"""Deterministic, budget-matched post-prefill selection controls.

No selectors receive answers, needle offsets, or benchmark labels.
Entity boundaries are a punctuation heuristic, not an entity recognizer.
"""
from __future__ import annotations
import random
import re
from bisect import bisect_left, bisect_right
import torch


def observation_scores(attentions, length: int, query_window: int, device, corrected=False) -> torch.Tensor:
    if isinstance(attentions, torch.Tensor):
        return attentions
    if not attentions or not any(a is not None for a in attentions):
        raise ValueError("Attention scoring requires eager attentions; no silent fallback")
    scores = torch.zeros(length, device=device, dtype=torch.float32)
    for attention in attentions:
        if attention is not None:
            # Slice BEFORE head reduction: do not create a second L x L tensor.
            observed = attention[0, :, -query_window:, :].float()
            scores += observed.sum(dim=(0, 1))
            if corrected:
                scores -= observed.amin(dim=1).sum(dim=0) * observed.shape[1]
    return scores.clamp_min(0)


def select_history(scores, start: int, end: int, capacity: int, chunk_size: int,
                   policy: str, seed: int, boundaries=(), recency_ties=False) -> tuple[list[int], list[tuple[int, int]]]:
    if capacity == 0:
        return [], []
    if end - start < capacity:
        raise ValueError("Historical candidate set smaller than capacity")
    values = scores.detach().cpu().tolist()
    if policy == "query_tokens":
        chosen = sorted(sorted(range(start, end), key=lambda i: (-values[i], i))[:capacity])
        return chosen, [(i, i + 1) for i in chosen]
    if policy == "adaptive_chunks":
        # Predeclared salience-concentration rule; research heuristic, not tuned on answers.
        historical = sorted((max(0.0, values[i]) for i in range(start, end)), reverse=True)
        concentration = sum(historical[:capacity]) / max(sum(historical), 1e-12)
        chunk_size = min(capacity, 16 if concentration > 0.5 else 32 if concentration > 0.25 else 64)
    if policy == "entity_chunks":
        # Whole sentence/record spans where possible. Oversized spans split at C.
        edges = [start] + sorted({b for b in boundaries if start < b < end}) + [end]
        chunks = [(s, min(s + chunk_size, b)) for a, b in zip(edges, edges[1:])
                  for s in range(a, b, chunk_size)]
    else:
        chunks = [(s, min(s + chunk_size, end)) for s in range(start, end, chunk_size)]
    if policy in {"ogc", "random_chunks"}:
        if capacity % chunk_size:
            raise ValueError("Fixed-chunk capacity must be divisible by chunk size")
        # A short tail must not turn K fixed chunks into K+1 fragments.
        chunks = [(a, b) for a, b in chunks if b - a == chunk_size]
    if policy == "random_chunks":
        random.Random(seed).shuffle(chunks)
    elif policy in {"ogc", "adaptive_chunks", "entity_chunks"}:
        # Mean salience permits comparisons across boundary span sizes.
        chunks.sort(key=lambda ab: (-sum(values[ab[0]:ab[1]]) / (ab[1] - ab[0]), -ab[0] if recency_ties else ab[0]))
    else:
        raise ValueError(f"Unknown selection policy: {policy}")
    chosen, ranges = [], []
    for a, b in chunks:
        take = min(b - a, capacity - len(chosen))
        if take:
            ranges.append((a, a + take))
            chosen.extend(range(a, a + take))
        if len(chosen) == capacity:
            break
    if len(chosen) != capacity:
        raise AssertionError("Selector failed to fill historical budget")
    return sorted(chosen), sorted(ranges)


def record_spans(tokenizer, text, token_count):
    """Sentence/newline units with exact full-tokenizer offsets, not token decodes.

    Decimal points and punctuation inside identifiers are not boundaries. The
    full text may include a held-out question; only document tokens are eligible.
    """
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"][:token_count]
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]
    edges = [0] + [m.end() for m in re.finditer(r"\n+|(?<=[.!?])\s+", text)] + [len(text)]
    spans = []
    for a, b in zip(edges, edges[1:]):
        lo, hi = bisect_right(ends, a), bisect_left(starts, b)
        if lo < min(hi, token_count):
            spans.append((lo, min(hi, token_count), text[a:b]))
    return spans


def select_records(scores, start, end, capacity, spans, query):
    """Whole-record selection with query-identifier coverage and attention residual.

    Generic identifiers come from the user query, never benchmark answers or
    needle offsets. Greedy marginal query-word coverage favors dispersed records.
    Complete fitting units are selected before residual token-level budget fill.
    """
    values = scores.detach().cpu().tolist()
    words = lambda text: set(re.findall(r"[a-z][a-z0-9_-]*", text.casefold()))
    stop = set("the a an and or for of to in is its only return current state with project answer".split())
    terms = words(query) - stop
    anchors = {m.casefold() for m in re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", query) if len(m) >= 8}
    candidates = []
    for a, b, text in spans:
        a, b = max(a, start), min(b, end)
        if a < b:
            candidates.append((a, b, bool(anchors and any(k in text.casefold() for k in anchors)),
                               words(text) & terms, sum(values[a:b]) / (b-a)))
    chosen, covered = set(), set()
    while candidates:
        candidates.sort(key=lambda x: (-int(x[2]), -len(x[3] - covered), -x[4], -x[0]))
        a, b, match, coverage, _ = candidates.pop(0)
        unit = set(range(a, b)) - chosen
        if len(unit) <= capacity - len(chosen):
            chosen.update(unit)
            covered.update(coverage)
        if len(chosen) == capacity:
            break
    # Never split an already retained record. Fill spare slots from remaining
    # tokens; oversized units cannot be guaranteed atomic under a hard budget.
    spare = sorted((i for i in range(start, end) if i not in chosen), key=lambda i: (-values[i], -i))
    chosen.update(spare[:capacity-len(chosen)])
    indices = sorted(chosen)
    ranges = []
    for i in indices:
        if ranges and ranges[-1][1] == i:
            ranges[-1] = (ranges[-1][0], i+1)
        else:
            ranges.append((i, i+1))
    if len(indices) != capacity:
        raise AssertionError("Record selector failed to fill history capacity")
    return indices, ranges
