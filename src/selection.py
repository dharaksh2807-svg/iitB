"""Deterministic, budget-matched post-prefill selection controls.

No selectors receive answers, needle offsets, or benchmark labels.
Entity boundaries are a punctuation heuristic, not an entity recognizer.
"""
from __future__ import annotations
import random
import torch


def observation_scores(attentions, length: int, query_window: int, device) -> torch.Tensor:
    if isinstance(attentions, torch.Tensor):
        return attentions
    if not attentions or not any(a is not None for a in attentions):
        raise ValueError("Attention scoring requires eager attentions; no silent fallback")
    scores = torch.zeros(length, device=device, dtype=torch.float32)
    for attention in attentions:
        if attention is not None:
            # Slice BEFORE head reduction: do not create a second L x L tensor.
            scores += attention[0, :, -query_window:, :].float().sum(dim=(0, 1))
    return scores


def select_history(scores, start: int, end: int, capacity: int, chunk_size: int,
                   policy: str, seed: int, boundaries=()) -> tuple[list[int], list[tuple[int, int]]]:
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
        chunks.sort(key=lambda ab: (-sum(values[ab[0]:ab[1]]) / (ab[1] - ab[0]), ab[0]))
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
