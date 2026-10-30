"""Independently seeded recall cases and paired, stratified bootstrap analysis."""
from __future__ import annotations
import hashlib
import random
import re
from collections import defaultdict

import numpy as np

from src.dataset import BenchmarkGenerator, generate_distractor_paragraph, normalize_text


def build_cases(tokenizer, lengths=(1024, 2048, 4096, 8192), repeats=5, seed=20261031):
    """Default 300 independent documents: 4 lengths x 5 depths x 3 types x 5 seeds.

    Multi-fact cases separate the two current records by roughly 30% of the body.
    Adversarial cases add plausible obsolete records next to the inquiry.
    Depths and lengths are measured after final tokenization, never assumed exact.
    """
    cases = []
    for length in lengths:
        for depth_name, depth in BenchmarkGenerator.DEPTH_SPEC:
            for kind in ("single_fact", "multi_fact", "adversarial"):
                for repeat in range(repeats):
                    identity = f"{seed}/{length}/{depth_name}/{kind}/{repeat}"
                    case_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
                    rng = random.Random(case_seed)
                    entity = f"PROJ-{rng.getrandbits(48):012X}"
                    port, obsolete_port = str(rng.randrange(10000, 60000)), str(rng.randrange(60001, 65535))
                    region = rng.choice(["eu-west", "ap-south", "us-east", "sa-east"]) + f"-{rng.randrange(1, 10)}"
                    fact1 = f"\n\nCURRENT record for project {entity}: primary database port is {port}.\n\n"
                    fact2 = f"\n\nCURRENT record for project {entity}: assigned deployment region is {region}.\n\n"
                    question = (f"For project {entity}, return only the current primary database port" +
                                (" and deployment region, separated by a comma." if kind == "multi_fact" else "."))
                    preamble = "Read the records below. Use CURRENT records only; ignore ARCHIVED records.\n\n"
                    suffix = (f"\n\nARCHIVED obsolete record for project {entity}: primary database port is {obsolete_port}.\n"
                              if kind == "adversarial" else "")
                    suffix += f"\n[Inquiry]\n{question}\nAnswer:"
                    facts = [(depth, fact1)]
                    if kind == "multi_fact":
                        facts.append(((depth + 0.30) % 1.0, fact2))
                    facts.sort()
                    overhead = len(tokenizer.encode(preamble + suffix + "".join(f for _, f in facts), add_special_tokens=False))
                    filler_target = max(64, length - overhead)
                    filler_ids = []
                    while len(filler_ids) < filler_target + 64:
                        filler_ids.extend(tokenizer.encode(generate_distractor_paragraph(rng) + "\n\n", add_special_tokens=False))

                    def assemble(count):
                        parts, cursor = [preamble], 0
                        for fraction, fact in facts:
                            boundary = int(count * fraction)
                            parts.append(tokenizer.decode(filler_ids[cursor:boundary], skip_special_tokens=True))
                            parts.append(fact)
                            cursor = boundary
                        parts.append(tokenizer.decode(filler_ids[cursor:count], skip_special_tokens=True))
                        text = "".join(parts) + suffix
                        if hasattr(tokenizer, "apply_chat_template"):
                            text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                                 tokenize=False, add_generation_prompt=True)
                        return text

                    # Token boundaries can merge on concatenation; correct filler count.
                    for _ in range(6):
                        document = assemble(filler_target)
                        actual = len(tokenizer.encode(document))
                        if abs(actual - length) <= 2:
                            break
                        filler_target += length - actual
                    answers = [port, region] if kind == "multi_fact" else [port]
                    if not all(answer in document for answer in answers):
                        raise AssertionError("Ground truth lost during construction")
                    offsets = []
                    for _, fact in facts:
                        char_pos = document.index(fact)
                        offsets.append(len(tokenizer.encode(document[:char_pos], add_special_tokens=False)))
                    cases.append(dict(item_id=identity, target_context_length=length, actual_token_length=actual,
                                      depth_category=depth_name, depth_fraction=depth, query_type=kind,
                                      document=document, question=question, ground_truth_answers=answers,
                                      seed=case_seed, fact_token_offsets=offsets,
                                      actual_fact_depths=[offset / actual for offset in offsets]))
    return cases


def score(prediction, answers):
    normalized = normalize_text(prediction)
    # Boundary matching rejects 5438 inside 15438; all facts are required.
    matches = [bool(re.search(r"(?<![\w.-])" + re.escape(normalize_text(a)) + r"(?![\w-]|\.\d)", normalized))
               for a in answers]
    canonical = normalize_text(", ".join(answers))
    return dict(is_correct=bool(matches) and all(matches), score_fraction=sum(matches) / max(1, len(matches)),
                exact_match=normalized == canonical, answer_matches=matches)


def paired_interval(records, reference, seed=17, samples=2000):
    """Paired bootstrap, stratified by context/depth/type; units are independent seeds."""
    by_id = {r["item_id"]: r for r in reference}
    if len(by_id) != len(reference) or len({r["item_id"] for r in records}) != len(records):
        raise ValueError("Duplicate question ids")
    if set(by_id) != {r["item_id"] for r in records}:
        raise ValueError("Paired uncertainty requires exactly matched questions")
    strata = defaultdict(list)
    for row in records:
        other = by_id[row["item_id"]]
        key = tuple(row[k] for k in ("context_length", "depth_category", "query_type"))
        strata[key].append(int(row["is_correct"]) - int(other["is_correct"]))
    if not strata:
        raise ValueError("No paired records")
    min_cases = min(map(len, strata.values()))
    if min_cases < 2:
        return dict(delta_accuracy_pp=100 * sum(sum(v) for v in strata.values()) / len(records),
                    paired_bootstrap_95_ci_pp=None, bootstrap_samples=0, bootstrap_seed=seed,
                    independent_pairs=len(records), min_cases_per_stratum=min_cases,
                    caveat="Interval withheld: singleton strata cannot estimate within-stratum variance")
    rng = np.random.default_rng(seed)
    estimates = np.zeros(samples)
    for values in strata.values():
        values = np.asarray(values)
        estimates += values[rng.integers(0, len(values), (samples, len(values)))].sum(axis=1)
    count = len(records)
    lo, hi = np.quantile(100 * estimates / count, [0.025, 0.975])
    return dict(delta_accuracy_pp=100 * sum(sum(v) for v in strata.values()) / count,
                paired_bootstrap_95_ci_pp=[float(lo), float(hi)], bootstrap_samples=samples,
                bootstrap_seed=seed, independent_pairs=count,
                min_cases_per_stratum=min_cases,
                caveat="Synthetic benchmark only; no multiple-comparison adjustment")


def summarize(records):
    summaries = {}
    for config in sorted({r["config_name"] for r in records}):
        rows = [r for r in records if r["config_name"] == config]
        summary = dict(total=len(rows), accuracy_pct=100 * np.mean([r["is_correct"] for r in rows]),
                       exact_match_pct=100 * np.mean([r["exact_match"] for r in rows]),
                       mean_kv_bytes=float(np.mean([r["compressed_kv_bytes"] for r in rows])))
        for category in ("context_length", "depth_category", "query_type"):
            summary["by_" + category] = {
                str(value): dict(n=len(subset), accuracy_pct=100 * np.mean([r["is_correct"] for r in subset]),
                                 mean_kv_bytes=float(np.mean([r["compressed_kv_bytes"] for r in subset])))
                for value in sorted({r[category] for r in rows})
                for subset in [[r for r in rows if r[category] == value]]}
        for name in ("prefill_time_ms", "scoring_time_ms", "selection_time_ms", "ttft_ms", "decode_time_ms", "total_time_ms"):
            values = [r[name] for r in rows if r[name] is not None]
            summary["mean_" + name] = float(np.mean(values)) if values else None
        summary["decode_tokens_per_sec"] = 1000 * sum(r["decode_steps"] for r in rows) / max(1e-12, sum(r["decode_time_ms"] for r in rows))
        summaries[config] = summary
    baseline = [r for r in records if r["config_name"] == "full"]
    if baseline:
        for config in summaries:
            rows = [r for r in records if r["config_name"] == config]
            summaries[config]["paired_vs_full"] = paired_interval(rows, baseline)
            if "/B" in config:
                budget = config.split("/B")[1].split("/")[0]
                peers = [name for name in summaries if name.startswith(("streaming/B", "random_chunks/B"))
                         and name.split("/B")[1].split("/")[0] == budget and name != config]
                summaries[config]["paired_budget_controls"] = {
                    peer: paired_interval(rows, [r for r in records if r["config_name"] == peer]) for peer in peers}
    for config, summary in summaries.items():
        summary["pareto_nondominated"] = not any(
            other["mean_kv_bytes"] <= summary["mean_kv_bytes"] and other["accuracy_pct"] >= summary["accuracy_pct"]
            and (other["mean_kv_bytes"] < summary["mean_kv_bytes"] or other["accuracy_pct"] > summary["accuracy_pct"])
            for name, other in summaries.items() if name != config)
    return summaries
