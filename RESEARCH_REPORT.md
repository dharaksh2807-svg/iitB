# KV-cache correctness audit and research report

Audit date: 2026-10-31. IIT Bombay Hackathon Problem 2.

## Conclusion and scope

The original OGC generation loop did not implement correct autoregressive
inference. It re-fed the last prompt token, discarded prefill logits, did not
refresh the cache reference after decode, and did not enforce the decode budget.
Its reported compressed bytes were calculated immediately after initial pruning,
not measured after generation. Historical OGC recall and latency claims therefore
do not validate the repaired implementation. All original `results/` files are
preserved unchanged.

The repaired pipeline uses prefill logits for token 1, consumes each subsequent
token once at its absolute position, updates returned caches, and physically
prunes after every decode step. Tensor and backing-storage bytes are measured
after generation. In a forced 100-token generation on the pinned 1.5B Qwen model,
the cache remains at exactly 260 entries: 7,454,720 bytes (7.109375 MiB), matching
unique retained tensor storage. This is a retained-KV measurement, not a measured
peak-memory reduction.

The audit executed 40 distinct relevant tests and independently verified 1,875
benchmark outputs: a 105-case 1K cohort, a 12-question 1K–8K budget pilot, and two
105-question chunk-size reruns. There are 114 unique questions across these
experiments; repeated conditions are paired evaluations, not extra independent
samples. At B=260 the boundary heuristic achieves 69.5% all-facts recall versus
84.8% full-cache recall on the expanded cohort, with 75.1% fewer mean retained KV
bytes. Canonical EM is 15.2% versus 34.3%, and multi-fact losses remain substantial.
The small long-context pilot provides promising operating points but cannot
establish general recall preservation. The full 300-question matrix and
peak-memory reductions remain unverified.

## Changes and rationale

| Files | Change |
| --- | --- |
| `src/generation.py`, `src/metrics.py` | Shared single-sequence generation contract, explicit positions/masks, EOS/count limits, synchronized component and wall timings, physical and backing-storage byte accounting. |
| `src/ogc_cache.py`, `src/selection.py` | Correct decode, exact rolling budgets, observation-based token/chunk selection, random contiguous controls, adaptive chunks, punctuation-boundary chunks; deferred selection when generation first crosses budget. |
| `src/baseline.py`, `src/streaming_cache.py`, `src/sliding_cache.py`, `src/h2o_cache.py` | Preserve policies and public result shapes while using the common inference contract. Pin the default model/tokenizer. No silent device fallback. Label static H2O approximation. |
| `src/research.py`, `run_research_benchmark.py` | Independent seeded cases, strict boundary-aware scoring plus canonical EM, fresh matched budgets, raw outputs, paired stratified bootstrap intervals, Pareto status, immutable manifests and source hashes. |
| `run_phase6_evaluation.py`, `run_phase7_benchmark.py`, `run_phase8_ablation.py` | Current CLI entry points run fresh comparisons rather than importing old results. Fix nested checkpoint telemetry and remove hard-coded byte coordinates from historical plotting helpers. |
| `tests/test_cache_correctness.py`, `tests/test_cache_pretrained.py`, `tests/test_research_benchmark.py` | Offline tiny-Qwen2 correctness tests, opt-in pinned-model checks, benchmark-validity tests. |
| `EXPERIMENT_PROTOCOL.md`, `README.md`, this report | Protocol v2, historical labeling, reproducible commands and limits. |

Only inference, benchmark, test, protocol, and research evidence files were
changed. No UI files were edited. The duplicate generation loops were consolidated
so baseline and compressed methods share stopping rules and timing definitions.

## Positions, RoPE and memory semantics

Original absolute `position_ids` are supplied for both prefill and decode.
Retained keys keep their existing RoPE phases. A compact single-query attention
mask exposes exactly the retained keys plus the current token. The installed
Transformers 5.20.1 Qwen2 API does not declare `cache_position`; it is passed
explicitly for APIs that do. That compatibility branch is exercised with an
adapter; an actual older Transformers installation is unverified.

Block prefill uses at most 256 query rows per forward and accumulates scores into
one length-L vector. OGC scores the last 48 observation queries; the static
H2O-style oracle scores all prefill queries. Tiny-model tests compare block scores
against dense attention and verify that pruning does not change cached RoPE keys.
Live attention output storage scales as O(layers × heads × block_size × L), rather
than retaining O(layers × heads × L²) matrices. Arithmetic is still quadratic and
the full prompt KV remains live until post-prefill pruning. The direct pruning
test helpers can still accept dense attention tensors; generation uses blocks.

Budgets are exact **after** pruning. A decode forward can transiently hold B+1
entries. The last generated token has not been consumed and is absent from the
final cache, so full-cache bytes correspond to prompt length plus N−1 decode
steps. No views of evicted backing storage are kept by pruning.

H2O here is a **static prefill oracle approximation**, not the dynamically updated
H2O algorithm in the [original paper](https://arxiv.org/abs/2306.14048).
StreamingLLM-style sink/recent retention uses absolute RoPE positions and does not
reproduce the original method's positional-shift behavior. The absolute-position
contract follows the distinction described in the
[Transformers cache documentation](https://huggingface.co/docs/transformers/v4.56.2/cache_explanation)
and was checked against the installed Qwen2/DynamicCache source.

## Research design

Model and tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. PyTorch 2.14.1,
Transformers 5.20.1, Python 3.12.14. Measured hardware is Apple Silicon M5/MPS;
CUDA is unavailable and unverified. Fresh comparisons share BF16 weights, eager
attention, block size 256, greedy argmax without repetition penalty, 35 output
tokens maximum, and identical EOS rules. Configuration execution order is seeded
and randomized per question. Warmups and model load are excluded.

At B=260/516/1028, StreamingLLM gets four sinks and B−4 recent slots; sliding
window gets B recent slots. OGC and its controls get four sinks, H=(B−4)/4 history
slots, and B−4−H recent slots. H2O-style gets H static heavy hitters and B−H recent
slots. Fixed OGC and random controls choose exactly H/C disjoint full chunks from
the same grid; this differs from historical overlapping greedy chunk selection.
Query-aware tokens use the same observation scores and history capacity. Adaptive
chunks use predeclared attention-concentration thresholds to choose 16/32/64.
Boundary chunks approximate sentence/record boundaries from token punctuation;
this is a heuristic, not an entity recognizer. No selector receives ground-truth
answers or needle offsets.

The new default suite contains 300 independently seeded documents: four lengths
(1K/2K/4K/8K), five depths (5/25/50/75/95%), three types, and five repetitions.
Multi-fact records occur at separated depths. Adversarial cases include plausible
obsolete records near the question. Full chat prompts, measured token lengths,
actual fact depths, seeds and answers are persisted. The generated 300-case chat
suite has 300 unique seeds and token length error at most two tokens.

Accuracy requires all answers with identifier boundaries; `15438` does not match
`5438`, and sentence-final punctuation is accepted. Canonical exact match is
reported separately. Paired differences use 2,000 seeded bootstrap resamples
stratified by context/depth/type. Intervals are withheld for pilot strata with
only one case, which cannot estimate within-stratum uncertainty. Intervals describe the synthetic suite and have no
multiple-comparison adjustment. Development data were inspected during this
audit, so these experiments are exploratory, not a blind confirmatory study.

## Executed validation

Forty distinct tests passed across the audit executions below. Raw commands,
counts, durations and log hashes are in
[`test-evidence.json`](artifacts/audit/test-evidence.json). These are selected
successful executions across audit commits, not one full discovery run on the
last commit. Full UI/server test discovery was not run.

| Test file | Passed |
| --- | ---: |
| `test_cache_correctness.py` | 17 |
| `test_cache_pretrained.py` | 2 |
| `test_research_benchmark.py` | 4 |
| `test_baseline.py` | 4 |
| `test_streaming_cache.py` | 3 |
| `test_additional_baselines.py` | 3 |
| `test_ogc_cache.py` | 3 |
| `test_benchmark_dataset.py` | 4 |

The final focused suite includes a deliberately corrupted byte measurement to
verify that the baseline's theoretical size check is independent of observation.
The pinned-model checks verify full/OGC no-pruning agreement against HF greedy
inference (repetition penalty 1.0), first-token agreement, and physical budgeting
through a forced 100-token decode.

Development failures remain visible: `pretrained-initial-failed.txt` used the
model's default repetition penalty in the HF reference and attempted to patch a
tokenizer property with `unittest.mock`. The corrected check sets repetition
penalty 1.0 and restores the tokenizer property explicitly. `research-pilot-4`
failed before measurement because Transformers no longer populated
`config._commit_hash`; snapshot-path verification replaced it. Its early manifest
may still say running because that early failure occurred outside the original
status handler. Later manifests handle load/warmup failures.

`research-pilot-v2-4` was a development run with an earlier scorer that rejected
answers followed by a period, an earlier chunk-tail policy, and different
allocation-fill settings. Some integration tests overlapped this development
run. Its raw outputs and scores are preserved and excluded from final comparative
claims. No recorded result value was corrected in place.

## Verified empirical results

### Expanded 1K cohort: 105 independent cases, nine policies, B=260

[Manifest](artifacts/research-1k-105/manifest.json),
[raw outputs](artifacts/research-1k-105/outputs.jsonl),
[summary](artifacts/research-1k-105/summary.json), and
[independent artifact validation](artifacts/research-1k-105/artifact-verification.json).
There are 7 independently seeded questions per depth/type cell, 21 per depth,
and 35 per type. All 945 expected outputs passed validation of question/config
pairing, decoded predictions and scores, exact budget histories, original absolute
positions, bytes against independent model geometry, and timing arithmetic.

| Policy | Correct / 105 | All-facts recall % | Canonical EM % | Mean retained KV MiB | Delta vs full, pp [paired 95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| Full cache | 89/105 | 84.8 | 34.3 | 28.58 | Reference |
| StreamingLLM-style | 14/105 | 13.3 | 1.9 | 7.11 | -71.4 [-77.1, -65.7] |
| Sliding window | 6/105 | 5.7 | 1.9 | 7.11 | -79.0 [-85.7, -72.4] |
| H2O prefill approximation | 14/105 | 13.3 | 1.9 | 7.11 | -71.4 [-77.1, -65.7] |
| OGC C=64, K=1 | 48/105 | 45.7 | 4.8 | 7.11 | -39.0 [-42.9, -34.3] |
| Query-aware tokens | 27/105 | 25.7 | 1.9 | 7.11 | -59.0 [-64.8, -53.3] |
| Random C=64, K=1 | 19/105 | 18.1 | 3.8 | 7.11 | -66.7 [-73.3, -60.0] |
| Adaptive chunks | 58/105 | 55.2 | 10.5 | 7.11 | -29.5 [-34.3, -23.8] |
| Punctuation-boundary chunks | 73/105 | 69.5 | 15.2 | 7.11 | -15.2 [-20.0, -9.5] |

The descriptive recall-memory frontier contains full cache and the boundary
heuristic. Boundary retention reduces mean retained KV bytes by 75.1%
versus this cohort's actual full-cache mean, while losing 15.2 percentage points
of all-facts recall. This is not lossless preservation. No peak-memory benefit
was measured.

OGC exceeds the budget-matched random control by 27.6 pp (paired 95% CI
[21.9, 33.3]) and StreamingLLM-style by 32.4 pp [28.6, 36.2]. Boundary retention
exceeds OGC by 23.8 pp [19.0, 28.6]; adaptive retention exceeds OGC by 9.5 pp
[4.8, 14.3]. These exploratory contrasts have no multiplicity correction.
Additional contrasts are saved separately in
[additional-comparisons.json](artifacts/research-1k-105/additional-comparisons.json);
the recorded raw outputs and summary were not edited.

The conclusion depends on the scoring rule. Under normalized canonical EM,
OGC exceeds random by only 1.0 pp, with a paired 95% CI [0.0, 2.9], which does not
establish a strictly positive gain at this interval level. Boundary retention
exceeds OGC by 10.5 pp [6.7, 14.3] under EM, but remains 19.0 pp below full cache
[-22.9, -15.2]. These separately derived contrasts are in
[canonical-em-comparisons.json](artifacts/research-1k-105/canonical-em-comparisons.json).
All-facts matching accepts surrounding prose containing the targets; it does not
prove clean answer formatting or rule out contradictory extra text. The much
lower EM scores prevent a claim of preserved exact-answer quality.

Aggregate scores hide a substantial failure on separated multi-fact retrieval:

| Query type (35 cases each) | Full % | OGC C=64 % | Adaptive % | Boundary % |
| --- | ---: | ---: | ---: | ---: |
| single_fact | 97.1 | 77.1 | 65.7 | 97.1 |
| multi_fact | 88.6 | 0.0 | 20.0 | 31.4 |
| adversarial | 68.6 | 60.0 | 80.0 | 80.0 |

A single 64-token historical chunk cannot directly cover two distant records.
Even boundary retention recovers both facts in only 11/35 cases, versus 31/35 for
full cache. Adaptive and boundary methods help this cohort but do not solve
multi-fact preservation. Fixed C=32/K=2 and C=16/K=4 controls were specified in the
protocol and are evaluated separately on the same questions; this is exploratory
follow-up rather than a new independent validation population.

| Target depth (21 cases each) | Full % | OGC C=64 % | Boundary % |
| --- | ---: | ---: | ---: |
| start | 95.2 | 33.3 | 61.9 |
| early_middle | 76.2 | 47.6 | 57.1 |
| middle | 81.0 | 23.8 | 76.2 |
| late_middle | 76.2 | 57.1 | 57.1 |
| end | 95.2 | 66.7 | 95.2 |

All remaining policy strata are available in the summary, including negative
results. These are 1K results; they must not be generalized to 8K from this cohort.

Mean synchronized timings below are measured milliseconds. Prefill includes
score reduction; selection includes initial selection plus all decode pruning,
which overlaps decode wall time. Columns are not additive. Output lengths vary
with EOS and response behavior, so these values are not controlled token-count
speedup estimates.

| Policy | Prefill ms | Score ms | Selection/pruning ms | TTFT ms | Decode wall ms | E2E wall ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full cache | 479.2 | 0.0 | 0.1 | 482.2 | 872.3 | 1354.8 | 24.4 |
| StreamingLLM-style | 474.6 | 0.0 | 145.1 | 486.1 | 861.1 | 1347.4 | 24.5 |
| Sliding window | 484.2 | 0.0 | 172.6 | 496.6 | 1041.6 | 1538.5 | 24.7 |
| H2O prefill approximation | 564.4 | 81.9 | 136.4 | 570.2 | 848.1 | 1418.6 | 24.6 |
| OGC C=64, K=1 | 493.6 | 8.8 | 143.2 | 506.0 | 854.0 | 1360.3 | 24.6 |
| Query-aware tokens | 487.8 | 8.7 | 139.9 | 500.3 | 825.4 | 1326.0 | 24.6 |
| Random C=64, K=1 | 484.5 | 0.0 | 144.9 | 496.5 | 854.4 | 1351.2 | 24.6 |
| Adaptive chunks | 492.7 | 9.1 | 145.5 | 505.2 | 859.0 | 1364.4 | 24.5 |
| Punctuation-boundary chunks | 485.2 | 8.8 | 155.0 | 501.3 | 898.8 | 1400.4 | 24.7 |

Decode throughput is similar across these policies on this cohort; reducing
retained KV did not establish an end-to-end speedup. Initial selection, decode
pruning, prefill/decode forward times and output counts are also recorded for
every question.

### Matched-budget context pilot: 12 questions, 25 configurations

[Manifest](artifacts/research-matched-12/manifest.json),
[raw outputs](artifacts/research-matched-12/outputs.jsonl),
[summary](artifacts/research-matched-12/summary.json), and
[independent verification](artifacts/research-matched-12/artifact-verification.json).
All 300 expected outputs completed. The target lengths are 1K, 2K, 4K and 8K.
At each length there is one early-middle multi-fact, one middle single-fact,
and one late-middle adversarial question. This pilot therefore confounds depth
with question type and has one case per stratum. Paired CIs are withheld rather
than reporting zero-width intervals. These are descriptive pilot results.

| Policy | B=260 recall % | B=516 recall % | B=1028 recall % |
| --- | ---: | ---: | ---: |
| StreamingLLM-style | 0.0 | 8.3 | 41.7 |
| Sliding window | 0.0 | 8.3 | 33.3 |
| H2O prefill approximation | 0.0 | 8.3 | 33.3 |
| OGC C=64 | 33.3 | 33.3 | 91.7 |
| Query-aware tokens | 33.3 | 33.3 | 41.7 |
| Random C=64 | 8.3 | 8.3 | 33.3 |
| Adaptive chunks | 50.0 | 91.7 | 91.7 |
| Punctuation-boundary chunks | 83.3 | 91.7 | 91.7 |

Full-cache recall on the same 12 questions is 91.7%.

Equal aggregate recall is not necessarily paired preservation. At B=516,
adaptive and boundary policies each correct the full baseline's 4K adversarial
error but lose its correct 1K multi-fact answer: only 10/12 correctness outcomes
agree. OGC at B=1028 has the same correctness outcome as full cache on all 12
questions, but this is a small pilot, not an equivalence result or a claim of
identical generated text. See [paired-case-outcomes.json](artifacts/research-matched-12/paired-case-outcomes.json).
Boundary recall at 1K also falls from 3/3 at B=260 to 2/3 at B=516 in this pilot;
increasing the budget does not guarantee monotonic recall under a changed
history/recent split and selection.

The table includes all budgets and policies; no best single pilot cell is treated
as a validated recall-preserving operating point. Successful 8K forwards establish
that this block-scoring path is feasible on the measured 16 GB MPS machine, not
that it reduces peak memory.

| Policy / budget | Mean measured retained KV MiB |
| --- | ---: |
| Full cache | 105.5469 |
| OGC, B=260 | 7.1094 |
| OGC, B=516 | 14.1094 |
| OGC, B=1028 | 28.1094 |
| Boundary, B=260 | 7.1094 |
| Boundary, B=516 | 14.1094 |
| Boundary, B=1028 | 28.1094 |

Short prompts may finish below a budget cap. Every physical byte value above is
computed from retained tensors; each raw cache history was checked against
`min(consumed_length, B)`. Per-context pilot recall, based on only three cases
per target length, is:

| Target context | Full % | OGC B260 % | OGC B516 % | OGC B1028 % | Boundary B260 % | Boundary B516 % | Boundary B1028 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 100.0 | 33.3 | 33.3 | 100.0 | 100.0 | 66.7 | 100.0 |
| 2048 | 100.0 | 33.3 | 33.3 | 100.0 | 100.0 | 100.0 | 100.0 |
| 4096 | 66.7 | 33.3 | 33.3 | 66.7 | 66.7 | 100.0 | 66.7 |
| 8192 | 100.0 | 33.3 | 33.3 | 100.0 | 66.7 | 100.0 | 100.0 |

Descriptive frontier for this pilot: `adaptive_chunks/B516`, `entity_chunks/B260`, `entity_chunks/B516`. Small-sample uncertainty prevents a general frontier claim.

The 105-case run used source commit `260e708`; this pilot includes later
metadata corrections (independent baseline geometry and withholding sparse CIs).
The shared generation and token/chunk selection implementation is the same.
Manifests retain exact source hashes for both. The three 1K pilot questions also
occur in the expanded cohort, so the cohorts are analyzed separately and are
not counted as independent extra 1K cases.


### Fixed chunk-size follow-up at B=260: C=64/32/16

The same 105 questions were rerun at C=32/K=2 and C=16/K=4, with a fresh full
baseline and matched random control in each run. This holds historical capacity
at 64 and recent capacity at 192, changing the number of disjoint chunks.
[All contrasts](artifacts/audit/chunk-size-comparisons.json) are exploratory paired
comparisons without multiplicity correction. These repetitions are not additional
independent questions.

| Policy | C | K | All-facts recall % | Canonical EM % | Multi-fact recall % | Measured KV MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ogc | 64 | 1 | 45.7 | 4.8 | 0.0 | 7.1094 |
| random_chunks | 64 | 1 | 18.1 | 3.8 | 0.0 | 7.1094 |
| ogc | 32 | 2 | 47.6 | 14.3 | 37.1 | 7.1094 |
| random_chunks | 32 | 2 | 17.1 | 2.9 | 2.9 | 7.1094 |
| ogc | 16 | 4 | 53.3 | 7.6 | 11.4 | 7.1094 |
| random_chunks | 16 | 4 | 20.0 | 2.9 | 0.0 | 7.1094 |

Paired all-facts differences relative to the corresponding C=64 policy:

- ogc C32 minus C64: +1.9 pp [-1.0, +4.8].
- random_chunks C32 minus C64: -1.0 pp [-5.7, +3.8].
- ogc C16 minus C64: +7.6 pp [+3.8, +11.4].
- random_chunks C16 minus C64: +1.9 pp [-3.8, +7.6].

The C=32 all-facts interval spans zero, despite its stronger multi-fact score.
Its canonical EM gain over C=64 is +9.5 pp [5.7, 12.4]. C=16 improves aggregate
all-facts recall by +7.6 pp [3.8, 11.4], but its EM gain is only +2.9 pp [0.0, 5.7].
C=32 is better for multi-fact recall and EM than C=16 in these runs; C=16 is
better for aggregate all-facts recall. Smaller chunks therefore do not provide
a uniform improvement across metrics or query types.

[Retention structure](artifacts/derived-analysis/retention-structure.json) records
the actual retained span lengths and contiguity. Query-token selection averages
22.4 disjoint historical runs, adaptive selection 2.3, and the boundary heuristic
5.5, versus one for fixed C=64. These are descriptive structural measurements,
not proof of semantic entity preservation or inferred adaptive C decisions.

Baseline repeatability on raw generated token IDs: C=32 run: 105/105 identical; C=16 run: 105/105 identical. The exact comparison is saved in [cross-run-baseline-agreement.json](artifacts/audit/cross-run-baseline-agreement.json).

- C=32: [manifest](artifacts/research-chunks32-105/manifest.json), [raw outputs](artifacts/research-chunks32-105/outputs.jsonl), [summary](artifacts/research-chunks32-105/summary.json), [independent artifact validation](artifacts/research-chunks32-105/artifact-verification.json).
- C=16: [manifest](artifacts/research-chunks16-105/manifest.json), [raw outputs](artifacts/research-chunks16-105/outputs.jsonl), [summary](artifacts/research-chunks16-105/summary.json), [independent artifact validation](artifacts/research-chunks16-105/artifact-verification.json).

The full 300-question / 25-condition matrix and fixed-chunk sweeps across every
budget/context cell were not executed. The larger population in this audit is
the 105-case 1K cohort; the long-context budget sweep remains a small pilot.


## Reproduction

Run from the repository root with the pinned dependencies in `requirements.txt`:

```bash
.venv/bin/python -m unittest discover -s tests -p test_cache_correctness.py -v
.venv/bin/python -m unittest discover -s tests -p test_research_benchmark.py -v
RUN_PRETRAINED_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_cache_pretrained.py -v

# Generate 300 cases without inference; existing directories are rejected.
.venv/bin/python run_research_benchmark.py --build-only --output artifacts/dataset-300-repro

# Audit's expanded 105-question 1K study, with all nine policies at B=260.
.venv/bin/python run_research_benchmark.py --lengths 1024 --repeats 7 --budgets 260 --output artifacts/1k-105-repro

# Matched-budget context pilot, using the saved preselected 12-question input.
.venv/bin/python run_research_benchmark.py --dataset artifacts/matched-contexts-12-input.json --output artifacts/matched-12-repro

# Full 300-question / 25-condition matrix (expensive, not implied to have run).
.venv/bin/python run_research_benchmark.py --output artifacts/full-matrix-repro

# Reproduce the actual 105-question fixed-size controls with saved inputs.
.venv/bin/python run_research_benchmark.py --dataset artifacts/research-1k-105/dataset.json --policies full ogc random_chunks --budgets 260 --chunk-size 32 --output artifacts/chunks32-105-repro
.venv/bin/python run_research_benchmark.py --dataset artifacts/research-1k-105/dataset.json --policies full ogc random_chunks --budgets 260 --chunk-size 16 --output artifacts/chunks16-105-repro
```

Each directory contains `dataset.json`, `manifest.json`, append-only
`outputs.jsonl`, and (on success) `summary.json`. Manifests include the exact
executed command, package versions, requested/resolved revision, dataset and
source hashes, git state, warmup/settings, status, and timestamps. To independently verify completed output decoding,
question pairing, budget histories, geometry-based physical bytes and timing
arithmetic, run:

```bash
.venv/bin/python artifacts/audit/verify_results.py artifacts/research-1k-105 artifacts/research-matched-12 artifacts/research-chunks32-105 artifacts/research-chunks16-105

# Recompute exploratory contrasts into a new directory, leaving raw results intact.
.venv/bin/python artifacts/audit/derived_analysis.py --output artifacts/analysis-repro
```

## Unresolved limits

- No isolated per-run peak-memory comparison. MPS allocation is sampled only at
  checkpoints, and RSS high water belongs to the process lifetime. CUDA allocator
  peak instrumentation exists but was not executed on CUDA.
- All methods select their first token from full-prefill logits. That token has
  already seen every fact. These tests measure post-prefill generation, not a
  strict demonstration that recall depends exclusively on compressed KV. A
  stronger held-out-query protocol must prune before consuming the question.
- Full-prompt prefill KV, quadratic attention compute, and repeated prefill at
  large lengths remain expensive on this hardware. Neither speedups nor peak
  reductions follow from the retained-cache byte plateau.
- One unpadded sequence, Qwen2 DynamicCache, native static RoPE only. Padded
  batches, other cache architectures, dynamic RoPE scaling, beam search,
  quantized caches, other model families and CUDA remain unverified.
- Entity retention currently approximates sentence/record boundaries from punctuation; true entity recognition and atomic entity-span retention are unverified.
- Synthetic identifier recall is a narrow task; it does not establish quality on
  natural documents, dialogue, reasoning, or production workloads. Boundary and
  adaptive retention are hypotheses whose negative results must remain visible.
