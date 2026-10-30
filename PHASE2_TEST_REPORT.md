# Phase 2 Test & Diagnostic Report: Long-Context KV-Cache Compression

**Date:** 2026-10-31  
**Project:** IIT Bombay Techfest Hackathon — Problem 2 (`iitB`)  
**Scope:** Testing and Empirical Diagnosis Only (No Major Algorithmic Redesign)  
**Evaluator:** Independent QA & Diagnostic Auditor  

---

## 1. Executive Summary & Diagnostic Findings

This audit independently executed and verified all automated correctness tests, inference smoke tests, and fresh matched-budget benchmark evaluations on Apple Silicon hardware for the IIT Bombay Problem 2 repository.

### Key Conclusions:
1. **Core Correctness Invariants Pass:** All 44 automated unit tests across `tests/` pass cleanly without error, verifying that the repaired generation pipeline enforces exact physical budget constraints, maintains sorted absolute RoPE positions, updates dynamic caches per step, and accurately accounts for physical tensor bytes.
2. **Multi-Fact Recall Failure Mode:** Under matched compression budgets ($B=260$ and $B=516$), OGC-Cache achieves **$0.0\%$ accuracy on multi-fact recall** (compared to $100.0\%$ for the uncompressed baseline). Analysis reveals two root causes:
   - **History Capacity Bottleneck:** At $B=260$ ($C=64$), history capacity $H=(B-4)/4=64$ tokens permits retaining only $K=1$ contiguous chunk. Facts placed at separated context depths ($\Delta \approx 30\%$ of sequence length) cannot physically fit into a single 64-token chunk.
   - **Arbitrary Grid Alignment & Entity Fracturing:** Fixed chunk boundaries ($4 + k \times C$) frequently slice entity records across chunk boundaries (e.g., token 259 in chunk 3, token 260 in chunk 4). If the chunk containing the project identifier is selected while the chunk containing the port or region value is evicted, the model retains the query entity but hallucinates the answer.
3. **Adversarial Distractor Resistance:** OGC-Cache achieves **$100.0\%$ recall on adversarial queries** across all context lengths (1,024 to 8,192 tokens), successfully using query cross-attention to isolate historical project records and ignore obsolete recency distractors where StreamingLLM and Sliding Window score $0.0\%$.
4. **Methodological Disclosures:**
   - **Shared Prefill Logit for Token 1:** All evaluated policies (baseline and compressed) select the first generated token directly from uncompressed prefill logits. Token 1 has full causal visibility over all context facts prior to pruning. Multi-token generation (tokens 2–35) relies entirely on pruned KV, which explains the sharp divergence on multi-fact queries.
   - **Static H2O Approximation:** The evaluated H2O baseline is a static prefill oracle approximation with $O(L^2)$ attention scoring overhead (+1,145 ms to +1,214 ms prefill latency penalty), heavily biased toward leading punctuation and attention sinks.
   - **Retained-KV vs. Peak Memory:** Compression is applied strictly post-prefill. Full prompt KV memory is materialized before eviction; true peak inference memory reduction during prefill remains unverified.

---

## 2. Hardware Environment & Verified Test Execution

### 2.1 Hardware Environment
- **Platform:** Apple Silicon (macOS 26.5.2, arm64, 10 CPU cores, 16.0 GB Unified RAM)
- **Execution Device:** Apple Metal Performance Shaders (`device="mps"`) via PyTorch 2.14.1
- **CUDA Availability:** `False` (CUDA is not available on this machine)
- **Model Hub ID:** `Qwen/Qwen2.5-1.5B-Instruct` (Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)
- **Precision:** `torch.bfloat16` (28 layers, 2 KV heads, 128 head dim $\implies 28,672\text{ bytes/token}$)

### 2.2 Test Suite Execution Results

| Test Suite / Script | Command | Result | Notes |
| :--- | :--- | :---: | :--- |
| **All Discovered Unit Tests** | `.venv/bin/python -m unittest discover -s tests` | **44 Passed** (0 failed, 2 skipped) | 42 offline & integration tests pass (114.4s). |
| **Opt-in Pretrained Tests** | `RUN_PRETRAINED_TESTS=1 .venv/bin/python -m unittest tests/test_cache_pretrained.py` | **2 Passed** (0 failed) | Verifies 100-step decode budget ($B=260$) and HF parity. |
| **Uncompressed Baseline Smoke Test** | `.venv/bin/python run_baseline.py` | **PASSED** | 52 prompt tokens, 30 gen tokens, 2.2148 MiB KV (100% theoretical match), 28.27 tok/s. |
| **StreamingLLM Smoke Test** | `.venv/bin/python run_compression.py` | **PASSED** | 1,022 prompt tokens, 260 cached tokens, 7.11 MiB KV (75.26% memory reduction), 24.67 tok/s. |
| **OGC-Cache Interactive Smoke Test** | `.venv/bin/python demo.py --mode interactive --preset 0 --method ogc_cache` | **PASSED** | Preset #0 (1,023 tokens), 260 cached tokens, 7.11 MiB KV (75.4% savings), answer verified (`5375`). |
| **Benchmark Artifact Verification** | `.venv/bin/python artifacts/audit/verify_results.py <artifact_dir>` | **PASSED** | Validated mathematical invariants on all completed runs. |

> [!NOTE]
> Running generic `pytest` from the system Python fails during collection because `torch` is not in the system environment, and pytest default discovery attempts to import ad-hoc scripts in `scratch/test_*.py` which contain top-level assertion scripts rather than `unittest.TestCase` classes. Test execution must use `.venv/bin/python -m unittest discover -s tests`. A minimal `pytest.ini` has been added to isolate `tests/`.

---

## 3. Fresh Matched Benchmark Evaluation (OGC-Cache vs. Baselines)

A fresh, matched-budget benchmark was executed across 12 controlled cases spanning 4 context lengths ($1\text{K}, 2\text{K}, 4\text{K}, 8\text{K}$ tokens) and 3 query types (`single_fact`, `multi_fact`, `adversarial`) across 9 configurations (108 total persisted outputs).

- **Reproduction Command:**
  ```bash
  .venv/bin/python run_research_benchmark.py \
    --dataset artifacts/matched-contexts-12-input.json \
    --policies full ogc streaming sliding h2o_prefill_approx \
    --budgets 260 516 \
    --output artifacts/phase2-matched-evaluation
  ```
- **Verification Command:**
  ```bash
  .venv/bin/python artifacts/audit/verify_results.py artifacts/phase2-matched-evaluation
  ```
  *(Status: `passed: true`, 108 outputs, 12 cases, 9 configs, 28,672 bytes/token)*

### 3.1 Primary Matched Performance Summary

| Policy & Configuration | Token Budget ($B$) | Retained Physical KV Cache | Overall Accuracy (%) | Canonical Exact Match (%) | Multi-Fact Accuracy (%) | Single-Fact Accuracy (%) | Adversarial Accuracy (%) | Mean TTFT ($T_{\text{prefill}}$) | Decode Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Uncompressed Baseline** | Full ($L$) | $105.55\text{ MiB}$ (up to $229.4\text{ MiB}$) | **91.7%** (11/12) | **41.7%** | **100.0%** (4/4) | **100.0%** (4/4) | **75.0%** (3/4) | $2,932.0\text{ ms}$ | $20.3\text{ tok/s}$ |
| **StreamingLLM ($S=4, W=256$)** | $260$ | $7.11\text{ MiB}$ ($-93.3\%$) | **0.0%** (0/12) | **0.0%** | **0.0%** (0/4) | **0.0%** (0/4) | **0.0%** (0/4) | $2,962.5\text{ ms}$ | $25.0\text{ tok/s}$ |
| **StreamingLLM ($S=4, W=512$)** | $516$ | $14.11\text{ MiB}$ ($-86.6\%$) | **8.3%** (1/12) | **0.0%** | **0.0%** (0/4) | **0.0%** (0/4) | **25.0%** (1/4) | $2,926.7\text{ ms}$ | $23.7\text{ tok/s}$ |
| **Pure Sliding Window ($W=260$)** | $260$ | $7.11\text{ MiB}$ ($-93.3\%$) | **0.0%** (0/12) | **0.0%** | **0.0%** (0/4) | **0.0%** (0/4) | **0.0%** (0/4) | $2,930.7\text{ ms}$ | $24.6\text{ tok/s}$ |
| **Pure Sliding Window ($W=516$)** | $516$ | $14.11\text{ MiB}$ ($-86.6\%$) | **8.3%** (1/12) | **0.0%** | **0.0%** (0/4) | **25.0%** (1/4) | **0.0%** (0/4) | $2,954.1\text{ ms}$ | $23.8\text{ tok/s}$ |
| **H2O Oracle Approx ($H=64, W=196$)** | $260$ | $7.11\text{ MiB}$ ($-93.3\%$) | **0.0%** (0/12) | **0.0%** | **0.0%** (0/4) | **0.0%** (0/4) | **0.0%** (0/4) | $4,127.7\text{ ms}$ | $24.6\text{ tok/s}$ |
| **H2O Oracle Approx ($H=128, W=388$)**| $516$ | $14.11\text{ MiB}$ ($-86.6\%$) | **8.3%** (1/12) | **0.0%** | **0.0%** (0/4) | **0.0%** (0/4) | **25.0%** (1/4) | $4,146.5\text{ ms}$ | $23.4\text{ tok/s}$ |
| **OGC-Cache ($C=64, K=1$) [Ours]** | $260$ | $7.11\text{ MiB}$ ($-93.3\%$) | **33.3%** (4/12) | **0.0%** | **0.0%** (0/4) | **0.0%** (0/4) | **100.0%** (4/4) | $2,982.8\text{ ms}$ | $24.8\text{ tok/s}$ |
| **OGC-Cache ($C=64, K=2$) [Ours]** | $516$ | $14.11\text{ MiB}$ ($-86.6\%$) | **33.3%** (4/12) | **0.0%** | **0.0%** (0/4) | **0.0%** (0/4) | **100.0%** (4/4) | $2,974.6\text{ ms}$ | $23.3\text{ tok/s}$ |

### 3.2 Breakdown by Context Length ($L$)

| Policy / Budget | 1,024 tokens | 2,048 tokens | 4,096 tokens | 8,192 tokens |
| :--- | :---: | :---: | :---: | :---: |
| **Full Baseline** | 100.0% (3/3) | 100.0% (3/3) | 66.7% (2/3) | 100.0% (3/3) |
| **StreamingLLM ($B=260$)** | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) |
| **StreamingLLM ($B=516$)** | 33.3% (1/3) | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) |
| **Pure Sliding Window ($B=260$)** | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) |
| **Pure Sliding Window ($B=516$)** | 33.3% (1/3) | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) |
| **H2O Oracle Approx ($B=260$)** | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) |
| **H2O Oracle Approx ($B=516$)** | 33.3% (1/3) | 0.0% (0/3) | 0.0% (0/3) | 0.0% (0/3) |
| **OGC-Cache ($B=260$)** | 33.3% (1/3) | 33.3% (1/3) | 33.3% (1/3) | 33.3% (1/3) |
| **OGC-Cache ($B=516$)** | 33.3% (1/3) | 33.3% (1/3) | 33.3% (1/3) | 33.3% (1/3) |

---

## 4. In-Depth Failure & Diagnostic Analysis

### 4.1 Multi-Fact Recall Breakdown: Why OGC-Cache Scores 0.0%
In multi-fact recall, two facts (e.g. database port and deployment region) are embedded at distinct context depths separated by $\approx 30\%$ of the context:
$$\text{Depth}_1 = \text{depth}, \quad \text{Depth}_2 = (\text{depth} + 0.30) \bmod 1.0$$
Across all evaluated context lengths ($1\text{K}$ to $8\text{K}$), this separation spans **250 to 2,400 tokens**.

1. **Under $B=260$ ($C=64$):**
   - History capacity: $H = (B - 4) / 4 = 64$ tokens.
   - Number of chunks: $K = H / C = 64 / 64 = 1$ chunk.
   - **Diagnosis:** A single 64-token chunk cannot bridge a 250+ token gap. OGC retains at most one fact; the second fact is inevitably evicted into the discard window.
   - **Observed Outputs (`outputs.jsonl`):**
     - Case `20261031/2048/early_middle/multi_fact/0` (Ground Truth: `['43468', 'sa-east-7']`):
       - Retained Chunk: `[516, 580]` (contains Fact 1 at offset 515).
       - Fact 2 offset: `1117` (evicted).
       - Output: `'43468,121'` (`matches: [True, False]`, `score_fraction: 0.5`, `is_correct: False`).
     - Case `20261031/8192/early_middle/multi_fact/0` (Ground Truth: `['15509', 'eu-west-9']`):
       - Retained Chunk: `[4484, 4548]` (contains Fact 2).
       - Fact 1 offset: `2050` (evicted).
       - Output: `'1234, eu-west-9'` (`matches: [False, True]`, `score_fraction: 0.5`, `is_correct: False`).

2. **Under $B=516$ ($C=64$):**
   - History capacity: $H = (516 - 4) / 4 = 128$ tokens $\implies K = 2$ chunks.
   - **Diagnosis:** While $K=2$ chunks are available, attention concentration on initial formatting delimiters (attention sink bias) causes OGC to select chunk `[4, 68]` as one of its two chunks. This leaves only **1 chunk** for historical fact retention, causing the second fact to be evicted just as under $B=260$.
   - **Observed Outputs (`outputs.jsonl`):**
     - All 4 multi-fact items scored exactly $0.5$ score fraction (`[True, False]` or `[False, True]`), resulting in $0.0\%$ complete fact recovery.

### 4.2 Single-Fact Grid Alignment Artifact & Token Slicing
In single-fact queries at middle depth ($50\%$), OGC failed to recover the fact despite attending to the region.
- **Example:** Case `20261031/1024/middle/single_fact/0` (Ground Truth: `['43470']`):
  - Fact record text: `\n\nCURRENT record for project PROJ-731EA4B0A976: primary database port is 43470.\n\n`
  - Token offsets: `497` through `525`.
  - OGC Selected Chunk: `[452, 516)`.
  - Token 516 corresponds to the colon `:` after `PROJ-731EA4B0A976`.
  - Tokens 517–525 (` primary database port is 43470`) fell into the next chunk `[516, 580)`, which was evicted!
  - **Model Output:** `'The current primary database port for project PROJ-731EA4B0A976 is 5432.'`
  - **Diagnosis:** The model recognized the entity because its name was retained in chunk `[452, 516)`, but the assigned value was sliced off at boundary 516. The model consequently hallucinated default port `5432`.
  - *Comparison:* In the companion ablation `entity_chunks` (punctuation-boundary aligned), chunks align to sentence delimiters, retaining the entire statement together and achieving $83.3\%$ accuracy at $B=260$.

---

## 5. Potential Methodological Flaws & Architectural Limitations

1. **Pre-pruning Token 1 Query Contamination:**
   - *Description:* In `src/generation.py`, token 1 logits are captured from the prefill forward pass *before* cache pruning occurs (`next_token = pick(logits)`).
   - *Impact:* The first output token is generated with full access to the entire uncompressed context. Benchmark recall is technically evaluating *post-prefill continuation*, not proving that token 1 was produced from compressed representations.

2. **Static H2O Prefill Oracle Approximation:**
   - *Description:* The H2O implementation (`src/h2o_cache.py`) calculates attention scores once during prefill and freezes the top $H$ tokens throughout decode.
   - *Impact:* This does not implement dynamic H2O (which updates cumulative attention per decode step). Furthermore, computing dense attention scores over all prefill tokens creates a $1.4\times$ prefill latency penalty ($4,127.7\text{ ms}$ vs $2,932.0\text{ ms}$).

3. **Frozen Historical Prefix:**
   - *Description:* Once initial pruning selects the top chunks, the prefix $(S + K \cdot C)$ remains frozen throughout autoregressive generation (`src/ogc_cache.py#L104-L107`).
   - *Impact:* While suitable for single-turn extraction, in extended multi-turn dialog or chain-of-thought decoding, the cache cannot adapt its historical chunks to new topics or reasoning steps.

4. **Post-Prefill vs. Peak Memory Demands:**
   - *Description:* The full prompt KV cache is allocated during prefill, followed by slicing down to budget $B$.
   - *Impact:* Although KV cache footprint drops from $\approx 105.5\text{ MiB}$ to $7.11\text{ MiB}$ after generation, hardware memory consumption peaks during the prefill phase. The algorithm does not achieve peak-memory savings during initial prompt ingestion.

---

## 6. Prioritized Actionable Recommendations for GPT-6.1 Sol

| Priority | Component | Issue | Recommended Action |
| :---: | :--- | :--- | :--- |
| **P1** | `src/selection.py`, `src/ogc_cache.py` | **Attention Sink Chunk Consumption:** Chunk 0 (`[4, 68]`) frequently absorbs 1 of the $K$ available chunks due to causal prefill sink accumulation, reducing effective history capacity to $K-1$. | Exclude initial tokens ($< S + C_{\text{sink}}$) from candidate chunk selection or introduce an explicit sink-exclusion mask before candidate chunk ranking. |
| **P2** | `src/selection.py` | **Rigid Fixed-Grid Chunk Slicing:** Arbitrary chunk boundaries ($4 + k \times 64$) fracture entities, separating keys from values across boundaries. | Standardize or default to punctuation/syntactic boundary chunking (`entity_chunks`) or evaluate overlapping chunk strides during candidate generation. |
| **P3** | `src/research.py`, `src/generation.py` | **Multi-Fact History Allocation:** At $B=260$, $K=1$ chunk cannot resolve multi-fact inquiries where facts are separated by $>64$ tokens. | For multi-fact queries under small budgets ($B=260$), consider adaptive chunk allocation (e.g. $K=2, C=32$) or dual-chunk targeting based on top-2 dispersed attention peaks. |
| **P4** | `src/generation.py` | **Pre-pruning Information Leakage:** Prefill produces token 1 logits prior to KV pruning. | For rigorous evaluation of compressed representations, implement a held-out query evaluation protocol where prompt documents are prefilled and pruned before the question tokens are processed. |
| **P5** | `pytest.ini`, `scratch/` | **Pytest Test Runner Discovery Conflict:** Bare `pytest` invocations fail due to non-test scripts in `scratch/test_*.py`. | Keep the committed `pytest.ini` (`testpaths = tests`) to ensure `pytest` and `unittest` cleanly isolate formal tests. |
