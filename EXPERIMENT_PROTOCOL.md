# Written Experiment Protocol: Squeezing Long-Context Inference into Less Memory

**Document Version:** 1.0.0  
**Phase:** PHASE 2 — Design the Experimental Protocol  
**Date:** 2026-10-19  
**Status:** Approved for Implementation (Phase 3 & Beyond)  

---

## 1. Objective and Scientific Scope

The objective of this research is to evaluate Key-Value (KV) cache compression techniques for long-context autoregressive language model inference. Long-context documents and chat histories place heavy demands on GPU/unified system memory because standard Transformer architectures retain key and value tensors for every past token.

This protocol establishes the experimental methodology, model specification, test matrix, evaluation dataset design, metric definitions, and acceptance criteria required to measure:
1. **Physical KV-cache memory reduction** achieved by cache pruning/eviction.
2. **Peak inference memory impacts** during prefill and decoding phases.
3. **Recall accuracy degradation** across varying document depths and context lengths.
4. **Latency and throughput trade-offs** under constrained cache budgets.

---

## 2. Model Selection and Specification

### 2.1 Primary Benchmark Model: `Qwen/Qwen2.5-1.5B-Instruct`
To satisfy the mandatory requirement for a 1–3B parameter model, we select **`Qwen/Qwen2.5-1.5B-Instruct`**.

| Attribute | Specification | Justification |
| :--- | :--- | :--- |
| **Model Hub ID** | `Qwen/Qwen2.5-1.5B-Instruct` | Official Alibaba Qwen repository |
| **Model Revision** | `main` | Exact commit hash to be pinned upon download |
| **Parameter Count** | **1,543,714,304 (1.54B)** | Directly in the 1–3B requirement range |
| **Architecture** | `Qwen2ForCausalLM` | Modern Transformer with Grouped-Query Attention (GQA) and RoPE |
| **Number of Layers** | 28 | Standard deep architecture |
| **Hidden Size ($d_{\text{model}}$)** | 1,536 | Projection dimension |
| **Attention Heads ($n_{\text{head}}$)** | 12 | Query heads |
| **KV Heads ($n_{\text{kv}}$)** | **2** | Grouped-Query Attention (6:1 query-to-KV ratio) |
| **Head Dimension ($d_{\text{head}}$)** | 128 | $1536 / 12 = 128$ |
| **Native Context Length** | **32,768 tokens** | Supports extensive long-context benchmarks without out-of-context distortion |
| **Tokenizer** | `Qwen2TokenizerFast` (BPE, 152,064 vocab) | Byte-level BPE with high tokenization density |
| **Weight Precision** | `torch.float16` (FP16) / `torch.bfloat16` | Standard inference precision |
| **Static Weight Size** | **~3.08 GB** | Consumes only ~19.3% of 16 GB unified RAM, leaving ~12.9 GB for KV cache and activations |
| **License** | **Apache 2.0** | Permissive, open-weights, no gating or authentication barriers |
| **Instruction Tuning** | Instruct-tuned | Enables reliable, deterministic extraction in question-answering tasks |

### 2.2 Secondary Debugging / Pilot Model: `Qwen/Qwen2.5-0.5B-Instruct`
* **Parameters:** 0.49 Billion (`Qwen/Qwen2.5-0.5B-Instruct`).
* **Role:** Used exclusively as a fast sanity check during code construction (Phase 3 & Phase 5) to verify tensor slicing hooks and data flow before initiating large benchmark runs with the 1.5B model.

### 2.3 Runtime Execution Environment
* **Hardware:** Apple M5 (arm64, 16.0 GB Unified Memory).
* **Execution Devices:**
  * **Primary:** Apple Silicon Metal Performance Shaders (`device="mps"`).
  * **Validation Fallback:** Apple Silicon CPU (`device="cpu"`), used to cross-verify numerical equivalence and ensure reproducibility regardless of platform accelerator.
* **Inference Runtime:** Hugging Face `transformers` v4.45+ / v5.x with PyTorch 2.x inside Python 3.12.14 `.venv`.

---

## 3. Experiment Matrix

The experimental matrix tests multiple input-context lengths against varying cache budgets, comparing an uncompressed baseline against the primary compression method: **StreamingLLM (Attention Sinks + Recent Rolling Window)**.

### 3.1 Input-Context Lengths ($L$)
Based on Phase 1 profiling of memory bandwidth and context capacity, we establish four standardized context lengths:
1. **$L_1 = 1,024$ tokens:** Short-context calibration and baseline verification.
2. **$L_2 = 2,048$ tokens:** Medium-context standard regime.
3. **$L_3 = 4,096$ tokens:** Long-context regime where KV cache memory begins to strain standard deployments.
4. **$L_4 = 8,192$ tokens:** Extended long-context regime demonstrating high-compression savings.

### 3.2 Cache Strategies and Budget Formulations

#### Configuration A: Full Uncompressed Baseline
* **Eviction Policy:** None (Full Dynamic Cache).
* **Retained Tokens:** $100\%$ of sequence length ($L$).
* **Cache Budget:** Unbounded ($B = L$).

#### Configuration B: StreamingLLM (Attention Sinks + Rolling Window)
* **Sink Budget ($S$):** Fixed at $S = 4$ initial tokens (positions 0, 1, 2, 3), retaining attention sinks that stabilize the softmax distribution.
* **Recent Window Budget ($W$):** Evaluated at three distinct operational budgets:
  1. **Budget 1 ($B_1$):** $W = 256 \implies \text{Total Cache Budget } B_1 = S + W = 4 + 256 = 260$ tokens.
  2. **Budget 2 ($B_2$):** $W = 512 \implies \text{Total Cache Budget } B_2 = S + W = 4 + 512 = 516$ tokens.
  3. **Budget 3 ($B_3$):** $W = 1024 \implies \text{Total Cache Budget } B_3 = S + W = 4 + 1024 = 1028$ tokens.

### 3.3 Complete Evaluation Matrix (16 Conditions)

| Condition ID | Context Length ($L$) | Strategy | Sink Tokens ($S$) | Recent Window ($W$) | Retained Tokens ($B$) | Theoretical KV Cache |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | 1,024 | Full Baseline | — | — | 1,024 (100%) | 28.67 MB |
| **C2** | 1,024 | StreamingLLM ($B_1$) | 4 | 256 | 260 (25.4%) | 7.28 MB |
| **C3** | 1,024 | StreamingLLM ($B_2$) | 4 | 512 | 516 (50.4%) | 14.45 MB |
| **C4** | 1,024 | StreamingLLM ($B_3$) | 4 | 1,024 | 1,024 (100%)* | 28.67 MB |
| **C5** | 2,048 | Full Baseline | — | — | 2,048 (100%) | 57.34 MB |
| **C6** | 2,048 | StreamingLLM ($B_1$) | 4 | 256 | 260 (12.7%) | 7.28 MB |
| **C7** | 2,048 | StreamingLLM ($B_2$) | 4 | 512 | 516 (25.2%) | 14.45 MB |
| **C8** | 2,048 | StreamingLLM ($B_3$) | 4 | 1,024 | 1,028 (50.2%) | 28.78 MB |
| **C9** | 4,096 | Full Baseline | — | — | 4,096 (100%) | 114.69 MB |
| **C10** | 4,096 | StreamingLLM ($B_1$) | 4 | 256 | 260 (6.3%) | 7.28 MB |
| **C11** | 4,096 | StreamingLLM ($B_2$) | 4 | 512 | 516 (12.6%) | 14.45 MB |
| **C12** | 4,096 | StreamingLLM ($B_3$) | 4 | 1,024 | 1,028 (25.1%) | 28.78 MB |
| **C13** | 8,192 | Full Baseline | — | — | 8,192 (100%) | 229.38 MB |
| **C14** | 8,192 | StreamingLLM ($B_1$) | 4 | 256 | 260 (3.2%) | 7.28 MB |
| **C15** | 8,192 | StreamingLLM ($B_2$) | 4 | 512 | 516 (6.3%) | 14.45 MB |
| **C16** | 8,192 | StreamingLLM ($B_3$) | 4 | 1,024 | 1,028 (12.5%) | 28.78 MB |

*\*Note: In condition C4 where prompt length $\le S+W$, no eviction is triggered.*

---

## 4. Controlled Long-Context Recall Dataset Design

To evaluate whether KV-cache compression degrades the model's ability to recall earlier information, we design a deterministic, synthetic evaluation benchmark: **Multi-Depth Needle-in-a-Haystack (MD-NIAH)**.

### 4.1 Document Generation & Distractor Structure
* **Domain:** Technical enterprise operations manuals and microservice deployment manifests.
* **Distractor Content:** Realistic paragraphs detailing cloud infrastructure, server metrics, network routing rules, and incident reports. Distractors are deterministic, seeded with fixed random seed `42`.
* **Zero Prior Knowledge:** All target facts consist of randomly generated high-entropy identifiers that cannot be guessed through pretraining prior knowledge (e.g. project codes, security salts, port numbers).

### 4.2 Needle Placements (Document Depths)
Each test document embeds verifiable key-value facts at 5 distinct normalized depth intervals:
1. **Depth 1 — Start ($5\%$ depth):** Placed right after introductory system tokens.
2. **Depth 2 — Early-Middle ($25\%$ depth):** Placed in the first quarter of the document.
3. **Depth 3 — Middle ($50\%$ depth):** Placed at the exact midpoint of the document.
4. **Depth 4 — Late-Middle ($75\%$ depth):** Placed in the third quarter of the document.
5. **Depth 5 — End ($95\%$ depth):** Placed immediately before the query prompt.

### 4.3 Query Types and Target Structure
* **Single-Fact Retrieval:** "What is the project code assigned to the cluster in the documentation? Answer with only the exact code."
  - *Example ground truth:* `OMEGA-8419`
* **Multi-Fact Relational Retrieval:** "Which database port is configured for the project with code `OMEGA-8419`? Answer with only the port number."
  - *Example ground truth:* `5438`
* **Scoring Mechanism:** Exact Match (EM) and Normalized Substring Matching (case-insensitive, whitespace-trimmed). An answer is marked correct if and only if the exact target identifier is recovered in the model's output.

---

## 5. Quantitative Metric Definitions and Formulas

### 5.1 KV-Cache Memory Savings ($\Delta \text{Mem}_{\text{KV}}$)
Measures the percentage reduction in physical memory occupied by retained Key and Value cache tensors:
$$\Delta \text{Mem}_{\text{KV}} = 100 \times \frac{\text{Mem}_{\text{KV}}^{\text{baseline}} - \text{Mem}_{\text{KV}}^{\text{compressed}}}{\text{Mem}_{\text{KV}}^{\text{baseline}}}$$
* **Units:** Percentage ($\%$).
* **Sign Convention:** Positive indicates memory reduction (savings). $0\%$ indicates no savings.

### 5.2 Peak Inference Memory Savings ($\Delta \text{Mem}_{\text{peak}}$)
Measures the percentage reduction in total peak process memory reached during the end-to-end inference lifecycle:
$$\Delta \text{Mem}_{\text{peak}} = 100 \times \frac{\text{Mem}_{\text{peak}}^{\text{baseline}} - \text{Mem}_{\text{peak}}^{\text{compressed}}}{\text{Mem}_{\text{peak}}^{\text{baseline}}}$$
* **Units:** Percentage ($\%$).
* **Sign Convention:** Positive indicates peak memory reduction.

### 5.3 Recall Accuracy ($\text{Acc}_{\text{recall}}$)
Measures the proportion of evaluation questions answered with exact ground-truth fidelity:
$$\text{Acc}_{\text{recall}} = 100 \times \frac{\sum_{i=1}^{Q} \mathbb{I}(\text{pred}_i == \text{target}_i)}{Q}$$
* **Units:** Percentage ($\%$).
* **Range:** $0.0\%$ to $100.0\%$.

### 5.4 Absolute Accuracy Change ($\Delta \text{Acc}$)
Measures the degradation or change in recall accuracy resulting from cache compression relative to the full-cache baseline:
$$\Delta \text{Acc} = \text{Acc}_{\text{recall}}^{\text{compressed}} - \text{Acc}_{\text{recall}}^{\text{baseline}}$$
* **Units:** Percentage points ($\text{pp}$).
* **Sign Convention:** Negative value indicates accuracy degradation. $0.0\text{ pp}$ indicates lossless compression. Positive indicates improved retrieval.

### 5.5 Latency and Throughput Metrics
* **Prefill Latency ($T_{\text{prefill}}$):** Elapsed time (ms) to encode prompt tokens and populate initial KV cache.
* **Time-to-First-Token (TTFT):** Elapsed time (ms) from prompt submission to first generated token.
* **Decoding Latency ($T_{\text{decode}}$):** Average milliseconds elapsed per generated token during autoregression:
  $$T_{\text{decode}} = \frac{T_{\text{total\_gen}} - T_{\text{prefill}}}{N_{\text{generated\_tokens}}}$$
* **Generation Throughput ($\Phi_{\text{gen}}$):** Tokens per second during decoding ($1000 / T_{\text{decode}}$).

---

## 6. Memory Tier Architecture & Instrumentation Protocol

To prevent conflation of different memory quantities, measurements are categorized into five rigorous tiers:

| Tier | Memory Quantity | Type | Instrumentation Source |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Physical KV-Cache Tensor Memory** | **Actual Runtime** | Direct tensor introspection: $\sum_{l=1}^{28} \left( \text{numel}(K_l) \times \text{elem\_size}(K_l) + \text{numel}(V_l) \times \text{elem\_size}(V_l) \right)$ |
| **Tier 2** | **Static Model Weight Memory** | **Actual & Theoretical** | $\sum_{p \in \text{params}} p.\text{numel}() \times p.\text{element\_size}() \approx 3.08 \text{ GB}$ (FP16) |
| **Tier 3** | **Peak Process Resident Memory (RSS)** | **Actual Runtime** | `psutil.Process().memory_info().rss` and `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` |
| **Tier 4** | **Metal Device Allocated Memory** | **Actual Runtime** | `torch.mps.current_allocated_memory()` (Apple Silicon MPS driver pool) |
| **Tier 5** | **Transient Prefill Activation Memory** | **Calculated / Estimated** | Difference: $\text{Peak RSS during prefill} - (\text{Weights} + \text{Full KV Cache})$ |

### Lifecycle Checkpoints
Every experimental run must sample memory at five distinct lifecycle checkpoints:
1. **Checkpoint 0 (Pre-load):** Initial empty process memory.
2. **Checkpoint 1 (Post-load):** Memory after model and tokenizer weights are instantiated and warmed up.
3. **Checkpoint 2 (Peak Prefill):** Peak memory reached during prompt matrix multiplication and attention computation.
4. **Checkpoint 3 (Post-Eviction Quiescent):** Memory immediately following cache pruning (for compressed runs) or prompt prefill completion (for baseline).
5. **Checkpoint 4 (Peak Decoding):** Peak memory reached during multi-token autoregressive generation.

---

## 7. Experimental Acceptance Criteria

Before declaring subsequent phases successful, the implementation must satisfy the following explicit criteria:

1. **Baseline Correctness (Phase 3):**
   * The uncompressed baseline must achieve **$100\%$ accuracy** on the test dataset at $L=1024$ and $L=2048$ within native context.
   * Calculated KV-cache size must match the theoretical formula within $0.01\%$.
2. **Compression Integrity (Phase 5):**
   * The cache adapter must perform **physical tensor eviction** (reducing tensor shape along the sequence dimension). Masking attention without reducing tensor shape is strictly rejected.
   * Evicted cache tensors must yield $\ge 70\%$ KV memory reduction at $L=4096$ with budget $B_3$ ($1028$ tokens).
3. **Evaluation Realism (Phase 6):**
   * The experiment must empirically expose the fundamental trade-off curve: as budget $W$ shrinks, facts placed at middle depths ($25\%$, $50\%$) will be evicted, producing a measurable drop in recall accuracy that cleanly contrasts with the baseline.
4. **Reproducibility:**
   * All results must be generated via deterministic seeds and reproducible command-line execution scripts.

---

## 8. Protocol Sign-Off

* **Author:** Antigravity Autonomous Research Assistant
* **Approval Status:** Ready for execution in Phase 3 (Build the Uncompressed Baseline).

## 9. Audit protocol v2 (2026-10-31; supersedes experimental claims above)

Historical phase 6–8 JSON and plots are preserved verbatim. They used different
dtypes, a 30-question suite, estimated OGC bytes, and an incorrect OGC generation
loop. They are exploratory historical evidence, not validation of the repaired
implementation. Current runners never merge them into fresh measurements.

### Inference contract

- One unpadded Qwen2 sequence with DynamicCache; pinned model and tokenizer revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- First output token is selected from the final prefill logits. Subsequent forwards
  consume the previously generated token once, at its original absolute position.
  The last generated token is not yet consumed and is absent from the final cache.
- Cached RoPE keys keep their original phases. No position rebasing or rerotation.
  Explicit absolute `position_ids`; `cache_position` is supplied when declared by
  the model API. Transformers 5.20.1 Qwen2 does not declare it. Decode uses an
  explicit compact all-visible mask for its single query. No static or sliding
  architecture caches, padded batches, beam search, or dynamic RoPE scaling are
  validated by this work.
- Stop at configured model/tokenizer EOS or the requested generation count;
  newline is ordinary output. Zero output tokens means prefill only; negative
  counts and inputs beyond the consumed model context are rejected.
- Compressed caches contain `min(consumed_tokens, B)` entries after selection and
  each decode step. A forward can transiently allocate B+1 entries before pruning.
  OGC freezes selected history and rotates the recent window. H2O-style retains
  static prefill heavy hitters, not dynamically accumulated decode heavy hitters.

### Memory and timing

All methods use the same eager backend, BF16, greedy argmax, repetition penalty
1.0, EOS rules, output cap, and prefill block size (default 256) in a comparison.
All weights are shared; each measured method reruns its own prefill and decode.
Model loading and one warmup per configuration are excluded. Config order is
randomized independently for each question with a recorded seed.

Block prefill bounds live attention outputs by O(layers × heads × block_size × L)
instead of retaining all L×L matrices. Query observation scores and full-prefill
H2O-style scores are accumulated and released block by block. Attention arithmetic
is still quadratic in L. The full prompt KV is materialized before pruning:
**this is post-prefill KV compression, not a demonstrated peak-memory saving**.
The public direct pruning helpers still accept dense attentions for tests;
`generate` does not request a full L×L prefill.

MPS/CUDA synchronization surrounds measured work. Timings are: block prefill
(including scoring), prefill forwards, score reduction, initial selection/pruning,
decode forwards, decode pruning, decode wall time, TTFT, and end-to-end wall time
(including tokenization and final text decoding). Selection time sums initial and
decode pruning; it overlaps decode wall time and must not be added to it again.
TTFT is wall time through initial selection and the first token's materialization.
Decode throughput counts only subsequent forward-produced tokens (N−1, excluding
the token obtained from prefill). Generation count and decode step count are saved.

Final tensor bytes are summed from retained K/V tensors after generation. Unique
backing-storage bytes are saved separately to detect views that retain evicted
storage. The uncompressed counterfactual uses observed full-prefill bytes/token
at the same *consumed* length, not the requested output cap. Actual full-baseline
bytes are available for each paired question; different EOS lengths can make
these differ from the counterfactual. No theoretical value is presented as a
measured compressed cache size.

MPS allocation values are checkpoint samples, not peaks. Process RSS peak is
process-lifetime high water, not resettable per question. CUDA allocator peak is
reset per measured generation, but CUDA is unverified on the current hardware.
A causal claim about peak-memory reduction requires isolated processes and
appropriate peak instrumentation; no such claim is made here.

### Research matrix and uncertainty

Default matrix: full baseline plus StreamingLLM-style sinks/recent (absolute RoPE,
not the original positional-shift implementation), pure sliding window, static
H2O-style prefill oracle approximation, OGC, query-aware individual tokens,
seeded random contiguous chunks, adaptive chunk sizes, and punctuation-boundary
chunks, each at B=260/516/1028 (25 conditions). OGC allocates S=4 and H=(B−4)/4
tokens to history, with W=B−4−H recent tokens. H2O-style uses the same H and W=B−H.
OGC uses disjoint candidate chunks, ordered by mean observation attention; its
candidate grid differs from the historical overlapping greedy windows. The
random control uses the identical grid and history/recent split. Run fixed C=16,
32,64 ablations with `--chunk-size` in separate fresh output directories.

Adaptive retention chooses C=16/32/64 from predeclared attention concentration
thresholds 0.5/0.25. Boundary retention uses token punctuation to approximate
sentence/record boundaries, with spans capped at C; it is not NER. These are
research hypotheses, not established improvements. Selection never receives
answers or ground-truth needle offsets. Every condition is reported, including
negative results, together with its recall-memory Pareto status.

Default synthetic suite: 4 context lengths (1K/2K/4K/8K) × 5 depths × 3 question
types × 5 independent seeds = 300 cases. Facts are independently generated per
case, multi-fact records occur at separated depths, and adversarial cases place
obsolete conflicting records near the inquiry. Token counts and actual fact
depths are measured after final chat-template tokenization. Ground truths, seeds,
complete prompts, outputs, token IDs, retained positions, and per-step cache
lengths are persisted. Supplied historical datasets are scored separately.

Report all-facts boundary-matched accuracy, canonical exact match, strata, and
paired accuracy differences with 2,000 seeded bootstrap resamples stratified by
length/depth/type. Singleton pilot strata provide no resampling variance and are
not evidence of precise uncertainty. CIs describe this synthetic population,
not general long-context performance; no multiple-comparison adjustment is made.

All policies use full-prefill logits for token 1. This token has already seen all
facts before pruning. Report this shared-information limitation: these are
post-prefill generation tests, not a strict demonstration that every recalled
fact was read solely from compressed KV. Future held-out-query evaluation should
prune before the question is consumed.

Fresh outputs require a new `--output` directory. Manifests save the immutable
revision, source hashes, git state, versions, seed, dataset hash, settings, command,
status, and timestamps. Correctness tests precede the benchmark. No minimum
accuracy outcome is an acceptance gate: the earlier 100% accuracy and expected
middle-depth degradation requirements are hypotheses, not results to enforce.
