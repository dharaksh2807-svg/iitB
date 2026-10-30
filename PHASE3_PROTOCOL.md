# Phase 3 frozen evaluation protocol

Base: `486b9b0fc9410c943f108ea57030d42db68538af`; implementation `04f2821`.
Default model and tokenizer remain pinned to Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16, eager attention, block size 256,
greedy decoding, EOS stop, 35 answer tokens. No frontend changes.

Development: four 1K questions, including two Phase 2 examples and two new
questions from seed 2026110300. Both continuation and held-query protocols run.
No accuracy claim is based on these four examples.

Freeze rule: no algorithm, capacity split, score threshold or prompt tuning after
held-out inference begins. Any implementation defect requires a separately
identified new run; raw results are never edited.

Held-out cohort: seed 2026110401, lengths 1024/2048, five depths, three types
(single/multi/adversarial), two independent repetitions per cell: 60 questions.
All 12 configurations run at B=260: full, streaming, sliding, static H2O
approximation, original OGC, residual-attention OGC, existing boundary/adaptive/
random controls, C32/K2 OGC, larger-history OGC, combined record selection.
Paired bootstrap uses the existing 2,000-resample stratified method, min 2 per cell.

Budget/context pilot: independent seed 2026110501, lengths 1024/2048/4096/8192,
one early-middle multi-fact, middle single-fact, late-middle adversarial case per
length: 12 questions. Compare full, streaming, sliding, static H2O approximation,
original OGC and combined record selection at B=260/516/1028: 16 conditions.
Intervals are withheld for singleton strata. This pilot is descriptive.

Both held-out experiments use document -> compress -> question -> answer.
The full chat prompt is tokenized once, then partitioned at the explicit Inquiry
boundary; no question token is forwarded before document pruning. Selection may
read the known question text lexically, but never benchmark answers or needle
offsets. Token 1 uses the final post-compression query forward's logits.
Question tokens count inside B, and pruning runs after every query/decode token.
Absolute RoPE positions are preserved. Cache length is min(consumed, B) after
each step; a forward can transiently allocate B+1 entries.

Original OGC keeps H=(B-4)/4 history slots. C32 changes granularity only.
Larger-history OGC and the combined policy use H=(B-4)/2 and W=B-4-H.
Combined selection: subtract per-head persistent attention floors; rank whole
sentence/newline records by generic query-identifier match, marginal query-word
coverage, then residual attention. Keep fitting complete records before filling
spare slots. No benchmark-specific project pattern, CURRENT/ARCHIVED rule, gold
value or gold offset is used. Oversized records cannot be guaranteed atomic.
The default engine and historical continuation protocol remain unchanged.

Primary success rule (declared before held-out inference): a positive lower
paired 95% bound for combined minus original OGC all-facts recall on the 60-case
cohort; no negative point change by context or query type; canonical EM must not
fall overall. All results and ablations, including negative ones, are reported.
Even a pass is evidence only for this structured synthetic population; the small
4K/8K pilot does not support a general default or accuracy guarantee. No multiple
comparison adjustment; analyses are exploratory.

Record physical final K/V payload and unique storage, query/selection/prefill/
decode/TTFT/wall times, checkpoint allocations and available CUDA peak counters.
MPS checkpoint samples and retained bytes are NOT peak-memory measurements.
BF16 batching can change later greedy tokens: a strict 12-token parity check
against batched HF diverged after six matching tokens. That negative log remains;
first-token HF parity, FP32 full-token parity, positional and storage tests pass.

Write new output directories, manifests, hashes, per-question raw predictions,
IDs and cache histories. Validate all artifacts independently before reporting.
Preserve all earlier files and the pre-existing uncommitted Phase 2 report edit.
Before pushing main, require remote HEAD to remain the recorded base above;
otherwise stop without merge or force push.
