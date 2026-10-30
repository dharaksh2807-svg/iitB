"""Shared single-sequence inference and telemetry (Qwen2 DynamicCache).

All policies prefill the full prompt in bounded query blocks. Full prompt KV
is still materialized: compression occurs after prefill, not during it.
Cached keys keep their original RoPE phase; positions are never renumbered.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.metrics import MemoryTracker, calculate_kv_cache_bytes, synchronize_device

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@dataclass(kw_only=True)
class GenerationTelemetry:
    evaluation_protocol: str = "continuation"
    document_tokens: int = 0
    query_tokens: int = 0
    query_time_ms: float = 0.0
    query_forward_time_ms: float = 0.0
    query_pruning_time_ms: float = 0.0
    query_cache_lengths: list[int] = field(default_factory=list)
    document_retained_positions: list[int] = field(default_factory=list)
    scoring_time_ms: float = 0.0
    prefill_forward_time_ms: float = 0.0
    selection_time_ms: float = 0.0  # initial selection + pruning on all decode steps
    initial_selection_time_ms: float = 0.0
    decode_pruning_time_ms: float = 0.0
    decode_forward_time_ms: float = 0.0
    decode_steps: int = 0  # excludes token selected directly from prefill logits
    generated_token_ids: list[int] = field(default_factory=list)
    retained_positions: list[int] = field(default_factory=list)
    cache_lengths: list[int] = field(default_factory=list)
    prefill_kv_bytes: int = 0
    cache_storage_bytes: int = 0
    cuda_peak_allocated_bytes: int | None = None
    memory_scope: str = "block-prefill scoring; full prompt KV; post-prefill compression; peak savings unverified"
    cache_position_mode: str = ""


def cache_length(cache: Any) -> int:
    return int(cache.layers[0].keys.shape[-2])


def retain(cache: Any, indices: list[int]) -> Any:
    """Allocate compact tensors, including for contiguous slices (views keep storage)."""
    if indices == list(range(cache_length(cache))):
        return cache
    for layer in cache.layers:
        if layer.keys is not None:
            idx = torch.tensor(indices, device=layer.keys.device, dtype=torch.long)
            layer.keys = layer.keys.index_select(-2, idx)
            layer.values = layer.values.index_select(-2, idx)
    return cache


def prefix_recent_indices(length: int, prefix: int, recent: int) -> list[int]:
    if length <= prefix + recent:
        return list(range(length))
    return list(range(prefix)) + list(range(length - recent, length))


def run_generation(engine: Any, prompt: str, max_new_tokens: int, temperature: float,
                   do_sample: bool, select: Callable | None = None,
                   prune: Callable | None = None, capture_attentions: bool = False) -> dict:
    if engine.model is None or engine.tokenizer is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if do_sample and temperature <= 0:
        raise ValueError("Sampling requires positive temperature")
    device = engine.device
    synchronize_device(device)
    wall_start = time.perf_counter()  # includes tokenization and output decoding
    inputs = engine.tokenizer(prompt, return_tensors="pt").to(device)
    ids = inputs["input_ids"]
    n = int(ids.shape[1])
    if n == 0 or n + max(0, max_new_tokens - 1) > engine.max_context_length:
        raise ValueError("Prompt plus consumed generation tokens exceeds model context (or empty prompt)")
    if ids.shape[0] != 1 or not bool(inputs.get("attention_mask", torch.ones_like(ids)).all()):
        raise ValueError("Only one unpadded sequence is supported")
    split = getattr(engine, "document_token_count", None)
    doc_n = n if split is None else int(split)
    if not 0 < doc_n <= n:
        raise ValueError("Document split must be inside the tokenized prompt")
    protocol = "held_query" if doc_n < n else "continuation"
    tracker = MemoryTracker(device)
    tracker.record("inference_start")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    signature = inspect.signature(engine.model.forward).parameters
    has_cache_position = "cache_position" in signature
    synchronize_device(device)
    start = time.perf_counter()
    cache = None
    scores = torch.zeros(doc_n, device=device, dtype=torch.float32) if capture_attentions else None
    corrected = getattr(engine, "attention_score_correction", False)
    floors = {}
    block_size = getattr(engine, "prefill_chunk_size", 256)
    if block_size < 1:
        raise ValueError("prefill_chunk_size must be positive")
    observation_start = max(0, doc_n - getattr(engine, "query_window", doc_n))
    scoring_ms = prefill_forward_ms = 0.0
    with torch.inference_mode():
        for block_start in range(0, doc_n, block_size):
            end = min(doc_n, block_start + block_size)
            query_positions = torch.arange(block_start, end, device=device)
            # Absolute causal block mask; keys are still dense during prefill.
            allowed = torch.arange(end, device=device)[None, :] <= query_positions[:, None]
            mask = torch.zeros((end - block_start, end), device=device, dtype=engine.dtype)
            mask.masked_fill_(~allowed, torch.finfo(engine.dtype).min)
            kwargs = {"logits_to_keep": 1} if "logits_to_keep" in signature else {}
            if has_cache_position:
                kwargs["cache_position"] = query_positions
            observe = capture_attentions and end > observation_start
            synchronize_device(device)
            forward_start = time.perf_counter()
            output = engine.model(input_ids=ids[:, block_start:end], past_key_values=cache,
                                  position_ids=query_positions[None], attention_mask=mask[None, None],
                                  use_cache=True, output_attentions=observe, **kwargs)
            synchronize_device(device)
            prefill_forward_ms += (time.perf_counter() - forward_start) * 1000
            cache = output.past_key_values
            scoring_start = time.perf_counter()
            if observe:
                if not output.attentions or any(a is None for a in output.attentions):
                    raise ValueError("Scoring requires an attention backend that returns weights")
                offset = max(0, observation_start - block_start)
                for layer_index, attention in enumerate(output.attentions):
                    observed = attention[0, :, offset:, :].float()
                    scores[:end] += observed.sum(dim=(0, 1))
                    if corrected:
                        minimum = observed.amin(dim=1)
                        if layer_index not in floors:
                            floors[layer_index] = torch.zeros((minimum.shape[0], doc_n), device=device)
                            floors[layer_index][:, :end] = minimum
                        else:
                            floors[layer_index][:, :end] = torch.minimum(floors[layer_index][:, :end], minimum)
                        del minimum
                    del observed
                del attention
                synchronize_device(device)
                scoring_ms += (time.perf_counter() - scoring_start) * 1000
            if end == doc_n:
                logits = output.logits[:, -1, :].clone()
            del output  # at most one block of attention outputs remains live
    if corrected and capture_attentions:
        correction_start = time.perf_counter()
        for floor in floors.values():
            scores -= floor.sum(dim=0) * (doc_n - observation_start)
        scores.clamp_min_(0)
        synchronize_device(device)
        scoring_ms += (time.perf_counter()-correction_start)*1000
        del floors
        del floor
    synchronize_device(device)
    prefill_ms = (time.perf_counter() - start) * 1000
    prefill_bytes = calculate_kv_cache_bytes(cache)[0]
    tracker.record("post_prefill", cache)
    tracker.record("peak_prefill_raw", cache)  # legacy name; checkpoint, NOT a true peak
    positions = list(range(doc_n))
    synchronize_device(device)
    start = time.perf_counter()
    if select is not None:
        cache, indices = select(cache, scores, ids[:, :doc_n])
        positions = [positions[i] for i in indices]
    del scores  # selection input is a single accumulated score vector
    synchronize_device(device)
    initial_selection_ms = (time.perf_counter() - start) * 1000
    tracker.record("post_prefill_pruned", cache)
    budget = getattr(engine, "cache_budget", getattr(engine, "budget_tokens", None))
    if budget is not None and cache_length(cache) > budget:
        raise AssertionError("Prefill pruning exceeded cache budget")
    document_positions = positions.copy()
    query_lengths = [cache_length(cache)]
    query_forward_ms = query_prune_ms = 0.0
    synchronize_device(device)
    query_start = time.perf_counter()
    with torch.inference_mode():
        for absolute in range(doc_n, n):
            mask = torch.zeros((1, 1, 1, cache_length(cache)+1), device=device, dtype=engine.dtype)
            kwargs = {"cache_position": torch.tensor([absolute], device=device)} if has_cache_position else {}
            synchronize_device(device)
            forward_start = time.perf_counter()
            output = engine.model(input_ids=ids[:, absolute:absolute+1], past_key_values=cache,
                                  position_ids=torch.tensor([[absolute]], device=device),
                                  attention_mask=mask, use_cache=True, output_attentions=False, **kwargs)
            synchronize_device(device)
            query_forward_ms += (time.perf_counter()-forward_start)*1000
            cache = output.past_key_values
            positions.append(absolute)
            prune_start = time.perf_counter()
            if prune is not None:
                cache, indices = prune(cache)
                positions = [positions[i] for i in indices]
            synchronize_device(device)
            query_prune_ms += (time.perf_counter()-prune_start)*1000
            query_lengths.append(cache_length(cache))
            if len(positions) != cache_length(cache) or (budget is not None and len(positions)>budget):
                raise AssertionError("Question processing exceeded cache budget")
            logits = output.logits[:, -1, :].clone()  # replaces document logits before token 1
            del output
    synchronize_device(device)
    query_ms = (time.perf_counter()-query_start)*1000 if n>doc_n else 0.0
    tracker.record("post_query", cache)
    lengths = [cache_length(cache)]

    def pick(scores):
        if do_sample:
            return torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1)
        return scores.argmax(dim=-1, keepdim=True)

    eos = getattr(engine.model.generation_config, "eos_token_id", None)
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos.add(engine.tokenizer.eos_token_id)
    generated = []
    ttft_ms = None
    if max_new_tokens:
        # Held-query mode uses ONLY logits from the final post-compression query forward.
        next_token = pick(logits)
        generated.append(int(next_token.item()))
        synchronize_device(device)
        ttft_ms = (time.perf_counter() - wall_start) * 1000
    del logits
    decode_start = time.perf_counter()
    prune_ms = forward_ms = 0.0
    steps = 0
    with torch.inference_mode():
        while len(generated) < max_new_tokens and generated[-1] not in eos:
            absolute = n + len(generated) - 1
            # Compact cache has gaps in absolute positions. All retained entries
            # precede this single query; an explicit mask avoids inferred offsets.
            mask = torch.zeros((1, 1, 1, cache_length(cache) + 1), device=device, dtype=engine.dtype)
            kwargs = {"cache_position": torch.tensor([absolute], device=device)} if has_cache_position else {}
            synchronize_device(device)
            start = time.perf_counter()
            output = engine.model(input_ids=next_token, past_key_values=cache, use_cache=True,
                                  position_ids=torch.tensor([[absolute]], device=device),
                                  attention_mask=mask, output_attentions=False, **kwargs)
            synchronize_device(device)
            forward_ms += (time.perf_counter() - start) * 1000
            cache = output.past_key_values  # do not assume in-place cache identity
            positions.append(absolute)
            start = time.perf_counter()
            if prune is not None:
                cache, indices = prune(cache)
                positions = [positions[i] for i in indices]
            synchronize_device(device)
            prune_ms += (time.perf_counter() - start) * 1000
            lengths.append(cache_length(cache))
            if lengths[-1] != len(positions) or (budget is not None and lengths[-1] > budget):
                raise AssertionError("Cache positions/budget diverged")
            next_token = pick(output.logits[:, -1, :])
            generated.append(int(next_token.item()))
            steps += 1
            del output
    synchronize_device(device)
    decode_ms = (time.perf_counter() - decode_start) * 1000
    tracker.record("post_decoding", cache)
    actual, _, meta = calculate_kv_cache_bytes(cache)
    # Counterfactual full cache at the same consumed length, using observed bytes/token.
    uncompressed = prefill_bytes // doc_n * (n + steps)
    text = engine.tokenizer.decode(generated, skip_special_tokens=True)
    synchronize_device(device)
    total_ms = (time.perf_counter() - wall_start) * 1000
    return dict(output_text=text, input_tokens=n, generated_tokens=len(generated),
                total_tokens=n + len(generated), prefill_time_ms=prefill_ms,
                decode_time_ms=decode_ms, total_time_ms=total_ms, ttft_ms=ttft_ms,
                decode_tokens_per_sec=1000 * steps / decode_ms if steps else 0.0,
                actual_cached_tokens=cache_length(cache), compressed_kv_bytes=actual,
                uncompressed_kv_bytes=uncompressed,
                kv_cache_savings_pct=100 * (1 - actual / uncompressed),
                checkpoints=tracker.summary(),
                telemetry=dict(evaluation_protocol=protocol, document_tokens=doc_n, query_tokens=n-doc_n,
                               query_time_ms=query_ms, query_forward_time_ms=query_forward_ms,
                               query_pruning_time_ms=query_prune_ms, query_cache_lengths=query_lengths,
                               document_retained_positions=document_positions, scoring_time_ms=scoring_ms, prefill_forward_time_ms=prefill_forward_ms,
                               selection_time_ms=initial_selection_ms + query_prune_ms + prune_ms,
                               initial_selection_time_ms=initial_selection_ms,
                               decode_pruning_time_ms=prune_ms, decode_forward_time_ms=forward_ms,
                               decode_steps=steps, generated_token_ids=generated,
                               retained_positions=positions, cache_lengths=lengths,
                               prefill_kv_bytes=prefill_bytes, cache_storage_bytes=meta["storage_bytes"],
                               cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                               cache_position_mode="explicit absolute" if has_cache_position else "not in model API; explicit position_ids + compact mask"))
