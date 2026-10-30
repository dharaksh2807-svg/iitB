"""Observation-guided chunk retention: explicitly post-prefill compression.

Generation accumulates observation attention in query blocks, bounding live
attention outputs to O(layers * heads * block_size * L). Full prompt KV remains
live until post-prefill compression. No peak-memory reduction claim is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from src.baseline import UncompressedBaselineEngine
from src.generation import GenerationTelemetry, cache_length, prefix_recent_indices, retain, run_generation
from src.selection import observation_scores, select_history

@dataclass
class OGCInferenceResult(GenerationTelemetry):
    """Encapsulates the metrics and output of an OGC-Cache inference run."""
    prompt: str
    generated_text: str
    output_tokens_generated: int
    input_tokens: int
    context_tokens: int
    sink_tokens: int
    recent_window: int
    chunk_size: int
    num_chunks: int
    retained_tokens: int
    selected_chunk_ranges: List[Tuple[int, int]]
    ttft_ms: Optional[float]
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    decode_tokens_per_sec: float
    uncompressed_kv_bytes: int
    compressed_kv_bytes: int
    kv_cache_savings_pct: float
    memory_metrics: Dict[str, Any]


class OGCCacheEngine(UncompressedBaselineEngine):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 sink_tokens: int = 4, recent_window: int = 192,
                 chunk_size: int = 64, num_chunks: int = 1, query_window: int = 48,
                 device: Optional[str] = None, dtype: torch.dtype = torch.bfloat16,
                 revision: Optional[str] = None, selection_policy: str = "ogc", seed: int = 42):
        super().__init__(model_name, device, dtype, revision=revision, attn_implementation="eager")
        if sink_tokens < 0 or min(recent_window, chunk_size, query_window) < 1 or num_chunks < 0:
            raise ValueError("Invalid sink/window/chunk/query configuration")
        if selection_policy not in {"ogc", "query_tokens", "random_chunks", "adaptive_chunks", "entity_chunks"}:
            raise ValueError("Unknown selection policy")
        self.sink_tokens, self.recent_window = sink_tokens, recent_window
        self.chunk_size, self.num_chunks = chunk_size, num_chunks
        self.query_window, self.selection_policy, self.seed = query_window, selection_policy, seed
        self.budget_tokens = sink_tokens + recent_window + chunk_size * num_chunks
        self._last_indices = []
        self._boundaries = []

    def _select_and_prune_cache(self, past_key_values: Any, attentions, seq_len: int):
        if seq_len <= self.budget_tokens:
            self._last_indices = list(range(seq_len))
            return past_key_values, seq_len, []
        capacity = self.chunk_size * self.num_chunks
        if self.selection_policy == "random_chunks" or capacity == 0:
            scores = torch.zeros(seq_len)
        else:
            scores = observation_scores(attentions, seq_len, self.query_window, self.device)
        historical, ranges = select_history(scores, self.sink_tokens, seq_len - self.recent_window,
                                             capacity, self.chunk_size, self.selection_policy,
                                             self.seed, self._boundaries)
        indices = list(range(self.sink_tokens)) + historical + list(range(seq_len - self.recent_window, seq_len))
        self._last_indices = indices
        return retain(past_key_values, indices), len(indices), ranges

    def generate(self, prompt: str, max_new_tokens: int = 35,
                 temperature: float = 0.0, do_sample: bool = False) -> OGCInferenceResult:
        selected_chunks = []
        pending_scores = None
        history_selected = False

        def select(cache, attention, ids):
            nonlocal selected_chunks, pending_scores, history_selected
            self._boundaries = []
            if self.selection_policy == "entity_chunks":
                # Token-only punctuation boundaries; never benchmark answer offsets.
                self._boundaries = [i + 1 for i, token in enumerate(ids[0].tolist())
                                    if any(c in self.tokenizer.decode([token]) for c in (".", "!", "?", "\n"))]
            length = cache_length(cache)
            history_selected = length > self.budget_tokens
            pending_scores = None if history_selected else attention
            cache, _, selected_chunks = self._select_and_prune_cache(cache, attention, length)
            return cache, self._last_indices

        def prune(cache):
            nonlocal pending_scores, history_selected, selected_chunks
            length = cache_length(cache)
            if not history_selected and length > self.budget_tokens:
                scores = pending_scores
                if scores is not None:
                    scores = torch.nn.functional.pad(scores, (0, length - scores.numel()))
                cache, _, selected_chunks = self._select_and_prune_cache(cache, scores, length)
                pending_scores = None
                history_selected = True
                return cache, self._last_indices
            # Freeze the selected historical prefix; evict oldest local tokens.
            prefix = self.sink_tokens + self.chunk_size * self.num_chunks
            indices = prefix_recent_indices(cache_length(cache), prefix, self.recent_window)
            return retain(cache, indices), indices

        r = run_generation(self, prompt, max_new_tokens, temperature, do_sample,
                           select=select, prune=prune,
                           capture_attentions=self.selection_policy != "random_chunks")
        telemetry = r.pop("telemetry")
        return OGCInferenceResult(
            prompt=prompt, generated_text=r["output_text"], output_tokens_generated=r["generated_tokens"],
            input_tokens=r["input_tokens"], context_tokens=r["total_tokens"],
            sink_tokens=self.sink_tokens, recent_window=self.recent_window,
            chunk_size=self.chunk_size, num_chunks=self.num_chunks,
            retained_tokens=r["actual_cached_tokens"], selected_chunk_ranges=selected_chunks,
            prefill_time_ms=r["prefill_time_ms"], decode_time_ms=r["decode_time_ms"],
            total_time_ms=r["total_time_ms"], ttft_ms=r["ttft_ms"],
            decode_tokens_per_sec=r["decode_tokens_per_sec"],
            uncompressed_kv_bytes=r["uncompressed_kv_bytes"], compressed_kv_bytes=r["compressed_kv_bytes"],
            kv_cache_savings_pct=r["kv_cache_savings_pct"], memory_metrics=r["checkpoints"], **telemetry)
