"""
StreamingLLM (Attention Sinks + Recent Rolling Window) Cache Adapter.
Implements real physical Key-Value tensor eviction, reducing cache memory
post-prefill and during autoregressive decoding.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.baseline import UncompressedBaselineEngine
from src.generation import GenerationTelemetry, run_generation, cache_length, prefix_recent_indices, retain
from src.metrics import (
    checkpoint_device_mib,
    MemoryTracker,
    MemoryCheckpoint,
    calculate_kv_cache_bytes,
    theoretical_kv_cache_bytes,
    get_process_rss_bytes,
    get_peak_rss_bytes,
    get_device_allocated_bytes,
)


@dataclass
class StreamingInferenceResult(GenerationTelemetry):
    prompt: str
    output_text: str
    input_tokens: int
    generated_tokens: int
    total_tokens: int
    sink_tokens: int
    window_tokens: int
    cache_budget_tokens: int
    actual_cached_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    ttft_ms: float
    decode_tokens_per_sec: float
    uncompressed_kv_bytes: int
    compressed_kv_bytes: int
    compressed_kv_mib: float
    kv_cache_savings_pct: float
    peak_process_rss_mib: float
    peak_device_allocated_mib: float
    device: str
    model_name: str
    checkpoints: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "policy": "StreamingLLM (Attention Sinks + Rolling Window)",
            "sink_tokens": self.sink_tokens,
            "window_tokens": self.window_tokens,
            "cache_budget_tokens": self.cache_budget_tokens,
            "input_tokens": self.input_tokens,
            "generated_tokens": self.generated_tokens,
            "actual_cached_tokens": self.actual_cached_tokens,
            "output_text": self.output_text,
            "telemetry": {name: getattr(self, name) for name in GenerationTelemetry.__dataclass_fields__},
            "timings": {
                "selection_time_ms": self.selection_time_ms,
                "initial_selection_time_ms": self.initial_selection_time_ms,
                "decode_pruning_time_ms": self.decode_pruning_time_ms,
                "decode_forward_time_ms": self.decode_forward_time_ms,
                "prefill_time_ms": round(self.prefill_time_ms, 2),
                "decode_time_ms": round(self.decode_time_ms, 2),
                "total_time_ms": round(self.total_time_ms, 2),
                "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
                "decode_tokens_per_sec": round(self.decode_tokens_per_sec, 2),
            },
            "memory": {
                "uncompressed_kv_bytes": self.uncompressed_kv_bytes,
                "compressed_kv_bytes": self.compressed_kv_bytes,
                "compressed_kv_mib": round(self.compressed_kv_mib, 4),
                "kv_cache_savings_pct": round(self.kv_cache_savings_pct, 2),
                "peak_process_rss_mib": round(self.peak_process_rss_mib, 2),
                "peak_device_allocated_mib": round(self.peak_device_allocated_mib, 2),
            },
            "checkpoints": self.checkpoints,
        }


class StreamingLLMEngine(UncompressedBaselineEngine):
    """
    Inference Engine implementing StreamingLLM Attention Sinks + Rolling Window eviction.
    Physically prunes Key and Value cache tensors to release physical storage.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        sink_tokens: int = 4,
        window_tokens: int = 256,
        revision: Optional[str] = None,
        attn_implementation: str = "sdpa",
    ):
        super().__init__(model_name_or_path=model_name_or_path, device=device, dtype=dtype,
                         revision=revision, attn_implementation=attn_implementation)
        self.sink_tokens = sink_tokens
        self.window_tokens = window_tokens
        self.cache_budget = sink_tokens + window_tokens
        if sink_tokens < 0 or window_tokens < 1:
            raise ValueError("Require nonnegative sinks and positive recent window")

    def prune_cache(self, past_key_values: Any) -> Tuple[Any, int]:
        """
        Physically slices Key and Value tensors along the sequence dimension.
        Retains [:sink_tokens] and [-window_tokens:], evicting intermediate tokens.
        Returns:
            (pruned_past_key_values, pruned_seq_len)
        """
        if past_key_values is None or not hasattr(past_key_values, "layers"):
            return past_key_values, 0

        first_layer = past_key_values.layers[0]
        if first_layer.keys is None:
            return past_key_values, 0

        current_seq_len = first_layer.keys.shape[-2]
        if current_seq_len <= self.cache_budget:
            return past_key_values, current_seq_len

        S = self.sink_tokens
        W = self.window_tokens

        # Slicing across all transformer layers
        for layer in past_key_values.layers:
            if layer.keys is not None and layer.values is not None:
                # Slicing creates new, contiguous physical tensors of size S + W
                layer.keys = torch.cat(
                    [layer.keys[..., :S, :], layer.keys[..., -W:, :]],
                    dim=-2,
                )
                layer.values = torch.cat(
                    [layer.values[..., :S, :], layer.values[..., -W:, :]],
                    dim=-2,
                )

        new_seq_len = past_key_values.layers[0].keys.shape[-2]
        return past_key_values, new_seq_len

    def generate(self, prompt: str, max_new_tokens: int = 35,
                 temperature: float = 0.0, do_sample: bool = False) -> StreamingInferenceResult:
        def prune(cache):
            indices = prefix_recent_indices(cache_length(cache), self.sink_tokens, self.window_tokens)
            return retain(cache, indices), indices

        r = run_generation(self, prompt, max_new_tokens, temperature, do_sample,
                           select=lambda cache, attention, ids: prune(cache), prune=prune)
        telemetry = r.pop("telemetry")
        return StreamingInferenceResult(
            prompt=prompt, **r, **telemetry, sink_tokens=self.sink_tokens,
            window_tokens=self.window_tokens, cache_budget_tokens=self.cache_budget,
            compressed_kv_mib=r["compressed_kv_bytes"] / 1024**2,
            peak_process_rss_mib=get_peak_rss_bytes() / 1024**2,
            peak_device_allocated_mib=checkpoint_device_mib(r["checkpoints"]),
            device=str(self.device), model_name=self.model_name)
