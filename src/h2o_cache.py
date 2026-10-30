"""
H2O (Heavy-Hitter Oracle) Key-Value Cache Eviction Engine.
Implements attention-score accumulation across transformer layers to identify
and retain static prefill 'Heavy Hitter' (H) key-value tokens alongside a local
rolling window (W) of recent tokens.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import DynamicCache

from src.baseline import UncompressedBaselineEngine
from src.generation import GenerationTelemetry, run_generation, cache_length, prefix_recent_indices, retain
from src.metrics import (
    checkpoint_device_mib,
    MemoryTracker,
    calculate_kv_cache_bytes,
    theoretical_kv_cache_bytes,
    get_process_rss_bytes,
    get_peak_rss_bytes,
    get_device_allocated_bytes,
)


@dataclass
class H2OInferenceResult(GenerationTelemetry):
    prompt: str
    output_text: str
    input_tokens: int
    generated_tokens: int
    total_tokens: int
    heavy_hitters: int
    recent_tokens: int
    cache_budget_tokens: int
    actual_cached_tokens: int
    retained_heavy_hitter_indices: List[int]
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
            "policy": "H2O-style prefill oracle approximation (static HH + recent; not dynamic H2O)",
            "heavy_hitters": self.heavy_hitters,
            "recent_tokens": self.recent_tokens,
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
            "retained_heavy_hitter_indices": self.retained_heavy_hitter_indices,
            "checkpoints": self.checkpoints,
        }


class H2OCacheEngine(UncompressedBaselineEngine):
    """
    Inference Engine implementing H2O-style prefill oracle approximation; not dynamic H2O.
    Identifies high-attention 'Heavy Hitter' keys during prefill and retains them
    along with a local recent window of tokens.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        heavy_hitters: int = 64,
        recent_tokens: int = 196,
        revision: Optional[str] = None,
        attn_implementation: str = "eager",
    ):
        super().__init__(model_name_or_path=model_name_or_path, device=device, dtype=dtype,
                         revision=revision, attn_implementation=attn_implementation)
        self.heavy_hitters = heavy_hitters
        self.recent_tokens = recent_tokens
        self.cache_budget = heavy_hitters + recent_tokens
        if heavy_hitters < 0 or recent_tokens < 1:
            raise ValueError("Require nonnegative heavy hitters and positive recent window")

    def prune_cache_with_attentions(
        self, past_key_values: Any, attentions: Optional[Tuple[torch.Tensor, ...]]
    ) -> Tuple[Any, int, List[int]]:
        """
        Calculates cumulative attention scores across all transformer layers and heads,
        selects top-H Heavy Hitters outside the recent window W, and physically slices
        the KV cache tensors.
        Returns:
            (pruned_cache, pruned_seq_len, retained_heavy_hitter_indices)
        """
        if past_key_values is None or not hasattr(past_key_values, "layers"):
            return past_key_values, 0, []

        first_layer = past_key_values.layers[0]
        if first_layer.keys is None:
            return past_key_values, 0, []

        seq_len = first_layer.keys.shape[-2]
        if seq_len <= self.cache_budget:
            return past_key_values, seq_len, list(range(max(0, seq_len - self.recent_tokens)))

        H = self.heavy_hitters
        W = self.recent_tokens
        num_candidates = seq_len - W
        candidate_indices = torch.arange(num_candidates, device=self.device)

        if attentions is not None and len(attentions) > 0 and attentions[0] is not None:
            # Accumulate attention scores over all layers
            if isinstance(attentions, torch.Tensor):
                cumulative_scores = attentions
            else:
                cumulative_scores = torch.zeros(seq_len, device=self.device, dtype=torch.float32)
                for layer_attn in attentions:
                    if layer_attn is not None:
                        cumulative_scores += layer_attn[0].float().sum(dim=(0, 1))

            candidate_scores = cumulative_scores[:num_candidates]
            actual_h = min(H, num_candidates)
            top_h_rel = torch.argsort(candidate_scores, descending=True, stable=True)[:actual_h]
            top_h_indices = candidate_indices[top_h_rel]
            top_h_indices, _ = torch.sort(top_h_indices)
        else:
            # Fallback if attentions were not captured (e.g. initial sink tokens)
            actual_h = min(H, num_candidates)
            top_h_indices = candidate_indices[:actual_h]

        # Recent window tokens
        recent_indices = torch.arange(num_candidates, seq_len, device=self.device)

        # Retained indices: H heavy hitters + W recent tokens
        retained_indices = torch.cat([top_h_indices, recent_indices])
        retained_indices, _ = torch.sort(retained_indices)

        # Physical slicing across all layers
        for layer in past_key_values.layers:
            if layer.keys is not None and layer.values is not None:
                layer.keys = layer.keys[..., retained_indices, :].contiguous()
                layer.values = layer.values[..., retained_indices, :].contiguous()

        new_seq_len = past_key_values.layers[0].keys.shape[-2]
        return past_key_values, new_seq_len, top_h_indices.tolist()

    def generate(self, prompt: str, max_new_tokens: int = 35,
                 temperature: float = 0.0, do_sample: bool = False) -> H2OInferenceResult:
        selected = []

        def select(cache, attention, ids):
            nonlocal selected
            length = cache_length(cache)
            cache, _, selected = self.prune_cache_with_attentions(cache, attention)
            indices = (list(range(length)) if length <= self.cache_budget else
                       sorted(selected + list(range(length - self.recent_tokens, length))))
            return cache, indices

        def prune(cache):
            indices = prefix_recent_indices(cache_length(cache), self.heavy_hitters, self.recent_tokens)
            return retain(cache, indices), indices

        r = run_generation(self, prompt, max_new_tokens, temperature, do_sample,
                           select=select, prune=prune, capture_attentions=True)
        telemetry = r.pop("telemetry")
        return H2OInferenceResult(
            prompt=prompt, **r, **telemetry, heavy_hitters=self.heavy_hitters,
            recent_tokens=self.recent_tokens, cache_budget_tokens=self.cache_budget,
            retained_heavy_hitter_indices=selected,
            compressed_kv_mib=r["compressed_kv_bytes"] / 1024**2,
            peak_process_rss_mib=get_peak_rss_bytes() / 1024**2,
            peak_device_allocated_mib=checkpoint_device_mib(r["checkpoints"]),
            device=str(self.device), model_name=self.model_name)
