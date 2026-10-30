"""
Uncompressed Baseline Inference Engine for Long-Context LLMs.
Implements reference uncompressed KV-cache inference with separate prefill/decode timing,
exact tensor introspection, and multi-tier memory instrumentation.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from src.generation import (GenerationTelemetry, DEFAULT_MODEL, DEFAULT_REVISION, run_generation)

from src.metrics import (
    checkpoint_device_mib,
    MemoryTracker,
    MemoryCheckpoint,
    calculate_kv_cache_bytes,
    theoretical_kv_cache_bytes,
    get_process_rss_bytes,
    get_peak_rss_bytes,
    get_device_allocated_bytes,
    get_device_driver_bytes,
)


@dataclass
class BaselineInferenceResult(GenerationTelemetry):
    prompt: str
    output_text: str
    input_tokens: int
    generated_tokens: int
    total_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    ttft_ms: float
    decode_tokens_per_sec: float
    kv_cache_bytes: int
    kv_cache_mib: float
    theoretical_kv_bytes: int
    theoretical_kv_mib: float
    kv_scaling_match_pct: float
    peak_process_rss_mib: float
    peak_device_allocated_mib: float
    device: str
    model_name: str
    checkpoints: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "input_tokens": self.input_tokens,
            "generated_tokens": self.generated_tokens,
            "total_tokens": self.total_tokens,
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
                "kv_cache_bytes": self.kv_cache_bytes,
                "kv_cache_mib": round(self.kv_cache_mib, 4),
                "theoretical_kv_bytes": self.theoretical_kv_bytes,
                "theoretical_kv_mib": round(self.theoretical_kv_mib, 4),
                "kv_scaling_match_pct": round(self.kv_scaling_match_pct, 4),
                "peak_process_rss_mib": round(self.peak_process_rss_mib, 2),
                "peak_device_allocated_mib": round(self.peak_device_allocated_mib, 2),
            },
            "checkpoints": self.checkpoints,
        }


class UncompressedBaselineEngine:
    """
    Reference inference pipeline operating with standard, uncompressed DynamicCache.
    Serves as the ground-truth benchmark reference for all compression methods.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        revision: Optional[str] = None,
        attn_implementation: str = "sdpa",
    ):
        self.model_name = model_name_or_path
        self.dtype = dtype
        self.revision = revision or (DEFAULT_REVISION if self.model_name == DEFAULT_MODEL else None)
        self.attn_implementation = attn_implementation
        self.prefill_chunk_size = 256

        # Device selection: prefer MPS on Apple Silicon, fallback to CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.memory_tracker = MemoryTracker(self.device)

        # Architecture properties
        self.num_layers: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0
        self.max_context_length: int = 0
        self.model_param_count: int = 0
        self.static_weights_bytes: int = 0
        self.element_size: int = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    def load_model(self) -> Dict[str, Any]:
        """Loads tokenizer and model weights, recording load duration and baseline footprint."""
        print(f"[BaselineEngine] Initializing pre-load memory tracking on device: {self.device}")
        self.memory_tracker.record("pre_load")
        load_start = time.time()

        print(f"[BaselineEngine] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, revision=self.revision)

        print(f"[BaselineEngine] Loading model weights (dtype={self.dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.revision, attn_implementation=self.attn_implementation,
            dtype=self.dtype, device_map=None,
        ).to(self.device)

        self.model.eval()
        self.sync_device()
        load_duration = time.time() - load_start

        # Extract architecture specs
        config = self.model.config
        self.num_layers = getattr(config, "num_hidden_layers", 28)
        self.num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 12))
        hidden_size = getattr(config, "hidden_size", 1536)
        num_query_heads = getattr(config, "num_attention_heads", 12)
        self.head_dim = getattr(config, "head_dim", hidden_size // num_query_heads)
        self.max_context_length = getattr(config, "max_position_embeddings", 32768)

        self.model_param_count = sum(p.numel() for p in self.model.parameters())
        self.static_weights_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        self.memory_tracker.record("post_load")

        meta = {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "max_context_length": self.max_context_length,
            "param_count": self.model_param_count,
            "static_weights_mib": round(self.static_weights_bytes / (1024**2), 2),
            "load_duration_sec": round(load_duration, 2),
        }
        print(f"[BaselineEngine] Model loaded in {load_duration:.2f}s | Params: {self.model_param_count:,} | Weights: {meta['static_weights_mib']} MiB")
        return meta

    def sync_device(self) -> None:
        """Synchronize hardware device stream for precise latency measurement."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> BaselineInferenceResult:
        r = run_generation(self, prompt, max_new_tokens, temperature, do_sample)
        actual = r.pop("compressed_kv_bytes")
        theo = r.pop("uncompressed_kv_bytes")
        r.pop("kv_cache_savings_pct")
        r.pop("actual_cached_tokens")
        telemetry = r.pop("telemetry")
        return BaselineInferenceResult(
            prompt=prompt, **r, **telemetry, kv_cache_bytes=actual,
            kv_cache_mib=actual / 1024**2, theoretical_kv_bytes=theo,
            theoretical_kv_mib=theo / 1024**2,
            kv_scaling_match_pct=100 * (1 - abs(actual - theo) / theo),
            peak_process_rss_mib=get_peak_rss_bytes() / 1024**2,
            peak_device_allocated_mib=checkpoint_device_mib(r["checkpoints"]),
            device=str(self.device), model_name=self.model_name)

    def evaluate_scaling(
        self,
        context_lengths: List[int],
        filler_token: str = "The quick brown fox jumps over the lazy dog. ",
    ) -> List[Dict[str, Any]]:
        """
        Evaluates how uncompressed KV-cache size and prefill latency scale
        across increasing sequence lengths.
        """
        print(f"\n[BaselineEngine] Commencing context length scaling evaluation: {context_lengths}")
        results: List[Dict[str, Any]] = []

        base_tokens = self.tokenizer.encode(filler_token, add_special_tokens=False)
        tokens_per_repeat = len(base_tokens)

        for target_len in context_lengths:
            num_repeats = max(1, target_len // tokens_per_repeat)
            raw_text = filler_token * num_repeats
            tokens = self.tokenizer.encode(raw_text, add_special_tokens=True)[:target_len]
            actual_len = len(tokens)
            prompt = self.tokenizer.decode(tokens)

            res = self.generate(prompt=prompt, max_new_tokens=5, do_sample=False)
            results.append({
                "target_len": target_len,
                "actual_input_tokens": res.input_tokens,
                "total_tokens": res.total_tokens,
                "kv_cache_mib": res.kv_cache_mib,
                "theoretical_kv_mib": res.theoretical_kv_mib,
                "prefill_time_ms": res.prefill_time_ms,
                "decode_tokens_per_sec": res.decode_tokens_per_sec,
                "peak_process_rss_mib": res.peak_process_rss_mib,
                "peak_device_allocated_mib": res.peak_device_allocated_mib,
            })
            print(
                f"  -> Context: {actual_len} tokens | Actual KV Cache: {res.kv_cache_mib:.2f} MiB | "
                f"Theo: {res.theoretical_kv_mib:.2f} MiB | Prefill: {res.prefill_time_ms:.1f} ms"
            )

        return results
