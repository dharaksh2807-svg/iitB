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

from src.metrics import (
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
class BaselineInferenceResult:
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
            "timings": {
                "prefill_time_ms": round(self.prefill_time_ms, 2),
                "decode_time_ms": round(self.decode_time_ms, 2),
                "total_time_ms": round(self.total_time_ms, 2),
                "ttft_ms": round(self.ttft_ms, 2),
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
    ):
        self.model_name = model_name_or_path
        self.dtype = dtype

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        print(f"[BaselineEngine] Loading model weights (dtype={self.dtype})")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self.dtype,
                device_map=None,
            ).to(self.device)
        except Exception as e:
            if self.device.type == "mps":
                print(f"[BaselineEngine] MPS load failed ({e}); falling back to CPU.")
                self.device = torch.device("cpu")
                self.memory_tracker.device = self.device
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=self.dtype,
                    device_map=None,
                ).to(self.device)
            else:
                raise e

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
        """
        Executes reference uncompressed autoregressive generation.
        Separates prefill phase and token decoding loop with individual timing and memory telemetry.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_tokens = input_ids.shape[1]

        if input_tokens > self.max_context_length:
            raise ValueError(
                f"Prompt length ({input_tokens} tokens) exceeds model maximum context ({self.max_context_length} tokens)."
            )

        # Fresh tracker for this inference invocation
        tracker = MemoryTracker(self.device)
        tracker.record("inference_start")

        # ----------------------------------------------------
        # Phase 1: Prefill (Prompt Encoding & Initial KV Cache)
        # ----------------------------------------------------
        self.sync_device()
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)

        self.sync_device()
        t_prefill_end = time.perf_counter()
        prefill_time_ms = (t_prefill_end - t_prefill_start) * 1000.0

        past_key_values = outputs.past_key_values
        tracker.record("post_prefill", past_key_values)

        # First token selection
        next_token_logits = outputs.logits[:, -1, :]
        if do_sample and temperature > 0.0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        ttft_ms = prefill_time_ms
        generated_token_ids: List[int] = [int(next_token[0, 0].item())]

        # ----------------------------------------------------
        # Phase 2: Autoregressive Decoding Loop
        # ----------------------------------------------------
        t_decode_start = time.perf_counter()
        eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for step in range(max_new_tokens - 1):
                if generated_token_ids[-1] == eos_token_id:
                    break

                out_step = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out_step.past_key_values

                step_logits = out_step.logits[:, -1, :]
                if do_sample and temperature > 0.0:
                    probs = torch.softmax(step_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

                generated_token_ids.append(int(next_token[0, 0].item()))

        self.sync_device()
        t_decode_end = time.perf_counter()
        decode_time_ms = (t_decode_end - t_decode_start) * 1000.0
        total_time_ms = prefill_time_ms + decode_time_ms

        generated_tokens = len(generated_token_ids)
        total_tokens = input_tokens + generated_tokens
        decode_tokens_per_sec = (generated_tokens / (decode_time_ms / 1000.0)) if decode_time_ms > 0 else 0.0

        tracker.record("post_decoding", past_key_values)

        # ----------------------------------------------------
        # Phase 3: KV Cache Tensor Introspection & Verification
        # ----------------------------------------------------
        # Determine exact sequence length currently retained in cache tensors
        if hasattr(past_key_values, "get_seq_length"):
            cache_seq_len = past_key_values.get_seq_length()
        elif hasattr(past_key_values, "layers") and past_key_values.layers[0].keys is not None:
            cache_seq_len = past_key_values.layers[0].keys.shape[-2]
        else:
            cache_seq_len = input_tokens + max(0, generated_tokens - 1)

        actual_kv_bytes, layers_measured, meta = calculate_kv_cache_bytes(past_key_values)
        theo_kv_bytes = theoretical_kv_cache_bytes(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            element_size=self.element_size,
            seq_len=cache_seq_len,
            batch_size=1,
        )

        scaling_match = (
            (1.0 - abs(actual_kv_bytes - theo_kv_bytes) / theo_kv_bytes) * 100.0
            if theo_kv_bytes > 0
            else 100.0
        )

        output_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return BaselineInferenceResult(
            prompt=prompt,
            output_text=output_text,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
            total_time_ms=total_time_ms,
            ttft_ms=ttft_ms,
            decode_tokens_per_sec=decode_tokens_per_sec,
            kv_cache_bytes=actual_kv_bytes,
            kv_cache_mib=actual_kv_bytes / (1024**2),
            theoretical_kv_bytes=theo_kv_bytes,
            theoretical_kv_mib=theo_kv_bytes / (1024**2),
            kv_scaling_match_pct=scaling_match,
            peak_process_rss_mib=get_peak_rss_bytes() / (1024**2),
            peak_device_allocated_mib=get_device_allocated_bytes(self.device) / (1024**2),
            device=str(self.device),
            model_name=self.model_name,
            checkpoints=tracker.summary(),
        )

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
