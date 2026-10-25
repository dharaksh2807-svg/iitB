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
from src.metrics import (
    MemoryTracker,
    MemoryCheckpoint,
    calculate_kv_cache_bytes,
    theoretical_kv_cache_bytes,
    get_process_rss_bytes,
    get_peak_rss_bytes,
    get_device_allocated_bytes,
)


@dataclass
class StreamingInferenceResult:
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
            "timings": {
                "prefill_time_ms": round(self.prefill_time_ms, 2),
                "decode_time_ms": round(self.decode_time_ms, 2),
                "total_time_ms": round(self.total_time_ms, 2),
                "ttft_ms": round(self.ttft_ms, 2),
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
    ):
        super().__init__(model_name_or_path=model_name_or_path, device=device, dtype=dtype)
        self.sink_tokens = sink_tokens
        self.window_tokens = window_tokens
        self.cache_budget = sink_tokens + window_tokens

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

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> StreamingInferenceResult:
        """
        Executes StreamingLLM compressed autoregressive generation.
        Separates prefill phase, performs post-prefill KV tensor eviction,
        and bounds cache size during decoding via rolling eviction.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_tokens = input_ids.shape[1]

        if input_tokens > self.max_context_length:
            raise ValueError(
                f"Prompt length ({input_tokens} tokens) exceeds model maximum context ({self.max_context_length})."
            )

        tracker = MemoryTracker(self.device)
        tracker.record("inference_start")

        # Theoretical uncompressed size if no eviction took place
        uncompressed_theo_bytes = theoretical_kv_cache_bytes(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            element_size=self.element_size,
            seq_len=input_tokens + max_new_tokens,
            batch_size=1,
        )

        # ----------------------------------------------------
        # Phase 1: Prefill (Prompt Encoding)
        # ----------------------------------------------------
        self.sync_device()
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)

        self.sync_device()
        t_prefill_end = time.perf_counter()
        prefill_time_ms = (t_prefill_end - t_prefill_start) * 1000.0

        past_key_values = outputs.past_key_values
        tracker.record("peak_prefill_raw", past_key_values)

        # ----------------------------------------------------
        # Phase 2: Post-Prefill Real Cache Pruning
        # ----------------------------------------------------
        past_key_values, cached_seq_len = self.prune_cache(past_key_values)
        tracker.record("post_prefill_pruned", past_key_values)

        next_token_logits = outputs.logits[:, -1, :]
        if do_sample and temperature > 0.0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        ttft_ms = prefill_time_ms
        generated_token_ids: List[int] = [int(next_token[0, 0].item())]
        eos_token_id = self.tokenizer.eos_token_id

        # ----------------------------------------------------
        # Phase 3: Autoregressive Decoding Loop with Rolling Eviction
        # ----------------------------------------------------
        t_decode_start = time.perf_counter()
        current_abs_pos = input_tokens

        with torch.no_grad():
            for step in range(max_new_tokens - 1):
                if generated_token_ids[-1] == eos_token_id:
                    break

                # Explicitly pass true absolute position to maintain RoPE consistency
                pos_ids = torch.tensor([[current_abs_pos]], device=self.device)

                out_step = self.model(
                    input_ids=next_token,
                    position_ids=pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out_step.past_key_values

                # Rolling eviction: ensure cache size does not exceed S + W
                past_key_values, cached_seq_len = self.prune_cache(past_key_values)

                step_logits = out_step.logits[:, -1, :]
                if do_sample and temperature > 0.0:
                    probs = torch.softmax(step_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

                generated_token_ids.append(int(next_token[0, 0].item()))
                current_abs_pos += 1

        self.sync_device()
        t_decode_end = time.perf_counter()
        decode_time_ms = (t_decode_end - t_decode_start) * 1000.0
        total_time_ms = prefill_time_ms + decode_time_ms

        generated_tokens = len(generated_token_ids)
        total_tokens = input_tokens + generated_tokens
        decode_tokens_per_sec = (generated_tokens / (decode_time_ms / 1000.0)) if decode_time_ms > 0 else 0.0

        tracker.record("post_decoding", past_key_values)

        # ----------------------------------------------------
        # Phase 4: Exact Compressed Cache Measurement
        # ----------------------------------------------------
        actual_kv_bytes, _, _ = calculate_kv_cache_bytes(past_key_values)

        # Calculate exact baseline uncompressed bytes for this exact sequence length
        uncompressed_bytes = theoretical_kv_cache_bytes(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            element_size=self.element_size,
            seq_len=input_tokens + max(0, generated_tokens - 1),
            batch_size=1,
        )

        savings_pct = (
            (100.0 * (uncompressed_bytes - actual_kv_bytes) / uncompressed_bytes)
            if uncompressed_bytes > 0
            else 0.0
        )

        output_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return StreamingInferenceResult(
            prompt=prompt,
            output_text=output_text,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            sink_tokens=self.sink_tokens,
            window_tokens=self.window_tokens,
            cache_budget_tokens=self.cache_budget,
            actual_cached_tokens=cached_seq_len,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
            total_time_ms=total_time_ms,
            ttft_ms=ttft_ms,
            decode_tokens_per_sec=decode_tokens_per_sec,
            uncompressed_kv_bytes=uncompressed_bytes,
            compressed_kv_bytes=actual_kv_bytes,
            compressed_kv_mib=actual_kv_bytes / (1024**2),
            kv_cache_savings_pct=savings_pct,
            peak_process_rss_mib=get_peak_rss_bytes() / (1024**2),
            peak_device_allocated_mib=get_device_allocated_bytes(self.device) / (1024**2),
            device=str(self.device),
            model_name=self.model_name,
            checkpoints=tracker.summary(),
        )
