"""
H2O (Heavy-Hitter Oracle) Key-Value Cache Eviction Engine.
Implements attention-score accumulation across transformer layers to identify
and retain dynamic 'Heavy Hitter' (H) key-value tokens alongside a local
rolling window (W) of recent tokens.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import DynamicCache

from src.baseline import UncompressedBaselineEngine
from src.metrics import (
    MemoryTracker,
    calculate_kv_cache_bytes,
    theoretical_kv_cache_bytes,
    get_process_rss_bytes,
    get_peak_rss_bytes,
    get_device_allocated_bytes,
)


@dataclass
class H2OInferenceResult:
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
            "policy": "H2O (Heavy-Hitter Oracle + Recent Window)",
            "heavy_hitters": self.heavy_hitters,
            "recent_tokens": self.recent_tokens,
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
            "retained_heavy_hitter_indices": self.retained_heavy_hitter_indices,
            "checkpoints": self.checkpoints,
        }


class H2OCacheEngine(UncompressedBaselineEngine):
    """
    Inference Engine implementing H2O (Heavy-Hitter Oracle) Cache Eviction.
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
    ):
        super().__init__(model_name_or_path=model_name_or_path, device=device, dtype=dtype)
        self.heavy_hitters = heavy_hitters
        self.recent_tokens = recent_tokens
        self.cache_budget = heavy_hitters + recent_tokens

    def load_model(self) -> Dict[str, Any]:
        """Loads tokenizer and model with eager attention implementation for attention extraction."""
        print(f"[H2OCacheEngine] Initializing pre-load memory tracking on device: {self.device}")
        self.memory_tracker.record("pre_load")
        load_start = time.time()

        print(f"[H2OCacheEngine] Loading tokenizer: {self.model_name}")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        print(f"[H2OCacheEngine] Loading model weights with eager attention (dtype={self.dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=None,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.use_cache = True

        cfg = self.model.config
        self.num_layers = getattr(cfg, "num_hidden_layers", 28)
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 16))
        hidden_size = getattr(cfg, "hidden_size", 2048)
        num_attn_heads = getattr(cfg, "num_attention_heads", 16)
        self.head_dim = getattr(cfg, "head_dim", hidden_size // num_attn_heads)
        self.max_context_length = getattr(cfg, "max_position_embeddings", 32768)

        self.model_param_count = sum(p.numel() for p in self.model.parameters())
        self.static_weights_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.memory_tracker.record("model_loaded")

        load_duration_sec = time.time() - load_start
        print(
            f"[H2OCacheEngine] Model loaded in {load_duration_sec:.2f}s | "
            f"Params: {self.model_param_count:,} | "
            f"Weights: {self.static_weights_bytes / (1024 ** 2):.1f} MiB"
        )
        return {
            "load_duration_sec": round(load_duration_sec, 2),
            "param_count": self.model_param_count,
            "weights_mib": round(self.static_weights_bytes / (1024 ** 2), 2),
        }

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
            cumulative_scores = torch.zeros(seq_len, device=self.device, dtype=torch.float32)
            for layer_attn in attentions:
                if layer_attn is not None:
                    key_scores = layer_attn.squeeze(0).mean(dim=0).sum(dim=0).to(torch.float32)
                    cumulative_scores += key_scores

            candidate_scores = cumulative_scores[:num_candidates]
            actual_h = min(H, num_candidates)
            top_h_rel = torch.topk(candidate_scores, actual_h).indices
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

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 35,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> H2OInferenceResult:
        """
        Executes H2O compressed autoregressive generation.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_tokens = input_ids.shape[1]

        if input_tokens > self.max_context_length:
            raise ValueError(
                f"Prompt length ({input_tokens}) exceeds maximum context ({self.max_context_length})."
            )

        tracker = MemoryTracker(self.device)
        tracker.record("inference_start")

        uncompressed_theo_bytes = theoretical_kv_cache_bytes(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            element_size=self.element_size,
            seq_len=input_tokens + max_new_tokens,
            batch_size=1,
        )

        # ---------------------------------------------------------
        # 1. Prompt Prefill Phase with Attention Extraction
        # ---------------------------------------------------------
        if self.device.type == "mps":
            torch.mps.synchronize()
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            prefill_out = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
                output_attentions=True,
            )

        if self.device.type == "mps":
            torch.mps.synchronize()
        t_prefill_end = time.perf_counter()
        prefill_time_ms = (t_prefill_end - t_prefill_start) * 1000.0

        past_key_values = prefill_out.past_key_values
        attentions = prefill_out.attentions
        next_token_logits = prefill_out.logits[:, -1, :].clone()
        tracker.record("prompt_prefill_raw")

        # ---------------------------------------------------------
        # 2. H2O Cache Pruning (Heavy-Hitters + Recent Window)
        # ---------------------------------------------------------
        t_slice_start = time.perf_counter()
        past_key_values, pruned_seq_len, retained_heavy_hitters = self.prune_cache_with_attentions(
            past_key_values, attentions
        )
        # Release full attention tensors from memory
        del attentions
        del prefill_out
        if self.device.type == "mps":
            torch.mps.empty_cache()

        t_slice_end = time.perf_counter()
        tracker.record("cache_pruned")

        # ---------------------------------------------------------
        # 3. Autoregressive Decoding Phase
        # ---------------------------------------------------------
        t_decode_start = time.perf_counter()
        if do_sample and temperature > 0.0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        ttft_ms = prefill_time_ms + (t_slice_end - t_slice_start) * 1000.0
        generated_ids: List[int] = [int(next_token[0, 0].item())]
        current_abs_pos = input_tokens

        H = self.heavy_hitters
        W = self.recent_tokens

        for step in range(max_new_tokens - 1):
            if generated_ids[-1] in (self.tokenizer.eos_token_id, 151645, 151643):
                break

            curr_pos = torch.tensor([[current_abs_pos]], dtype=torch.long, device=self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=curr_pos,
                )

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]

            if do_sample and temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)
            current_abs_pos += 1

            # Bound cache size during decoding: retain H heavy hitters + W recent tokens
            curr_cached_len = past_key_values.layers[0].keys.shape[-2]
            if curr_cached_len > self.cache_budget:
                for layer in past_key_values.layers:
                    if layer.keys is not None:
                        layer.keys = torch.cat(
                            [layer.keys[..., :H, :], layer.keys[..., -W:, :]], dim=-2
                        ).contiguous()
                        layer.values = torch.cat(
                            [layer.values[..., :H, :], layer.values[..., -W:, :]], dim=-2
                        ).contiguous()

        if self.device.type == "mps":
            torch.mps.synchronize()
        t_decode_end = time.perf_counter()
        decode_time_ms = (t_decode_end - t_decode_start) * 1000.0
        total_time_ms = prefill_time_ms + decode_time_ms
        tracker.record("decoding_end")

        generated_tokens = len(generated_ids)
        decode_tokens_per_sec = (generated_tokens / (decode_time_ms / 1000.0)) if decode_time_ms > 0 else 0.0
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        final_cached_tokens = past_key_values.layers[0].keys.shape[-2]
        compressed_kv_bytes, _, _ = calculate_kv_cache_bytes(past_key_values)
        savings_pct = (
            (100.0 * (uncompressed_theo_bytes - compressed_kv_bytes) / uncompressed_theo_bytes)
            if uncompressed_theo_bytes > 0
            else 0.0
        )

        return H2OInferenceResult(
            prompt=prompt,
            output_text=output_text,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            total_tokens=input_tokens + generated_tokens,
            heavy_hitters=self.heavy_hitters,
            recent_tokens=self.recent_tokens,
            cache_budget_tokens=self.cache_budget,
            actual_cached_tokens=final_cached_tokens,
            retained_heavy_hitter_indices=retained_heavy_hitters,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
            total_time_ms=total_time_ms,
            ttft_ms=ttft_ms,
            decode_tokens_per_sec=decode_tokens_per_sec,
            uncompressed_kv_bytes=uncompressed_theo_bytes,
            compressed_kv_bytes=compressed_kv_bytes,
            compressed_kv_mib=compressed_kv_bytes / (1024 ** 2),
            kv_cache_savings_pct=savings_pct,
            peak_process_rss_mib=get_peak_rss_bytes() / (1024 ** 2),
            peak_device_allocated_mib=get_device_allocated_bytes(self.device) / (1024 ** 2),
            device=str(self.device),
            model_name=self.model_name,
            checkpoints=tracker.summary(),
        )
