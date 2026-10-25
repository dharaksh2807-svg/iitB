"""
Observation-Guided Chunked Cache (OGC-Cache) Engine for Long-Context LLM Inference.

Research Modification for Phase 8:
Addresses the mid-context blind spot in StreamingLLM and the point-wise token
fracturing in H2O by extracting query-directed observation attention and
retaining contiguous historical chunks alongside attention sinks and recent window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import MemoryTracker, theoretical_kv_cache_bytes


@dataclass
class OGCInferenceResult:
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
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    decode_tokens_per_sec: float
    uncompressed_kv_bytes: int
    compressed_kv_bytes: int
    kv_cache_savings_pct: float
    memory_metrics: Dict[str, float]


class OGCCacheEngine:
    """
    Observation-Guided Chunked (OGC) Cache Engine.
    
    Partitions the KV cache into three functional zones:
      1. S Initial Attention Sinks (stabilizes softmax & RoPE)
      2. K Contiguous Historical Chunks of size C (query-directed semantic retention)
      3. W Recent Window Tokens (immediate local context & generation prompt)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        sink_tokens: int = 4,
        recent_window: int = 192,
        chunk_size: int = 64,
        num_chunks: int = 1,
        query_window: int = 48,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.sink_tokens = sink_tokens
        self.recent_window = recent_window
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.query_window = query_window
        self.budget_tokens = sink_tokens + (chunk_size * num_chunks) + recent_window
        self.dtype = dtype

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.element_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4

        self.num_layers = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self.max_context_length = 0
        self.model_param_count = 0
        self.static_weights_bytes = 0

        self.memory_tracker = MemoryTracker(self.device)
        self.memory_tracker.record("pre_load")

    def load_model(self) -> Dict[str, Any]:
        """Loads tokenizer and weights with eager attention for attention extraction."""
        load_start = time.time()
        print(f"[OGCCacheEngine] Initializing pre-load memory tracking on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"[OGCCacheEngine] Loading model weights with eager attention (dtype={self.dtype})"
        )
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
            f"[OGCCacheEngine] Model loaded in {load_duration_sec:.2f}s | "
            f"Params: {self.model_param_count:,} | "
            f"Weights: {self.static_weights_bytes / (1024 ** 2):.1f} MiB"
        )
        return {
            "load_duration_sec": load_duration_sec,
            "params": self.model_param_count,
            "weights_mib": self.static_weights_bytes / (1024 ** 2),
        }

    def _select_and_prune_cache(
        self,
        past_key_values: Any,
        attentions: Optional[Tuple[torch.Tensor, ...]],
        seq_len: int,
    ) -> Tuple[Any, int, List[Tuple[int, int]]]:
        """
        Extracts observation-window attention, performs 1D chunk pooling,
        and physically slices the KV cache.
        """
        S = self.sink_tokens
        W = self.recent_window
        C = self.chunk_size
        K = self.num_chunks

        if seq_len <= self.budget_tokens:
            return past_key_values, seq_len, []

        sink_indices = list(range(min(S, seq_len)))
        recent_start = max(0, seq_len - W)
        recent_indices = list(range(recent_start, seq_len))

        cand_start = S
        cand_end = max(cand_start, seq_len - W)
        num_candidates = cand_end - cand_start

        selected_chunk_ranges: List[Tuple[int, int]] = []
        chunk_indices: List[int] = []

        if num_candidates >= C and attentions is not None:
            # Query window attention accumulation
            W_q = min(self.query_window, seq_len)
            query_start = seq_len - W_q
            query_scores = torch.zeros(seq_len, device=self.device, dtype=torch.float32)

            for layer_attn in attentions:
                # layer_attn: [batch, heads, seq_len, seq_len]
                head_sum = layer_attn[0].sum(dim=0).to(torch.float32)
                query_scores += head_sum[query_start:].sum(dim=0)

            # 1D Window Pooling across candidates [cand_start, cand_end - C + 1]
            window_scores: List[Tuple[float, int]] = []
            for s_idx in range(cand_start, cand_end - C + 1):
                chunk_sc = query_scores[s_idx : s_idx + C].sum().item()
                window_scores.append((chunk_sc, s_idx))

            # Sort descending by pooled score
            window_scores.sort(key=lambda x: x[0], reverse=True)

            selected_starts: List[int] = []
            for sc, s_idx in window_scores:
                # Avoid overlapping chunks
                if not any(abs(s_idx - prev) < C for prev in selected_starts):
                    selected_starts.append(s_idx)
                    selected_chunk_ranges.append((s_idx, s_idx + C))
                    if len(selected_starts) == K:
                        break

            for s_idx, e_idx in selected_chunk_ranges:
                chunk_indices.extend(range(s_idx, e_idx))
        elif num_candidates >= C:
            # Fallback uniform chunk if attentions were absent
            s_idx = cand_start + (num_candidates - C) // 2
            selected_chunk_ranges.append((s_idx, s_idx + C))
            chunk_indices.extend(range(s_idx, s_idx + C))

        retained = sorted(list(set(sink_indices + chunk_indices + recent_indices)))
        retained_tensor = torch.tensor(retained, device=self.device, dtype=torch.long)

        # Physical slicing across all layers
        for layer in past_key_values.layers:
            if layer.keys is not None and layer.values is not None:
                layer.keys = layer.keys[..., retained_tensor, :].contiguous()
                layer.values = layer.values[..., retained_tensor, :].contiguous()

        new_seq_len = past_key_values.layers[0].keys.shape[-2]
        return past_key_values, new_seq_len, selected_chunk_ranges

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 35,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> OGCInferenceResult:
        """Executes OGC-Cache compressed inference."""
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
            seq_len=input_tokens,
        )

        # 1. Prompt Prefill with Attention Extraction
        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                output_attentions=True,
            )
        prefill_duration_ms = (time.perf_counter() - prefill_start) * 1000.0
        tracker.record("prefill_completed")

        past_key_values = outputs.past_key_values
        attentions = outputs.attentions

        # 2. Query-Directed Chunk Pruning
        past_key_values, retained_len, selected_chunks = self._select_and_prune_cache(
            past_key_values=past_key_values,
            attentions=attentions,
            seq_len=input_tokens,
        )

        # Free transient attention tensors
        del outputs
        del attentions
        if self.device.type == "mps":
            torch.mps.empty_cache()

        tracker.record("cache_pruned")

        compressed_theo_bytes = theoretical_kv_cache_bytes(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            element_size=self.element_size,
            seq_len=retained_len,
        )
        savings_pct = (
            ((uncompressed_theo_bytes - compressed_theo_bytes) / uncompressed_theo_bytes) * 100.0
            if uncompressed_theo_bytes > 0
            else 0.0
        )

        # 3. Autoregressive Generation
        cur_token = input_ids[:, -1:]
        generated_token_ids: List[int] = []
        decode_start = time.perf_counter()

        for step in range(max_new_tokens):
            pos_ids = torch.tensor([[input_tokens + step]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                step_out = self.model(
                    input_ids=cur_token,
                    past_key_values=past_key_values,
                    position_ids=pos_ids,
                    use_cache=True,
                )

            next_token_logits = step_out.logits[:, -1, :]
            if do_sample and temperature > 0.0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            tok_id = next_token.item()
            generated_token_ids.append(tok_id)
            cur_token = next_token

            decoded_text = self.tokenizer.decode(generated_token_ids)
            if tok_id == self.tokenizer.eos_token_id or "\n" in decoded_text:
                break

        decode_duration_ms = (time.perf_counter() - decode_start) * 1000.0
        total_duration_ms = prefill_duration_ms + decode_duration_ms
        tracker.record("decoding_completed")

        num_generated = len(generated_token_ids)
        decode_tps = (num_generated / (decode_duration_ms / 1000.0)) if decode_duration_ms > 0 else 0.0
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        return OGCInferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            output_tokens_generated=num_generated,
            input_tokens=input_tokens,
            context_tokens=input_tokens + num_generated,
            sink_tokens=self.sink_tokens,
            recent_window=self.recent_window,
            chunk_size=self.chunk_size,
            num_chunks=self.num_chunks,
            retained_tokens=retained_len,
            selected_chunk_ranges=selected_chunks,
            prefill_time_ms=prefill_duration_ms,
            decode_time_ms=decode_duration_ms,
            total_time_ms=total_duration_ms,
            decode_tokens_per_sec=decode_tps,
            uncompressed_kv_bytes=uncompressed_theo_bytes,
            compressed_kv_bytes=compressed_theo_bytes,
            kv_cache_savings_pct=savings_pct,
            memory_metrics=tracker.summary(),
        )
