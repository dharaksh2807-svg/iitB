"""
Pure Sliding Window Cache Adapter (No Attention Sinks).
Maintains only the most recent W tokens in the Key-Value cache,
evicting all earlier tokens including initial tokens (S=0).
Used as a critical baseline to isolate and measure the importance
of attention sinks in StreamingLLM.
"""

from typing import Dict, Any, Optional, Tuple
import torch

from src.streaming_cache import StreamingLLMEngine, StreamingInferenceResult


class PureSlidingWindowEngine(StreamingLLMEngine):
    """
    Pure Local Sliding-Window Cache Engine.
    Discards all tokens outside the recent W-token window (sink_tokens = 0).
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        window_tokens: int = 256,
    ):
        # Explicitly set sink_tokens = 0
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
            sink_tokens=0,
            window_tokens=window_tokens,
        )

    def prune_cache(self, past_key_values: Any) -> Tuple[Any, int]:
        """
        Physically slices Key and Value tensors to retain strictly the last W tokens.
        All initial tokens (sinks) are evicted when context length exceeds W.
        """
        if past_key_values is None or not hasattr(past_key_values, "layers"):
            return past_key_values, 0

        first_layer = past_key_values.layers[0]
        if first_layer.keys is None:
            return past_key_values, 0

        current_seq_len = first_layer.keys.shape[-2]
        if current_seq_len <= self.window_tokens:
            return past_key_values, current_seq_len

        W = self.window_tokens

        # Pure sliding window slice: retain strictly the last W tokens
        for layer in past_key_values.layers:
            if layer.keys is not None and layer.values is not None:
                layer.keys = layer.keys[..., -W:, :].contiguous()
                layer.values = layer.values[..., -W:, :].contiguous()

        new_seq_len = past_key_values.layers[0].keys.shape[-2]
        return past_key_values, new_seq_len
