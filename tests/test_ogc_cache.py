"""
Unit and integration tests for Observation-Guided Chunked Cache (OGC-Cache).
"""

import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from src.ogc_cache import OGCCacheEngine
from src.metrics import theoretical_kv_cache_bytes


class TestOGCCacheEngine(unittest.TestCase):
    """Validation test suite for OGCCacheEngine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = OGCCacheEngine(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            sink_tokens=4,
            recent_window=128,
            chunk_size=32,
            num_chunks=1,
            query_window=32,
            dtype=torch.bfloat16,
        )
        cls.engine.load_model()

    def test_01_ogc_cache_physical_sizing(self):
        """Verifies that OGC-Cache bounds physical tensor shape to exact budget."""
        prompt = "Antigravity long context verification prompt. " * 30
        inputs = self.engine.tokenizer(prompt, return_tensors="pt").to(self.engine.device)
        seq_len = inputs["input_ids"].shape[1]
        self.assertGreater(seq_len, 200)

        with torch.no_grad():
            outputs = self.engine.model(
                input_ids=inputs["input_ids"],
                use_cache=True,
                output_attentions=True,
            )

        past_key_values = outputs.past_key_values
        pruned_cache, retained_len, selected_chunks = self.engine._select_and_prune_cache(
            past_key_values=past_key_values,
            attentions=outputs.attentions,
            seq_len=seq_len,
        )

        expected_budget = self.engine.sink_tokens + (self.engine.chunk_size * self.engine.num_chunks) + self.engine.recent_window
        self.assertEqual(retained_len, expected_budget)
        self.assertEqual(len(selected_chunks), 1)

        # Check physical tensor size on layer 0
        k_tensor = pruned_cache.layers[0].keys
        self.assertEqual(k_tensor.shape[-2], expected_budget)

        actual_bytes = sum(
            layer.keys.numel() * layer.keys.element_size() +
            layer.values.numel() * layer.values.element_size()
            for layer in pruned_cache.layers
        )
        expected_bytes = theoretical_kv_cache_bytes(
            num_layers=self.engine.num_layers,
            num_kv_heads=self.engine.num_kv_heads,
            head_dim=self.engine.head_dim,
            element_size=self.engine.element_size,
            seq_len=expected_budget,
        )
        self.assertEqual(actual_bytes, expected_bytes)
        print(f"  -> test_01_ogc_cache_physical_sizing passed ({actual_bytes} bytes, {expected_budget} tokens).")

    def test_02_query_guided_chunk_selection(self):
        """Verifies that synthetic query attention successfully isolates the target chunk."""
        seq_len = 300
        S = 4
        W = 128
        C = 32
        K = 1

        # Create dummy attention weights targeting index 45
        dummy_attentions = []
        for _ in range(self.engine.num_layers):
            attn = torch.zeros(1, self.engine.num_kv_heads, seq_len, seq_len, device=self.engine.device)
            # Query window [seq_len-32 : seq_len] attends strongly to index 45
            attn[:, :, seq_len - 32 :, 45] = 10.0
            dummy_attentions.append(attn)

        dummy_cache = DynamicCache()
        for layer_idx in range(self.engine.num_layers):
            k = torch.zeros(1, 2, seq_len, 128, device=self.engine.device, dtype=torch.bfloat16)
            v = torch.zeros(1, 2, seq_len, 128, device=self.engine.device, dtype=torch.bfloat16)
            dummy_cache.update(k, v, layer_idx)

        pruned, retained_len, selected_chunks = self.engine._select_and_prune_cache(
            past_key_values=dummy_cache,
            attentions=tuple(dummy_attentions),
            seq_len=seq_len,
        )

        self.assertEqual(len(selected_chunks), 1)
        start, end = selected_chunks[0]
        self.assertTrue(start <= 45 < end, f"Expected chunk to contain index 45, got range {start}..{end}")
        print(f"  -> test_02_query_guided_chunk_selection passed. Chunk {start}..{end} contains target 45.")

    def test_03_inference_execution(self):
        """Verifies end-to-end execution of generate() with OGC-Cache."""
        prompt = "Question: What is 25 + 75? Answer directly with only the number:\nAnswer:"
        result = self.engine.generate(prompt=prompt, max_new_tokens=15)

        self.assertIsNotNone(result.generated_text)
        self.assertGreater(result.output_tokens_generated, 0)
        self.assertGreater(result.prefill_time_ms, 0)
        self.assertGreater(result.decode_tokens_per_sec, 0)
        print(f"  -> test_03_inference_execution passed. Output: {repr(result.generated_text)}")


if __name__ == "__main__":
    unittest.main()
