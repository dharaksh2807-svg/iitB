"""
Automated Unit and Integration Tests for Additional Baselines (Phase 7):
1. Pure Sliding Window Cache (No Sinks, S=0)
2. H2O Heavy-Hitter Oracle Cache (H > 0, W > 0)
"""

import unittest
import torch
from transformers import DynamicCache

from src.sliding_cache import PureSlidingWindowEngine
from src.h2o_cache import H2OCacheEngine
from src.metrics import calculate_kv_cache_bytes


class TestAdditionalBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "mps" if torch.backends.mps.is_available() else "cpu"
        cls.model_id = "Qwen/Qwen2.5-1.5B-Instruct"

        # Initialize engines
        cls.engine_sliding = PureSlidingWindowEngine(
            model_name_or_path=cls.model_id,
            device=cls.device,
            window_tokens=128,
        )
        cls.engine_sliding.load_model()

        cls.engine_h2o = H2OCacheEngine(
            model_name_or_path=cls.model_id,
            device=cls.device,
            heavy_hitters=32,
            recent_tokens=96,
        )
        cls.engine_h2o.load_model()

    def test_01_sliding_window_prunes_correctly(self):
        """Verify PureSlidingWindowEngine strictly keeps W tokens with S=0."""
        cache = DynamicCache()
        layers = 28
        heads = 2
        dim = 128
        seq_len = 500

        for _ in range(layers):
            k = torch.randn(1, heads, seq_len, dim, device=self.device, dtype=torch.float16)
            v = torch.randn(1, heads, seq_len, dim, device=self.device, dtype=torch.float16)
            cache.update(k, v, len(cache))

        pruned_cache, new_len = self.engine_sliding.prune_cache(cache)
        self.assertEqual(new_len, 128)
        self.assertEqual(pruned_cache.layers[0].keys.shape[-2], 128)
        self.assertEqual(self.engine_sliding.sink_tokens, 0)
        self.assertEqual(self.engine_sliding.window_tokens, 128)

        # Expected size: 128 tokens * 28,672 bytes/token = 3,670,016 bytes
        expected_bytes = 128 * 28 * 2 * 128 * 2 * 2
        actual_bytes, _, _ = calculate_kv_cache_bytes(pruned_cache)
        self.assertEqual(actual_bytes, expected_bytes)
        print(f"  -> test_01_sliding_window_prunes_correctly passed ({actual_bytes} bytes).")

    def test_02_h2o_cache_prunes_and_retains_heavy_hitters(self):
        """Verify H2OCacheEngine selects H heavy hitters and W recent tokens."""
        cache = DynamicCache()
        layers = 28
        heads = 2
        dim = 128
        seq_len = 300

        for _ in range(layers):
            k = torch.randn(1, heads, seq_len, dim, device=self.device, dtype=torch.float16)
            v = torch.randn(1, heads, seq_len, dim, device=self.device, dtype=torch.float16)
            cache.update(k, v, len(cache))

        # Synthetic attention tensor: [1, 12, seq_len, seq_len]
        # Make token 42 have very high attention across all layers
        attentions = []
        for _ in range(layers):
            attn = torch.ones(1, 12, seq_len, seq_len, device=self.device, dtype=torch.float32) * 0.01
            attn[..., :, 42] = 5.0 # high attention on token 42
            attentions.append(attn)

        pruned_cache, new_len, heavy_hitters = self.engine_h2o.prune_cache_with_attentions(
            cache, tuple(attentions)
        )
        self.assertEqual(new_len, 32 + 96) # H=32, W=96 -> 128 tokens
        self.assertIn(42, heavy_hitters) # Token 42 MUST be chosen as a heavy hitter!
        self.assertEqual(len(heavy_hitters), 32)
        print(f"  -> test_02_h2o_cache_prunes_and_retains_heavy_hitters passed (Token 42 in HH: {42 in heavy_hitters}).")

    def test_03_inference_execution_both_engines(self):
        """Verify end-to-end inference execution for both additional baselines."""
        prompt = "Question: What is 12 + 18? Answer:"
        
        # Test Sliding Window
        res_sw = self.engine_sliding.generate(prompt=prompt, max_new_tokens=15, do_sample=False)
        self.assertTrue(len(res_sw.output_text) > 0)
        self.assertGreaterEqual(res_sw.actual_cached_tokens, 1)

        # Test H2O
        res_h2o = self.engine_h2o.generate(prompt=prompt, max_new_tokens=15, do_sample=False)
        self.assertTrue(len(res_h2o.output_text) > 0)
        self.assertGreaterEqual(res_h2o.actual_cached_tokens, 1)
        print(f"  -> test_03_inference_execution_both_engines passed (SW: {res_sw.output_text!r}, H2O: {res_h2o.output_text!r}).")


if __name__ == "__main__":
    unittest.main()
