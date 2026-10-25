"""
Unit and Integration Tests for Phase 5 StreamingLLM Cache Compression.
"""

import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.metrics import calculate_kv_cache_bytes, theoretical_kv_cache_bytes
from src.streaming_cache import StreamingLLMEngine


class TestStreamingLLMCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[TestStreamingLLM] Initializing StreamingLLMEngine (S=4, W=128)...")
        cls.engine = StreamingLLMEngine(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            device="mps" if torch.backends.mps.is_available() else "cpu",
            dtype=torch.float16,
            sink_tokens=4,
            window_tokens=128,
        )
        cls.engine.load_model()

    def test_01_cache_pruning_physical_tensor_reduction(self):
        """Verify that prune_cache physically slices tensor shapes to S + W."""
        # Create mock 28-layer DynamicCache with seq_len=500
        class MockLayer:
            def __init__(self):
                self.keys = torch.zeros((1, 2, 500, 128), dtype=torch.float16)
                self.values = torch.zeros((1, 2, 500, 128), dtype=torch.float16)

        class MockCache:
            def __init__(self):
                self.layers = [MockLayer() for _ in range(28)]

        mock_cache = MockCache()
        before_bytes, _, _ = calculate_kv_cache_bytes(mock_cache)
        # 500 * 28,672 = 14,336,000 bytes

        pruned_cache, pruned_len = self.engine.prune_cache(mock_cache)
        self.assertEqual(pruned_len, 4 + 128)  # S=4, W=128 -> 132 tokens
        self.assertEqual(pruned_cache.layers[0].keys.shape[-2], 132)
        self.assertEqual(pruned_cache.layers[0].values.shape[-2], 132)

        after_bytes, _, _ = calculate_kv_cache_bytes(pruned_cache)
        # 132 * 28,672 = 3,784,704 bytes
        self.assertEqual(after_bytes, 132 * 28672)
        expected_savings = (1.0 - 132 / 500) * 100.0  # 73.6%
        actual_savings = (1.0 - after_bytes / before_bytes) * 100.0
        self.assertAlmostEqual(actual_savings, expected_savings, places=2)
        print(f"  -> test_01_cache_pruning_physical_tensor_reduction passed. Pruned 500 -> 132 tokens ({actual_savings:.1f}% savings).")

    def test_02_streaming_inference_execution(self):
        """Verify StreamingLLM generation runs on real model and records savings."""
        prompt = "System initialization: The primary server port is 5438. " * 30 + "Question: What is 2 + 2? Answer:"
        res = self.engine.generate(prompt=prompt, max_new_tokens=10, do_sample=False)

        self.assertGreater(res.input_tokens, 132)
        self.assertEqual(res.actual_cached_tokens, 132)  # Strictly bounded to 4 + 128
        self.assertGreater(res.kv_cache_savings_pct, 50.0)
        self.assertIn("4", res.output_text)
        print(f"  -> test_02_streaming_inference_execution passed. Savings: {res.kv_cache_savings_pct:.2f}% | Output: '{res.output_text.strip()}'")

    def test_03_memory_checkpoints_show_pruning(self):
        """Verify checkpoints explicitly record raw prefill peak vs post-prefill pruned memory."""
        prompt = "Cluster status telemetry entry. " * 30 + "Status check: All systems nominal."
        res = self.engine.generate(prompt=prompt, max_new_tokens=5, do_sample=False)
        cps = res.checkpoints

        self.assertIn("peak_prefill_raw", cps)
        self.assertIn("post_prefill_pruned", cps)
        self.assertIn("post_decoding", cps)

        # Pre-pruned raw cache must be larger than post-pruned cache
        raw_prefill_mib = cps["peak_prefill_raw"]["kv_cache_mib"]
        pruned_mib = cps["post_prefill_pruned"]["kv_cache_mib"]
        self.assertGreater(raw_prefill_mib, pruned_mib)
        print(f"  -> test_03_memory_checkpoints_show_pruning passed. Raw prefill: {raw_prefill_mib:.2f} MiB -> Pruned: {pruned_mib:.2f} MiB.")


if __name__ == "__main__":
    unittest.main()
