"""
Unit and Integration Tests for Phase 3 Uncompressed Baseline.
"""

import unittest
import torch
from src.metrics import calculate_kv_cache_bytes, theoretical_kv_cache_bytes
from src.baseline import UncompressedBaselineEngine


class TestBaselineInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[TestBaseline] Initializing UncompressedBaselineEngine...")
        # Use Qwen2.5-1.5B-Instruct on available device (prefer mps or cpu)
        cls.engine = UncompressedBaselineEngine(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            device="mps" if torch.backends.mps.is_available() else "cpu",
            dtype=torch.float16,
        )
        cls.meta = cls.engine.load_model()

    def test_01_model_architecture_and_parameters(self):
        """Verify model parameters match Qwen2.5-1.5B specs."""
        self.assertEqual(self.meta["num_layers"], 28)
        self.assertEqual(self.meta["num_kv_heads"], 2)
        self.assertEqual(self.meta["head_dim"], 128)
        self.assertEqual(self.meta["param_count"], 1543714304)
        print("  -> test_01_model_architecture_and_parameters passed.")

    def test_02_kv_cache_formula_conformance(self):
        """Verify that calculate_kv_cache_bytes matches theoretical formula exactly."""
        # Theoretical bytes for seq_len=100
        theo = theoretical_kv_cache_bytes(
            num_layers=28,
            num_kv_heads=2,
            head_dim=128,
            element_size=2,  # float16
            seq_len=100,
            batch_size=1,
        )
        # Expected: 1 * 2 * 28 * 2 * 100 * 128 * 2 = 2,867,200 bytes
        self.assertEqual(theo, 2867200)

        # Create dummy DynamicCache mock
        class MockLayer:
            def __init__(self):
                self.keys = torch.zeros((1, 2, 100, 128), dtype=torch.float16)
                self.values = torch.zeros((1, 2, 100, 128), dtype=torch.float16)

        class MockCache:
            def __init__(self):
                self.layers = [MockLayer() for _ in range(28)]

        mock_cache = MockCache()
        actual, layers, _ = calculate_kv_cache_bytes(mock_cache)
        self.assertEqual(layers, 28)
        self.assertEqual(actual, theo)
        print("  -> test_02_kv_cache_formula_conformance passed (Exact 2,867,200 bytes).")

    def test_03_baseline_generation_and_timing(self):
        """Verify uncompressed baseline generation produces valid output and distinct timings."""
        prompt = "Question: What is 15 + 25? Answer:"
        res = self.engine.generate(prompt=prompt, max_new_tokens=15, do_sample=False)

        self.assertGreater(res.input_tokens, 0)
        self.assertGreater(res.generated_tokens, 0)
        self.assertEqual(res.total_tokens, res.input_tokens + res.generated_tokens)
        self.assertIn("40", res.output_text)  # 15 + 25 = 40

        # Timing checks
        self.assertGreater(res.prefill_time_ms, 0.0)
        self.assertGreater(res.decode_time_ms, 0.0)
        self.assertGreater(res.total_time_ms, 0.0)
        self.assertGreater(res.decode_tokens_per_sec, 0.0)

        # KV cache checks
        self.assertAlmostEqual(res.kv_scaling_match_pct, 100.0, places=2)
        print(f"  -> test_03_baseline_generation_and_timing passed. Output: '{res.output_text.strip()}' | KV match: {res.kv_scaling_match_pct:.2f}%")

    def test_04_prefill_and_decode_memory_checkpoints(self):
        """Verify memory checkpoints exist and distinguish phases."""
        prompt = "System initialization test prompt."
        res = self.engine.generate(prompt=prompt, max_new_tokens=5, do_sample=False)
        cps = res.checkpoints

        self.assertIn("inference_start", cps)
        self.assertIn("post_prefill", cps)
        self.assertIn("post_decoding", cps)

        # Post-prefill KV bytes should reflect input length
        # Post-decoding KV bytes should reflect input length + generated tokens
        self.assertGreater(cps["post_decoding"]["kv_cache_mib"], cps["post_prefill"]["kv_cache_mib"])
        print("  -> test_04_prefill_and_decode_memory_checkpoints passed.")


if __name__ == "__main__":
    unittest.main()
