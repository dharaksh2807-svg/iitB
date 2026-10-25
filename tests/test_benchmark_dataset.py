"""
Unit Tests for Phase 4 Controlled Long-Context Recall Benchmark Dataset.
"""

import json
import unittest
from src.dataset import BenchmarkGenerator, score_response, normalize_text
from transformers import AutoTokenizer


class TestBenchmarkDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n[TestDataset] Loading tokenizer and dataset files...")
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", use_fast=True)
        with open("data/pilot_dataset.json") as f:
            cls.pilot_items = json.load(f)
        with open("data/benchmark_dataset.json") as f:
            cls.benchmark_items = json.load(f)

    def test_01_dataset_integrity_and_depth_coverage(self):
        """Verify pilot and full benchmark datasets cover all 5 required document depths."""
        expected_depths = {"start", "early_middle", "middle", "late_middle", "end"}

        pilot_depths = {item["depth_category"] for item in self.pilot_items}
        self.assertEqual(pilot_depths, expected_depths)

        bench_depths = {item["depth_category"] for item in self.benchmark_items}
        self.assertEqual(bench_depths, expected_depths)
        self.assertEqual(len(self.pilot_items), 5)
        self.assertEqual(len(self.benchmark_items), 30)
        print("  -> test_01_dataset_integrity_and_depth_coverage passed.")

    def test_02_ground_truth_presence_in_documents(self):
        """Verify that all ground-truth facts are embedded in document text without truncation."""
        for item in self.pilot_items:
            for gt in item["ground_truth_answers"]:
                self.assertIn(gt, item["document"], f"Ground truth {gt} missing in pilot item {item['item_id']}")

        for item in self.benchmark_items:
            for gt in item["ground_truth_answers"]:
                self.assertIn(gt, item["document"], f"Ground truth {gt} missing in benchmark item {item['item_id']}")
        print("  -> test_02_ground_truth_presence_in_documents passed (100% presence verified).")

    def test_03_query_types_and_scoring_logic(self):
        """Verify scoring logic handles single-fact and multi-fact queries deterministically."""
        # Single-fact match
        correct, frac, _ = score_response("The assigned database port is 5438.", ["5438"])
        self.assertTrue(correct)
        self.assertEqual(frac, 1.0)

        # Single-fact mismatch
        correct, frac, _ = score_response("The assigned database port is 8080.", ["5438"])
        self.assertFalse(correct)
        self.assertEqual(frac, 0.0)

        # Multi-fact complete match
        correct, frac, _ = score_response("Region: ap-south-1, Port: 7820", ["ap-south-1", "7820"])
        self.assertTrue(correct)
        self.assertEqual(frac, 1.0)

        # Multi-fact partial match (1 of 2)
        correct, frac, _ = score_response("Region: ap-south-1, Port: unknown", ["ap-south-1", "7820"])
        self.assertFalse(correct)
        self.assertEqual(frac, 0.5)
        print("  -> test_03_query_types_and_scoring_logic passed.")

    def test_04_deterministic_generation_reproducibility(self):
        """Verify that identical seeds produce identical documents and ground truth."""
        gen1 = BenchmarkGenerator(self.tokenizer, base_seed=999)
        gen2 = BenchmarkGenerator(self.tokenizer, base_seed=999)

        item1 = gen1.build_item("test_repro", 1024, "middle", 0.5, "single_fact", 12345)
        item2 = gen2.build_item("test_repro", 1024, "middle", 0.5, "single_fact", 12345)

        self.assertEqual(item1.document, item2.document)
        self.assertEqual(item1.question, item2.question)
        self.assertEqual(item1.ground_truth_answers, item2.ground_truth_answers)
        print("  -> test_04_deterministic_generation_reproducibility passed.")


if __name__ == "__main__":
    unittest.main()
