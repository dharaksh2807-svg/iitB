"""Small deterministic tests for benchmark validity, without model weights."""
import unittest
from dataclasses import replace
import torch
from src.research import build_cases, score, paired_interval, summarize
from run_research_benchmark import make_engines, record_result
from src.generation import DEFAULT_MODEL, DEFAULT_REVISION


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class ResearchValidity(unittest.TestCase):
    def test_300_independently_seeded_cases_and_distributed_facts(self):
        cases = build_cases(CharacterTokenizer())
        self.assertEqual(len(cases), 300)
        self.assertEqual(len({c["seed"] for c in cases}), 300)
        self.assertEqual({c["target_context_length"] for c in cases}, {1024, 2048, 4096, 8192})
        for case in cases:
            self.assertLessEqual(abs(case["actual_token_length"] - case["target_context_length"]), 2)
            self.assertTrue(all(a in case["document"] for a in case["ground_truth_answers"]))
            if case["query_type"] == "multi_fact":
                self.assertEqual(len(case["fact_token_offsets"]), 2)
                self.assertGreater(abs(case["fact_token_offsets"][0] - case["fact_token_offsets"][1]), 50)
            if case["query_type"] == "adversarial":
                self.assertIn("ARCHIVED obsolete", case["document"])
        self.assertEqual(cases[:15], build_cases(CharacterTokenizer(), lengths=[1024], repeats=5)[:15])

    def test_answer_boundaries_and_multifact_scoring(self):
        self.assertFalse(score("15438", ["5438"])["is_correct"])
        self.assertTrue(score("The port is 5438.", ["5438"])["is_correct"])
        self.assertFalse(score("5438.2", ["5438"])["is_correct"])
        self.assertFalse(score("5438", ["5438", "eu-west-1"])["is_correct"])
        self.assertTrue(score("Port: 5438; region: eu-west-1", ["5438", "eu-west-1"])["is_correct"])
        self.assertFalse(score("eu-west-10", ["eu-west-1"])["is_correct"])
        self.assertTrue(score("5438, eu-west-1", ["5438", "eu-west-1"])["exact_match"])

    def test_paired_bootstrap_and_missing_pairs(self):
        def row(i, success):
            return dict(item_id=str(i), context_length=1024, depth_category="middle", query_type="single_fact", is_correct=success)
        a, b = [row(i, i % 2 == 0) for i in range(20)], [row(i, True) for i in range(20)]
        result = paired_interval(a, b)
        self.assertEqual(result["delta_accuracy_pp"], -50)
        self.assertLess(result["paired_bootstrap_95_ci_pp"][0], -50)
        self.assertGreater(result["paired_bootstrap_95_ci_pp"][1], -50)
        self.assertEqual(result, paired_interval(a, b))
        self.assertEqual(paired_interval(a, a)["paired_bootstrap_95_ci_pp"], [0, 0])
        with self.assertRaises(ValueError):
            paired_interval(a[:-1], b)

    def test_all_matched_budgets_and_policy_settings(self):
        engines = make_engines(DEFAULT_MODEL, DEFAULT_REVISION, "cpu", torch.float32,
                               [260, 516, 1028], ["full", "streaming", "sliding", "h2o_prefill_approx", "ogc", "query_tokens", "random_chunks", "adaptive_chunks", "entity_chunks"],
                               64, 256, 17)
        self.assertEqual(len(engines), 25)
        for config, engine in engines.items():
            self.assertEqual(engine.revision, DEFAULT_REVISION)
            self.assertEqual(engine.attn_implementation, "eager")
            self.assertEqual(engine.prefill_chunk_size, 256)
            if config != "full":
                budget = int(config.split("/B")[1].split("/")[0])
                self.assertEqual(getattr(engine, "cache_budget", getattr(engine, "budget_tokens", None)), budget)


if __name__ == "__main__":
    unittest.main()
