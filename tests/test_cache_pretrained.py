"""Opt-in checks on the pinned 1.5B model: RUN_PRETRAINED_TESTS=1."""
import os
import unittest
from unittest.mock import patch
import torch
from src.baseline import UncompressedBaselineEngine
from src.ogc_cache import OGCCacheEngine


@unittest.skipUnless(os.environ.get("RUN_PRETRAINED_TESTS") == "1", "Opt-in pretrained integration")
class PretrainedCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = UncompressedBaselineEngine(dtype=torch.bfloat16, attn_implementation="eager")
        cls.base.load_model()

    def ogc(self, **kwargs):
        engine = OGCCacheEngine(device=str(self.base.device), dtype=torch.bfloat16, **kwargs)
        engine.model, engine.tokenizer = self.base.model, self.base.tokenizer
        engine.max_context_length = self.base.max_context_length
        return engine

    def test_no_pruning_agrees_with_hf_and_first_token(self):
        prompt = "Question: What is 15 + 25? Answer:"
        inputs = self.base.tokenizer(prompt, return_tensors="pt").to(self.base.device)
        with torch.inference_mode():
            expected = self.base.model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                                repetition_penalty=1.0).tolist()[0][inputs.input_ids.shape[1]:]
        self.assertEqual(self.base.generate(prompt, 12).generated_token_ids, expected)
        self.assertEqual(self.ogc().generate(prompt, 12).generated_token_ids, expected)

    def test_long_decode_physical_budget_and_first_token(self):
        prompt = "The current deployment record is nominal. " * 100 + "Continue the report:"
        engine = self.ogc()
        saved_eos = engine.tokenizer.eos_token_id
        try:
            engine.tokenizer.eos_token_id = None
            with patch.object(engine.model.generation_config, "eos_token_id", None):
                result = engine.generate(prompt, 100)
                baseline = self.base.generate(prompt, 1)
        finally:
            engine.tokenizer.eos_token_id = saved_eos
        self.assertEqual(result.generated_token_ids[0], baseline.generated_token_ids[0])
        self.assertEqual(len(result.generated_token_ids), 100)
        self.assertEqual(result.cache_lengths, [260] * 100)
        self.assertEqual(result.retained_tokens, 260)
        self.assertEqual(result.compressed_kv_bytes, 260 * 28672)
        self.assertEqual(result.cache_storage_bytes, result.compressed_kv_bytes)
        self.assertEqual(result.retained_positions[-1], result.input_tokens + 98)


    def test_held_query_first_token_matches_hf(self):
        prompt = "Document: the port is 54321. Question: return the port. Answer:"
        inputs = self.base.tokenizer(prompt, return_tensors="pt").to(self.base.device)
        n = inputs.input_ids.shape[1]
        with torch.inference_mode():
            expected = self.base.model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                                repetition_penalty=1.0).tolist()[0][n:]
        engine = self.ogc()
        engine.document_token_count = n // 2
        result = engine.generate(prompt, 12)
        # BF16 batched vs tokenwise GEMMs can diverge at later near-tied logits.
        # Full-sequence equivalence is independently covered with FP32 tiny Qwen2.
        self.assertEqual(result.generated_token_ids[0], expected[0])
        self.assertEqual(result.query_tokens, n-n//2)

    def test_held_query_long_generation_real_storage(self):
        prompt = "The current deployment record is nominal. " * 100 + "Continue the report:"
        engine = self.ogc(selection_policy="entity_balanced")
        n = len(engine.tokenizer.encode(prompt))
        engine.document_token_count = n-20
        saved_eos = engine.tokenizer.eos_token_id
        try:
            engine.tokenizer.eos_token_id = None
            with patch.object(engine.model.generation_config, "eos_token_id", None):
                result = engine.generate(prompt, 100)
        finally:
            engine.tokenizer.eos_token_id = saved_eos
        self.assertEqual(result.query_cache_lengths, [260]*21)
        self.assertEqual(result.cache_lengths, [260]*100)
        self.assertEqual(result.retained_positions[-1], n+98)
        self.assertEqual(result.compressed_kv_bytes, 260*28672)
        self.assertEqual(result.compressed_kv_bytes, result.cache_storage_bytes)


if __name__ == "__main__":
    unittest.main()
