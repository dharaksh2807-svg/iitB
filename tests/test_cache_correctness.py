"""Offline correctness tests: small real Qwen2/RoPE, no pretrained downloads."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, BatchEncoding, DynamicCache

from src.baseline import UncompressedBaselineEngine
from src.streaming_cache import StreamingLLMEngine
from src.sliding_cache import PureSlidingWindowEngine
from src.h2o_cache import H2OCacheEngine
from src.ogc_cache import OGCCacheEngine
from src.generation import retain, run_generation
from src.metrics import calculate_kv_cache_bytes, checkpoint_device_mib, synchronize_device
from src.selection import select_history


class NumericTokenizer:
    eos_token_id = None

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[int(x) for x in text.split()]])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


class CacheCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        cfg = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=2048, eos_token_id=None,
                          attention_dropout=0.0, tie_word_embeddings=False)
        cfg._attn_implementation = "eager"
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.prompt = " ".join(str(i % 30 + 1) for i in range(40))

    def engine(self, cls=OGCCacheEngine, **kwargs):
        engine = cls(device="cpu", dtype=torch.float32, **kwargs)
        engine.model, engine.tokenizer = self.model, NumericTokenizer()
        engine.max_context_length = 2048
        return engine

    def tokens(self, result):
        return result.generated_token_ids

    def test_first_token_is_prefill_argmax_and_prompt_not_repeated(self):
        engine = self.engine(sink_tokens=2, recent_window=6, chunk_size=4)
        calls = []
        def hook(model, args, kwargs):
            calls.append(kwargs["input_ids"].clone())
        handle = self.model.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            result = engine.generate(self.prompt, max_new_tokens=4)
        finally:
            handle.remove()
        with torch.no_grad():
            ref = self.model(**engine.tokenizer(self.prompt)).logits[:, -1].argmax().item()
        self.assertEqual(result.generated_token_ids[0], ref)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[1].item(), ref)

    def test_absolute_positions_mask_and_exact_budget_for_long_decode(self):
        engine = self.engine(sink_tokens=2, recent_window=6, chunk_size=4, num_chunks=2)
        calls = []
        def hook(model, args, kwargs):
            if kwargs["input_ids"].shape[1] == 1:
                calls.append((kwargs["position_ids"].item(), tuple(kwargs["attention_mask"].shape)))
        handle = self.model.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            result = engine.generate(self.prompt, 100)
        finally:
            handle.remove()
        self.assertEqual([p for p, _ in calls], list(range(40, 139)))
        self.assertTrue(all(shape == (1, 1, 1, 17) for _, shape in calls))
        self.assertEqual(result.cache_lengths, [16] * 100)
        self.assertEqual(result.retained_positions[-6:], list(range(133, 139)))
        self.assertEqual(result.retained_positions[:2], [0, 1])
        self.assertEqual(result.retained_positions, sorted(set(result.retained_positions)))
        self.assertEqual(result.compressed_kv_bytes, 2 * 2 * 2 * 16 * 8 * 4)
        self.assertEqual(result.cache_storage_bytes, result.compressed_kv_bytes)
        self.assertEqual(result.uncompressed_kv_bytes, 2 * 2 * 2 * 139 * 8 * 4)

    def test_all_policies_agree_with_hf_generate_when_no_pruning(self):
        inputs = NumericTokenizer()(self.prompt)
        with torch.no_grad():
            ref = self.model.generate(**inputs, max_new_tokens=7, do_sample=False,
                                      pad_token_id=0).tolist()[0][40:]
        engines = [self.engine(UncompressedBaselineEngine), self.engine(StreamingLLMEngine),
                   self.engine(PureSlidingWindowEngine), self.engine(H2OCacheEngine), self.engine()]
        for engine in engines:
            with self.subTest(engine=type(engine).__name__):
                self.assertEqual(self.tokens(engine.generate(self.prompt, 7)), ref)

    def test_pruning_does_not_rerotate_or_rebase_keys(self):
        engine = self.engine(sink_tokens=2, recent_window=6, chunk_size=4)
        with torch.no_grad():
            output = self.model(**engine.tokenizer(self.prompt), output_attentions=True, use_cache=True)
        keys = [layer.keys.clone() for layer in output.past_key_values.layers]
        cache, length, ranges = engine._select_and_prune_cache(output.past_key_values, output.attentions, 40)
        for expected, layer in zip(keys, cache.layers):
            torch.testing.assert_close(layer.keys, expected[..., engine._last_indices, :], rtol=0, atol=0)
        self.assertEqual(length, 12)

    def test_block_prefill_matches_dense_logits_and_observation_scores(self):
        for query_window in [11, 40]:
            engine = self.engine(UncompressedBaselineEngine)
            engine.query_window = query_window
            engine.prefill_chunk_size = 8
            with torch.no_grad():
                dense = self.model(**engine.tokenizer(self.prompt), output_attentions=True, use_cache=True)
            expected = sum(a[0, :, -query_window:, :].float().sum((0, 1)) for a in dense.attentions)
            scores = []
            shapes = []
            def hook(model, args, output):
                if output.attentions:
                    shapes.extend(a.shape[-2] for a in output.attentions)
            handle = self.model.register_forward_hook(hook)
            try:
                result = run_generation(engine, self.prompt, 1, 0, False,
                                        select=lambda cache, attention, ids: (scores.append(attention.clone()) or cache, list(range(40))),
                                        capture_attentions=True)
            finally:
                handle.remove()
            self.assertEqual(result["telemetry"]["generated_token_ids"], [dense.logits[0, -1].argmax().item()])
            torch.testing.assert_close(scores[0], expected, rtol=1e-5, atol=1e-5)
            self.assertLessEqual(max(shapes), 8)

    def test_zero_one_eos_and_context_limit(self):
        engine = self.engine()
        self.assertEqual(engine.generate(self.prompt, 0).generated_token_ids, [])
        self.assertIsNone(engine.generate(self.prompt, 0).ttft_ms)
        one = engine.generate(self.prompt, 1)
        self.assertEqual(one.decode_steps, 0)
        self.assertEqual(one.decode_tokens_per_sec, 0)
        first = one.generated_token_ids[0]
        with patch.object(self.model.generation_config, "eos_token_id", [first, 31]):
            self.assertEqual(engine.generate(self.prompt, 20).generated_token_ids, [first])
        for count in [-1, 2010]:
            with self.assertRaises(ValueError):
                engine.generate(self.prompt, count)

    def test_newline_is_not_implicit_stop(self):
        engine = self.engine()
        engine.tokenizer.decode = lambda ids, **kwargs: "\n"
        self.assertEqual(len(engine.generate(self.prompt, 8).generated_token_ids), 8)

    def test_repeatability_and_short_prompt_growing_past_budget(self):
        for policy in ["ogc", "query_tokens", "random_chunks", "adaptive_chunks", "entity_chunks"]:
            engine = self.engine(sink_tokens=2, recent_window=6, chunk_size=4, selection_policy=policy)
            a, b = engine.generate("1 2 3 4", 40), engine.generate("1 2 3 4", 40)
            self.assertEqual(a.generated_token_ids, b.generated_token_ids)
            self.assertEqual(a.retained_positions, b.retained_positions)
            self.assertEqual(a.cache_lengths[-1], 12)
            self.assertLessEqual(max(a.cache_lengths), 12)

    def test_compressed_other_policies_exact_budget_and_positions(self):
        engines = [self.engine(StreamingLLMEngine, sink_tokens=2, window_tokens=10),
                   self.engine(PureSlidingWindowEngine, window_tokens=12),
                   self.engine(H2OCacheEngine, heavy_hitters=4, recent_tokens=8)]
        for engine in engines:
            with self.subTest(engine=type(engine).__name__):
                result = engine.generate(self.prompt, 30)
                self.assertEqual(result.cache_lengths, [12] * 30)
                self.assertEqual(result.retained_positions[-1], 68)
                self.assertEqual(result.cache_storage_bytes, result.compressed_kv_bytes)

    def test_explicit_cache_position_on_older_api_and_replaced_cache(self):
        underlying = self.model
        seen = []
        class Adapter:
            generation_config = underlying.generation_config
            def forward(self, input_ids, cache_position=None, **kwargs):
                seen.extend(cache_position.tolist())
                out = underlying(input_ids=input_ids, **kwargs)
                replacement = DynamicCache()
                for i, layer in enumerate(out.past_key_values.layers):
                    replacement.update(layer.keys.clone(), layer.values.clone(), i)
                out.past_key_values = replacement
                return out
            __call__ = forward
        engine = self.engine(sink_tokens=2, recent_window=6, chunk_size=4)
        engine.model = Adapter()
        result = engine.generate(self.prompt, 6)
        self.assertEqual(seen, list(range(45)))
        self.assertEqual(result.retained_tokens, 12)
        self.assertEqual(result.cache_position_mode, "explicit absolute")

    def test_matched_budget_selection_and_controls(self):
        scores = torch.zeros(1500)
        scores[400:464] = 10
        for budget in [260, 516, 1028]:
            capacity = budget - 4 - 192
            for policy in ["ogc", "query_tokens", "random_chunks", "adaptive_chunks", "entity_chunks"]:
                indices, ranges = select_history(scores, 4, 1308, capacity, 32, policy, 4, [399, 470])
                self.assertEqual(len(set(indices)), capacity)
                self.assertTrue(all(4 <= i < 1308 for i in indices))
                self.assertEqual((indices, ranges), select_history(scores, 4, 1308, capacity, 32, policy, 4, [399, 470]))
        selected, _ = select_history(scores, 4, 1308, 64, 32, "query_tokens", 4)
        self.assertEqual(selected, list(range(400, 464)))
        a, _ = select_history(scores, 4, 1308, 64, 32, "random_chunks", 4)
        b, _ = select_history(scores, 4, 1308, 64, 32, "random_chunks", 5)
        self.assertNotEqual(a, b)

    def test_storage_views_and_physical_pruning(self):
        key, value = torch.zeros(1, 1, 100, 4), torch.zeros(1, 1, 100, 4)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=key[..., -10:, :], values=value[..., -10:, :])])
        size, _, meta = calculate_kv_cache_bytes(cache)
        self.assertEqual(size, 320)
        self.assertEqual(meta["storage_bytes"], 3200)
        cache = retain(cache, [0, 2, 4])
        self.assertEqual(calculate_kv_cache_bytes(cache)[0], 96)
        self.assertEqual(calculate_kv_cache_bytes(cache)[2]["storage_bytes"], 96)

    def test_nested_checkpoints_and_synchronization(self):
        self.assertEqual(checkpoint_device_mib({"a": {"device_allocated_mib": 3}, "b": {"device_allocated_mib": 9}}), 9)
        with patch("torch.cuda.synchronize") as cuda, patch("torch.mps.synchronize") as mps:
            synchronize_device(torch.device("cuda"))
            synchronize_device(torch.device("mps"))
            synchronize_device(torch.device("cpu"))
            cuda.assert_called_once()
            mps.assert_called_once()

    def test_budget_crossing_uses_pending_query_scores(self):
        engine = self.engine(sink_tokens=2, recent_window=6, chunk_size=4, selection_policy="query_tokens")
        # A prompt below budget must defer selection until decoding crosses it.
        result = engine.generate("1 2 3 4 5 6 7 8 9 10", 20)
        self.assertEqual(result.cache_lengths[:3], [10, 11, 12])
        self.assertEqual(result.cache_lengths[3:], [12] * 17)
        self.assertTrue(result.selected_chunk_ranges)

    def test_benchmark_extracts_actual_bytes_and_nested_telemetry(self):
        from run_research_benchmark import record_result
        engine = self.engine(sink_tokens=2, recent_window=6, chunk_size=4)
        result = engine.generate(self.prompt, 3)
        result.memory_metrics["post_decoding"]["device_allocated_mib"] = 9.0
        item = dict(item_id="fixture", target_context_length=40, depth_category="middle",
                    query_type="single_fact", ground_truth_answers=["unused"])
        row = record_result(result, item, "ogc")
        self.assertEqual(row["compressed_kv_bytes"], result.compressed_kv_bytes)
        self.assertEqual(row["checkpoint_max_device_allocated_mib"], 9.0)
        self.assertEqual(row["decode_steps"], 2)

    def test_timing_contains_selection_and_matches_step_definition(self):
        result = self.engine(sink_tokens=2, recent_window=6, chunk_size=4).generate(self.prompt, 8)
        self.assertEqual(result.decode_steps, 7)
        self.assertGreater(result.ttft_ms, result.prefill_time_ms)
        self.assertGreater(result.total_time_ms, result.prefill_time_ms + result.selection_time_ms)
        self.assertAlmostEqual(result.selection_time_ms, result.initial_selection_time_ms + result.decode_pruning_time_ms)
        self.assertAlmostEqual(result.decode_tokens_per_sec, 7000 / result.decode_time_ms)


if __name__ == "__main__":
    unittest.main()
