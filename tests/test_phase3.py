"""Targeted Phase 3 regression tests; tiny real Qwen2, no weight download."""
import re
import unittest
from unittest.mock import patch
import torch
from transformers import BatchEncoding
import test_cache_correctness as fixtures
NumericTokenizer = fixtures.NumericTokenizer
from src.generation import run_generation
from src.baseline import UncompressedBaselineEngine
from src.ogc_cache import OGCCacheEngine
from src.selection import observation_scores, select_records, record_spans
from run_research_benchmark import make_engines, prepare_item


class OffsetNumericTokenizer(NumericTokenizer):
    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        if return_offsets_mapping:
            return {'offset_mapping':[(m.start(),m.end()) for m in re.finditer(r'\d+',text)]}
        return super().__call__(text,return_tensors or 'pt')


class Phase3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CacheCorrectness.setUpClass()
        cls.model=fixtures.CacheCorrectness.model
        cls.prompt=fixtures.CacheCorrectness.prompt

    def engine(self, **kwargs):
        e=OGCCacheEngine(device='cpu',dtype=torch.float32,**kwargs)
        e.model=self.model;e.tokenizer=OffsetNumericTokenizer();e.max_context_length=2048
        return e

    def test_floor_subtraction_cancels_sink_not_salient_record(self):
        a=torch.zeros(1,4,8,40)
        a[:,:,:,4:8]=10 # persistent sink-adjacent distraction
        a[:,:,7,20:24]=5 # query-specific signal
        raw=observation_scores((a,),40,8,'cpu')
        corrected=observation_scores((a,),40,8,'cpu',corrected=True)
        self.assertGreater(raw[4],raw[20]);self.assertEqual(corrected[4],0)
        self.assertGreater(corrected[20],0)
        from transformers import DynamicCache
        cache=DynamicCache()
        for i in range(2): cache.update(torch.zeros(1,2,40,8),torch.zeros(1,2,40,8),i)
        e=self.engine(sink_tokens=2,recent_window=6,chunk_size=4,selection_policy='ogc_corrected')
        e._select_and_prune_cache(cache,(a,),40)
        self.assertEqual(e._last_indices[:2],[0,1])
        self.assertTrue(any(i in e._last_indices for i in range(20,24)));self.assertNotIn(4,e._last_indices)

    def test_corrected_block_scores_equal_dense_across_window_boundary(self):
        e=self.engine();e.prefill_chunk_size=8;e.query_window=11;e.attention_score_correction=True
        # Disable pruning in this score equivalence check.
        e.budget_tokens=1000
        with torch.no_grad(): out=self.model(**e.tokenizer(self.prompt),output_attentions=True,use_cache=True)
        expected=observation_scores(out.attentions,40,11,'cpu',corrected=True)
        scores=[]
        run_generation(e,self.prompt,1,0,False,select=lambda c,a,ids:(scores.append(a.clone()) or c,list(range(40))),capture_attentions=True)
        torch.testing.assert_close(scores[0],expected,rtol=1e-5,atol=1e-5)

    def test_whole_relationships_disperse_under_sink_noise(self):
        # Two distant complete records, each fitting capacity; irrelevant sink huge.
        scores=torch.zeros(180);scores[:40]=1000
        spans=[(4,36,'system preamble'),(40,58,'Record ITEM-123456 port is 54321.'),
               (100,118,'Record ITEM-123456 region is eu-west-3.'),(120,155,'unrelated filler')]
        selected,ranges=select_records(scores,4,160,40,spans,'Return ITEM-123456 port and region')
        self.assertTrue(set(range(40,58)).issubset(selected))
        self.assertTrue(set(range(100,118)).issubset(selected))
        self.assertEqual(len(selected),40)

    def test_boundaries_do_not_split_identifiers_or_decimals(self):
        class Chars:
            def __call__(self,text,**kw): return {'offset_mapping':[(i,i+1) for i in range(len(text))]}
        text='Record ITEM-AB12 value 12.34.\nRecord ITEM-AB12 region east-3.\n'
        spans=record_spans(Chars(),text,len(text))
        self.assertEqual(len(spans),2)
        self.assertIn('ITEM-AB12 value 12.34.',spans[0][2])

    def test_held_query_no_pruning_matches_continuation_and_hf(self):
        e=self.engine();e.document_token_count=25
        held=e.generate(self.prompt,8)
        e.document_token_count=None
        continuation=e.generate(self.prompt,8)
        self.assertEqual(held.generated_token_ids,continuation.generated_token_ids)
        self.assertEqual(held.query_tokens,15)
        self.assertEqual(held.evaluation_protocol,'held_query')
        self.assertEqual(held.query_cache_lengths,list(range(25,41)))

    def test_first_answer_token_comes_from_pruned_query_forward(self):
        e=self.engine(sink_tokens=2,recent_window=6,chunk_size=4)
        e.document_token_count=25
        calls=[]
        def hook(model,args,kwargs,output):
            absolute=kwargs['position_ids'][0,-1].item();calls.append((absolute,kwargs['input_ids'].shape[1]))
            output.logits.fill_(-100)
            output.logits[:,-1,7 if absolute==39 else 3]=100
        h=self.model.register_forward_hook(hook,with_kwargs=True)
        try: r=e.generate(self.prompt,1)
        finally:h.remove()
        self.assertEqual(r.generated_token_ids,[7])
        self.assertEqual(calls,[(24,25)]+[(i,1) for i in range(25,40)])
        self.assertEqual(r.query_cache_lengths,[12]*16)
        self.assertLess(max(r.document_retained_positions),25)
        self.assertEqual(r.retained_positions[-1],39)

    def test_long_query_and_decode_budget_shapes_positions_repeat(self):
        for policy in ['ogc','ogc_corrected','entity_balanced']:
            with self.subTest(policy=policy):
                e=self.engine(sink_tokens=2,recent_window=6,chunk_size=4,selection_policy=policy)
                e.document_token_count=20
                r=e.generate(self.prompt,100);again=e.generate(self.prompt,100)
                self.assertEqual(r.generated_token_ids,again.generated_token_ids)
                self.assertEqual(r.query_cache_lengths,[12]*21)
                self.assertEqual(r.cache_lengths,[12]*100)
                self.assertEqual(r.retained_positions[-1],138)
                self.assertEqual(r.compressed_kv_bytes,2*2*2*12*8*4)
                self.assertEqual(r.cache_storage_bytes,r.compressed_kv_bytes)
                self.assertGreater(r.ttft_ms,r.prefill_time_ms+r.query_time_ms)

    def test_query_eos_is_teacher_forced_not_answer_stop(self):
        e=self.engine();e.document_token_count=25
        # Token 30 appears early in the query; teacher forcing must continue afterward.
        with patch.object(self.model.generation_config,'eos_token_id',30):
            r=e.generate(self.prompt,1)
        self.assertEqual(len(r.query_cache_lengths),16)
        self.assertEqual(len(r.generated_token_ids),1)

    def test_reserved_query_cannot_evict_recent_record_before_answer(self):
        spans=[(2,24,'irrelevant filler'),(24,28,'Record ITEM-123456 port zone'),(28,30,'tail')]
        outputs=[]
        for reserve in [False,True]:
            e=self.engine(sink_tokens=2,recent_window=6,chunk_size=4,
                          selection_policy='entity_balanced',reserve_query_space=reserve)
            e.selection_query='Return ITEM-123456 port zone'
            e.document_token_count=30
            with patch('src.ogc_cache.record_spans',return_value=spans):
                outputs.append(e.generate(self.prompt,1))
        old,new=outputs
        self.assertFalse(set(range(24,28)) & set(old.retained_positions))
        self.assertTrue(set(range(24,28)).issubset(new.retained_positions))
        self.assertEqual(new.document_budget_tokens,6)
        self.assertEqual(new.reserved_query_tokens,6)
        self.assertEqual(new.query_cache_lengths,[min(6+i,12) for i in range(11)])
        self.assertEqual(new.retained_tokens,12)

    def test_invalid_split_and_matching_factory_budget(self):
        e=self.engine()
        for split in [0,-1,41]:
            e.document_token_count=split
            with self.assertRaises(ValueError):e.generate(self.prompt,1)
        engines=make_engines('Qwen/Qwen2.5-1.5B-Instruct',None,'cpu',torch.float32,[260,516,1028],['ogc','ogc_corrected','entity_balanced'],64,256,19)
        for name,engine in engines.items():
            budget=int(name.split('/B')[1].split('/')[0]);self.assertEqual(engine.budget_tokens,budget)
        self.assertEqual(engines['entity_balanced/B260/H128'].recent_window,128)


if __name__=='__main__':unittest.main()
