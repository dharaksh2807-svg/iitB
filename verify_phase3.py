"""Validate new Phase 3 artifacts without editing any raw measurements."""
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from src.research import score
from run_research_benchmark import make_engines


def verify(path):
    p=Path(path);m=json.loads((p/'manifest.json').read_text());assert m['status']=='completed'
    cfg=m['config'];data=json.loads((p/'dataset.json').read_text());items={x['item_id']:x for x in data}
    assert len(items)==len(data)
    assert hashlib.sha256((p/'dataset.json').read_bytes()).hexdigest()==m['dataset_sha256']
    rows=[json.loads(x) for x in (p/'outputs.jsonl').read_text().splitlines()]
    engines=make_engines(cfg['model'],cfg['revision'],'cpu',getattr(torch,cfg['dtype']),cfg['budgets'],cfg['policies'],cfg['chunk_size'],cfg['prefill_chunk_size'],cfg['seed'],cfg['modified_history_fraction'])
    assert len(rows)==len(data)*len(engines)==m['outputs']
    assert [r['execution_index'] for r in rows]==list(range(len(rows)))
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
    bpt=2*28*2*128*{'float32':4,'float16':2,'bfloat16':2}[cfg['dtype']]
    by_case=defaultdict(list)
    for r in rows:
        item=items[r['item_id']];n=len(tokenizer.encode(item['document']));doc=item.get('document_token_count',n) if cfg['protocol']=='held_query' else n
        assert n==r['actual_input_tokens']==item['actual_token_length']
        e=engines[r['config_name']];steps=r['decode_steps'];b=getattr(e,'cache_budget',getattr(e,'budget_tokens',n+steps))
        assert r['evaluation_protocol']==cfg['protocol']
        assert r['document_tokens']==doc and r['query_tokens']==n-doc
        doc_budget=r.get('document_budget_tokens')
        doc_budget=b if doc_budget is None else doc_budget
        reservation=min(n-doc,e.recent_window) if getattr(e,'reserve_query_space',False) else 0
        assert r.get('reserved_query_tokens',0)==reservation
        assert doc_budget==b-reservation
        initial=min(doc,doc_budget)
        assert r['query_cache_lengths']==[min(initial+i,b) for i in range(n-doc+1)]
        assert len(r['document_retained_positions'])==initial
        assert max(r['document_retained_positions'],default=-1)<doc
        if 'document_kv_bytes' in r:assert r['document_kv_bytes']==initial*bpt
        if r.get('reserved_query_tokens',0):assert doc_budget+r['reserved_query_tokens']==b
        assert r['cache_lengths']==[min(n+i,b) for i in range(steps+1)]
        pos=r['retained_positions'];assert pos==sorted(set(pos)) and pos[-1]==n+steps-1
        assert len(pos)==min(n+steps,b)==r['actual_cached_tokens']
        assert r['compressed_kv_bytes']==r['cache_storage_bytes']==len(pos)*bpt
        assert r['prefill_kv_bytes']==doc*bpt
        assert r['uncompressed_counterfactual_kv_bytes']==(n+steps)*bpt
        assert r['generated_tokens']==len(r['generated_token_ids'])==steps+1
        assert tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True)==r['prediction']
        assert all(r[k]==v for k,v in score(r['prediction'],item['ground_truth_answers']).items())
        assert r['total_time_ms']>=r['ttft_ms']+r['decode_time_ms']
        assert r['ttft_ms']>=r['prefill_time_ms']+r['initial_selection_time_ms']+r['query_time_ms']
        assert abs(r['selection_time_ms']-r['initial_selection_time_ms']-r['query_pruning_time_ms']-r['decode_pruning_time_ms'])<1e-6
        by_case[r['item_id']].append(r)
    for subset in by_case.values():
        assert {r['config_name'] for r in subset}==set(engines)
        if cfg['protocol']=='continuation':assert len({r['generated_token_ids'][0] for r in subset})==1
    result=dict(passed=True,cases=len(data),outputs=len(rows),protocol=cfg['protocol'],bytes_per_token=bpt,
                checks='pairing, raw output decoding/scoring, document/query/decode budgets, original positions, physical bytes, timing decomposition')
    (p/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    for path in sys.argv[1:]: verify(path)
