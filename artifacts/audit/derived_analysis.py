"""Reproduce exploratory contrasts into a NEW directory; never edit raw outputs."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.research import paired_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def load(name):
        p = args.artifact_root / name
        assert json.loads((p / 'manifest.json').read_text())['status'] == 'completed'
        return [json.loads(line) for line in (p / 'outputs.jsonl').read_text().splitlines()]
    studies={64:load('research-1k-105'),32:load('research-chunks32-105'),16:load('research-chunks16-105')}
    def contrast(a,b,metric='is_correct'):
        return paired_interval([dict(r,is_correct=r[metric]) for r in a],
                               [dict(r,is_correct=r[metric]) for r in b])
    def select(rows,name):
        return [r for r in rows if r['config_name']==name]
    main_rows=studies[64]
    extra={}
    for left,right in [('entity_chunks/B260','ogc/B260/C64'),('adaptive_chunks/B260','ogc/B260/C64'),('entity_chunks/B260','adaptive_chunks/B260')]:
        extra[left+' minus '+right]=contrast(select(main_rows,left),select(main_rows,right))
    em={}
    for left,right in [('ogc/B260/C64','random_chunks/B260/C64'),('entity_chunks/B260','ogc/B260/C64'),('entity_chunks/B260','full')]:
        em[left+' minus '+right]=contrast(select(main_rows,left),select(main_rows,right),'exact_match')
    chunks={};repeat={}
    baseline={r['item_id']:r['generated_token_ids'] for r in select(main_rows,'full')}
    for c in (32,16):
        other={r['item_id']:r['generated_token_ids'] for r in select(studies[c],'full')}
        assert set(other)==set(baseline)
        mismatches=[key for key in baseline if baseline[key]!=other[key]]
        repeat[str(c)]=dict(questions=len(baseline),identical=len(baseline)-len(mismatches),differing_item_ids=mismatches)
        for policy in ('ogc','random_chunks'):
            a=select(studies[c],f'{policy}/B260/C{c}');b=select(main_rows,f'{policy}/B260/C64')
            chunks[f'{policy} C{c} minus C64']=dict(all_facts=contrast(a,b),canonical_em=contrast(a,b,'exact_match'))
    structure={}
    for name in ('ogc/B260/C64','random_chunks/B260/C64','query_tokens/B260','adaptive_chunks/B260','entity_chunks/B260'):
        subset=select(main_rows,name);lengths=Counter();run_counts=[]
        for row in subset:
            for a,b in row['selected_chunk_ranges']:
                lengths[b-a]+=1
            history=row['retained_positions'][4:68]
            run_counts.append(sum(i==0 or p!=history[i-1]+1 for i,p in enumerate(history)))
        structure[name]=dict(mean_contiguous_history_runs=sum(run_counts)/len(run_counts),
                             selected_span_length_histogram=dict(sorted(lengths.items())),
                             note='Actual selected span lengths, including partial fills; not inferred adaptive C or semantic entity quality')
    for filename,value in [('additional-comparisons.json',extra),('canonical-em-comparisons.json',em),
                           ('chunk-size-comparisons.json',chunks),('cross-run-baseline-agreement.json',repeat),
                           ('retention-structure.json',structure)]:
        (args.output / filename).write_text(json.dumps(value,indent=2))
    print(f'Wrote reproducible derived analyses to {args.output}')


if __name__=='__main__':
    main()
