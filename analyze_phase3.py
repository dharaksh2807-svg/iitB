"""Produce new before/after tables, paired contrasts and measured-KV plots."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.research import paired_interval


def analyze(study, output):
    p=Path(study);m=json.loads((p/'manifest.json').read_text())
    assert m['status']=='completed' and json.loads((p/'verification.json').read_text())['passed']
    rows=[json.loads(line) for line in (p/'outputs.jsonl').read_text().splitlines()]
    groups=defaultdict(list)
    for row in rows: groups[row['config_name']].append(row)
    table=[];contrasts={};plot_points=[]
    for name,rs in sorted(groups.items()):
        value=dict(config=name,n=len(rs),correct=sum(r['is_correct'] for r in rs),
                   recall_pct=100*np.mean([r['is_correct'] for r in rs]),
                   canonical_em_pct=100*np.mean([r['exact_match'] for r in rs]),
                   mean_kv_bytes=float(np.mean([r['compressed_kv_bytes'] for r in rs])),
                   mean_prefill_ms=float(np.mean([r['prefill_time_ms'] for r in rs])),
                   mean_query_ms=float(np.mean([r['query_time_ms'] for r in rs])),
                   mean_selection_ms=float(np.mean([r['selection_time_ms'] for r in rs])),
                   mean_ttft_ms=float(np.mean([r['ttft_ms'] for r in rs])),
                   mean_decode_ms=float(np.mean([r['decode_time_ms'] for r in rs])),
                   mean_wall_ms=float(np.mean([r['total_time_ms'] for r in rs])))
        value['by_context']={str(k):100*np.mean([r['is_correct'] for r in rs if r['context_length']==k]) for k in sorted({r['context_length'] for r in rs})}
        value['by_depth']={k:100*np.mean([r['is_correct'] for r in rs if r['depth_category']==k]) for k in sorted({r['depth_category'] for r in rs})}
        value['by_type']={k:100*np.mean([r['is_correct'] for r in rs if r['query_type']==k]) for k in sorted({r['query_type'] for r in rs})}
        if name.startswith(('ogc/','ogc_corrected/','ogc_c32/','ogc_history/','entity_balanced/','entity_reserved/')):
            budget=int(name.split('/B')[1].split('/')[0])
            history=int(name.split('/H')[1]) if '/H' in name else (budget-4)//4
            eligible=[r for r in rs if r['document_tokens']>budget]
            value['sink_occupancy_eligible_cases']=len(eligible)
            value['mean_sink_adjacent_history_tokens']=(float(np.mean([sum(4<=i<68 for i in r['document_retained_positions'][4:4+history]) for r in eligible])) if eligible else None)
        table.append(value)
        plot_points.append(dict(config=name,kv_mib=value['mean_kv_bytes']/1024**2,recall_pct=value['recall_pct']))
        if name!='full':
            budget=int(name.split('/B')[1].split('/')[0]);old=f'ogc/B{budget}/C64'
            if old in groups:
                contrasts[name]=dict(all_facts=paired_interval(rs,groups[old]),canonical_em=paired_interval([dict(r,is_correct=r['exact_match']) for r in rs],[dict(r,is_correct=r['exact_match']) for r in groups[old]]))
    promotion={}
    if len(groups.get('full',[]))>=60 and m['config']['protocol']=='held_query':
        for entry in table:
            if not entry['config'].startswith(('entity_balanced/','entity_reserved/')):continue
            b=int(entry['config'].split('/B')[1].split('/')[0]);base=next(x for x in table if x['config']==f'ogc/B{b}/C64')
            ci=contrasts[entry['config']]['all_facts']['paired_bootstrap_95_ci_pp']
            promotion[entry['config']]=dict(paired_lower_bound_positive=ci is not None and ci[0]>0,
                 nonnegative_each_context=all(v>=base['by_context'][k] for k,v in entry['by_context'].items()),
                 nonnegative_each_type=all(v>=base['by_type'][k] for k,v in entry['by_type'].items()),
                 canonical_em_not_lower=bool(entry['canonical_em_pct']>=base['canonical_em_pct']))
            if entry['config'].startswith('entity_reserved/'):
                promotion[entry['config']]['nonnegative_each_depth']=all(v>=base['by_depth'][k] for k,v in entry['by_depth'].items())
            promotion[entry['config']]['passed']=all(promotion[entry['config']].values())
    reservation_contrasts={}
    for name,rs in groups.items():
        if not name.startswith('entity_reserved/'):continue
        peer=name.replace('entity_reserved/','entity_balanced/',1)
        if peer in groups:
            reservation_contrasts[name]=dict(all_facts=paired_interval(rs,groups[peer]),canonical_em=paired_interval([dict(r,is_correct=r['exact_match']) for r in rs],[dict(r,is_correct=r['exact_match']) for r in groups[peer]]))
    result=dict(study=str(p),protocol=m['config']['protocol'],table=table,contrasts_vs_original_ogc=contrasts,
                contrasts_vs_unreserved_records=reservation_contrasts,
                scoped_improvement_rule=promotion,plot_points=plot_points,
                caveat='Exploratory synthetic recall. No multiplicity adjustment. Singleton intervals withheld; plots show measured retained KV, not peak memory.')
    (output/(p.name+'.json')).write_text(json.dumps(result,indent=2))
    text=f"# {p.name}: {m['config']['protocol']}\n\n|Configuration|n|All-facts %|Canonical EM %|KV MiB|Prefill ms|Query ms|TTFT ms|Wall ms|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for v in table:
        text+=f"|{v['config']}|{v['n']}|{v['recall_pct']:.1f}|{v['canonical_em_pct']:.1f}|{v['mean_kv_bytes']/1024**2:.4f}|{v['mean_prefill_ms']:.1f}|{v['mean_query_ms']:.1f}|{v['mean_ttft_ms']:.1f}|{v['mean_wall_ms']:.1f}|\n"
    text+='\nQuery/selection/decode durations overlap component totals; do not add selection to wall time.\n'
    (output/(p.name+'.md')).write_text(text)
    fig,ax=plt.subplots(figsize=(10,6))
    families=defaultdict(list)
    for point in plot_points:families[point['config'].split('/')[0]].append(point)
    for family,points in sorted(families.items()):
        points.sort(key=lambda x:x['kv_mib'])
        ax.plot([x['kv_mib'] for x in points],[x['recall_pct'] for x in points],marker='o',label=family)
    ax.set_xlabel('Mean measured retained KV tensor memory (MiB)')
    ax.set_ylabel('All-facts recall (%)');ax.set_ylim(-3,103);ax.grid(alpha=.25)
    ax.set_title(f"{m['config']['protocol']}: {m['cases']} questions\n"+('Pilot: one case per stratum; no confidence intervals' if any(c['all_facts']['paired_bootstrap_95_ci_pp'] is None for c in contrasts.values()) else 'Paired uncertainty and canonical EM reported in accompanying tables'))
    ax.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.15))
    fig.tight_layout();fig.savefig(output/(p.name+'-recall-memory.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    print(p.name,len(rows),'outputs',promotion,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--studies',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for study in args.studies: analyze(study,out)
