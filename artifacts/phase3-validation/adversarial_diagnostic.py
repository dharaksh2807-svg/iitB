"""Post-hoc retention diagnosis; gold spans are used only after frozen inference.

Run from the repository root with PYTHONPATH=. and a fresh --output path.
This reconstructs static-policy cache positions; it does not rerun inference.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from src.generation import DEFAULT_MODEL, DEFAULT_REVISION, prefix_recent_indices
from src.research import score


def diagnose(study):
    data = {x['item_id']: x for x in json.loads((study / 'dataset.json').read_text())}
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION,
                                               local_files_only=True)
    rows = [json.loads(s) for s in (study / 'outputs.jsonl').read_text().splitlines()]
    records = []
    for row in rows:
        if row['query_type'] != 'adversarial':
            continue
        name = row['config_name']
        assert name == 'full' or name in {'ogc/B260/C64', 'entity_balanced/B260/H128',
                                         'entity_reserved/B260/H128'}
        text = data[row['item_id']]['document']
        offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
        facts = {}
        for kind, pattern in [('current', r'CURRENT record for project[^\n]+'),
                              ('obsolete', r'ARCHIVED obsolete record for project[^\n]+')]:
            match = re.search(pattern, text)
            assert match
            facts[kind] = {i for i, (a, b) in enumerate(offsets)
                           if b > match.start() and a < match.end()}
        positions = row['document_retained_positions'].copy()
        history = 64 if name.startswith('ogc/') else 128
        for absolute in range(row['document_tokens'], row['actual_input_tokens']):
            positions.append(absolute)
            if name != 'full':
                positions = [positions[i] for i in prefix_recent_indices(
                    len(positions), 4 + history, 260 - 4 - history)]
        assert len(positions) == row['query_cache_lengths'][-1]
        obsolete = re.search(r'ARCHIVED obsolete record[^\n]+port is (\d+)', text).group(1)
        records.append(dict(
            item_id=row['item_id'], config=name, is_correct=row['is_correct'],
            prediction=row['prediction'],
            obsolete_in_prediction=score(row['prediction'], [obsolete])['is_correct'],
            initial_complete_current=facts['current'].issubset(row['document_retained_positions']),
            preanswer_complete_current=facts['current'].issubset(positions),
            preanswer_complete_obsolete=facts['obsolete'].issubset(positions),
            final_complete_current=facts['current'].issubset(row['retained_positions'])))
    groups = defaultdict(list)
    for row in records:
        groups[row['config']].append(row)
    fields = ['is_correct', 'obsolete_in_prediction', 'initial_complete_current',
              'preanswer_complete_current', 'preanswer_complete_obsolete', 'final_complete_current']
    summary = {name: dict(n=len(rs), **{f: sum(x[f] for x in rs) for f in fields},
                         incorrect_with_complete_current=sum(
                             not x['is_correct'] and x['preanswer_complete_current'] for x in rs))
               for name, rs in groups.items()}
    return dict(note='Diagnostic only; uses gold record spans after frozen inference. Preanswer positions reconstructed from recorded document retention and static prefix/recent pruning; no extra model run.',
                summary=summary, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.study)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result['summary'], indent=2))
