import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent


def compact_indexes(values):
    ranges = []
    for value in values:
        if ranges and value == ranges[-1][-1] + 1:
            ranges[-1].append(value)
        else:
            ranges.append([value])
    return ','.join(str(v[0]) if len(v) == 1 else f'{v[0]}-{v[-1]}' for v in ranges)


parser = argparse.ArgumentParser()
parser.add_argument('section', choices=['catalog', 'targets'])
parser.add_argument('start', type=int)
parser.add_argument('stop', type=int)
args = parser.parse_args()
started = datetime.now(timezone.utc).isoformat()
data = json.loads((BASE / 'input.snapshot.json').read_text())
catalog = list(data['catalog'].items())
index = {cid: i for i, (cid, _) in enumerate(catalog)}
for i in range(args.start, args.stop):
    if args.section == 'catalog':
        cid, paragraph = catalog[i]
        print(f'CATALOG {i} | {cid} | source split={paragraph["source_split"]}')
        print(paragraph['text'])
    else:
        target = data['proposals'][i]
        print(f'TARGET {i} | {target["target_sample"]} | {target["split"]}')
        print('QUESTION:', target['question'])
        print('REFERENCES:', json.dumps(target['reference_answers'], ensure_ascii=False))
        print('TARGET DOCUMENTS:', json.dumps(target['target_documents'], ensure_ascii=False))
        print('ORDERED CATALOG INDEXES:', compact_indexes([index[cid] for cid in target['catalog_ids']]))
        print()
record_path = BASE / 'execution.json'
record = json.loads(record_path.read_text())
record['commands'].append({
    'cmd': f'bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B {BASE.relative_to(Path.cwd())}/read_review_input.py {args.section} {args.start} {args.stop}',
    'started_at': started,
    'ended_at': datetime.now(timezone.utc).isoformat(),
    'exit_code': 0,
    'purpose': 'Emit complete input text for semantic review; emitted is not itself a semantic safety verdict.',
})
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
