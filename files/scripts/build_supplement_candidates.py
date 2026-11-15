"""Propose conservative development-only donor paragraphs for PLANNER review.

This script cannot authorize a negative-evidence review or freeze data itself.
"""
import bisect
import re
from collections import defaultdict

from transformers import AutoTokenizer
from main80.common import read_json, write_json
from main80.data import json_rows, map_longbench, source_index, split_groups
from main80.metrics import normalize
from main80.rules import DATASETS

STOP = set('a an the and or to of in on at for with from by as is are was were be been being do does did have has had can could will would should what which who whom where when why how many much their they it its this that these those also than then based used use using proposed given all any both each other only study studies system systems approach approaches method methods'.split())
ACADEMIC = set('algorithm algorithms model models modeling dataset datasets data corpus corpora neural network networks experiment experiments research paper papers training trained learning embedding embeddings vector vectors subspace regression classification recognition entailment translation evaluation evaluate evaluated accuracy metric metrics performance linguistic linguistics language languages'.split())


def main():
    config = read_json('data_sources.json')
    qasper = {}
    for path in config['qasper_origins']:
        qasper.update(json_rows(path))
    index = source_index(qasper, json_rows(config['hotpotqa_origin']), json_rows(config['musique_origin']))
    rows, errors = map_longbench({d: json_rows(config['longbench'][d]) for d in DATASETS}, index)
    assert not errors
    orders, report = split_groups(rows)
    by_id = {r['sample_id']: r for r in rows}
    dev = [by_id[i] for split in ('explore', 'validation') for i in orders[split]]
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True, trust_remote_code=False)
    segments, tokens = {}, {}
    for donor in dev:
        encoded = tokenizer(donor['document'], add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        tokens[donor['sample_id']] = ids
        starts, ends = [a for a, _ in offsets], [b for _, b in offsets]
        pieces = []
        for line in re.finditer(r'[^\n]+', donor['document']):
            if not 8 <= len(line.group().split()) <= 260:
                continue
            a, b = bisect.bisect_right(ends, line.start()), bisect.bisect_left(starts, line.end())
            if b <= a:
                continue
            text = tokenizer.decode(ids[a:b])
            pieces.append({'token_interval': [a, b], 'char_interval': [line.start(), line.end()], 'text': text})
        segments[donor['sample_id']] = pieces
    catalog, proposals = {}, []
    for target in dev:
        if not target['evidence_char_spans']:
            continue
        forbidden = set(normalize(target['question']).split()) - STOP
        forbidden.update(word for answer in target['answers'] for word in normalize(answer).split() if len(word) >= 3 and word not in STOP)
        if target['dataset'] == 'qasper':
            forbidden.update(ACADEMIC)
        forbidden = {w for w in forbidden if len(w) >= 3}
        aliases = [normalize(a) for a in target['answers'] if normalize(a)]
        need = max(0, 16384 - len(tokens[target['sample_id']]))
        picked, seen_text, available = [], set(), 0
        for donor in dev:
            # A scientific-paper target receives non-academic encyclopedic background;
            # named-entity Wikipedia QA receives unrelated scientific-paper background.
            if (target['dataset'] == 'qasper') == (donor['dataset'] == 'qasper'):
                continue
            if donor['group'] == target['group'] or set(donor['input_documents']) & set(target['input_documents']):
                continue
            previous_end = -1
            for segment in segments[donor['sample_id']]:
                a, b = segment['token_interval']
                normalized = normalize(segment['text'])
                words = set(normalized.split())
                if a < previous_end or normalized in seen_text or words & forbidden:
                    continue
                padded = ' ' + normalized + ' '
                if any(' ' + alias + ' ' in padded for alias in aliases):
                    continue
                cid = f'{donor["sample_id"]}:{a}:{b}'
                catalog[cid] = {**segment, 'source_sample': donor['sample_id'], 'source_dataset': donor['dataset'],
                    'source_documents': donor['input_documents'], 'source_group': donor['group'], 'source_split': donor['split']}
                picked.append(cid)
                previous_end = b
                seen_text.add(normalized)
                available += b - a
                if available >= need:
                    break
            if available >= need:
                break
        proposals.append({'target_sample': target['sample_id'], 'dataset': target['dataset'], 'split': target['split'],
            'question': target['question'], 'reference_answers': target['answers'], 'target_documents': target['input_documents'],
            'needed_supplement_tokens': need, 'available_proposed_tokens': available, 'forbidden_words': sorted(forbidden),
            'catalog_ids': picked, 'review_status': 'PENDING', 'note': 'Lexical exclusions alone do not certify absence of additional answering evidence; PLANNER semantic review required.'})
    write_json('logs/runs/20261113-main80-prep/supplement-proposals.json', {'proposals': proposals, 'catalog': catalog})
    print('proposals', len(proposals), 'unique paragraphs', len(catalog), 'complete proposed fills', sum(p['available_proposed_tokens'] >= p['needed_supplement_tokens'] for p in proposals))


if __name__ == '__main__':
    main()
