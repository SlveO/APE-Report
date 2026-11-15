"""Verify actual frozen source reconstruction, evidence, pairing and history."""
from collections import Counter
from itertools import product

from transformers import AutoTokenizer

from .common import read_json, write_json, now
from .data import json_rows, overlap_report, question_key
from .freeze import CHAT_END, CHAT_START
from .jobs import JobFactory
from .methods import chunks
from .metrics import normalize
from .rules import COUNTS, DATASETS, LENGTHS, PARAMS


def main():
    factory = JobFactory('datasets/frozen-main80')
    config = read_json('data_sources.json')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True, trust_remote_code=False)
    base = factory.tables['longbench']
    by_id = {r['sample_id']: r for r in base}
    for dataset in DATASETS:
        public = {r['_id']: r for r in json_rows(config['longbench'][dataset])}
        selected = [r for r in base if r['dataset'] == dataset]
        assert len(selected) == 200 and len({r['public_id'] for r in selected}) == 200
        assert {r['public_id'] for r in selected} == set(public)
        for row in selected:
            original = public[row['public_id']]
            assert row['document'] == original['context'] and row['question'] == original['input'] and row['answers'] == original['answers']
            assert row['source_ids'] and row['support_documents']
    overlap = overlap_report(base)
    tokenized = {}
    offsets = {}
    for row in base:
        encoded = tokenizer(row['document'], add_special_tokens=False, return_offsets_mapping=True)
        tokenized[row['sample_id']], offsets[row['sample_id']] = encoded['input_ids'], encoded['offset_mapping']
        assert len(encoded['input_ids']) == row['_length']
    verified = 0
    for row in factory.tables['development']:
        original = by_id[row['sample_id']]
        assert row['split'] in ('explore', 'validation')
        source = tokenized[row['sample_id']]
        retained = [i for a, b in row['retained_source_intervals'] for i in range(a, b)]
        assert retained == sorted(set(retained))
        keep = set(retained)
        evidence = {i for i, (a, b) in enumerate(offsets[row['sample_id']]) if a < b and any(a < y and b > x for x, y in original['evidence_char_spans'])}
        assert evidence and evidence <= keep
        reconstructed = [source[i] for i in retained]
        for supplement in row['supplement_sources']:
            donor = by_id[supplement['sample_id']]
            assert donor['split'] in ('explore', 'validation') and donor['group'] != original['group']
            assert supplement['review']['contains_additional_answer_evidence'] is False and supplement['review']['basis']
            for a, b in supplement['source_token_intervals']:
                fragment = tokenized[donor['sample_id']][a:b]
                assert 0 <= a < b <= len(tokenized[donor['sample_id']])
                text = ' ' + normalize(tokenizer.decode(fragment)) + ' '
                assert not any(normalize(answer) and ' ' + normalize(answer) + ' ' in text for answer in row['answers'])
                reconstructed.extend(fragment)
        assert reconstructed == row['document_tokens'] and len(reconstructed) == row['L']
        for n in COUNTS:
            parts = chunks(reconstructed, count=n)
            assert len(parts) == n and sum(parts, []) == reconstructed
        verified += 1
    coverage = read_json('datasets/frozen-main80/nested-coverage.json')
    for split, levels in coverage.items():
        for n, check in levels.items():
            assert check['valid'] and not check['missing_dataset_length_cells']
    # Every applicable exploration input retains all 16 T/S, all three block counts.
    for size in (50, 35, 20, 10, 6):
        limits = dict(factory.initial_limits(), explore=size)
        jobs = factory.make('explore', limits)
        for d, length, n, param in product(DATASETS, LENGTHS, COUNTS, PARAMS):
            assert any(factory.tables[j['table']][j['index']]['dataset'] == d and j['key']['L'] == length and j['N'] == n and tuple(j['param']) == param for j in jobs)
    litm = factory.tables['litm']
    assert len(litm) == 1500
    originals = {p: {question_key(r['question']): r for r in json_rows(path)} for p, path in config['litm'].items()}
    prompt = __import__('pathlib').Path(config['litm_prompt']).read_text().rstrip('\n')
    for sid in factory.orders['litm']:
        versions = [r for r in litm if r['sample_id'] == sid]
        assert sorted(r['position'] for r in versions) == [0, 9, 19]
        for row in versions:
            raw = originals[str(row['position'])][question_key(row['question'])]
            docs = '\n'.join(f'Document [{i+1}](Title: {d["title"]}) {d["text"]}' for i, d in enumerate(raw['ctxs']))
            assert row['document'] == docs and row['answers'] == raw['answers']
            assert row['prefix'] + docs + row['suffix'] == CHAT_START + prompt.format(question=row['question'], search_results=docs) + CHAT_END
    raw_dial = json_rows(config['doc2dial_dialogues'])
    originals = {d['dial_id']: d for documents in raw_dial['dial_data'].values() for conversations in documents.values() for d in conversations}
    assert len(factory.tables['doc2dial']) == 500
    for row in factory.tables['doc2dial']:
        conversation = originals[row['source']['dial_id']]
        previous = [t for t in conversation['turns'] if t['turn_id'] < row['source']['target_turn']]
        assert row['history'] == [{k: t[k] for k in ('turn_id', 'role', 'utterance')} for t in previous]
        assert row['history_turn_ids'] == [t['turn_id'] for t in previous]
        assert all(t['utterance'].replace('\n', '\t').replace('\r', '\t').strip() in row['suffix'] for t in previous)
    applicability = read_json('datasets/frozen-main80/applicability.json')
    report = {'role': 'PLANNER', 'status': 'PASS', 'at': now(), 'longbench_ids': dict(Counter(r['dataset'] for r in base)),
        'splits': {s: len(factory.orders[s]) for s in ('explore', 'validation', 'test')},
        'verified_constructed_inputs': verified, 'development_grid_counts': dict(Counter(f'{r["split"]}/{r["dataset"]}/{r["L"]}' for r in factory.tables['development'] if r['L'] in LENGTHS)),
        'inapplicable': [r for r in applicability if not r['applicable']],
        'nested_144_configuration_coverage': True, 'paired_litm_questions': 500, 'complete_history_doc2dial': 500,
        'group_and_single_document_overlap': overlap,
        'evidence_mapping_policy_counts': dict(Counter(p for r in base for p in r['evidence_mapping_policy'])),
        'raw_public_content_unchanged': True, 'supplement_sources_development_only': True,
        'no_model_generation_or_formal_timing': True}
    write_json('logs/runs/20261113-main80-prep/data-validation.json', report)
    print('PASS: 600 source IDs, every constructed token/evidence interval, all allowed nested144 levels, LITM500 triples, Doc2Dial500 full histories')


if __name__ == '__main__':
    main()
