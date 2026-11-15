"""Resolve public origins, freeze split/order and construct development inputs.

Run only after asset acquisition. Emits unresolved mappings instead of freezing them.
"""
import argparse
import random

from transformers import AutoTokenizer

from .common import ROOT, inside, read_json, write_json
from .data import check_litm, construct, json_rows, map_longbench, source_index, split_groups
from .rules import DATASETS, LENGTHS

CHAT_START = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
CHAT_END = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'


def longbench_prompt(row, templates):
    prefix, suffix = templates[row['dataset']].split('{context}')
    return {**row, 'prefix': CHAT_START + prefix, 'suffix': suffix.format(input=row['question']) + CHAT_END,
            'max_new_tokens': 128 if row['dataset'] == 'qasper' else 32,
            'scorer': 'LongBench_v1_qa_f1'}


def litm_rows(raw, prompt):
    order, indexed = check_litm(raw)
    rows = []
    for i, key in enumerate(order):
        for pos in (0, 9, 19):
            r = indexed[pos][key]
            before, after = prompt.rstrip('\n').split('{search_results}')
            rows.append({'sample_id': f'litm:{i:04d}', 'dataset': 'litm', 'question': r['question'], 'answers': r['answers'],
                'position': pos, 'split': 'test', 'group': ['litm-question', r['question']],
                'document': '\n'.join(f'Document [{j+1}](Title: {d["title"]}) {d["text"]}' for j, d in enumerate(r['ctxs'])),
                'prefix': CHAT_START + before.format(question=r['question']),
                'suffix': after.format(question=r['question']) + CHAT_END,
                'max_new_tokens': 128, 'original_document_count': 20, 'scorer': 'LITM_best_subspan_em',
                'public_document_ids': [d.get('id') for d in r['ctxs']]})
    return rows, [f'litm:{i:04d}' for i in range(len(order))]


def doc2dial_rows(dial, documents):
    def text2line(text):
        return text.replace('\n', '\t').replace('\r', '\t').strip()
    def tag(role, text):
        return f'<{role}>\t{text2line(text)}'
    rows = []
    for domain, docs in dial['dial_data'].items():
        for doc_id, conversations in docs.items():
            doc = documents['doc_data'][domain][doc_id]
            for conversation in conversations:
                history = []
                for turn in conversation['turns']:
                    if turn['role'] == 'agent' and turn['references'] and history:
                        sample_id = f'{conversation["dial_id"]}_{turn["turn_id"] - 1}'
                        suffix = '\t'.join([tag('last_turn', history[-1]['utterance'])] + [tag(t['role'], t['utterance']) for t in reversed(history)])
                        rows.append({'sample_id': sample_id, 'dataset': 'doc2dial', 'split': 'test',
                            'group': ['doc2dial', domain, doc_id], 'source': {'domain': domain, 'doc_id': doc_id, 'dial_id': conversation['dial_id'], 'target_turn': turn['turn_id']},
                            'answers': [text2line(turn['utterance'])], 'history': [dict(t) for t in history], 'history_turn_ids': [t['turn_id'] for t in history],
                            'prefix': CHAT_START + tag('title', doc_id) + '\t<doc_context>\t',
                            'document': text2line(doc['doc_text']), 'suffix': '\t' + suffix + CHAT_END,
                            'max_new_tokens': 128, 'original_document_count': 1, 'scorer': 'DialDoc21_sacrebleu',
                            'template_adaptation': 'official tags/text2line/reverse history; all earlier turns retained; full document before request history for reusable document-only KV'})
                    # Unlike the historical reference script, do not omit ungrounded history.
                    history.append({'turn_id': turn['turn_id'], 'role': turn['role'], 'utterance': turn['utterance']})
    rows.sort(key=lambda r: r['sample_id'])
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate Doc2Dial public request IDs')
    random.Random(42).shuffle(rows)
    if len(rows) < 500:
        raise ValueError('Doc2Dial has fewer than 500 grounded target turns')
    rows = rows[:500]
    return rows, [r['sample_id'] for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', default='data_sources.json')
    parser.add_argument('--output', default='datasets/frozen-main80')
    args = parser.parse_args()
    config = read_json(args.sources)
    output = inside(args.output)
    if (output / 'frozen.json').exists():
        raise ValueError('frozen manifest exists; use a distinct output for a new dataset version')
    qasper = {}
    for path in config['qasper_origins']:
        qasper.update(json_rows(path))
    index = source_index(qasper, json_rows(config['hotpotqa_origin']), json_rows(config['musique_origin']))
    raw = {d: json_rows(config['longbench'][d]) for d in DATASETS}
    rows, errors = map_longbench(raw, index)
    write_json(output / 'mapping-errors.json', errors)
    if errors:
        raise ValueError(f'{len(errors)} unresolved original mappings; data not frozen')
    order, report = split_groups(rows)
    templates = read_json(config['longbench_templates'])
    rows = [longbench_prompt(r, templates) for r in rows]
    tokenizer = AutoTokenizer.from_pretrained(str(inside(config['model'])), local_files_only=True, trust_remote_code=False)
    reviews = read_json(config['supplement_reviews']) if config.get('supplement_reviews') else {}
    dev = [r for r in rows if r['split'] != 'test']
    by_id = {r['sample_id']: r for r in rows}
    donors = [by_id[i] for split in ('explore', 'validation') for i in order[split]]
    constructions, applicability = [], []
    donor_cache = {}
    for row in dev:
        for length in (*LENGTHS, 6144, 12288):
            result = construct(row, length, tokenizer, donors, reviews, donor_cache)
            applicability.append({'sample_id': row['sample_id'], 'dataset': row['dataset'], 'split': row['split'],
                                  **{k: v for k, v in result.items() if k != 'document_tokens'}})
            if result['applicable']:
                constructions.append({**row, **result})
    for split in ('explore', 'validation'):
        for d in DATASETS:
            for length in LENGTHS:
                if not any(r['split'] == split and r['dataset'] == d and r['L'] == length for r in constructions):
                    write_json(output / 'applicability.json', applicability)
                    raise ValueError(f'no verified applicable development sample for {split}/{d}/{length}')
    original_orders = {k: list(v) for k, v in order.items()}
    # Freeze nested orders by data applicability only, before any model output.
    for split in ('explore', 'validation'):
        queues = {}
        for d in DATASETS:
            ids = [sid for sid in order[split] if by_id[sid]['dataset'] == d]
            complete = {sid for sid in ids if all(any(c['sample_id'] == sid and c['L'] == l for c in constructions) for l in LENGTHS)}
            queues[d] = [sid for sid in ids if sid in complete] + [sid for sid in ids if sid not in complete]
        order[split] = []
        while any(queues.values()):
            for d in DATASETS:
                if queues[d]:
                    order[split].append(queues[d].pop(0))
    nested = {}
    for split, levels in (('explore', (len(order['explore']), 35, 20, 10, 6)), ('validation', (len(order['validation']), 30, 20, 10))):
        nested[split] = {}
        for size in dict.fromkeys(levels):
            chosen = order[split][:size]
            missing = [[d, l] for d in DATASETS for l in LENGTHS if not any(c['sample_id'] in chosen and c['dataset'] == d and c['L'] == l for c in constructions)]
            nested[split][str(size)] = {'sample_ids': chosen, 'missing_dataset_length_cells': missing, 'valid': not missing}
    write_json(output / 'nested-coverage.json', nested)
    if any(not check['valid'] for levels in nested.values() for check in levels.values()):
        raise ValueError('an authorized nested subset loses dataset/144-configuration coverage')
    litm, litm_order = litm_rows({int(p): json_rows(path) for p, path in config['litm'].items()}, inside(config['litm_prompt']).read_text())
    doc, doc_order = doc2dial_rows(json_rows(config['doc2dial_dialogues']), json_rows(config['doc2dial_documents']))
    for row in [*rows, *litm, *doc]:
        row['_length'] = len(tokenizer.encode(row['document'], add_special_tokens=False))
    write_json(output / 'longbench.json', rows)
    write_json(output / 'development.json', constructions)
    write_json(output / 'litm.json', litm)
    write_json(output / 'doc2dial.json', doc)
    write_json(output / 'applicability.json', applicability)
    write_json(output / 'split-report.json', report)
    write_json(output / 'frozen.json', {'protocol': 'ape-v3-data', 'sources': config, 'counts': {'longbench': 600, 'litm': 1500, 'doc2dial': 500},
        'orders': {**order, 'litm': litm_order, 'doc2dial': doc_order}, 'original_group_orders': original_orders,
        'nested_order_policy': 'Within each dataset: all-three-length applicable first, then others, preserving seeded group order in each class; interleave datasets. Frozen before model execution.',
        'files': ['longbench.json', 'development.json', 'litm.json', 'doc2dial.json', 'applicability.json', 'split-report.json', 'nested-coverage.json'],
        'construction_policy': 'verified evidence only; absent negative-evidence supplement review becomes inapplicable, never score-based replacement'})
    print(f'frozen {len(rows)} LongBench IDs, {len(constructions)} development inputs, {len(litm)} position inputs, {len(doc)} application requests')


if __name__ == '__main__':
    main()
