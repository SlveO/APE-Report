"""Experiment 5: stream only MAIN80 diagnostic attention and saved answers on CPU.

Inputs under restored/ remain read-only. No answer generation, F1 re-scoring,
parameter selection, or downstream experiment analysis is performed.
"""
from __future__ import annotations

from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import csv
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'pc-analysis-output'
OUT = BASE / 'stages/05_attention'
RUN = ROOT / 'restored/output/runs/main80-20261114T041503'
FROZEN = RUN / 'snapshot/datasets/frozen-main80'
TOKENIZER = RUN / 'snapshot/models/Llama-3.1-8B-Instruct/tokenizer.json'
REFERENCE = '227116e4a004153272640cfe31087a033861b8e9'
HEADS, LAYERS = [0, 8, 16, 24], [0, 15, 31]
DATASETS = ['qasper', 'hotpotqa', 'musique']
REGIONS = ['prefix', 'evidence', 'background', 'request', 'decoded_history']

for name in ['tables', 'figures', 'reports', 'sources', 'cache', 'tmp', 'logs']:
    (OUT / name).mkdir(parents=True, exist_ok=True)
for key in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TEMP', 'TMP']:
    os.environ[key] = str(OUT / ('tmp' if key in ['TEMP', 'TMP'] else 'cache'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tokenizers import Tokenizer


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(key):
    return json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def stream(path):
    with path.open(encoding='utf-8') as handle:
        for line, text in enumerate(handle, 1):
            if text.strip():
                yield line, json.loads(text)


def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def csvwrite(name, rows, empty_columns=('issue', 'source_file', 'source_line')):
    columns = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_columns)
    with (OUT / 'tables' / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def intervals(indices):
    result = []
    for value in sorted(set(indices)):
        if result and result[-1][1] == value:
            result[-1][1] += 1
        else:
            result.append([value, value + 1])
    return result


def metadata(record):
    return {k: record[k] for k in ['dataset', 'sample_id', 'L', 'N', 'T', 'S']}


def savefig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def draw_case(pair, arrays, evidence):
    """Two common-scale views: all four query heads and actual decode steps."""
    ev = evidence[(pair['sample_id'], pair['L'])]
    labels = [('default', pair['_default']), ('cell_best', pair['_best'])]
    for mode in ['query_heads', 'decode_mean_heads']:
        fig, axes = plt.subplots(3, 2, figsize=(13, 8.5), layout='constrained')
        common_bins = max(len(v['mean_bins']) for _, record in labels for values in arrays[identity(record['key'])].values() for s, v in values.items() if mode == 'decode_mean_heads' or s == 0)
        common_steps = max(record['output_tokens'] for _, record in labels)
        for col, (label, record) in enumerate(labels):
            kid = identity(record['key'])
            p, length = record['prefix_tokens'], record['L']
            for row, layer in enumerate(LAYERS):
                ax = axes[row, col]
                saved = arrays[kid][layer]
                if mode == 'query_heads':
                    matrix = saved[0]['head_bins']
                    ys, ticks = HEADS, range(4)
                else:
                    matrix = np.full((common_steps, common_bins), np.nan)
                    for s in saved:
                        matrix[s, :len(saved[s]['mean_bins'])] = saved[s]['mean_bins']
                    ys, ticks = [f'{s}' + (' Q' if s == 0 else ' D') for s in range(common_steps)], range(common_steps)
                image = ax.imshow(np.ma.masked_less_equal(matrix, 0), aspect='auto', interpolation='nearest', cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=1))
                ax.set_yticks(list(ticks), ys, fontsize=8)
                ax.set_ylabel('头' if mode == 'query_heads' else '实际步骤（0=请求末token）')
                ax.set_title(f"层{layer} · {label} · T/S={record['T']}/{record['S']}", fontsize=10)
                ax.set_xlabel('物理缓存token索引（每箱64 token）')
                ax.axvline(p / 64 - .5, color='white', linewidth=.8, linestyle='--')
                ax.axvline((p + length) / 64 - .5, color='red', linewidth=.8, linestyle='--')
                for a, b in ev['corrected_document_intervals']:
                    ax.plot([(p + a) / 64 - .5, (p + b) / 64 - .5], [-.68, -.68], color='#00bf63', linewidth=3, clip_on=False)
                xticks = np.linspace(0, matrix.shape[1] - 1, 5)
                ax.set_xticks(xticks, [str(round((x + .5) * 64)) for x in xticks], fontsize=8)
        fig.colorbar(image, ax=axes.ravel().tolist(), label='每箱权重质量（对数色标）', shrink=.8)
        divergence = [s for s in range(min(pair['_default']['output_tokens'], pair['_best']['output_tokens'])) if pair['_default']['output_token_ids'][:s] != pair['_best']['output_token_ids'][:s]]
        history_note = f"共同步骤从step{min(divergence)}起历史不同" if divergence else '共同步骤的生成token历史相同'
        subtitle = f'绿色=完整证据；白线=前缀末；红线=文档末；{history_note}'
        if mode == 'decode_mean_heads':
            subtitle += '；空白=未执行步骤/无缓存token'
        fig.suptitle(f"{pair['dataset']} · L={pair['L']}, N={pair['N']} · {mode}\n{subtitle}", fontsize=10)
        savefig(fig, f"{mode}_{pair['pair_id']}")


def draw_overviews(query_pairs, pairs, evidence):
    for metric, ylabel, name in [
        ('evidence_fraction_context', '证据占文档权重/%', 'query_evidence_overview'),
        ('context_mass', '文档分支总权重/%', 'query_document_mass_overview'),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained', sharey=metric == 'context_mass')
        for pair in pairs:
            length = int(pair['L'])
            ax = axes[0 if length == 4096 else 1, DATASETS.index(pair['dataset'])]
            rows = sorted((r for r in query_pairs if str(r['pair_id']) == str(pair['pair_id'])), key=lambda r: int(r['layer']))
            values = []
            for prefix, label in [('default_', '默认APE'), ('best_', '探索格点最佳')]:
                yy = [float(r[prefix + metric]) * 100 for r in rows]
                values.extend(yy)
                ax.plot(range(3), yy, marker='o', label=label)
            if metric == 'evidence_fraction_context':
                reference = float(evidence[(pair['sample_id'], length)]['evidence_fraction_document']) * 100
                ax.axhline(reference, color='gray', linestyle='--', linewidth=.8, label='文档均匀权重参照')
                values.append(reference)
            ax.set_xticks(range(3), LAYERS)
            ax.set_xlabel('层（指定四头等权平均）')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{pair['dataset']} · {length//1024}K/{pair['N']}\n保存F1: {float(pair['default_f1_0_1'])*100:.2f} → {float(pair['best_f1_0_1'])*100:.2f}")
            ax.set_ylim(0, 100 if metric == 'context_mass' else max(values) * 1.3)
            ax.grid(axis='y', alpha=.2)
        axes[0, 0].legend(fontsize=8, loc='lower left' if metric == 'evidence_fraction_context' else 'upper right')
        title = '完整证据占文档分支权重的比例（各面板独立纵轴）' if metric == 'evidence_fraction_context' else '文档分支总权重（统一纵轴）'
        fig.suptitle('同输入请求末token：' + title, fontsize=12)
        savefig(fig, name)


def main():
    records = {}
    for line, record in stream(RUN / 'attempts.jsonl'):
        if record['key']['stage'] == 'attention':
            record['_line'] = line
            record['_role'] = 'default' if 'default' in record['logical_labels'] else 'cell_best'
            kid = identity(record['key'])
            assert kid not in records
            assert record['status'] == 'success' and record['attempt'] == 1
            assert record['method'] == 'ape_fixed' and record['split'] == 'explore'
            assert record['cache_mode'] == 'cold' and record['cache_hit'] is False
            assert math.isfinite(record['score']) and 0 <= record['score'] <= 1
            records[kid] = record
    assert len(records) == 12
    ids = {r['sample_id'] for r in records.values()}
    assert len(ids) == 3
    frozen, candidates, rule = (read(RUN / name) for name in ['snapshot/datasets/frozen-main80/frozen.json', 'candidates.json', 'rule.json'])
    all_development = read(FROZEN / 'development.json')
    dev = {(r['sample_id'], r['L']): r for r in all_development if r['sample_id'] in ids and r['L'] in (4096, 16384)}
    sources = {r['sample_id']: r for r in read(FROZEN / 'longbench.json') if r['sample_id'] in ids}
    assert len(dev) == 6
    for dataset in DATASETS:
        first = next(s for s in frozen['orders']['explore'] if s in sources and sources[s]['dataset'] == dataset)
        for length in [4096, 16384]:
            eligible = {r['sample_id'] for r in all_development if r['dataset'] == dataset and r['split'] == 'explore' and r['L'] == length}
            assert first == next(s for s in frozen['orders']['explore'] if s in eligible)

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    audit = {(r['sample_id'], int(r['L'])): r for r in csvread(BASE / 'stages/01_audit/tables/construction_checks.csv')}
    encoded, evidence, evidence_rows, evidence_examples = {}, {}, [], []
    for sid, source in sources.items():
        enc = tokenizer.encode(source['document'], add_special_tokens=False)
        required = {i for i, (a, b) in enumerate(enc.offsets) if b > a and any(a < d and b > c for c, d in source['evidence_char_spans'])}
        encoded[sid] = (enc, required)
    for (sid, length), row in dev.items():
        enc, required = encoded[sid]
        retained = [i for a, b in row['retained_source_intervals'] for i in range(a, b)]
        corrected = [j for j, old in enumerate(retained) if old in required]
        saved = {i for a, b in row['evidence_token_intervals'] for i in range(a, b)}
        expected = audit[(sid, length)]
        assert len(required) == int(expected['n_required_evidence_tokens'])
        assert len(required - set(retained)) == int(expected['n_missing_required_tokens']) == 0
        assert [enc.ids[i] for i in retained] == row['document_tokens'][:len(retained)]
        mask = np.zeros(length, dtype=bool)
        mask[corrected] = True
        ev = {'sample_id': sid, 'dataset': row['dataset'], 'L': length, 'evidence_tokens': len(corrected), 'background_tokens': length - len(corrected), 'evidence_fraction_document': len(corrected) / length, 'required_source_evidence_tokens': len(required), 'missing_required_tokens': 0, 'saved_evidence_tokens': len(saved), 'saved_mask_extra_tokens': len(saved - set(corrected)), 'saved_mask_omitted_tokens': len(set(corrected) - saved), 'corrected_document_intervals': intervals(corrected), 'saved_document_intervals': row['evidence_token_intervals'], 'source_character_intervals': source['evidence_char_spans'], 'retained_source_intervals': row['retained_source_intervals'], 'supplement_tokens': length - len(retained), 'unit': 'token counts; half-open intervals; fraction 0-1', 'source_file': 'snapshot/datasets/frozen-main80/longbench.json + development.json; snapshot tokenizer.json'}
        ev['evidence_mapping_policy'] = source['evidence_mapping_policy']
        evidence[(sid, length)] = {**ev, '_mask': mask}
        evidence_rows.append(ev)
        evidence_examples.append({**ev, 'question': source['question'], 'reference_answers': source['answers'], 'evidence_texts': [source['document'][a:b] for a, b in source['evidence_char_spans']], 'corrected_evidence_token_ids': [row['document_tokens'][i] for i in corrected]})
    csvwrite('evidence_regions.csv', evidence_rows)
    (OUT / 'sources/evidence_mapping_examples.json').write_text(json.dumps(evidence_examples, ensure_ascii=False, indent=2), encoding='utf-8')

    pair_groups = defaultdict(dict)
    for kid, record in records.items():
        pair_groups[(record['dataset'], record['sample_id'], record['L'], record['N'])][record['_role']] = record
    pairs, attempts = [], []
    generation_config = read(RUN / 'snapshot/models/Llama-3.1-8B-Instruct/generation_config.json')
    stop_ids = generation_config['eos_token_id']
    stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
    for index, key in enumerate(sorted(pair_groups, key=lambda k: (DATASETS.index(k[0]), k[2]))):
        a, b = (pair_groups[key][role] for role in ['default', 'cell_best'])
        assert (a['T'], a['S']) == (.9, .9)
        assert [b['T'], b['S']] == candidates['cells'][f'{key[2]}/{key[3]}']
        assert a['reference_answers'] == b['reference_answers'] == dev[(key[1], key[2])]['answers']
        assert a['prefix_tokens'] == b['prefix_tokens'] and a['request_tokens'] == b['request_tokens'] and a['block_lengths'] == b['block_lengths']
        for role, record in [('default', a), ('cell_best', b)]:
            record['_pair_id'] = index
            attempts.append({**metadata(record), 'pair_id': index, 'role': role, 'logical_labels': record['logical_labels'], 'attempt': record['attempt'], 'status': record['status'], 'saved_f1_0_1': record['score'], 'f1_percent': record['score'] * 100, 'answer': record['answer'], 'reference_answers': record['reference_answers'], 'output_tokens': record['output_tokens'], 'output_token_ids': record['output_token_ids'], 'eos_last': record['output_token_ids'][-1] in stop_ids, 'max_new_tokens': dev[(key[1], key[2])]['max_new_tokens'], **{f: record[f] for f in ['prefix_tokens', 'request_tokens', 'query_start_position', 'block_lengths', 'original_document_count', 'cache_mode', 'cache_hit', 'ttft_seconds', 'total_seconds', 'cache_build_seconds', 'cache_get_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes']}, 'source_file': 'restored/output/runs/main80-20261114T041503/attempts.jsonl', 'source_line': record['_line'], 'unit': 'F1 0-1 / x100; tokens; seconds with diagnostic capture; raw memory bytes'})
        pairs.append({'pair_id': index, 'dataset': key[0], 'sample_id': key[1], 'L': key[2], 'N': key[3], 'default_T': a['T'], 'default_S': a['S'], 'best_T': b['T'], 'best_S': b['S'], 'frozen_dynamic_T': rule['cells'][f'{key[2]}/{key[3]}'][0], 'frozen_dynamic_S': rule['cells'][f'{key[2]}/{key[3]}'][1], 'default_f1_0_1': a['score'], 'best_f1_0_1': b['score'], 'difference_pp': (b['score'] - a['score']) * 100, 'default_answer': a['answer'], 'best_answer': b['answer'], 'reference_answers': a['reference_answers'], 'answers_same': a['answer'] == b['answer'], 'token_sequences_same': a['output_token_ids'] == b['output_token_ids'], 'default_output_tokens': a['output_tokens'], 'best_output_tokens': b['output_tokens'], 'default_source_line': a['_line'], 'best_source_line': b['_line'], 'source_file': 'restored/output/runs/main80-20261114T041503/attempts.jsonl', 'unit': 'F1 0-1; difference percentage points; tokens', '_default': a, '_best': b})
    csvwrite('diagnostic_attempts.csv', attempts)
    csvwrite('answer_pairs.csv', [{k: v for k, v in row.items() if not k.startswith('_')} for row in pairs])

    arrays = defaultdict(lambda: defaultdict(dict))
    head_rows, layer_rows, audits, bins, query_head_bins, anomalies = [], [], [], [], [], []
    seen = set()
    for line, capture in stream(RUN / 'attention.jsonl'):
        kid = identity(capture['key'])
        assert kid in records
        record = records[kid]
        step, layer = capture['step'], capture['layer']
        shape_id = (kid, layer, step)
        if shape_id in seen:
            anomalies.append({'issue': 'duplicate key/layer/step', 'source_line': line})
        seen.add(shape_id)
        weights = np.asarray(capture['token_probabilities'], dtype=float)
        branches = np.asarray(capture['branch_masses'], dtype=float)
        p, length, q = (record[f] for f in ['prefix_tokens', 'L', 'request_tokens'])
        expected = p + length + q + step
        assert weights.shape == (4, expected) and branches.shape == (4, 2)
        assert np.isfinite(branches).all() and (branches >= 0).all()
        assert math.isfinite(capture['max_abs_output_error']) and math.isfinite(capture['tolerance']) and capture['tolerance'] > 0
        assert capture['heads'] == HEADS and layer in LAYERS
        assert [capture['T'], capture['S']] == [record['T'], record['S']]
        assert capture['diagnostic_roles'] == record['logical_labels']
        assert capture['query_kind'] == ('query_last_token' if step == 0 else 'decode_token')
        nonfinite = int((~np.isfinite(weights)).sum())
        negative = int((weights < 0).sum())
        if nonfinite or negative or capture['max_abs_output_error'] > capture['tolerance']:
            anomalies.append({'issue': 'weight value or saved reconstruction threshold failure', 'source_line': line})
        mask = evidence[(record['sample_id'], length)]['_mask']
        wb = np.array([weights[:, start:start + 64].sum(axis=1) for start in range(0, expected, 64)]).T
        arrays[kid][layer][step] = {'head_bins': wb if step == 0 else None, 'mean_bins': wb.mean(axis=0)}
        if step == 0:
            for j, head in enumerate(HEADS):
                for b, mass in enumerate(wb[j]):
                    query_head_bins.append({**metadata(record), 'pair_id': record['_pair_id'], 'role': record['_role'], 'layer': layer, 'head': head, 'step': 0, 'bin': b, 'token_start': 64 * b, 'token_end_exclusive': min(64 * (b + 1), expected), 'mass': float(mass), 'source_line': line, 'unit': 'raw saved weight sum per 64 physical-cache tokens; one head; query last token'})
        per_head = []
        for j, head in enumerate(HEADS):
            w, document = weights[j], weights[j, p:p + length]
            masses = {'prefix': float(w[:p].sum()), 'evidence': float(document[mask].sum()), 'background': float(document[~mask].sum()), 'request': float(w[p + length:p + length + q].sum()), 'decoded_history': float(w[p + length + q:].sum())}
            context = masses['evidence'] + masses['background']
            other = masses['prefix'] + masses['request'] + masses['decoded_history']
            conditional = document / context if context > 0 else np.zeros(length)
            nz = conditional[conditional > 0]
            block_edges = np.cumsum([0, *record['block_lengths']])
            result = {**metadata(record), 'pair_id': record['_pair_id'], 'role': record['_role'], 'layer': layer, 'head': head, 'step': step, 'query_kind': capture['query_kind'], **{k + '_mass': v for k, v in masses.items()}, 'context_mass': context, 'other_mass': other, 'total_mass': context + other, 'context_branch_mass': float(branches[j, 0]), 'other_branch_mass': float(branches[j, 1]), 'context_branch_error': context - float(branches[j, 0]), 'other_branch_error': other - float(branches[j, 1]), 'evidence_fraction_context': masses['evidence'] / context if context else None, 'evidence_token_density': masses['evidence'] / int(mask.sum()), 'background_token_density': masses['background'] / int((~mask).sum()), 'evidence_enrichment_vs_uniform_context': masses['evidence'] / context / mask.mean() if context else None, 'context_entropy_nats': float(-(nz * np.log(nz)).sum()), 'context_max_conditional_token': float(conditional.max()), 'block_masses': [float(document[a:b].sum()) for a, b in zip(block_edges[:-1], block_edges[1:])], 'source_file': 'restored/output/runs/main80-20261114T041503/attention.jsonl', 'source_line': line, 'unit': 'raw saved mass 0-1 with BF16 rounding; fractions/density/enrichment; entropy nats'}
            head_rows.append(result)
            per_head.append(result)
        layer_row = {**metadata(record), 'pair_id': record['_pair_id'], 'role': record['_role'], 'layer': layer, 'step': step, 'query_kind': capture['query_kind'], 'n_heads': 4}
        for field in [*[k + '_mass' for k in REGIONS], 'context_mass', 'other_mass', 'total_mass', 'evidence_fraction_context', 'evidence_enrichment_vs_uniform_context', 'context_entropy_nats']:
            layer_row[field] = float(np.mean([r[field] for r in per_head]))
        layer_rows.append({**layer_row, 'source_line': line, 'unit': 'arithmetic mean over heads 0/8/16/24; saved mass; conditional evidence fraction; entropy nats'})
        audits.append({**metadata(record), 'pair_id': record['_pair_id'], 'role': record['_role'], 'layer': layer, 'step': step, 'query_kind': capture['query_kind'], 'n_heads': 4, 'saved_token_count': weights.shape[1], 'expected_token_count': expected, 'nonfinite_count': nonfinite, 'negative_count': negative, 'max_abs_output_error_saved': capture['max_abs_output_error'], 'tolerance_saved': capture['tolerance'], 'error_to_tolerance': capture['max_abs_output_error'] / capture['tolerance'], 'within_saved_tolerance': capture['max_abs_output_error'] <= capture['tolerance'], 'max_normalization_abs_error': float(np.abs(weights.sum(axis=1) - 1).max()), 'max_context_branch_abs_error': max(abs(r['context_branch_error']) for r in per_head), 'max_other_branch_abs_error': max(abs(r['other_branch_error']) for r in per_head), 'forward_input_last_token_id': None if step == 0 else record['output_token_ids'][step - 1], 'predicted_output_token_id': record['output_token_ids'][step], 'source_file': 'restored/output/runs/main80-20261114T041503/attention.jsonl', 'source_line': line, 'unit': 'token count; saved pre-o_proj absolute error and tolerance; probability mass error'})
        for b, mass in enumerate(wb.mean(axis=0)):
            bins.append({**metadata(record), 'pair_id': record['_pair_id'], 'role': record['_role'], 'layer': layer, 'step': step, 'bin': b, 'token_start': 64 * b, 'token_end_exclusive': min(64 * (b + 1), expected), 'mass_mean_heads': float(mass), 'source_line': line, 'unit': 'sum within 64 physical-cache tokens; mean of four heads'})
    coverage = []
    for kid, record in records.items():
        for layer in LAYERS:
            actual = sorted(arrays[kid][layer])
            passed = actual == list(range(record['output_tokens']))
            coverage.append({**metadata(record), 'pair_id': record['_pair_id'], 'role': record['_role'], 'layer': layer, 'n_steps': len(actual), 'query_steps': int(0 in actual), 'decode_steps': len(actual) - int(0 in actual), 'expected_steps': record['output_tokens'], 'actual_steps': actual, 'passed': passed, 'unit': 'one query forward plus output_tokens-1 decode forwards; final EOS not forwarded'})
            if not passed:
                anomalies.append({'issue': 'missing or extra query/decode steps', **metadata(record), 'layer': layer})
    assert len(audits) == sum(r['output_tokens'] * 3 for r in records.values())
    csvwrite('capture_audit.csv', audits)
    csvwrite('step_coverage.csv', coverage)
    csvwrite('head_steps.csv', head_rows)
    csvwrite('layer_steps.csv', layer_rows)
    csvwrite('physical_cache_bins.csv', bins)
    csvwrite('query_head_bins.csv', query_head_bins)
    csvwrite('anomalies.csv', anomalies)

    lookup = {(r['pair_id'], r['role'], r['layer'], r['head'], r['step']): r for r in head_rows}
    paired_heads, paired_layers, histories = [], [], []
    fields = [*[k + '_mass' for k in REGIONS], 'context_mass', 'evidence_fraction_context', 'evidence_enrichment_vs_uniform_context', 'context_entropy_nats']
    for pair in pairs:
        a, b = pair['_default'], pair['_best']
        shared_steps = min(a['output_tokens'], b['output_tokens'])
        for step in range(shared_steps):
            same_history = a['output_token_ids'][:step] == b['output_token_ids'][:step]
            histories.append({'pair_id': pair['pair_id'], 'dataset': pair['dataset'], 'sample_id': pair['sample_id'], 'L': pair['L'], 'N': pair['N'], 'step': step, 'same_history_before_forward': same_history, 'history_length_generated_tokens': step, 'default_generated_history_ids': a['output_token_ids'][:step], 'best_generated_history_ids': b['output_token_ids'][:step], 'unit': 'paired forward step; generated prefix before current prediction'})
            for layer in LAYERS:
                group = []
                for head in HEADS:
                    aa, bb = (lookup[(pair['pair_id'], role, layer, head, step)] for role in ['default', 'cell_best'])
                    result = {k: pair[k] for k in ['pair_id', 'dataset', 'sample_id', 'L', 'N']}
                    result.update(layer=layer, head=head, step=step, same_history_before_forward=same_history, default_attention_line=aa['source_line'], best_attention_line=bb['source_line'])
                    for field in fields:
                        result['default_' + field] = aa[field]
                        result['best_' + field] = bb[field]
                        result['delta_' + field] = bb[field] - aa[field]
                    result['unit'] = 'candidate minus default; raw mass/fraction; entropy nats; paired head/step'
                    paired_heads.append(result)
                    group.append(result)
                aggregate = {k: result[k] for k in ['pair_id', 'dataset', 'sample_id', 'L', 'N', 'layer', 'step', 'same_history_before_forward']}
                aggregate['n_heads'] = 4
                for prefix in ['default_', 'best_', 'delta_']:
                    for field in fields:
                        aggregate[prefix + field] = float(np.mean([r[prefix + field] for r in group]))
                paired_layers.append({**aggregate, 'unit': 'equal mean of four specified heads; candidate minus default; mass/fraction/nats'})
    csvwrite('paired_head_steps.csv', paired_heads)
    csvwrite('paired_layer_steps.csv', paired_layers)
    csvwrite('decode_history_alignment.csv', histories)
    query_pairs = [r for r in paired_layers if r['step'] == 0]
    csvwrite('query_region_pairs.csv', query_pairs)
    selected, case_counts = [], []
    for label, predicate, sign in [('改善', lambda x: x > 1e-12, -1), ('退化', lambda x: x < -1e-12, 1), ('持平', lambda x: abs(x) <= 1e-12, 1)]:
        available = sorted((r for r in pairs if predicate(r['difference_pp'])), key=lambda r: (sign * r['difference_pp'], r['sample_id'], r['L'], r['N']))
        case_counts.append({'case_type': label, 'n_pairs': len(available), 'denominator': 6, 'unit': 'paired input conditions; 3 unique questions', 'selection_rule': 'maximum positive/negative saved F1 delta, zero/ties sample_id then L/N ascending'})
        if available:
            selected.append({'case_type': label, **{k: v for k, v in available[0].items() if not k.startswith('_')}})
    csvwrite('selected_cases.csv', selected)
    csvwrite('case_type_counts.csv', case_counts)

    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 9, 'axes.unicode_minus': False})
    for pair in pairs:
        draw_case(pair, arrays, evidence)
    draw_overviews(query_pairs, pairs, evidence)

    quality = {r['case_type']: r['n_pairs'] for r in case_counts}
    summary = {'experiment_stage': 5, 'status': 'COMPLETED', 'research_status': 'SUBMITTED', 'physical_generations': len(records), 'unique_question_ids': len(ids), 'paired_input_conditions': len(pairs), 'attention_rows': len(audits), 'head_step_rows': len(head_rows), 'layer_coverage_rows': len(coverage), 'anomaly_count': len(anomalies), 'quality_pair_counts': quality, 'max_saved_output_reconstruction_error': max(r['max_abs_output_error_saved'] for r in audits), 'max_error_to_saved_tolerance': max(r['error_to_tolerance'] for r in audits), 'max_normalization_abs_error': max(r['max_normalization_abs_error'] for r in audits), 'max_context_branch_abs_error': max(r['max_context_branch_abs_error'] for r in audits), 'max_other_branch_abs_error': max(r['max_other_branch_abs_error'] for r in audits), 'evidence_missing_input_constructions': sum(r['missing_required_tokens'] > 0 for r in evidence_rows), 'sparse_saved_evidence_input_constructions': sum(r['saved_mask_omitted_tokens'] > 0 for r in evidence_rows), 'score_recomputed_count': 0, 'new_generations': 0, 'rule_modified': False, 'next_stage_executed': False, 'bootstrap_executed': False, 'analysis_seed_reserved': 20261116}
    (OUT / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'STAGE_STATUS.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    changes = subprocess.check_output(['git', 'diff', '--name-only', f'{REFERENCE}..HEAD'], cwd=ROOT, text=True).splitlines()
    env = {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'tokenizers': version('tokenizers'), 'analysis_commit': commit, 'reference_commit': REFERENCE, 'changes_from_reference': changes, 'new_dependencies': [], 'stochastic_analysis': False, 'reserved_bootstrap_seed': 20261116, 'tokenizer': str(TOKENIZER.relative_to(ROOT)), 'input': '12 attention attempts only; attention.jsonl streamed; six frozen development inputs and three sources; pre-audit evidence counts', 'outputs': str(OUT), 'environment_policy': 'Python -s -B; user site and bytecode disabled; all cache/temp paths under stage output'}
    (OUT / 'environment.json').write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
