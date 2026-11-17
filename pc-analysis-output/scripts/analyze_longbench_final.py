"""Experiment 6 only: five-method paired LongBench test statistics on CPU.

Uses previously verified saved F1 values. No re-scoring, model execution,
parameter selection, downstream experiment analysis, or evidence mutation.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from importlib.metadata import version
import csv
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'pc-analysis-output'
OUT = BASE / 'stages/06_longbench_final'
RUN = ROOT / 'restored/output/runs/main80-20261114T041503'
FROZEN = RUN / 'snapshot/datasets/frozen-main80'
REFERENCE = '227116e4a004153272640cfe31087a033861b8e9'
for name in ['tables', 'reports', 'figures', 'sources', 'logs', 'cache', 'tmp']:
    (OUT / name).mkdir(parents=True, exist_ok=True)
for key in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TEMP', 'TMP']:
    os.environ[key] = str(OUT / ('tmp' if key in ['TEMP', 'TMP'] else 'cache'))

import numpy as np

DS = ['qasper', 'hotpotqa', 'musique']
METHODS = ['sequential', 'pcw', 'ape_default', 'ape_fixed', 'ape_dynamic']
SEED, BOOT = 20261116, 5000
METRICS = {'score': 'F1 0-1', 'ttft_seconds': 's', 'total_seconds': 's', 'cache_build_seconds': 's', 'cache_get_seconds': 's', 'dynamic_selection_seconds': 's', 'output_tokens': 'tokens including EOS', 'peak_allocated_bytes': 'GiB', 'peak_reserved_bytes': 'GiB', 'after_ttft_seconds': 's'}
COLS = [(m, f) for m in METHODS for f in METRICS]
METHOD_PAIRS = list(combinations(METHODS, 2))
RATIOS = [(a, b, f) for a, b in METHOD_PAIRS for f in ['ttft_seconds', 'total_seconds']]
CONTRIB = ['mapped_4096', 'mapped_8192', 'mapped_16384', 'N_1', 'N_2', 'N_3', 'support_overlap', 'support_nonoverlap']
GRID = [(l, n) for l in [4096, 8192, 16384] for n in [4, 8, 16]]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def csvout(name, rows, empty_fields=('issue', 'sample_id', 'source_line')):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or list(empty_fields)
    with (OUT / 'tables' / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict, tuple)) else v for k, v in r.items()})


def serial(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def nearest(length, actual_n):
    return min(GRID, key=lambda c: ((math.log2(length) - math.log2(c[0]))**2 + (math.log2(actual_n) - math.log2(c[1]))**2, c[0], c[1]))


def ci(values):
    return np.quantile(values, [.025, .975], axis=0)


def clustered_means(values, groups, seed_key):
    unique = sorted(set(groups))
    lookup = {g: j for j, g in enumerate(unique)}
    sizes = np.zeros(len(unique))
    sums = np.zeros((len(unique), values.shape[1]))
    for row, group in zip(values, groups):
        sizes[lookup[group]] += 1
        sums[lookup[group]] += row
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *seed_key]))
    result = []
    for start in range(0, BOOT, 100):
        idx = rng.integers(0, len(unique), size=(min(100, BOOT - start), len(unique)))
        result.append(sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)[:, None])
    return np.concatenate(result), len(unique)


def main():
    frozen = read(FROZEN / 'frozen.json')
    test_ids = set(frozen['orders']['test'])
    assert len(test_ids) == 510
    source = {r['sample_id']: r for r in read(FROZEN / 'longbench.json')}
    assert all(source[s]['split'] == 'test' for s in test_ids)
    rule = read(RUN / 'rule.json')
    source_path = 'restored/output/runs/main80-20261114T041503/attempts.jsonl'
    records, by_id = [], defaultdict(dict)
    bad = []
    with (RUN / 'attempts.jsonl').open(encoding='utf-8') as handle:
        for line, text in enumerate(handle, 1):
            r = json.loads(text)
            if r['key']['stage'] not in ['baseline', 'longbench_final'] or r['sample_id'] not in test_ids:
                continue
            assert r['method'] in METHODS
            assert r['key']['stage'] == ('baseline' if r['method'] in METHODS[:3] else 'longbench_final')
            assert r['method'] not in by_id[r['sample_id']]
            assert r['status'] == 'success' and r['attempt'] == 1
            assert r['cache_mode'] == 'cold' and not r['cache_hit'] and r['timing_condition'] == 'cold'
            assert r['reference_answers'] == source[r['sample_id']]['answers']
            for field in ['score', *list(METRICS)[1:-1]]:
                assert isinstance(r[field], (int, float)) and math.isfinite(r[field]) and r[field] >= 0
            assert r['score'] <= 1 and r['total_seconds'] >= r['ttft_seconds'] > 0
            assert len(r['output_token_ids']) == r['output_tokens']
            if not r['answer'].strip():
                bad.append({'issue': 'empty answer retained with zero score', 'sample_id': r['sample_id'], 'method': r['method'], 'source_line': line})
                assert r['score'] == 0
            r['_line'] = line
            r['after_ttft_seconds'] = r['total_seconds'] - r['ttft_seconds']
            by_id[r['sample_id']][r['method']] = r
            records.append(r)
    assert len(records) == 2550 and set(by_id) == test_ids
    assert all(set(rows) == set(METHODS) for rows in by_id.values())

    dev_sources = [r for r in source.values() if r['split'] in ['explore', 'validation']]
    dev_support = {doc for r in dev_sources for doc in r['support_documents']}
    dev_docs = {doc for r in dev_sources for doc in r['input_documents']}
    info, membership, paired_samples, requests = {}, [], [], []
    eos = read(RUN / 'snapshot/models/Llama-3.1-8B-Instruct/generation_config.json')['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    for sid in sorted(test_ids):
        s, rr = source[sid], by_id[sid]
        r = rr['ape_dynamic']
        mapped = nearest(r['L'], r['N'])
        assert r['L'] == s['_length'] and r['N'] == math.ceil(r['L'] / 7500)
        expected_blocks = [min(7500, r['L'] - start) for start in range(0, r['L'], 7500)]
        assert (r['T'], r['S']) == tuple(rule['cells'][f'{mapped[0]}/{mapped[1]}'])
        assert (rr['ape_fixed']['T'], rr['ape_fixed']['S']) == tuple(rule['fixed'])
        assert (rr['ape_default']['T'], rr['ape_default']['S']) == (.9, .9)
        for method, other in rr.items():
            assert (other['dataset'], other['L'], other['N'], other['block_lengths'], other['group']) == (s['dataset'], r['L'], r['N'], expected_blocks, s['group'])
            assert other['source'] == s['source_ids']
            assert other['prefix_tokens'] == r['prefix_tokens']
            assert other['request_tokens'] == r['request_tokens'] + (r['prefix_tokens'] - 1 if method == 'pcw' else 0)
        same_params = (r['T'], r['S']) == (rr['ape_fixed']['T'], rr['ape_fixed']['S'])
        same_tokens = r['output_token_ids'] == rr['ape_fixed']['output_token_ids']
        if same_params:
            assert same_tokens
        shared_support = sorted(set(s['support_documents']) & dev_support)
        shared_docs = sorted(set(s['input_documents']) & dev_docs)
        group = serial(s['group'])
        meta = {'dataset': s['dataset'], 'sample_id': sid, 'group': group, 'L': r['L'], 'actual_N': r['N'], 'original_document_count': r['original_document_count'], 'mapped_L': mapped[0], 'mapped_N': mapped[1], 'N_outside_grid': not 4 <= r['N'] <= 16, 'L_below_grid': r['L'] < 4096, 'L_above_grid': r['L'] > 16384, 'support_overlap_development': bool(shared_support), 'single_document_overlap_development': bool(shared_docs), 'shared_support_documents': shared_support, 'shared_input_documents': shared_docs, 'support_documents': s['support_documents'], 'fixed_dynamic_same_parameters': same_params, 'fixed_dynamic_same_tokens': same_tokens, 'default_fixed_same_tokens': rr['ape_default']['output_token_ids'] == rr['ape_fixed']['output_token_ids'], 'default_dynamic_same_tokens': rr['ape_default']['output_token_ids'] == r['output_token_ids'], 'three_APE_same_tokens': rr['ape_default']['output_token_ids'] == rr['ape_fixed']['output_token_ids'] == r['output_token_ids']}
        info[sid] = meta
        membership.append({**meta, 'methods_present': list(rr), 'five_method_complete': True, 'source_file': source_path, 'unit': 'paired question; L document tokens; actual_N encoded blocks; original document count separate'})
        delta = r['score'] - rr['ape_fixed']['score']
        paired_samples.append({**meta, 'fixed_T': rr['ape_fixed']['T'], 'fixed_S': rr['ape_fixed']['S'], 'dynamic_T': r['T'], 'dynamic_S': r['S'], 'fixed_f1_0_1': rr['ape_fixed']['score'], 'dynamic_f1_0_1': r['score'], 'delta_f1_0_1': delta, 'delta_pp': delta * 100, 'fixed_answer': rr['ape_fixed']['answer'], 'dynamic_answer': r['answer'], 'reference_answers': r['reference_answers'], 'question': s['question'], 'fixed_output_tokens': rr['ape_fixed']['output_tokens'], 'dynamic_output_tokens': r['output_tokens'], 'fixed_source_line': rr['ape_fixed']['_line'], 'dynamic_source_line': r['_line'], **{m + '_f1_0_1': rr[m]['score'] for m in METHODS}, **{m + '_' + field: rr[m][field] for m in ['ape_fixed', 'ape_dynamic'] for field in ['ttft_seconds', 'total_seconds', 'after_ttft_seconds']}, 'source_file': source_path, 'unit': 'F1 0-1; delta pp; seconds; token counts'})
        for method, other in rr.items():
            requests.append({**{k: meta[k] for k in ['dataset', 'sample_id', 'L', 'actual_N', 'mapped_L', 'mapped_N', 'group', 'support_overlap_development']}, 'method': method, 'stage': other['key']['stage'], 'task_key': other['key'], 'task_N': other['key']['N'], 'attempt': other['attempt'], 'logical_labels': other['logical_labels'], 'status': other['status'], 'T': other['T'], 'S': other['S'], 'saved_f1_0_1': other['score'], 'answer': other['answer'], 'reference_answers': other['reference_answers'], 'output_token_ids': other['output_token_ids'], 'max_new_tokens': s['max_new_tokens'], 'cap_hit': other['output_tokens'] == s['max_new_tokens'], 'eos_last': other['output_token_ids'][-1] in eos, **{field: other[field] for field in METRICS if field != 'score'}, **{field: other[field] for field in ['prefix_tokens', 'request_tokens', 'query_start_position', 'block_lengths', 'window_order', 'original_document_count', 'stage_calls', 'started_at', 'ended_at', 'cache_mode', 'cache_hit', 'timing_condition']}, 'source_file': source_path, 'source_line': other['_line'], 'unit': 'saved F1 0-1; raw memory bytes; all timing seconds; EOS included in output tokens'})
    prior_overlap = {r['sample_id'] for r in readcsv(BASE / 'stages/01_audit/tables/splits_overlap_affected_ids.csv') if (r['left_split'], r['right_split'], r['overlap_type'], r['side']) == ('development', 'test', 'support_document', 'right')}
    assert prior_overlap == {sid for sid, x in info.items() if x['support_overlap_development']}
    csvout('record_issues.csv', bad)
    csvout('membership.csv', membership)
    csvout('per_request_records.csv', requests)
    csvout('dynamic_fixed_per_sample.csv', paired_samples)
    csvout('development_overlap_affected_ids.csv', [r for r in membership if r['support_overlap_development'] or r['single_document_overlap_development']])

    parent = {sid: sid for sid in test_ids}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen_groups, seen_docs = {}, {}
    for sid in sorted(test_ids):
        for seen, keys in [(seen_groups, [info[sid]['group']]), (seen_docs, source[sid]['support_documents'])]:
            for key in keys:
                if key in seen:
                    union(sid, seen[key])
                seen[key] = sid
    components = {sid: find(sid) for sid in test_ids}
    csvout('bootstrap_membership.csv', [{'dataset': info[sid]['dataset'], 'sample_id': sid, 'frozen_group': info[sid]['group'], 'support_connected_component': components[sid], 'unit': 'paired source cluster; all five methods retained'} for sid in sorted(test_ids)])

    scopes = [('test510', sorted(test_ids))]
    scopes += [(f'mapped_{length}_4', [sid for sid in sorted(test_ids) if info[sid]['mapped_L'] == length]) for length in [4096, 8192, 16384]]
    scopes += [(f'actual_N_{n}', [sid for sid in sorted(test_ids) if info[sid]['actual_N'] == n]) for n in [1, 2, 3]]
    scopes += [(label, [sid for sid in sorted(test_ids) if predicate(info[sid])]) for label, predicate in [
        ('support_overlap', lambda x: x['support_overlap_development']),
        ('support_nonoverlap', lambda x: not x['support_overlap_development']),
        ('fixed_dynamic_same_parameters', lambda x: x['fixed_dynamic_same_parameters']),
        ('fixed_dynamic_changed_parameters', lambda x: not x['fixed_dynamic_same_parameters']),
        ('L_inside_grid', lambda x: not x['L_below_grid'] and not x['L_above_grid']),
        ('L_outside_grid', lambda x: x['L_below_grid'] or x['L_above_grid']),
    ]]
    quality, paired, efficiency, performance, diagnostics, scope_rows, contributions = [], [], [], [], [], [], []

    def values_for(ids):
        values = []
        for sid in ids:
            rr, meta = by_id[sid], info[sid]
            row = [rr[m][f] / 2**30 if METRICS[f] == 'GiB' else rr[m][f] for m, f in COLS]
            row += [rr[a][f] / rr[b][f] for a, b, f in RATIOS]
            delta = rr['ape_dynamic']['score'] - rr['ape_fixed']['score']
            flags = [meta['mapped_L'] == l for l in [4096, 8192, 16384]] + [meta['actual_N'] == n for n in [1, 2, 3]] + [meta['support_overlap_development'], not meta['support_overlap_development']]
            row += [delta * flag for flag in flags]
            values.append(row)
        return np.array(values, dtype=float)

    def emit(scope, dataset, a, b, n_groups, included_ds, cluster_type, weights_note):
        common = {'scope': scope, 'dataset': dataset, 'n_questions': len(a), 'n_clusters': n_groups, 'datasets_in_scope': included_ds, 'cluster_type': cluster_type, 'aggregation': weights_note, 'seed': SEED, 'replicates': BOOT}
        for method in METHODS:
            j = COLS.index((method, 'score'))
            lo, hi = ci(b[:, j])
            quality.append({**common, 'method': method, 'f1_0_1': float(a[:, j].mean()), 'f1_percent': float(a[:, j].mean() * 100), 'ci_low_0_1': float(lo), 'ci_high_0_1': float(hi), 'ci_low_percent': float(lo * 100), 'ci_high_percent': float(hi * 100), 'unit': 'saved F1 0-1; display F1 x100'})
            for field, unit in METRICS.items():
                if field == 'score':
                    continue
                j = COLS.index((method, field))
                lo, hi = ci(b[:, j])
                efficiency.append({**common, 'method': method, 'metric': field, 'mean': float(a[:, j].mean()), 'median': float(np.median(a[:, j])), 'p90': float(np.quantile(a[:, j], .9)), 'mean_ci_low': float(lo), 'mean_ci_high': float(hi), 'unit': unit, 'cache_condition': 'cold miss; no attention observer; document encoding included'})
        for ma, mb in METHOD_PAIRS:
            ia, ib = [COLS.index((m, 'score')) for m in [ma, mb]]
            delta = a[:, ib] - a[:, ia]
            lo, hi = ci(b[:, ib] - b[:, ia])
            paired.append({**common, 'reference_method': ma, 'comparison_method': mb, 'delta_comparison_minus_reference_0_1': float(delta.mean()), 'delta_pp': float(delta.mean() * 100), 'ci_low_pp': float(lo * 100), 'ci_high_pp': float(hi * 100), 'wins_comparison': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()), 'losses_comparison': int((delta < -1e-12).sum()), 'unit': 'F1 percentage points; counts of paired questions'})
            for field in ['ttft_seconds', 'total_seconds']:
                ia, ib = [COLS.index((m, field)) for m in [ma, mb]]
                ir = len(COLS) + RATIOS.index((ma, mb, field))
                dl, dh = ci(b[:, ib] - b[:, ia])
                rl, rh = ci(b[:, ir])
                ml, mh = ci(b[:, ia] / b[:, ib])
                performance.append({**common, 'reference_method': ma, 'comparison_method': mb, 'metric': field, 'reference_mean_s': float(a[:, ia].mean()), 'comparison_mean_s': float(a[:, ib].mean()), 'delta_comparison_minus_reference_s': float((a[:, ib] - a[:, ia]).mean()), 'delta_ci_low_s': float(dl), 'delta_ci_high_s': float(dh), 'paired_speedup_mean': float(a[:, ir].mean()), 'paired_speedup_median': float(np.median(a[:, ir])), 'paired_speedup_p90': float(np.quantile(a[:, ir], .9)), 'paired_speedup_mean_ci_low': float(rl), 'paired_speedup_mean_ci_high': float(rh), 'ratio_of_means': float(a[:, ia].mean() / a[:, ib].mean()), 'ratio_of_means_ci_low': float(ml), 'ratio_of_means_ci_high': float(mh), 'unit': 'seconds; speedups reference/comparison (>1 comparison faster)'})

    for scope_index, (scope, scope_ids) in enumerate(scopes):
        arrays, boots, sizes, clusters, included = [], [], [], [], []
        for ds_index, dataset in enumerate(DS):
            ids = [sid for sid in scope_ids if info[sid]['dataset'] == dataset]
            scope_rows.append({'scope': scope, 'dataset': dataset, 'n_questions': len(ids), 'n_requests': len(ids) * 5, 'unit': 'paired questions / five-method generation records'})
            if not ids:
                continue
            a = values_for(ids)
            groups = [info[sid]['group'] for sid in ids]
            b, n_groups = clustered_means(a, groups, [scope_index, ds_index, 0])
            diagnostics.append({'scope': scope, 'dataset': dataset, 'n_questions': len(ids), 'n_clusters': n_groups, 'cluster_type': 'frozen paper/evidence group', 'largest_cluster_questions': max(Counter(groups).values()), 'seed_key': [SEED, scope_index, ds_index, 0], 'replicates': BOOT, 'unit': 'all five methods retained per sampled source group'})
            emit(scope, dataset, a, b, n_groups, [dataset], 'frozen_group', 'question-weighted within dataset')
            arrays.append(a); boots.append(b); sizes.append(len(a)); clusters.append(n_groups); included.append(dataset)
        a = np.concatenate(arrays)
        pooled_boot = np.average(boots, axis=0, weights=sizes)
        emit(scope, 'pooled_requests', a, pooled_boot, sum(clusters), included, 'frozen_group', 'fixed dataset request proportions in this scope')
        macro_boot = np.mean(boots, axis=0)
        # Point values use dataset-equal weights while raw win/tie/loss counts stay paired-ID counts.
        before = (len(quality), len(paired), len(efficiency), len(performance))
        emit(scope, 'macro_equal_present_dataset', a, macro_boot, sum(clusters), included, 'frozen_group', 'equal weight for each present dataset; main scope has all three')
        macro_mean = np.mean([v.mean(axis=0) for v in arrays], axis=0)
        for r in quality[before[0]:]:
            value = macro_mean[COLS.index((r['method'], 'score'))]
            r.update(f1_0_1=float(value), f1_percent=float(value * 100))
        for r in paired[before[1]:]:
            value = macro_mean[COLS.index((r['comparison_method'], 'score'))] - macro_mean[COLS.index((r['reference_method'], 'score'))]
            r.update(delta_comparison_minus_reference_0_1=float(value), delta_pp=float(value * 100))
        # Efficiency main tables use pooled request distributions; macro timing rows are omitted.
        del efficiency[before[2]:]
        del performance[before[3]:]
        if scope == 'test510':
            for j, label in enumerate(CONTRIB):
                col = len(COLS) + len(RATIOS) + j
                lo, hi = ci(macro_boot[:, col])
                contributions.append({'stratum': label, 'overall_n_questions': 510, 'contribution_to_overall_delta_pp': float(macro_mean[col] * 100), 'ci_low_pp': float(lo * 100), 'ci_high_pp': float(hi * 100), 'unit': 'dataset-equal overall dynamic-minus-fixed F1 percentage-point contribution; zero outside stratum', 'seed': SEED, 'replicates': BOOT})

    # Source-sharing sensitivity uses the identical cohort and frozen parameters.
    arrays, boots, clusters = [], [], []
    for ds_index, dataset in enumerate(DS):
        ids = sorted(sid for sid in test_ids if info[sid]['dataset'] == dataset)
        a = values_for(ids)
        groups = [components[sid] for sid in ids]
        b, n_groups = clustered_means(a, groups, [99, ds_index, 1])
        emit('test510_support_component_sensitivity', dataset, a, b, n_groups, [dataset], 'shared_support_component', 'question-weighted within dataset; identical cohort')
        arrays.append(a); boots.append(b); clusters.append(n_groups)
        diagnostics.append({'scope': 'test510_support_component_sensitivity', 'dataset': dataset, 'n_questions': len(ids), 'n_clusters': n_groups, 'cluster_type': 'same frozen group or any shared support document connected component', 'largest_cluster_questions': max(Counter(groups).values()), 'seed_key': [SEED, 99, ds_index, 1], 'replicates': BOOT, 'unit': 'all five methods retained per sampled source component'})
    emit('test510_support_component_sensitivity', 'macro_equal_present_dataset', np.concatenate(arrays), np.mean(boots, axis=0), sum(clusters), DS, 'shared_support_component', 'equal datasets; all170 questions/dataset')
    # Equal170 per dataset makes the sensitivity point means and pooled time distributions identical.
    for name, data in [('scope_counts.csv', scope_rows), ('quality.csv', quality), ('quality_paired.csv', paired), ('efficiency_descriptive.csv', efficiency), ('efficiency_paired.csv', performance), ('bootstrap_diagnostics.csv', diagnostics), ('delta_contributions.csv', contributions)]:
        csvout(name, data)

    mapping, distributions, output_match = [], [], []
    for dataset in [*DS, 'ALL']:
        ids = [sid for sid in sorted(test_ids) if dataset == 'ALL' or info[sid]['dataset'] == dataset]
        for length in [4096, 8192, 16384]:
            selected = [sid for sid in ids if info[sid]['mapped_L'] == length]
            mapping.append({'dataset': dataset, 'mapped_L': length, 'mapped_N': 4, 'T': rule['cells'][f'{length}/4'][0], 'S': rule['cells'][f'{length}/4'][1], 'n_questions': len(selected), 'denominator': len(ids), 'fraction': len(selected) / len(ids), 'unit': 'paired test IDs; mapping uses actual L/N'})
        for n in [1, 2, 3]:
            selected = [sid for sid in ids if info[sid]['actual_N'] == n]
            distributions.append({'dataset': dataset, 'actual_N': n, 'n_questions': len(selected), 'denominator': len(ids), 'fraction': len(selected) / len(ids), 'L_min': min(info[sid]['L'] for sid in selected), 'L_max': max(info[sid]['L'] for sid in selected), 'unit': 'paired IDs; actual encoded block count; document tokens'})
        for a, b in combinations(['ape_default', 'ape_fixed', 'ape_dynamic'], 2):
            output_match.append({'dataset': dataset, 'method_a': a, 'method_b': b, 'n_questions': len(ids), 'same_parameters': sum((by_id[s][a]['T'], by_id[s][a]['S']) == (by_id[s][b]['T'], by_id[s][b]['S']) for s in ids), 'same_token_sequences': sum(by_id[s][a]['output_token_ids'] == by_id[s][b]['output_token_ids'] for s in ids), 'same_scores': sum(abs(by_id[s][a]['score'] - by_id[s][b]['score']) <= 1e-12 for s in ids), 'unit': 'paired ID counts; EOS included in token equality'})
    csvout('rule_mapping_counts.csv', mapping)
    csvout('actual_N_distribution.csv', distributions)
    csvout('APE_output_equivalence.csv', output_match)

    limits, cap_conditioned = [], []
    for dataset in DS:
        for method in METHODS:
            rows = [r for r in requests if r['dataset'] == dataset and r['method'] == method]
            limits.append({'dataset': dataset, 'method': method, 'n_requests': len(rows), 'max_new_tokens': rows[0]['max_new_tokens'], 'cap_hits': sum(r['cap_hit'] for r in rows), 'cap_hit_rate': np.mean([r['cap_hit'] for r in rows]), 'eos_last_count': sum(r['eos_last'] for r in rows), 'eos_at_cap_count': sum(r['eos_last'] and r['cap_hit'] for r in rows), 'output_tokens_mean': np.mean([r['output_tokens'] for r in rows]), 'denominator': len(rows), 'unit': 'requests; rates0-1; output tokens include EOS'})
            for hit in [False, True]:
                subset = [r for r in rows if r['cap_hit'] == hit]
                if not subset:
                    continue
                cap_conditioned.append({'dataset': dataset, 'method': method, 'cap_hit': hit, 'n_requests': len(subset), 'denominator_dataset_method': len(rows), 'mean_f1_percent': np.mean([r['saved_f1_0_1'] for r in subset]) * 100, 'mean_total_seconds': np.mean([r['total_seconds'] for r in subset]), 'mean_ttft_seconds': np.mean([r['ttft_seconds'] for r in subset]), 'mean_output_tokens': np.mean([r['output_tokens'] for r in subset]), 'unit': 'F1 x100; seconds; tokens; descriptive output-length grouping'})
    csvout('generation_limits.csv', limits)
    csvout('generation_cap_records.csv', [r for r in requests if r['cap_hit']])
    csvout('generation_cap_conditioned.csv', cap_conditioned)

    length_groups = []
    for label, predicate in [('all510', lambda r: True), ('same_tokens', lambda r: r['fixed_dynamic_same_tokens']), ('different_tokens', lambda r: not r['fixed_dynamic_same_tokens']), ('dynamic_longer', lambda r: r['dynamic_output_tokens'] > r['fixed_output_tokens']), ('dynamic_shorter', lambda r: r['dynamic_output_tokens'] < r['fixed_output_tokens']), ('equal_length_different_tokens', lambda r: r['dynamic_output_tokens'] == r['fixed_output_tokens'] and not r['fixed_dynamic_same_tokens'])]:
        rows = [r for r in paired_samples if predicate(r)]
        length_groups.append({'group': label, 'n_pairs': len(rows), 'denominator': 510, 'delta_f1_pp_pooled': np.mean([r['delta_pp'] for r in rows]), 'mean_fixed_output_tokens': np.mean([r['fixed_output_tokens'] for r in rows]), 'mean_dynamic_output_tokens': np.mean([r['dynamic_output_tokens'] for r in rows]), 'mean_total_delta_s': np.mean([r['ape_dynamic_total_seconds'] - r['ape_fixed_total_seconds'] for r in rows]), 'mean_ttft_delta_s': np.mean([r['ape_dynamic_ttft_seconds'] - r['ape_fixed_ttft_seconds'] for r in rows]), 'unit': 'paired IDs; pooled descriptive means; F1pp/seconds/tokens'})
    csvout('paired_output_length_groups.csv', length_groups)
    cases, case_counts = [], []
    for dataset in [*DS, 'ALL']:
        rows = [r for r in paired_samples if dataset == 'ALL' or r['dataset'] == dataset]
        for label, predicate, sign in [('改善', lambda x: x > 1e-12, -1), ('退化', lambda x: x < -1e-12, 1), ('持平', lambda x: abs(x) <= 1e-12, 1)]:
            selected = sorted((r for r in rows if predicate(r['delta_pp'])), key=lambda r: (sign * r['delta_pp'], r['sample_id']))
            case_counts.append({'scope': dataset, 'type': label, 'n_pairs': len(selected), 'denominator': len(rows), 'unit': 'paired questions'})
            if selected:
                cases.append({'selection_scope': dataset, 'case_type': label, 'selection_rule': 'maximum positive/negative dynamic-minus-fixed saved F1; zero/ties smallest sample_id', **selected[0]})
    csvout('selected_cases.csv', cases)
    csvout('case_type_counts.csv', case_counts)

    original_summary = read(RUN / 'summary.json')
    summary_checks = []
    for row in original_summary['longbench_test_by_method']:
        ids = [sid for sid in test_ids if info[sid]['dataset'] == row['dataset']]
        value = np.mean([by_id[sid][row['method']]['score'] for sid in ids])
        assert set(row['selected_test_ids']) == set(ids) and row['n'] == len(ids)
        summary_checks.append({'section': 'longbench_test_by_method', 'dataset': row['dataset'], 'method': row['method'], 'field': 'mean_f1', 'original': row['mean_f1'], 'current': value, 'difference': value - row['mean_f1'], 'n_questions': len(ids), 'unit': 'saved F1 0-1'})
    for row in original_summary['longbench_five_method_paired']:
        ids = [sid for sid in test_ids if info[sid]['dataset'] == row['dataset']]
        assert set(row['sample_ids']) == set(ids) and row['paired_n'] == len(ids)
        for method in METHODS:
            value = np.mean([by_id[sid][method]['score'] for sid in ids])
            summary_checks.append({'section': 'longbench_five_method_paired', 'dataset': row['dataset'], 'method': method, 'field': 'mean_f1', 'original': row['mean_f1'][method], 'current': value, 'difference': value - row['mean_f1'][method], 'n_questions': len(ids), 'unit': 'saved F1 0-1'})
    assert max(abs(r['difference']) for r in summary_checks) < 1e-12
    csvout('original_summary_crosscheck.csv', summary_checks)
    workers = [r for r in readcsv(BASE / 'stages/01_audit/tables/runtime_workers.csv') if r['stage'] in ['baseline', 'longbench_final']]
    csvout('model_load_workers.csv', workers)
    csvout('server_environment.csv', readcsv(BASE / 'stages/01_audit/tables/runtime_environment.csv'))
    csvout('baseline600_reference.csv', readcsv(BASE / 'stages/01_longbench_baseline/tables/baseline_quality.csv'))
    (OUT / 'sources/original_frozen_rule.json').write_text(json.dumps(rule, ensure_ascii=False, indent=2), encoding='utf-8')
    assert all(r['started_at'] > rule['frozen_at'] for r in records if r['key']['stage'] == 'longbench_final')
    main_quality = [r for r in quality if r['scope'] == 'test510' and r['dataset'] == 'macro_equal_present_dataset']
    primary_pairs = [r for r in paired if r['scope'] == 'test510' and r['reference_method'] == 'ape_fixed' and r['comparison_method'] == 'ape_dynamic' and r['dataset'] != 'pooled_requests']
    results = {'experiment_stage': 6, 'status': 'COMPLETED', 'research_status': 'SUBMITTED', 'n_questions': 510, 'n_datasets': 3, 'n_generations': 2550, 'reused_baseline_generations': 1530, 'new_formal_stage_saved_generations': 1020, 'quality_macro': main_quality, 'dynamic_minus_fixed': primary_pairs, 'frozen_group_count': sum(r['n_clusters'] for r in diagnostics if r['scope'] == 'test510'), 'support_component_count': sum(r['n_clusters'] for r in diagnostics if r['scope'] == 'test510_support_component_sensitivity'), 'rule_branches_used': [r for r in mapping if r['dataset'] == 'ALL'], 'N_outside_grid_count': sum(x['N_outside_grid'] for x in info.values()), 'L_below_grid_count': sum(x['L_below_grid'] for x in info.values()), 'L_above_grid_count': sum(x['L_above_grid'] for x in info.values()), 'support_overlap_development_ids': len(prior_overlap), 'single_document_overlap_development_ids': sum(x['single_document_overlap_development'] for x in info.values()), 'fixed_dynamic_same_parameters': sum(x['fixed_dynamic_same_parameters'] for x in info.values()), 'fixed_dynamic_same_tokens': sum(x['fixed_dynamic_same_tokens'] for x in info.values()), 'three_APE_same_tokens': sum(x['three_APE_same_tokens'] for x in info.values()), 'cap_hit_count': sum(r['cap_hit'] for r in requests), 'empty_answers': len(bad), 'score_recomputed_count': 0, 'rule_modified': False, 'test_used_for_parameter_selection': False, 'next_stage_executed': False, 'seed': SEED, 'replicates': BOOT}
    (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'STAGE_STATUS.json').write_text(json.dumps({k: v for k, v in results.items() if k not in ['quality_macro', 'dynamic_minus_fixed', 'rule_branches_used']}, ensure_ascii=False, indent=2), encoding='utf-8')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    env = {'analysis_commit': commit, 'reference_commit': REFERENCE, 'changes_from_reference': subprocess.check_output(['git', 'diff', '--name-only', f'{REFERENCE}..HEAD'], cwd=ROOT, text=True).splitlines(), 'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'matplotlib': version('matplotlib'), 'new_dependencies': [], 'seed': SEED, 'replicates': BOOT, 'bootstrap': 'percentile paired cluster within dataset; source-group sizes carried; five methods jointly resampled; macro equal datasets, efficiency fixed original dataset request proportions', 'subgroup_intervals': 'clusters restricted to the named subgroup; conditional sample variation within subgroup', 'sensitivity': 'same frozen group OR any shared support document connected component; frozen parameters and same510 IDs', 'rng': 'SeedSequence([20261116, scope_index, dataset_index, 0]); sensitivity [20261116,99,dataset_index,1]', 'cache_and_temp': str(OUT), 'analysis_input': source_path, 'score_policy': 'saved score, verified in pre-audit; no F1 recomputation'}
    (OUT / 'environment.json').write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'stage': 6, 'quality_macro': {r['method']: r['f1_percent'] for r in main_quality}, 'dynamic_minus_fixed': primary_pairs, 'cap_hits': results['cap_hit_count'], 'status': 'SUBMITTED'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
