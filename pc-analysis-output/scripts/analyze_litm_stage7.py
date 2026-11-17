"""Experiment 7 only: cached official first-line LITM EM and question-paired CIs.

Reuses the pre-audit's 501 corrected labels; verifies nine original-answer
fixtures. No model generation, rule changes, or Doc2Dial analysis.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from importlib.metadata import version
import csv
import json
import math
import os
import subprocess
import sys
import string

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'pc-analysis-output'
OUT = BASE / 'stages/07_litm'
RUN = ROOT / 'restored/output/runs/main80-20261114T041503'
FROZEN = RUN / 'snapshot/datasets/frozen-main80'
AUDIT = BASE / 'stages/01_audit/tables'
REFERENCE = '227116e4a004153272640cfe31087a033861b8e9'
for name in ['tables', 'reports', 'figures', 'sources', 'logs', 'cache', 'tmp']:
    (OUT / name).mkdir(parents=True, exist_ok=True)
for key in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TEMP', 'TMP']:
    os.environ[key] = str(OUT / ('tmp' if key in ['TEMP', 'TMP'] else 'cache'))
import numpy as np
import regex

METHODS = ['sequential', 'ape_fixed', 'ape_dynamic']
POSITIONS = [0, 9, 19]
SEED, BOOT = 20261116, 5000
SCORES = ['official_first_line_em', 'saved_full_answer_em']
METRICS = {**{s: 'EM 0-1' for s in SCORES}, 'ttft_seconds': 's', 'total_seconds': 's', 'cache_build_seconds': 's', 'cache_get_seconds': 's', 'dynamic_selection_seconds': 's', 'output_tokens': 'tokens including EOS', 'peak_allocated_bytes': 'GiB', 'peak_reserved_bytes': 'GiB', 'after_ttft_seconds': 's'}
COLS = [(m, f) for m in METHODS for f in METRICS]
PAIRS = list(combinations(METHODS, 2))
RATIOS = [(a, b, f) for a, b in PAIRS for f in ['ttft_seconds', 'total_seconds']]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def csvout(name, rows, empty_fields=('issue', 'sample_id', 'source_line')):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields)
    with (OUT / 'tables' / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(regex.sub(r'\b(a|an|the)\b', ' ', text).split())


def fixture_em(answer, references):
    normalized = normalize(answer)
    return float(any(normalize(ref) in normalized for ref in references))


def nearest(length, n):
    grid = [(l, count) for l in [4096, 8192, 16384] for count in [4, 8, 16]]
    return min(grid, key=lambda c: ((math.log2(length) - math.log2(c[0]))**2 + (math.log2(n) - math.log2(c[1]))**2, c[0], c[1]))


def ci(values):
    return np.quantile(values, [.025, .975], axis=0)


def win_tie_loss(delta):
    return {'wins_comparison': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()), 'losses_comparison': int((delta < -1e-12).sum())}


def main():
    frozen = read(FROZEN / 'frozen.json')
    selected = set(frozen['orders']['litm'])
    assert len(selected) == 500
    inputs = read(FROZEN / 'litm.json')
    source = {(r['sample_id'], r['position']): r for r in inputs}
    assert len(inputs) == len(source) == 1500
    assert {r['sample_id'] for r in inputs} == selected
    input_audit = readcsv(AUDIT / 'scoring_litm_input_checks.csv')
    assert len(input_audit) == 1500 and all(r['status'] == 'PASS' for r in input_audit)
    assert {(r['sample_id'], int(r['position'])) for r in input_audit} == set(source)
    csvout('preaudit_input_checks.csv', input_audit)
    changes = readcsv(AUDIT / 'scoring_litm_first_line_changed_IDs.csv')
    corrected = {(r['sample_id'], r['method'], int(r['position'])): r for r in changes}
    assert len(changes) == len(corrected) == 501
    rule = read(RUN / 'rule.json')
    eos = read(RUN / 'snapshot/models/Llama-3.1-8B-Instruct/generation_config.json')['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    records, lookup, issues = [], {}, []
    with (RUN / 'attempts.jsonl').open(encoding='utf-8') as handle:
        for line, text in enumerate(handle, 1):
            r = json.loads(text)
            if r['key']['stage'] != 'litm':
                continue
            key = (r['sample_id'], r['method'], r['position'])
            assert key not in lookup and r['sample_id'] in selected
            assert r['method'] in METHODS and r['position'] in POSITIONS
            assert r['status'] == 'success' and r['attempt'] == 1
            assert r['cache_mode'] == 'cold' and not r['cache_hit'] and r['timing_condition'] == 'cold'
            s = source[(r['sample_id'], r['position'])]
            assert r['reference_answers'] == s['answers'] and s['answers']
            assert r['group'] == s['group'] and r['split'] == 'test'
            assert r['L'] == s['_length'] and r['N'] == 1 and r['block_lengths'] == [r['L']]
            assert r['original_document_count'] == s['original_document_count'] == 20
            assert r['output_tokens'] == len(r['output_token_ids']) and r['output_tokens'] <= s['max_new_tokens'] == 128
            assert r['score'] in [0, 1]
            for f in METRICS:
                if f in SCORES or f == 'after_ttft_seconds':
                    continue
                assert isinstance(r[f], (int, float)) and math.isfinite(r[f]) and r[f] >= 0
            assert r['total_seconds'] >= r['ttft_seconds'] > 0
            r['_line'] = line
            r['saved_full_answer_em'] = r['score']
            r['official_first_line_em'] = r['score']
            r['after_ttft_seconds'] = r['total_seconds'] - r['ttft_seconds']
            r['_changed'] = key in corrected
            if key in corrected:
                c = corrected[key]
                assert int(c['attempts_line']) == line and int(c['attempt']) == r['attempt']
                assert (int(c['L']), int(c['N'])) == (r['L'], r['N'])
                if r['T'] is not None:
                    assert (float(c['T']), float(c['S'])) == (r['T'], r['S'])
                assert float(c['saved_score']) == float(c['full_answer_em']) == r['score'] == 1
                assert float(c['official_first_line_em']) == 0
                r['official_first_line_em'] = float(c['official_first_line_em'])
            r['_first_line'] = r['answer'].split('\n')[0].strip()
            r['_cap'] = r['output_tokens'] == s['max_new_tokens']
            r['_eos'] = r['output_token_ids'][-1] in eos
            mapped = nearest(r['L'], r['N'])
            if r['method'] == 'ape_fixed':
                assert (r['T'], r['S']) == tuple(rule['fixed'])
            if r['method'] == 'ape_dynamic':
                assert (r['T'], r['S']) == tuple(rule['cells'][f'{mapped[0]}/{mapped[1]}']) == (1, 1)
            if not r['answer'].strip():
                issues.append({'issue': 'empty generated answer, zero retained', 'sample_id': r['sample_id'], 'source_line': line})
                assert r['score'] == r['official_first_line_em'] == 0
            lookup[key] = r
            records.append(r)
    assert len(records) == 4500 and len(lookup) == 4500
    assert all(r['started_at'] > rule['frozen_at'] for r in records)
    assert set(corrected).issubset(lookup)
    assert sum(r['_changed'] for r in records) == 501
    ids = sorted(selected)
    groups = {json.dumps(source[(sid, 0)]['group'], ensure_ascii=False) for sid in ids}
    assert len(groups) == 500

    membership = []
    for sid in ids:
        variants = [source[(sid, p)] for p in POSITIONS]
        assert all(r['question'] == variants[0]['question'] and r['answers'] == variants[0]['answers'] for r in variants)
        distractors = []
        for pos, s in zip(POSITIONS, variants):
            doc_ids = s['public_document_ids']
            assert len(doc_ids) == 20 and doc_ids[pos] is None
            other = doc_ids[:pos] + doc_ids[pos + 1:]
            assert all(isinstance(v, str) and v for v in other) and len(set(other)) == 19
            distractors.append(other)
            rr = [lookup[(sid, m, pos)] for m in METHODS]
            assert len({(r['L'], r['N'], r['prefix_tokens'], r['request_tokens'], r['query_start_position']) for r in rr}) == 1
            assert all(r['query_start_position'] == r['prefix_tokens'] + r['L'] for r in rr)
            membership.append({'sample_id': sid, 'position': pos, 'question': s['question'], 'reference_answers': s['answers'], 'group': s['group'], 'L': rr[0]['L'], 'actual_N': 1, 'original_documents': 20, 'mapped_L': 4096, 'mapped_N': 4, 'dynamic_T': 1, 'dynamic_S': 1, 'public_document_ids': doc_ids, 'source_field_null_all_methods': all(r['source'] is None for r in rr), 'full_title_text_and_gold_check': 'preaudit PASS', 'distractor_ID_order_equal_all_positions': True, 'N_below_development_grid': True, 'L_below_development_grid': True, 'sequential_line': rr[0]['_line'], 'fixed_line': rr[1]['_line'], 'dynamic_line': rr[2]['_line'], 'unit': 'one question-position input; source IDs null only for verified gold; 3 paired generations'})
        assert distractors[0] == distractors[1] == distractors[2]
    csvout('membership.csv', membership)
    csvout('record_issues.csv', issues)

    fixtures = readcsv(AUDIT / 'scoring_litm_first_line_minimal_evidence.csv')
    fixture_rows = []
    for f in fixtures:
        r = lookup[(f['sample_id'], f['method'], int(f['position']))]
        assert r['answer'] == f['answer'] and r['reference_answers'] == json.loads(f['reference_answers'])
        full = fixture_em(r['answer'], r['reference_answers'])
        first = fixture_em(r['_first_line'], r['reference_answers'])
        assert full == r['saved_full_answer_em'] and first == r['official_first_line_em']
        fixture_rows.append({**f, 'verified_full_em': full, 'verified_official_first_line_em': first, 'passed': True, 'unit': 'nine original-answer fixtures, 18 official EM function evaluations'})
    assert len(fixture_rows) == 9
    csvout('official_entrypoint_fixtures.csv', fixture_rows)

    requests = []
    for r in records:
        requests.append({'sample_id': r['sample_id'], 'method': r['method'], 'position': r['position'], 'group': r['group'], 'task_key': r['key'], 'task_N': r['key']['N'], 'attempt': r['attempt'], 'status': r['status'], 'logical_labels': r['logical_labels'], 'L': r['L'], 'actual_N': r['N'], 'original_document_count': 20, 'T': r['T'], 'S': r['S'], 'answer': r['answer'], 'first_line': r['_first_line'], 'reference_answers': r['reference_answers'], 'score_corrected_by_preaudit': r['_changed'], 'first_line_empty': not bool(r['_first_line']), 'has_newline': '\n' in r['answer'], 'n_lines': len(r['answer'].split('\n')), 'post_first_line_characters': len(r['answer']) - len(r['answer'].split('\n')[0]), 'cap_hit': r['_cap'], 'eos_last': r['_eos'], 'cap_after_first_newline': r['_cap'] and '\n' in r['answer'], 'cap_unterminated_first_line': r['_cap'] and not r['_eos'] and '\n' not in r['answer'], 'output_token_ids': r['output_token_ids'], **{f: r[f] for f in METRICS}, **{f: r[f] for f in ['prefix_tokens', 'request_tokens', 'query_start_position', 'block_lengths', 'stage_calls', 'cache_mode', 'cache_hit', 'timing_condition', 'started_at', 'ended_at']}, 'source_file': 'restored/output/runs/main80-20261114T041503/attempts.jsonl', 'source_line': r['_line'], 'source': r['source'], 'unit': 'EM0-1; raw memory bytes; seconds; tokens include EOS; cached corrected score separate from saved'})
    csvout('per_request_records.csv', requests)
    csvout('scoring_changed_records.csv', [r for r in requests if r['score_corrected_by_preaudit']])
    csvout('generation_cap_records.csv', [r for r in requests if r['cap_hit']])
    csvout('bootstrap_questions.csv', [{'bootstrap_index': j, 'sample_id': sid, 'group': source[(sid, 0)]['group'], 'positions_retained': POSITIONS, 'methods_retained': METHODS, 'n_generations': 9, 'unit': 'one independent resampling question with all nine outcomes'} for j, sid in enumerate(ids)])

    values = []
    for sid in ids:
        positions = []
        for pos in POSITIONS:
            rr = {m: lookup[(sid, m, pos)] for m in METHODS}
            row = [rr[m][f] / 2**30 if METRICS[f] == 'GiB' else rr[m][f] for m, f in COLS]
            row += [rr[a][f] / rr[b][f] for a, b, f in RATIOS]
            positions.append(row)
        values.append(positions)
    a = np.array(values, dtype=float)
    rng = np.random.default_rng(SEED)
    chunks = []
    for start in range(0, BOOT, 100):
        draw = rng.integers(0, len(ids), size=(min(100, BOOT - start), len(ids)))
        chunks.append(a[draw].mean(axis=1))
    boot = np.concatenate(chunks)
    quality, paired, efficiency, performance, scoring_impact, scope_shifts = [], [], [], [], [], []
    for pos in [*POSITIONS, 'ALL']:
        av = a.mean(axis=1) if pos == 'ALL' else a[:, POSITIONS.index(pos), :]
        bv = boot.mean(axis=1) if pos == 'ALL' else boot[:, POSITIONS.index(pos), :]
        actual = a.reshape(-1, a.shape[-1]) if pos == 'ALL' else av
        common = {'position': pos, 'n_questions': 500, 'n_requests_per_method': 1500 if pos == 'ALL' else 500, 'resampling_unit': 'question, all three positions and methods jointly retained', 'seed': SEED, 'replicates': BOOT}
        for method in METHODS:
            for score in SCORES:
                j = COLS.index((method, score)); lo, hi = ci(bv[:, j])
                quality.append({**common, 'method': method, 'score_metric': score, 'em_0_1': float(av[:, j].mean()), 'em_percent': float(av[:, j].mean() * 100), 'ci_low_percent': float(lo * 100), 'ci_high_percent': float(hi * 100), 'unit': 'EM0-1, display EM x100; ALL first averages three positions within question'})
            ja, jb = [COLS.index((method, f)) for f in SCORES]
            lo, hi = ci(bv[:, ja] - bv[:, jb])
            rr = [r for r in requests if r['method'] == method and (pos == 'ALL' or r['position'] == pos)]
            scoring_impact.append({**common, 'method': method, 'changed_generations': sum(r['score_corrected_by_preaudit'] for r in rr), 'changed_questions': len({r['sample_id'] for r in rr if r['score_corrected_by_preaudit']}), 'multiline_generations': sum(r['has_newline'] for r in rr), 'first_line_empty': sum(r['first_line_empty'] for r in rr), 'official_minus_saved_pp': float((av[:, ja] - av[:, jb]).mean() * 100), 'ci_low_pp': float(lo * 100), 'ci_high_pp': float(hi * 100), 'unit': 'corrected official-first-line minus saved-full-answer EM percentage points; counts'})
            for field, unit in METRICS.items():
                if field in SCORES:
                    continue
                j = COLS.index((method, field)); lo, hi = ci(bv[:, j])
                efficiency.append({**common, 'method': method, 'metric': field, 'mean': float(actual[:, j].mean()), 'median': float(np.median(actual[:, j])), 'p90': float(np.quantile(actual[:, j], .9)), 'mean_ci_low': float(lo), 'mean_ci_high': float(hi), 'unit': unit, 'cache_condition': 'cold miss; no attention observer; document encoding included'})
        for ma, mb in PAIRS:
            for score in SCORES:
                ja, jb = [COLS.index((m, score)) for m in [ma, mb]]
                delta = av[:, jb] - av[:, ja]; lo, hi = ci(bv[:, jb] - bv[:, ja])
                paired.append({**common, 'reference_method': ma, 'comparison_method': mb, 'score_metric': score, 'delta_0_1': float(delta.mean()), 'delta_pp': float(delta.mean() * 100), 'ci_low_pp': float(lo * 100), 'ci_high_pp': float(hi * 100), **win_tie_loss(delta), 'win_tie_loss_unit': 'question mean across three positions for ALL; one question at named position otherwise', 'unit': 'EM percentage points; W/T/L denominator500'})
            ia, ib = [COLS.index((m, SCORES[0])) for m in [ma, mb]]
            fa, fb = [COLS.index((m, SCORES[1])) for m in [ma, mb]]
            shift_boot = (bv[:, ib] - bv[:, ia]) - (bv[:, fb] - bv[:, fa])
            sl, sh = ci(shift_boot)
            scope_shifts.append({**common, 'reference_method': ma, 'comparison_method': mb, 'saved_full_pair_delta_pp': float((av[:, fb] - av[:, fa]).mean() * 100), 'official_first_line_pair_delta_pp': float((av[:, ib] - av[:, ia]).mean() * 100), 'official_minus_saved_pair_delta_pp': float(((av[:, ib] - av[:, ia]) - (av[:, fb] - av[:, fa])).mean() * 100), 'shift_ci_low_pp': float(sl * 100), 'shift_ci_high_pp': float(sh * 100), 'unit': 'difference of paired method effects under official versus saved score scope; EMpp'})
            for field in ['ttft_seconds', 'total_seconds']:
                ia, ib = [COLS.index((m, field)) for m in [ma, mb]]
                ir = len(COLS) + RATIOS.index((ma, mb, field))
                dl, dh = ci(bv[:, ib] - bv[:, ia]); rl, rh = ci(bv[:, ir]); ml, mh = ci(bv[:, ia] / bv[:, ib])
                performance.append({**common, 'reference_method': ma, 'comparison_method': mb, 'metric': field, 'reference_mean_s': float(actual[:, ia].mean()), 'comparison_mean_s': float(actual[:, ib].mean()), 'delta_comparison_minus_reference_s': float((actual[:, ib] - actual[:, ia]).mean()), 'delta_ci_low_s': float(dl), 'delta_ci_high_s': float(dh), 'paired_speedup_mean': float(actual[:, ir].mean()), 'paired_speedup_median': float(np.median(actual[:, ir])), 'paired_speedup_p90': float(np.quantile(actual[:, ir], .9)), 'speedup_mean_ci_low': float(rl), 'speedup_mean_ci_high': float(rh), 'ratio_of_means': float(actual[:, ia].mean() / actual[:, ib].mean()), 'ratio_of_means_ci_low': float(ml), 'ratio_of_means_ci_high': float(mh), 'unit': 's; speedup reference/comparison, >1 comparison faster; median/P90 across actual requests'})

    coefficients = {'middle_minus_ends_mean': np.array([-.5, 1, -.5]), 'last_minus_first': np.array([-1, 0, 1]), 'middle_minus_first': np.array([-1, 1, 0]), 'last_minus_middle': np.array([0, -1, 1])}
    contrasts, paired_contrasts, question_scores = [], [], []
    for score in SCORES:
        for name, coef in coefficients.items():
            for method in METHODS:
                j = COLS.index((method, score)); point = a[:, :, j] @ coef; bc = boot[:, :, j] @ coef; lo, hi = ci(bc)
                contrasts.append({'score_metric': score, 'method': method, 'contrast': name, 'coefficients_positions_0_9_19': coef.tolist(), 'n_questions': 500, 'contrast_pp': float(point.mean() * 100), 'ci_low_pp': float(lo * 100), 'ci_high_pp': float(hi * 100), 'unit': 'EM percentage points; within-question position contrast', 'seed': SEED, 'replicates': BOOT})
            for ma, mb in PAIRS:
                ja, jb = [COLS.index((m, score)) for m in [ma, mb]]
                point = (a[:, :, jb] - a[:, :, ja]) @ coef
                bc = (boot[:, :, jb] - boot[:, :, ja]) @ coef; lo, hi = ci(bc)
                paired_contrasts.append({'score_metric': score, 'reference_method': ma, 'comparison_method': mb, 'contrast': name, 'n_questions': 500, 'delta_contrast_pp': float(point.mean() * 100), 'ci_low_pp': float(lo * 100), 'ci_high_pp': float(hi * 100), **win_tie_loss(point), 'unit': 'comparison minus reference within-question contrast, EMpp', 'seed': SEED, 'replicates': BOOT})
    for i, sid in enumerate(ids):
        row = {'sample_id': sid, 'question': source[(sid, 0)]['question'], 'reference_answers': source[(sid, 0)]['answers'], 'group': source[(sid, 0)]['group']}
        for method in METHODS:
            for score in SCORES:
                j = COLS.index((method, score))
                row.update({f'{method}_{score}_position_{p}': float(a[i, k, j]) for k, p in enumerate(POSITIONS)})
                row[f'{method}_{score}_mean_three_positions'] = float(a[i, :, j].mean())
        row['dynamic_minus_fixed_mean_em_0_1'] = row['ape_dynamic_official_first_line_em_mean_three_positions'] - row['ape_fixed_official_first_line_em_mean_three_positions']
        row['dynamic_fixed_each_position_equal'] = all(lookup[(sid, 'ape_dynamic', p)]['official_first_line_em'] == lookup[(sid, 'ape_fixed', p)]['official_first_line_em'] for p in POSITIONS)
        row['unit'] = 'one question, all3 positions and3 methods; official/saved EM0-1'
        question_scores.append(row)
    for name, data in [('quality.csv', quality), ('quality_paired.csv', paired), ('efficiency_descriptive.csv', efficiency), ('efficiency_paired.csv', performance), ('scoring_scope_impact.csv', scoring_impact), ('method_pair_scoring_scope_shift.csv', scope_shifts), ('position_contrasts.csv', contrasts), ('paired_position_contrasts.csv', paired_contrasts), ('per_question_scores.csv', question_scores)]:
        csvout(name, data)

    limits, cap_conditioned, output_match, length_groups = [], [], [], []
    for pos in [*POSITIONS, 'ALL']:
        for method in METHODS:
            rr = [r for r in requests if r['method'] == method and (pos == 'ALL' or r['position'] == pos)]
            limits.append({'position': pos, 'method': method, 'n_requests': len(rr), 'n_questions': len({r['sample_id'] for r in rr}), 'max_new_tokens': 128, 'cap_hits': sum(r['cap_hit'] for r in rr), 'cap_hit_fraction': np.mean([r['cap_hit'] for r in rr]), 'eos_last_count': sum(r['eos_last'] for r in rr), 'eos_at_cap_count': sum(r['eos_last'] and r['cap_hit'] for r in rr), 'cap_after_first_newline_count': sum(r['cap_after_first_newline'] for r in rr), 'cap_unterminated_first_line_count': sum(r['cap_unterminated_first_line'] for r in rr), 'multiline_count': sum(r['has_newline'] for r in rr), 'first_line_empty_count': sum(r['first_line_empty'] for r in rr), 'score_changed_at_cap_count': sum(r['cap_hit'] and r['score_corrected_by_preaudit'] for r in rr), 'unit': 'actual requests; rates0-1; EOS included at cap'})
            for cap in [False, True]:
                chosen = [r for r in rr if r['cap_hit'] == cap]
                if chosen:
                    cap_conditioned.append({'position': pos, 'method': method, 'cap_hit': cap, 'n_requests': len(chosen), 'n_questions': len({r['sample_id'] for r in chosen}), 'denominator_requests': len(rr), 'official_em_percent': np.mean([r['official_first_line_em'] for r in chosen]) * 100, 'saved_full_em_percent': np.mean([r['saved_full_answer_em'] for r in chosen]) * 100, 'mean_total_s': np.mean([r['total_seconds'] for r in chosen]), 'mean_ttft_s': np.mean([r['ttft_seconds'] for r in chosen]), 'mean_output_tokens': np.mean([r['output_tokens'] for r in chosen]), 'unit': 'descriptive actual-generation groups; EMx100/s/tokens'})
        for ma, mb in PAIRS:
            positions = POSITIONS if pos == 'ALL' else [pos]
            rr = [(lookup[(sid, ma, p)], lookup[(sid, mb, p)]) for sid in ids for p in positions]
            output_match.append({'position': pos, 'reference_method': ma, 'comparison_method': mb, 'n_condition_pairs': len(rr), 'n_questions': 500, 'same_parameters': sum((x['T'], x['S']) == (y['T'], y['S']) for x, y in rr), 'same_token_sequences': sum(x['output_token_ids'] == y['output_token_ids'] for x, y in rr), 'same_official_scores': sum(x['official_first_line_em'] == y['official_first_line_em'] for x, y in rr), 'unit': 'same question/position request pairs; output tokens include EOS'})
    for ma, mb in PAIRS:
        rr = [(lookup[(sid, ma, p)], lookup[(sid, mb, p)]) for sid in ids for p in POSITIONS]
        for label, predicate in [('all', lambda x,y: True), ('same_tokens', lambda x,y: x['output_token_ids']==y['output_token_ids']), ('different_tokens', lambda x,y: x['output_token_ids']!=y['output_token_ids']), ('comparison_longer', lambda x,y: y['output_tokens']>x['output_tokens']), ('comparison_shorter', lambda x,y: y['output_tokens']<x['output_tokens'])]:
            chosen = [(x,y) for x,y in rr if predicate(x,y)]
            length_groups.append({'reference_method': ma, 'comparison_method': mb, 'group': label, 'n_condition_pairs': len(chosen), 'n_questions': len({x['sample_id'] for x,y in chosen}), 'mean_reference_tokens': np.mean([x['output_tokens'] for x,y in chosen]), 'mean_comparison_tokens': np.mean([y['output_tokens'] for x,y in chosen]), 'mean_total_delta_s': np.mean([y['total_seconds']-x['total_seconds'] for x,y in chosen]), 'mean_ttft_delta_s': np.mean([y['ttft_seconds']-x['ttft_seconds'] for x,y in chosen]), 'official_em_delta_pp': np.mean([y['official_first_line_em']-x['official_first_line_em'] for x,y in chosen])*100, 'unit': 'descriptive question-position pairs, not independent1500 resampling; s/tokens/EMpp'})
    csvout('generation_limits.csv', limits)
    csvout('generation_cap_conditioned.csv', cap_conditioned)
    csvout('output_equivalence.csv', output_match)
    csvout('paired_output_length_groups.csv', length_groups)

    selected_cases, selected_positions, case_counts = [], [], []
    for label, predicate, sign in [('改善',lambda v:v>1e-12,-1),('退化',lambda v:v< -1e-12,1),('持平',lambda v:abs(v)<=1e-12,1)]:
        chosen = sorted([r for r in question_scores if predicate(r['dynamic_minus_fixed_mean_em_0_1'])],key=lambda r:(sign*r['dynamic_minus_fixed_mean_em_0_1'],r['sample_id']))
        case_counts.append({'case_type': label, 'n_questions': len(chosen), 'denominator': 500, 'unit': 'question mean official EM over all3 positions'})
        if chosen:
            selected_cases.append({'case_type': label, **chosen[0], 'selection_rule': 'max positive/negative mean3-position dynamic-minus-fixed; zero/ties smallest sample_id'})
            for p in POSITIONS:
                x,y = [lookup[(chosen[0]['sample_id'],m,p)] for m in ['ape_fixed','ape_dynamic']]
                selected_positions.append({'case_type': label, 'sample_id': x['sample_id'], 'position': p, 'question': source[(x['sample_id'],p)]['question'], 'fixed_T':x['T'],'fixed_S':x['S'],'dynamic_T':y['T'],'dynamic_S':y['S'],'fixed_original_answer':x['answer'],'dynamic_original_answer':y['answer'],'fixed_first_line':x['_first_line'],'dynamic_first_line':y['_first_line'],'reference_answers':x['reference_answers'],'fixed_official_em':x['official_first_line_em'],'dynamic_official_em':y['official_first_line_em'],'fixed_saved_full_em':x['score'],'dynamic_saved_full_em':y['score'],'fixed_output_tokens':x['output_tokens'],'dynamic_output_tokens':y['output_tokens'],'fixed_source_line':x['_line'],'dynamic_source_line':y['_line'],'source_file':'restored/output/runs/main80-20261114T041503/attempts.jsonl','unit':'all3 positions retained for selected question; EM0-1; tokens'})
    csvout('selected_cases.csv', selected_cases)
    csvout('selected_case_positions.csv', selected_positions)
    csvout('case_type_counts.csv', case_counts)

    original = read(RUN / 'summary.json')['litm_paired']
    assert original['selected_n'] == original['paired_n'] == 500 and set(original['sample_ids']) == selected
    checks = []
    for m in METHODS:
        for p in [*POSITIONS,'ALL']:
            observed = next(r['em_0_1'] for r in quality if r['method']==m and r['position']==p and r['score_metric']=='saved_full_answer_em')
            saved = original['methods'][m]['overall_accuracy'] if p=='ALL' else original['methods'][m]['position_accuracy'][str(p)]
            checks.append({'method':m,'position':p,'score_scope':'saved full answer only','original_summary_em':saved,'current_full_em':observed,'difference':observed-saved,'unit':'EM0-1; official-first-line results kept separate'})
    assert max(abs(r['difference']) for r in checks) < 1e-12
    csvout('original_summary_crosscheck.csv', checks)
    csvout('model_load_workers.csv', [r for r in readcsv(AUDIT/'runtime_workers.csv') if r['stage']=='litm'])
    csvout('server_environment.csv', readcsv(AUDIT/'runtime_environment.csv'))
    (OUT/'sources/original_frozen_rule.json').write_text(json.dumps(rule,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'sources/input_sources.json').write_text(json.dumps({'raw_position_files':frozen['sources']['litm'],'prompt':'restored/datasets/raw-main80/references/litm_qa.prompt','frozen':'snapshot/datasets/frozen-main80/litm.json','source_join':'sample_id/position -> frozen question -> original per-position question; title/text multiset audited','cached_corrections':'pc-analysis-output/stages/01_audit/tables/scoring_litm_first_line_changed_IDs.csv'},ensure_ascii=False,indent=2),encoding='utf-8')
    summary = {'experiment_stage':7,'status':'COMPLETED','research_status':'SUBMITTED','n_questions':500,'n_position_inputs':1500,'n_generation_records':4500,'main_score':'official_first_line_em','main_quality':[r for r in quality if r['score_metric']==SCORES[0]],'dynamic_fixed_pairs':[r for r in paired if r['score_metric']==SCORES[0] and r['reference_method']=='ape_fixed'],'corrected_labels_reused':501,'corrected_question_ids':len({r['sample_id'] for r in changes}),'newly_rescored_generation_records':0,'fixture_pairs_verified':9,'fixture_em_function_calls':18,'all_actual_N':1,'all_original_document_count':20,'L_min':min(r['L'] for r in records),'L_max':max(r['L'] for r in records),'dynamic_parameters':[1,1],'distractor_order_same_questions':500,'cap_hits':sum(r['_cap'] for r in records),'eos_last_count':sum(r['_eos'] for r in records),'eos_at_cap_count':sum(r['_eos'] and r['_cap'] for r in records),'empty_first_lines':sum(not r['_first_line'] for r in records),'normalized_empty_reference_questions':sum(any(not normalize(v) for v in source[(sid,0)]['answers']) for sid in ids),'seed':SEED,'replicates':BOOT,'rule_modified':False,'next_stage_executed':False}
    (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'STAGE_STATUS.json').write_text(json.dumps({k:v for k,v in summary.items() if k not in ['main_quality','dynamic_fixed_pairs']},ensure_ascii=False,indent=2),encoding='utf-8')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    environment = {'analysis_commit':commit,'reference_commit':REFERENCE,'changes_from_reference':subprocess.check_output(['git','diff','--name-only',f'{REFERENCE}..HEAD'],cwd=ROOT,text=True).splitlines(),'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'matplotlib':version('matplotlib'),'regex':version('regex'),'new_dependencies':[],'seed':SEED,'replicates':BOOT,'rng':'default_rng(20261116); sorted500 sample_ids; 100-replicate chunks; 500 question draws with replacement per replicate','bootstrap':'paired question, all3 positions and3 methods retained; percentile2.5/97.5','overall_EM':'question-wise mean of three positions, then mean500 questions','efficiency_quantiles':'actual500 requests/position or1500 requests/overall; CI always resamples500 questions jointly','scoring':'reuse audited correction table; verify9 original-answer pairs using official first-line and full-answer function','all_cache_temp_output':str(OUT)}
    (OUT/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'stage':7,'main_quality':[{k:r[k] for k in ['method','position','em_percent','ci_low_percent','ci_high_percent']} for r in quality if r['score_metric']==SCORES[0]],'primary_dynamic_fixed':summary['dynamic_fixed_pairs'],'cap_hits':summary['cap_hits'],'research_status':'SUBMITTED'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
