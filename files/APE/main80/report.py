"""Truthful partial exports; corpus BLEU and quality comparisons join by ID."""
import csv
from collections import defaultdict
from statistics import mean, median

from .common import Ledger, inside, now, write_json
from .metrics import corpus_doc2dial


def export(directory, factory, limits, expected):
    directory = inside(directory)
    ledger = Ledger(directory / 'attempts.jsonl')
    selected = factory.selected_ids(limits)
    successes = list(ledger.success.values())
    # Empty outputs are failed requests, but their official zero score must not
    # disappear from a quality comparison merely because another method answered.
    outcomes = list(ledger.scored.values())
    empty_failures = [r for r in outcomes if r['status'] != 'success']
    groups = defaultdict(list)
    retained = []
    for r in outcomes:
        stage = r['key']['stage']
        scope = {'longbench_final': 'test'}.get(stage, stage)
        if scope in selected and r['sample_id'] not in selected[scope]:
            continue
        groups[(stage, r['dataset'], r['method'], r.get('position'), r.get('cache_mode'))].append(r)
        retained.append(r)
    quality = []
    for (stage, dataset, method, position, cache), rows in groups.items():
        entry = {'stage': stage, 'dataset': dataset, 'method': method, 'position': position, 'cache_mode': cache,
                 'n': len(rows), 'mean_sample_score': mean(r['score'] for r in rows),
                 'scope': 'available valid scoring outcomes; empty generated responses retain official zero and failed status; compare methods using paired sections'}
        if dataset == 'doc2dial':
            refs = {r['sample_id']: r['answers'][0] for r in factory.tables['doc2dial'] if r['sample_id'] in selected['doc2dial']}
            complete_refs = {r['sample_id']: refs[r['sample_id']] for r in rows}
            entry['corpus_bleu'] = corpus_doc2dial(rows, complete_refs)
        entry['mean_ttft_seconds'] = mean(r['ttft_seconds'] for r in rows)
        entry['median_total_seconds'] = median(r['total_seconds'] for r in rows)
        quality.append(entry)
    # Baseline on full600 and the final test subset must never be conflated.
    baseline_test = []
    for dataset in ('qasper', 'hotpotqa', 'musique'):
        for method in ('sequential', 'pcw', 'ape_default', 'ape_fixed', 'ape_dynamic'):
            stages = {'baseline'} if method in ('sequential', 'pcw', 'ape_default') else {'longbench_final'}
            rows = [r for r in retained if r['key']['stage'] in stages and r['dataset'] == dataset and r['method'] == method and r['sample_id'] in selected['test']]
            baseline_test.append({'dataset': dataset, 'method': method, 'n': len(rows), 'selected_test_ids': [r['sample_id'] for r in rows], 'mean_f1': mean(r['score'] for r in rows) if rows else None})
    paired = []
    for dataset in ('qasper', 'hotpotqa', 'musique'):
        rows_by_method = {m: {r['sample_id']: r for r in retained if r['dataset'] == dataset and r['method'] == m and r['key']['stage'] in {'baseline', 'longbench_final'} and r['sample_id'] in selected['test']}
                          for m in ('sequential', 'pcw', 'ape_default', 'ape_fixed', 'ape_dynamic')}
        ids = sorted(set.intersection(*(set(v) for v in rows_by_method.values())))
        paired.append({'dataset': dataset, 'paired_n': len(ids), 'sample_ids': ids,
                       'mean_f1': {m: mean(rows_by_method[m][i]['score'] for i in ids) if ids else None for m in rows_by_method}})
    pending = [job['key'] for job in expected if not ledger.done(job['key'])]
    errors = [r for r in ledger.records if r['status'] != 'success']
    litm_conditions = [(m, p) for m in ('sequential', 'ape_fixed', 'ape_dynamic') for p in (0, 9, 19)]
    litm_by = {(m, p): {r['sample_id']: r for r in retained if r['dataset'] == 'litm' and r['method'] == m and r['position'] == p} for m, p in litm_conditions}
    litm_ids = sorted(set(selected['litm']).intersection(*(set(v) for v in litm_by.values())))
    litm_paired = {'selected_n': len(selected['litm']), 'paired_n': len(litm_ids), 'sample_ids': litm_ids,
        'excluded_ids': sorted(set(selected['litm']) - set(litm_ids)), 'methods': {}}
    for method in ('sequential', 'ape_fixed', 'ape_dynamic'):
        litm_paired['methods'][method] = {'position_accuracy': {str(p): mean(litm_by[method, p][i]['score'] for i in litm_ids) if litm_ids else None for p in (0, 9, 19)},
            'overall_accuracy': mean(litm_by[method, p][i]['score'] for i in litm_ids for p in (0, 9, 19)) if litm_ids else None}
    doc_conditions = [(m, c) for m in ('sequential', 'ape_fixed', 'ape_dynamic') for c in ('cold', 'hot')]
    doc_by = {(m, c): {r['sample_id']: r for r in retained if r['dataset'] == 'doc2dial' and r['method'] == m and r['cache_mode'] == c} for m, c in doc_conditions}
    doc_ids = sorted(set(selected['doc2dial']).intersection(*(set(v) for v in doc_by.values())))
    doc_paired = {'selected_n': len(selected['doc2dial']), 'paired_n': len(doc_ids), 'sample_ids': doc_ids,
        'excluded_ids': sorted(set(selected['doc2dial']) - set(doc_ids)), 'conditions': []}
    refs = {r['sample_id']: r['answers'][0] for r in factory.tables['doc2dial'] if r['sample_id'] in doc_ids}
    for method, cache in doc_conditions:
        rows = [doc_by[method, cache][i] for i in doc_ids]
        doc_paired['conditions'].append({'method': method, 'cache_mode': cache,
            'corpus_bleu': corpus_doc2dial(rows, refs) if rows else None,
            'mean_ttft_seconds': mean(r['ttft_seconds'] for r in rows) if rows else None,
            'mean_total_seconds': mean(r['total_seconds'] for r in rows) if rows else None,
            'mean_resume_cache_rebuild_seconds': mean(r.get('resume_cache_rebuild_seconds', 0) for r in rows) if rows else None})
    write_json(directory / 'incomplete.json', {'at': now(), 'pending_jobs': pending, 'failed_attempts': errors, 'incomplete_lines': ledger.incomplete})
    result = {'at': now(), 'status': 'SUBMITTED' if not pending else 'PARTIAL', 'requires_PLANNER_review': True,
              'quality': quality, 'longbench_test_by_method': baseline_test, 'longbench_five_method_paired': paired,
              'litm_paired': litm_paired, 'doc2dial_paired': doc_paired,
              'successful_records': len(successes), 'retained_analysis_scope_records': len(retained),
              'failed_empty_responses_scored_zero': len(empty_failures),
              'outside_current_sample_scope': len(outcomes) - len(retained), 'pending': len(pending), 'failed_attempts': len(errors),
              'limits': limits, 'scores_are_observations_not_claimed_improvements': True}
    write_json(directory / 'summary.json', result)
    columns = ['sample_id', 'dataset', 'method', 'split', 'L', 'N', 'T', 'S', 'position', 'cache_mode', 'score', 'ttft_seconds', 'total_seconds', 'output_tokens', 'peak_allocated_bytes', 'peak_reserved_bytes', 'status', 'attempt']
    with (directory / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ledger.records)
    return result
