"""Stage 1 only: CPU audit of configuration, runtime records and stop legality.

No benchmark re-scoring, efficiency ranking, confidence interval, attention plot,
GPU/model call, dependency installation or file digest is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

CAMPAIGN = 'main80-20261114T041503'
STAGES = ['baseline', 'explore', 'validation', 'length_check', 'attention', 'longbench_final', 'litm', 'doc2dial']


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def stream(path):
    with path.open(encoding='utf-8-sig') as handle:
        for line, text in enumerate(handle, 1):
            if text.strip():
                yield line, json.loads(text)


def identity(key):
    return json.dumps(key, sort_keys=True, separators=(',', ':'))


def write_csv(path, rows, default_fields=('issue',)):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(default_fields)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md(rows, fields):
    table = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |']
    for row in rows:
        table.append('| ' + ' | '.join(str(row.get(field, '')).replace('|', '/') for field in fields) + ' |')
    return '\n'.join(table)


def bj(text):
    return datetime.fromisoformat(text).astimezone(timezone(timedelta(hours=8))).isoformat(timespec='seconds') if text else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = repo / 'pc-analysis-output' / 'stages' / '01_audit'
    tables, reports = output / 'tables', output / 'reports'
    for folder in (tables, reports, output / 'tmp', output / 'cache'):
        folder.mkdir(parents=True, exist_ok=True)
    for name in ('TMP', 'TEMP', 'TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
        os.environ[name] = str(output / ('cache' if name in ('XDG_CACHE_HOME', 'MPLCONFIGDIR') else 'tmp'))
    restored = repo / 'restored'
    run = restored / 'output' / 'runs' / CAMPAIGN
    snapshot = run / 'snapshot'
    attempts = [(line, row) for line, row in stream(run / 'attempts.jsonl')]
    records = {identity(row['key']): row for _, row in attempts}
    events = list(stream(run / 'events.jsonl'))
    config = load(run / 'config.json')
    model_config = load(snapshot / 'models' / 'Llama-3.1-8B-Instruct' / 'config.json')
    generation = load(snapshot / 'models' / 'Llama-3.1-8B-Instruct' / 'generation_config.json')
    env = load(restored / 'logs' / 'runs' / '20261113-main80-prep' / 'environment.json')
    review = load(snapshot / 'logs' / 'runs' / '20261113-main80-prep' / 'review.json')
    model_source = load(restored / 'logs' / 'runs' / '20261113-main80-assets' / 'model-manifest.json')
    model_stability = load(restored / 'logs' / 'runs' / '20261113-main80-assets' / 'model-source-stability.json')
    rule, candidates, delivery, state = [load(run / (name + '.json')) for name in ('rule', 'candidates', 'delivery', 'state')]
    eos = generation['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    sources = []

    def ref(relative, marker):
        path = restored / relative
        for line, text in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if marker in text:
                value = f'restored/{relative}:{line}'
                sources.append({'source': value, 'marker': marker, 'text': text.strip()})
                return value
        raise ValueError(f'expected source marker missing: {relative}: {marker}')

    method_rel = f'output/runs/{CAMPAIGN}/snapshot/APE/main80/methods.py'
    worker_rel = f'output/runs/{CAMPAIGN}/snapshot/APE/main80/worker.py'
    refs = {
        'load': ref(method_rel, 'def load(cls'),
        'prepare': ref(method_rel, 'def prepare(self'),
        'chunks': ref(method_rel, 'def chunks('),
        'encode': ref(method_rel, 'def encode(self'),
        'merge_pcw': ref(method_rel, 'def merge_pcw('),
        'start': ref(method_rel, 'def run(self'),
        'cache': ref(method_rel, 'encoding_param ='),
        'request': ref(method_rel, 'History/question are intentionally'),
        'greedy': ref(method_rel, 'token = int(logits.argmax'),
        'stops': ref(method_rel, 'stops = set(eos'),
        'total': ref(method_rel, 'total = time.perf_counter() - started'),
        'memory': ref(method_rel, "'peak_allocated_bytes':"),
        'hot': ref(worker_rel, "if cold['output_token_ids'] != result['output_token_ids']"),
        'model_event': ref(worker_rel, "'event': 'model_loaded'"),
        'attention': ref(worker_rel, 'with AttentionCapture('),
        'restore': ref(worker_rel, 'def recovery_timing('),
        'device': ref(f'output/runs/{CAMPAIGN}/snapshot/scripts/with_env.sh', 'CUDA_VISIBLE_DEVICES=0'),
        'hardware_plan': ref('docs/EXPERIMENT_PLAN.md', 'RTX 4090'),
        'template_doc2dial': ref(f'output/runs/{CAMPAIGN}/snapshot/APE/main80/freeze.py', "'template_adaptation'"),
    }

    issues = []

    def issue(kind, line='', sample_id='', field='', value='', source='attempts.jsonl'):
        issues.append({'issue': kind, 'source': source, 'source_line': line, 'sample_id': sample_id, 'field': field, 'value': str(value)[:400]})

    # Worker bundles are the concrete scheduled jobs, including their data indices.
    jobs = {}
    bundles = []
    for path in sorted((run / 'workers').glob('*.bundle.json')):
        bundle = load(path)
        bundles.append((path.name, bundle))
        for job in bundle['jobs']:
            kid = identity(job['key'])
            if kid in jobs:
                issue('duplicate scheduled key', source=path.name, value=kid)
            jobs[kid] = {**job, 'bundle': path.name}
    row_metadata = {}
    dataset_templates = []
    for table in ('longbench', 'development', 'litm', 'doc2dial'):
        rows = load(snapshot / 'datasets' / 'frozen-main80' / (table + '.json'))
        by_dataset = defaultdict(list)
        for index, row in enumerate(rows):
            row_metadata[(table, index)] = {k: row.get(k) for k in ('sample_id', 'dataset', 'position', 'L', '_length', 'max_new_tokens', 'template_adaptation')}
            by_dataset[row['dataset']].append(row)
        for dataset, part in sorted(by_dataset.items()):
            dataset_templates.append({
                'table': table, 'dataset': dataset, 'frozen_rows': len(part),
                'max_new_tokens': ','.join(str(x) for x in sorted(set(r['max_new_tokens'] for r in part))),
                'nonempty_prefix': sum(bool(r.get('prefix')) for r in part),
                'nonempty_suffix': sum(bool(r.get('suffix')) for r in part),
                'prefix_variants': len(set(r['prefix'] for r in part)),
                'document_tokens_rows': sum('document_tokens' in r for r in part),
                'document_text_rows': sum('document' in r for r in part),
                'adaptation': '; '.join(sorted(set(r.get('template_adaptation', '') for r in part))),
                'source': f'snapshot/datasets/frozen-main80/{table}.json',
            })
    same_input_groups = defaultdict(set)
    for kid, job in jobs.items():
        key = job['key']
        meta = row_metadata[(job['table'], job['index'])]
        for field in ('sample_id', 'max_new_tokens'):
            value = key['sample_id'] if field == 'sample_id' else job[field]
            if value != meta[field]:
                issue('job/frozen-row mismatch', sample_id=key['sample_id'], field=field, value=value, source=job['bundle'])
        # Baseline and final LongBench share the same frozen row; cache/parameters do not alter it.
        group = (job['table'], key['sample_id'], key['L'], job['N'], key['position'])
        same_input_groups[group].add(job['index'])
    for group, indices in same_input_groups.items():
        if len(indices) != 1:
            issue('method/template row index mismatch', sample_id=group[1], value=sorted(indices), source='workers/*.bundle.json')
    write_csv(tables / 'runtime_templates.csv', dataset_templates)

    numeric_fields = ['L', 'N', 'original_document_count', 'query_start_position', 'output_tokens', 'prefix_tokens', 'request_tokens', 'ttft_seconds', 'total_seconds',
                      'cache_build_seconds', 'cache_get_seconds', 'dynamic_selection_seconds', 'resume_cache_rebuild_seconds',
                      'peak_allocated_bytes', 'peak_reserved_bytes']
    fields = numeric_fields + ['T', 'S', 'output_token_ids', 'block_lengths', 'window_order', 'cache_mode', 'cache_hit', 'timing_condition',
                               'stage_calls', 'started_at', 'ended_at', 'attempt', 'logical_labels']
    availability = defaultdict(Counter)
    groups, cache_groups = defaultdict(list), defaultdict(Counter)
    req_starts, req_ends = Counter(), Counter()
    for _, event in events:
        if event['event'] in ('request_started', 'request_ended'):
            counter = req_starts if event['event'] == 'request_started' else req_ends
            counter[(identity(event['key']), event['attempt'])] += 1
    max_output = 0
    cap_equal = cap_eos = tokens_valid = output_length_valid = request_event_pairs = 0
    for line, record in attempts:
        key = record['key']
        kid = identity(key)
        job = jobs.get(kid)
        stage, dataset, method, cache = key['stage'], record['dataset'], record['method'], record['cache_mode']
        group = (stage, dataset, method, cache)
        for field in fields:
            a = availability[(stage, field)]
            a['denominator'] += 1
            a['present'] += field in record
            a['nonnull'] += record.get(field) is not None
            if field in numeric_fields and record.get(field) is not None:
                value = record[field]
                valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
                a['valid_nonnegative_finite'] += valid
                if not valid:
                    issue('invalid numeric field', line, record['sample_id'], field, value)
        if job is None:
            issue('record missing worker job', line, record['sample_id'], value=kid)
            continue
        cap = job['max_new_tokens']
        tokens, length = record.get('output_token_ids', []), record.get('output_tokens')
        array_valid = isinstance(tokens, list) and all(isinstance(x, int) and 0 <= x < model_config['vocab_size'] for x in tokens)
        tokens_valid += array_valid
        length_valid = length == len(tokens) and isinstance(length, int) and 1 <= length <= cap
        output_length_valid += length_valid
        if not array_valid or not length_valid:
            issue('invalid output tokens/length', line, record['sample_id'], 'output_tokens', length)
        has_eos = bool(tokens and tokens[-1] in eos)
        hit = length == cap
        cap_equal += hit
        cap_eos += hit and has_eos
        max_output = max(max_output, length)
        if any(x in eos for x in tokens[:-1]):
            issue('EOS before final output token', line, record['sample_id'])
        if not hit and not has_eos:
            issue('generation stopped below cap without EOS', line, record['sample_id'])
        if record['L'] != sum(record['block_lengths']) or record['N'] != len(record['block_lengths']) or min(record['block_lengths']) <= 0:
            issue('L/N/block_lengths inconsistent', line, record['sample_id'])
        if record['L'] != key['L'] or (key['N'] is not None and record['N'] != key['N']):
            issue('actual L/N differ from task request', line, record['sample_id'])
        if record['ttft_seconds'] > record['total_seconds']:
            issue('TTFT exceeds total', line, record['sample_id'])
        if record['peak_allocated_bytes'] > record['peak_reserved_bytes']:
            issue('allocated exceeds reserved', line, record['sample_id'])
        # PCW reattaches prefix[1:] to its request; avoid counting that prefix twice.
        full_input = record['L'] + record['request_tokens'] + (1 if method == 'pcw' else record['prefix_tokens'])
        if full_input + cap > model_config['max_position_embeddings']:
            issue('original full input plus cap exceeds model window', line, record['sample_id'])
        event_id = (kid, record['attempt'])
        event_valid = req_starts[event_id] == 1 and req_ends[event_id] == 1
        request_event_pairs += event_valid
        if not event_valid:
            issue('request event pair incomplete/duplicate', line, record['sample_id'], value=(req_starts[event_id], req_ends[event_id]))
        expected_timing = {'cold': 'cold', 'hot': 'application_GPU_KV', 'search': 'search_immutable_KV'}[cache]
        if record['timing_condition'] != expected_timing and not (cache == 'hot' and record['timing_condition'] == 'recovered_application_GPU_KV'):
            issue('cache/timing condition mismatch', line, record['sample_id'])
        if cache == 'cold' and record['cache_hit']:
            issue('cold request incorrectly marks hit', line, record['sample_id'])
        if cache == 'hot' and (not record['cache_hit'] or record['cache_build_seconds'] != 0):
            issue('hot miss or nonzero in-request build', line, record['sample_id'])
        if method.startswith('ape_'):
            if any(record.get(field) not in (0.5, 0.7, 0.9, 1.0) for field in ('T', 'S')):
                issue('APE T/S outside executed grid', line, record['sample_id'])
            expected_query_calls = length
            if record['stage_calls']['query'] != expected_query_calls:
                issue('APE query call count differs from generated token count', line, record['sample_id'])
        elif record.get('T') is not None or record.get('S') is not None:
            issue('T/S recorded for stock sequential/PCW', line, record['sample_id'])
        groups[group].append({'cap': cap, 'length': length, 'hit': hit, 'eos': has_eos})
        layer = (stage, method, cache, record['timing_condition'])
        cache_groups[layer].update(n=1, cache_hits=int(record['cache_hit']), build_zero=int(record['cache_build_seconds'] == 0),
                                   query_calls_present=int('query' in record['stage_calls']), recovery_nonzero=int(record.get('resume_cache_rebuild_seconds', 0) > 0))
    field_rows = []
    for (stage, field), counts in sorted(availability.items()):
        field_rows.append({'stage': stage, 'field': field, **counts, 'unit': 'record count; denominator=records in stage'})
    write_csv(tables / 'runtime_field_availability.csv', field_rows)
    generation_rows = []
    for group, part in sorted(groups.items()):
        generation_rows.append(dict(zip(('stage', 'dataset', 'method', 'cache_mode'), group)) | {
            'n': len(part), 'max_new_tokens': ','.join(str(x) for x in sorted(set(r['cap'] for r in part))),
            'cap_hit_count': sum(r['hit'] for r in part), 'eos_last_count': sum(r['eos'] for r in part),
            'eos_at_cap_count': sum(r['eos'] and r['hit'] for r in part),
            'output_tokens_min': min(r['length'] for r in part), 'output_tokens_max': max(r['length'] for r in part),
            'denominator': len(part), 'unit': 'record count; output_tokens includes final EOS',
        })
    write_csv(tables / 'runtime_generation_limits.csv', generation_rows)
    write_csv(tables / 'runtime_cache_layers.csv', [dict(zip(('stage', 'method', 'cache_mode', 'timing_condition'), group)) | dict(counts)
                                                  for group, counts in sorted(cache_groups.items())])
    hot_pairs = []
    for line, record in attempts:
        if record['cache_mode'] != 'hot':
            continue
        cold_key = dict(record['key'], cache='cold')
        cold = records.get(identity(cold_key))
        equal = cold is not None and cold['output_token_ids'] == record['output_token_ids']
        record_after_cold = cold is not None and datetime.fromisoformat(record['started_at']) >= datetime.fromisoformat(cold['ended_at'])
        if not equal or not record_after_cold:
            issue('hot/cold pairing or order mismatch', line, record['sample_id'])
        hot_pairs.append({'sample_id': record['sample_id'], 'method': record['method'], 'hot_source_line': line,
                          'cold_present': cold is not None, 'tokens_equal': equal, 'hot_after_cold': record_after_cold,
                          'hot_cache_hit': record['cache_hit'], 'hot_cache_build_zero': record['cache_build_seconds'] == 0,
                          'hot_output_tokens': record['output_tokens'], 'hot_query_calls': record['stage_calls']['query'],
                          'hot_request_tokens': record['request_tokens'], 'hot_resume_cache_rebuild_seconds': record.get('resume_cache_rebuild_seconds', 0),
                          'query_hook_scope': 'APE query forward count; stock sequential/PCW hook stage is unset'})
    write_csv(tables / 'runtime_hot_pairs.csv', hot_pairs)

    # Inventory uses unique inputs; repeated methods and cold/hot requests are conditions.
    baseline = [(line, row) for line, row in attempts if row['key']['stage'] == 'baseline']
    final = [(line, row) for line, row in attempts if row['key']['stage'] == 'longbench_final']
    five_methods = ['sequential', 'pcw', 'ape_default', 'ape_fixed', 'ape_dynamic']
    common_ids = set.intersection(*[
        {(row['dataset'], row['sample_id']) for _, row in baseline + final if row['method'] == method}
        for method in five_methods
    ])
    scopes = {
        'longbench_full600': (baseline, {(method, 'cold') for method in ('sequential', 'pcw', 'ape_default')}, 'question'),
        'longbench_common510': ([(line, row) for line, row in baseline + final if (row['dataset'], row['sample_id']) in common_ids],
                                {(method, 'cold') for method in five_methods}, 'question'),
        'litm_question_positions1500': ([(line, row) for line, row in attempts if row['key']['stage'] == 'litm'],
                                       {(method, 'cold') for method in ('sequential', 'ape_fixed', 'ape_dynamic')}, 'question_position'),
        'doc2dial_targets500': ([(line, row) for line, row in attempts if row['key']['stage'] == 'doc2dial'],
                               {(method, cache) for method in ('sequential', 'ape_fixed', 'ape_dynamic') for cache in ('cold', 'hot')}, 'target_turn'),
    }
    unique_inputs, input_summary, input_distribution, input_joint = [], [], [], []
    for scope, (scope_records, expected_conditions, input_unit) in scopes.items():
        inputs = defaultdict(list)
        for line, record in scope_records:
            inputs[(record['dataset'], record['sample_id'], record['position'])].append((line, record))
        for (dataset, sid, position), observed in sorted(inputs.items(), key=lambda item: str(item[0])):
            signatures = {(r['L'], r['N'], r['original_document_count'], tuple(r['block_lengths'])) for _, r in observed}
            conditions = Counter((r['method'], r['cache_mode']) for _, r in observed)
            conditions_ok = set(conditions) == expected_conditions and all(n == 1 for n in conditions.values())
            metadata_ok = len(signatures) == 1
            line, record = observed[0]
            if not conditions_ok or not metadata_ok:
                issue('input metadata or condition pairing mismatch', line, sid, value=scope)
            L, N = record['L'], record['N']
            unique_inputs.append({'scope': scope, 'dataset': dataset, 'sample_id': sid, 'position': position,
                                  'input_unit': input_unit, 'L': L, 'N': N, 'original_document_count': record['original_document_count'],
                                  'block_lengths': json.dumps(record['block_lengths']),
                                  'observed_conditions': len(observed), 'expected_conditions': len(expected_conditions),
                                  'conditions_complete_once': conditions_ok, 'input_metadata_equal_across_conditions': metadata_ok,
                                  'outside_grid_rectangle': L < 4096 or L > 16384 or N < 4 or N > 16,
                                  'outside_grid_L_range': L < 4096 or L > 16384,
                                  'outside_grid_N_range': N < 4 or N > 16,
                                  'outside_exact_nine_nodes': L not in (4096, 8192, 16384) or N not in (4, 8, 16),
                                  'source_line': line, 'source': 'attempts.jsonl; grouping retains one row per unique input'})
        datasets = sorted({row['dataset'] for row in unique_inputs if row['scope'] == scope})
        for dataset in datasets + ['ALL']:
            selected = [row for row in unique_inputs if row['scope'] == scope and (dataset == 'ALL' or row['dataset'] == dataset)]
            denominator = len(selected)
            input_summary.append({'scope': scope, 'dataset': dataset, 'unique_inputs': denominator,
                                  'unique_question_or_target_ids': len({row['sample_id'] for row in selected}),
                                  'input_unit': input_unit, 'conditions_per_input': len(expected_conditions),
                                  'input_metadata_equal_count': sum(row['input_metadata_equal_across_conditions'] for row in selected),
                                  'complete_condition_count': sum(row['conditions_complete_once'] for row in selected),
                                  'L_min': min(row['L'] for row in selected), 'L_max': max(row['L'] for row in selected),
                                  'N_values': ','.join(str(x) for x in sorted({row['N'] for row in selected})),
                                  'original_document_count_values': ','.join(str(x) for x in sorted({row['original_document_count'] for row in selected})),
                                  'outside_grid_rectangle_count': sum(row['outside_grid_rectangle'] for row in selected),
                                  'outside_grid_rectangle_ratio': sum(row['outside_grid_rectangle'] for row in selected) / denominator,
                                  'outside_grid_L_range_count': sum(row['outside_grid_L_range'] for row in selected),
                                  'outside_grid_N_range_count': sum(row['outside_grid_N_range'] for row in selected),
                                  'outside_exact_nine_nodes_count': sum(row['outside_exact_nine_nodes'] for row in selected),
                                  'denominator': denominator, 'unit': 'unique input count; L in document tokens; ratio 0-1',
                                  'rectangle_definition': '4096<=L<=16384 and 4<=actual_N<=16',
                                  'exact_nodes_definition': 'L in {4096,8192,16384} and actual_N in {4,8,16}'})
            for field in ('L', 'N', 'original_document_count'):
                counts = Counter(row[field] for row in selected)
                for value, count in sorted(counts.items()):
                    input_distribution.append({'scope': scope, 'dataset': dataset, 'field': field, 'value': value,
                                               'count': count, 'denominator': denominator, 'ratio': count / denominator,
                                               'input_unit': input_unit, 'unit': 'L=tokens; N=encoding blocks; original_document_count=source documents; ratio 0-1'})
            counts = Counter((row['L'], row['N'], row['original_document_count']) for row in selected)
            for (length, count_blocks, count_docs), count in sorted(counts.items()):
                input_joint.append({'scope': scope, 'dataset': dataset, 'L': length, 'N': count_blocks,
                                    'original_document_count': count_docs, 'count': count, 'denominator': denominator,
                                    'ratio': count / denominator, 'input_unit': input_unit, 'unit': 'unique input count; document tokens; ratio 0-1'})
    write_csv(tables / 'runtime_unique_inputs.csv', unique_inputs)
    write_csv(tables / 'runtime_input_scope_summary.csv', input_summary)
    write_csv(tables / 'runtime_input_distributions.csv', input_distribution)
    write_csv(tables / 'runtime_input_LN_joint.csv', input_joint)

    event_counts = Counter(row['event'] for _, row in events)
    write_csv(tables / 'runtime_event_counts.csv', [{'event': event, 'count': count, 'unit': 'event count'} for event, count in sorted(event_counts.items())])
    timeline = []
    worker_rows = []
    active = None
    for line, event in events:
        kind = event['event']
        if kind == 'worker_started':
            active = {'stage': event['stage'], 'method': event['method'], 'started_at': event['at'], 'source_start_line': line,
                      'command': json.dumps(event['command']), 'cwd': event['cwd']}
        elif kind == 'model_loaded':
            if active is not None:
                active['model_loaded_at'] = event['at']
                active['model_load_seconds'] = event['seconds']
        elif kind == 'worker_ended':
            if active is None or active['stage'] != event['stage'] or active['method'] != event['method']:
                issue('worker event order mismatch', line, source='events.jsonl')
            else:
                active.update(ended_at=event['at'], exit_code=event['exit_code'], source_end_line=line)
                worker_rows.append(active)
                active = None
        if kind not in ('request_started', 'request_ended', 'model_loaded'):
            timeline.append({'at_utc': event['at'], 'at_beijing': bj(event['at']), 'event': kind, 'stage': event.get('stage', ''),
                             'method': event.get('method', ''), 'exit_code': event.get('exit_code', ''), 'source': 'events.jsonl', 'source_line': line})
    for event, when, source in [('prep_accepted', review['at'], 'snapshot/logs/runs/20261113-main80-prep/review.json'),
                                ('campaign_started', state['started_at'], 'state.json'),
                                ('exploration_candidates_frozen', candidates['frozen_at'], 'candidates.json'),
                                ('dynamic_rule_frozen', rule['frozen_at'], 'rule.json'),
                                ('campaign_submitted', delivery['at'], 'delivery.json')]:
        timeline.append({'at_utc': when, 'at_beijing': bj(when), 'event': event, 'source': source})
    timeline.sort(key=lambda r: datetime.fromisoformat(r['at_utc']))
    write_csv(tables / 'runtime_stage_timeline.csv', timeline)
    write_csv(tables / 'runtime_workers.csv', worker_rows)
    independent_start = min(event['at'] for _, event in events if event['event'] == 'worker_started' and event['stage'] in ('length_check', 'attention', 'longbench_final', 'litm', 'doc2dial'))
    if datetime.fromisoformat(rule['frozen_at']) >= datetime.fromisoformat(independent_start):
        issue('rule frozen after independent checks start', source='rule.json')
    if delivery['status'] != 'SUBMITTED' or state['state'] != 'SUBMITTED':
        issue('final status differs from SUBMITTED', source='delivery.json/state.json')

    environment_rows = [{'item': key, 'value': value, 'source': 'logs/runs/20261113-main80-prep/environment.json', 'evidence_kind': 'preparation runtime observation'} for key, value in env.items()]
    environment_rows += [{'item': key, 'value': value, 'source': 'snapshot/logs/runs/20261113-main80-prep/review.json.runtime_versions', 'evidence_kind': 'accepted runtime versions'} for key, value in review['runtime_versions'].items()]
    environment_rows += [
        {'item': 'model_source', 'value': model_source['source'] + ' @ ' + model_source['fetch_branch'], 'source': 'logs/runs/20261113-main80-assets/model-manifest.json', 'evidence_kind': 'source metadata'},
        {'item': 'source_metadata_stable', 'value': model_stability['source_observed_before_after_agree'], 'source': 'logs/runs/20261113-main80-assets/model-source-stability.json', 'evidence_kind': 'source metadata'},
        {'item': 'physical_gpu', 'value': config['physical_gpu'], 'source': 'config.json + snapshot/scripts/with_env.sh', 'evidence_kind': 'executed configuration'},
        {'item': 'GPU model / total memory', 'value': '方案登记 RTX 4090 48GB；本campaign硬件遥测待补充', 'source': refs['hardware_plan'], 'evidence_kind': 'plan declaration'},
        {'item': 'GPU UUID / driver / power limit', 'value': '待补充', 'source': 'no campaign hardware telemetry in manifest', 'evidence_kind': 'missing observation'},
        {'item': 'dtype/batch/seed', 'value': f'{config["dtype"]}/{config["batch_size"]}/{config["seed"]}', 'source': 'config.json + snapshot/APE/main80/methods.py', 'evidence_kind': 'executed configuration'},
        {'item': 'CUDA build', 'value': env['torch'].split('+')[-1], 'source': 'environment.json.torch', 'evidence_kind': 'PyTorch build tag'},
        {'item': 'PC audit python', 'value': sys.version.replace('\n', ' '), 'source': sys.executable, 'evidence_kind': 'current CPU execution'},
        {'item': 'PC audit platform', 'value': platform.platform(), 'source': 'platform.platform()', 'evidence_kind': 'current CPU execution'},
    ]
    write_csv(tables / 'runtime_environment.csv', environment_rows)
    write_csv(tables / 'runtime_model_source_files.csv', [
        {'path': row['Path'], 'bytes': row['Size'], 'observed_source_revision': row['Revision'],
         'fetch_branch': model_source['fetch_branch'], 'source': 'logs/runs/20261113-main80-assets/model-manifest.json',
         'unit': 'bytes; Revision copied from existing source metadata'} for row in model_source['required_files']
    ])
    write_csv(tables / 'runtime_source_lines.csv', sources)
    write_csv(tables / 'runtime_integrity_issues.csv', issues)

    stage_summary = []
    for stage in STAGES:
        part = [(line, row) for line, row in attempts if row['key']['stage'] == stage]
        stage_summary.append({'stage': stage, 'records': len(part), 'cache_modes': ','.join(sorted(set(r['cache_mode'] for _, r in part))),
                              'timing_conditions': ','.join(sorted(set(r['timing_condition'] for _, r in part))),
                              'max_new_tokens': ','.join(str(x) for x in sorted(set(jobs[identity(r['key'])]['max_new_tokens'] for _, r in part))),
                              'cap_hits': sum(r['output_tokens'] == jobs[identity(r['key'])]['max_new_tokens'] for _, r in part),
                              'first_started_beijing': bj(min(r['started_at'] for _, r in part)),
                              'last_ended_beijing': bj(max(r['ended_at'] for _, r in part)),
                              'denominator': len(part)})
    write_csv(tables / 'runtime_stage_overview.csv', stage_summary)
    hot_ape = [r for r in hot_pairs if r['method'].startswith('ape_')]
    all_valid_numeric = all(row.get('valid_nonnegative_finite', row['denominator']) == row['denominator'] for row in field_rows if row['field'] in numeric_fields)
    formal_caps = [row for row in generation_rows if row['stage'] in ('baseline', 'longbench_final', 'litm', 'doc2dial')]
    report = f'''# 第1阶段：运行配置、计时与生成记录审核

我们按 worker bundle、运行快照和逐请求事件审核了 {len(attempts):,} 条生成记录。记录与任务上限全部连接，输入/输出长度、EOS、缓存条件和计时字段检查共记录 {len(issues)} 项异常。第1阶段仅整理配置与数据合法性，正式质量复算、效率比较和注意力图在后续阶段开展。研究状态保持 `{delivery['status']}`。

## 1 实际模型、环境与硬件登记

本 campaign 从 `models/Llama-3.1-8B-Instruct` 离线读取模型和 tokenizer，实际来源为 ModelScope 用户镜像 `{model_source['source']}` 的 `{model_source['fetch_branch']}` 分支。获取清单逐文件记录观察到的 Revision；获取前后 Path/Size/Revision 元数据比较结果为 `{model_stability['source_observed_before_after_agree']}`。模型权重保留在服务器，本 PC 包保存配置、索引和 tokenizer。模型架构为 {model_config['num_hidden_layers']} 层、{model_config['num_attention_heads']} 个注意力头、{model_config['num_key_value_heads']} 个 KV 头，上下文配置 {model_config['max_position_embeddings']:,} tokens。

准备阶段记录 Python {env['python'].split()[0]}、PyTorch {env['torch']}、Transformers {env['transformers']}、tokenizers {env['tokenizers']}、FlashAttention {env['flash_attn']}；冻结审核另记录 SacreBLEU {review['runtime_versions']['sacrebleu']}。with_env 使用 core-tf44 和 main80-scoring overlay，模型加载指定 FlashAttention 2、BF16、batch=1，种子42。PyTorch `cu124` 是其构建标签。

硬件记录分为执行配置和方案登记：执行配置为 physical_gpu=0、`CUDA_VISIBLE_DEVICES=0`、进程内 `cuda:0`；方案登记“RTX 4090 48GB”。本包缺少该 campaign 的设备名称、GPU UUID、实际总显存、驱动版本与功耗上限采集。报告的实际硬件栏应补充这些字段后填定；历史 R2 的 L20 注释对应另一入口。来源：`{refs['device']}`、`{refs['hardware_plan']}`、`{refs['load']}`。

完整版本和来源见 `runtime_environment.csv`，模型获取元数据见 `restored/logs/runs/20261113-main80-assets/model-manifest.json`。

## 2 输入模板和编码方法

任务 bundle 用 table/index 指向同一冻结行，再注入 N、方法和参数。我们检查 {len(jobs):,} 个任务与冻结行的 sample_id/max_new_tokens，并按 table、sample_id、L、N、position 检查各方法所用索引，共 {len(same_input_groups):,} 个输入条件组。具体异常见 `runtime_integrity_issues.csv`。

LongBench 官方提示在 context 处分割，LITM 保留公开问题和20篇资料文本，三者共同使用 Llama user/assistant envelope。Doc2Dial 的全篇资料前置，history/current question 保留在 suffix，`<last_turn>` 与倒序角色历史共同组成请求；所有方法使用同一适配。`runtime_templates.csv` 列出四张冻结表的字段与上限。Doc2Dial 的 last_turn 取历史最后一轮，可为 user 或 agent；该轮在 last_turn 与角色历史中均出现，完整角色历史保留，最近用户轮次可能更早。500个目标中28个紧邻前一轮为agent，按官方 turn_id-1 构造合法ID，具体计数见评分审核，来源 `{refs['template_doc2dial']}`。

顺序方法先编码公共 prefix+全部资料。PCW 对每个资料块加入独立 BOS 并在 position=0 编码，最长窗口稳定排首位，合并 KV 时保留一个 BOS，再以 prefix[1:]+suffix 发起请求。APE 每块共享公共 prefix，分别编码资料，再合并 prefix+各块 KV；query 使用 T/S 对上下文和其他分支进行调整。PCW 使用 stock Llama，APE 使用前缀/上下文/query 补丁。三种路径的 RoPE/缓存位置组织随方法定义变化。来源 `{refs['encode']}`、`{refs['merge_pcw']}`。

本次 PCW/APE 均通过 Python 逐块循环调用 forward，各块在同一 GPU 上顺序执行，之后合并独立编码的 KV，batch=1。常规输入按每块最多7500 document tokens 分块；开发网格显式指定 N。原始篇数与实际块数分别保存于 original_document_count 和 N。来源 `{refs['chunks']}`。

### 2.1 以唯一输入为分母的 L/N 盘点

LongBench完整基线每题取一条输入（3方法为条件）；共同测试取5方法交集ID（每题5个条件）；LITM按问题×位置去重，三个位置仍隶属于同一问题；Doc2Dial按目标轮次去重（3方法×冷/热为6个条件）。下表中的 unique_inputs 是输入数量，unique_question_or_target_ids 保留问题/目标ID数量。各方法/冷热条件之间仅核对L、N、原始资料篇数及逐块长度的一致性，时延与质量留待后续分析。

{md(input_summary, ['scope','dataset','unique_inputs','unique_question_or_target_ids','conditions_per_input','L_min','L_max','N_values','original_document_count_values','complete_condition_count','input_metadata_equal_count','outside_grid_rectangle_count','denominator'])}

网格范围定义为4096≤L≤16384且4≤实际N≤16；精确九节点定义为L∈{{4096,8192,16384}}且N∈{{4,8,16}}。`runtime_input_scope_summary.csv` 同时保存范围外比例、单独L/N越界计数和精确节点之外计数；`runtime_input_distributions.csv` 给出每个实际L、N及原始篇数的频数，`runtime_input_LN_joint.csv` 给出联合分布，均以唯一输入为分母。`runtime_unique_inputs.csv` 保存逐输入源行和完整配对检查结果。

## 3 生成配置与合法性

config 声明 do_sample=False、num_beams=1、seed=42；Engine 实际手工 `logits.argmax` 逐 token 贪心解码。模型 generation_config 的 do_sample=True、temperature=0.6、top_p=0.9 来自模型文件，执行入口仅使用其中 EOS 集 `{sorted(eos)}`。终止条件为 EOS 或达到任务 max_new_tokens；输出 token 数包含最终 EOS，decode 文本去除特殊 token。来源 `{refs['greedy']}`、`{refs['stops']}`。

有效 token ID 数组 {tokens_valid:,}/{len(attempts):,}，输出长度与数组/上限一致 {output_length_valid:,}/{len(attempts):,}。总体命中上限 {cap_equal:,} 条，其中 EOS 恰在上限 {cap_eos:,} 条。最长输出 {max_output} tokens。我们同时检查了正块长、L等于块长和、N等于块数、完整原始输入加生成上限落在模型窗口内；PCW 计算时将已附加在 request_tokens 中的 prefix[1:] 只计一次。

开发阶段实际沿用每数据集生成上限：Qasper为128，HotpotQA/MuSiQue为32。这个设置覆盖explore、validation、length_check和attention，来源为 `snapshot/APE/main80/freeze.py:21` 的行构造及随后开发行复制。后续参数比较应按这些实际执行上限解释各数据集结果。

各阶段数量、上限和上限命中如下。命中数在本阶段用作数据审核，长度与质量/时延的联合分析留给后续阶段。

{md(stage_summary, ['stage','records','cache_modes','max_new_tokens','cap_hits'])}

按数据集/方法/冷热细分见 `runtime_generation_limits.csv`，包含分母、EOS交叉计数和输出长度最小/最大值。

## 4 计时与显存字段定义

Engine.run 在 CUDA synchronize 后重置峰值统计，计时起点位于 prepare 前。TTFT 包括 prepare 分词、动态或固定选参分支、KV 获取/构建、请求 forward、首 token argmax 与同步；total 继续包含完整 decode 和文本解码。每个 token 的 argmax 后同步。`cache_build_seconds` 为缓存未命中时的资料编码区间（末端同步），`cache_get_seconds` 是 CPU 命中判断区间；`dynamic_selection_seconds` 对所有方法均记录参数分支的 CPU 时间，动态规则 choose 的耗时已经计入 TTFT/total。来源 `{refs['start']}`、`{refs['total']}`。

`peak_allocated_bytes` 和 `peak_reserved_bytes` 为该方法进程在重置峰值后的总峰值，含模型及存活缓存。重置峰值保留分配器 reserved 池；请求开始时还可能保留上一资料缓存，cold 分支在编码前清空 self.encoded。bytes 后续可按 bytes/2^30 展示为 GiB。来源 `{refs['memory']}`。

全部 {len(attempts):,} 条记录保存输入/输出长度、TTFT、total、cache_build/get、selection、resume rebuild、allocated/reserved；所检查数值字段均非负且有限的结果为 `{all_valid_numeric}`。T/S 在 sequential/PCW 中为 null，APE 中有值；实际字段可用性按阶段列于 `runtime_field_availability.csv`。

## 5 搜索、冷、热、诊断和加载分层

search 的 KV 缓存身份忽略 T/S，复用同一资料编码，timing_condition=`search_immutable_KV`；正式 cold 每次编码，timing_condition=`cold`；Doc2Dial hot 要求同进程、同方法/配置/文档的 GPU KV 命中，timing_condition=`application_GPU_KV`。资料缓存不含 suffix，因此历史/当前问题每次进入 query。来源 `{refs['cache']}`、`{refs['request']}`。

我们连接了 {len(hot_pairs):,} 个 cold/hot 对，输出 token 一致 {sum(r['tokens_equal'] for r in hot_pairs):,} 对，hot 在对应 cold 结束后启动 {sum(r['hot_after_cold'] for r in hot_pairs):,} 对，hot 均命中且本请求 cache_build=0 的对数 {sum(r['hot_cache_hit'] and r['hot_cache_build_zero'] for r in hot_pairs):,}。APE 的 {len(hot_ape):,} 条 hot 记录有 query 调用，调用数与输出 token 数逐题相同。sequential/PCW 的 stage_calls hook 未设置 query 阶段，query计数固定0；其新请求由代码完整 forward/decode 路径及独立 request_started/request_ended 事件核对。worker 每次 hot 都调用 engine.run，再检查 cold/hot token 列表。来源 `{refs['hot']}`。

attention 的12项配置在独立 worker/stage 中启用 observer，采样层0/15/31、头0/8/16/24、请求尾查询和实际decode步，其计时与常规请求分开。第1阶段仅审核该路径的配置和事件；完整注意力内容留到后续阶段。模型加载独立记录 model_loaded，含加载后 CUDA 同步，排除于请求计时之外。恢复资料 KV 的路径会把重建时间加回 TTFT/total，并另记 resume_cache_rebuild_seconds；本 campaign 的 resume_cache_rebuilt 事件数为 {event_counts['resume_cache_rebuilt']}。来源 `{refs['attention']}`、`{refs['model_event']}`、`{refs['restore']}`。

按 stage/method/cache/timing 的记录数、命中和重建分层见 `runtime_cache_layers.csv`；model_load_seconds 仅原值保存在 `runtime_workers.csv`。

## 6 启动、冻结、退出与提交顺序

事件包含 worker_started/model_loaded/worker_ended 各 {event_counts['worker_started']}/{event_counts['model_loaded']}/{event_counts['worker_ended']} 条，stage_submitted {event_counts['stage_submitted']} 条。worker退出码全部为0的结果为 `{all(r['exit_code'] == 0 for r in worker_rows)}`。逐请求 started/ended 事件各 {event_counts['request_started']:,}/{event_counts['request_ended']:,} 条，记录与唯一事件对连接 {request_event_pairs:,}/{len(attempts):,}。

准备审定时间为北京时间 {bj(review['at'])}；campaign 于 {bj(state['started_at'])} 启动；探索候选于 {bj(candidates['frozen_at'])} 固定；动态规则于 {bj(rule['frozen_at'])} 冻结；独立长度检查/注意力/测试首个 worker 启动于 {bj(independent_start)}；最终提交时间为 {bj(delivery['at'])}。候选和规则的冻结时间来自相应 JSON，events 另记录各阶段提交。timeline 保存 UTC 与北京时间，便于跨来源连接。

`delivery.json.status={delivery['status']}`，`state.json.state={state['state']}`，delivery 保留 requires_PLANNER_review={delivery['requires_PLANNER_review']}。启动快照和早期 launch observation 记录的是当时进度；当前提交状态据 delivery/state 读取。

## 7 第1阶段待补充与产物

- 硬件遥测待补充：GPU实际名称/UUID、总显存、驱动版本、功耗上限；现有方案和执行设备编号分别列示。
- 后续评分与统计阶段应继续按已冻结参数开展，保留上述生成长度和缓存分层；本阶段未运行 GPU、重新生成答案或修改证据。
- `runtime_integrity_issues.csv` 保存本脚本发现的最小异常证据；当前 {len(issues)} 行。
- 复现命令：`C:\\Users\\Osheepne\\anaconda3\\python.exe -B pc-analysis-output/scripts/audit_stage1_runtime.py`。
- 当前解释器 `{sys.executable}`，Python {platform.python_version()}；仅使用标准库，临时/缓存指向本阶段输出目录。全部输出在 `pc-analysis-output/stages/01_audit/`，本次未读取 attention 权重内容。
'''
    (reports / 'STAGE1_RUNTIME.md').write_text(report, encoding='utf-8')
    result = {'records': len(attempts), 'scheduled_jobs': len(jobs), 'worker_bundles': len(bundles), 'workers_exit_zero': sum(r['exit_code'] == 0 for r in worker_rows),
              'unique_request_event_pairs': request_event_pairs, 'valid_token_arrays': tokens_valid, 'valid_output_lengths': output_length_valid,
              'cap_hits': cap_equal, 'eos_at_cap': cap_eos, 'hot_pairs': len(hot_pairs), 'hot_token_equal_pairs': sum(r['tokens_equal'] for r in hot_pairs),
              'issues': len(issues), 'status': delivery['status'], 'scope': 'stage1 runtime/configuration audit only'}
    (tables / 'runtime_audit_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
