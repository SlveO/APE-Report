"""Serial GPU worker, launched by the campaign controller in a fresh process."""
import argparse
import sys
import time
import traceback

from .common import Ledger, ROOT, append_json, identity, inside, now, read_json, write_json
from .diagnostics import AttentionCapture
from .methods import Engine, SampleError
from .metrics import score


def recovery_timing(result, seconds, peak=None):
    result = dict(result, resume_cache_rebuild_seconds=seconds)
    if seconds:
        result['gpu_hot_ttft_after_rebuild_seconds'] = result['ttft_seconds']
        result['gpu_hot_total_after_rebuild_seconds'] = result['total_seconds']
        result['ttft_seconds'] += seconds
        result['total_seconds'] += seconds
        result['timing_condition'] = 'recovered_application_GPU_KV'
        for name, value in (peak or {}).items():
            result['resume_cache_rebuild_' + name] = value
            result[name] = max(result.get(name) or 0, value)
    return result


def execute(bundle):
    model_path = inside(bundle['model'])
    ledger = Ledger(bundle['ledger'])
    tables = {}
    started = time.perf_counter()
    engine = Engine.load(model_path, bundle['method'])
    engine.sync()
    append_json(bundle['events'], {'event': 'model_loaded', 'at': now(), 'method': bundle['method'], 'seconds': time.perf_counter() - started})
    rule = read_json(bundle['rule']) if bundle.get('rule') else None
    successful = 0
    for job in bundle['jobs']:
        if time.time() >= bundle['deadline_epoch']:
            raise TimeoutError('campaign deadline reached')
        if ledger.done(job['key']):
            continue
        if bundle.get('selection'):
            selected = read_json(bundle['selection'])
            scope = {'longbench_final': 'test'}.get(job['key']['stage'], job['key']['stage'])
            if scope in selected and job['key']['sample_id'] not in selected[scope]:
                continue
        if job['table'] not in tables:
            tables[job['table']] = read_json(f'{bundle["data"]}/{job["table"]}.json')
        row = dict(tables[job['table']][job['index']])
        if job['N'] is not None:
            row['N'] = job['N']
        write_json(bundle['heartbeat'], {'state': 'RUNNING', 'at': now(), 'job': job['key'], 'pid': __import__('os').getpid()})
        record = {'key': job['key'], 'attempt': ledger.next_attempt(job['key']), 'started_at': now(),
            'sample_id': row['sample_id'], 'dataset': row['dataset'], 'split': row['split'], 'group': row['group'],
            'source': row.get('source', row.get('source_ids')), 'method': job['method'],
            'reference_answers': row['answers'], 'logical_labels': job['logical_labels'], 'position': row.get('position')}
        append_json(bundle['events'], {'event': 'request_started', 'at': now(), 'key': job['key'], 'attempt': record['attempt']})
        request_start = time.perf_counter()
        try:
            param = tuple(job['param'] or (0.9, 0.9))
            recovery_build = 0.0
            recovery_peak = {}
            if job['cache'] == 'hot':
                cold_key = dict(job['key'], cache='cold')
                if ledger.outcome(cold_key) is None:
                    raise SampleError('hot request lacks a completed paired cold generation')
                if engine.encoded is None:
                    if engine.device.type == 'cuda':
                        import torch
                        engine.sync()
                        torch.cuda.reset_peak_memory_stats(engine.device)
                    recovery_build = engine.restore_cache(row, param, rule)
                    if engine.device.type == 'cuda':
                        recovery_peak = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(engine.device),
                                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(engine.device)}
                    append_json(bundle['events'], {'event': 'resume_cache_rebuilt', 'at': now(), 'key': job['key'],
                        'seconds': recovery_build, 'generation_repeated': False, 'included_in_total_response_and_ttft': True})
            if job['key']['stage'] == 'attention':
                with AttentionCapture(engine.model, bundle['attention'], {'key': job['key'], 'T': param[0], 'S': param[1], 'diagnostic_roles': job['logical_labels']}):
                    result = engine.run(row, param=param, cache_mode=job['cache'])
            else:
                result = engine.run(row, param=param, rule=rule, cache_mode=job['cache'])
            record.update(recovery_timing(result, recovery_build, recovery_peak))
            if job['cache'] == 'hot':
                cold = ledger.outcome(dict(job['key'], cache='cold'))
                if cold['output_token_ids'] != result['output_token_ids']:
                    raise RuntimeError('cold/hot greedy outputs differ for the same request and immutable cache')
            record.update(status='success', score=score(row['dataset'], result['answer'], row['answers']))
            successful += 1
        except SampleError as e:
            record.update(recovery_timing(e.evidence, recovery_build, recovery_peak) if e.evidence else {})
            record.update(status='sample_error', error=str(e), elapsed_seconds=time.perf_counter() - request_start)
            if isinstance(e.evidence.get('answer'), str) and not e.evidence['answer'].strip() and e.evidence.get('output_token_ids'):
                record.update(score=0.0, score_status='official_empty_response_zero; request remains failed')
                if job['cache'] == 'hot':
                    cold = ledger.outcome(dict(job['key'], cache='cold'))
                    if cold is not None and cold['output_token_ids'] != record['output_token_ids']:
                        record.update(status='system_error', error='cold/hot greedy token mismatch, including empty response', ended_at=now())
                        ledger.add(record)
                        raise RuntimeError(record['error'])
            append_json(bundle['events'], {'event': 'sample_error', 'at': now(), 'key': job['key'], 'error': str(e)})
        except BaseException as e:
            record.update(status='system_error', error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc(),
                          elapsed_seconds=time.perf_counter() - request_start, ended_at=now())
            ledger.add(record)
            raise
        record['ended_at'] = now()
        ledger.add(record)
        append_json(bundle['events'], {'event': 'request_ended', 'at': now(), 'key': job['key'], 'attempt': record['attempt'], 'status': record['status']})
    write_json(bundle['heartbeat'], {'state': 'SUBMITTED', 'at': now(), 'successful_this_worker': successful})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args()
    execute(read_json(args.bundle))


if __name__ == '__main__':
    main()
