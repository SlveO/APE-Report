"""Detached, single-instance, 80-hour controller. Does not query GPU/resources."""
import argparse
import ctypes
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .common import Ledger, ROOT, append_json, identity, inside, now, read_json, read_records, write_json
from .jobs import JobFactory, STAGES
from .report import export
from .rules import freeze_rule, select_exploration

PYTHON = '/home/shenxiaoyang/miniconda3/envs/ape/bin/python'
BASE = ROOT / 'output/runs'
REQUIRED_CHECKS = ('data', 'methods_cpu', 'methods_full_model', 'cache_generation', 'recovery', 'background', 'scope', 'scoring')


def validate(config):
    fixed = {'protocol': 'ape-plan-exec/v1', 'task_id': 'MAIN80', 'plan_version': 'v3.0',
             'model': 'models/Llama-3.1-8B-Instruct', 'physical_gpu': 0, 'dtype': 'bfloat16',
             'batch_size': 1, 'do_sample': False, 'num_beams': 1, 'seed': 42, 'budget_seconds': 288000}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f'fixed campaign protocol mismatch: {key}')
    for key in ('data', 'model', 'prep_review'):
        inside(config[key])
    review = read_json(config['prep_review'])
    if review.get('role') != 'PLANNER' or review.get('status') != 'ACCEPTED':
        raise ValueError('PREP has not been accepted by PLANNER')
    for name in REQUIRED_CHECKS:
        check = review.get('checks', {}).get(name, {})
        if check.get('status') != 'PASS' or not check.get('evidence'):
            raise ValueError(f'missing preparation check: {name}')
        for path in check['evidence']:
            if not inside(path).is_file():
                raise ValueError(f'missing evidence: {path}')
    if not review.get('frozen_inputs'):
        raise ValueError('PREP review lacks directly comparable frozen inputs/code')
    for item in review['frozen_inputs']:
        if inside(item['path']).read_bytes() != inside(item['accepted_copy']).read_bytes():
            raise ValueError(f'input/code changed after PREP acceptance: {item["path"]}')
    if review.get('runtime_versions') != runtime_versions():
        raise ValueError('effective dependency versions changed after PREP acceptance')
    for item in review.get('model_required_files', []):
        if inside(config['model']).joinpath(item['path']).stat().st_size != item['bytes']:
            raise ValueError(f'model structure changed after PREP acceptance: {item["path"]}')
    frozen = read_json(f'{config["data"]}/frozen.json')
    if frozen.get('counts') != {'longbench': 600, 'litm': 1500, 'doc2dial': 500}:
        raise ValueError('frozen dataset counts mismatch')
    return review


def runtime_versions():
    from importlib.metadata import version
    return {name: version(name) for name in ('torch', 'transformers', 'tokenizers', 'flash-attn', 'numpy', 'regex',
        'sacrebleu', 'portalocker', 'colorama', 'tabulate', 'lxml', 'safetensors')}


def source_files(config):
    code = sorted((ROOT / 'APE/main80').glob('*.py')) + [ROOT / 'APE/ape/ape_llama.py', ROOT / 'APE/ape/__init__.py', ROOT / 'scripts/with_env.sh', ROOT / 'scripts/run_campaign.sh']
    data = [inside(f'{config["data"]}/frozen.json')]
    frozen = read_json(data[0])
    data.extend(inside(f'{config["data"]}/{name}') for name in frozen['files'])
    model = [inside(config['model']) / name for name in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors.index.json')]
    return code + data + model + [inside(config['prep_review'])]


def snapshot(directory, config, resume=False):
    destination = directory / 'snapshot'
    files = source_files(config)
    names = [str(p.relative_to(ROOT)) for p in files]
    if resume:
        if read_json(directory / 'source-files.json') != names:
            raise ValueError('source file set changed; not an identical resume')
        for path in files:
            original = destination / path.relative_to(ROOT)
            if original.read_bytes() != path.read_bytes():
                raise ValueError(f'content changed; cannot impersonate same campaign: {path.relative_to(ROOT)}')
        for name in ('candidates.json', 'rule.json'):
            artifact, sealed = directory / name, destination / 'generated' / name
            if sealed.exists() and not artifact.exists():
                # The canonical sealed file is committed first, before dependents.
                # A crash before publishing its identical presentation copy is safe.
                write_json(artifact, read_json(sealed))
                append_json(directory / 'events.jsonl', {'event': 'generated_artifact_restored', 'at': now(), 'name': name, 'basis': 'committed immutable content copy; no generation repeated'})
            if artifact.exists() != sealed.exists() or (artifact.exists() and artifact.read_bytes() != sealed.read_bytes()):
                raise ValueError(f'frozen generated artifact changed: {name}')
    else:
        for path in files:
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        write_json(directory / 'source-files.json', names)
        # Read an existing commit reference; do not compute Git/blob digests.
        head = (ROOT / 'APE/.git/HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = ROOT / 'APE/.git' / head[5:]
            observed = ref.read_text().strip() if ref.is_file() else 'packed ref; see existing HEAD'
        else:
            observed = head
        write_json(directory / 'code-version.json', {'observed_HEAD': head, 'observed_existing_commit': observed,
            'working_code_identity': 'source-files.json and full source snapshots; direct byte comparison on resume'})


def seal_generated(directory, name, content):
    artifact = directory / name
    sealed = directory / 'snapshot/generated' / name
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.exists():
        old = read_json(sealed)
        comparable = {k: v for k, v in old.items() if k != 'frozen_at'}
        if comparable != content:
            raise ValueError(f'cannot replace frozen {name}')
    else:
        if artifact.exists():
            raise ValueError(f'unsealed generated artifact requires review: {name}')
        write_json(sealed, {**content, 'frozen_at': now()})
    if artifact.exists():
        if sealed.read_bytes() != artifact.read_bytes():
            raise ValueError(f'sealed {name} changed')
    else:
        write_json(artifact, read_json(sealed))


def _parent_death():
    """Only our child: exit on controller failure; terminal exit is harmless."""
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        os._exit(91)


def start_browser(directory, url):
    previous, _ = read_records(directory / 'browser-launches.jsonl')
    profile = inside(f'.runtime/browser/{directory.name}/{len(previous)}')
    profile.mkdir(parents=True, exist_ok=True)
    browser = shutil.which('google-chrome') or shutil.which('firefox')
    if not browser:
        raise RuntimeError('no installed browser for independent monitor window')
    if 'firefox' in browser:
        command = [browser, '--no-remote', '--new-instance', '--profile', str(profile), '--new-window', url]
    else:
        command = [browser, f'--user-data-dir={profile}', '--new-window', '--no-first-run', '--no-default-browser-check',
                   '--disable-gpu', '--disable-dev-shm-usage', '--disable-background-networking',
                   '--disable-component-update', '--disable-sync', '--metrics-recording-only', url]
    log = (directory / 'browser.log').open('ab')
    p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    write_json(directory / 'browser.json', {'pid': p.pid, 'command': command, 'url': url, 'at': now()})
    append_json(directory / 'browser-launches.jsonl', {'pid': p.pid, 'command': command, 'url': url, 'at': now()})
    return p


def confirm_browser(directory, since, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        receipts, _ = read_records(directory / 'browser-seen.jsonl')
        if any(r.get('rendered') is True and r.get('at', '') >= since and
               any(browser in r.get('user_agent', '') for browser in ('Firefox/', 'Chrome/')) for r in receipts):
            return
        time.sleep(0.2)
    raise RuntimeError('independent browser did not render this campaign page')


def launch_monitor(directory):
    old = directory / 'monitor.json'
    if old.exists():
        import urllib.request
        state = read_json(old)
        try:
            with urllib.request.urlopen(state['url'] + 'status.json', timeout=2) as response:
                if json.load(response)['campaign'] == directory.name:
                    return state['url']
        except (OSError, ValueError):
            pass
    log = (directory / 'monitor.log').open('ab')
    subprocess.Popen([PYTHON, '-B', '-u', '-m', 'main80.monitor', '--directory', str(directory), '--campaign', directory.name],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if old.exists():
            state = read_json(old)
            # A fresh monitor replaces its file; verify the actual endpoint below.
            import urllib.request
            try:
                with urllib.request.urlopen(state['url'] + 'status.json', timeout=1) as response:
                    if json.load(response)['campaign'] == directory.name:
                        return state['url']
            except (OSError, ValueError):
                pass
        time.sleep(0.1)
    raise RuntimeError('monitor did not start')


def bucket(length):
    return min((4096, 8192, 16384, 32768, 65536, 131072), key=lambda x: abs(math.log2(max(length or 1, 1) / x)))


def speed_key(method, length):
    return ('ape' if method.startswith('ape_') else method, bucket(length))


def rates(records):
    values = defaultdict(list)
    for r in records:
        if r['status'] == 'success' and r['key']['stage'] != 'attention':
            # Cold and hot requests cannot share a latency estimate.
            cache = 'hot_recovered' if r.get('timing_condition') == 'recovered_application_GPU_KV' else r.get('cache_mode', 'cold')
            key = (*speed_key(r['method'], r['L']), cache)
            values[key].append(r['total_seconds'])
    return [{'method': k[0], 'length_bucket': k[1], 'cache_mode': k[2], 'count': len(v),
             'mean': mean(v), 'low': min(v), 'high': max(v)} for k, v in sorted(values.items())]


def estimate(jobs, ledger, speed, conservative=False, partial=False):
    total_low, total_high = 0.0, 0.0
    observed_jobs = 0
    by_key = {(r['method'], r['length_bucket'], r['cache_mode']): r for r in speed if r['count'] >= 3}
    for job in jobs:
        if ledger.finished(job['key']):
            continue
        method, length = speed_key(job['method'], job['key']['L'])
        cache = job['cache']
        observation = by_key.get((method, length, cache))
        proxy = False
        if observation is None and conservative and cache in {'hot', 'search'}:
            observation = by_key.get((method, length, 'cold'))
            proxy = observation is not None
        if observation is None:
            # No invented speed for an unobserved method/length/cache stratum.
            if partial:
                continue
            return None
        observed_jobs += 1
        total_low += 0 if proxy else observation['low']
        total_high += observation['high']
    return (total_low, total_high) if observed_jobs or not partial else None


def all_jobs(factory, limits, candidates):
    # Conservative validation count before exploration completes (three distinct choices).
    temporary = candidates or {'global': [0.5, 0.5], 'cells': {f'{l}/{n}': [0.7, 0.7] for l in (4096, 8192, 16384) for n in (4, 8, 16)}}
    return [j for stage in STAGES for j in factory.make(stage, limits, temporary)]


def selected_records(ledger, jobs, directory):
    groups = defaultdict(list)
    for job in jobs:
        key = job['key']
        groups[(key['sample_id'], key['L'], key['N'])].append(key)
    paired = []
    for group, keys in groups.items():
        if all(ledger.outcome(key) is not None for key in keys):
            paired.extend(ledger.outcome(key) for key in keys)
        else:
            append_json(directory / 'events.jsonl', {'event': 'invalid_pair_excluded', 'at': now(),
                'sample_cell': group, 'missing_keys': [k for k in keys if ledger.outcome(k) is None],
                'basis': 'failed or missing output; never score-based, no replacement'})
    return paired


def update_status(directory, state, factory, candidates=None, final=False):
    ledger = Ledger(directory / 'attempts.jsonl')
    expected = all_jobs(factory, state['limits'], candidates)
    elapsed = max(0, time.time() - state['started_epoch'])
    speed = rates(ledger.records)
    eta = estimate(expected, ledger, speed)
    forecast = estimate(expected, ledger, speed, conservative=True)
    forecast_complete = forecast is not None
    if forecast is None:
        forecast = estimate(expected, ledger, speed, conservative=True, partial=True)
    notices, _ = read_records(directory / 'events.jsonl')
    noteworthy = [e for e in notices if e['event'] in {'sampling_reduced', 'system_error', 'budget_blocked', 'deadline', 'sample_error', 'source_incomplete'}]
    heartbeat = read_json(directory / 'worker-state.json') if (directory / 'worker-state.json').exists() else {}
    stamp = time.time()
    status = {'campaign': directory.name, 'state': state['state'], 'stage': state.get('stage'), 'updated_at': now(),
        'completed': sum(ledger.done(j['key']) for j in expected), 'planned': len(expected),
        'processed_including_failed': sum(ledger.finished(j['key']) for j in expected),
        'failed': sum(r['status'] != 'success' for r in ledger.records),
        'elapsed_seconds': elapsed, 'remaining_seconds': max(0, state['deadline_epoch'] - stamp),
        'eta_range': [datetime.fromtimestamp(stamp + x, timezone.utc).isoformat() for x in eta] if eta else None,
        'rates': speed, 'current_job': heartbeat.get('job'), 'notices': noteworthy[-10:], 'monitor_url': state.get('monitor_url'),
        'budget_forecast_seconds': forecast, 'forecast_complete': forecast_complete,
        'forecast_policy': 'observed strata; cold upper bound for unseen search/hot only; incomplete forecast covers known strata only, unknown future costs remain unestimated; not a reliable ETA'}
    write_json(directory / 'status.json', status)
    write_json(directory / 'state.json', state)
    if final:
        export(directory, factory, state['limits'], expected)
    return forecast, ledger, expected


def reduce_budget(directory, state, factory, candidates):
    eta, ledger, expected = update_status(directory, state, factory, candidates)
    if eta is None:
        return
    locked = set()
    if 'explore' in state['completed_stages']:
        locked.add('explore')
    if 'validation' in state['completed_stages']:
        locked.add('validation')
    if 'longbench_final' in state['completed_stages']:
        locked.add('test')
    if 'doc2dial' in state['completed_stages']:
        locked.update(('litm', 'doc2dial'))
    while eta[1] + state['export_reserve_seconds'] > state['deadline_epoch'] - time.time():
        lower = factory.lower(state['limits'], locked)
        if lower is None:
            state['state'] = 'BLOCKED'
            append_json(directory / 'events.jsonl', {'event': 'budget_blocked', 'at': now(), 'estimate_seconds': eta, 'limits': state['limits']})
            return
        before = factory.selected_ids(state['limits'])
        state['limits'] = lower
        after = factory.selected_ids(lower)
        write_json(directory / 'sampling.json', after)
        append_json(directory / 'events.jsonl', {'event': 'sampling_reduced', 'at': now(), 'estimate_seconds': eta,
            'remaining_seconds': state['deadline_epoch'] - time.time(), 'before_ids': before, 'after_ids': after,
            'reason': 'observed method/length stratified speed; no score used'})
        expected = all_jobs(factory, lower, candidates)
        eta = estimate(expected, ledger, rates(ledger.records), conservative=True)
        if eta is None:
            eta = estimate(expected, ledger, rates(ledger.records), conservative=True, partial=True)
        if eta is None:
            break


def stop_own_worker(worker):
    if worker is not None and worker.poll() is None:
        try:
            os.killpg(worker.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)


def control(directory, lock_fd):
    config = read_json(directory / 'config.json')
    validate(config)
    snapshot(directory, config, resume=True)
    state = read_json(directory / 'state.json')
    factory = JobFactory(config['data'])
    candidates = read_json(directory / 'candidates.json') if (directory / 'candidates.json').exists() else None
    worker = None
    try:
        state['state'] = 'RUNNING'
        state['controller_pid'] = os.getpid()
        update_status(directory, state, factory, candidates)
        state['monitor_url'] = launch_monitor(directory)
        browser_started = now()
        start_browser(directory, state['monitor_url'])
        confirm_browser(directory, browser_started)
        # Wall-clock budget starts before the first formal worker, including loading.
        for stage in STAGES:
            if stage in state['completed_stages']:
                continue
            if time.time() >= state['deadline_epoch']:
                raise TimeoutError('80-hour deadline reached')
            state['stage'] = stage
            reduce_budget(directory, state, factory, candidates)
            if state['state'] == 'BLOCKED':
                break
            jobs = factory.make(stage, state['limits'], candidates)
            methods = list(dict.fromkeys(j['method'] for j in jobs))
            for method in methods:
                ledger = Ledger(directory / 'attempts.jsonl')
                # An explicit sample error remains terminal unless a new authorized
                # correction creates a new campaign; do not silently retry generations.
                if any(r['status'] == 'system_error' for r in ledger.records):
                    raise RuntimeError('unresolved systematic error; dependent execution remains stopped for PLANNER review')
                terminal_data = {identity(r['key']) for r in ledger.records if r['status'] == 'sample_error'}
                pending = [j for j in jobs if j['method'] == method and not ledger.done(j['key']) and identity(j['key']) not in terminal_data]
                if not pending:
                    continue
                serial = len(list((directory / 'workers').glob('*.bundle.json'))) if (directory / 'workers').exists() else 0
                stem = directory / 'workers' / f'{serial:03d}-{stage}-{method}'
                stem.parent.mkdir(parents=True, exist_ok=True)
                rule_path = str(directory / 'rule.json') if (directory / 'rule.json').exists() and stage not in {'explore', 'validation', 'attention'} else None
                bundle = {'model': config['model'], 'data': config['data'], 'method': method, 'jobs': pending,
                    'ledger': str(directory / 'attempts.jsonl'), 'events': str(directory / 'events.jsonl'),
                    'heartbeat': str(directory / 'worker-state.json'), 'selection': str(directory / 'sampling.json'),
                    'attention': str(directory / 'attention.jsonl'), 'rule': rule_path, 'deadline_epoch': state['deadline_epoch']}
                bundle_path = str(stem) + '.bundle.json'
                write_json(bundle_path, bundle)
                command = [PYTHON, '-B', '-u', '-m', 'main80.worker', '--bundle', bundle_path]
                append_json(directory / 'events.jsonl', {'event': 'worker_started', 'at': now(), 'stage': stage, 'method': method, 'command': command, 'cwd': str(ROOT)})
                with Path(str(stem) + '.stdout').open('ab') as out, Path(str(stem) + '.stderr').open('ab') as err:
                    worker = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, start_new_session=True, preexec_fn=_parent_death)
                # Deadline termination must remain independent of JSON/status/export IO.
                def expire(owned=worker):
                    if owned.poll() is None:
                        stop_own_worker(owned)
                        write_json(directory / 'deadline-event.json', {'at': now(), 'owned_worker_pid': owned.pid})
                watchdog = threading.Timer(max(0, state['deadline_epoch'] - time.time()), expire)
                watchdog.daemon = True
                watchdog.start()
                write_json(directory / 'worker.json', {'pid': worker.pid, 'command': command, 'cwd': str(ROOT), 'started_at': now()})
                last_estimate = time.monotonic()
                while worker.poll() is None:
                    remaining = state['deadline_epoch'] - time.time()
                    if remaining <= 0:
                        stop_own_worker(worker)
                        raise TimeoutError('80-hour deadline reached')
                    try:
                        worker.wait(timeout=min(5, remaining))
                    except subprocess.TimeoutExpired:
                        pass
                    if time.time() >= state['deadline_epoch']:
                        stop_own_worker(worker)
                        raise TimeoutError('80-hour deadline reached')
                    update_status(directory, state, factory, candidates)
                    if time.monotonic() - last_estimate >= 600:
                        reduce_budget(directory, state, factory, candidates)
                        last_estimate = time.monotonic()
                        if state['state'] == 'BLOCKED':
                            stop_own_worker(worker)
                append_json(directory / 'events.jsonl', {'event': 'worker_ended', 'at': now(), 'stage': stage, 'method': method, 'exit_code': worker.returncode})
                watchdog.cancel()
                watchdog.join(timeout=1)
                if time.time() >= state['deadline_epoch']:
                    raise TimeoutError('80-hour deadline reached')
                if worker.returncode != 0:
                    raise RuntimeError(f'worker stopped: {stage}/{method}, exit {worker.returncode}')
                if state['state'] == 'BLOCKED':
                    break
            if state['state'] == 'BLOCKED':
                break
            ledger = Ledger(directory / 'attempts.jsonl')
            if stage == 'explore':
                candidates = select_exploration(selected_records(ledger, factory.make(stage, state['limits'], candidates), directory))
                seal_generated(directory, 'candidates.json', candidates)
            elif stage == 'validation':
                rule = freeze_rule(selected_records(ledger, factory.make(stage, state['limits'], candidates), directory), candidates)
                seal_generated(directory, 'rule.json', {**rule, 'frozen_before': 'length_check/attention/test'})
            elif stage == 'attention':
                from .plots import render
                append_json(directory / 'events.jsonl', {'event': 'attention_figures_exported', 'at': now(), 'figures': render(directory)})
            state['completed_stages'].append(stage)
            append_json(directory / 'events.jsonl', {'event': 'stage_submitted', 'at': now(), 'stage': stage})
            update_status(directory, state, factory, candidates, final=True)
        if state['state'] == 'RUNNING':
            state['state'] = 'SUBMITTED'
    except BaseException as exc:
        stop_own_worker(worker)
        state['state'] = 'BLOCKED'
        append_json(directory / 'events.jsonl', {'event': 'deadline' if isinstance(exc, TimeoutError) else 'system_error', 'at': now(),
            'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
    finally:
        update_status(directory, state, factory, candidates, final=True)
        write_json(directory / 'delivery.json', {'role': 'EXECUTOR', 'status': state['state'], 'at': now(),
            'evidence': ['summary.json', 'incomplete.json', 'attempts.jsonl', 'events.jsonl', 'rule.json'], 'requires_PLANNER_review': True})
        os.close(lock_fd)


def launch(config_path=None, campaign=None):
    lock_path = inside('.runtime/main80.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('a MAIN80 controller already holds the single-instance lock')
    if campaign:
        if Path(campaign).name != campaign:
            raise ValueError('campaign must be an ID, not a path')
        directory = inside(BASE / campaign)
        config = read_json(directory / 'config.json')
        validate(config)
        snapshot(directory, config, resume=True)
        state = read_json(directory / 'state.json')
        if state['state'] == 'SUBMITTED' or time.time() >= state['deadline_epoch']:
            raise ValueError('campaign finished or deadline expired; cannot resume')
        if any(r['status'] == 'system_error' for r in Ledger(directory / 'attempts.jsonl').records):
            raise ValueError('systematic failure requires PLANNER correction/review; dependent work cannot auto-resume')
        if config_path and read_json(config_path) != config:
            raise ValueError('configuration changed; create a new campaign')
    else:
        config = read_json(config_path)
        validate(config)
        campaign = datetime.now(timezone.utc).strftime('main80-%Y%m%dT%H%M%S')
        directory = inside(BASE / campaign)
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'config.json', config)
        snapshot(directory, config)
        factory = JobFactory(config['data'])
        epoch = time.time()
        write_json(directory / 'sampling.json', factory.selected_ids(factory.initial_limits()))
        write_json(directory / 'state.json', {'campaign': campaign, 'state': 'RUNNING', 'stage': 'baseline',
            'started_at': now(), 'started_epoch': epoch, 'deadline_epoch': epoch + 288000,
            'export_reserve_seconds': 900, 'limits': factory.initial_limits(), 'completed_stages': []})
    command = [PYTHON, '-B', '-u', '-m', 'main80.campaign', '_control', '--campaign', campaign, '--lock-fd', str(lock.fileno())]
    with (directory / 'controller.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock.fileno(),))
    write_json(directory / 'launcher.json', {'controller_pid': process.pid, 'cwd': str(ROOT), 'command': command, 'at': now()})
    lock.close()
    print(json.dumps({'campaign': campaign, 'controller_pid': process.pid, 'directory': str(directory),
                      'status_command': f'bash scripts/run_campaign.sh status --campaign {campaign}'}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'resume', 'status', 'check', '_control'))
    parser.add_argument('--config')
    parser.add_argument('--campaign')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.action == 'start':
        if not args.config:
            parser.error('--config required')
        launch(args.config)
    elif args.action == 'resume':
        launch(args.config, args.campaign)
    elif args.action == 'status':
        if not args.campaign or Path(args.campaign).name != args.campaign:
            parser.error('valid --campaign required')
        from .monitor import current_status
        print(json.dumps(current_status(inside(BASE / args.campaign)), ensure_ascii=False, indent=2))
    elif args.action == 'check':
        validate(read_json(args.config))
        print('PREP accepted and frozen campaign configuration valid; no worker started')
    else:
        control(inside(BASE / args.campaign), args.lock_fd)


if __name__ == '__main__':
    main()
