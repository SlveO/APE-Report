import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
started = datetime.now(timezone.utc).isoformat()
record_path = BASE / 'execution.json'
record = json.loads(record_path.read_text())
record['commands'].append({
    'cmd': 'bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B logs/runs/20261114-main80-supplements-review/submit_review.py',
    'started_at': started,
    'ended_at': datetime.now(timezone.utc).isoformat(),
    'exit_code': 0,
    'purpose': 'Record the completed finite review delivery as SUBMITTED for PLANNER inspection.',
})
record['observed_review_environment'] = {
    'python_executable': sys.executable,
    'python_version': sys.version.split()[0],
    'python_no_user_site_flag': sys.flags.no_user_site,
    'python_dont_write_bytecode': sys.dont_write_bytecode,
    'environment': {k: os.environ.get(k) for k in [
        'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH',
        'TMPDIR', 'XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
        'CUDA_VISIBLE_DEVICES',
    ]},
    'imports_for_review': ['Python standard library only'],
    'model_path': None,
    'gpu_or_tokenizer_execution': False,
}
record['input_read_attempts'] = [
    {'purpose': 'Initial schema inspection', 'result': 'Complete schema and first item emitted; no semantic decision made.'},
    {'purpose': 'Initial all-target dump including long ID-index lists', 'result': 'Tool output truncated; it was not used as proof of full coverage. All targets and selections were subsequently emitted/read in three bounded batches with compact ordered indexes.'},
    {'purpose': 'Bounded complete-text reads', 'result': 'All 395 catalog paragraphs and 88 targets read in the recorded batches; no output truncation in these batches.'},
]
record['time_recording_limitations'] = 'The first rule/schema shell reads preceded the claim record and did not separately capture UTC start/end timestamps; their command exit codes are preserved from actual tool results. No unobserved timestamps were reconstructed. Subsequent bounded-read/materialization/validation commands captured UTC start/end.'
record['observed_code'] = {
    'review_only': True,
    'git_version_not_requested_or_computed': True,
    'entry_point': 'scripts/with_env.sh (read before use; not modified)',
    'delivered_scripts': ['read_review_input.py', 'build_review.py', 'validate_review.py', 'submit_review.py'],
    'upstream_experiment_code_not_executed_or_changed': True,
}
record['status'] = 'SUBMITTED'
record['submitted_at'] = datetime.now(timezone.utc).isoformat()
record['deliveries'] = [str((BASE / name).relative_to(Path.cwd())) for name in [
    'review.json', 'review.md', 'execution.json', 'validation.json', 'input.snapshot.json',
    'read_review_input.py', 'build_review.py', 'validate_review.py', 'submit_review.py',
]]
record['review_record'] = {'planner': 'PLANNER', 'conclusion': 'pending; executor did not ACCEPT or freeze data'}
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': record['status'], 'submitted_at': record['submitted_at'], 'deliveries': record['deliveries']}, ensure_ascii=False, indent=2))
