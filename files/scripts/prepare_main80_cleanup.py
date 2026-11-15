"""One-shot, authorized v3 cleanup. Manifest is durable before the first unlink."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'logs/runs/20261113-main80-prep'


def selection():
    selected = {ROOT / 'docs/AI_INFRA_LAB.md': 'authorized_learning_plan'}
    obsolete = {'20260926-r2p-a1', '20260926-r2p-a2'}
    for run in (ROOT / 'logs/runs').iterdir():
        if not run.is_dir() or run.is_symlink():
            continue
        for p in run.rglob('*'):
            if not p.is_file() or p.is_symlink():
                continue
            reason = None
            if run.name in obsolete:
                reason = 'superseded_R2_proposal_and_draft'
            elif run.name in {'20260926-r2-a3', '20260926-r2-a4'}:
                if p.name.startswith(('gate', 'GATE_', 'selftest', 'SELFTEST_')) or 'gate-attempt-1' in p.parts or p.name == 'validated-uuid.txt':
                    reason = 'obsolete_resource_gate_and_duplicate_parser_tests'
                if p.name == 'version.txt' and 'hf-home' in p.parts:
                    reason = 'historical_empty_cache_marker'
            elif run.name == '20260926-r1env-a1':
                if p.name.startswith('gpu-preflight') or p.name in {'STATIC_CHECKS.md.txt', 'demo-help.log', 'demo-help.exit'}:
                    reason = 'obsolete_preflight_or_duplicate_static_check'
                elif 'freeze' in p.name and 'cpuonly' not in p.name:
                    reason = 'duplicate_freeze_preserving_cpuonly_effective_snapshot'
            elif run.name == '20260926-r2-a2':
                if p.name not in {'demo_ape.before.py.txt', 'git-diff.before.patch.txt'} and any(term in p.name.lower() for term in ('gate', 'du.', 'query', 'available', 'apps_', 'counted_', 'mem')):
                    reason = 'obsolete_resource_gate_keep_failure_and_code_evidence'
            elif run.name in {'20260924-r1ms-a1', '20260925-r1ms-a2'}:
                if 'PROPOSAL' in p.name or 'sha256' in p.name or p.name == 'local-preflight.txt':
                    reason = 'superseded_acquisition_proposal_or_old_digest_record'
            elif run.name == '20260925-r1d-a2' and p.name == 'sha256-check.txt':
                reason = 'redundant_old_digest_record_keep_structure_and_offline_results'
            if reason:
                selected[p] = reason
    return sorted(((p, reason) for p, reason in selected.items() if p.exists()), key=lambda x: str(x[0]))


def main():
    RECORD.mkdir(parents=True, exist_ok=True)
    target = RECORD / 'cleanup-manifest.json'
    if target.exists():
        raise SystemExit('manifest already exists; inspect cleanup-result.json, do not rerun blindly')
    entries = []
    for p, reason in selection():
        assert p.is_relative_to(ROOT) and p.resolve().is_relative_to(ROOT) and not p.is_symlink()
        assert '20261113-workspace-cleanup' not in p.parts
        entries.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'reason': reason})
    with target.open('x') as f:
        json.dump({'authorization': 'U_MAIN80', 'created_at': datetime.now(timezone.utc).isoformat(), 'paths': entries}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    deleted = []
    for entry in entries:
        (ROOT / entry['path']).unlink()
        deleted.append(entry['path'])
    (RECORD / 'cleanup-result.json').write_text(json.dumps({'deleted': deleted, 'count': len(deleted), 'recovery': 'No new backup. Authorized discarded logs and learning plan are not guaranteed recoverable. Retained originals and prior cleanup records were not edited.'}, indent=2))
    print(json.dumps({'manifest': str(target.relative_to(ROOT)), 'deleted_files': len(deleted)}))


if __name__ == '__main__':
    main()
