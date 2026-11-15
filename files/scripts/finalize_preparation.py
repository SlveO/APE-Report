"""PLANNER sign-off: verify submitted evidence and freeze exact launch inputs.

No model execution, downloads, dependency changes or digest calculation.
"""
import ast
import difflib
import shutil
import subprocess
from pathlib import Path

from main80.campaign import runtime_versions, source_files
from main80.common import ROOT, inside, now, read_json, write_json


def main():
    destination = inside('logs/runs/20261113-main80-prep')
    if (destination / 'review.json').exists():
        raise ValueError('PREP is already signed; do not rewrite accepted evidence')
    config = read_json('campaign.json')
    data = read_json(destination / 'data-validation.json')
    assert data['status'] == 'PASS' and data['longbench_ids'] == dict.fromkeys(('qasper', 'hotpotqa', 'musique'), 200)
    assert data['paired_litm_questions'] == 500 and data['complete_history_doc2dial'] == 500
    assert all(not v['evidence_group_overlap'] and not v['question_overlap'] for v in data['group_and_single_document_overlap'].values())
    for path in ('logs/runs/20261113-main80-methods-cpu/exit.txt', 'logs/runs/20261114-main80-methods-full/launcher-exit.txt'):
        assert inside(path).read_text().strip() == '0'
    full = read_json('logs/runs/20261114-main80-methods-full/execution.json')
    assert full['outcome'] == 'PASS' and full['full_model_generations'] == 16
    full_dir = ROOT / 'logs/runs/20261114-main80-methods-full'
    for path in ('APE/main80/methods.py', 'APE/main80/diagnostics.py', 'APE/ape/ape_llama.py'):
        archived = full_dir / 'code' / (path.replace('/', '__') + '.txt')
        assert archived.read_bytes() == (ROOT / path).read_bytes(), 'tested model logic changed: ' + path
    protocol = read_json('logs/runs/20261114-main80-protocol-tests-r3/command.json')
    assert protocol['exit_code'] == 0
    assert 'Ran 12 tests' in inside('logs/runs/20261114-main80-protocol-tests-r3/stdout').read_text()
    background = read_json(destination / 'background-test-latest.json')
    assert background['status'] == 'PASS' and background['actual_browser_rendered'] and background['five_second_HTML_refresh'] and background['independent_deadline_watchdog']
    assets = read_json('logs/runs/20261113-main80-assets/model-structure.json')
    assert len(assets['required_files']) == 13 and assets['index_entries'] == 291
    assert read_json('logs/runs/20261113-main80-assets/model-source-stability.json')['source_observed_before_after_agree']
    deleted = read_json(destination / 'cleanup-manifest.json')['paths']
    assert all(not inside(item['path']).exists() for item in deleted)
    assert inside('docs/share/[参考]ape项目申报书.doc').is_file()
    sources = sorted((ROOT / 'APE/main80').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    forbidden = {'sha256', 'sha1', 'md5', 'blake2b', 'hash_file', 'device_count', 'get_device_properties', 'mem_get_info'}
    for path in sources:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                assert name not in forbidden, f'forbidden operation: {path}:{node.lineno}:{name}'
            if isinstance(node, ast.ImportFrom):
                assert not any(part in (node.module or '').lower() for part in ('chatrag', 'chatqa'))
    for name in ('.runtime', 'models/Llama-3.1-8B-Instruct', 'datasets/frozen-main80', 'output/runs'):
        inside(name)
    for path in (ROOT / 'APE/main80').rglob('*.pyc'):
        raise ValueError('bytecode write in new source tree: ' + str(path))
    scope = {'role': 'PLANNER', 'status': 'PASS', 'at': now(), 'deleted_exact_manifest_files': len(deleted),
        'original_application_preserved': True, 'no_old_ChatQA_imports': True, 'no_project_digest_or_GPU_preflight_calls': True,
        'configuration_paths_confined': True, 'python_bytecode_disabled': True,
        'basis': 'actual code AST/path inspection, scoped command records, offline full-model and browser evidence; timing/memory observations are not release gates'}
    write_json(destination / 'scope-validation.json', scope)
    # Textual diffs read existing Git objects and compare text; never create blobs.
    patches = []
    for name in ('ape/ape_llama.py', 'experiments/Longbench/pred_parallel9.py'):
        before = subprocess.check_output(['git', '-C', str(ROOT / 'APE'), 'show', 'HEAD:' + name]).decode()
        after = (ROOT / 'APE' / name).read_text()
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), 'existing-HEAD/' + name, 'working/' + name))
    for path in sorted((ROOT / 'APE/main80').glob('*.py')):
        patches.extend(difflib.unified_diff([], path.read_text().splitlines(True), '/dev/null', str(path.relative_to(ROOT))))
    (destination / 'implementation.diff.txt').write_text(''.join(patches))
    checks = {
        'data': ['logs/runs/20261113-main80-prep/data-validation.json', 'datasets/frozen-main80/nested-coverage.json'],
        'methods_cpu': ['logs/runs/20261113-main80-methods-cpu/tests.txt', 'logs/runs/20261113-main80-methods-cpu/execution.json'],
        'methods_full_model': ['logs/runs/20261114-main80-methods-full/delivery.md', 'logs/runs/20261114-main80-methods-full/execution.json'],
        'cache_generation': ['logs/runs/20261114-main80-methods-full/ape_fixed/result.json', 'logs/runs/20261114-main80-protocol-tests-r3/stdout'],
        'recovery': ['logs/runs/20261114-main80-protocol-tests-r3/stdout'],
        'background': ['logs/runs/20261113-main80-prep/background-test-latest.json', str(Path(background['directory']).relative_to(ROOT) / 'browser-seen.jsonl')],
        'scope': ['logs/runs/20261113-main80-prep/scope-validation.json', 'logs/runs/20261113-main80-prep/cleanup-manifest.json'],
        'scoring': ['logs/runs/20261113-main80-assets/scoring-overlay-validation.json', 'logs/runs/20261114-main80-protocol-tests-r3/stdout']}
    paths = [p for p in source_files(config) if p != destination / 'review.json']
    paths += [ROOT / name for name in ('campaign.json', 'data_sources.json', 'scripts/observe_campaign.py', 'APE/experiments/Longbench/pred_parallel9.py', 'datasets/prepared-main80/supplement-reviews.json')]
    frozen = []
    for path in dict.fromkeys(paths):
        relative = path.relative_to(ROOT)
        copy = destination / 'accepted-inputs' / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        assert path.read_bytes() == copy.read_bytes()
        frozen.append({'path': str(relative), 'accepted_copy': str(copy.relative_to(ROOT))})
    review = {'protocol': 'ape-plan-exec/v1', 'task_id': 'PREP', 'plan_version': 'v3.0', 'role': 'PLANNER', 'status': 'ACCEPTED', 'at': now(),
        'checks': {k: {'status': 'PASS', 'evidence': v} for k, v in checks.items()},
        'frozen_inputs': frozen, 'runtime_versions': runtime_versions(), 'model_required_files': assets['required_files'],
        'limits': ['Functional short checks do not prove benchmark gains. APE answered the changed synthetic green-river query incorrectly; original output retained.',
            '18 of 450 development length constructions are inapplicable; all allowed nested levels still preserve three datasets and144 configurations.',
            'PCW GPU single-window comparison uses the predeclared BF16 numerical tolerance and matching argmax, not bit identity.'],
        'formal_campaign_started': False, 'authorization_ref': 'U_MAIN80', 'dependency_authorization_ref': 'U_MAIN80_SCORING'}
    write_json(destination / 'review.json', review)
    print('PREP ACCEPTED; directly frozen', len(frozen), 'input/code files; MAIN80 may be set READY by PLANNER')


if __name__ == '__main__':
    main()
