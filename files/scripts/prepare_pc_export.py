"""Prepare a standalone, lossless MAIN80 data export; no Git or digests.

Original files are read only. Large files are stored in <=45MiB pieces. Identical
copies are represented by aliases after direct byte comparison, never checksums.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 45 * 1024 * 1024
RUN = 'main80-20261114T041503'


def same(left, right):
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(1024*1024), b.read(1024*1024)
            if x != y:
                return False
            if not x:
                return True


def select():
    files = set()
    for name in ('AGENTS.md', 'README.md', 'campaign.json', 'data_sources.json', 'APE/README.md', 'APE/setup.py', 'APE/demo_ape.py', 'APE/experiments/Longbench/pred_parallel9.py'):
        files.add(ROOT / name)
    for directory in ('APE/main80', 'APE/ape', 'scripts', 'datasets/raw-main80', 'datasets/frozen-main80', 'datasets/prepared-main80', 'environment/locks', 'environment/snapshots', f'output/runs/{RUN}'):
        files.update(p for p in (ROOT / directory).rglob('*') if p.is_file() and not p.is_symlink())
    files.update((ROOT / 'docs').glob('*.md'))
    files.update(p for p in (ROOT / 'APE').glob('LICENSE*') if p.is_file())
    for directory in (ROOT / 'logs/runs').glob('*main80*'):
        files.update(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink())
    for name in ('README.md', 'LICENSE', 'USE_POLICY.md', 'config.json', 'generation_config.json', 'model.safetensors.index.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json'):
        files.add(ROOT / 'models/Llama-3.1-8B-Instruct' / name)
    included, omitted = [], []
    for p in sorted(files):
        relative = p.relative_to(ROOT)
        if any(part in {'.git', '.codex', '.agents', '.aws', '.runtime', '__pycache__'} for part in relative.parts) or p.suffix in {'.pyc', '.whl', '.lock'}:
            omitted.append({'path': str(relative), 'reason': 'runtime_control_cache_or_platform_package'})
            continue
        assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
        included.append(p)
    return included, omitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='exports/main80-pc-analysis')
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError('export must remain inside workspace')
    destination.mkdir(parents=True, exist_ok=False)
    sources, omitted = select()
    manifest = {'format': 'ape-pc-export/v1', 'campaign': RUN, 'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'PREPARED_NOT_UPLOADED', 'max_piece_bytes': CHUNK, 'files': [], 'omitted_selected_files': omitted,
        'excluded_roots': ['.git', '.codex', '.agents', '.aws', '.runtime', '.overlays', 'model weights and download partials', 'environment installers', 'docs/share original application', 'unrelated historical experiment runs'],
        'integrity_policy': 'file inventory, exact byte comparison, JSON structure and sample IDs; no file digest computation'}
    canonical_by_size = {}
    verified = 0
    for number, source in enumerate(sources):
        relative = str(source.relative_to(ROOT))
        size = source.stat().st_size
        alias = next((prior for prior in canonical_by_size.get(size, []) if same(source, ROOT / prior['path'])), None)
        entry = {'path': relative, 'bytes': size}
        if alias is not None:
            entry.update(storage='alias', alias_of=alias['path'])
        elif size > CHUNK:
            pieces = []
            with source.open('rb') as stream:
                index = 0
                while True:
                    data = stream.read(CHUNK)
                    if not data:
                        break
                    piece = Path('parts') / f'{number:05d}' / f'{index:04d}.part'
                    target = destination / piece
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    pieces.append({'path': str(piece), 'bytes': len(data)})
                    index += 1
            entry.update(storage='parts', parts=pieces)
            with source.open('rb') as original:
                for piece in pieces:
                    with (destination / piece['path']).open('rb') as stored:
                        while True:
                            b = stored.read(1024*1024)
                            if not b:
                                break
                            assert original.read(len(b)) == b
                assert not original.read(1)
        else:
            path = Path('files') / relative
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert same(source, target), relative
            entry.update(storage='file', stored_path=str(path))
        if alias is None:
            canonical_by_size.setdefault(size, []).append(entry)
        manifest['files'].append(entry)
        verified += 1
    manifest['counts'] = {'logical_files': len(manifest['files']), 'verified_by_direct_comparison': verified,
        'aliases': sum(e['storage'] == 'alias' for e in manifest['files']),
        'multipart_files': sum(e['storage'] == 'parts' for e in manifest['files']),
        'logical_bytes': sum(e['bytes'] for e in manifest['files']),
        'stored_data_bytes': sum(e['bytes'] for e in manifest['files'] if e['storage'] != 'alias')}
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(ROOT / 'scripts/pc_analysis_client.py', destination / 'pc_analysis.py')
    (destination / '.gitattributes').write_text('* -text\n')
    (destination / '.gitignore').write_text('restored/\n__pycache__/\n.venv/\n.DS_Store\nThumbs.db\npc-analysis-output/\n')
    (destination / 'README.md').write_text(f'''# APE MAIN80 PC分析包

Campaign: `{RUN}`。计算已提交，研究结果仍待正式审核。原始文件未移动、删改或重跑。

包含本次全部逐题答案/评分/时延/显存、事件/命令/退出/原始stdout及stderr、搜索和冻结规则、完整token注意力及12幅SVG、启动快照、准备与审定证据、冻结及公开原始数据、当前代码/脚本、模型来源/配置/索引/分词器、环境版本/锁。模型权重、下载partial、运行环境二进制、浏览器/账户配置、原申报书和无关历史实验未纳入。

## PC端使用

Python3.10+，基础分析无第三方依赖、无需GPU。取得仓库后在其根目录执行：

```text
python -B pc_analysis.py verify
python -B pc_analysis.py summary
python -B pc_analysis.py restore --output restored
```

`summary`直接读取包内逐题数据和自动汇总；`restore`按manifest恢复原始相对目录，之后可用pandas、Jupyter或Excel分析。已有非一致目标文件会报错，不覆盖PC修改。

GitHub常规Git单文件上限为100MiB。这里超过45MiB的文件按字节拆分；相同内容经直接比较后存为别名。`restore`会重建全部原文件，包含200MB完整注意力文件。没有删减或压缩实验数值，没有计算文件摘要。`.gitattributes`禁止换行转换以保留原始字节。

恢复后的主要入口：

- `output/runs/{RUN}/attempts.jsonl`：18,116条逐题记录。
- 同目录`summary.json`、`results.csv`、`events.jsonl`、`rule.json`、`candidates.json`、`attention.jsonl`及`attention-figures/`。
- `datasets/frozen-main80/`：样本、分组、构造区间及嵌套顺序。
- `docs/EXPERIMENT_PLAN.md`、`docs/REPRODUCTION_STAGE.md`和`APE/main80/SOURCES.md`：协议、授权、来源与模板适配。

原始命令保留服务器路径作为证据；PC分析不执行这些GPU命令。快照是启动时的历史内容，当前计算状态以campaign的`delivery.json`为准。自动汇总不等同最终验收。原始数据来源及许可位于`datasets/raw-main80/ASSET_MANIFEST.json`，模型为ModelScope用户镜像master，非Meta官方发布或不可变revision。

本地包目前尚未提交或上传，接收仓库及Git必需哈希例外等待用户确认。
''')
    print(json.dumps({'export': str(destination.relative_to(ROOT)), **manifest['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
