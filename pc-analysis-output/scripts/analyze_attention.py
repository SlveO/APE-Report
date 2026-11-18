"""Stream MAIN80 attention evidence and audit stops/model-loading on CPU.

Run: python -B pc-analysis-output/scripts/analyze_attention.py
Sources remain read-only. All products are written below pc-analysis-output.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CAMPAIGN = 'main80-20261114T041503'
HEADS = [0, 8, 16, 24]
LAYERS = [0, 15, 31]


def key_id(key):
    return json.dumps(key, sort_keys=True, separators=(',', ':'))


def lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                yield line_no, json.loads(line)


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows, fields, digits=4):
    out = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |']
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, '')
            values.append(f'{value:.{digits}f}' if isinstance(value, float) else str(value))
        out.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--restored', type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    restored = (args.restored or repo / 'restored').resolve()
    out = repo / 'pc-analysis-output'
    tables, figures, reports = (out / x for x in ('tables', 'figures', 'reports'))
    for directory in (tables, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)
    run = restored / 'output' / 'runs' / CAMPAIGN
    attempts = [record for _, record in lines(run / 'attempts.jsonl')]
    by_key = {key_id(record['key']): record for record in attempts}
    caps = {}
    for bundle_path in sorted((run / 'workers').glob('*.bundle.json')):
        bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
        for job in bundle['jobs']:
            caps[key_id(job['key'])] = job['max_new_tokens']
    gen_config = json.loads((restored / 'models' / 'Llama-3.1-8B-Instruct' / 'generation_config.json').read_text(encoding='utf-8'))
    stop_ids = gen_config['eos_token_id']
    stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
    details = []
    for line_no, record in enumerate(attempts, 1):
        key = record['key']
        cap = caps.get(key_id(key))
        tokens = record.get('output_token_ids', [])
        n = record.get('output_tokens', 0)
        eos = bool(tokens and tokens[-1] in stop_ids)
        hit = n == cap if cap is not None else None
        # Include equality to the generation cap even when its final token is EOS.
        reason = 'eos_at_cap' if hit and eos else 'length_cap' if hit else 'eos' if eos else 'other'
        details.append({
            'source_line': line_no, 'stage': key['stage'], 'dataset': record['dataset'],
            'sample_id': record['sample_id'], 'method': record['method'],
            'cache_mode': record['cache_mode'], 'timing_condition': record['timing_condition'],
            'L': record['L'], 'N': record['N'], 'T': record['T'], 'S': record['S'],
            'max_new_tokens': cap, 'output_tokens': n, 'cap_hit': hit, 'eos_last': eos,
            'stop_reason': reason, 'last_token_id': tokens[-1] if tokens else None,
            'recorded_score': record.get('score'), 'ttft_seconds': record.get('ttft_seconds'),
            'total_seconds': record.get('total_seconds'), 'source': 'attempts.jsonl; cap from workers/*.bundle.json; EOS from generation_config.json',
        })
    write_csv(tables / 'generation_detail.csv', details)
    groups = defaultdict(list)
    for row in details:
        group = tuple(row[x] for x in ('stage', 'dataset', 'method', 'cache_mode', 'timing_condition'))
        groups[group].append(row)
    stop_summary = []
    for group, rows in sorted(groups.items()):
        stats = {field: value for field, value in zip(('stage', 'dataset', 'method', 'cache_mode', 'timing_condition'), group)}
        stats.update(n=len(rows), denominator=len(rows), max_new_tokens=','.join(str(x) for x in sorted(set(r['max_new_tokens'] for r in rows if r['max_new_tokens'] is not None))),
                     cap_hit_count=sum(r['cap_hit'] is True for r in rows),
                     eos_last_count=sum(r['eos_last'] for r in rows), missing_cap_count=sum(r['max_new_tokens'] is None for r in rows),
                     cap_hit_rate=sum(r['cap_hit'] is True for r in rows) / len(rows),
                     mean_output_tokens=float(np.mean([r['output_tokens'] for r in rows])),
                     median_output_tokens=float(np.median([r['output_tokens'] for r in rows])),
                     p90_output_tokens=float(np.quantile([r['output_tokens'] for r in rows], .9)),
                     unit='tokens; rate in 0-1')
        stop_summary.append(stats)
    write_csv(tables / 'generation_stops.csv', stop_summary)
    conditional = []
    for group, rows in sorted(groups.items()):
        for hit in (False, True):
            selected = [r for r in rows if r['cap_hit'] is hit]
            if not selected:
                continue
            result = dict(zip(('stage', 'dataset', 'method', 'cache_mode', 'timing_condition'), group))
            result.update(cap_hit=hit, n=len(selected), denominator=len(rows),
                          mean_recorded_score=float(np.mean([r['recorded_score'] for r in selected])),
                          score_unit='F1/EM 0-1; Doc2Dial sentence BLEU 0-100 (descriptive length grouping only)',
                          mean_output_tokens=float(np.mean([r['output_tokens'] for r in selected])),
                          mean_total_seconds=float(np.mean([r['total_seconds'] for r in selected])),
                          mean_ttft_seconds=float(np.mean([r['ttft_seconds'] for r in selected])))
            conditional.append(result)
    write_csv(tables / 'generation_cap_conditioned.csv', conditional)
    loads = []
    cache_rebuilds = []
    for line_no, event in lines(run / 'events.jsonl'):
        if event.get('event') == 'model_loaded':
            loads.append({'source_line': line_no, 'method': event['method'], 'at': event['at'], 'seconds': event['seconds'], 'unit': 'seconds; fresh worker load+cuda sync'})
        elif event.get('event') == 'resume_cache_rebuilt':
            cache_rebuilds.append({'source_line': line_no, **event})
    write_csv(tables / 'model_load_events.csv', loads)
    write_csv(tables / 'resume_cache_rebuild_events.csv', cache_rebuilds)
    load_summary = []
    for method in sorted(set(row['method'] for row in loads)):
        values = [row['seconds'] for row in loads if row['method'] == method]
        load_summary.append({'method': method, 'n': len(values), 'mean_seconds': float(np.mean(values)), 'median_seconds': float(np.median(values)), 'p90_seconds': float(np.quantile(values, .9))})
    write_csv(tables / 'model_load_summary.csv', load_summary)

    # The raw attention file is streamed line by line. Keep only 64-token bins.
    head_rows, diagnostic_rows, anomalies = [], [], []
    binned = defaultdict(lambda: defaultdict(list))
    metadata = {}
    seen = set()
    for line_no, row in lines(run / 'attention.jsonl'):
        key = row['key']
        kid = key_id(key)
        record = by_key[kid]
        metadata[kid] = record
        weights = np.asarray(row['token_probabilities'], dtype=float)
        branches = np.asarray(row['branch_masses'], dtype=float)
        p, length, request = record['prefix_tokens'], record['L'], record['request_tokens']
        step, layer = row['step'], row['layer']
        expected_length = p + length + request + step
        identity = (kid, layer, step)
        if identity in seen:
            anomalies.append({'source_line': line_no, 'issue': 'duplicate key/layer/step'})
        seen.add(identity)
        if row['heads'] != HEADS or layer not in LAYERS or weights.shape != (4, expected_length):
            anomalies.append({'source_line': line_no, 'issue': 'head/layer/length mismatch', 'observed_shape': str(weights.shape), 'expected_length': expected_length})
        if not np.isfinite(weights).all() or (weights < 0).any():
            anomalies.append({'source_line': line_no, 'issue': 'nonfinite or negative probability'})
        if row['max_abs_output_error'] > row['tolerance']:
            anomalies.append({'source_line': line_no, 'issue': 'reconstruction exceeds saved tolerance'})
        query_kind = 'query_last_token' if step == 0 else 'decode_token'
        if row['query_kind'] != query_kind:
            anomalies.append({'source_line': line_no, 'issue': 'query kind inconsistent with step'})
        for j, head in enumerate(row['heads']):
            w = weights[j]
            total_mass = float(w.sum())
            context_mass = float(w[p:p+length].sum())
            other_mass = float(w[:p].sum() + w[p+length:].sum())
            block_mass = [float(w[p+sum(record['block_lengths'][:b]):p+sum(record['block_lengths'][:b+1])].sum()) for b in range(record['N'])]
            entropy = float(-(w[w > 0] * np.log(w[w > 0])).sum())
            head_rows.append({
                'source_line': line_no, 'sample_id': record['sample_id'], 'L': length, 'N': record['N'],
                'T': row['T'], 'S': row['S'], 'layer': layer, 'head': head, 'step': step, 'query_kind': query_kind,
                'token_count': len(w), 'prefix_tokens': p, 'document_tokens': length, 'request_tokens': request,
                'prefix_mass': float(w[:p].sum()), 'context_mass': context_mass,
                'request_mass': float(w[p+length:p+length+request].sum()), 'decoded_history_mass': float(w[p+length+request:].sum()),
                'total_mass': total_mass, 'context_branch_mass': float(branches[j, 0]), 'other_branch_mass': float(branches[j, 1]),
                'context_mass_error': context_mass - float(branches[j, 0]), 'other_mass_error': other_mass - float(branches[j, 1]),
                'entropy_nats': entropy, 'max_token_probability': float(w.max()),
                'max_block_mass': max(block_mass), 'block_mass_json': json.dumps(block_mass, separators=(',', ':')),
                'unit': 'saved probability mass (BF16 branch mixing); entropy nats',
            })
        mean = weights.mean(axis=0)
        bins = np.asarray([mean[i:i+64].sum() for i in range(0, len(mean), 64)])
        binned[kid][layer].append((step, bins))
        diagnostic_rows.append({'source_line': line_no, 'sample_id': record['sample_id'], 'L': length, 'N': record['N'],
                                'T': row['T'], 'S': row['S'], 'layer': layer, 'step': step, 'query_kind': row['query_kind'],
                                'token_count': weights.shape[1], 'expected_token_count': expected_length,
                                'max_abs_output_error': row['max_abs_output_error'], 'tolerance': row['tolerance'],
                                'error_to_tolerance': row['max_abs_output_error'] / row['tolerance']})
    write_csv(tables / 'attention_head_steps.csv', head_rows)
    write_csv(tables / 'attention_record_audit.csv', diagnostic_rows)
    write_csv(tables / 'attention_anomalies.csv', anomalies)
    bins_csv, cases_csv = [], []
    for number, (kid, layers) in enumerate(binned.items()):
        record = metadata[kid]
        figure_name = f'attention_{number:02d}'
        fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), constrained_layout=True)
        maximum_bins = max(len(a) for values in layers.values() for _, a in values)
        maximum_mass = max(float(a.max()) for values in layers.values() for _, a in values)
        for ax, layer in zip(axes, LAYERS):
            values = sorted(layers[layer], key=lambda x: x[0])
            matrix = np.full((len(values), maximum_bins), np.nan)
            expected_steps = set(range(record['output_tokens']))
            actual_steps = {step for step, _ in values}
            if actual_steps != expected_steps:
                anomalies.append({'sample_id': record['sample_id'], 'L': record['L'], 'N': record['N'], 'T': record['T'], 'S': record['S'], 'layer': layer, 'issue': 'missing or extra decode step'})
            for i, (step, values_i) in enumerate(values):
                matrix[i, :len(values_i)] = values_i
                for b, mass in enumerate(values_i):
                    bins_csv.append({'case': number, 'sample_id': record['sample_id'], 'L': record['L'], 'N': record['N'],
                                     'T': record['T'], 'S': record['S'], 'layer': layer, 'step': step,
                                     'bin': b, 'token_start_inclusive': b*64, 'token_end_exclusive': min((b+1)*64, record['prefix_tokens'] + record['L'] + record['request_tokens'] + step),
                                     'mass_mean_four_heads': float(mass), 'unit': 'probability mass per 64 physical cache tokens'})
            im = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=maximum_mass)
            ax.axvline(record['prefix_tokens']/64-.5, color='white', lw=.7, alpha=.75)
            ax.axvline((record['prefix_tokens']+record['L'])/64-.5, color='red', lw=.8, alpha=.8)
            ax.set_title(f'Layer {layer}; heads 0, 8, 16, 24 averaged')
            ax.set_ylabel('Step (0 = last query token)')
            ax.set_xlabel('64-token bin in physical cache order: prefix / document / request / decode history')
            fig.colorbar(im, ax=ax, label='Probability mass')
        fig.suptitle(f'{record["dataset"]} | L={record["L"]}, N={record["N"]}, T/S={record["T"]}/{record["S"]}\n{record["sample_id"]}', fontsize=10)
        for ext in ('png', 'svg'):
            fig.savefig(figures / f'{figure_name}.{ext}', dpi=160)
        plt.close(fig)
        rows_for_case = [r for r in diagnostic_rows if r['sample_id'] == record['sample_id'] and r['L'] == record['L'] and r['N'] == record['N'] and r['T'] == record['T'] and r['S'] == record['S']]
        cases_csv.append({'case': number, 'figure_png': f'{figure_name}.png', 'sample_id': record['sample_id'], 'dataset': record['dataset'],
                          'L': record['L'], 'N': record['N'], 'T': record['T'], 'S': record['S'],
                          'diagnostic_roles': json.dumps(record['logical_labels']), 'score_f1_0_1': record['score'],
                          'output_tokens': record['output_tokens'], 'raw_records': len(rows_for_case),
                          'max_output_reconstruction_error': max(r['max_abs_output_error'] for r in rows_for_case),
                          'max_error_to_tolerance': max(r['error_to_tolerance'] for r in rows_for_case),
                          'ttft_seconds_diagnostic': record['ttft_seconds'], 'total_seconds_diagnostic': record['total_seconds'],
                          'unit': 'F1 0-1; seconds; diagnostic observer enabled'})
    write_csv(tables / 'attention_bins.csv', bins_csv)
    write_csv(tables / 'attention_cases.csv', cases_csv)
    write_csv(tables / 'attention_anomalies.csv', anomalies)
    query_context = []
    for case in cases_csv:
        for layer in LAYERS:
            selected = [r for r in head_rows if r['sample_id'] == case['sample_id'] and r['L'] == case['L'] and r['N'] == case['N'] and r['T'] == case['T'] and r['S'] == case['S'] and r['layer'] == layer and r['step'] == 0]
            query_context.append({**{k:case[k] for k in ('case','dataset','sample_id','L','N','T','S')}, 'layer': layer,
                                  'mean_context_mass_first_query': float(np.mean([r['context_mass'] for r in selected])),
                                  'mean_prefix_mass_first_query': float(np.mean([r['prefix_mass'] for r in selected])),
                                  'mean_request_mass_first_query': float(np.mean([r['request_mass'] for r in selected])),
                                  'denominator_heads': len(selected), 'unit': 'saved probability mass'})
    write_csv(tables / 'attention_first_query_mass.csv', query_context)
    head_max_sum_error = max(abs(row['total_mass'] - 1) for row in head_rows)
    max_context_error = max(abs(row['context_mass_error']) for row in head_rows)
    max_other_error = max(abs(row['other_mass_error']) for row in head_rows)
    formal = [row for row in stop_summary if row['stage'] in ('baseline', 'longbench_final', 'litm', 'doc2dial')]
    report = f'''# 注意力、生成停止和计时方法审核

我们在 PC 端流式读取恢复后的 `{CAMPAIGN}/attention.jsonl`，逐行连接对应 `attempts.jsonl.key`。本次读取 {len(diagnostic_rows)} 条注意力记录、{len(cases_csv)} 项配置、{len(head_rows)} 条 head-step 记录，发现 {len(anomalies)} 项结构/数值异常。逐层实际 step 集合与各配置的 output_tokens 对照，step=0 是请求尾 token，step>0 是已生成 token 再次送入网络后的解码查询。

## 1 记录内容和图像解释

采样层固定为 0、15、31，采样头固定为 0、8、16、24。每个保存查询只取该次 forward 的最后 token。上下文 logits 使用 `1/(sqrt(head_dim)*T)`，other logits 使用 `1/sqrt(head_dim)`；在各分支内部 softmax 后，分别乘实际 BF16 context/other 分支质量，恢复 token 概率。APE 对 context 的分支 log-sum-exp 乘 S×T，再与 other 分支共同 softmax。保存的顺序为共享前缀、按编码顺序拼接的全部文档块、请求、已解码历史。

全部保存权重的和偏离 1 的最大绝对值为 {head_max_sum_error:.8g}；context token 概率和与保存 branch mass 的最大绝对差为 {max_context_error:.8g}，other 对应差为 {max_other_error:.8g}。这些数值来自 BF16 分支质量与浮点重建。实际输出重建误差最大为 {max(r['max_abs_output_error'] for r in diagnostic_rows):.8g}，最大误差/保存容差为 {max(r['error_to_tolerance'] for r in diagnostic_rows):.6f}。

`attention_00.png` 至 `attention_11.png` 按四个指定头取平均，再按连续 64 个物理缓存 token 求和。横轴为缓存位置/64，纵轴含请求尾查询和随后实际解码步；白线表示前缀末端，红线表示文档末端；颜色是每箱概率质量。图像保留实际回答长度，所以各配置纵轴步数随生成结束变化。原始文档内容、支撑证据 token 区间和每个 token 的文本标签未保存在 attention 行内；本图解释为物理缓存概率分布。`attention_head_steps.csv` 给出各头分支、文档块和请求概率，`attention_bins.csv` 可重绘图像。

{md_table(cases_csv, ['case','dataset','L','N','T','S','score_f1_0_1','output_tokens','raw_records','max_output_reconstruction_error'])}

## 2 生成上限和停止规则

我们从实际 worker bundle 的每项任务读取 max_new_tokens，从模型 generation_config 读取 EOS IDs={sorted(stop_ids)}。记录的 output_tokens 包含最终 EOS token，文本 decode 时剔除特殊 token。上限命中定义为 output_tokens=max_new_tokens，另保留 EOS 恰好落在上限的交叉情况。生成长度字段和 token ID 数组逐题保留在原始记录；`generation_detail.csv` 导出每题的上限/停止/时延。{sum(r['max_new_tokens'] is None for r in details)} 条记录缺少可连接的任务上限。

模型 generation_config 中 do_sample=True、temperature=0.6、top_p=0.9；本 campaign 的 Engine.run 手工使用 logits.argmax 生成，每步选择最大概率 token，实际执行贪心解码。终止条件为当前 token 命中上述 EOS 集或步数达到任务上限。

{md_table(formal, ['stage','dataset','method','cache_mode','n','max_new_tokens','cap_hit_count','cap_hit_rate','eos_last_count','mean_output_tokens','median_output_tokens','p90_output_tokens'])}

`generation_cap_conditioned.csv` 同时给出上限命中组/提前结束组的时延、输出长度和记录分数，组间样本由实际停止行为形成。Doc2Dial 其中的记录分数是单条 BLEU，正式质量对比使用主报告中的 corpus BLEU。

## 3 计时范围和缓存条件

Engine.run 开始前执行 CUDA synchronize 并重置峰值统计；started 位于 prepare 前。TTFT 包括文档/请求分词、参数选择、KV 查询或构建、请求 forward、首 token argmax 和同步，total 还包括完整 decode 和最后文本解码。每次 token argmax 后同步；cache_build 为未命中时的编码区间并在结束同步，cache_get 为 CPU 缓存键命中判断。dynamic_selection_seconds 是 choose 或默认/固定参数分支的 CPU 区间，对所有方法都有该字段。选参耗时已计入 TTFT 与 total。

本次 PCW 与 APE 编码以逐块循环调用 forward 执行，batch=1，各块在同一 GPU 顺序计算，再合并独立编码的 KV。正式 cold 每次重新编码资料；search 允许同一资料的 T/S 扫描复用参数无关的 KV；hot 只复用同一进程、同一方法/参数/文档的 GPU KV，仍重新处理 history/question 的 suffix，重新 forward 和 greedy decode。worker 将 cold/hot output_token_ids 逐项比较。资料、历史和当前请求在模板中的实际内容由冻结数据提供。缓存身份不包含 suffix，处理不同后续问题时仍使用新 suffix。

峰值 allocated/reserved 是重置峰值之后、该方法进程的总显存峰值，包含模型与存活缓存；reserved 还包含分配器保留池。冷请求开始时可能仍有上一个请求缓存，清空 self.encoded 在 cold 编码分支发生。数值按 bytes 保存，主表以 GiB=bytes/2^30 展示。12 项 attention 配置启用 observer，内部浮点重建和写 JSON 增加开销，其时延在诊断组单列。

模型加载在各个 fresh worker 的 model_loaded 事件中独立计时，含加载和 CUDA 同步，不进入请求时延。模型加载事件 {len(loads)} 项，resume_cache_rebuilt 事件 {len(cache_rebuilds)} 项。

{md_table(load_summary, ['method','n','mean_seconds','median_seconds','p90_seconds'])}

## 4 来源与复现

- `restored/output/runs/{CAMPAIGN}/attempts.jsonl`：key、answer/output_token_ids、prefix_tokens/L/request_tokens、block_lengths、time/memory。
- 同目录 `workers/*.bundle.json`：逐项 max_new_tokens；`events.jsonl`：model_loaded、resume_cache_rebuilt。
- `restored/models/Llama-3.1-8B-Instruct/generation_config.json`：EOS IDs。
- `restored/output/runs/{CAMPAIGN}/snapshot/APE/main80/{{methods,worker,diagnostics,plots}}.py`：实际计时与注意力实现；同快照 `APE/ape/ape_llama.py`：上下文/other 分支计算。
- 执行：`python -B pc-analysis-output/scripts/analyze_attention.py`。本脚本使用现有 NumPy 和 Matplotlib，无模型/GPU调用。
'''
    (reports / 'ATTENTION_EFFICIENCY_AUDIT.md').write_text(report, encoding='utf-8')
    result = {'attention_records': len(diagnostic_rows), 'cases': len(cases_csv), 'head_steps': len(head_rows), 'anomalies': len(anomalies), 'generation_records': len(details), 'missing_caps': sum(r['max_new_tokens'] is None for r in details), 'model_loads': len(loads), 'cache_rebuilds': len(cache_rebuilds), 'maximum_probability_sum_error': head_max_sum_error}
    (tables / 'attention_execution_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
