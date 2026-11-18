"""Experiment-6 static figures from the derived tables; no raw data re-analysis."""
from pathlib import Path
import csv
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'pc-analysis-output/stages/06_longbench_final'
for key in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TEMP', 'TMP']:
    os.environ[key] = str(OUT / ('tmp' if key in ['TEMP', 'TMP'] else 'cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['sequential', 'pcw', 'ape_default', 'ape_fixed', 'ape_dynamic']
LABELS = ['顺序编码', 'PCW', '默认APE', '固定APE', '动态APE']
COLORS = ['#6c757d', '#9467bd', '#4c78a8', '#54a24b', '#f58518']
DATASETS = ['qasper', 'hotpotqa', 'musique', 'macro_equal_present_dataset']


def rows(name):
    with (OUT / 'tables' / name).open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'axes.unicode_minus': False})
    quality, paired, efficiency = rows('quality.csv'), rows('quality_paired.csv'), rows('efficiency_descriptive.csv')
    q = [r for r in quality if r['scope'] == 'test510']
    fig, ax = plt.subplots(figsize=(12, 5.2), layout='constrained')
    x = np.arange(4)
    for j, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        rr = [next(r for r in q if r['dataset'] == dataset and r['method'] == method) for dataset in DATASETS]
        yy = np.array([float(r['f1_percent']) for r in rr])
        err = [yy - [float(r['ci_low_percent']) for r in rr], [float(r['ci_high_percent']) for r in rr] - yy]
        ax.bar(x + (j - 2) * .15, yy, .15, label=label, color=color)
        ax.errorbar(x + (j - 2) * .15, yy, yerr=err, fmt='none', ecolor='black', capsize=2, linewidth=.7)
    ax.set_xticks(x, ['Qasper\n170题', 'HotpotQA\n170题', 'MuSiQue\n170题', '三数据集等权\n510题'])
    ax.set_ylabel('F1 ×100（误差线为95%来源组Bootstrap区间）')
    ax.set_ylim(0, 80)
    ax.set_title('共同510题的五方法质量', pad=36)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.12), fontsize=9)
    save(fig, 'five_method_quality')

    rr = [next(r for r in paired if r['scope'] == 'test510' and r['dataset'] == ds and r['reference_method'] == 'ape_fixed' and r['comparison_method'] == 'ape_dynamic') for ds in DATASETS]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    for j, r in enumerate(rr):
        value, lo, hi = [float(r[k]) for k in ['delta_pp', 'ci_low_pp', 'ci_high_pp']]
        ax.errorbar(value, 3 - j, xerr=[[value - lo], [hi - value]], fmt='o', capsize=4, color='#f58518' if j < 3 else '#222222')
        ax.annotate(f'{value:+.2f} [{lo:.2f}, {hi:.2f}]', (hi, 3 - j), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(range(4), ['等权宏平均 · 510题', 'MuSiQue · 170题', 'HotpotQA · 170题', 'Qasper · 170题'])
    ax.set_xlim(-5.5, 6)
    ax.set_xlabel('动态APE − 固定APE / F1百分点')
    ax.set_title('配对质量差值与95%置信区间（450个冻结来源组）')
    save(fig, 'dynamic_fixed_quality_difference')

    macro = {r['method']: float(r['f1_percent']) for r in q if r['dataset'] == 'macro_equal_present_dataset'}
    matrix = np.array([[macro[b] - macro[a] for a in METHODS] for b in METHODS])
    fig, ax = plt.subplots(figsize=(7.5, 6.2), layout='constrained')
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-4, vmax=4)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{matrix[i,j]:+.2f}', ha='center', va='center', color='white' if abs(matrix[i,j]) > 2.6 else 'black')
    ax.set_xticks(range(5), LABELS, rotation=20)
    ax.set_yticks(range(5), LABELS)
    ax.set_xlabel('参照方法'); ax.set_ylabel('比较方法')
    ax.set_title('共同510题：行方法 − 列方法的宏F1差值')
    fig.colorbar(im, ax=ax, label='F1百分点（对应配对区间见quality_paired.csv）')
    save(fig, 'all_method_delta_matrix')

    membership = rows('membership.csv')
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for length, color, label in [(4096, '#4c78a8', '映射4K/4 · 146题 · T/S=1/1'), (8192, '#f58518', '映射8K/4 · 76题 · T/S=.7/.7'), (16384, '#54a24b', '映射16K/4 · 288题 · T/S=.7/1')]:
        selected = [r for r in membership if int(r['mapped_L']) == length]
        ax.scatter([int(r['L']) for r in selected], [int(r['actual_N']) for r in selected], s=18, alpha=.5, color=color, label=label)
    ax.scatter([l for l in [4096, 8192, 16384] for n in [4, 8, 16]], [n for l in [4096, 8192, 16384] for n in [4, 8, 16]], marker='x', color='black', s=65, label='开发九格')
    ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384, 32768], ['1K', '2K', '4K', '8K', '16K', '32K'])
    ax.set_yticks([1, 2, 3, 4, 8, 16], ['1', '2', '3', '4', '8', '16'])
    ax.axhline(4, color='gray', linestyle='--', linewidth=.7)
    ax.set_ylim(.8, 21)
    ax.set_xlabel('实际文档长度L / tokenizer tokens（log2坐标）')
    ax.set_ylabel('实际编码块数N（log2坐标）')
    ax.set_title('正式输入与开发九格：510题均使用N=4列的映射')
    ax.legend(fontsize=9, loc='upper left')
    save(fig, 'actual_LN_and_rule_mapping')

    contribution = [r for r in rows('delta_contributions.csv') if r['stratum'].startswith('mapped_')]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    yy = np.array([float(r['contribution_to_overall_delta_pp']) for r in contribution])
    low = yy - [float(r['ci_low_pp']) for r in contribution]
    high = [float(r['ci_high_pp']) for r in contribution] - yy
    ax.bar(range(3), yy, color=['#4c78a8', '#f58518', '#54a24b'], width=.6)
    ax.errorbar(range(3), yy, yerr=[low, high], fmt='none', ecolor='black', capsize=5)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set_xticks(range(3), ['映射4K/4\n146题 · 1/1', '映射8K/4\n76题 · .7/.7', '映射16K/4\n288题 · .7/1'])
    ax.set_ylabel('对全510题宏F1差值的贡献 / 百分点')
    ax.set_title('分支贡献相加为动态−固定总体差值 −0.283个百分点')
    save(fig, 'rule_branch_delta_contributions')

    main_eff = [r for r in efficiency if r['scope'] == 'test510' and r['dataset'] == 'pooled_requests']
    for name, metrics, titles, units in [
        ('cold_latency', ['ttft_seconds', 'total_seconds'], ['首token时间', '总响应时间'], ['秒', '秒']),
        ('output_length_and_after_ttft', ['output_tokens', 'after_ttft_seconds'], ['生成长度（含EOS）', '首token之后的剩余响应时间'], ['token', '秒']),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for ax, metric, title, unit in zip(axes, metrics, titles, units):
            data = [next(r for r in main_eff if r['metric'] == metric and r['method'] == method) for method in METHODS]
            yy = np.array([float(r['mean']) for r in data])
            err = [yy - [float(r['mean_ci_low']) for r in data], [float(r['mean_ci_high']) for r in data] - yy]
            ax.bar(range(5), yy, color=COLORS)
            ax.errorbar(range(5), yy, yerr=err, fmt='none', ecolor='black', capsize=3)
            ax.set_xticks(range(5), LABELS, rotation=20)
            ax.set_ylabel(unit); ax.set_title(title)
        fig.suptitle('同510题冷请求均值（误差线为来源组Bootstrap均值95%区间）')
        save(fig, name)

    sample_pairs = rows('dynamic_fixed_per_sample.csv')
    fig, ax = plt.subplots(figsize=(8.5, 5.3), layout='constrained')
    xx = [int(r['dynamic_output_tokens']) - int(r['fixed_output_tokens']) for r in sample_pairs]
    yy = [float(r['ape_dynamic_total_seconds']) - float(r['ape_fixed_total_seconds']) for r in sample_pairs]
    color = [float(r['delta_pp']) for r in sample_pairs]
    im = ax.scatter(xx, yy, c=color, cmap='RdBu_r', vmin=-100, vmax=100, s=28, alpha=.65, edgecolors='none')
    ax.axhline(0, color='gray', linewidth=.6); ax.axvline(0, color='gray', linewidth=.6)
    ax.set_xlabel('动态−固定：生成长度差 / token（含EOS）')
    ax.set_ylabel('动态−固定：总响应时间差 / 秒')
    ax.set_title('逐题长度、时延与质量差异（510对；相同输出393对）')
    fig.colorbar(im, ax=ax, label='动态−固定 F1百分点')
    save(fig, 'paired_length_latency_quality')
    print('8 experiment-6 figures written as PNG and SVG.')


if __name__ == '__main__':
    main()
