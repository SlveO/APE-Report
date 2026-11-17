"""Stage-8 static figures from the derived Doc2Dial tables only.

No rescoring, model execution, evidence mutation, or other-stage analysis.
"""
from pathlib import Path
import csv
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'pc-analysis-output/stages/08_doc2dial'
for key in ['MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TEMP', 'TMP']:
    os.environ[key] = str(OUT / ('tmp' if key in ['TEMP', 'TMP'] else 'cache'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['sequential', 'ape_fixed', 'ape_dynamic']
LABEL = dict(zip(METHODS, ['顺序编码', '固定APE', '动态APE']))
COLORS = ['#4c78a8', '#54a24b', '#f58518']
PLOT_NAMES = [
    'corpus_bleu', 'quality_differences', 'cold_hot_latency', 'cache_speedups',
    'output_length_and_caps', 'history_subgroups', 'actual_LN_mapping',
    'paired_length_latency',
]


def rows(name):
    with (OUT / 'tables' / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def error_values(records, point='mean', low='mean_ci_low', high='mean_ci_high'):
    y = np.array([float(r[point]) for r in records])
    lo = np.array([float(r[low]) for r in records])
    hi = np.array([float(r[high]) for r in records])
    return y, np.array([y - lo, hi - y])


def main():
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10, 'axes.unicode_minus': False})
    quality = rows('quality.csv')
    quality_pairs = rows('quality_paired.csv')
    efficiency = rows('efficiency_descriptive.csv')
    performance = rows('efficiency_paired.csv')
    limits = rows('generation_limits.csv')
    membership = rows('membership.csv')
    paired_targets = rows('dynamic_fixed_per_target.csv')

    def q(method, scope='all500', cluster='dialogue', cache='cold'):
        return next(r for r in quality if r['scope'] == scope and r['cluster_unit'] == cluster and r['method'] == method and r['cache_mode'] == cache)

    def e(method, cache, metric):
        return next(r for r in efficiency if r['scope'] == 'all500' and r['cluster_unit'] == 'dialogue' and r['method'] == method and r['cache_mode'] == cache and r['metric'] == metric)

    def cache_perf(method, metric):
        return next(r for r in performance if r['scope'] == 'all500' and r['cluster_unit'] == 'dialogue' and r['comparison_type'] == 'cache' and r['reference_method'] == method and r['metric'] == metric)

    # 1. The cold/hot responses are identical; show the shared corpus score once.
    rr = [q(m) for m in METHODS]
    y, err = error_values(rr, 'corpus_bleu', 'ci_low', 'ci_high')
    fig, ax = plt.subplots(figsize=(8.5, 5), layout='constrained')
    ax.bar(range(3), y, color=COLORS, width=.62)
    ax.errorbar(range(3), y, yerr=err, fmt='none', ecolor='black', capsize=5)
    for j, r in enumerate(rr):
        ax.text(j, float(r['ci_high']) + .08, f"{float(r['corpus_bleu']):.3f}", ha='center')
    ax.set_xticks(range(3), [LABEL[m] for m in METHODS])
    ax.set_ylabel('corpus BLEU / 0–100')
    ax.set_ylim(0, 4.5)
    ax.set_title('Doc2Dial共同500目标：冷/热答案一致，共用同一质量值\n误差线：366个对话簇配对Bootstrap 95%区间')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    save(fig, 'corpus_bleu')

    # 2. Method comparisons share the identical target/condition sample weights.
    method_pairs = [('sequential', 'ape_fixed'), ('sequential', 'ape_dynamic'), ('ape_fixed', 'ape_dynamic')]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout='constrained')
    all_lows, all_highs = [], []
    for offset, cluster, color, marker, label in [(.11, 'dialogue', '#f58518', 'o', '对话聚类 · 366簇'), (-.11, 'document', '#4c78a8', 'D', '资料聚类 · 172簇')]:
        for j, (a, b) in enumerate(method_pairs):
            r = next(r for r in quality_pairs if r['scope'] == 'all500' and r['cluster_unit'] == cluster and r['comparison_type'] == 'method' and r['reference_method'] == a and r['comparison_method'] == b and r['reference_cache'] == 'cold')
            v, lo, hi = [float(r[k]) for k in ['delta_bleu', 'ci_low', 'ci_high']]
            ax.errorbar(v, 2 - j + offset, xerr=[[v - lo], [hi - v]], fmt=marker, color=color, capsize=4, label=label if j == 0 else None)
            all_lows.append(lo); all_highs.append(hi)
    ax.set_yticks([2, 1, 0], [LABEL[b] + ' − ' + LABEL[a] for a, b in method_pairs])
    ax.axvline(0, color='gray', linestyle='--', linewidth=.8)
    ax.set_xlim(min(all_lows) - .08, max(all_highs) + .08)
    ax.set_xlabel('配对corpus BLEU差：比较方法 − 参照方法 / BLEU点')
    ax.set_title('方法差值及95%区间：对话为主，资料聚类作敏感性\n每次共同抽样后重新累加n-gram统计并计算语料BLEU')
    ax.legend(loc='lower left', fontsize=9)
    save(fig, 'quality_differences')

    # 3. Cold/hot latency statistics are computed on each condition's 500 requests.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['ttft_seconds', 'total_seconds'], ['首token时间', '总响应时间']):
        for offset, cache, label, alpha, hatch in [(-.18, 'cold', '冷请求', 1, None), (.18, 'hot', 'GPU资料KV热请求', .48, '//')]:
            data = [e(m, cache, metric) for m in METHODS]
            y, err = error_values(data)
            ax.bar(np.arange(3) + offset, y, width=.34, color=COLORS, alpha=alpha, hatch=hatch, label=label)
            ax.errorbar(np.arange(3) + offset, y, yerr=err, fmt='none', ecolor='black', capsize=3, linewidth=.8)
        ax.set_xticks(range(3), [LABEL[m] for m in METHODS])
        ax.set_ylabel('秒'); ax.set_title(title)
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle('相同500目标的冷/热请求均值（95%对话聚类Bootstrap区间）')
    save(fig, 'cold_hot_latency')

    # 4. These are ratios of mean times, not means of per-request ratios.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout='constrained')
    for ax, metric, title in zip(axes, ['ttft_seconds', 'total_seconds'], ['首token时间', '总响应时间']):
        data = [cache_perf(m, metric) for m in METHODS]
        y, err = error_values(data, 'ratio_of_means', 'ratio_of_means_ci_low', 'ratio_of_means_ci_high')
        ax.errorbar(range(3), y, yerr=err, fmt='o', color='#222222', capsize=5)
        for j, value in enumerate(y):
            ax.annotate(f'{value:.4f}×', (j, value), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
        ax.axhline(1, color='gray', linestyle='--', linewidth=.8)
        ax.set_xticks(range(3), [LABEL[m] for m in METHODS])
        ax.set_xlim(-.35, 2.35)
        ax.set_ylabel('冷请求均值 / 热请求均值')
        ax.set_title(title)
        if metric == 'ttft_seconds':
            ax.set_ylim(.9, max(float(r['ratio_of_means_ci_high']) for r in data) + .4)
        else:
            ax.set_ylim(.998, max(float(r['ratio_of_means_ci_high']) for r in data) + .012)
    fig.suptitle('GPU资料KV复用：均值耗时比及95%对话聚类区间（大于1表示热请求更快）', fontsize=12)
    save(fig, 'cache_speedups')

    # 5. 13a words and model generation tokens have explicitly distinct axes.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    generated = [float(q(m)['system_length_13a']) / 500 for m in METHODS]
    reference = [float(q(m)['reference_length_13a']) / 500 for m in METHODS]
    axes[0].bar(np.arange(3) - .17, generated, .32, color=COLORS, label='生成文本')
    axes[0].bar(np.arange(3) + .17, reference, .32, color='#b9bdc5', label='参考答案')
    for j in range(3):
        axes[0].text(j - .17, generated[j] + 2, f'{generated[j]:.2f}', ha='center', fontsize=9)
        axes[0].text(j + .17, reference[j] + 2, f'{reference[j]:.2f}', ha='center', fontsize=9)
    axes[0].set_xticks(range(3), [LABEL[m] for m in METHODS])
    axes[0].set_ylabel('每目标平均13a分词数（两类文本同单位）')
    axes[0].set_title('生成与参考长度')
    data = [next(r for r in limits if r['method'] == m and r['cache_mode'] == 'cold') for m in METHODS]
    y = [float(r['cap_fraction']) * 100 for r in data]
    axes[1].bar(range(3), y, color=COLORS, width=.6)
    for j, r in enumerate(data):
        axes[1].text(j, y[j] + 2, f"{r['cap_hits']}/500\n{y[j]:.1f}%", ha='center', fontsize=9)
    axes[1].set_xticks(range(3), [LABEL[m] for m in METHODS])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('达到128模型token上限的请求 / %')
    axes[1].set_title('生成上限命中率（冷/热相同）')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=2, fontsize=9)
    fig.suptitle('Doc2Dial长输出：语料长度与128-token停止条件分别登记')
    save(fig, 'output_length_and_caps')

    # 6. The two flagged groups overlap; subgroup corpora are scored separately.
    scopes = [('all500', '全部目标', 500), ('clean_user_no_repeat', '上一轮user且参考未重复', 467), ('consecutive_agent', '连续agent', 28), ('reference_repeat', '参考在历史精确重复', 7)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.7), layout='constrained', sharey=True)
    for ax, (scope, title, count) in zip(axes, scopes):
        data = [q(m, scope) for m in METHODS]
        y, err = error_values(data, 'corpus_bleu', 'ci_low', 'ci_high')
        ax.bar(range(3), y, color=COLORS, width=.6)
        ax.errorbar(range(3), y, yerr=err, fmt='none', ecolor='black', capsize=3, linewidth=.8)
        ax.set_xticks(range(3), [LABEL[m] for m in METHODS], rotation=25)
        ax.set_title(f'{title}\n{count}目标 · {data[0]["n_clusters"]}对话簇', fontsize=10)
        ax.set_ylim(0, 4.6); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    axes[0].set_ylabel('各子语料独立重算corpus BLEU / 0–100')
    overlap = sum(r['consecutive_agent_target'] == r['reference_repeated_in_history'] == 'True' for r in membership)
    fig.suptitle(f'历史条件分层及95%对话聚类区间；连续agent与参考重复组有{overlap}目标重叠', fontsize=12)
    save(fig, 'history_subgroups')

    # 7. Original document count remains one; encoded N is the plotted quantity.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained', gridspec_kw={'width_ratios': [1.45, 1]})
    counts = []
    for length, color in [(4096, '#4c78a8'), (8192, '#f58518')]:
        selected = [r for r in membership if int(r['mapped_L']) == length]
        counts.append(len(selected))
        axes[0].scatter([int(r['L']) for r in selected], [int(r['actual_N']) for r in selected], color=color, s=22, alpha=.5)
    axes[0].scatter([l for l in [4096, 8192, 16384] for n in [4, 8, 16]], [n for l in [4096, 8192, 16384] for n in [4, 8, 16]], color='black', marker='x', s=40)
    axes[0].set_xscale('log', base=2); axes[0].set_yscale('log', base=2)
    axes[0].set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384], ['256', '512', '1K', '2K', '4K', '8K', '16K'])
    axes[0].set_yticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'])
    axes[0].set_xlim(160, 23000); axes[0].set_ylim(.85, 20)
    axes[0].axhline(4, color='gray', linestyle='--', linewidth=.7)
    axes[0].set_xlabel('实际文档长度L / 模型token（log2）')
    axes[0].set_ylabel('实际编码块数N（log2）')
    axes[0].set_title('实际输入与开发九格（黑叉）')
    axes[1].bar(range(2), counts, color=['#4c78a8', '#f58518'], width=.6)
    for j, count in enumerate(counts):
        axes[1].text(j, count + 8, str(count), ha='center')
    axes[1].set_xticks(range(2), ['4K/4 · T/S=1/1\n实际N=1', '8K/4 · T/S=.7/.7\n实际N=2'])
    axes[1].set_ylim(0, 560); axes[1].set_ylabel('目标数 / 500')
    axes[1].set_title('冻结规则分支使用数量')
    n2 = [r for r in membership if int(r['actual_N']) == 2]
    n2_docs = len({r['document_cluster'] for r in n2})
    n2_dialogues = len({r['dialogue_cluster'] for r in n2})
    axes[1].text(.98, .92, f'N=2的{len(n2)}目标\n来自{n2_docs}篇资料、{n2_dialogues}个对话', transform=axes[1].transAxes, ha='right', va='top', fontsize=9)
    fig.suptitle('每目标原始资料数为1；自然分块后493目标N=1、7目标N=2', fontsize=12)
    save(fig, 'actual_LN_mapping')

    # 8. Same-target length/time differences, without treating sentence scores as corpus effects.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained', sharex=True, sharey=True)
    for ax, cache, title in zip(axes, ['cold', 'hot'], ['冷请求', '同进程GPU资料KV热请求']):
        for equal, color, label in [('True', '#8c8c8c', '输出token相同'), ('False', '#4c78a8', '输出token不同')]:
            selected = [r for r in paired_targets if r['same_tokens'] == equal]
            xx = [int(r['dynamic_output_tokens']) - int(r['fixed_output_tokens']) for r in selected]
            yy = [float(r[f'dynamic_{cache}_total_s']) - float(r[f'fixed_{cache}_total_s']) for r in selected]
            ax.scatter(xx, yy, color=color, s=16 if equal == 'False' else 12, alpha=.5, edgecolors='none', label=f'{label} · {len(selected)}对')
        ax.axhline(0, color='gray', linewidth=.6); ax.axvline(0, color='gray', linewidth=.6)
        ax.set_xlabel('动态−固定：输出长度差 / 模型token')
        ax.set_ylabel('动态−固定：总响应时间差 / 秒')
        ax.set_title(title)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('动态−固定：相同500目标的输出长度与响应时间差')
    save(fig, 'paired_length_latency')
    print('Stage 8: eight PNG/SVG figure pairs written: ' + ', '.join(PLOT_NAMES))


if __name__ == '__main__':
    main()
