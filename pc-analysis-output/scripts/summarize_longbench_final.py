"""Experiment-6 Chinese report and audit from the saved paired-analysis tables."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'pc-analysis-output'
OUT = BASE / 'stages/06_longbench_final'
METHODS = ['sequential', 'pcw', 'ape_default', 'ape_fixed', 'ape_dynamic']
LABEL = dict(zip(METHODS, ['顺序编码', 'PCW', '默认APE', '固定APE', '动态APE']))
DS = ['qasper', 'hotpotqa', 'musique']
MACRO = 'macro_equal_present_dataset'


def readcsv(name):
    with (OUT / 'tables' / name).open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def csvout(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / 'tables' / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def md(headers, rows):
    def text(v):
        return str(v).replace('|', '\\|').replace('\n', '<br>')
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + '\n'.join('| ' + ' | '.join(text(v) for v in row) + ' |' for row in rows) + '\n'


def main():
    result = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    env = json.loads((OUT / 'environment.json').read_text(encoding='utf-8'))
    quality, paired, efficiency, performance = [readcsv(name) for name in ['quality.csv', 'quality_paired.csv', 'efficiency_descriptive.csv', 'efficiency_paired.csv']]
    membership, cases, limits, cap_groups, length_groups, workers, contributions, bootstrap = [readcsv(name) for name in ['membership.csv', 'selected_cases.csv', 'generation_limits.csv', 'generation_cap_conditioned.csv', 'paired_output_length_groups.csv', 'model_load_workers.csv', 'delta_contributions.csv', 'bootstrap_diagnostics.csv']]

    def q(dataset, method, scope='test510'):
        return next(r for r in quality if r['scope'] == scope and r['dataset'] == dataset and r['method'] == method)
    def pair(dataset, a='ape_fixed', b='ape_dynamic', scope='test510'):
        return next(r for r in paired if r['scope'] == scope and r['dataset'] == dataset and r['reference_method'] == a and r['comparison_method'] == b)
    def e(method, metric, dataset='pooled_requests', scope='test510'):
        return next(r for r in efficiency if r['scope'] == scope and r['dataset'] == dataset and r['method'] == method and r['metric'] == metric)
    def perf(metric, a='ape_fixed', b='ape_dynamic', dataset='pooled_requests', scope='test510'):
        return next(r for r in performance if r['scope'] == scope and r['dataset'] == dataset and r['reference_method'] == a and r['comparison_method'] == b and r['metric'] == metric)
    def interval(r, low='ci_low_pp', high='ci_high_pp', digits=2):
        return f"[{float(r[low]):.{digits}f}, {float(r[high]):.{digits}f}]"
    def beijing(stamp):
        return datetime.fromisoformat(stamp).astimezone(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

    main_quality = [q(dataset, method) for dataset in [*DS, MACRO] for method in METHODS]
    main_pairs = [r for r in paired if r['scope'] == 'test510' and r['dataset'] in [*DS, MACRO]]
    main_eff = [r for r in efficiency if r['scope'] == 'test510' and r['dataset'] in [*DS, 'pooled_requests']]
    primary = [pair(dataset) for dataset in [*DS, MACRO]]
    csvout('main_quality.csv', main_quality)
    csvout('main_quality_paired.csv', main_pairs)
    csvout('main_efficiency.csv', main_eff)
    csvout('primary_dynamic_fixed.csv', primary)

    inputs = []
    for dataset in [*DS, 'ALL']:
        selected = [r for r in membership if dataset == 'ALL' or r['dataset'] == dataset]
        lengths = np.array([int(r['L']) for r in selected])
        originals = [int(r['original_document_count']) for r in selected]
        inputs.append({'dataset': dataset, 'n_questions': len(selected), 'L_min': lengths.min(), 'L_mean': lengths.mean(), 'L_median': np.median(lengths), 'L_p90': np.quantile(lengths, .9), 'L_max': lengths.max(), 'N_outside_grid': sum(r['N_outside_grid'] == 'True' for r in selected), 'L_below_grid': sum(r['L_below_grid'] == 'True' for r in selected), 'L_above_grid': sum(r['L_above_grid'] == 'True' for r in selected), 'original_documents_min': min(originals), 'original_documents_max': max(originals), 'unit': 'document tokens; paired questions; original document count'})
    csvout('input_L_statistics.csv', inputs)
    mapping_rows = []
    for length in [4096, 8192, 16384]:
        scope = f'mapped_{length}_4'
        row = pair('pooled_requests', scope=scope)
        contrib = next(r for r in contributions if r['stratum'] == f'mapped_{length}')
        mapping = next(r for r in readcsv('rule_mapping_counts.csv') if r['dataset'] == 'ALL' and int(r['mapped_L']) == length)
        mapping_rows.append({'mapped_L': length, 'mapped_N': 4, 'T': mapping['T'], 'S': mapping['S'], 'n_questions': row['n_questions'], **{dataset + '_n': sum(r['dataset'] == dataset and int(r['mapped_L']) == length for r in membership) for dataset in DS}, 'fixed_f1_percent_pooled': q('pooled_requests', 'ape_fixed', scope)['f1_percent'], 'dynamic_f1_percent_pooled': q('pooled_requests', 'ape_dynamic', scope)['f1_percent'], 'within_branch_delta_pp': row['delta_pp'], 'within_branch_ci_low_pp': row['ci_low_pp'], 'within_branch_ci_high_pp': row['ci_high_pp'], 'wins': row['wins_comparison'], 'ties': row['ties'], 'losses': row['losses_comparison'], 'contribution_to_overall_delta_pp': contrib['contribution_to_overall_delta_pp'], 'contribution_ci_low_pp': contrib['ci_low_pp'], 'contribution_ci_high_pp': contrib['ci_high_pp'], 'unit': 'within branch pooled paired F1pp; contribution to full510 equal-dataset macro F1pp'})
    csvout('mapping_interpretation.csv', mapping_rows)

    provenance = [
        ('样本范围', 'attempts.jsonl:key.stage/sample_id/method + snapshot frozen.json:orders.test', 'baseline三方法与final两方法; 按同ID严格连接', '每DS170题; 总510题/2550条; 每法510', 'membership.csv; scope_counts.csv'),
        ('质量主表', 'attempts.jsonl:score,reference_answers + frozen longbench.json:answers', '保存score; 参考严格连接; 空生成保留零分; 成功first attempt', '每DS题目均值; 宏平均三DS等权; 展示乘100', 'main_quality.csv; per_request_records.csv'),
        ('质量差及胜平负', '同ID两方法score', '同510配对; 差=比较−参照; 绝对差≤1e-12记平', 'F1百分点; 胜/平/负按题数; 95%配对聚类Bootstrap', 'main_quality_paired.csv; primary_dynamic_fixed.csv'),
        ('置信区间', 'snapshot longbench.json:group/support_documents', 'DS内有放回抽来源组, 同组五法共同保留; 参数冻结', 'seed20261116; 5000次; 2.5%/97.5%分位数; 450组主/413组件敏感性', 'bootstrap_diagnostics.csv; bootstrap_membership.csv'),
        ('规则映射', 'attempts.jsonl:L,N,T,S + rule.json:cells,mapping', '用实际N; log2空间最近格; 平距较小L/N', '题数/510; 自然块长≤7500; 映射贡献采用完整主表分母', 'rule_mapping_counts.csv; mapping_interpretation.csv; membership.csv'),
        ('资料重合', 'snapshot longbench.json:input_documents/support_documents; dev=explore∪validation', '按公开资料ID求交集; 与前置审核affected IDs核对', '支撑重合29题; 单篇输入重合59题; 绝对质量分层', 'development_overlap_affected_ids.csv; quality.csv'),
        ('时间', 'attempts.jsonl:ttft_seconds,total_seconds,cache_build_seconds,cache_get_seconds,dynamic_selection_seconds', 'cold miss; 关闭注意力采集; 同ID配对', '均值/中位数/P90/均值95%CI; s; 选参展示µs', 'main_efficiency.csv; efficiency_paired.csv'),
        ('加速比', '同ID两方法ttft/total', '逐对比值=参照耗时/比较耗时; 同一配对组重采样', '比值均值/中位数/P90; 均值耗时之比另列; 无量纲', 'efficiency_paired.csv'),
        ('显存', 'attempts.jsonl:peak_allocated_bytes,peak_reserved_bytes', '请求前重置峰值计数; 含模型、开始时存活缓存和本次请求', 'GiB=bytes/2^30; allocated/reserved分别报告', 'main_efficiency.csv'),
        ('生成停止', 'output_tokens/output_token_ids + frozen max_new_tokens + model generation_config:eos_token_id', 'cap=token数等于上限; EOS=最后ID属于停止集合', 'Qasper128; HotpotQA/MuSiQue32; EOS计入长度', 'generation_limits.csv; generation_cap_records.csv'),
        ('案例', 'dynamic_fixed_per_sample.csv:score,answer,reference,params,source_line', '最大正差/负差/零差; 同分按sample_id字典序', '全局及各DS各类型; 保留原答案', 'selected_cases.csv; case_type_counts.csv'),
        ('模型加载/执行顺序', '前置审核runtime_workers.csv, 来自events.jsonl', '只取baseline三worker和longbench_final两worker', '加载s; worker完整起止时间; 排除加载于请求时延', 'model_load_workers.csv'),
    ]
    csvout('provenance.csv', [dict(zip(['result', 'raw_fields', 'filter_and_join', 'aggregation_units_denominator', 'output_tables'], r)) for r in provenance])
    dictionary = [
        ('task_N / actual_N', '任务键N为null时由自然分块产生actual_N; 实际参数和块数取结果记录', 'task key / block count'),
        ('L', '完整文档token长度; 不含prefix/request; natural chunks最长7500', 'tokenizer tokens'),
        ('original_document_count', '原公开输入资料数, 与分块后编码块数分别登记', 'documents'),
        ('dataset=macro_equal_present_dataset', '对datasets_in_scope内数据集等权; 主表为3个, 分支4K仅Qasper/HotpotQA', 'F1 weighted mean'),
        ('dataset=pooled_requests', '按该scope实际题数比例保留数据集权重; 质量分层及效率采用该口径', 'request/question-weighted mean'),
        ('bootstrap subgroup', '仅对子组内原始来源组抽样; 分母随组大小变化; 五法共同保留', 'conditional source composition variation'),
        ('context cache timing', 'cold miss每次重新编码资料; 获取时间为缓存条件查询分支', 'seconds'),
        ('after_ttft_seconds', 'total_seconds−ttft_seconds, 含首token后生成及最终文本解码等剩余过程', 'seconds'),
        ('dynamic_selection_seconds', 'prepare之后选择/固定参数分支的计时; 动态实际查冻结规则', 'seconds; report µs'),
        ('CI scope', '固定模型/模板/规则下的问题及来源组构成变动; 当前每方法一次连续worker运行', 'percentile interval; no repeated timing runs'),
    ]
    csvout('field_dictionary.csv', [dict(zip(['field', 'definition', 'unit_or_scope'], r)) for r in dictionary])

    flags = [
        {'phenomenon': '开发与正式实际N分布不同', 'evidence': '开发N4/8/16; 测试N1/2/3=177/104/229; 510/510网格外', 'affected': '全部动态规则评价', 'handling': '冻结最近格映射; 单列实际N、原资料数和三个使用分支', 'followup': '在开发数据用自然分块N1–3搜索/验证后重新冻结, 用新增独立测试评价'},
        {'phenomenon': '正式动态仅使用三个分支', 'evidence': '4K/4=146;8K/4=76;16K/4=288; 同参288条答案一致', 'affected': '动态相对固定的收益来源', 'handling': '按分支和实际N统计; 对全510的贡献逐项保存', 'followup': '针对8K/4的退化在证据完整开发集复核; 当前保持原规则'},
        {'phenomenon': '开发/测试单篇支撑资料重合', 'evidence': '24篇支撑资料,29个MuSiQue测试ID; 单篇输入资料涉及59测试ID', 'affected': '独立测试样本解释及源资料相关性', 'handling': '保留冻结510主表; 支撑重合29/不重合141的MuSiQue分层;450组与413连通组件CI', 'followup': '新增按单篇支撑资料隔离的独立测试'},
        {'phenomenon': '毫秒级计时差存在于连续worker运行', 'evidence': '五方法各一个worker时段; 同参/同token288题也有计时差', 'affected': '动态TTFT平均缩短2.33ms的归因', 'handling': '报告时间差、输出长度、选参µs和worker顺序; CI反映样本来源变化', 'followup': '同参同输出请求随机交错重复测时, 分离选参及执行时序影响'},
        {'phenomenon': '短回答任务达到生成上限', 'evidence': '41/2550; MuSiQue5–8/170/方法; Qasper0', 'affected': '质量与总响应时间', 'handling': '保留原分数; EOS/cap分组; 保存全部41条原答案与来源', 'followup': '在开发集对相同问题成对增加生成上限, 检查截断与重复输出'},
        {'phenomenon': '硬件登记缺少campaign遥测', 'evidence': '方案RTX4090 48GB; 记录物理GPU0/BF16/batch1及软件版本; manifest无GPU型号遥测', 'affected': '跨硬件效率对照', 'handling': '分别列方案登记与实际执行配置, 保留单campaign效率', 'followup': '未来实验保存实际GPU型号/显存/驱动运行元数据'},
    ]
    csvout('audit_findings.csv', flags)

    report = '# 实验第六阶段：LongBench五方法正式比较\n\n'
    report += '## 摘要\n\n我们在MAIN80冻结测试集上比较顺序编码、PCW、默认APE、调优固定APE和动态APE。Qasper、HotpotQA、MuSiQue各170题，共510题和2550条冷请求生成；三方法基线复用同ID的1530条记录，固定与动态APE来自正式阶段1020条记录。我们引用已审核的逐题F1，以数据集等权汇总，采用5000次配对来源组Bootstrap计算区间。五方法宏F1依次为46.21、43.67、43.86、43.09和42.81。动态相对固定下降0.28个百分点，95%区间为[-1.68,1.09]，胜平负47/415/48。实际编码块数均为1–3，全部映射至开发九格N=4列；4K分支贡献+0.72个百分点，8K分支贡献-1.00个百分点，16K分支同参且答案持平。动态平均TTFT缩短2.33毫秒，总响应增加10.90毫秒，输出长度由10.82增至11.23 token。测试与开发共享支撑资料的29题单列，改用来源连通组件后区间相近。结果显示，本次冻结动态规则总体点估计略低于固定规则，其改善与退化集中在两个短中长度映射分支，效率需结合输出长度、缓存条件和实际逐块实现阅读。\n\n关键词：LongBench；自适应并行编码；动态参数；配对统计；冷请求效率\n\n'
    report += '## 1 研究问题与设置\n\n我们评价五种编码方法在共同测试ID上的质量和冷请求效率，重点检查冻结动态规则相对固定T/S=0.7/1.0的变化及来源。默认APE为0.9/0.9，动态按文档长度L、实际编码块数N查原冻结九格。测试分数仅用于本阶段评价，参数和回退规则保留实验第三阶段冻结结果。\n\n'
    report += '模型为Llama-3.1-8B-Instruct，执行物理GPU0、bfloat16、batch1、greedy生成，生成种子配置42。服务器记录Python3.10.20、PyTorch2.6.0+cu124、Transformers4.44.2、FlashAttention2.7.4.post1。方案登记RTX4090 48GB；本campaign缺GPU型号、总显存、驱动及功率遥测，具体登记见server_environment.csv。当前结果来自单模型、单campaign。\n\n'
    report += '五方法共用冻结完整公开文档、问题、参考和聊天模板。自然分块最大7500 tokenizer tokens，逐块长度与五方法结果一致。顺序编码先处理prefix+全部文档；PCW各窗以一个BOS独立编码，合并时保留唯一BOS，并将prefix去BOS后移至请求；APE先编码共享prefix，再按块编码文档并合并KV。PCW/APE在本实现中通过Python循环逐块调用GPU forward。最终请求位置和模板适配按实际代码登记。\n\n'
    report += '全部2550条为cold miss，各请求重新构建资料KV，计时关闭注意力观察器。TTFT从请求开始计，包含prepare分词/构造、选参、缓存查询/编码和首token；总响应继续覆盖完整生成及文本解码。模型加载独立记录。质量使用LongBench英文token F1：规范化后计算token重叠，多参考取最高值；原score为0–1，展示乘100。评分实现及官方来源沿用[前置审核](../../01_audit/reports/AUDIT.md)，本阶段评分重算0次。\n\n'
    report += '## 2 样本连接、冻结顺序与统计\n\n1020条longbench_final记录加上同ID的1530条baseline记录，严格形成510题五方法配对，每数据集170题。全部成功且attempt=1，重复成功、失败、空答案、缺参考及非法数值均为0。源任务键N为null时，实际N由自然分块产生；选参使用结果N与methods.prepare实际块数。membership.csv及per_request_records.csv保存原键、实际参数、逻辑标签和源行。\n\n'
    baseline_reference = readcsv('baseline600_reference.csv')
    report += md(['三方法', '完整600题基线宏F1', '共同510题宏F1'], [[LABEL[m], f"{float(next(r['f1_percent'] for r in baseline_reference if r['dataset']=='macro_equal_dataset' and r['method']==m)):.2f}", f"{float(q(MACRO,m)['f1_percent']):.2f}"] for m in METHODS[:3]])
    report += '\n完整600题基线由[实验第一阶段](../../01_longbench_baseline/reports/BASELINE_REPORT.md)交付。当前510题为冻结测试划分，每个方法按同ID比较；600与510的分母和样本构成分别保留。\n\n'
    report += '主区间按数据集内原论文/完整证据组重采样，Qasper116、HotpotQA170、MuSiQue164组，共450组；每次有放回抽取同数来源组，保留组内所有题和五方法。组大小变化时按该次抽到的题数归一，三个数据集F1再等权平均。seed20261116，5000次，取2.5%与97.5%分位数。每个区间对应一个指定对比。效率汇总保留原数据集请求比例；分支及N子组的CI仅反映该子组内来源构成变化。\n\n'
    report += '另把同冻结组或共享任一支撑资料的测试题连通，得到116/169/128、合计413个来源组件，在同510题及同冻结规则下作敏感性。区间反映所选问题和资料来源的样本变动；本campaign每方法各有一个连续worker时段，重复执行的时延变化尚缺观测。全部组成员及随机种子子序列见bootstrap_membership.csv、bootstrap_diagnostics.csv。\n\n'
    rule = json.loads((OUT / 'sources/original_frozen_rule.json').read_text(encoding='utf-8'))
    report += f"规则冻结时间为北京时间{beijing(rule['frozen_at'])}，固定/动态正式worker分别于{beijing(next(r['started_at'] for r in workers if r['stage']=='longbench_final' and r['method']=='ape_fixed'))}、{beijing(next(r['started_at'] for r in workers if r['method']=='ape_dynamic'))}启动。三方法基线先于规则冻结生成；候选及验证选择的输入范围由实验第三阶段审核，正式阶段参数与rule.json逐题吻合。\n\n"

    report += '## 3 五方法质量\n\n'
    report += md(['方法', 'Qasper/170', 'HotpotQA/170', 'MuSiQue/170', '等权宏F1/510', '宏F1的95%区间'], [[LABEL[m], *[f"{float(q(ds,m)['f1_percent']):.2f}" for ds in DS], f"{float(q(MACRO,m)['f1_percent']):.2f}", interval(q(MACRO,m), 'ci_low_percent', 'ci_high_percent')] for m in METHODS])
    report += '\n![共同测试五方法质量](../figures/five_method_quality.png)\n\n顺序编码的宏F1点估计最高，PCW与默认APE接近；调优固定和动态APE的宏F1分别低于默认APE0.77、1.06个百分点，对应区间[-2.82,1.33]、[-3.13,1.08]。动态相对顺序下降3.40个百分点，区间[-6.10,-0.66]。全部10个方法对在各数据集及宏平均上的差值、区间、胜平负见main_quality_paired.csv。\n\n'
    report += md(['数据集', '动态−固定/百分点', '95%区间', '胜/平/负', '分母'], [[ds if ds!=MACRO else '三数据集等权', f"{float(pair(ds)['delta_pp']):+.3f}", interval(pair(ds)), f"{pair(ds)['wins_comparison']}/{pair(ds)['ties']}/{pair(ds)['losses_comparison']}", pair(ds)['n_questions']] for ds in [*DS,MACRO]])
    report += '\nQasper点估计略升，HotpotQA下降，MuSiQue只有1题分数提高、169题持平。全体宏F1下降0.283个百分点，区间包含两种方向的变动。改用413个来源连通组件，差值点估计相同、区间为' + interval(pair(MACRO, scope='test510_support_component_sensitivity')) + '个百分点。\n\n![动态相对固定的配对区间](../figures/dynamic_fixed_quality_difference.png)\n\n![全部方法宏平均差值](../figures/all_method_delta_matrix.png)\n\n'

    report += '## 4 实际L/N、网格映射与分支贡献\n\n'
    nrows = readcsv('actual_N_distribution.csv')
    report += md(['数据集', 'N=1', 'N=2', 'N=3', '长度范围/token'], [[ds if ds!='ALL' else '合计', *[next(r['n_questions'] for r in nrows if r['dataset']==ds and r['actual_N']==str(n)) for n in [1,2,3]], f"{next(r['L_min'] for r in inputs if r['dataset']==ds)}–{next(r['L_max'] for r in inputs if r['dataset']==ds)}"] for ds in [*DS,'ALL']])
    report += '\n开发网格为L=4K/8K/16K、N=4/8/16，正式510题的N均低于开发范围。L低于4K的74题、高于16K的3题，合计77/510（15.10%）；低端Qasper69、HotpotQA5，高端3题均Qasper。规则按log₂(L)、log₂(actual_N)欧氏距离取最近格，所有输入落到N=4列。\n\n![实际输入和开发九格](../figures/actual_LN_and_rule_mapping.png)\n\n'
    report += md(['映射格', '动态T/S', '题数 Q/H/M', '分支内动态−固定/百分点', '分支内95%区间', '对总体差值贡献/百分点'], [[f"{r['mapped_L']}/4", f"{r['T']}/{r['S']}", f"{r['n_questions']} ({r['qasper_n']}/{r['hotpotqa_n']}/{r['musique_n']})", f"{float(r['within_branch_delta_pp']):+.3f}", f"[{float(r['within_branch_ci_low_pp']):.2f},{float(r['within_branch_ci_high_pp']):.2f}]", f"{float(r['contribution_to_overall_delta_pp']):+.3f}"] for r in mapping_rows])
    report += '\n分支内结果按该分支实际题目混合汇总；4K分支仅含Qasper和HotpotQA。总体贡献把其他分支差值置零，沿用全510题三数据集等权分母，三项相加为-0.283。4K分支贡献+0.720，8K分支贡献-1.004；16K分支的288题采用固定参数0.7/1.0，答案token完全相同。\n\n'
    report += '按实际N汇总，N=1的177题动态−固定为+1.505个百分点，N=2的104题为-3.950，N=3的229题为0；各数据集组成与单独差值保存在quality.csv。三种APE同时生成相同token序列290/510；固定/动态相同393/510，其中288题同参、105题改参后输出仍相同。分数持平415题还包含22题token不同但F1相同。\n\n![映射分支对总体的贡献](../figures/rule_branch_delta_contributions.png)\n\n'

    report += '## 5 开发资料重合与独立来源敏感性\n\n测试与探索/验证集在规范化问题和完整证据组上无重合；单篇输入资料重合影响59个测试ID，单篇支撑资料重合24篇、影响29个测试ID，全部为MuSiQue。原始构造来源、资料ID交集及受影响样本清单保存在development_overlap_affected_ids.csv，并与前置审核清单逐ID吻合。\n\n'
    report += md(['MuSiQue方法', '支撑重合29题F1', '无支撑重合141题F1'], [[LABEL[m], f"{float(q('musique',m,'support_overlap')['f1_percent']):.2f}", f"{float(q('musique',m,'support_nonoverlap')['f1_percent']):.2f}"] for m in METHODS])
    report += '\n重合29题的绝对F1低于其余141题，五方法均呈该样本构成下的表现。重合29题动态与固定全部持平；非重合141题动态提高0.071个百分点，只有1题分数上升。保留全部510题主表，来源聚类敏感性及资料分层分别给出，冻结规则保持原值。\n\n'

    report += '## 6 冷请求效率、缓存与显存\n\n以下每方法均为共同510题，时间单位秒，输出长度含EOS。均值的95%区间及所有指标中位数/P90见main_efficiency.csv。\n\n'
    report += md(['方法', 'TTFT均值', 'TTFT中位数/P90', '总响应均值', '总响应中位数/P90', '输出token均值'], [[LABEL[m], f"{float(e(m,'ttft_seconds')['mean']):.4f}", f"{float(e(m,'ttft_seconds')['median']):.4f}/{float(e(m,'ttft_seconds')['p90']):.4f}", f"{float(e(m,'total_seconds')['mean']):.4f}", f"{float(e(m,'total_seconds')['median']):.4f}/{float(e(m,'total_seconds')['p90']):.4f}", f"{float(e(m,'output_tokens')['mean']):.3f}"] for m in METHODS])
    report += '\n![冷请求时延](../figures/cold_latency.png)\n\n'
    report += md(['动态相对固定指标', '动态−固定/毫秒', '95%配对区间/毫秒', '逐题加速比均值/中位数', '均值耗时之比'], [[field, f"{float(perf(field)['delta_comparison_minus_reference_s'])*1000:+.3f}", f"[{float(perf(field)['delta_ci_low_s'])*1000:.3f},{float(perf(field)['delta_ci_high_s'])*1000:.3f}]", f"{float(perf(field)['paired_speedup_mean']):.5f}/{float(perf(field)['paired_speedup_median']):.5f}", f"{float(perf(field)['ratio_of_means']):.5f}"] for field in ['ttft_seconds','total_seconds']])
    report += '\n逐题加速比为参照耗时÷比较耗时，先逐题求比再平均；均值耗时比为两方法平均耗时相除。动态相对固定的两种总响应比值分别为1.01112和0.99456，按同一批请求的两种定义保留。PCW相对顺序的总响应均值缩短0.1797秒、均值耗时比1.1086；动态相对顺序总响应增加0.1697秒，区间[0.1373,0.2025]秒。\n\n'
    report += md(['方法', '缓存构建均值/秒', '缓存获取均值/微秒', '参数分支均值/微秒', 'allocated均值/GiB', 'reserved均值/GiB'], [[LABEL[m], f"{float(e(m,'cache_build_seconds')['mean']):.4f}", f"{float(e(m,'cache_get_seconds')['mean'])*1e6:.3f}", f"{float(e(m,'dynamic_selection_seconds')['mean'])*1e6:.3f}", f"{float(e(m,'peak_allocated_bytes')['mean']):.3f}", f"{float(e(m,'peak_reserved_bytes')['mean']):.3f}"] for m in METHODS])
    report += '\n动态选参均值24.005微秒，固定参数分支1.806微秒。两方法TTFT均值相差-2.332毫秒，同参同输出请求仍存在计时差；这批记录每方法来自一个连续worker时段，需交错重复测时分解参数开销与执行时序的影响。缓存构建是完整逐块资料编码，缓存获取为cold条件下的查询开销。显存GiB=bytes/2³⁰，请求开始重置峰值统计，覆盖模型、开始时存活缓存及本次请求；reserved还保留worker内分配器历史。\n\n'
    report += '五方法worker的模型加载依次为' + '、'.join(f"{float(next(r['model_load_seconds'] for r in workers if r['method']==m)):.3f}" for m in METHODS) + '秒，单列于model_load_workers.csv，排除于请求时延。三个基线worker早于参数搜索，两个正式worker在冻结后顺序执行；完整起止时间按UTC保留。\n\n'
    report += '## 7 输出长度、生成上限及质量/时延关系\n\n'
    report += md(['分组', '配对数', '固定/动态token均值', '动态−固定总响应/毫秒', '动态−固定F1/百分点'], [[r['group'], r['n_pairs'], f"{float(r['mean_fixed_output_tokens']):.3f}/{float(r['mean_dynamic_output_tokens']):.3f}", f"{float(r['mean_total_delta_s'])*1000:+.2f}", f"{float(r['delta_f1_pp_pooled']):+.3f}"] for r in length_groups])
    report += '\n上述为实际请求混合的描述性分组。393题输出相同，平均总响应差-6.10毫秒；117题输出不同，动态平均多1.803 token、总响应多68.02毫秒。更长67题与更短40题共同决定全体平均多0.414 token、总响应多10.90毫秒；两类的F1点估计均下降。\n\n![输出长度与首token后时间](../figures/output_length_and_after_ttft.png)\n\n![逐题输出长度、时延和质量](../figures/paired_length_latency_quality.png)\n\n'
    report += md(['数据集', *[LABEL[m]+' 上限命中/170' for m in METHODS]], [[ds, *[f"{next(r['cap_hits'] for r in limits if r['dataset']==ds and r['method']==m)}/170" for m in METHODS]] for ds in DS])
    report += '\n生成上限为Qasper128、HotpotQA/MuSiQue32 token。共同测试2550条中41条达到上限（1.61%），2509条以EOS结束，EOS恰在上限为0。MuSiQue每方法有5–8条达到32 token；原始答案、分数、token及源行号在generation_cap_records.csv。\n\n'
    cap = next(r for r in cap_groups if r['dataset']=='musique' and r['method']=='ape_dynamic' and r['cap_hit']=='True')
    no_cap = next(r for r in cap_groups if r['dataset']=='musique' and r['method']=='ape_dynamic' and r['cap_hit']=='False')
    report += f"以MuSiQue动态APE为例，达到上限的{cap['n_requests']}题平均F1={float(cap['mean_f1_percent']):.2f}、总响应{float(cap['mean_total_seconds']):.3f}秒；其余{no_cap['n_requests']}题为{float(no_cap['mean_f1_percent']):.2f}、{float(no_cap['mean_total_seconds']):.3f}秒。分组由实际生成长度决定，保留题目、质量及耗时共同阅读，后续成对提高上限时继续使用同题比较。\n\n"

    report += '## 8 统一规则选择的案例\n\n按动态−固定F1最大正差、最大负差及零差最小sample_id选全局案例；并列按ID字典序。逐数据集相同规则的案例同时保存。\n\n'
    for index, r in enumerate([r for r in cases if r['selection_scope']=='ALL'],1):
        report += f"### 8.{index} {r['case_type']}：{r['sample_id']}\n\n"
        report += f"输入L={r['L']}、实际N={r['actual_N']}，映射{r['mapped_L']}/{r['mapped_N']}；固定T/S={r['fixed_T']}/{r['fixed_S']}，动态={r['dynamic_T']}/{r['dynamic_S']}。F1原值{r['fixed_f1_0_1']}→{r['dynamic_f1_0_1']}，差{float(r['delta_pp']):+.3f}个百分点。源文件attempts.jsonl第{r['fixed_source_line']}/{r['dynamic_source_line']}行。\n\n"
        report += f"- 问题：{r['question']}\n- 固定原答案：{r['fixed_answer']}\n- 动态原答案：{r['dynamic_answer']}\n- 参考答案：{r['reference_answers']}\n\n"

    report += '## 9 审核、讨论与补实验事项\n\n'
    report += '保存summary.json的longbench_test_by_method与longbench_five_method_paired中30项F1均值及全部对应ID、分母与本次复算吻合，浮点差小于10⁻¹²；本次新增配对区间、来源敏感性、规则贡献及效率分层。正式任务数与任务工厂和冻结测试划分一致。\n\n'
    report += md(['具体问题', '证据及范围', '本阶段处理', '所需补充'], [[r['phenomenon'],r['evidence'],r['handling'],r['followup']] for r in flags])
    report += '\n开发构造证据缺失的具体ID及配置已在前置审核和实验2/3报告登记；它影响参数候选和验证选择的输入。本次正式测试使用完整公开文档，按原冻结规则评价。后续重建开发构造、补自然N=1–3网格并重新冻结时，需配套新增独立测试。当前保留全部原始记录和参数。\n\n'
    report += '## 10 结论与复现\n\n在本次共同510题中，顺序编码宏F1最高，PCW总响应均值最短。调优固定APE和动态APE均低于默认APE的宏F1点估计；动态相对固定小幅下降0.283个百分点，95%配对区间[-1.676,1.092]，主要由4K映射收益与8K映射损失相抵后形成。16K分支同参，288题答案完全持平。正式自然块数全部位于开发N范围之外，动态实际只使用N=4列的三个分支。\n\n'
    report += '动态相对固定平均TTFT缩短2.332毫秒，总响应增加10.902毫秒，输出长度均值增加0.414 token；这批冷请求上的总响应变化与生成长度共同报告。按共享单篇支撑资料连接来源后，质量差区间保持相近。材料保留单模型、单campaign、模板适配、逐块执行、worker时段及缺少硬件遥测的实际设置。\n\n'
    report += f"实际分析提交{env['analysis_commit']}，相对参考提交无变化；依赖版本、seed20261116、5000次重采样及各scope随机子序列见environment.json。全部脚本、CSV、PNG/SVG及运行命令见本阶段README；关键字段与来源见provenance.csv。原始证据只读，未安装依赖、运行模型或改动冻结规则。研究状态保留SUBMITTED，执行停止在第六阶段。\n"
    (OUT / 'reports/LONGBENCH_FINAL_REPORT.md').write_text(report, encoding='utf-8')

    audit = '# 第六阶段审核记录\n\n'
    audit += '已核对冻结510题/2550条、五方法逐ID连接、actual L/N与块长、参考答案、冷缓存、attempt、T/S和summary均值。保存score已在前置审核核对官方实现；本阶段复用原分数进行统计。\n\n'
    audit += md(['审核项','实际结果','证据表'], [['样本及逻辑生成','每DS170题; baseline1530+final1020; 无复用logical_labels合并','membership.csv; per_request_records.csv'], ['缺失/异常','失败0、空答案0、缺参考0、非有限/负值0; 全attempt1','record_issues.csv; 本阶段脚本断言'], ['原汇总对照','两个测试汇总部分30项F1、ID与分母一致','original_summary_crosscheck.csv'], ['冻结和映射','全部T/S吻合rule; 正式worker在冻结后','rule_mapping_counts.csv; model_load_workers.csv'], ['Bootstrap','450冻结组;413支撑连通组件;5法共同抽样','bootstrap_membership.csv; bootstrap_diagnostics.csv']])
    audit += '\n' + md(['现象','影响','证据','处理与补充'], [[r['phenomenon'],r['affected'],r['evidence'],r['handling']+'；'+r['followup']] for r in flags])
    audit += '\n最小证据：membership.csv逐ID保存网格外标志、映射、资料交集和同参输出标签；generation_cap_records.csv保存41条上限原答案；dynamic_fixed_per_sample.csv保存510对分数、原答案、参数及源行。冻结候选/验证阶段的构造问题沿用前置审核清单，本阶段保留正式完整文档输入。\n\n'
    audit += '代码及数据来源：restored/output/runs/main80-20261114T041503/下的attempts.jsonl、summary.json、rule.json、snapshot/datasets/frozen-main80/；方法及评分依据同次snapshot/APE/main80/methods.py、rules.py、jobs.py、metrics.py。原始资料不写入，CPU分析产物位于本阶段目录。研究状态SUBMITTED，后续阶段未执行。\n'
    (OUT / 'reports/AUDIT.md').write_text(audit, encoding='utf-8')

    index = '# 实验第六阶段：LongBench正式五方法比较\n\n本阶段完成，研究状态SUBMITTED；执行停止于本阶段。\n\n'
    links = [('分析报告','reports/LONGBENCH_FINAL_REPORT.md'),('审核记录','reports/AUDIT.md'),('五方法质量主表','tables/main_quality.csv'),('全部方法对质量差','tables/main_quality_paired.csv'),('动态与固定主要比较','tables/primary_dynamic_fixed.csv'),('510题五方法成员与分层标签','tables/membership.csv'),('2550条逐请求原分数与计时','tables/per_request_records.csv'),('动态固定逐题原答案','tables/dynamic_fixed_per_sample.csv'),('规则分支解释','tables/mapping_interpretation.csv'),('全部分支/N/重合层统计','tables/quality_paired.csv'),('效率均值、中位数、P90及区间','tables/main_efficiency.csv'),('配对时间差和两种加速比','tables/efficiency_paired.csv'),('输出长度与时延关系','tables/paired_output_length_groups.csv'),('上限与EOS','tables/generation_limits.csv'),('原答案案例','tables/selected_cases.csv'),('来源组及连通组件','tables/bootstrap_membership.csv'),('原汇总对照','tables/original_summary_crosscheck.csv'),('关键数字来源与公式','tables/provenance.csv'),('字段及权重解释','tables/field_dictionary.csv'),('环境和种子','environment.json')]
    index += ''.join(f'- [{title}]({target})\n' for title,target in links)
    index += '\nfigures/含8张静态图，各保存PNG和SVG；sources/保留原冻结规则。质量图的数据来自main_quality、main_quality_paired、mapping_interpretation；效率图来自main_efficiency、dynamic_fixed_per_sample；输入图来自membership。主F1按三数据集等权，分支内展示按实际题目混合汇总；每项均保留分母、单位与区间。\n\n'
    index += '从仓库根目录复现：\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n$env:PYTHONDONTWRITEBYTECODE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_longbench_final.py\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/plot_longbench_final.py\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_longbench_final.py\n```\n\n使用现有Python3.12.7、NumPy1.26.4、Matplotlib3.9.2，无新增依赖。seed20261116，5000次配对聚类Bootstrap；分组、子序列、软件版本及分析提交见environment.json。各脚本输出和缓存位于pc-analysis-output，本阶段只读取已恢复证据。F1重新评分0次。\n'
    (OUT / 'README.md').write_text(index, encoding='utf-8')
    root_index = (BASE / 'README.md').read_text(encoding='utf-8')
    root_index = root_index.replace('当前已完成实验第五阶段：注意力诊断。','当前已完成实验第六阶段：LongBench正式五方法比较。')
    if '[实验6：LongBench五方法正式比较]' not in root_index:
        root_index = root_index.replace('- [实验5：注意力诊断]', '- [实验6：LongBench五方法正式比较](stages/06_longbench_final/README.md)\n- [实验5：注意力诊断]')
    root_index = root_index.replace('本轮已停在实验5；','本轮已停在实验6；')
    (BASE / 'README.md').write_text(root_index, encoding='utf-8')
    print('Experiment 6 report, audit, provenance and material index written.')


if __name__ == '__main__':
    main()
