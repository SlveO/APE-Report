"""Assemble experiment-5 reports and dictionaries from the derived CPU tables."""
from pathlib import Path
from statistics import mean
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'pc-analysis-output'
OUT = BASE / 'stages/05_attention'


def readcsv(name):
    with (OUT / 'tables' / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def writecsv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / 'tables' / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + '\n'.join('| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', '<br>') for v in row) + ' |' for row in rows) + '\n'


def main():
    result = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    env = json.loads((OUT / 'environment.json').read_text(encoding='utf-8'))
    pairs, attempts, evidence, query, history = (readcsv(name) for name in ['answer_pairs.csv', 'diagnostic_attempts.csv', 'evidence_regions.csv', 'query_region_pairs.csv', 'decode_history_alignment.csv'])
    timing = []
    for length in ['4096', '16384', 'ALL']:
        for role in ['default', 'cell_best']:
            rows = [r for r in attempts if r['role'] == role and (length == 'ALL' or r['L'] == length)]
            for field in ['ttft_seconds', 'total_seconds', 'cache_build_seconds', 'cache_get_seconds', 'output_tokens', 'peak_allocated_bytes', 'peak_reserved_bytes']:
                values = np.array([float(r[field]) for r in rows])
                unit = 'token including EOS' if field == 'output_tokens' else 'GiB' if field.endswith('_bytes') else 's'
                if unit == 'GiB':
                    values /= 2**30
                timing.append({'L': length, 'role': role, 'metric': field, 'n_requests': len(rows), 'n_question_ids': len({r['sample_id'] for r in rows}), 'mean': values.mean(), 'median': np.median(values), 'p90': np.quantile(values, .9), 'unit': unit, 'observer_enabled': True, 'cache_condition': 'cold, KV miss', 'ci_low': '', 'ci_high': '', 'ci_method': 'descriptive diagnostic observations; no resampling', 'source_file': 'diagnostic_attempts.csv'})
    writecsv('diagnostic_efficiency.csv', timing)
    stops = []
    for role in ['default', 'cell_best']:
        rows = [r for r in attempts if r['role'] == role]
        stops.append({'role': role, 'n': len(rows), 'cap_hit_count': sum(int(r['output_tokens']) == int(r['max_new_tokens']) for r in rows), 'eos_last_count': sum(r['eos_last'] == 'True' for r in rows), 'output_tokens_min': min(int(r['output_tokens']) for r in rows), 'output_tokens_max': max(int(r['output_tokens']) for r in rows), 'denominator': len(rows), 'unit': 'generation records / output tokens including EOS'})
    writecsv('generation_stops.csv', stops)
    with (BASE / 'stages/01_audit/tables/runtime_workers.csv').open(encoding='utf-8-sig') as handle:
        workers = [r for r in csv.DictReader(handle) if r['stage'] == 'attention']
    writecsv('model_load_workers.csv', workers)

    alignments = []
    for pair in pairs:
        rows = [r for r in history if r['pair_id'] == pair['pair_id']]
        diverged = [int(r['step']) for r in rows if r['same_history_before_forward'] == 'False']
        alignments.append({k: pair[k] for k in ['pair_id', 'dataset', 'sample_id', 'L', 'N']} | {'common_step_pairs': len(rows), 'same_history_step_pairs': len(rows) - len(diverged), 'diverged_history_step_pairs': len(diverged), 'first_diverged_step': min(diverged) if diverged else '', 'default_unpaired_late_steps': int(pair['default_output_tokens']) - len(rows), 'best_unpaired_late_steps': int(pair['best_output_tokens']) - len(rows), 'unit': 'paired forward steps including step0; compare output_token_ids[:step]'})
    writecsv('decode_alignment_summary.csv', alignments)

    dictionary = [
        ('diagnostic_attempts.csv', '一条真实生成', '12', 'stage=attention, success, attempt=1, explore, cold', 'sample_id/L/N/T/S; source_line', '保存F1/答案/token/计时/显存; observer开启'),
        ('answer_pairs.csv', '同题同L/N的两参数配对', '6; 3个问题', '默认0.9/0.9 对探索格点最佳', 'pair_id; sample_id/L/N', '差值=(best-default)*100; F1原值0-1'),
        ('evidence_regions.csv', '一份冻结文档构造', '6', '三题×两个长度', 'sample_id/L', '字符证据→当前token偏移→retained区间投影; 半开区间'),
        ('capture_audit.csv', '一层一次forward最后查询', str(result['attention_rows']), '12任务; query+全部实际decode', 'pair_id/role/layer/step; source_line', '保存重建误差/容差; token质量和分支质量误差'),
        ('step_coverage.csv', '配置×层', '36', '层0/15/31', 'pair_id/role/layer', '实际step=0..output_tokens-1'),
        ('head_steps.csv', '一个指定头的一次查询', str(result['head_step_rows']), '头0/8/16/24', 'pair_id/role/layer/head/step', '五区域原始权重质量; 文档条件熵; 证据占比'),
        ('layer_steps.csv', '一层一次查询的四头均值', str(result['attention_rows']), '四头等权', 'pair_id/role/layer/step', '比例为mean_h(Eh/Ch); 熵为头内先算后平均'),
        ('paired_head_steps.csv', '同题同层同头同step的参数配对', '共同step×3层×4头', '共同保存step; 保留same_history标签', 'pair_id/layer/head/step', 'best-default; decode历史分歧另标'),
        ('paired_layer_steps.csv', '逐头配对差异的四头均值', '共同step×3层', '与paired_head_steps相同', 'pair_id/layer/step', '四头等权'),
        ('query_region_pairs.csv', '同输入请求末token的层级配对', '18', 'step=0; 六对×三层', 'pair_id/layer', '图中证据占文档比例及原始区域质量'),
        ('decode_history_alignment.csv', '同题同step的生成历史', str(len(history)), '两配置共同step', 'pair_id/step', '先比较完整output_token_ids[:step]'),
        ('physical_cache_bins.csv', '64-token物理箱×实际步骤×层', '见CSV行数', '四头先平均、箱内求和', 'pair_id/role/layer/step/bin', 'decode热图数据; 最末箱按实际token数'),
        ('query_head_bins.csv', '64-token物理箱×头×层', '12配置×3层×4头×箱', 'step=0; 保留逐头', 'pair_id/role/layer/head/bin', 'query逐头热图数据; 原始质量'),
        ('selected_cases.csv', '改善/退化/持平选例', '改善1、退化0、持平1', '正负最大差; 同分按sample_id/L/N', 'pair_id', '保留两答案、参考、配置、源行号'),
        ('diagnostic_efficiency.csv', '长度×参数角色×指标', '每格3请求; ALL每角色6', '只含注意力诊断冷请求', 'L/role/metric', '均值/中位数/P90; 时间s、显存GiB; CI留空'),
    ]
    writecsv('table_dictionary.csv', [dict(zip(['file', 'grain', 'denominator', 'filter', 'join_key', 'definition_and_units'], row)) for row in dictionary])
    field_definitions = [
        ('L', '实际document_tokens长度', 'token', '与原始文本字符数、原始文档数分别登记'),
        ('N', '实际编码块数', 'block', '本阶段4/16; 原始文档数另列1/10/17'),
        ('step', '0=请求末token; s>0输入前s个已生成token后取最后查询', 'forward index', '终止EOS之后没有forward'),
        ('prefix_mass', '共享前缀位置权重之和', 'raw mass', 'prefix独立于文档与请求'),
        ('evidence_mass', '完整证据token掩码内权重之和E', 'raw mass', '掩码来自字符证据投影'),
        ('background_mass', '文档中证据掩码以外权重之和', 'raw mass', '含未标为证据的原文、构造补充token'),
        ('request_mass', '完整suffix请求权重之和', 'raw mass', '包括模板、当前问题及assistant起始标记'),
        ('decoded_history_mass', '已生成并再次送入网络的token权重之和', 'raw mass', 'step0为0; s步长度为s'),
        ('context_mass', 'C=evidence_mass+background_mass', 'raw mass', '等于文档分支质量，保留数值舍入'),
        ('evidence_fraction_context', '每个头E/C; 层级值为mean_h(Eh/Ch)', 'fraction 0-1', '逐头先求比例，再四头等权平均'),
        ('evidence_enrichment_vs_uniform_context', '(E/C)/(证据token数/L)', 'ratio', '1对应文档内均匀权重参照'),
        ('context_entropy_nats', '-sum_i p_i ln(p_i), p_i=w_i/C, i属于文档', 'nats', '文档内归一; 四头值先算后平均'),
        ('max_abs_output_error_saved', '服务器比较重建weighted-V与实际o_proj前输出的最大绝对误差', 'activation absolute error', 'PC核对保存值与tolerance; 未保存Q/K/V/target向量'),
        ('tolerance_saved', '0.03+0.02*max(abs(target))', 'activation absolute error', '由服务器诊断保存'),
        ('difference_pp', '(探索格点最佳保存F1-默认保存F1)*100', 'percentage points', '保存score直接复用'),
        ('same_history_before_forward', '两方法output_token_ids[:step]完全相同', 'boolean', '共同输入token历史; 参数对应的隐状态各自生成'),
    ]
    writecsv('field_dictionary.csv', [dict(zip(['field', 'definition', 'unit', 'notes'], row)) for row in field_definitions])

    findings = [
        {'finding': '保存证据区间稀疏', 'affected': '6/6构造; 12项诊断', 'evidence': '完整证据token q43/h1468/m1833; 保存区间q10/h235/m494', 'impact': '直接按保存区间会将部分证据计入背景', 'handling': '按原字符证据和现有tokenizer重建掩码、投影retained区间; 原始输入及字段保留', 'required_followup': '本阶段区域统计已修正; 关联上游映射问题见前置审核'},
        {'finding': 'decode历史分歧', 'affected': 'Qasper4K从step5; Qasper16K从step1', 'evidence': 'decode_history_alignment.csv逐步列出两历史token ID', 'impact': '后续注意力同时随参数和已生成内容变化', 'handling': '同输入step0为主; decode逐步保留same_history标签及未配对尾部数', 'required_followup': '需要隔离参数效应时补固定同一token序列的teacher-forced诊断'},
        {'finding': '未保存weighted-V独立重建输入', 'affected': '231条捕获', 'evidence': '原始字段保存weights/branch/error/tolerance; 缺Q/K/V和实际target向量', 'impact': 'PC可核对保存误差、容差和质量一致性; weighted-V误差数值来自服务器采集', 'handling': '明确误差的来源和检验步骤', 'required_followup': '若需PC独立重建,为选定layer/head/step补存Q/K/V/target及实际分支输出'},
        {'finding': '探索候选与冻结动态规则的角色不同', 'affected': '16K/16的三个参数比较', 'evidence': '候选0.5/0.7; rule.json冻结0.7/1.0', 'impact': '本阶段解释探索候选的注意力与答案', 'handling': '表中同时列探索参数与冻结参数; 本阶段保持冻结规则', 'required_followup': '冻结规则收益在后续正式配对实验评价'},
    ]
    writecsv('audit_findings.csv', findings)
    parameters = [{k: r[k] for k in ['pair_id', 'dataset', 'sample_id', 'L', 'N', 'default_T', 'default_S', 'best_T', 'best_S', 'frozen_dynamic_T', 'frozen_dynamic_S']} for r in pairs]
    (OUT / 'sources/parameters_used.json').write_text(json.dumps({'comparison': 'default versus exploration_cell_best; both execution labels ape_fixed', 'source': 'same run candidates.json + rule.json + attempts.jsonl', 'pairs': parameters}, ensure_ascii=False, indent=2), encoding='utf-8')

    report = '# 实验第五阶段：注意力诊断\n\n'
    report += '## 1 研究问题与实际材料\n\n我们检查T/S变化如何伴随文档、证据、背景和请求区域的注意力分配变化，并连接相同输入上的答案。材料来自main80-20261114T041503的12条attention生成及完整attention.jsonl。每数据集按冻结顺序取首个有效探索题，三道原始问题各构造4K/4块、16K/16块，形成六组参数配对。全部执行标签为ape_fixed，逻辑角色分别为default与exploration_cell_best。我们直接引用保存score，评分重算0次。\n\n'
    report += '默认T/S=0.9/0.9；4K/4探索候选为1.0/1.0，16K/16为0.5/0.7。16K/16的冻结动态规则已回退到0.7/1.0，本阶段按原诊断任务解释探索候选。模型为Llama-3.1-8B-Instruct，采集使用BF16分支混合；正式时延运行关闭观察器，诊断运行保留逐步重建与JSON写出。原始文档数Qasper/HotpotQA/MuSiQue分别为1/10/17，实际编码块数为4或16。\n\n'
    report += '## 2 采集定义、区域和数值审核\n\n'
    report += f"我们流式读取{result['attention_rows']}条层级捕获、{result['head_step_rows']}条head-step数据。层为0/15/31，头为0/8/16/24；36项配置×层检查均覆盖step=0到output_tokens−1。step0取完整请求的最后token，step=s>0取前s个已生成token送回网络后的最后查询，用于预测下一token。终止EOS之后直接退出。物理缓存长度逐条等于prefix_tokens+L+request_tokens+step。\n\n"
    report += '记文档token logits为zᶜᵢ=q·kᵢ/(√d·T)，其他token logits为zᵒⱼ=q·kⱼ/√d。文档分支LSE乘S×T后与其他分支LSE作softmax，得到实际BF16分支质量mᶜ/mᵒ。保存token权重为wᶜᵢ=mᶜ·softmax(zᶜ)ᵢ及wᵒⱼ=mᵒ·softmax(zᵒ)ⱼ。T同时改变文档内分布和分支分配，S通过LSE缩放改变分支分配。\n\n'
    report += f"保存的o_proj前指定头输出重建误差最大为{result['max_saved_output_reconstruction_error']:.9f}，最大误差/容差为{result['max_error_to_saved_tolerance']:.6f}，全部低于保存容差0.03+0.02×max|target|。PC逐条核对保存误差与容差，并检查token质量和分支质量；原始记录缺Q/K/V和target向量，weighted-V误差数值取服务器采集结果。token权重和偏离1最大{result['max_normalization_abs_error']:.9f}，文档/其他区域质量与对应分支质量的最大绝对差分别为{result['max_context_branch_abs_error']:.3g}/{result['max_other_branch_abs_error']:.3g}。原始质量保留BF16舍入，文档条件比例单独用E/C归一。\n\n"
    report += '五个区域为共享前缀、完整证据、文档内其余背景、完整suffix请求、已生成历史。请求包含问题及模板标记。字符证据经本机既有tokenizers与运行快照tokenizer.json转换，再按retained_source_intervals投影到实际构造位置。三个问题的证据全部保留；六种构造的保存区间均偏稀疏，本次区域统计使用重建后的完整掩码。\n\n'
    report += md(['数据集', '完整证据token', '原保存标记token', '4K证据比例', '16K证据比例'], [[dataset, next(r['evidence_tokens'] for r in evidence if r['dataset'] == dataset), next(r['saved_evidence_tokens'] for r in evidence if r['dataset'] == dataset), *[f"{float(next(r['evidence_fraction_document'] for r in evidence if r['dataset'] == dataset and r['L'] == length))*100:.3f}%" for length in ['4096', '16384']]] for dataset in ['qasper', 'hotpotqa', 'musique']])
    report += '\nQasper采用exact_evidence_with_whitespace_alignment标注；HotpotQA和MuSiQue采用whole_public_support_document_version_changed策略，证据范围覆盖指定公共支撑文档。表中证据粒度随数据来源保留。背景包含其余原文和构造补充token。\n\n'
    report += '## 3 同输入请求末token与质量结果\n\nF1原值为0–1，以下乘100展示；差值统一为探索候选−默认。证据占文档比例先在各头求Eh/Ch，再对四头等权平均。下表列第31层，全部三层及逐头值保存在query_region_pairs.csv和paired_head_steps.csv。\n\n'
    report += md(['数据集/输入', '默认F1', '候选F1', '差/百分点', '文档质量 默认→候选', '证据占文档% 默认→候选'], [[f"{r['dataset']} {int(r['L'])//1024}K/{r['N']}", f"{float(r['default_f1_0_1'])*100:.2f}", f"{float(r['best_f1_0_1'])*100:.2f}", f"{float(r['difference_pp']):+.2f}", f"{float(q['default_context_mass']):.4f}→{float(q['best_context_mass']):.4f}", f"{float(q['default_evidence_fraction_context'])*100:.3f}→{float(q['best_evidence_fraction_context'])*100:.3f}"] for r in pairs for q in query if q['pair_id'] == r['pair_id'] and q['layer'] == '31'])
    report += '\n4K/4从0.9/0.9改为1.0/1.0后，三题在三个采样层的文档总质量均上升；证据占文档比例按题、层变化。Qasper的F1由30.77升至80.00，HotpotQA保持100，MuSiQue保持0。16K/16改为0.5/0.7后，三题在三个采样层的文档总质量均下降；三题F1分别保持0、100、0。参数对分支总分配的变化在这些输入中更一致，证据条件占比和答案分数按任务呈现不同表现。\n\n'
    report += '![请求末token文档分支总质量](../figures/query_document_mass_overview.png)\n\n![请求末token证据比例](../figures/query_evidence_overview.png)\n\n文档分支总质量图使用统一0–100%纵轴。证据比例图各面板从0起、独立设定纵轴，展示Qasper小幅比例变化；灰虚线为证据token数/L对应的文档内均匀权重参照。三题分别来自三个数据集，层、头及两长度保留为同题内观测；本阶段按实际记录描述，不对这些头和步骤作跨题Bootstrap，质量CI留空。\n\n'
    report += '## 4 实际decode及历史配对\n\n'
    report += md(['输入', '共同步骤数', '相同token历史步骤', '分歧步骤', '首次分歧step', '默认/候选未配对尾部'], [[f"{r['dataset']} {int(r['L'])//1024}K/{r['N']}", r['common_step_pairs'], r['same_history_step_pairs'], r['diverged_history_step_pairs'], r['first_diverged_step'] if r['first_diverged_step'] != '' else '无', f"{r['default_unpaired_late_steps']}/{r['best_unpaired_late_steps']}"] for r in alignments])
    report += '\n相同历史标签依据完整output_token_ids[:step]判定，隐状态仍沿各配置各自计算。Qasper4K在step5起生成历史不同，16K在step1起不同，后续图像同时包含参数与已生成内容的变化。HotpotQA和MuSiQue两长度的共同步骤均保留相同token历史。未配对尾部仅在各配置原始轨迹展示。\n\n'
    report += '## 5 注意力图与统一选例\n\n每组参数共用10⁻⁶至1的对数色标。query_heads图展示每个头；decode_mean_heads图先对四头平均，两侧使用共同token与step坐标，空白表示该配置未执行的步骤或尚无对应缓存token，标题标注生成历史首次分歧位置。每个色块为连续64个物理缓存token的质量之和，最末箱按实际长度；绿色顶线为完整证据、白线为前缀末、红线为文档末。一个箱可混合多个区域，区域精确统计按逐token掩码求和。物理缓存索引与分块复用的RoPE位置分别记录，query_start_position等于prefix加最长1024-token块，图横轴使用拼接后的物理顺序。\n\n'
    selected = readcsv('selected_cases.csv')
    for row in selected:
        report += f"### 5.{selected.index(row)+1} {row['case_type']}案例：{row['dataset']} {int(row['L'])//1024}K/{row['N']}\n\n"
        report += f"样本ID：`{row['sample_id']}`。默认T/S={row['default_T']}/{row['default_S']}，候选{row['best_T']}/{row['best_S']}；保存F1分别{float(row['default_f1_0_1'])*100:.2f}、{float(row['best_f1_0_1'])*100:.2f}。attempts.jsonl源行{row['default_source_line']}/{row['best_source_line']}。\n\n"
        report += f"- 默认原答案：{row['default_answer']}\n- 候选原答案：{row['best_answer']}\n- 参考答案：{row['reference_answers']}\n\n"
        report += f"![{row['case_type']}案例逐头注意力](../figures/query_heads_{row['pair_id']}.png)\n\n![{row['case_type']}案例实际decode](../figures/decode_mean_heads_{row['pair_id']}.png)\n\n"
    report += '选例规则为最大改善、最大退化、零差最小sample_id/L/N，并列按同序排序。本阶段六对中改善1、退化0、持平5，退化类型留空。Qasper16K两答案内容不同而F1均为0，全部原答案保存在answer_pairs.csv。\n\n'
    report += '## 6 诊断效率与生成停止\n\n'
    report += md(['角色', '诊断请求', 'TTFT均值/秒', '总响应均值/秒', '总响应中位数/秒', '总响应P90/秒', '输出token均值'], [[role, 6, *[f"{next(r['mean'] for r in timing if r['L']=='ALL' and r['role']==role and r['metric']==field):.4f}" for field in ['ttft_seconds', 'total_seconds']], f"{next(r['median'] for r in timing if r['L']=='ALL' and r['role']==role and r['metric']=='total_seconds'):.4f}", f"{next(r['p90'] for r in timing if r['L']=='ALL' and r['role']==role and r['metric']=='total_seconds'):.4f}", f"{next(r['mean'] for r in timing if r['L']=='ALL' and r['role']==role and r['metric']=='output_tokens'):.2f}"] for role in ['default', 'cell_best']])
    report += '\n以上时间覆盖观察器中的逐步FP32权重重建、误差核对、CPU转移和JSON写出，单独登记为诊断请求。12条均以EOS结束，生成上限命中0/12；默认输出范围2–16 token，候选2–18 token。Qasper16K候选由默认4 token变为18 token，诊断总响应由2.629升至5.761秒，保存F1均为0。缓存构建/获取与显存的均值、中位数、P90保存在diagnostic_efficiency.csv，显存单位GiB=bytes/2³⁰，峰值包含模型及存活缓存。\n\n'
    report += f"本阶段唯一worker模型加载{float(workers[0]['model_load_seconds']):.6f}秒，事件源行见model_load_workers.csv，加载单列于请求计时之外。三题的两个长度共同保留，角色均值分母为6请求、3个原始问题。\n\n"
    report += '## 7 结论、补充记录与复现\n\n我们核对了12项诊断的完整采集：所有指定层、头、实际步骤和保存误差通过检查，并修正六份构造的证据区域标注。4K/4的探索候选普遍增加文档分支质量，伴随Qasper这一个配对的F1改善；16K/16候选普遍降低文档分支质量，三个答案分数持平。证据条件占比的变化随层和问题而异，结果须与总文档质量、具体答案一同阅读。\n\n'
    report += '需补充的记录有两类：若需在PC独立复算weighted-V误差，补存选定层/头/步骤的Q/K/V、target及实际分支输出；若需分离生成历史变化对decode注意力的作用，对Qasper两长度补同一固定token序列的teacher-forced诊断。当前原始记录缺这两类输入，本轮登记建议，保持原冻结参数。\n\n'
    report += f"本地分析提交为{env['analysis_commit']}，与参考提交相同。使用已有Python、NumPy、Matplotlib和tokenizers，版本见environment.json；随机分析未执行，统一Bootstrap种子20261116保留于环境记录。运行命令见本阶段README。原始证据只读，新增产物在pc-analysis-output内。研究状态保留SUBMITTED，执行停止在第五阶段。\n"
    (OUT / 'reports/ATTENTION_REPORT.md').write_text(report, encoding='utf-8')

    audit_text = '# 第五阶段审核记录\n\n'
    audit_text += f"范围为12条attention生成、{result['attention_rows']}条注意力捕获及六份开发构造，独立问题3个。所有捕获流式读取，无结构/非有限值/负权重/保存容差异常。以下问题影响区域解释、decode配对或PC复核所需数据。\n\n"
    audit_text += md(['现象', '范围', '证据', '影响与处理', '后续所需'], [[r['finding'], r['affected'], r['evidence'], r['impact'] + '；' + r['handling'], r['required_followup']] for r in findings])
    audit_text += '\n具体最小证据见sources/evidence_mapping_examples.json、tables/decode_history_alignment.csv、tables/capture_audit.csv和tables/answer_pairs.csv。稀疏证据位置的根因及全campaign受影响范围沿用前置审核报告，本阶段只处理六份实际诊断输入。\n\n'
    audit_text += '代码依据：运行快照main80/jobs.py第55–64行定义诊断选题与候选；main80/diagnostics.py第25–49行定义权重重建、容差和存储顺序；ape/ape_llama.py第170–197行定义T/S、分支混合与o_proj前观察；main80/methods.py第203–223行定义请求与生成循环。这些文件位于restored/APE/，运行快照对应副本在该次snapshot/APE/。数据依据为同次snapshot/datasets/frozen-main80/和attention.jsonl。\n\n'
    audit_text += '复核结论：231条层级捕获、924条head-step，36项配置×层覆盖完整；保存重建最大误差0.006171465，最大误差/容差0.112259。六组step0输入匹配；两个Qasper轨迹的历史分歧另标；保存原始质量的归一偏差与BF16分支质量相符。原始记录、候选、冻结规则和研究状态保留。\n'
    (OUT / 'reports/AUDIT.md').write_text(audit_text, encoding='utf-8')

    index = '# 实验第五阶段：注意力诊断材料\n\n研究状态SUBMITTED；本阶段完成后停止。\n\n'
    index += '- [分析报告](reports/ATTENTION_REPORT.md)\n- [审核记录](reports/AUDIT.md)\n- [六组答案与参数配对](tables/answer_pairs.csv)\n- [完整证据区域与原保存标记](tables/evidence_regions.csv)\n- [231条捕获审核](tables/capture_audit.csv)\n- [36项步骤覆盖](tables/step_coverage.csv)\n- [924条逐头区域权重](tables/head_steps.csv)\n- [同输入请求末token比较](tables/query_region_pairs.csv)\n- [解码历史对齐](tables/decode_alignment_summary.csv)\n- [原答案案例](tables/selected_cases.csv)\n- [诊断效率](tables/diagnostic_efficiency.csv)\n- [生成停止](tables/generation_stops.csv)\n- [模型加载](tables/model_load_workers.csv)\n- [数据表字典](tables/table_dictionary.csv)\n- [字段、公式与单位](tables/field_dictionary.csv)\n- [环境和提交](environment.json)\n\n'
    index += 'figures/中包含6组逐头query图、6组实际decode图、1张证据比例总图和1张文档分支质量总图，均有PNG及SVG。图表数据来自query_head_bins.csv、physical_cache_bins.csv和query_region_pairs.csv；区域以逐token掩码精确求和，图像每箱64 token。sources/保留证据映射最小记录及实际参数。\n\n'
    index += '从仓库根目录复现：\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n$env:PYTHONDONTWRITEBYTECODE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_attention_stage5.py\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_attention_stage5.py\n```\n\n本阶段无随机重采样，未重新评分、生成答案或安装依赖。Python/NumPy/Matplotlib/tokenizers均复用现有环境；缓存、临时文件与新增产物位于本阶段目录。各时间只对应观察器开启的诊断条件。\n'
    (OUT / 'README.md').write_text(index, encoding='utf-8')
    root_index = (BASE / 'README.md').read_text(encoding='utf-8')
    root_index = root_index.replace('当前已完成实验第四阶段：6K/12K长度检查。', '当前已完成实验第五阶段：注意力诊断。')
    if '[实验5：注意力诊断]' not in root_index:
        root_index = root_index.replace('- [实验4：长度检查]', '- [实验5：注意力诊断](stages/05_attention/README.md)\n- [实验4：长度检查]')
    root_index = root_index.replace('本轮已停在实验4；', '本轮已停在实验5；')
    (BASE / 'README.md').write_text(root_index, encoding='utf-8')
    print('Experiment 5 reports, dictionaries, timing tables and material index written.')


if __name__ == '__main__':
    main()
