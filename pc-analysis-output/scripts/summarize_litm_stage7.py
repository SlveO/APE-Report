"""Assemble the experiment-7 Chinese draft, audit and provenance from CPU tables."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/07_litm'
METHODS=['sequential','ape_fixed','ape_dynamic']
LABEL=dict(zip(METHODS,['顺序编码','固定APE','动态APE']))
POSITIONS=['0','9','19']
OFFICIAL='official_first_line_em'
FULL='saved_full_answer_em'
ENTRY='https://github.com/nelson-liu/lost-in-the-middle/blob/main/scripts/evaluate_qa_responses.py'
METRIC='https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py'


def readcsv(name):
    with (OUT/'tables'/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def csvout(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def md(headers,rows):
    def text(v):return str(v).replace('|','\\|').replace('\n','<br>')
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(text(v) for v in row)+' |' for row in rows)+'\n'


def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    env=json.loads((OUT/'environment.json').read_text(encoding='utf-8'))
    quality,paired,contrasts,paired_contrasts,efficiency,performance,impact,limits,caps,matching,lengths,cases,case_positions,workers=[readcsv(n) for n in ['quality.csv','quality_paired.csv','position_contrasts.csv','paired_position_contrasts.csv','efficiency_descriptive.csv','efficiency_paired.csv','scoring_scope_impact.csv','generation_limits.csv','generation_cap_conditioned.csv','output_equivalence.csv','paired_output_length_groups.csv','selected_cases.csv','selected_case_positions.csv','model_load_workers.csv']]
    def q(m,p='ALL',s=OFFICIAL):return next(r for r in quality if r['method']==m and r['position']==p and r['score_metric']==s)
    def pair(p='ALL',a='ape_fixed',b='ape_dynamic',s=OFFICIAL):return next(r for r in paired if r['position']==p and r['reference_method']==a and r['comparison_method']==b and r['score_metric']==s)
    def contrast(m,name,s=OFFICIAL):return next(r for r in contrasts if r['method']==m and r['contrast']==name and r['score_metric']==s)
    def pc(name,a='ape_fixed',b='ape_dynamic'):return next(r for r in paired_contrasts if r['reference_method']==a and r['comparison_method']==b and r['contrast']==name and r['score_metric']==OFFICIAL)
    def e(m,metric,p='ALL'):return next(r for r in efficiency if r['method']==m and r['metric']==metric and r['position']==p)
    def perf(metric,a='ape_fixed',b='ape_dynamic',p='ALL'):return next(r for r in performance if r['position']==p and r['reference_method']==a and r['comparison_method']==b and r['metric']==metric)
    def interval(r,lo='ci_low_pp',hi='ci_high_pp',digits=2):return f"[{float(r[lo]):.{digits}f}, {float(r[hi]):.{digits}f}]"
    def beijing(s):return datetime.fromisoformat(s).astimezone(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

    csvout('main_quality.csv',[r for r in quality if r['score_metric']==OFFICIAL])
    csvout('main_quality_paired.csv',[r for r in paired if r['score_metric']==OFFICIAL])
    csvout('main_position_contrasts.csv',[r for r in contrasts if r['score_metric']==OFFICIAL])
    csvout('main_paired_position_contrasts.csv',[r for r in paired_contrasts if r['score_metric']==OFFICIAL])
    membership=readcsv('membership.csv')
    input_stats=[]
    for p in [*POSITIONS,'ALL']:
        rows=[r for r in membership if p=='ALL' or r['position']==p]
        values=np.array([int(r['L']) for r in rows])
        input_stats.append({'position':p,'n_question_position_inputs':len(rows),'n_questions':500,'L_min':values.min(),'L_mean':values.mean(),'L_median':np.median(values),'L_p90':np.quantile(values,.9),'L_max':values.max(),'actual_N':1,'original_documents':20,'mapped_L':4096,'mapped_N':4,'dynamic_T':1,'dynamic_S':1,'unit':'document tokens; original20 documents, actual1 encoding block; ALL has1500 position inputs'})
    csvout('input_distribution.csv',input_stats)

    provenance=[
        ('样本与原始输入','attempts.jsonl:key.stage/sample_id/method/position + snapshot litm.json','stage=litm; 500 IDs×3位置×3方法; all success attempt1 cold miss','4500生成;1500输入;500问题','membership.csv;per_request_records.csv'),
        ('文档集合与位置','前置scoring_litm_input_checks.csv + frozen public_document_ids','前置逐title/text多重集合、唯一gold/参考/模板PASS; 本阶段删gold后19篇ID顺序相同','1500输入, 每份20篇;500问题配对','preaudit_input_checks.csv;membership.csv'),
        ('官方首行EM','answer/refs + 前置501条修正表','先取首个换行前内容并strip; 按官方规范化子串EM; 保存分数不改','0–1; 展示×100; 501条修正147问题;9实例复核','scoring_changed_records.csv;official_entrypoint_fixtures.csv;main_quality.csv'),
        ('总体EM和方法差','每题三位置三方法EM','每题三位置等权平均再汇总; 方法差=比较−参照','分母500问题; 每法1500请求; EM百分点','per_question_scores.csv;main_quality_paired.csv'),
        ('位置效应','同问题EM0/EM9/EM19','中间−两端均值: [-.5,1,-.5];末−首:[-1,0,1]','500问题配对;各95%区间单独计算','main_position_contrasts.csv;main_paired_position_contrasts.csv'),
        ('重采样','sorted500 sample_id以及全部3位置/3方法向量','default_rng(20261116),5000次;每次500问题有放回;2.5/97.5分位数','500问题;9条件共同保留','bootstrap_questions.csv;environment.json'),
        ('效率','ttft,total,cache_build/get,dynamic_selection,peak allocated/reserved','cold miss;文档重新编码;无注意力观察器','均值/中位数/P90;均值CI问题配对;秒/GiB','efficiency_descriptive.csv;efficiency_paired.csv'),
        ('加速比','同题同位置的两方法耗时','逐请求比值=参照/比较;另算两均值之比','总体均值/中位数/P90基于1500实际请求;CI基于500问题','efficiency_paired.csv'),
        ('生成上限/EOS','output_tokens/token_ids; frozen max_new_tokens; generation_config:eos_token_id','cap=128;EOS看最后token;首行是否完成由换行及EOS记录','上限1465/4500;EOS3046;EOS恰上限11','generation_limits.csv;generation_cap_records.csv'),
        ('案例','dynamic−fixed的每题三位置平均官方EM','最大正差/负差/零差;同分sample_id排序;保留全部位置','三题×三位置;原答案/首行/参考/参数/源行号','selected_cases.csv;selected_case_positions.csv'),
    ]
    csvout('provenance.csv',[dict(zip(['result','raw_fields','filter_and_definition','denominator_units','outputs'],r)) for r in provenance])
    dictionary=[
        ('source','生成记录该字段为null;原始来源从冻结sample_id/position与question连接三份原作者位置文件','nullable metadata, not a missing reference'),
        ('public_document_ids','已核对的gold公开记录ID为None;其余19篇ID非空且唯一, 三位置的相对顺序相同','document IDs'),
        ('position','gold资料零基索引0/9/19, 对应第1/10/20篇','original-document index'),
        ('actual_N','实际编码块数1;20篇资料共同自然分块,长度未达7500-token分块阈值','encoded block count'),
        ('saved_full_answer_em','原campaign将整段回答送入best_subspan_em所得保存标签','EM0-1; retained unchanged'),
        ('official_first_line_em','取answer.split(newline)[0].strip再按官方规范化子串匹配任一参考','EM0-1; cached pre-audit correction'),
        ('ALL','先求每题三位置均值,再求500题均值;总体W/T/L按每题均值比较','500 question units,1500 requests/method'),
        ('efficiency median/P90','总体在实际1500次请求上取分位数;CI将三位置保留在问题内','s/tokens/GiB'),
        ('after_ttft_seconds','总响应−首token时间,含剩余生成与最终文本解码','seconds'),
        ('cap_after_first_newline','达到128 token时已出现换行;官方EM只读取换行前文本','request count'),
        ('cap_unterminated_first_line','到128 token、无EOS且未出现换行','request count'),
        ('stage_calls','APE阶段计数器;顺序路径的stage=None故该计数为0,执行耗时照常记录','instrumented stage calls'),
        ('confidence interval','固定模型、规则、模板下问题样本构成变化;各方法一个连续worker时段','paired question percentile interval'),
    ]
    csvout('field_dictionary.csv',[dict(zip(['field','definition','unit_or_scope'],r)) for r in dictionary])

    findings=[
        {'phenomenon':'原保存评分采用全文,官方入口采用首行','evidence':'501/4500标签由1变0,涉及147问题;末位动态−固定由全文-1.0变为首行+4.4pp','affected':'位置EM、总体EM及所有配对质量结论','handling':'复用前置501条修正;9原答案实例复核;全文分数另表保留','followup':'未来生成与评分入口统一使用作者首行约定'},
        {'phenomenon':'全部输入为单编码块且低于开发长度下界','evidence':'原20篇;L2532–3573,N1;全部映4K/4,动态1/1','affected':'动态规则迁移和位置效应解释','handling':'明确资料位置在一个编码块内变化;固定.7/1与动态1/1实际参数另列','followup':'同20篇完整资料控制实际N=1/4/8/16,分开研究资料位置与编码块位置'},
        {'phenomenon':'当前最低点在末位','evidence':'三方法位置19均低于位置9;末−首为-9.2/-16.0/-8.2pp','affected':'位置敏感性的结论','handling':'同时报告中间−两端和末−首;保留三点原曲线','followup':'扩展预先冻结的位置点,检查位置曲线形状'},
        {'phenomenon':'高比例达到生成上限,多数首行已完成','evidence':'1465/4500达到128;1391已换行;69无EOS且首行未换行;393次评分修正在cap组','affected':'长输出的质量与总响应','handling':'保留全部原始生成,将EOS、上限、首行状态和质量一并统计','followup':'统一比较EOS/128与首个换行停止;对69类未完成首行条件成对提高上限'},
        {'phenomenon':'缺逐token时标','evidence':'仅保存TTFT和总响应;首行换行时刻缺失','affected':'首行停止规则的实际加速量','handling':'按已有完整生成计时报告;输出状态单列','followup':'补首个换行时刻或逐token时标,实测停止规则的时延'},
        {'phenomenon':'单块APE同输出路径耗时较高','evidence':'顺序/动态1044对token相同,动态平均总响应仍多1.406秒','affected':'单块效率与输出长度的归因','handling':'单列同输出组;记录APE两分支query及完整冷请求实现','followup':'同输入同输出随机交错重复测时,拆分prefill/query与后续decode'},
    ]
    csvout('audit_findings.csv',findings)
    source_notes=f'''# 官方评分来源查阅\n\n本阶段于2026-11-16查阅作者main分支的[评测入口]({ENTRY})与[指标实现]({METRIC})。入口get_metrics_for_example在第57行先保留回答首行并去首尾空白，再调用best_subspan_em；指标按SQuAD式规范化后检验任一参考是否为回答子串。\n\n保存的作者指标文本位于restored/datasets/raw-main80/references/litm_metrics.py.txt。本次主分数来自冻结生成及前置审核修正表，九个原答案实例分别复核全文和首行两次函数结果。在线main作为本次观察的作者入口，原始campaign证据保持只读。\n'''
    (OUT/'sources/official_scoring_notes.md').write_text(source_notes,encoding='utf-8')

    report='# 实验第七阶段：Lost in the Middle位置实验\n\n'
    report+='## 摘要\n\n我们在Lost in the Middle原作者20篇资料版本上，比较顺序编码、固定APE和冻结动态APE。500个问题分别将同一gold资料置于位置0、9、19，共4500条冷请求生成。我们核对文档多重集合、唯一gold、干扰资料相对顺序和参考答案，复用前置审核的501条官方首行评分修正，并以问题为单位保留三位置、三方法作5000次Bootstrap。三方法总体首行EM为58.47%、58.47%、58.33%；动态相对固定差-0.13个百分点，95%区间[-2.40,2.20]。动态在末位提高4.40个百分点，在开头和中间分别下降3.40、1.40个百分点。三条曲线最低点均在末位；固定的末位相对开头下降16.0个百分点，动态下降8.2个百分点。20篇资料实际均编码为一块，动态规则全部映射为1/1。动态平均总响应4.122秒，较固定缩短0.129秒，较顺序增加1.681秒；生成上限命中32.56%，其中大部分已生成完整首行。结果呈现位置间收益转移及单块执行开销，整体质量接近持平。\n\n关键词：Lost in the Middle；证据位置；官方首行EM；动态参数；配对重采样\n\n'
    report+='## 1 研究问题、材料与实际设置\n\n我们评价证据资料在开头、中间和末端时的问答质量，检查固定和动态APE对位置差异及效率的影响。输入取原作者nq-open-20_total_documents_gold_at_0/9/19三个版本，各500个相同问题；位置为零基，对应第1、10、20篇。固定参数0.7/1.0，动态使用实验第三阶段已冻结规则。\n\n'
    report+='每份冻结输入保留完整20篇资料，不裁剪或填充，gold资料全文和19篇干扰资料相同；删除gold后，500题的干扰ID序列在三个位置完全一致。标题/正文多重集合、唯一gold及其位置、问题、参考和共同模板由前置1500条检查逐项核对，本阶段再次连接输入及实际记录。公开gold记录ID为None，其他19篇ID非空且无重复；生成source字段为null，来源通过冻结question/position连接原始位置文件。\n\n'
    report+='实际文档长度L=2532–3573 tokenizer tokens，全部低于自然分块7500-token阈值，实际N=1。20篇检索资料在一个编码块内移动，三个方法的prefix/request长度和query起始位置逐输入相同。动态全映射至开发九格4K/4，实际参数为1.0/1.0；固定为0.7/1.0。输入L与N均低于开发网格下界。\n\n'
    report+='模型为Llama-3.1-8B-Instruct，物理GPU0、bfloat16、batch1、greedy，生成配置seed42，统一max_new_tokens=128，停止规则为EOS或上限，关闭注意力观察器。软件版本及硬件登记沿用server_environment.csv：PyTorch2.6.0+cu124、Transformers4.44.2、FlashAttention2.7.4.post1；方案登记RTX4090 48GB，本campaign缺GPU型号/总显存/驱动遥测。所有结果来自同一模型和campaign。\n\n'

    report+='## 2 官方评分入口与原保存口径\n\n'
    report+=f'作者[评测入口]({ENTRY})先取回答的首个换行前内容并去首尾空白；[best_subspan_em]({METRIC})将回答和参考转小写、删除ASCII标点与英文冠词、合并空白，再检验任一规范化参考是否为规范化回答的子串。主指标为EM原值0–1，以下乘100展示。\n\n'
    report+='原campaign把完整回答传入同一指标。前置审核已逐条核对4500条，得到501条由1改为0的首行标签，涉及147题。本阶段按sample_id/method/position连接这些标签，并核对源行、attempt、L/N和保存分数；其余3999条沿用已审核分数。另取9个原答案实例验证全文与首行函数结果，共18次函数调用。原始score保留，未重新给4500条生成评分。\n\n'
    report+=md(['方法','原全文总体EM/%','官方首行总体EM/%','差/百分点','修正条数/1500'],[[LABEL[m],f"{float(q(m,s=FULL)['em_percent']):.2f}",f"{float(q(m)['em_percent']):.2f}",f"{float(next(r['official_minus_saved_pp'] for r in impact if r['method']==m and r['position']=='ALL')):+.2f}",next(r['changed_generations'] for r in impact if r['method']==m and r['position']=='ALL')] for m in METHODS])
    report+='\n原summary.json按全文口径汇总的12个位置/总体均值与本次一致；首行修正改变方法对比。末位动态−固定从全文-1.0个百分点变为官方首行+4.4个百分点；总体从-1.4变为-0.133个百分点。末位固定APE有81/500条从全文命中改为首行未命中，动态为54条、顺序为56条；固定与动态修正条数相差27条，对应末位方法差变化5.4个百分点。完整差异和原答案保存在scoring_changed_records.csv、method_pair_scoring_scope_shift.csv。\n\n'
    report+='以litm:0003的动态APE、位置19为例，参考为“October 27, 1904”或“1904”；首行叙述1932与1844，后续段落出现1904，全文得1而首行得0。该最小实例的完整答案、首行、参考与源行在official_entrypoint_fixtures.csv保留。\n\n![评分入口对位置曲线的影响](../figures/scoring_scope_comparison.png)\n\n'

    report+='## 3 统计与分母\n\n4500条记录均成功、attempt=1、cold miss，各条件500条，无重复成功、失败、空答案、缺参考或非法值；首行为空及规范化空参考均为0。Bootstrap单位为500个问题，固定seed20261116，进行5000次有放回抽样；每次抽500个问题，完整保留该题三位置及三方法。区间取2.5%与97.5%分位数，反映这批问题样本构成的变动。\n\n'
    report+='总体EM先对每题三位置等权平均，再对500题平均；总体方法差及胜/平/负也按题均值计算，分母500。每位置胜/平/负按该位置的500题计算。效率均值对应每方法1500次实际冷请求，中位数/P90在这1500条上计算；均值CI仍按500问题共同保留三位置抽样。每个区间对应一个指定差值或位置对比。\n\n'

    report+='## 4 官方首行质量与方法配对\n\n'
    report+=md(['方法','位置0 EM/%','位置9 EM/%','位置19 EM/%','总体EM/%','总体95%区间'],[[LABEL[m],*[f"{float(q(m,p)['em_percent']):.2f}" for p in POSITIONS],f"{float(q(m)['em_percent']):.2f}",interval(q(m),'ci_low_percent','ci_high_percent')] for m in METHODS])
    report+='\n![官方首行位置曲线](../figures/official_position_curve.png)\n\n'
    report+=md(['动态−固定','差/EM百分点','95%区间','胜/平/负','统计分母'],[[p if p!='ALL' else '每题三位置均值',f"{float(pair(p)['delta_pp']):+.3f}",interval(pair(p)),f"{pair(p)['wins_comparison']}/{pair(p)['ties']}/{pair(p)['losses_comparison']}",'500问题'] for p in [*POSITIONS,'ALL']])
    report+='\n动态在末位提高4.4个百分点，区间[0.6,8.2]；开头下降3.4，中间下降1.4。总体成功的题目×位置条件数为固定877/1500、动态875/1500，按问题聚合后差-0.133个百分点，区间[-2.4,2.2]。顺序与固定总体点估计相同，动态与顺序的总体差同为-0.133个百分点，完整三方法对比见main_quality_paired.csv。\n\n![动态与固定的配对差](../figures/dynamic_fixed_difference.png)\n\n'

    report+='## 5 中间相对两端与末位下降\n\n定义中间缺口G=EM₉−(EM₀+EM₁₉)/2，末首差D=EM₁₉−EM₀，均在同一问题内计算后汇总。负值分别表示中间低于两端均值、末位低于首位。\n\n'
    report+=md(['方法','G/百分点','G的95%区间','D/百分点','D的95%区间'],[[LABEL[m],f"{float(contrast(m,'middle_minus_ends_mean')['contrast_pp']):+.2f}",interval(contrast(m,'middle_minus_ends_mean')),f"{float(contrast(m,'last_minus_first')['contrast_pp']):+.2f}",interval(contrast(m,'last_minus_first'))] for m in METHODS])
    report+='\n三条曲线最低点均在位置19。固定APE开头67.0%、末位51.0%，末首差-16.0个百分点；动态开头63.6%、末位55.4%，末首差-8.2。动态相对固定的末首差改善+7.8个百分点，区间' + interval(pc('last_minus_first')) + '；动态相对顺序为+1.0个百分点，区间' + interval(pc('last_minus_first',a='sequential')) + '。\n\n'
    report+='中间缺口按另一组分母比较：动态G=-3.5、固定G=-1.6，动态−固定为-1.9个百分点，区间'+interval(pc('middle_minus_ends_mean'))+'。末位恢复抬高了动态的两端均值，同时中间EM降低1.4个百分点，形成该G差。固定相对顺序的G点估计改善1.8个百分点，既包含中间提高1.2，也包含两端均值降低0.6。两个位置指标分别呈现开头、末端和中间的变化来源。\n\n![位置对比](../figures/position_contrasts.png)\n\n'

    report+='## 6 冷请求效率与同输出比较\n\n以下每方法1500请求、500问题；模型加载单列，不含注意力采集。TTFT覆盖prepare、参数分支、资料编码和首token；总响应继续覆盖完整生成与文本解码。\n\n'
    report+=md(['方法','TTFT均值/秒','TTFT中位数/P90','总响应均值/秒','总响应中位数/P90','输出token均值'],[[LABEL[m],f"{float(e(m,'ttft_seconds')['mean']):.4f}",f"{float(e(m,'ttft_seconds')['median']):.4f}/{float(e(m,'ttft_seconds')['p90']):.4f}",f"{float(e(m,'total_seconds')['mean']):.4f}",f"{float(e(m,'total_seconds')['median']):.4f}/{float(e(m,'total_seconds')['p90']):.4f}",f"{float(e(m,'output_tokens')['mean']):.3f}"] for m in METHODS])
    report+='\n![冷请求效率](../figures/cold_efficiency.png)\n\n'
    report+=md(['比较−参照','总响应差/秒','95%区间/秒','逐题加速比均值/中位数','均值耗时比'],[[LABEL[b]+'−'+LABEL[a],f"{float(perf('total_seconds',a,b)['delta_comparison_minus_reference_s']):+.4f}",interval(perf('total_seconds',a,b),'delta_ci_low_s','delta_ci_high_s',4),f"{float(perf('total_seconds',a,b)['paired_speedup_mean']):.4f}/{float(perf('total_seconds',a,b)['paired_speedup_median']):.4f}",f"{float(perf('total_seconds',a,b)['ratio_of_means']):.4f}"] for a,b in [('sequential','ape_fixed'),('sequential','ape_dynamic'),('ape_fixed','ape_dynamic')]])
    report+='\n加速比为参照耗时÷比较耗时，逐请求比值平均与两方法均值之比分别保存。动态相对固定总响应缩短0.1293秒，区间[-0.2327,-0.0297]，平均输出少2.299 token；TTFT缩短0.475毫秒，区间[-0.719,-0.240]毫秒。动态与固定的同token输出只有189/1500对；该组总响应平均差-0.0008秒。\n\n'
    report+='顺序与动态有1044/1500对输出token完全相同，占69.6%；该同输出组动态总响应仍平均增加1.4058秒，输出均值相同为61.90 token。源码中APE仍执行共享prefix和context阶段、query时文档/其他两分支注意力及混合；本批N=1输入保留该执行路径。整体动态总响应为顺序的1.689倍，增加1.6815秒，区间[1.6008,1.7618]。\n\n'
    report+=md(['方法','缓存构建均值/秒','缓存获取均值/微秒','选参分支均值/微秒','allocated均值/GiB','reserved均值/GiB'],[[LABEL[m],f"{float(e(m,'cache_build_seconds')['mean']):.4f}",f"{float(e(m,'cache_get_seconds')['mean'])*1e6:.3f}",f"{float(e(m,'dynamic_selection_seconds')['mean'])*1e6:.3f}",f"{float(e(m,'peak_allocated_bytes')['mean']):.3f}",f"{float(e(m,'peak_reserved_bytes')['mean']):.3f}"] for m in METHODS])
    report+='\n显存GiB=bytes/2³⁰，峰值包含模型、请求开始时存活缓存及本次请求，reserved保留worker分配器历史。全部时间、长度和显存的均值/中位数/P90及均值95%区间见efficiency_descriptive.csv，各位置分别保留。三worker的模型加载分别为'+ '、'.join(f"{float(next(r['model_load_seconds'] for r in workers if r['method']==m)):.3f}" for m in METHODS)+'秒。\n\n'
    report+='三个方法依次运行，起止为'+ '；'.join(LABEL[m]+': '+beijing(next(r['started_at'] for r in workers if r['method']==m))+' 至 '+beijing(next(r['ended_at'] for r in workers if r['method']==m)) for m in METHODS)+'。时间CI反映问题构成变化；交错重复运行的测时变动需后续保存。\n\n'

    report+='## 7 生成上限、首行完成状态与时延\n\n'
    report+=md(['方法','达到128上限/1500','EOS终止','EOS恰在上限','上限时已换行','首行未结束/无EOS'],[[LABEL[m],next(r['cap_hits'] for r in limits if r['method']==m and r['position']=='ALL'),next(r['eos_last_count'] for r in limits if r['method']==m and r['position']=='ALL'),next(r['eos_at_cap_count'] for r in limits if r['method']==m and r['position']=='ALL'),next(r['cap_after_first_newline_count'] for r in limits if r['method']==m and r['position']=='ALL'),next(r['cap_unterminated_first_line_count'] for r in limits if r['method']==m and r['position']=='ALL')] for m in METHODS])
    report+='\n合计1465/4500达到上限（32.56%），3046条以EOS结束，其中11条EOS恰在上限。1465条上限记录中1391条已出现换行（94.95%），69条无EOS且未换行，余5条无换行但以EOS恰在上限终止。官方EM只读取首行，已换行后的继续生成仍计入实际总响应。501条评分修正中393条位于上限组。\n\n'
    report+=md(['方法/上限组','请求数','官方首行EM/%','原全文EM/%','平均总响应/秒','平均输出token'],[[LABEL[m]+(' 达上限' if hit=='True' else ' 未达上限'),r['n_requests'],f"{float(r['official_em_percent']):.2f}",f"{float(r['saved_full_em_percent']):.2f}",f"{float(r['mean_total_s']):.3f}",f"{float(r['mean_output_tokens']):.2f}"] for m in METHODS for hit in ['False','True'] for r in caps if r['method']==m and r['position']=='ALL' and r['cap_hit']==hit])
    report+='\n上述为按实际生成长度形成的描述性分组。上限组的首行/全文差与总响应一同保留，原答案和EOS条件可追溯到generation_cap_records.csv。现有记录缺少首个换行时刻或逐token时标；首行停止规则的实际耗时需要补测。对于69条未完成首行的条件，可成对提高上限并继续用官方首行EM检查答案完成情况。\n\n![上限状态与评分影响](../figures/cap_and_scoring_effects.png)\n\n![同题长度和时延差](../figures/paired_output_latency.png)\n\n'

    report+='## 8 统一选例\n\n以动态−固定的每题三位置平均官方EM选择最大改善、最大退化及零差案例，并列按sample_id排序；所有位置及两方法完整原答案均保留在selected_case_positions.csv。总体持平按三位置均值定义，逐位置是否完全相同另列。\n\n'
    for i,r in enumerate(cases,1):
        report+=f"### 8.{i} {r['case_type']}：{r['sample_id']}\n\n问题：{r['question']}。参考：{r['reference_answers']}。三位置平均EM差{float(r['dynamic_minus_fixed_mean_em_0_1'])*100:+.2f}个百分点；逐位置分数全部相同={r['dynamic_fixed_each_position_equal']}。固定0.7/1.0，动态1/1。\n\n"
        report+=md(['位置','固定首行EM','动态首行EM','固定/动态输出token','源行 固定/动态'],[[v['position'],v['fixed_official_em'],v['dynamic_official_em'],v['fixed_output_tokens']+'/'+v['dynamic_output_tokens'],v['fixed_source_line']+'/'+v['dynamic_source_line']] for v in case_positions if v['sample_id']==r['sample_id']])
        report+='\n'
    report+='![三个案例的全部位置](../figures/selected_case_positions.png)\n\n'

    report+='## 9 审核、讨论与补实验\n\n'
    report+=md(['现象','证据/范围','本阶段处理','建议补充'],[[r['phenomenon'],r['evidence'],r['handling'],r['followup']] for r in findings])
    report+='\n当前位置实验的20篇资料均在一个编码块内，动态规则只取1/1分支；其总体质量接近顺序编码，1044对生成token相同。固定参数改变了开头和末位间的得分分布；动态恢复末位分数，同时开头分数降低，总体几乎抵消。开发N=4/8/16与本批实际N=1的条件差异需要在后续受控块数实验中检验。\n\n'
    report+='后续实验建议按同题同资料设计：控制实际编码块数并保留三位置；统一比较首个换行停止与现行EOS/128停止；对未完成首行条件增加输出上限；保存换行/逐token时标并进行随机交错的重复计时。本轮只登记这些建议，保持原始记录和冻结规则。\n\n'
    report+='## 10 结论与复现\n\n官方首行EM下，顺序、固定和动态APE总体为58.47%、58.47%、58.33%。动态相对固定总体差-0.133个百分点，95%区间[-2.40,2.20]；末位提高4.4个百分点、开头下降3.4个百分点。三曲线最低点均为末位。动态使末首落差从固定的16.0缩至8.2个百分点，中间相对两端缺口则从1.6增至3.5个百分点，两个位置指标对应不同的变化来源。\n\n'
    report+='动态平均总响应较固定缩短0.129秒，较顺序增加1.681秒。实际N=1、长输出和128-token上限共同构成这批效率条件；上限记录中1391条已完成首行，后续段落的生成耗时保留在当前测量中。官方入口修正改变501条标签，主表采用首行评分，全文结果用于审核对照。\n\n'
    report+=f"分析提交{env['analysis_commit']}，与参考提交相同。CPU使用已有Python、NumPy、Matplotlib和regex；具体版本及seed20261116、5000次抽样方法见environment.json。三个本阶段脚本及运行命令见README，关键字段、筛选、分母和公式见provenance.csv。研究状态保持SUBMITTED，执行停止在实验第七阶段。\n"
    (OUT/'reports/LITM_REPORT.md').write_text(report,encoding='utf-8')

    audit='# 第七阶段审核记录\n\n'
    audit+='本阶段读取4500条LITM生成，连接500题×3位置×3方法。全部成功、attempt1、cold miss；参考、实际L/N、prefix/request和query位置一致。源全文分数保留，主表另列官方首行分数。\n\n'
    audit+=md(['审核项','实际结果','证据'],[['输入与位置','1500前置title/text、gold、模板检查PASS;500题删除gold后干扰顺序一致','preaudit_input_checks.csv;membership.csv'],['来源字段','生成source为null;gold公开ID为None;通过question和position连接原作者输入','membership.csv;sources/input_sources.json'],['评分入口','501条修正147题;9fixture/18函数调用通过;12个原全文汇总均值匹配','official_entrypoint_fixtures.csv;original_summary_crosscheck.csv'],['统计单位','500问题;三位置及三方法共同保留;总体W/T/L按题均值','bootstrap_questions.csv;main_quality_paired.csv'],['实际配置','原资料20篇;实际N1;L2532–3573;动态1/1;固定.7/1','input_distribution.csv;per_request_records.csv']])
    audit+='\n'+md(['现象','影响','证据','处理及补充'],[[r['phenomenon'],r['affected'],r['evidence'],r['handling']+'；'+r['followup']] for r in findings])
    audit+='\n最小证据保留在official_entrypoint_fixtures.csv、scoring_changed_records.csv、selected_case_positions.csv和generation_cap_records.csv；每条含sample_id/method/position、源行与原答案。原作者入口来源见sources/official_scoring_notes.md。关键数字与公式见provenance.csv，原始证据只读，研究状态SUBMITTED，下一阶段未执行。\n'
    (OUT/'reports/AUDIT.md').write_text(audit,encoding='utf-8')

    links=[('阶段报告','reports/LITM_REPORT.md'),('审核记录','reports/AUDIT.md'),('官方首行质量主表','tables/main_quality.csv'),('方法配对差和胜平负','tables/main_quality_paired.csv'),('位置差','tables/main_position_contrasts.csv'),('位置效应的方法差','tables/main_paired_position_contrasts.csv'),('500题九条件分数','tables/per_question_scores.csv'),('1500输入及文档顺序','tables/membership.csv'),('4500原答案和逐请求计时','tables/per_request_records.csv'),('501条首行修正及原答案','tables/scoring_changed_records.csv'),('评分口径对配对结论的影响','tables/method_pair_scoring_scope_shift.csv'),('九个官方入口实例','tables/official_entrypoint_fixtures.csv'),('效率均值/中位数/P90及CI','tables/efficiency_descriptive.csv'),('配对加速比与时间差','tables/efficiency_paired.csv'),('上限与EOS','tables/generation_limits.csv'),('同输出长度与时延分组','tables/paired_output_length_groups.csv'),('案例全部位置及原答案','tables/selected_case_positions.csv'),('500重采样单位','tables/bootstrap_questions.csv'),('原汇总核对','tables/original_summary_crosscheck.csv'),('来源与公式','tables/provenance.csv'),('字段含义','tables/field_dictionary.csv'),('环境与种子','environment.json')]
    links.insert(-1,('官方评分来源查阅','sources/official_scoring_notes.md'))
    index='# 实验第七阶段：Lost in the Middle\n\n本阶段完成；研究状态SUBMITTED，执行停止于本阶段。\n\n'+''.join(f'- [{title}]({path})\n' for title,path in links)
    index+='\nfigures/含8组PNG/SVG图。质量曲线来自main_quality.csv，评分口径图来自quality.csv，位置差来自main_position_contrasts.csv，上限图来自generation_limits/scoring_scope_impact，案例图来自selected_case_positions，长度时延图来自per_request_records.csv。主指标为官方首行EM，全文分数另列。\n\n'
    index+='本阶段依赖已交付前置审核的scoring_litm_input_checks.csv、scoring_litm_first_line_changed_IDs.csv和九实例表；复用修正标签，保持原score。CPU命令从仓库根目录运行：\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n$env:PYTHONDONTWRITEBYTECODE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_litm_stage7.py\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/plot_litm_stage7.py\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_litm_stage7.py\n```\n\n使用已有依赖，版本见environment.json。seed20261116，5000次问题配对Bootstrap，三位置/三方法共同保留。总体每方法1500请求、500问题；质量先按题平均三位置，效率分位数在实际请求上计算。新增文件、缓存和临时目录均位于pc-analysis-output；未执行Doc2Dial阶段。\n'
    (OUT/'README.md').write_text(index,encoding='utf-8')
    root=(BASE/'README.md').read_text(encoding='utf-8')
    root=root.replace('当前已完成实验第六阶段：LongBench正式五方法比较。','当前已完成实验第七阶段：Lost in the Middle位置实验。')
    if '[实验7：Lost in the Middle]' not in root:
        root=root.replace('- [实验6：LongBench五方法正式比较]','- [实验7：Lost in the Middle](stages/07_litm/README.md)\n- [实验6：LongBench五方法正式比较]')
    root=root.replace('本轮已停在实验6；','本轮已停在实验7；')
    (BASE/'README.md').write_text(root,encoding='utf-8')
    print('Experiment 7 report, audit, provenance and index written.')


if __name__=='__main__':
    main()
