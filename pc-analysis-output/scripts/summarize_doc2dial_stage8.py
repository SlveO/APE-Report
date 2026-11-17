"""Chinese stage-8 report/audit/provenance from the derived Doc2Dial tables."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/08_doc2dial'
METHODS=['sequential','ape_fixed','ape_dynamic'];CACHES=['cold','hot']
LABEL=dict(zip(METHODS,['顺序编码','固定APE','动态APE']))
CACHE_LABEL={'cold':'冷请求','hot':'热请求'}
EVAL='https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/sharedtask_utils.py'
TEMPLATE='https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/subtask2/seq2seq_utils.py'
BLEU='https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/metrics/bleu.py'


def readcsv(name):
    with (OUT/'tables'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def md(headers,rows):
    def txt(v):return str(v).replace('|','\\|').replace('\n','<br>')
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(txt(v) for v in r)+' |' for r in rows)+'\n'


def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    env=json.loads((OUT/'environment.json').read_text(encoding='utf-8'))
    quality,pairs,efficiency,performance,membership,limits,caps,cases,workers,diagnostics=[readcsv(n) for n in ['quality.csv','quality_paired.csv','efficiency_descriptive.csv','efficiency_paired.csv','membership.csv','generation_limits.csv','generation_cap_conditioned.csv','selected_cases.csv','model_load_workers.csv','bootstrap_diagnostics.csv']]
    def q(m,c='cold',scope='all500',unit='dialogue'):return next(r for r in quality if r['scope']==scope and r['cluster_unit']==unit and r['method']==m and r['cache_mode']==c)
    def qp(a='ape_fixed',b='ape_dynamic',c='cold',scope='all500',unit='dialogue'):return next(r for r in pairs if r['scope']==scope and r['cluster_unit']==unit and r['reference_method']==a and r['comparison_method']==b and r['reference_cache']==c and r['comparison_cache']==c)
    def e(m,c,metric,unit='dialogue'):return next(r for r in efficiency if r['cluster_unit']==unit and r['method']==m and r['cache_mode']==c and r['metric']==metric)
    def perf(a,b,metric,unit='dialogue'):return next(r for r in performance if r['cluster_unit']==unit and (r['reference_method'],r['reference_cache'])==a and (r['comparison_method'],r['comparison_cache'])==b and r['metric']==metric)
    def interval(r,lo='ci_low',hi='ci_high',digits=3):
        return '—（仅1资料簇）' if r[lo]=='' else f"[{float(r[lo]):.{digits}f}, {float(r[hi]):.{digits}f}]"
    def bj(s):return datetime.fromisoformat(s).astimezone(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    csvout('main_quality.csv',[r for r in quality if r['scope']=='all500' and r['cluster_unit']=='dialogue'])
    csvout('main_quality_document_sensitivity.csv',[r for r in quality if r['scope']=='all500' and r['cluster_unit']=='document'])
    csvout('main_quality_paired.csv',[r for r in pairs if r['scope']=='all500' and r['cluster_unit']=='dialogue'])
    csvout('main_quality_paired_document_sensitivity.csv',[r for r in pairs if r['scope']=='all500' and r['cluster_unit']=='document'])
    csvout('main_efficiency.csv',[r for r in efficiency if r['cluster_unit']=='dialogue'])
    csvout('cache_effects.csv',[r for r in performance if r['cluster_unit']=='dialogue' and r['comparison_type']=='cache'])
    csvout('method_efficiency_effects.csv',[r for r in performance if r['cluster_unit']=='dialogue' and r['comparison_type']=='method'])
    dist=[]
    for n in [1,2]:
        rr=[r for r in membership if int(r['actual_N'])==n];values=np.array([int(r['L']) for r in rr])
        dist.append({'actual_N':n,'n_targets':len(rr),'n_dialogues':len({r['dialogue_cluster'] for r in rr}),'n_documents':len({r['document_cluster'] for r in rr}),'L_min':values.min(),'L_mean':values.mean(),'L_median':np.median(values),'L_p90':np.quantile(values,.9),'L_max':values.max(),'mapped_L':int(rr[0]['mapped_L']),'mapped_N':int(rr[0]['mapped_N']),'dynamic_T':rr[0]['dynamic_T'],'dynamic_S':rr[0]['dynamic_S'],'original_document_count':1,'unit':'document tokens; actual encoded block counts; target/dialogue/document counts'})
    csvout('input_distribution.csv',dist)
    hist=[]
    for agent in [False,True]:
        for repeat in [False,True]:
            rr=[r for r in membership if (r['consecutive_agent_target']=='True')==agent and (r['reference_repeated_in_history']=='True')==repeat]
            hist.append({'previous_agent':agent,'reference_repeated_in_history':repeat,'n_targets':len(rr),'n_generations':len(rr)*6,'n_dialogues':len({r['dialogue_cluster'] for r in rr}),'n_documents':len({r['document_cluster'] for r in rr}),'denominator_targets':500,'unit':'mutually exclusive original dialogue conditions; six requests per target'})
    csvout('history_condition_counts.csv',hist)
    csvout('domain_counts.csv',[{'domain':d,'n_targets':sum(r['domain']==d for r in membership),'unit':'target count /500'} for d in sorted({r['domain'] for r in membership})])

    provenance=[
        ('目标/参考连接','frozen doc2dial.json:source/history/answers + attempts:key/source/reference_answers','500唯一ID; id=dial_id_(target_agent_turn-1);每目标六条件','500目标;366对话;172资料;3000生成','membership.csv;target_inputs.csv;per_request_records.csv'),
        ('完整历史与模板','history/history_turn_ids/prefix/suffix/template_adaptation; 前置500输入检查','资料前置;last_turn为紧邻上一发言;逆序完整历史;引用原agent目标','历史1–15轮;当前请求24–443 tokenizer tokens;28连续agent;7参考历史重复','preaudit_input_checks.csv;history_condition_counts.csv'),
        ('资料KV热复用','attempts cache_hit/build/stage_calls/token_ids + 前置events证据','1500同题冷热对;hot GPU KV hit;query/decode重新执行;无资料重建','1500次热请求;输出及参考完全相同;同方法worker进程','cold_hot_pairs.csv;hot_generation_evidence.csv'),
        ('语料级BLEU','完整answer与一条对应reference;13a;C1..4/T1..4及系统/参考长度','先按目标提取充分统计,全语料累加后计分;评分签名对照服务器2.4.3','BLEU0–100;每条件500目标;BP与各阶精度另列','bleu_sufficient_statistics.csv;main_quality.csv;original_summary_crosscheck.csv'),
        ('本地评分引擎','audit_stage1_scoring.py纯函数+NumPy向量化计分','runpy关闭main;六服务器统计/BLEU和5个边界fixture核对','独立实现;本机未安装sacrebleu;signature为服务器配置','scorer_validation.csv;environment.json'),
        ('主要CI','source:domain/doc_id/dial_id;六条件充分统计向量','366对话簇有放回抽同数簇;所有目标/六条件共同保留;每次重算corpus BLEU','seed20261116;5000次;2.5%/97.5%分位数','bootstrap_diagnostics.csv;quality_paired.csv'),
        ('资料敏感性CI','source:domain/doc_id','172资料簇共同抽样;保留资料下全部对话/目标/六条件','N2子组仅1资料时CI留空;点分数保留','main_quality_document_sensitivity.csv;scope_counts.csv'),
        ('效率与加速比','TTFT/total/cache_build/get/selection/output/peak字段','cold/hot和方法按同目标配对;均值CI按来源簇;中位数/P90在实际500请求上计算','时间s;显存GiB=bytes/2^30;逐对比均值与均值比另列','main_efficiency.csv;cache_effects.csv;efficiency_paired.csv'),
        ('生成上限/EOS','output_tokens/output_token_ids;max_new_tokens;generation_config EOS IDs','cap=128;EOS=末token属于停止集合;冷热分别计数','各方法/条件分母500;输出含EOS','generation_limits.csv;generation_cap_records.csv'),
        ('语言问题定向审核','1条西语参考原始target+冻结参考+英文请求/输出','按公开原始对话核对;词/重音候选筛查;主表保留500并另列499敏感性','确认1条,原参考和原评分保留','reference_language_candidates.csv;sources/reference_language_evidence.json'),
        ('案例','保存单目标默认corpus BLEU分数/原答案','最大正差/负差/零差,并列按ID;冷热相同仍保留四条源行','辅助选例分数0–100;总体结论取corpus BLEU','selected_cases.csv;dynamic_fixed_per_target.csv'),
    ]
    csvout('provenance.csv',[dict(zip(['result','source_fields','filter_definition','denominator_units','outputs'],r)) for r in provenance])
    definitions=[
        ('group','原冻结group为[doc2dial,domain,doc_id],对应资料;主对话簇另用domain/doc_id/dial_id','source cluster'),
        ('last_turn','紧邻目标之前的原始发言;28条为agent;latest_user_utterance另保存更早user发言','original utterance'),
        ('reference_answer_id','dial_id_(target_agent_turn_id-1),并按source target agent轮次取参考','strict ID join'),
        ('cache_mode hot','同目标在cold后重复提交;只复用GPU资料/方法前缀KV,本次重新处理完整历史并生成','same-process GPU KV condition'),
        ('quality corpus BLEU','完整生成;13a mixed case,1–4gram,exp smoothing,effective_order=False','0–100 BLEU points'),
        ('server_setting_signature','原服务器SacreBLEU2.4.3设置签名,本地为复核该设置的独立实现','signature is server configuration'),
        ('saved_single_target_default_corpus_bleu','服务器将单目标作为单条corpus调用所得分数,用于案例','auxiliary0–100; not averaged for main BLEU'),
        ('one document subgroup','N2七目标共一篇资料,其资料间变化缺观察;CI字段留空','within-document dialogue CI separately available'),
        ('ratio_of_means','参照平均时间÷比较平均时间;cache比较参照cold、比较hot','dimensionless, >1 comparison faster'),
        ('paired_speedup_mean','先逐目标计算参照/比较,再对500目标平均','dimensionless, CI samples paired clusters'),
        ('peak allocated/reserved','请求开始重置峰值,覆盖模型、存活资料缓存及本次请求;reserved含分配器历史','GiB in descriptive table; raw bytes in request table'),
        ('confidence scope','固定模型、模板、冻结参数下来源簇/目标构成变化;当前每法一个连续worker','paired dialogue main, document sensitivity'),
    ]
    csvout('field_dictionary.csv',[dict(zip(['field','definition','unit_scope'],r)) for r in definitions])
    findings=[
        {'phenomenon':'历史相关性与目标角色','evidence':'500目标来自366对话/172资料;28上一轮为agent,7参考历史重复,交集2','impact':'参考连接、last_turn解释及CI独立单位','handling':'500主表保留;467普通条件/33标记条件分层;对话主CI/资料敏感性','followup':'原始角色条件分别复核,新增规范user→agent目标对照'},
        {'phenomenon':'热请求为相同目标再次提交','evidence':'1500热请求同worker GPU KV命中,build/rebuild0,答案token全同','impact':'热缓存效率条件','handling':'资料KV与历史/请求/答案重新计算分开解释;保留事件行和阶段调用','followup':'增加同资料下不同目标轮次的连续应用热测'},
        {'phenomenon':'N2仅一篇资料','evidence':'493目标N1取1/1;7目标N2来自5对话/同一studentaid资料,取.7/.7','impact':'规则映射和多块子组质量解释','handling':'子语料单独累计计分;N2资料CI留空,5对话CI注明同资料内变化','followup':'增加独立N2及更大N资料的开发/验证/冻结后测试'},
        {'phenomenon':'高生成上限命中率且输出较参考长','evidence':'每法70%/76.2%/70%;13a系统长度为参考约5.64/5.75/5.64倍','impact':'语料precision和decode总响应','handling':'记录cap/EOS,按实际长度分组,原始答案/参考保留','followup':'统一简洁模板及输出长度控制对照,保存结束原因/逐token时标'},
        {'phenomenon':'英文请求/西语参考','evidence':'确认1085067334266d41636126e5b9ee9519_5原target6为西语,连接正确','impact':'该目标词面匹配和案例解释','handling':'原500保留;原始target/历史存最小证据;499目标敏感性另列','followup':'原标注人工语言核对后,另版本定义语言匹配的参考条件'},
        {'phenomenon':'毫秒差与连续worker顺序','evidence':'每方法一个worker,cold后立即hot;模型加载与请求分开','impact':'小幅方法差的时延归因及峰值reserved历史','handling':'报告源簇CI、顺序、输出长度、两种加速比及显存口径','followup':'同资料生命周期内交错重复测时,分解prefill/query/decode'},
    ]
    csvout('audit_findings.csv',findings)
    notes=f'''# 官方评分及模板来源\n\n本阶段于2026-11-17查阅[DialDoc21评分入口]({EVAL})、[seq2seq模板]({TEMPLATE})和[SacreBLEU2.4.3]({BLEU})。评分入口按agent目标构造dial_id与target_turn−1连接ID,调用语料级sacrebleu；模板保留last_turn标签、倒序历史、title和doc_context。\n\n本campaign将完整资料置于历史前,包装为共用Llama聊天输入,逐方法保持同一适配。服务器SacreBLEU2.4.3签名为`{result['server_signature']}`。PC未安装SacreBLEU,使用前置独立标准库实现和NumPy向量化充分统计计分,对照服务器六组数值/充分统计和边界fixture。\n\n官方历史脚本会跳过未grounding轮次,当前实现保留全部历史；本批500目标的历史中该类轮次为0。原始参考/模板的前置500项检查全部PASS。\n'''
    (OUT/'sources/official_scoring_and_template.md').write_text(notes,encoding='utf-8')

    report='# 实验第八阶段：Doc2Dial多轮问答与资料KV复用\n\n'
    report+='## 摘要\n\n我们使用Doc2Dial的500个grounded agent目标轮次,比较顺序编码、固定APE与冻结动态APE,分别观测冷请求和同进程GPU资料KV热请求,共3000条生成。目标来自366个对话和172篇资料,完整保留历史并按原始agent轮次ID连接参考。我们复算语料级BLEU,按对话进行5000次配对聚类Bootstrap,资料聚类作为敏感性。顺序、固定和动态BLEU为3.360、3.251、3.418,冷热完全相同；动态相对固定提高0.167分,95%区间[-0.205,0.562],资料敏感性区间[-0.192,0.514]。1500次热请求均命中资料KV并重新生成答案,平均TTFT从0.159/0.213/0.213秒降至0.040/0.064/0.065秒,总响应分别节省0.120/0.149/0.144秒。493目标实际N=1,7目标N=2且共用一篇资料；生成上限命中率70%–76.2%,生成文本为参考长度约5.6倍。我们保留连续agent、历史参考重复及1条语言差异参考的原始目标,另列分层与敏感性。结果为质量点估计小幅改善、冷热质量持平,KV复用收益主要体现在首token时间。\n\n关键词：Doc2Dial；语料BLEU；GPU KV复用；动态APE；多轮历史\n\n'

    report+='## 1 研究问题、目标与输入\n\n我们评价动态选参在公共服务多轮问答中的质量变化,并测量资料GPU KV复用对TTFT及总响应的作用。材料来自MAIN80冻结500个目标,领域分布为va154、dmv125、ssa123、studentaid98。每个目标包含完整原始资料、此前全部历史、目标agent轮次和一条参考回答,三方法各生成cold/hot两次。\n\n'
    report+='参考ID为dial_id与target_agent_turn−1拼接。这里的后缀取紧邻上一轮ID；28个目标的上一轮角色为agent,其最新user问题位于更早历史。我们分别保存last_turn与latest_user_utterance。完整历史为1–15轮,实际请求为24–443个模型token。500项原始输入/历史/资料/参考/模板核对及500项六条件配对全部通过。\n\n'
    report+=f'共同模板沿用作者[标签与倒序历史规则]({TEMPLATE}),将完整资料前置到title/doc_context之后,再提交last_turn和逆序完整历史,外层为同一Llama聊天封套。last_turn发言同时出现在完整历史中。资料前置使资料KV可在后续请求中复用,三方法采用同一适配。代码保留未grounding历史,本批该类历史轮次为0。\n\n'
    report+='模型为Llama-3.1-8B-Instruct,物理GPU0、bfloat16、batch1、greedy、seed42、统一128新token上限,EOS或上限终止,关闭注意力观察器。运行记录PyTorch2.6.0+cu124、Transformers4.44.2、FlashAttention2.7.4.post1；方案登记RTX4090 48GB,本campaign缺GPU型号/总显存/驱动遥测。配置和来源登记保存在server_environment.csv。当前结果来自同一模型、同一campaign及共同模板。\n\n'

    report+='## 2 缓存内容与热请求重新生成\n\n每个目标先cold后hot,两次请求的历史、last_turn及参数相同。每方法由一个worker处理,hot复用其进程中的匹配资料KV。\n\n'
    report+=md(['内容','冷请求','热请求'],[['标题/方法前缀及完整资料','重新编码并构建GPU KV','匹配同一资料/配置的GPU KV'],['完整历史、当前last_turn','prepare后进入query forward','再次prepare并进入query forward'],['答案','执行greedy生成循环','重新执行greedy生成循环'],['缓存构建/恢复','记录实际构建时间','1500次构建0、恢复重建0']])
    report+='\n1500个同目标/方法冷热对全部命中核对：独立request_started/ended事件完整、cold在hot之前、cold miss/hot hit、实际配置一致、输出token和参考完全相同。APE热请求的prefix/context调用为0、query调用等于本次输出token数；顺序方法使用普通forward路径,其APE-only计数器为0。相关事件行、原记录行和调用数保存在hot_generation_evidence.csv及cold_hot_pairs.csv。资料文件加载、GPU资料KV、历史请求和答案生成分别按以上实际路径说明。\n\n'

    report+='## 3 评分与统计口径\n\n'
    report+=f'质量按[DialDoc21评分入口]({EVAL})使用完整回答和严格ID连接的agent参考,复算corpus BLEU。SacreBLEU设置对照[2.4.3实现]({BLEU})：13a、mixed case、1–4gram、exp平滑、effective_order=False、一条参考。服务器签名为`{result["server_signature"]}`。\n\n'
    report+='本地未安装SacreBLEU；我们复用前置审核的标准库充分统计实现,用NumPy批量重算Bootstrap语料分数。六组服务器BLEU及系统/参考长度、各阶匹配数和总数逐项吻合,另通过5个零匹配、缺4gram及平滑边界实例。本阶段未安装依赖。单目标保存score用于选例；主表先累计所有目标的ngram计数/长度,再计算精度几何均值和长度惩罚。每次Bootstrap也重新累计后计分。\n\n'
    report+='按用户规划,主CI以366个对话聚类,资料172簇作敏感性。每次有放回抽取与原来同数的簇,保留簇内全部目标及六个方法/缓存条件。原冻结group字段对应资料,主对话簇另用domain/doc_id/dial_id。最大对话簇4目标、最大资料簇13目标。seed20261116,5000次,取2.5%和97.5%分位数；各对比区间单独计算。效率均值除以本次抽到的目标数,中位数/P90取实际每条件500请求。\n\n'
    report+='区间反映目标及其对话/资料来源的样本构成变化,参数和模板在重采样中固定。六条件配对共同保留,冷热相同答案在每次抽样中的语料分数仍一致。N=2子组只有1篇资料,资料间变动缺少第二篇观察,其资料CI留空；5个对话的CI对应这一篇资料内的对话变化。源簇成员、抽到的目标数量范围和随机子序列见membership.csv、bootstrap_diagnostics.csv。\n\n'

    report+='## 4 语料级BLEU与方法差\n\nBLEU以原0–100值展示,每条件500个目标。\n\n'
    report+=md(['方法','冷BLEU','热BLEU','对话聚类95%区间','资料聚类95%区间'],[[LABEL[m],f"{float(q(m)['corpus_bleu']):.3f}",f"{float(q(m,'hot')['corpus_bleu']):.3f}",interval(q(m)),interval(q(m,unit='document'))] for m in METHODS])
    report+='\n![三方法语料BLEU](../figures/corpus_bleu.png)\n\n'
    report+=md(['比较−参照','BLEU差','对话CI','资料CI'],[[LABEL[b]+'−'+LABEL[a],f"{float(qp(a,b)['delta_bleu']):+.3f}",interval(qp(a,b)),interval(qp(a,b,unit='document'))] for a,b in [('sequential','ape_fixed'),('sequential','ape_dynamic'),('ape_fixed','ape_dynamic')]])
    report+='\n动态相对固定的点估计提高0.167 BLEU,对话区间[-0.205,0.562]、资料区间[-0.192,0.514]；动态相对顺序提高0.058,两种区间也跨0。冷热答案逐token相同,各方法冷热BLEU差及其配对区间均为0。主表保持全部500个原始目标。\n\n![主要质量差的两种聚类区间](../figures/quality_differences.png)\n\n'
    report+='生成长度与精度分解如下,单位均为13a分词数或百分精度。BP在三方法主语料均为1。\n\n'
    report+=md(['方法','生成/参考13a分词数','长度比','1/2/3/4gram精度%'],[[LABEL[m],q(m)['system_length_13a']+'/'+q(m)['reference_length_13a'],f"{float(q(m)['length_ratio']):.3f}",'/'.join(f'{float(v):.3f}' for v in json.loads(q(m)['precision_1_4_percent']))] for m in METHODS])
    report+='\n参考共9475个13a token,均值18.95；生成约106.94–109.02个13a token/目标。完整输出比参考长约5.6倍且高阶ngram匹配比例低,形成表中约3.3–3.4的语料BLEU。模型输出token均值另表记录,与13a分词单位分别标注。\n\n'

    report+='## 5 实际编码块数、规则映射及历史条件\n\n'
    report+=md(['实际N','目标数','对话/资料数','L范围/token','映射格','动态T/S','顺序/固定/动态子语料BLEU'],[[r['actual_N'],r['n_targets'],f"{r['n_dialogues']}/{r['n_documents']}",f"{r['L_min']}–{r['L_max']}",f"{r['mapped_L']}/{r['mapped_N']}",f"{r['dynamic_T']}/{r['dynamic_S']}",'/'.join(f"{float(q(m,scope='actual_N_'+str(r['actual_N']))['corpus_bleu']):.3f}" for m in METHODS)] for r in dist])
    report+='\n493目标取1/1,7目标取0.7/0.7,固定均为0.7/1.0。全部实际N低于开发网格最小N=4；N2的7目标均来自同一篇studentaid资料、5个对话。N1动态−固定+0.210 BLEU,区间[-0.160,0.584]；N2为-3.123,同资料内对话CI[-5.018,0.240],资料间CI留空。总体由子组ngram计数相加后求BLEU,分组分数按各自子语料计算。\n\n![实际L/N与映射](../figures/actual_LN_mapping.png)\n\n'
    report+=md(['上一轮agent','参考历史重复','目标数','六条件生成数'],[[r['previous_agent'],r['reference_repeated_in_history'],r['n_targets'],r['n_generations']] for r in hist])
    report+='\n28个连续agent目标与7个历史参考重复目标相交2个,合计33目标、198条生成。467目标的上一轮为user且参考未在历史精确重复。500目标的完整历史保留,分层按原始输入条件定义。\n\n'
    report+=md(['原始条件','目标数','顺序BLEU','固定BLEU','动态BLEU','动态−固定及对话CI'],[[name,q('sequential',scope=scope)['n_targets'],*[f"{float(q(m,scope=scope)['corpus_bleu']):.3f}" for m in METHODS],f"{float(qp(scope=scope)['delta_bleu']):+.3f} {interval(qp(scope=scope))}"] for scope,name in [('clean_user_no_repeat','上一轮user且无历史参考重复'),('consecutive_agent','上一轮agent'),('reference_repeat','参考历史精确重复'),('history_flagged_union','两标记的并集')]])
    report+='\n例如22198e14bb6cd2a93ec1f6540101934b_2的目标与前一个agent文本均为gh,最近user文本为hg。该目标按官方agent轮次ID连接,原始片段及匹配历史轮次在history_condition_evidence.csv保存。\n\n![历史条件分层](../figures/history_subgroups.png)\n\n'

    report+='## 6 冷热效率与缓存收益\n\n每行500个目标请求,均值/中位数/P90单位为秒,输出token含EOS。\n\n'
    report+=md(['方法/条件','TTFT均值','TTFT中位数/P90','总响应均值','总响应中位数/P90','输出token均值'],[[LABEL[m]+'/'+CACHE_LABEL[c],f"{float(e(m,c,'ttft_seconds')['mean']):.4f}",f"{float(e(m,c,'ttft_seconds')['median']):.4f}/{float(e(m,c,'ttft_seconds')['p90']):.4f}",f"{float(e(m,c,'total_seconds')['mean']):.4f}",f"{float(e(m,c,'total_seconds')['median']):.4f}/{float(e(m,c,'total_seconds')['p90']):.4f}",f"{float(e(m,c,'output_tokens')['mean']):.3f}"] for m in METHODS for c in CACHES])
    report+='\n![冷热请求时延](../figures/cold_hot_latency.png)\n\n'
    report+=md(['方法/指标','热−冷/毫秒','对话CI/毫秒','逐对加速比均值/中位数','均值耗时比及CI'],[[LABEL[m]+'/'+metric,f"{float(perf((m,'cold'),(m,'hot'),metric)['delta_comparison_minus_reference_s'])*1000:+.2f}",f"[{float(perf((m,'cold'),(m,'hot'),metric)['delta_ci_low_s'])*1000:.2f},{float(perf((m,'cold'),(m,'hot'),metric)['delta_ci_high_s'])*1000:.2f}]",f"{float(perf((m,'cold'),(m,'hot'),metric)['paired_speedup_mean']):.3f}/{float(perf((m,'cold'),(m,'hot'),metric)['paired_speedup_median']):.3f}",f"{float(perf((m,'cold'),(m,'hot'),metric)['ratio_of_means']):.3f} {interval(perf((m,'cold'),(m,'hot'),metric),'ratio_of_means_ci_low','ratio_of_means_ci_high')} "] for m in METHODS for metric in ['ttft_seconds','total_seconds']])
    report+='\n加速比定义为cold时间÷hot时间；逐对比值均值与均值时间比按各自公式保留。平均TTFT的冷/热均值比分别为3.968、3.310、3.290,总响应均值比分别为1.037、1.025、1.025。热请求主要节省资料编码,输出生成仍完整执行。动态热请求首token后剩余平均5.741秒,占热总响应98.89%。\n\n![缓存加速比](../figures/cache_speedups.png)\n\n'
    report+=md(['方法/条件','KV构建均值/秒','获取均值/微秒','参数分支均值/微秒','allocated均值/GiB','reserved均值/GiB'],[[LABEL[m]+'/'+CACHE_LABEL[c],f"{float(e(m,c,'cache_build_seconds')['mean']):.4f}",f"{float(e(m,c,'cache_get_seconds')['mean'])*1e6:.3f}",f"{float(e(m,c,'dynamic_selection_seconds')['mean'])*1e6:.3f}",f"{float(e(m,c,'peak_allocated_bytes')['mean']):.3f}",f"{float(e(m,c,'peak_reserved_bytes')['mean']):.3f}"] for m in METHODS for c in CACHES])
    report+='\n峰值按请求开始重置,覆盖模型、开始时存活资料KV及本次请求,GiB=bytes/2³⁰,reserved含worker内分配器历史。每个指标的完整中位数/P90/均值CI和资料敏感性见efficiency_descriptive.csv。\n\n'
    report+=md(['动态−固定','总响应差/秒','对话CI/秒','TTFT差/毫秒','TTFT对话CI/毫秒'],[[CACHE_LABEL[c],f"{float(perf(('ape_fixed',c),('ape_dynamic',c),'total_seconds')['delta_comparison_minus_reference_s']):+.4f}",interval(perf(('ape_fixed',c),('ape_dynamic',c),'total_seconds'),'delta_ci_low_s','delta_ci_high_s',4),f"{float(perf(('ape_fixed',c),('ape_dynamic',c),'ttft_seconds')['delta_comparison_minus_reference_s'])*1000:+.3f}",f"[{float(perf(('ape_fixed',c),('ape_dynamic',c),'ttft_seconds')['delta_ci_low_s'])*1000:.3f},{float(perf(('ape_fixed',c),('ape_dynamic',c),'ttft_seconds')['delta_ci_high_s'])*1000:.3f}]"] for c in CACHES])
    report+='\n动态相对固定平均少2.320个输出token,冷总响应缩短0.0888秒、热缩短0.0831秒,两种总响应差区间均跨0。顺序与动态有296/500目标输出token相同,该组动态冷/热总响应仍分别平均多2.479/2.452秒。源码中APE query继续执行文档/其他两分支注意力和混合,资料循环逐块forward；单块占493/500的这批输入保留该执行路径。\n\n'
    report+='三方法worker按顺序执行,模型加载为'+ '、'.join(f"{float(next(r['model_load_seconds'] for r in workers if r['method']==m)):.3f}" for m in METHODS)+'秒,排除于请求时延。起止为'+ '；'.join(LABEL[m]+': '+bj(next(r['started_at'] for r in workers if r['method']==m))+' 至 '+bj(next(r['ended_at'] for r in workers if r['method']==m)) for m in METHODS)+'。完整metadata见model_load_workers.csv。\n\n'

    report+='## 7 输出长度、EOS与上限\n\n'
    report+=md(['方法（每种缓存500）','命中128/500','比例','EOS终止','EOS恰在上限'],[[LABEL[m],next(r['cap_hits'] for r in limits if r['method']==m and r['cache_mode']=='cold'),f"{float(next(r['cap_fraction'] for r in limits if r['method']==m and r['cache_mode']=='cold'))*100:.1f}%",next(r['eos_last_count'] for r in limits if r['method']==m and r['cache_mode']=='cold'),next(r['eos_at_cap_count'] for r in limits if r['method']==m and r['cache_mode']=='cold')] for m in METHODS])
    report+='\n冷热token相同,上表计数在两种缓存条件分别相同。全部3000条中2162条达到上限（72.07%）,840条以EOS结束,其中2条EOS恰在上限。每个方法的输出中位数和P90均为128 token。\n\n'
    report+=md(['方法/冷请求分组','目标数','子语料BLEU','总响应均值/秒','输出token均值','13a长度比'],[[LABEL[m]+(' 达上限' if flag=='True' else ' 未达上限'),r['n_targets'],f"{float(r['corpus_bleu']):.3f}",f"{float(r['mean_total_s']):.3f}",f"{float(r['mean_output_tokens']):.2f}",f"{float(r['system_reference_length_ratio']):.3f}"] for m in METHODS for flag in ['False','True'] for r in caps if r['method']==m and r['cache_mode']=='cold' and r['cap_hit']==flag])
    report+='\n按实际输出长度形成的子语料分别累计ngram计数。动态达到上限的350目标平均冷响应6.717秒、BLEU3.120；其余150目标为4.157秒、BLEU4.569。原生成、参考、结束条件和源行在generation_cap_records.csv保留。完整生成长度及相同/不同输出分组用于解释耗时,不调整本次生成或评分。\n\n![生成/参考长度与上限](../figures/output_length_and_caps.png)\n\n![逐目标长度与时延差](../figures/paired_length_latency.png)\n\n'

    report+='## 8 案例与原始语言问题\n\n案例按动态−固定的保存单目标默认corpus BLEU选择最大正差、最大负差和零差,并列按sample_id排序。该辅助分数用于选例；总体统计始终使用语料级BLEU。冷热答案相同,四条源行均保留于selected_cases.csv。\n\n'
    for i,r in enumerate(cases,1):
        report+=f"### 8.{i} {r['case_type']}：{r['sample_id']}\n\nlast_turn：{r['last_turn_utterance']}\n\n参考：{r['reference_answer']}\n\n固定T/S={r['fixed_T']}/{r['fixed_S']},动态={r['dynamic_T']}/{r['dynamic_S']}；单目标辅助BLEU {float(r['fixed_auxiliary_bleu']):.3f}→{float(r['dynamic_auxiliary_bleu']):.3f},差{float(r['auxiliary_delta_bleu']):+.3f}。输出token {r['fixed_output_tokens']}→{r['dynamic_output_tokens']}。原始记录行：固定冷/热{r['fixed_cold_line']}/{r['fixed_hot_line']},动态冷/热{r['dynamic_cold_line']}/{r['dynamic_hot_line']}。\n\n"
        if r['case_type']=='改善':report+='动态答案直接回应纽约驾驶许可问题,与参考共享较长的条件/限制表述；固定答案以对话复述展开并达到生成上限。两份原答案在案例CSV完整并列。\n\n'
        elif r['case_type']=='退化':report+='两份生成均回答dependent coverage免费,动态继续补充列表和扩展条件,与参考的ngram精度下降且回答变长。参考原文包含Nothing.s拼写,按原始目标保留。\n\n'
        else:report+='该例请求和两份输出为英文,参考为西语,两份单目标词面得分均为0。原始target第6轮的西语文本与冻结参考匹配,仅去首尾空白；连接正确。原始target、历史及来源span证据保存在sources/reference_language_evidence.json。\n\n'
    report+='对500参考按明确西语词及重音字符作定向候选检索并人工确认,得到这一条西语参考。主表保留全部500目标；移除该条的499目标敏感性为顺序3.367、固定3.258、动态3.426,动态−固定+0.168分,与主表+0.167接近。候选检索规则和实际文本在reference_language_candidates.csv,原始资料不改写。\n\n'

    report+='## 9 审核、补实验与复现\n\n'
    report+=md(['具体问题','证据和范围','本阶段处理','建议补充'],[[r['phenomenon'],r['evidence'],r['handling'],r['followup']] for r in findings])
    report+='\n本次计分与服务器六组语料分数、充分统计及配置签名一致。所有3000条均成功、attempt=1,无重复成功、空答案、缺参考或非法值。目标ID、完整资料、历史、模板和参考连接通过前置检查与本阶段字段连接。源证据、冻结规则和主样本范围保留,发现的角色/历史/语言条件单列处理。\n\n'
    report+='建议后续对同资料不同目标轮次做连续热测,增加独立N2及更大N资料,统一比较简洁回答模板与生成长度控制,保存逐token/结束事件并交错重复测时。语言修订和参考规范化另定义数据版本。以上均登记为补实验事项,本轮只完成本地CPU复算。\n\n'
    report+='## 10 结论\n\n在共同500目标上,动态BLEU3.418较固定3.251高0.167,主对话CI[-0.205,0.562],资料CI[-0.192,0.514]；相对顺序3.360高0.058。冷热答案和质量完全相同。动态的总体点估计小幅提高,子语料N2的7目标明显下降,该子组来源仅一篇资料。\n\n'
    report+='热GPU资料KV使平均TTFT降至0.040–0.065秒,冷/热TTFT均值比3.29–3.97；总响应平均节省0.120–0.149秒,均值比1.025–1.037。平均113–116个模型token的生成占据主要响应时间,70%–76.2%的上限命中率与长于参考的文本共同影响效率和BLEU精度。实际模板、资料/历史缓存区分、同题热复用、源簇相关性及单模型/单campaign设置随结果保留。\n\n'
    report+=f"实际分析提交{env['analysis_commit']},与参考提交相同。环境版本、独立评分引擎、seed20261116、5000次重采样及随机子序列在environment.json；数据字段、筛选、分母、公式和来源在provenance.csv。研究状态保持SUBMITTED,执行停止在第八阶段,后续综合报告待用户审阅规划后进行。\n"
    (OUT/'reports/DOC2DIAL_REPORT.md').write_text(report,encoding='utf-8')

    audit='# 第八阶段审核记录\n\n'
    audit+='已核对500目标/3000生成、六条件配对、原始参考ID、完整历史、资料KV命中与热重新生成、规则L/N及独立评分充分统计。主聚类为366对话,资料172作为敏感性；N2资料CI因仅一簇留空。\n\n'
    audit+=md(['项目','实际证据','产物'],[['参考/历史/模板','500前置检查全部PASS,28连续agent、7参考历史重复、交集2','preaudit_input_checks;target_inputs;history_condition_counts'],['热请求','1500 hot全部命中、build/rebuild0,输出token/答案/参考一致;APE query调用完整','cold_hot_pairs;hot_generation_evidence'],['语料BLEU','六组服务器分数及十项统计一致;独立向量化通过5fixture','original_summary_crosscheck;scorer_validation'],['参考语言','1条英语请求/西语参考来自原始agent目标,主表保留并单列499敏感性','reference_language_candidates;sources/reference_language_evidence.json']])
    audit+='\n'+md(['发现','影响','证据','处理及补充'],[[r['phenomenon'],r['impact'],r['evidence'],r['handling']+'；'+r['followup']] for r in findings])
    audit+='\n原始证据路径为restored/output/runs/main80-20261114T041503/attempts.jsonl、summary.json、snapshot/datasets/frozen-main80/doc2dial.json和同次snapshot/APE/main80/{freeze,methods,worker,metrics,rules}.py。原始对话/资料路径由restored/data_sources.json登记。源行保存在逐请求和案例CSV,热事件行保存在热生成证据CSV；新增产物仅在pc-analysis-output。原始数据、规则及研究状态保留。\n'
    (OUT/'reports/AUDIT.md').write_text(audit,encoding='utf-8')
    links=[('阶段报告','reports/DOC2DIAL_REPORT.md'),('审核记录','reports/AUDIT.md'),('语料BLEU主表','tables/main_quality.csv'),('资料聚类质量敏感性','tables/main_quality_document_sensitivity.csv'),('方法/冷热质量配对','tables/main_quality_paired.csv'),('质量配对的资料敏感性','tables/main_quality_paired_document_sensitivity.csv'),('500目标成员与簇','tables/membership.csv'),('完整历史及输入处理','tables/target_inputs.csv'),('3000原答案/参考/计时','tables/per_request_records.csv'),('每目标ngram充分统计','tables/bleu_sufficient_statistics.csv'),('1500冷热对','tables/cold_hot_pairs.csv'),('热重新生成及事件证据','tables/hot_generation_evidence.csv'),('效率均值/中位数/P90/CI','tables/main_efficiency.csv'),('缓存收益','tables/cache_effects.csv'),('全部配对加速比','tables/efficiency_paired.csv'),('实际L/N与映射','tables/input_distribution.csv'),('历史角色/参考重复数量','tables/history_condition_counts.csv'),('生成上限与EOS','tables/generation_limits.csv'),('完整原答案案例','tables/selected_cases.csv'),('语言问题原始证据','sources/reference_language_evidence.json'),('官方评分与模板来源','sources/official_scoring_and_template.md'),('关键数字来源及公式','tables/provenance.csv'),('字段含义','tables/field_dictionary.csv'),('重采样簇与子序列','tables/bootstrap_diagnostics.csv'),('环境与种子','environment.json')]
    index='# 实验第八阶段：Doc2Dial\n\n本阶段完成,研究状态SUBMITTED；执行停止于本阶段。\n\n'+''.join(f'- [{t}]({p})\n' for t,p in links)
    index+='\nfigures/含8组PNG/SVG,源数据为quality/quality_paired、efficiency_descriptive/efficiency_paired、generation_limits、membership及dynamic_fixed_per_target各CSV。指标BLEU为0–100原值；13a分词数和模型token分开记录。主分析按对话,资料作敏感性。\n\n'
    index+='复现依赖已交付的前置审核CSV及scripts/audit_stage1_scoring.py纯函数。评分helper通过runpy加载,main未执行。CPU不需要安装SacreBLEU；对照设置为服务器2.4.3签名,六语料统计和分数核对保存在original_summary_crosscheck.csv。\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n$env:PYTHONDONTWRITEBYTECODE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_doc2dial_stage8.py\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/plot_doc2dial_stage8.py\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/summarize_doc2dial_stage8.py\n```\n\n使用既有Python、NumPy、Matplotlib,无新依赖。seed20261116,5000次配对簇Bootstrap,每次重新累计语料计数。六条件共同保留；中位数/P90基于各条件500真实请求。缓存、临时文件和产物均位于本阶段目录。\n'
    index+='\n[交付检查结果](logs/validation.json)记录16项表格分母、六组服务器评分一致性、1500冷热配对、单资料子组空CI及图表/引用检查。[检查脚本](../../scripts/validate_doc2dial_stage8.py)可在以上分析和报告生成后运行：\n\n```powershell\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/validate_doc2dial_stage8.py\n```\n'
    (OUT/'README.md').write_text(index,encoding='utf-8')
    root=(BASE/'README.md').read_text(encoding='utf-8')
    root=root.replace('当前已完成实验第七阶段：Lost in the Middle位置实验。','当前已完成实验第八阶段：Doc2Dial多轮问答与KV复用。')
    if '[实验8：Doc2Dial]' not in root:root=root.replace('- [实验7：Lost in the Middle]','- [实验8：Doc2Dial](stages/08_doc2dial/README.md)\n- [实验7：Lost in the Middle]')
    root=root.replace('本轮已停在实验7；','本轮已停在实验8；')
    (BASE/'README.md').write_text(root,encoding='utf-8')
    print('Stage8 report, audit, provenance, main tables and material index written.')


if __name__=='__main__':main()
