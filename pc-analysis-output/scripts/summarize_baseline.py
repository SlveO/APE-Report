"""Readable material handoff for experiment 1; uses completed baseline CSVs only."""
from pathlib import Path
import csv,json,shutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/01_longbench_baseline'
METHODS=('sequential','pcw','ape_default')
LABEL={'sequential':'顺序编码','pcw':'PCW','ape_default':'默认APE'}
DISPLAY={'qasper':'Qasper','hotpotqa':'HotpotQA','musique':'MuSiQue','macro_equal_dataset':'等权宏平均'}
DS=('qasper','hotpotqa','musique')
def rows(name):return list(csv.DictReader((OUT/'tables'/name).open(encoding='utf-8-sig')))
def n(r,k):return float(r[k])
def md(header,data):return '| '+' | '.join(header)+' |\n|'+'|'.join(['---']*len(header))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in data)+'\n'
def main():
    quality=rows('baseline_quality.csv');pairs=rows('baseline_quality_paired.csv');eff=rows('baseline_efficiency.csv');perf=rows('baseline_efficiency_paired.csv');comp=rows('official_comparison.csv');cases=rows('baseline_cases.csv');ns=rows('baseline_quality_by_N.csv');caps=rows('baseline_generation_limits.csv')
    def q(ds,m):return next(r for r in quality if r['dataset']==ds and r['method']==m)
    def e(m,metric):return next(r for r in eff if r['dataset']=='pooled600_stratified_mean_CI' and r['method']==m and r['metric']==metric)
    def p(ds,a,b):return next(r for r in pairs if r['dataset']==ds and r['method_a']==a and r['method_b']==b)
    out=OUT/'reports/BASELINE_REPORT.md'
    detail=OUT/'reports/STATISTICS_DETAIL.md'
    if out.exists() and not detail.exists():shutil.copyfile(out,detail)
    text='# 实验第一阶段：三方法基线及官方对照分析\n\n'
    text+='## 1 我们完成了什么\n\n我们从服务器MAIN80逐题记录中提取LongBench的1,800次基线生成，覆盖Qasper、HotpotQA、MuSiQue各200题，共600题。我们复算了答案F1，完成相同题目的方法配对、置信区间、冷请求效率统计，并查阅APE论文公开成绩进行对照。前置审核单独保存在01_audit；本文按研究实验顺序作为第一阶段交付。\n\n'
    text+='基线统一使用Llama-3.1-8B-Instruct、BF16、batch=1、贪心生成；默认APE为T=0.9、S=0.9。三方法共用完整公开资料和问题，按最多7500 document tokens分块，实际N=1–3；全部请求为cold。Qasper输出上限128，另外两任务32。\n\n'
    text+='## 2 基线质量：顺序编码最高，默认APE与PCW接近\n\n下表为F1×100；每项任务分母均为200，宏平均对三个任务等权。\n\n'
    text+=md(['数据集','顺序编码','PCW','默认APE'],[[DISPLAY[ds],*[f"{n(q(ds,m),'f1_percent'):.2f}" for m in METHODS]] for ds in [*DS,'macro_equal_dataset']])
    a=p('macro_equal_dataset','sequential','ape_default');b=p('macro_equal_dataset','sequential','pcw');c=p('macro_equal_dataset','pcw','ape_default')
    text+=f"\n默认APE相对顺序编码低{abs(n(a,'delta_b_minus_a_pp')):.2f}个百分点；PCW低{abs(n(b,'delta_b_minus_a_pp')):.2f}个百分点。默认APE与PCW相差{n(c,'delta_b_minus_a_pp'):+.3f}个百分点，宏平均成绩接近。\n\n"
    text+=md(['数据集','APE−顺序（百分点）','95%配对区间','APE胜／平／负'],[[DISPLAY[ds],f"{n(p(ds,'sequential','ape_default'),'delta_b_minus_a_pp'):+.3f}",f"[{n(p(ds,'sequential','ape_default'),'ci_low_pp'):.4f}, {n(p(ds,'sequential','ape_default'),'ci_high_pp'):.4f}]",'/'.join(p(ds,'sequential','ape_default')[k] for k in ('win_b','tie','loss_b'))] for ds in [*DS,'macro_equal_dataset']])
    text+='\n宏平均差值的区间上界接近0；HotpotQA的区间整体为负。区间采用seed20261116、5000次配对聚类Bootstrap：在每个数据集内按原论文/完整证据组抽样，保留组内所有题目，再对三任务等权。三个数据集的簇数为142、200、194。完整的三组方法对比和胜平负见[配对统计](../tables/baseline_quality_paired.csv)。\n\n'
    text+='## 3 与官方论文对比：同名模型、各自公开输入设置\n\n我们选用[APE论文arXiv v1表2](https://arxiv.org/html/2502.05431v1#S5.T2)中Llama-3.1-8B-Instruct的结果。顺序编码选128K上下文行；PCW和APE选Contriever检索前20个4000-word块的行。下面统一重算Qasper、HotpotQA、MuSiQue三任务宏平均。\n\n'
    text+=md(['方法','本次Qasper','官方Qasper','本次HotpotQA','官方HotpotQA','本次MuSiQue','官方MuSiQue'],[[LABEL[m],*[v for ds in DS for v in (f"{n(next(r for r in comp if r['dataset']==ds and r['method']==m),'local_f1_percent'):.2f}",f"{n(next(r for r in comp if r['dataset']==ds and r['method']==m),'paper_f1_percent'):.2f}")]] for m in METHODS])
    text+='\n'
    text+=md(['方法','本次三任务宏平均','官方三任务宏平均','本次−官方（百分点）'],[[LABEL[m],f"{n(next(r for r in comp if r['dataset']=='macro_equal_dataset' and r['method']==m),'local_f1_percent'):.2f}",f"{n(next(r for r in comp if r['dataset']=='macro_equal_dataset' and r['method']==m),'paper_f1_percent'):.2f}",f"{n(next(r for r in comp if r['dataset']=='macro_equal_dataset' and r['method']==m),'local_minus_paper_pp'):+.2f}"] for m in METHODS])
    text+='\n官方APE相对官方PCW的三任务宏平均高4.37个百分点；本次两者接近。官方APE相对其128K顺序行的三任务宏平均低1.23个百分点，本次低2.11。论文原表Avg覆盖八个任务；此处按本次三个任务重新求均值，数据来源和公式保存在[官方对照CSV](../tables/official_comparison.csv)。\n\n'
    text+='两组设置的具体差异如下。\n\n'
    text+=md(['条件','本次基线','官方相关设置'],[
        ['输入资料','三方法共用完整公开文本','表2的PCW/APE为检索文本块；顺序为128K行'],
        ['分块单位','最多7500 tokenizer tokens，实际1–3块','表2检索块以4000 words计；表6以7500 tokens分段'],
        ['APE参数','所有题固定默认0.9/0.9','论文§4.4描述验证集贪心搜索T/S，逐任务终值未公开列出'],
        ['公开入口筛选','每数据集完整200题','观察提交pred_parallel9.py按公开length≥8000筛选'],
        ['速度条件','实际问答、cold、batch1、输出上限128/32','论文§5.4为H100/vLLM、2K–128K输入、问题/输出各256、batch1/4']])
    text+='\n输入处理同时改变证据、干扰文本和顺序；参数搜索改变APE的注意力调整强度。这些条件应与数值一起记录，随后在相应实验中逐项研究。官方公开入口的length≥8000筛选若用于本次600题，只保留Qasper5、HotpotQA143、MuSiQue190题；筛选子集的描述性结果另表保存，主基线保留全部600题。论文表格对应逐题ID和分母尚缺公开记录。\n\n'
    text+='另据[论文表6](https://arxiv.org/html/2502.05431v1#S6.T6)，完整长文本分段实验采用Llama-3-8B，顺序/APE的Qasper为32.83/39.14、HotpotQA44.24/50.11、MuSiQue20.47/25.79；模型和顺序窗口条件与本次不同，作为补充参考。官方代码观察提交52513bcb的入口还存在query调用与接口参数未配套的情况，原文文件和行号保存在sources/及[设置对照](../tables/official_setting_comparison.csv)。本次执行按MAIN80快照实际传入T/S。\n\n'
    text+='## 4 效率结果：PCW较快，默认APE首token略快但总响应更慢\n\n以下每个方法均为相同600题、cold请求。\n\n'
    text+=md(['方法','平均TTFT（秒）','平均总响应（秒）','总响应中位数（秒）','总响应P90（秒）','平均输出token'],[[LABEL[m],f"{n(e(m,'ttft_seconds'),'mean'):.3f}",f"{n(e(m,'total_seconds'),'mean'):.3f}",f"{n(e(m,'total_seconds'),'median'):.3f}",f"{n(e(m,'total_seconds'),'p90'):.3f}",f"{n(e(m,'output_tokens'),'mean'):.3f}"] for m in METHODS])
    seq=n(e('sequential','total_seconds'),'mean');ape=n(e('ape_default','total_seconds'),'mean');pcw=n(e('pcw','total_seconds'),'mean')
    text+=f"\nPCW平均总响应缩短{(1-pcw/seq)*100:.2f}%，均值耗时比为{seq/pcw:.3f}倍；默认APE平均总响应增加{(ape/seq-1)*100:.2f}%，均值耗时比为{seq/ape:.3f}倍。配对的APE−顺序总响应差为0.166秒，95%区间[0.138,0.193]秒。\n\n"
    text+='PCW的平均输出也较短（8.905对11.075 token），总响应改善包含生成长度变化。默认APE与顺序输出长度相近（11.033对11.075），而TTFT之后的平均耗时从约0.304增至0.526秒，差异集中于后续生成路径。源码中APE每步执行上下文/其他分支的注意力调整，本阶段将它作为与该耗时表现对应的实现条件。\n\n'
    text+=f"达到生成上限的记录为{sum(int(r['cap_hits']) for r in caps)}/1800，其中{sum(int(r['eos_at_cap']) for r in caps)}次EOS恰在上限。各任务/方法的上限与EOS计数见[生成长度表](../tables/baseline_generation_limits.csv)。\n\n"
    text+='TTFT含预处理、选参分支及资料KV构建；总响应再含完整生成和文本解码。模型加载独立登记。显存峰值包含模型及存活缓存，单位GiB；完整均值、中位数、P90和均值区间见[效率表](../tables/baseline_efficiency.csv)，逐题加速比与均值耗时比见[配对效率表](../tables/baseline_efficiency_paired.csv)。官方4.5倍加速来自其长上下文速度配置，本次采用上表实际问答冷请求口径。\n\n'
    text+='## 5 退化主要出现在哪里\n\nHotpotQA按实际N分层：N=1的25题，顺序与默认APE均为65.15；N=2的89题从56.67降至46.58，下降10.09个百分点；N=3的86题从57.37降至56.14。分层保留各自样本数，详见[实际N分层](../tables/baseline_quality_by_N.csv)。\n\n'
    text+='我们按统一规则抽取改善、退化及持平案例。一个HotpotQA退化例的问题询问Roud Folk Song Index，参考为821，顺序回答821，默认APE回答7734，F1从1降至0，输入为8851 tokens、2块。原始答案、问题、参考与两方法源行号见[案例表](../tables/baseline_cases.csv)。这为后续检查分块与注意力参数的影响提供了具体样本。\n\n'
    text+='## 6 本阶段交付与停止点\n\n本次得到三条基线结论：顺序编码质量最高；PCW以约2.09个百分点宏平均F1下降换取较短平均总响应；默认APE与PCW质量接近，且当前实现下总响应慢于顺序。官方对照补齐了公开数值、设置差异和代码入口条件。\n\n基线使用原始完整资料。开发裁剪的证据问题留在前置审核及后续参数阶段处理；正式五方法比较使用独立510题。研究状态保持SUBMITTED，当前停在实验第一阶段，下一阶段按用户安排推进。\n'
    out.write_text(text,encoding='utf-8')
    status={'experiment_stage':1,'name':'LongBench full600 three-method baseline','delivery':'COMPLETED','research_status':'SUBMITTED','records':1800,'unique_ids':600,'official_comparison':'paper v1 Table2 Llama3.1; Table6 supplementary','next_stage_executed':False}
    (OUT/'STAGE_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__':main()
