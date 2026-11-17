"""Experiment 1 only: full600 LongBench baseline, paired statistics, official comparison."""
from pathlib import Path
import collections, csv, itertools, json, os, re, string, sys
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/01_longbench_baseline'
RUN=ROOT/'restored/output/runs/main80-20261114T041503'
for folder in ('tables','reports','figures','logs','cache','tmp'):
    (OUT/folder).mkdir(parents=True,exist_ok=True)
for key in ('MPLCONFIGDIR','XDG_CACHE_HOME','TEMP','TMP'):
    os.environ[key]=str(OUT/('cache' if 'CACHE' in key or key=='MPLCONFIGDIR' else 'tmp'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEED=20261116
BOOT=5000
DS=('qasper','hotpotqa','musique')
METHODS=('sequential','pcw','ape_default')
LABEL={'sequential':'顺序编码','pcw':'PCW','ape_default':'默认APE'}
PAPER='https://arxiv.org/html/2502.05431v1'
PAPER2=PAPER+'#S5.T2'
PAPER6=PAPER+'#S6.T6'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvout(name, rows):
    rows=list(rows)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def norm(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
def f1(s,a):
    x,y=norm(s).split(),norm(a).split();z=sum((collections.Counter(x)&collections.Counter(y)).values())
    return 2*z/(len(x)+len(y)) if z else 0.
def ci(a):return np.quantile(a,[.025,.975],axis=0)
def bootstrap(values,groups,rng):
    unique=sorted(set(groups));lookup={g:i for i,g in enumerate(unique)}
    sums=np.zeros((len(unique),values.shape[1]));sizes=np.zeros(len(unique))
    for v,g in zip(values,groups):sums[lookup[g]]+=v;sizes[lookup[g]]+=1
    result=[]
    for k in range(0,BOOT,100):
        idx=rng.integers(0,len(unique),size=(min(100,BOOT-k),len(unique)))
        result.append(sums[idx].sum(axis=1)/sizes[idx].sum(axis=1)[:,None])
    return np.concatenate(result)
def table(rows,columns):
    def val(r,k):
        x=r.get(k,'');return (f'{x:.3f}' if isinstance(x,float) else str(x)).replace('|','/')
    return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(val(r,k) for k in columns)+' |' for r in rows)+'\n'

rows=[]
with (RUN/'attempts.jsonl').open(encoding='utf-8') as f:
    for line,text in enumerate(f,1):
        r=json.loads(text)
        if r['key']['stage']=='baseline':r['_line']=line;rows.append(r)
assert len(rows)==1800 and all(r['status']=='success' and r['cache_mode']=='cold' for r in rows)
frozen=read(RUN/'snapshot/datasets/frozen-main80/longbench.json')
source={r['sample_id']:r for r in frozen}
eos=set(read(RUN/'snapshot/models/Llama-3.1-8B-Instruct/generation_config.json')['eos_token_id'])
for r in rows:
    assert r['reference_answers']==source[r['sample_id']]['answers']
    r['_f1']=max(f1(r['answer'],a) for a in r['reference_answers'])
assert max(abs(r['_f1']-r['score']) for r in rows)<1e-12
FIELDS={'ttft_seconds':'s','total_seconds':'s','cache_build_seconds':'s','cache_get_seconds':'s','dynamic_selection_seconds':'s','output_tokens':'token_including_EOS','peak_allocated_bytes':'GiB','peak_reserved_bytes':'GiB'}
metrics=['f1',*FIELDS]
columns=[(m,f) for m in METHODS for f in metrics]
ratio_columns=[(a,b,f) for a,b in itertools.combinations(METHODS,2) for f in ('ttft_seconds','total_seconds')]
quality=[];paired=[];efficiency=[];perf=[];individual=[];by_ds={};arrays={};boots={};all_groups={};caps=[];n_strata=[];cases=[]
rng=np.random.default_rng(SEED)
for ds in DS:
    by={m:{r['sample_id']:r for r in rows if r['dataset']==ds and r['method']==m} for m in METHODS}
    ids=sorted(set.intersection(*(set(v) for v in by.values())))
    assert len(ids)==200 and all(len(v)==200 for v in by.values())
    groups=[json.dumps(by[METHODS[0]][s]['group'],sort_keys=True) for s in ids]
    data=[]
    for sid in ids:
        vals=[]
        for m,f in columns:
            r=by[m][sid];x=r['_f1'] if f=='f1' else r[f]
            vals.append(x/2**30 if FIELDS.get(f)=='GiB' else x)
        vals.extend(by[a][sid][f]/by[b][sid][f] for a,b,f in ratio_columns)
        data.append(vals)
    a=np.array(data,float);b=bootstrap(a,groups,rng)
    arrays[ds]=a;boots[ds]=b;by_ds[ds]=(ids,by);all_groups[ds]=groups
    for sid in ids:
        for m in METHODS:
            r=by[m][sid]
            individual.append({'dataset':ds,'sample_id':sid,'method':m,'source_line':r['_line'],'f1_0_1':r['_f1'],'saved_f1_0_1':r['score'],'L_tokens':r['L'],'N_blocks':r['N'],'T':r['T'],'S':r['S'],'output_tokens':r['output_tokens'],'ttft_seconds':r['ttft_seconds'],'total_seconds':r['total_seconds'],'group':r['group'],'source':'attempts.jsonl; snapshot/longbench.json:answers'})
    for n in sorted(set(by[METHODS[0]][sid]['N'] for sid in ids)):
        nids=[sid for sid in ids if by[METHODS[0]][sid]['N']==n]
        for m in METHODS:
            n_strata.append({'dataset':ds,'N_blocks':n,'method':m,'n_unique_ids':len(nids),'mean_f1_percent':np.mean([by[m][s]['_f1'] for s in nids])*100,'unit':'F1 x100','scope':'full600 baseline within dataset and actual N'})
    for kind,predicate,descending in [('改善',lambda x:x>1e-12,True),('退化',lambda x:x< -1e-12,False),('持平',lambda x:abs(x)<=1e-12,False)]:
        eligible=[(by['ape_default'][sid]['_f1']-by['sequential'][sid]['_f1'],sid) for sid in ids]
        eligible=sorted([x for x in eligible if predicate(x[0])],key=lambda x:((-x[0] if descending else x[0]),x[1]))
        if eligible:
            delta,sid=eligible[0];sr,ar=by['sequential'][sid],by['ape_default'][sid]
            cases.append({'dataset':ds,'type':kind,'sample_id':sid,'selection_rule':'max positive/negative APE-sequential; tie smallest ID','delta_f1_0_1':delta,'question':source[sid]['question'],'reference_answers':ar['reference_answers'],'sequential_answer':sr['answer'],'ape_answer':ar['answer'],'N_blocks':ar['N'],'L_tokens':ar['L'],'ape_T':ar['T'],'ape_S':ar['S'],'sequential_line':sr['_line'],'ape_line':ar['_line']})
    for m in METHODS:
        rr=[by[m][sid] for sid in ids];limit=source[ids[0]]['max_new_tokens']
        caps.append({'dataset':ds,'method':m,'n':len(rr),'limit':limit,'cap_hits':sum(r['output_tokens']==limit for r in rr),'eos_at_cap':sum(r['output_tokens']==limit and r['output_token_ids'][-1] in eos for r in rr),'eos_terminated':sum(r['output_token_ids'][-1] in eos for r in rr),'output_tokens_mean':np.mean([r['output_tokens'] for r in rr]),'unit':'requests; output tokens include EOS'})

all_a=np.concatenate(list(arrays.values()))
macro_b=np.mean(list(boots.values()),axis=0)
for scope in [*DS,'macro_equal_dataset']:
    a=all_a if scope=='macro_equal_dataset' else arrays[scope]
    b=macro_b if scope=='macro_equal_dataset' else boots[scope]
    nc=sum(len(set(v)) for v in all_groups.values()) if scope=='macro_equal_dataset' else len(set(all_groups[scope]))
    for m in METHODS:
        j=columns.index((m,'f1'));lo,hi=ci(b[:,j])
        quality.append({'dataset':scope,'method':m,'n':len(a),'clusters':nc,'f1_0_1':a[:,j].mean(),'f1_percent':a[:,j].mean()*100,'ci_low_percent':lo*100,'ci_high_percent':hi*100,'unit':'F1 x100','seed':SEED,'replicates':BOOT,'scope':'full600; each dataset200; macro equal datasets'})
        for f,unit in FIELDS.items():
            j=columns.index((m,f));lo,hi=ci(b[:,j])
            efficiency.append({'dataset':'pooled600_stratified_mean_CI' if scope=='macro_equal_dataset' else scope,'method':m,'metric':f,'n':len(a),'clusters':nc,'mean':a[:,j].mean(),'median':np.median(a[:,j]),'p90':np.quantile(a[:,j],.9),'ci_mean_low':lo,'ci_mean_high':hi,'unit':unit,'cache':'cold miss; no diagnostics','seed':SEED,'replicates':BOOT})
    for ma,mb in itertools.combinations(METHODS,2):
        ia,ib=columns.index((ma,'f1')),columns.index((mb,'f1'));d=a[:,ib]-a[:,ia];lo,hi=ci(b[:,ib]-b[:,ia])
        paired.append({'dataset':scope,'method_a':ma,'method_b':mb,'n':len(a),'delta_b_minus_a_pp':d.mean()*100,'ci_low_pp':lo*100,'ci_high_pp':hi*100,'win_b':int((d>1e-12).sum()),'tie':int((abs(d)<=1e-12).sum()),'loss_b':int((d< -1e-12).sum()),'unit':'F1 percentage points','bootstrap':'paired paper/evidence-group within dataset; macro equal datasets','seed':SEED,'replicates':BOOT})
        for f in ('ttft_seconds','total_seconds'):
            ia,ib=columns.index((ma,f)),columns.index((mb,f));ir=len(columns)+ratio_columns.index((ma,mb,f));lo,hi=ci(b[:,ib]-b[:,ia]);rl,rh=ci(b[:,ir]);ml,mh=ci(b[:,ia]/b[:,ib])
            perf.append({'dataset':'pooled600_stratified_CI' if scope=='macro_equal_dataset' else scope,'method_a':ma,'method_b':mb,'metric':f,'n':len(a),'mean_a_s':a[:,ia].mean(),'mean_b_s':a[:,ib].mean(),'delta_b_minus_a_s':(a[:,ib]-a[:,ia]).mean(),'delta_ci_low_s':lo,'delta_ci_high_s':hi,'paired_speedup_mean_a_over_b':a[:,ir].mean(),'paired_speedup_median_a_over_b':np.median(a[:,ir]),'speedup_mean_ci_low':rl,'speedup_mean_ci_high':rh,'ratio_of_means_a_over_b':a[:,ia].mean()/a[:,ib].mean(),'ratio_of_means_ci_low':ml,'ratio_of_means_ci_high':mh,'unit':'s; dimensionless ratios (>1 B faster)','seed':SEED,'replicates':BOOT})

csvout('baseline_quality.csv',quality);csvout('baseline_quality_paired.csv',paired)
csvout('baseline_efficiency.csv',efficiency);csvout('baseline_efficiency_paired.csv',perf)
csvout('baseline_per_sample.csv',individual);csvout('baseline_generation_limits.csv',caps)
csvout('baseline_quality_by_N.csv',n_strata);csvout('baseline_cases.csv',cases)
# Paper values are transcribed from arXiv v1 Tables 2/6; all macro means below use exactly these three tasks.
official=[]
def add_official(tableid,model,method,setting,values):
    for ds,score in zip(DS,values):official.append({'table':tableid,'model':model,'method':method,'setting':setting,'dataset':ds,'f1_percent':score,'unit':'F1 x100','paper_task_count_in_original_table':8 if tableid=='2' else 11,'paper_sample_n':'not reported in table','url':PAPER2 if tableid=='2' else PAPER6})
add_official('2','Llama-3.1-8B-Instruct','sequential_128K','full context;128K model',(47.20,53.46,28.35))
add_official('2','Llama-3.1-8B-Instruct','sequential_model_row','paper model-name row; window not explicitly labeled in Table2',(46.81,43.97,22.18))
add_official('2','Llama-3.1-8B-Instruct','sequential_RAG','Contriever top20 chunks x200 words',(42.33,49.97,30.62))
add_official('2','Llama-3.1-8B-Instruct','pcw','Contriever top20 chunks x4000 words',(41.52,49.47,21.23))
add_official('2','Llama-3.1-8B-Instruct','ape','Contriever top20 chunks x4000 words',(43.03,55.41,26.88))
add_official('6','Llama-3-8B-Instruct','sequential','long-context task; limited sequential window',(32.83,44.24,20.47))
add_official('6','Llama-3-8B-Instruct','ape','full long text; segments <7500 tokens',(39.14,50.11,25.79))
for key,group in itertools.groupby(list(official),key=lambda r:(r['table'],r['model'],r['method'],r['setting'])):
    group=list(group);r=dict(group[0]);r.update(dataset='macro_three_tasks',f1_percent=np.mean([x['f1_percent'] for x in group]));official.append(r)
csvout('official_reference.csv',official)
comparison=[]
for m,ref in [('sequential','sequential_128K'),('pcw','pcw'),('ape_default','ape')]:
    for ds in [*DS,'macro_equal_dataset']:
        q=next(q for q in quality if q['dataset']==ds and q['method']==m)
        o=next(o for o in official if o['table']=='2' and o['method']==ref and o['dataset']==('macro_three_tasks' if ds=='macro_equal_dataset' else ds))
        comparison.append({'dataset':ds,'method':m,'local_n':q['n'],'local_f1_percent':q['f1_percent'],'paper_f1_percent':o['f1_percent'],'local_minus_paper_pp':q['f1_percent']-o['f1_percent'],'local_input':'full public document; natural blocks <=7500 tokens; cold','paper_input':o['setting'],'paper_model':o['model'],'paper_table':2,'unit':'F1 x100; descriptive difference in percentage points','paired_comparison':False,'url':PAPER2})
csvout('official_comparison.csv',comparison)
settings=[
    ('model','Llama-3.1-8B-Instruct (ModelScope mirror master)','Table2 Llama-3.1; Table6 Llama-3','主表选Table2同名模型；Table6补充输入方式对照'),
    ('inputs','full public context; no retrieval; all200 IDs/dataset','Table2 PCW/APE: Contriever top20 x4000 words','论文输入为检索文本；本次为完整文档'),
    ('segmentation','<=7500 tokenizer tokens/block;actual N1-3','Table2 4000-word retrieval chunk; Table6 <7500-token segments','words与tokens分开标注；本次N分布另表'),
    ('T/S','default0.9/0.9 for every baseline sample','paper section4.4 validation search0.1..1.0 step0.1; uses S*T','官方README示例默认0.9/0.9；论文逐任务最终值未列出'),
    ('sample selection','all600 public IDs','pred_parallel9.py:99 skips public length<8000; paper table denominator unspecified','入口代码筛选事实与论文样本分母分别登记'),
    ('official entry','main80 snapshot passes T/S explicitly; three datasets fixed','pred_parallel9.py:183 passes only model; query definition requires model,T,S','按实际snapshot记录本次设置；官方入口需接口配套'),
    ('efficiency','cold requests; batch1; output128/32; sequential block forwards; TF4.44.2','paper section5.4 H100/vLLM; batch1/4; query256/generation256;contexts2K-128K','并列呈现测量设置；本次给600题配对加速比'),
    ('hardware','physical GPU0; RTX4090 48GB is plan registration;telemetry pending','paper efficiency H100','本次设备实际型号与驱动采集待补'),
]
csvout('official_setting_comparison.csv',[dict(zip(('item','local','official','interpretation'),s),source=PAPER) for s in settings])
filter_rows=[]
for ds in DS:
    ids,by=by_ds[ds];eligible=[sid for sid in ids if source[sid].get('original_length_field',0)>=8000]
    for m in METHODS:
        filter_rows.append({'dataset':ds,'method':m,'all_ids':200,'public_length_ge8000_ids':len(eligible),'mean_f1_percent_in_subset':np.mean([by[m][sid]['_f1'] for sid in eligible])*100 if eligible else None,'unit':'IDs; F1 x100','scope':'descriptive sensitivity to observed upstream entry predicate; full600 baseline retained'})
csvout('official_code_filter_sensitivity.csv',filter_rows)
loads=[];active=None
for line,text in enumerate((RUN/'events.jsonl').open(encoding='utf-8'),1):
    e=json.loads(text)
    if e.get('event')=='worker_started':active=e
    if e.get('event')=='model_loaded' and active and active.get('stage')=='baseline':loads.append({'method':e['method'],'load_seconds':e['seconds'],'event_line':line,'time_utc':e['at'],'included_in_request_timing':False})
csvout('baseline_model_load.csv',loads)
# Figures are authored from the same result objects written to CSV.
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(4)
for j,m in enumerate(METHODS):
    vals=[next(q for q in quality if q['dataset']==ds and q['method']==m) for ds in [*DS,'macro_equal_dataset']]
    ax.bar(x+(j-1)*.24,[v['f1_percent'] for v in vals],.24,label=LABEL[m])
ax.set_xticks(x,['Qasper','HotpotQA','MuSiQue','三任务等权平均']);ax.set_ylabel('F1 ×100');ax.set_title('实验1：完整600题三方法基线');ax.legend();save(fig,'baseline_quality')
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,f,title in zip(axs,('ttft_seconds','total_seconds'),('首token时间','总响应时间')):
    vals=[next(e for e in efficiency if e['dataset']=='pooled600_stratified_mean_CI' and e['method']==m and e['metric']==f)['mean'] for m in METHODS]
    ax.bar([LABEL[m] for m in METHODS],vals,color=['#4c78a8','#72b7b2','#f58518']);ax.set_ylabel('秒');ax.set_title(title+'（600题均值）')
save(fig,'baseline_latency')
fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,ds in zip(axs,DS):
    rr=[r for r in comparison if r['dataset']==ds];xx=np.arange(3)
    ax.bar(xx-.17,[r['local_f1_percent'] for r in rr],.34,label='本次完整文档')
    ax.bar(xx+.17,[r['paper_f1_percent'] for r in rr],.34,label='官方表2（各自输入设置）')
    ax.set_xticks(xx,[LABEL[m] for m in METHODS]);ax.set_title(ds);ax.set_ylabel('F1 ×100')
axs[0].legend(fontsize=8);save(fig,'official_comparison')

qmacro=[q for q in quality if q['dataset']=='macro_equal_dataset']
pmacro=[p for p in paired if p['dataset']=='macro_equal_dataset']
emacro=[e for e in efficiency if e['dataset']=='pooled600_stratified_mean_CI' and e['metric'] in ('ttft_seconds','total_seconds','output_tokens')]
fmacro=[p for p in perf if p['dataset']=='pooled600_stratified_CI']
text='# 实验第一阶段：LongBench三方法基线与官方结果对照\n\n'
text+='## 1 本阶段做了什么\n\n我们从baseline阶段1800条记录复算完整600题（Qasper、HotpotQA、MuSiQue各200题）的顺序编码、PCW和默认APE。三方法逐题使用相同原始完整资料、问题和参考，均为冷请求，默认APE T/S=0.9/0.9。F1保留0–1原值，本文乘100展示。前置审核另存于01_audit；本阶段直接交付基线质量、效率及官方对照。参数探索等后续实验尚待逐阶段开展。\n\n'
text+='## 2 本次质量结果\n\n'
display=[]
for ds in [*DS,'macro_equal_dataset']:
    r={'数据集':ds,'题数':600 if ds=='macro_equal_dataset' else 200}
    for m in METHODS:r[LABEL[m]]=next(q['f1_percent'] for q in quality if q['dataset']==ds and q['method']==m)
    display.append(r)
text+=table(display,['数据集','题数',*LABEL.values()])
text+='\n默认APE相对顺序编码的宏平均下降2.106个百分点；PCW下降2.094个百分点；默认APE与PCW相差−0.012个百分点。HotpotQA的APE下降5.021个百分点，是宏平均下降的主要来源。Qasper下降0.217、MuSiQue下降1.081个百分点。\n\n'
text+=table(paired,['dataset','method_a','method_b','delta_b_minus_a_pp','ci_low_pp','ci_high_pp','win_b','tie','loss_b'])
text+='\n差值统一B−A，胜/平/负按同一题的F1比较。区间使用seed20261116、5000次配对证据组Bootstrap；各数据集组数为142/200/194，宏平均对三个数据集等权。区间描述样本/证据组构成变化。宏平均APE−顺序的区间上界非常接近0，报告同时呈现差值、完整区间与逐题胜平负。\n\n'
text+='## 3 与官方论文的数值对照\n\n官方来源固定为[APE论文arXiv v1表2](https://arxiv.org/html/2502.05431v1#S5.T2)。主表选同名Llama-3.1-8B模型：顺序取128K完整上下文行，PCW和APE取Contriever检索20个4000-word块的行。下表差值为本次−论文表值；两组使用各自公布的输入设置。\n\n'
text+=table(comparison,['dataset','method','local_f1_percent','paper_f1_percent','local_minus_paper_pp'])
text+='\n官方这三任务重算宏平均为顺序43.003、PCW37.407、APE41.773；本次分别高2.428、5.931、1.552个百分点。官方APE相对PCW在这三任务的宏平均高4.367个百分点，本次两者相近；官方APE相对128K顺序的三任务宏平均低1.230个百分点，本次低2.106。论文表内原Avg覆盖八个任务，这里统一重算三个任务平均。\n\n'
text+='差异解释先从输入组织展开：本次三方法共用完整文本，官方PCW/APE采用检索结果；本次全部实际N为1–3。前者保留全部原文和干扰信息，后者改变证据与背景的组成。其次，本次固定默认0.9/0.9，论文按验证集搜索参数，逐任务最终值未列出。本阶段先登记这些实际条件，参数与输入机制的进一步分析进入对应后续阶段。\n\n'
text+='[论文表6](https://arxiv.org/html/2502.05431v1#S6.T6)另报告完整长文本分段，但采用Llama-3-8B；三个任务的顺序/APE分别为Qasper32.83/39.14、HotpotQA44.24/50.11、MuSiQue20.47/25.79。模型和顺序上下文窗口与本次不同，作为补充来源列入official_reference.csv。\n\n'
text+='官方复现入口在观察提交52513bcb的pred_parallel9.py:99按公开length字段过滤<8000样本，:121分7500-token块，:183的query调用仅传model；同期定义要求model/temperature/scale。该入口默认任务和本次三任务也不同。代码筛选的本次子集范围另见official_code_filter_sensitivity.csv；主基线保留全部600题。论文表2/6的逐题ID和精确分母尚缺公开对应记录。下载来源与原文件保存在sources/。\n\n'
text+='## 4 本次冷请求效率\n\n以下总体均值、中位数/P90在600题请求上计算；均值CI在数据集内按证据组重采样后等权聚合。\n\n'
text+=table(emacro,['method','metric','n','mean','median','p90','ci_mean_low','ci_mean_high','unit'])
text+='\n逐题加速比=A耗时/B耗时，大于1表示B更快；均值比另列。\n\n'
text+=table(fmacro,['method_a','method_b','metric','ratio_of_means_a_over_b','paired_speedup_mean_a_over_b','paired_speedup_median_a_over_b','delta_b_minus_a_s','delta_ci_low_s','delta_ci_high_s'])
text+='\nTTFT计入预处理、参数分支和资料KV构建，总响应继续计入生成及文本解码；模型加载另表。峰值显存含模型与存活缓存，以GiB=bytes/2^30展示。本次PCW/APE在一张GPU逐块forward后合并KV。各数据集输出长度、EOS、上限命中见baseline_generation_limits.csv，实际N分层见baseline_quality_by_N.csv。\n\n'
text+='官方速度实验使用H100、vLLM、2K–128K上下文、固定256-token问题和生成、batch1/4；本次为实际问答、batch1、128/32输出上限及冷资料编码。论文4.5倍取自该长上下文速度实验；本次速度结论使用上面的600题实测配对值。硬件执行编号为GPU0，方案登记RTX4090 48GB，设备实际遥测缺项沿用前置审核记录。\n\n'
text+='## 5 案例、结论与阶段停止点\n\n我们按默认APE−顺序的最大正差、最大负差及零差最小ID选取每数据集案例，原答案、参考、配置和源行号保存在baseline_cases.csv。质量和时延变化结合输出长度及实际N讨论。\n\n'
text+='本阶段得到的基线是：顺序编码宏平均F1最高；默认APE与PCW质量接近，默认APE在HotpotQA退化较大。官方对照把同名模型、输入方式、参数与速度设置分别列示；本次方法比较使用同600题配对。600题基线包括探索/验证/测试分割，下一正式五方法比较采用独立510题范围。\n\n本阶段至此停止，研究状态保留SUBMITTED。前置审核的开发裁剪问题记录在调参材料中，本阶段输入为原始完整资料。后续各实验由用户逐阶段安排。\n'
(OUT/'reports/STATISTICS_DETAIL.md').write_text(text,encoding='utf-8')
env={'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'seed':SEED,'replicates':BOOT,'scope':'baseline only,1800 records/full600 IDs','metric':'English LongBench maximum reference token F1','reference_commit':'227116e4a004153272640cfe31087a033861b8e9','research_status':'SUBMITTED','official_paper':PAPER,'later_experiments_executed':False}
(OUT/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
result={'experiment':1,'name':'LongBench full600 three-method baseline','status':'completed','research_status':'SUBMITTED','quality_macro':qmacro,'paired_macro':pmacro,'efficiency_macro':emacro,'efficiency_paired_macro':fmacro,'official_comparison':comparison,'records':1800,'unique_ids':600}
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
index='# 实验第一阶段：LongBench完整600题三方法基线\n\n- [结果与官方对比分析](reports/BASELINE_REPORT.md)\n- [质量基线CSV](tables/baseline_quality.csv)\n- [方法间配对差异及95%CI](tables/baseline_quality_paired.csv)\n- [官方表值与本次差值](tables/official_comparison.csv)\n- [官方全部参考行及设置](tables/official_reference.csv)\n- [设置差异](tables/official_setting_comparison.csv)\n- [效率均值/中位/P90及CI](tables/baseline_efficiency.csv)\n- [配对耗时与加速比](tables/baseline_efficiency_paired.csv)\n- [逐题可追溯结果](tables/baseline_per_sample.csv)\n- [原答案案例](tables/baseline_cases.csv)\n- [质量图](figures/baseline_quality.png)\n- [冷请求时延图](figures/baseline_latency.png)\n- [官方对照图](figures/official_comparison.png)\n\n图表均另存SVG，官方代码原文及获取URL保存在sources/；env、种子和范围见environment.json。\n\n复现（在仓库根目录）：\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_baseline.py\n```\n\n使用已有NumPy1.26.4和Matplotlib3.9.2；原始证据只读，所有新增文件均在pc-analysis-output/。前置审核另存01_audit，本目录按实验顺序编号为第一阶段；后续实验未在本脚本执行。研究状态SUBMITTED。\n'
(OUT/'README.md').write_text(index,encoding='utf-8')
(BASE/'README.md').write_text('# APE MAIN80 分实验分析\n\n当前已完成实验第一阶段：LongBench完整600题三方法基线及官方对照。研究状态SUBMITTED。\n\n- [实验1结果与材料](stages/01_longbench_baseline/README.md)\n- [前置数据审核](stages/01_audit/README.md)\n\n前置审核目录01_audit的数字沿用早期执行顺序；实验阶段按研究规划编号，第一阶段为三方法基线。后续参数探索等实验待逐阶段安排。\n\n实际提交227116e4a004153272640cfe31087a033861b8e9。依赖、种子、命令和证据链接见各阶段索引。既有初算与获取日志保留。\n',encoding='utf-8')
print(json.dumps({'records':1800,'unique_ids':600,'quality_macro':qmacro,'efficiency_macro':emacro},ensure_ascii=False,indent=2))
import runpy
runpy.run_path(str(BASE/'scripts/summarize_baseline.py'),run_name='__main__')
