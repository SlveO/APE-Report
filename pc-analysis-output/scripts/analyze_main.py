"""Independent CPU audit, paired LongBench F1, clustered CIs and latency summaries.
Run: python -B pc-analysis-output/scripts/analyze_main.py
"""
from pathlib import Path
import collections, csv, itertools, json, math, platform, re, string, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'restored'
OUT=ROOT/'pc-analysis-output'
RUN=SRC/'output/runs/main80-20261114T041503'
SEED=20261116
BOOT=5000
rng=np.random.default_rng(SEED)
for d in ('tables','reports','figures'): (OUT/d).mkdir(parents=True,exist_ok=True)
def dump(name,rows):
    rows=list(rows)
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows: w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def norm(x):
    x=str(x).lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',x).split())
def f1(pred,ref):
    p=norm(pred).split();r=norm(ref).split();n=sum((collections.Counter(p)&collections.Counter(r)).values())
    return 2*n/(len(p)+len(r)) if n else 0.
def cluster_key(r): return json.dumps(r.get('group',r['sample_id']),sort_keys=True)
def boot_means(a,groups):
    a=np.asarray(a,float)
    if a.ndim==1: a=a[:,None]
    unique=sorted(set(groups));lookup={g:i for i,g in enumerate(unique)}
    sums=np.zeros((len(unique),a.shape[1]));sizes=np.zeros(len(unique))
    for vals,g in zip(a,groups): sums[lookup[g]]+=vals;sizes[lookup[g]]+=1
    result=[]
    for start in range(0,BOOT,100):
        ix=rng.integers(0,len(unique),size=(min(100,BOOT-start),len(unique)))
        result.append(sums[ix].sum(axis=1)/sizes[ix].sum(axis=1)[:,None])
    return np.concatenate(result)
def ci(x): return tuple(np.quantile(x,[.025,.975],axis=0))
def mdtable(rows,keys,formats=None):
    if not rows:return ''
    def fmt(r,k):
        v=r.get(k,'')
        return (f'{v:.3f}' if isinstance(v,float) else str(v)).replace('|','/')
    return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(fmt(r,k) for k in keys)+' |' for r in rows)
rows=[];bad_lines=[]
with (RUN/'attempts.jsonl').open(encoding='utf-8') as f:
    for n,line in enumerate(f,1):
        try:
            r=json.loads(line);r['_line']=n;rows.append(r)
        except Exception as e:bad_lines.append({'line':n,'error':str(e)})
keys=collections.Counter(json.dumps(r['key'],sort_keys=True) for r in rows)
issues=[]
num_fields=['L','N','T','S','score','ttft_seconds','total_seconds','cache_build_seconds','cache_get_seconds','dynamic_selection_seconds','output_tokens','peak_allocated_bytes','peak_reserved_bytes','resume_cache_rebuild_seconds']
for r in rows:
    for f in num_fields:
        v=r.get(f)
        if v is not None and (not isinstance(v,(float,int)) or not math.isfinite(v) or (v<0 and f not in ('T','S'))):issues.append({'line':r['_line'],'sample_id':r['sample_id'],'field':f,'value':str(v)})
    if not r.get('reference_answers'):issues.append({'line':r['_line'],'sample_id':r['sample_id'],'field':'reference_answers','value':'missing'})
    if not r.get('answer','').strip():issues.append({'line':r['_line'],'sample_id':r['sample_id'],'field':'answer','value':'empty'})
stage_counts=collections.Counter(r['key']['stage'] for r in rows)
expected={'baseline':1800,'explore':6672,'validation':992,'length_check':120,'attention':12,'longbench_final':1020,'litm':4500,'doc2dial':3000}
dump('record_counts.csv',[{'stage':s,'expected':v,'observed':stage_counts[s],'difference':stage_counts[s]-v,'unit':'generation_record','source':'attempts.jsonl:key.stage'} for s,v in expected.items()])
dump('record_issues.csv',issues or [{'issue':'none','scope':'all 18116 parsed records'}])
dump('record_parse_errors.csv',bad_lines or [{'issue':'none'}])
valid=[r for r in rows if r['status']=='success']
lb=[r for r in valid if r['dataset'] in ('qasper','hotpotqa','musique')]
scorechecks=[]
for r in lb:
    new=max((f1(r['answer'],a) for a in r['reference_answers']),default=0.)
    r['_f1']=new
    scorechecks.append({'line':r['_line'],'stage':r['key']['stage'],'dataset':r['dataset'],'sample_id':r['sample_id'],'method':r['method'],'recomputed_f1_0_1':new,'saved_f1_0_1':r['score'],'difference':new-r['score']})
dump('longbench_score_recheck.csv',scorechecks)
frozen=json.loads((SRC/'datasets/frozen-main80/frozen.json').read_text(encoding='utf-8'))
testids=set(frozen['orders']['test'])
methods5=['sequential','pcw','ape_default','ape_fixed','ape_dynamic']
methods3=methods5[:3]
quality=[];pairs=[];membership=[];bootstores={};matrices={}
for scope,methods in [('baseline600',methods3),('common510',methods5)]:
    ds_boot=[];ds_arrays=[]
    for ds in ('qasper','hotpotqa','musique'):
        by={m:{r['sample_id']:r for r in lb if r['method']==m and r['dataset']==ds and r['key']['stage']==('baseline' if m in methods3 else 'longbench_final') and (scope=='baseline600' or r['sample_id'] in testids)} for m in methods}
        ids=sorted(set.intersection(*(set(v) for v in by.values())))
        a=np.array([[by[m][sid]['_f1'] for m in methods] for sid in ids])
        groups=[cluster_key(by[methods[0]][sid]) for sid in ids]
        b=boot_means(a,groups);bootstores[scope,ds]=b;matrices[scope,ds]=(ids,by,a)
        ds_boot.append(b);ds_arrays.append(a)
        for sid,g in zip(ids,groups):membership.append({'scope':scope,'dataset':ds,'sample_id':sid,'cluster':g,'methods':methods,'source':'attempts.jsonl;frozen.json:orders.test'})
        for j,m in enumerate(methods):
            lo,hi=ci(b[:,j]);quality.append({'scope':scope,'dataset':ds,'method':m,'n':len(ids),'clusters':len(set(groups)),'f1_0_1':a[:,j].mean(),'f1_percent':100*a[:,j].mean(),'ci_low_0_1':lo,'ci_high_0_1':hi,'ci_low_percent':lo*100,'ci_high_percent':hi*100,'unit':'F1 x100','bootstrap':'dataset-stratified evidence-group cluster percentile','seed':SEED,'replicates':BOOT})
        for ia,ib in itertools.combinations(range(len(methods)),2):
            d=a[:,ib]-a[:,ia];lo,hi=ci(b[:,ib]-b[:,ia])
            pairs.append({'scope':scope,'dataset':ds,'method_a':methods[ia],'method_b':methods[ib],'n':len(ids),'delta_b_minus_a_0_1':d.mean(),'delta_percentage_points':100*d.mean(),'ci_low_pp':100*lo,'ci_high_pp':100*hi,'win_b':int((d>1e-12).sum()),'tie':int((abs(d)<=1e-12).sum()),'loss_b':int((d< -1e-12).sum()),'unit':'F1 percentage point','bootstrap':'paired evidence-group cluster','seed':SEED,'replicates':BOOT})
    bmacro=np.mean(ds_boot,axis=0)
    means=np.mean([x.mean(axis=0) for x in ds_arrays],axis=0)
    for j,m in enumerate(methods):
        lo,hi=ci(bmacro[:,j]);quality.append({'scope':scope,'dataset':'macro_equal_dataset','method':m,'n':sum(map(len,ds_arrays)),'f1_0_1':means[j],'f1_percent':means[j]*100,'ci_low_0_1':lo,'ci_high_0_1':hi,'ci_low_percent':lo*100,'ci_high_percent':hi*100,'unit':'F1 x100','bootstrap':'dataset-stratified evidence-group cluster percentile','seed':SEED,'replicates':BOOT})
    allarray=np.concatenate(ds_arrays)
    for ia,ib in itertools.combinations(range(len(methods)),2):
        lo,hi=ci(bmacro[:,ib]-bmacro[:,ia]);d=allarray[:,ib]-allarray[:,ia]
        pairs.append({'scope':scope,'dataset':'macro_equal_dataset','method_a':methods[ia],'method_b':methods[ib],'n':len(allarray),'delta_b_minus_a_0_1':means[ib]-means[ia],'delta_percentage_points':100*(means[ib]-means[ia]),'ci_low_pp':100*lo,'ci_high_pp':100*hi,'win_b':int((d>1e-12).sum()),'tie':int((abs(d)<=1e-12).sum()),'loss_b':int((d< -1e-12).sum()),'unit':'F1 percentage point','bootstrap':'paired stratified evidence-group cluster','seed':SEED,'replicates':BOOT})
dump('longbench_quality.csv',quality);dump('longbench_paired.csv',pairs);dump('longbench_membership.csv',membership)
cases=[]
for ds in ('qasper','hotpotqa','musique'):
    ids,by,a=matrices['common510',ds]
    data=[(by['ape_dynamic'][sid]['_f1']-by['ape_fixed'][sid]['_f1'],sid) for sid in ids]
    for label,pred,reverse in [('improvement',lambda d:d>1e-12,True),('degradation',lambda d:d< -1e-12,False),('tie',lambda d:abs(d)<=1e-12,False)]:
        options=[(d,sid) for d,sid in data if pred(d)]
        options=sorted(options,key=lambda x:((-x[0] if reverse else x[0]),x[1]))
        if options:
            d,sid=options[0];rf=by['ape_fixed'][sid];rd=by['ape_dynamic'][sid]
            cases.append({'dataset':ds,'case_type':label,'sample_id':sid,'selection_rule':'largest positive/negative delta; tie lexicographic ID','delta_f1_0_1':d,'fixed_f1':rf['_f1'],'dynamic_f1':rd['_f1'],'fixed_T':rf['T'],'fixed_S':rf['S'],'dynamic_T':rd['T'],'dynamic_S':rd['S'],'L':rd['L'],'N':rd['N'],'fixed_answer':rf['answer'],'dynamic_answer':rd['answer'],'references':rd['reference_answers'],'fixed_source_line':rf['_line'],'dynamic_source_line':rd['_line']})
dump('longbench_cases.csv',cases)
# Timing descriptives: never merge search reuse, formal cold, app hot, attention.
fields={'ttft_seconds':'s','total_seconds':'s','cache_build_seconds':'s','cache_get_seconds':'s','dynamic_selection_seconds':'s','resume_cache_rebuild_seconds':'s','output_tokens':'token_including_EOS','peak_allocated_bytes':'GiB','peak_reserved_bytes':'GiB'}
groups=collections.defaultdict(list)
for r in valid: groups[(r['key']['stage'],r['dataset'],r['method'],r['cache_mode'],r['cache_hit'])].append(r)
timing=[]
for k,rs in sorted(groups.items()):
    for field,unit in fields.items():
        a=np.array([r[field] for r in rs if isinstance(r.get(field),(float,int))],float)
        if not len(a):continue
        if unit=='GiB':a/=2**30
        timing.append(dict(zip(('stage','dataset','method','cache_mode','cache_hit'),k),metric=field,unit=unit,n=len(a),denominator='valid successful requests in exact stage/dataset/method/cache/hit group',mean=a.mean(),median=np.median(a),p90=np.quantile(a,.9),minimum=a.min(),maximum=a.max(),source='attempts.jsonl:'+field))
dump('efficiency_descriptive.csv',timing)
# Paired latency comparisons: method ratios computed per request; resample cluster together.
perf=[]
formal=[]
for r in valid:
    stage=r['key']['stage']
    if stage=='baseline' or stage=='longbench_final':
        if r['sample_id'] in testids:formal.append(('longbench_common510',r))
    elif stage in ('litm','doc2dial'): formal.append((stage,r))
for scope in ('longbench_common510','litm','doc2dial'):
    sr=[r for s,r in formal if s==scope]
    caches=sorted(set(r['cache_mode'] for r in sr))
    for cache in caches:
        dslist=sorted(set(r['dataset'] for r in sr))
        scopes=dslist+(['all'] if len(dslist)>1 else [])
        for ds in scopes:
            use=[r for r in sr if r['cache_mode']==cache and (ds=='all' or r['dataset']==ds)]
            ms=[m for m in methods5 if any(r['method']==m for r in use)]
            by={m:{(r['dataset'],r['sample_id'],r.get('position')):r for r in use if r['method']==m} for m in ms}
            ids=sorted(set.intersection(*(set(x) for x in by.values())),key=str)
            if not ids:continue
            gr=[(cluster_key(by[ms[0]][sid]) if scope=='longbench_common510' else (sid[1].rsplit('_',1)[0] if scope=='doc2dial' else sid[1])) for sid in ids]
            for ia,ib in itertools.combinations(ms,2):
                for metric in ('ttft_seconds','total_seconds'):
                    va=np.array([by[ia][sid][metric] for sid in ids]);vb=np.array([by[ib][sid][metric] for sid in ids])
                    a=np.column_stack((va,vb,vb-va,va/vb))
                    b=boot_means(a,gr)
                    dl,dh=ci(b[:,2]);rl,rh=ci(b[:,3]);mol,moh=ci(b[:,0]/b[:,1])
                    perf.append({'scope':scope,'dataset':ds,'cache_mode':cache,'method_a':ia,'method_b':ib,'metric':metric,'n_requests':len(ids),'n_clusters':len(set(gr)),'mean_a_seconds':va.mean(),'mean_b_seconds':vb.mean(),'delta_b_minus_a_seconds':(vb-va).mean(),'delta_ci_low_seconds':dl,'delta_ci_high_seconds':dh,'paired_speedup_mean_a_over_b':(va/vb).mean(),'paired_speedup_median_a_over_b':np.median(va/vb),'speedup_ci_low':rl,'speedup_ci_high':rh,'ratio_of_means_a_over_b':va.mean()/vb.mean(),'ratio_of_means_ci_low':mol,'ratio_of_means_ci_high':moh,'unit':'seconds and dimensionless ratio','bootstrap':'paired cluster; LB evidence group; LITM question with positions; D2D dialogue with turns','seed':SEED,'replicates':BOOT})
# cold/hot within method, by dialogue
for method in ('sequential','ape_fixed','ape_dynamic'):
    by={c:{r['sample_id']:r for r in valid if r['dataset']=='doc2dial' and r['method']==method and r['cache_mode']==c} for c in ('cold','hot')}
    ids=sorted(set(by['cold'])&set(by['hot']));gr=[s.rsplit('_',1)[0] for s in ids]
    for metric in ('ttft_seconds','total_seconds'):
        a=np.array([[by['cold'][s][metric],by['hot'][s][metric]] for s in ids]);a=np.column_stack((a,a[:,1]-a[:,0],a[:,0]/a[:,1]));b=boot_means(a,gr);dl,dh=ci(b[:,2]);rl,rh=ci(b[:,3])
        perf.append({'scope':'doc2dial_cold_hot','dataset':'doc2dial','cache_mode':'paired_cold_hot','method_a':method+'_cold','method_b':method+'_hot','metric':metric,'n_requests':len(ids),'n_clusters':len(set(gr)),'mean_a_seconds':a[:,0].mean(),'mean_b_seconds':a[:,1].mean(),'delta_b_minus_a_seconds':a[:,2].mean(),'delta_ci_low_seconds':dl,'delta_ci_high_seconds':dh,'paired_speedup_mean_a_over_b':a[:,3].mean(),'paired_speedup_median_a_over_b':np.median(a[:,3]),'speedup_ci_low':rl,'speedup_ci_high':rh,'ratio_of_means_a_over_b':a[:,0].mean()/a[:,1].mean(),'unit':'seconds and dimensionless ratio','bootstrap':'paired dialogue cluster; all turns and both cache conditions together','seed':SEED,'replicates':BOOT})
dump('efficiency_paired.csv',perf)
# Source linkage: standalone dictionary for every raw field, plus analysis definitions.
desc={
'L':('token','actual document/context tokens after construction, excludes prefix/request'), 'N':('block','actual encoded context block count; key.N can be null for natural construction'),
'key':('JSON','logical generation key: stage/sample/method/L/N/parameters/position/cache; excludes attempt'),
'logical_labels':('labels','candidate roles sharing exactly one generation; never duplicated in raw-record denominators'),
'group':('source cluster','source evidence/paper component, used for LongBench cluster bootstrap'),
'original_document_count':('document','original source documents before token block construction'),
'T':('dimensionless','attention temperature of saved actual method'), 'S':('dimensionless','context attention scaling of saved actual method'),
'score':('0-1 F1/EM or 0-100 sentence BLEU','saved score; recompute quality with task metric, corpus BLEU aggregated separately'),
'output_token_ids':('token IDs','generated tokens including terminal EOS when produced'), 'output_tokens':('token including EOS','len(output_token_ids)'),
'peak_allocated_bytes':('byte','peak total process tensor allocation since reset; model and alive caches included'),
'peak_reserved_bytes':('byte','peak total process allocator reserve since reset; retained allocator memory included'),
'cache_mode':('condition','search/cold/hot labels from campaign'), 'cache_hit':('boolean','actual same-process GPU KV hit'),
'reference_answers':('string list','saved references joined to sample_id; task-specific independent checks'),
'query_start_position':('token index','prefix + encoded documents location'),
}
for f in ('ttft_seconds','total_seconds','cache_build_seconds','cache_get_seconds','dynamic_selection_seconds','resume_cache_rebuild_seconds'):desc[f]=('s',{'ttft_seconds':'request start through first argmax token, synchronized; includes prepare and cache work','total_seconds':'request start through generation and answer text decoding','cache_build_seconds':'synchronized document/prefix KV encoding on miss; zero on hit','cache_get_seconds':'host cache lookup duration','dynamic_selection_seconds':'host parameter-selection duration; already included in total','resume_cache_rebuild_seconds':'separate recovery cache rebuild duration' }[f])
allfields=sorted({f for r in rows for f in r if not f.startswith('_')})
dump('data_dictionary.csv',[{'field':f,'unit':desc.get(f,('see raw JSON','literal raw evidence field'))[0],'definition':desc.get(f,('','literal raw evidence field'))[1],'source':'attempts.jsonl','example_type':type(next(r[f] for r in rows if f in r and r[f] is not None)).__name__ if any(r.get(f) is not None for r in rows) else 'null'} for f in allfields])
# Automatic summary comparison is a targeted check on reported means, not a file comparison.
saved=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
checks=[]
for ent in saved['longbench_five_method_paired']:
    for m,value in ent['mean_f1'].items():
        rr=next(q for q in quality if q['scope']=='common510' and q['dataset']==ent['dataset'] and q['method']==m)
        checks.append({'section':'longbench_five_method_paired','dataset':ent['dataset'],'method':m,'saved':value,'recomputed':rr['f1_0_1'],'difference':rr['f1_0_1']-value})
dump('summary_recheck.csv',checks)
env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'seed':SEED,'bootstrap_replicates':BOOT,'analysis_date_local':'2026-11-16','campaign':'main80-20261114T041503','raw_file':'restored/output/runs/main80-20261114T041503/attempts.jsonl','analysis_status':'draft; SUBMITTED preserved'}
(OUT/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
# Static figures use the exact CSV values.
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(10,4.6));x=np.arange(4);width=.15
for j,m in enumerate(methods5):
    qs=[next(q for q in quality if q['scope']=='common510' and q['dataset']==ds and q['method']==m) for ds in ('qasper','hotpotqa','musique','macro_equal_dataset')]
    ax.bar(x+(j-2)*width,[q['f1_percent'] for q in qs],width,label=m)
ax.set_xticks(x,['Qasper','HotpotQA','MuSiQue','等权宏平均']);ax.set_ylabel('F1 ×100');ax.set_title('LongBench 共同510题五方法比较');ax.legend(ncol=3,fontsize=8);fig.tight_layout()
for ext in ('png','svg'):fig.savefig(OUT/'figures'/('longbench_quality.'+ext),dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,metric,label in zip(axs,['ttft_seconds','total_seconds'],['首 token 时间（秒）','总响应时间（秒）']):
    vals=[]
    for m in methods5:
        rr=[r for s,r in formal if s=='longbench_common510' and r['method']==m];vals.append([r[metric] for r in rr])
    ax.boxplot(vals,tick_labels=methods5,showfliers=False);ax.tick_params(axis='x',rotation=30);ax.set_ylabel(label)
fig.suptitle('LongBench共同510题：正式冷请求');fig.tight_layout()
for ext in ('png','svg'):fig.savefig(OUT/'figures'/('longbench_latency.'+ext),dpi=180)
plt.close(fig)
summary={'records':len(rows),'stages':dict(stage_counts),'statuses':dict(collections.Counter(r['status'] for r in rows)),'duplicate_key_count':sum(v-1 for v in keys.values() if v>1),'attempt_distribution':dict(collections.Counter(r['attempt'] for r in rows)),'parse_errors':len(bad_lines),'record_issues':len(issues),'longbench_scored_records':len(scorechecks),'max_longbench_score_difference':max(abs(c['difference']) for c in scorechecks),'quality':quality,'pairs':pairs,'efficiency_paired':perf}
(OUT/'main_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
text='# 主统计复算记录\n\n本次从逐题答案独立复算LongBench F1，并按冻结测试ID连接方法。所有置信区间采用固定种子20261116、5000次配对聚类百分位重采样。LongBench在每个数据集内按原论文/证据组重采样，再对三个数据集等权；LITM时延以问题为簇共同保留三位置，Doc2Dial时延以对话为簇保留目标轮及缓存条件。区间描述该campaign样本组构成变化，硬件重复运行变动未采样。\n\n'
text+=f'原始记录{len(rows)}，状态{summary["statuses"]}，重复逻辑键{summary["duplicate_key_count"]}，非法/缺失字段项{len(issues)}。LongBench逐题评分最大绝对差{summary["max_longbench_score_difference"]:.3g}。\n\n'
text+='## LongBench\n\n'+mdtable(quality,['scope','dataset','method','n','f1_percent','ci_low_percent','ci_high_percent'])+'\n\n## 配对差异\n\n'+mdtable(pairs,['scope','dataset','method_a','method_b','n','delta_percentage_points','ci_low_pp','ci_high_pp','win_b','tie','loss_b'])+'\n\n'
text+='## 效率口径\n\n均值=Σx/n；P90使用NumPy线性分位数；配对加速比=同一请求A耗时/B耗时，表内分别给均值、中位数与均值比。差值统一B−A。所有秒字段保留原始单位；显存以GiB=bytes/2^30展示，含模型与存活缓存。描述表按stage/dataset/method/cache_mode/cache_hit分层，搜索复用、正式冷请求、应用热请求、注意力诊断分别统计。模型加载表由注意力审计脚本从events独立生成。\n\n'
text+='具体样本成员、逐题评分、字段字典及全部指标见tables CSV；表中source、line、scope和denominator提供追溯。\n'
(OUT/'reports/MAIN_STATISTICS.md').write_text(text,encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('records','stages','statuses','duplicate_key_count','attempt_distribution','record_issues','max_longbench_score_difference')},ensure_ascii=False,indent=2))
