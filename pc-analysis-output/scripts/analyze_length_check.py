"""Experiment 4 only: saved 6K/12K fixed-vs-dynamic records and frozen mapping.
No score re-computation, new model generation, test-score use or rule modification.
"""
from pathlib import Path
from collections import defaultdict
from itertools import product
from statistics import mean
import csv,json,math,os,shutil,sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/04_length_check'
RUN=ROOT/'restored/output/runs/main80-20261114T041503'
for sub in ('tables','reports','figures','sources','cache','tmp','logs'):(OUT/sub).mkdir(parents=True,exist_ok=True)
for key in ('MPLCONFIGDIR','XDG_CACHE_HOME','TEMP','TMP'):os.environ[key]=str(OUT/('cache' if key in ('MPLCONFIGDIR','XDG_CACHE_HOME') else 'tmp'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DS=('qasper','hotpotqa','musique')
METHODS=('ape_fixed','ape_dynamic')
CELLS=list(product((6144,12288),(4,8,16)))
GRID=list(product((4096,8192,16384),(4,8,16)))
SEED=20261116;BOOT=5000
FIELDS={'score':'F1 0-1','ttft_seconds':'s','total_seconds':'s','cache_build_seconds':'s','cache_get_seconds':'s','dynamic_selection_seconds':'s','output_tokens':'token including EOS','peak_allocated_bytes':'GiB','peak_reserved_bytes':'GiB'}
COLS=list(product(METHODS,FIELDS))
RATIOS=('ttft_seconds','total_seconds')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def writecsv(name,rows):
    rows=list(rows)
    if not rows:return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def ci(values):return np.quantile(values,[.025,.975],axis=0)
def nearest(l,n):return min(GRID,key=lambda c:((math.log2(l)-math.log2(c[0]))**2+(math.log2(n)-math.log2(c[1]))**2,c[0],c[1]))
def md(cols,rows):return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

def dataset_bootstrap(inputs,scope_index):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,scope_index]))
    output=[];diagnostics=[]
    for d in DS:
        keys=[k for k in inputs if k[0]==d]
        groups={k:json.dumps(inputs[k]['ape_fixed']['group'],ensure_ascii=False,sort_keys=True) for k in keys}
        unique=sorted(set(groups.values()));lookup={g:i for i,g in enumerate(unique)}
        count=np.zeros((len(unique),6));sums=np.zeros((len(unique),6,len(COLS)+len(RATIOS)))
        for k in keys:
            g=lookup[groups[k]];c=CELLS.index((k[2],k[3]));count[g,c]+=1;values=[]
            for m,f in COLS:
                v=inputs[k][m][f];values.append(v/2**30 if FIELDS[f]=='GiB' else v)
            values.extend(inputs[k]['ape_fixed'][f]/inputs[k]['ape_dynamic'][f] for f in RATIOS)
            sums[g,c]+=values
        draws=[]
        for start in range(0,BOOT,100):
            idx=rng.integers(0,len(unique),size=(min(100,BOOT-start),len(unique)))
            den=count[idx].sum(axis=1);assert (den>0).all()
            draws.append(sums[idx].sum(axis=1)/den[:,:,None])
        output.append(np.concatenate(draws))
        diagnostics.append({'scope':'all10' if scope_index==0 else 'evidence_complete9','dataset':d,'unique_ids':len({k[1] for k in keys}),'evidence_groups':len(unique),'paired_input_conditions':len(keys),'replicates':BOOT,'seed':SEED,'rng':'SeedSequence([seed, scope_index]); dataset order qasper/hotpotqa/musique','unit':'ID / source evidence group / condition count'})
    return np.array(output),diagnostics

def main():
    rule=read(RUN/'rule.json');frozen=read(RUN/'snapshot/datasets/frozen-main80/frozen.json')
    ids=frozen['orders']['validation'][:10]
    copy=OUT/'sources/original_frozen_rule.json'
    if not copy.exists():shutil.copyfile(RUN/'rule.json',copy)
    evidence={(r['sample_id'],int(r['L'])):r for r in rcsv(BASE/'stages/01_audit/tables/construction_checks.csv') if int(r['L']) in (6144,12288) and r['split']=='validation'}
    records=[];by_input=defaultdict(dict)
    with (RUN/'attempts.jsonl').open(encoding='utf-8') as f:
        for line,text in enumerate(f,1):
            r=json.loads(text)
            if r['key']['stage']!='length_check':continue
            assert r['status']=='success' and r['cache_mode']=='cold' and not r['cache_hit']
            r['_line']=line;r['_complete']=int(evidence[(r['sample_id'],r['L'])]['n_missing_required_tokens'])==0
            k=(r['dataset'],r['sample_id'],r['L'],r['N']);assert r['method'] not in by_input[k]
            by_input[k][r['method']]=r;records.append(r)
    assert len(records)==120 and set(r['sample_id'] for r in records)==set(ids) and len(by_input)==60
    paired_rows=[]
    for k,rr in by_input.items():
        assert set(rr)==set(METHODS)
        a,b=rr['ape_fixed'],rr['ape_dynamic'];mapped=nearest(k[2],k[3]);p=tuple(rule['cells'][f'{mapped[0]}/{mapped[1]}'])
        assert (a['T'],a['S'])==tuple(rule['fixed']) and (b['T'],b['S'])==p
        assert a['block_lengths']==b['block_lengths'] and a['reference_answers']==b['reference_answers'] and a['_complete']==b['_complete']
        same=(a['T'],a['S'])==(b['T'],b['S'])
        if same:assert a['answer']==b['answer'] and a['output_token_ids']==b['output_token_ids']
        paired_rows.append({'dataset':k[0],'sample_id':k[1],'L':k[2],'N':k[3],'mapped_L':mapped[0],'mapped_N':mapped[1],'fixed_T':a['T'],'fixed_S':a['S'],'dynamic_T':b['T'],'dynamic_S':b['S'],'parameters_same':same,'answers_same':a['answer']==b['answer'],'tokens_same':a['output_token_ids']==b['output_token_ids'],'evidence_complete':a['_complete'],'missing_evidence_tokens':int(evidence[(k[1],k[2])]['n_missing_required_tokens']),'fixed_f1_0_1':a['score'],'dynamic_f1_0_1':b['score'],'difference_0_1':b['score']-a['score'],'fixed_answer':a['answer'],'dynamic_answer':b['answer'],'reference_answers':a['reference_answers'],'fixed_line':a['_line'],'dynamic_line':b['_line'],**{'fixed_'+f:a[f] for f in FIELDS if f!='score'},**{'dynamic_'+f:b[f] for f in FIELDS if f!='score'},'unit':'saved F1 0-1; seconds; output tokens; raw peak bytes'})
    complete={k:v for k,v in by_input.items() if v['ape_fixed']['_complete']}
    scopes={'all10':dict(by_input),'evidence_complete9':complete}
    scope_counts=[];quality=[];dataset_quality=[];eff=[];effpairs=[];diagnostics=[];mapping=[];summaries={}
    for scope_index,(scope,inputs) in enumerate(scopes.items()):
        n_by_ds=np.array([len({k[1] for k in inputs if k[0]==d}) for d in DS]);weights=n_by_ds/n_by_ds.sum()
        arr,diag=dataset_bootstrap(inputs,scope_index);diagnostics.extend(diag)
        qb=arr[:,:,:,COLS.index(('ape_fixed','score'))].mean(axis=0);qd=arr[:,:,:,COLS.index(('ape_dynamic','score'))].mean(axis=0)
        scope_counts.append({'scope':scope,'n_unique_ids':int(n_by_ds.sum()),'n_qasper':int(n_by_ds[0]),'n_hotpotqa':int(n_by_ds[1]),'n_musique':int(n_by_ds[2]),'n_pairs':len(inputs),'n_generations':len(inputs)*2,'n_lengths':2,'n_N':3,'quality_aggregation':'datasets and six cells equal','efficiency_aggregation':'pooled requests; bootstrap dataset weights fixed to original ID counts','unit':'ID / paired condition / generation count'})
        for cell in [*CELLS,None]:
            use=[k for k in inputs if cell is None or (k[2],k[3])==cell]
            for d in DS:
                dk=[k for k in use if k[0]==d]
                dataset_quality.append({'scope':scope,'dataset':d,'L':'ALL' if cell is None else cell[0],'N':'ALL' if cell is None else cell[1],'n_unique_ids':len({k[1] for k in dk}),'n_pairs':len(dk),'fixed_f1_percent':mean(inputs[k]['ape_fixed']['score'] for k in dk)*100,'dynamic_f1_percent':mean(inputs[k]['ape_dynamic']['score'] for k in dk)*100,'unit':'F1 x100'})
            fixed=mean(mean(inputs[k]['ape_fixed']['score'] for k in use if k[0]==d) for d in DS)
            dynamic=mean(mean(inputs[k]['ape_dynamic']['score'] for k in use if k[0]==d) for d in DS)
            aa=qb.mean(axis=1) if cell is None else qb[:,CELLS.index(cell)];bb=qd.mean(axis=1) if cell is None else qd[:,CELLS.index(cell)]
            al,ah=ci(aa);bl,bh=ci(bb);dl,dh=ci(bb-aa)
            diffs=[inputs[k]['ape_dynamic']['score']-inputs[k]['ape_fixed']['score'] for k in use]
            quality.append({'scope':scope,'L':'ALL' if cell is None else cell[0],'N':'ALL' if cell is None else cell[1],'n_unique_ids':len({k[1] for k in use}),'n_pairs':len(use),'fixed_f1_percent':fixed*100,'dynamic_f1_percent':dynamic*100,'difference_pp':(dynamic-fixed)*100,'difference_ci_low_pp':float(dl*100),'difference_ci_high_pp':float(dh*100),'fixed_ci_low':float(al*100),'fixed_ci_high':float(ah*100),'dynamic_ci_low':float(bl*100),'dynamic_ci_high':float(bh*100),'win_dynamic':sum(x>1e-12 for x in diffs),'tie':sum(abs(x)<=1e-12 for x in diffs),'loss_dynamic':sum(x< -1e-12 for x in diffs),'unit':'F1 x100; paired macro difference percentage points','seed':SEED,'replicates':BOOT})
            for method,f in COLS:
                if f=='score':continue
                vals=np.array([inputs[k][method][f] for k in use],float)
                if FIELDS[f]=='GiB':vals/=2**30
                boot=(arr[:,:,:,COLS.index((method,f))]*weights[:,None,None]).sum(axis=0)
                boot=boot.mean(axis=1) if cell is None else boot[:,CELLS.index(cell)]
                lo,hi=ci(boot)
                eff.append({'scope':scope,'L':'ALL' if cell is None else cell[0],'N':'ALL' if cell is None else cell[1],'method':method,'metric':f,'n_requests':len(use),'mean':float(vals.mean()),'median':float(np.median(vals)),'p90':float(np.quantile(vals,.9)),'mean_ci_low':float(lo),'mean_ci_high':float(hi),'unit':FIELDS[f],'cache':'cold miss; document encoding included','seed':SEED,'replicates':BOOT})
            for f in RATIOS:
                va=np.array([inputs[k]['ape_fixed'][f] for k in use]);vb=np.array([inputs[k]['ape_dynamic'][f] for k in use]);rat=va/vb
                ma=(arr[:,:,:,COLS.index(('ape_fixed',f))]*weights[:,None,None]).sum(axis=0);mb=(arr[:,:,:,COLS.index(('ape_dynamic',f))]*weights[:,None,None]).sum(axis=0);br=(arr[:,:,:,len(COLS)+RATIOS.index(f)]*weights[:,None,None]).sum(axis=0)
                if cell is None:ma,mb,br=ma.mean(axis=1),mb.mean(axis=1),br.mean(axis=1)
                else:ma,mb,br=ma[:,CELLS.index(cell)],mb[:,CELLS.index(cell)],br[:,CELLS.index(cell)]
                dl,dh=ci(mb-ma);rl,rh=ci(br);ml,mh=ci(ma/mb)
                effpairs.append({'scope':scope,'L':'ALL' if cell is None else cell[0],'N':'ALL' if cell is None else cell[1],'metric':f,'n_pairs':len(use),'mean_fixed_s':float(va.mean()),'mean_dynamic_s':float(vb.mean()),'difference_dynamic_minus_fixed_s':float((vb-va).mean()),'difference_ci_low_s':float(dl),'difference_ci_high_s':float(dh),'paired_speedup_mean_fixed_over_dynamic':float(rat.mean()),'paired_speedup_median':float(np.median(rat)),'speedup_mean_ci_low':float(rl),'speedup_mean_ci_high':float(rh),'ratio_of_means_fixed_over_dynamic':float(va.mean()/vb.mean()),'ratio_of_means_ci_low':float(ml),'ratio_of_means_ci_high':float(mh),'unit':'s; dimensionless ratios (>1 dynamic faster)','seed':SEED,'replicates':BOOT})
        summaries[scope]=quality[-1]
    for cell in CELLS:
        rr=[r for r in paired_rows if (r['L'],r['N'])==cell]
        r=rr[0];mapping.append({'L':cell[0],'N':cell[1],'mapped_L':r['mapped_L'],'mapped_N':r['mapped_N'],'fixed_T':r['fixed_T'],'fixed_S':r['fixed_S'],'dynamic_T':r['dynamic_T'],'dynamic_S':r['dynamic_S'],'n_pairs':len(rr),'n_parameters_same':sum(x['parameters_same'] for x in rr),'n_tokens_same':sum(x['tokens_same'] for x in rr),'n_answers_same':sum(x['answers_same'] for x in rr),'unit':'input tokens / blocks / matched pairs'})
    missing=[r for r in paired_rows if not r['evidence_complete']]
    selected=[];availability=[]
    for label,predicate,sign in [('改善',lambda x:x>1e-12,-1),('退化',lambda x:x< -1e-12,1),('持平',lambda x:abs(x)<=1e-12,1)]:
        rr=sorted([r for r in paired_rows if predicate(r['difference_0_1'])],key=lambda r:(sign*r['difference_0_1'],r['sample_id'],r['L'],r['N']))
        availability.append({'case_type':label,'n_pairs':len(rr),'selection_rule':'largest positive/negative saved F1 delta; zero or ties smallest ID then L,N'})
        if rr:selected.append({'case_type':label,**rr[0]})
    for r in paired_rows:r['source_file']='restored/output/runs/main80-20261114T041503/attempts.jsonl'
    worker=[r for r in rcsv(BASE/'stages/01_audit/tables/runtime_workers.csv') if r['stage']=='length_check']
    limits=[r for r in rcsv(BASE/'stages/01_audit/tables/runtime_generation_limits.csv') if r['stage']=='length_check']
    limit_by_key={(r['dataset'],r['method']):int(r['max_new_tokens']) for r in limits}
    caps=[{'dataset':r['dataset'],'sample_id':r['sample_id'],'L':r['L'],'N':r['N'],'method':r['method'],'output_tokens':r['output_tokens'],'max_new_tokens':limit_by_key[(r['dataset'],r['method'])],'saved_f1_0_1':r['score'],'answer':r['answer'],'reference_answers':r['reference_answers'],'total_seconds':r['total_seconds'],'evidence_complete':r['_complete'],'source_file':'restored/output/runs/main80-20261114T041503/attempts.jsonl','source_line':r['_line'],'unit':'tokens; saved F1 0-1; seconds'} for r in records if r['output_tokens']>=limit_by_key[(r['dataset'],r['method'])]]
    writecsv('generation_limits.csv',limits);writecsv('generation_cap_records.csv',caps)
    for name,data in [('scope_counts.csv',scope_counts),('paired_records.csv',paired_rows),('mapping_and_output_checks.csv',mapping),('quality_paired.csv',quality),('dataset_cell_quality.csv',dataset_quality),('efficiency_descriptive.csv',eff),('efficiency_paired.csv',effpairs),('evidence_missing_pairs.csv',missing),('selected_cases.csv',selected),('case_type_availability.csv',availability),('bootstrap_units.csv',diagnostics),('model_load_workers.csv',worker)]:writecsv(name,data)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,scope,title in zip(axs,scopes,('全部10题','证据完整9题')):
        rr=[r for r in quality if r['scope']==scope and r['L']!='ALL'];xx=np.arange(6)
        for dx,m,label,color in [(-.18,'fixed','固定APE','#4c78a8'),(.18,'dynamic','动态APE','#f58518')]:
            yy=np.array([r[m+'_f1_percent'] for r in rr]);err=np.array([[r[m+'_f1_percent']-r[m+'_ci_low'] for r in rr],[r[m+'_ci_high']-r[m+'_f1_percent'] for r in rr]])
            ax.bar(xx+dx,yy,.36,label=label,color=color);ax.errorbar(xx+dx,yy,yerr=err,fmt='none',ecolor='black',capsize=2)
        ax.set_xticks(xx,[f'{int(r["L"])//1024}K/{r["N"]}' for r in rr],rotation=25);ax.set_title(title);ax.set_ylabel('F1 ×100（三数据集等权）');ax.set_xlabel('长度 / 实际块数');ax.legend()
    fig.suptitle('实验4：新长度条件下固定与动态APE');save(fig,'length_quality')
    fig,axs=plt.subplots(1,2,figsize=(10,4.3))
    for ax,f,title in zip(axs,('ttft_seconds','total_seconds'),('首token时间','总响应时间')):
        for m,label in zip(METHODS,('固定APE','动态APE')):
            rr=[r for r in eff if r['scope']=='all10' and r['metric']==f and r['method']==m and r['L']!='ALL']
            ax.plot(range(6),[r['mean'] for r in rr],marker='o',label=label)
        ax.set_xticks(range(6),['6K/4','6K/8','6K/16','12K/4','12K/8','12K/16'],rotation=25);ax.set_ylabel('秒（每格10个冷请求均值）');ax.set_title(title);ax.legend()
    save(fig,'length_latency')
    status={'experiment_stage':4,'status':'COMPLETED','research_status':'SUBMITTED','physical_generations':120,'paired_input_conditions':60,'unique_question_ids':10,'complete_generations':108,'complete_paired_conditions':54,'same_parameters_identical_output_pairs':sum(r['parameters_same'] and r['tokens_same'] for r in paired_rows),'score_recomputed_count':0,'test_scores_used':False,'rule_modified':False,'next_stage_executed':False}
    result={**status,'quality_overall':summaries,'mapping':mapping,'scope_counts':scope_counts,'missing_evidence_ids':sorted({r['sample_id'] for r in missing}),'cap_hit_count':len(caps),'eos_last_count':sum(int(r['eos_last_count']) for r in limits)}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'STAGE_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'seed':SEED,'replicates':BOOT,'bootstrap':'paired source evidence groups within dataset; all six L/N variants retained; parameters/rule fixed; percentile CI','quality_weights':'each dataset 1/3, six cells 1/6','efficiency_weights':'pooled requests; dataset bootstrap weights original question proportions 4/10,3/10,3/10 or 4/9,3/9,2/9','input':'length_check saved score and timing only; pre-audit evidence tags; frozen rule and validation-ID order','analysis_commit':'227116e4a004153272640cfe31087a033861b8e9','new_dependencies':[]},ensure_ascii=False,indent=2),encoding='utf-8')
    report='# 实验第四阶段：6K／12K长度检查\n\n'
    report+='## 1 数据与分析方法\n\n我们引用冻结顺序中的首10个验证问题，在6144和12288 document tokens、N=4/8/16下的120条保存记录。固定与动态APE形成60组同题/L/N配对；Qasper4题、HotpotQA3题、MuSiQue3题，每题六种输入条件。所有请求为冷请求且资料KV未命中。直接复用已有score和计时，评分重算次数为0。\n\n'
    report+='这10题来自验证集，长度条件为冻结后新增的6K和12K。我们保持规则不变，检查新长度的映射与表现。F1按0–100展示，先对数据集内题目平均，再对三个数据集及六格等权；胜平负按输入配对计数。效率均值、中位数和P90采用实际请求分布；各数据集题数权重在重采样中保持原比例。\n\n原始来源为restored/output/runs/main80-20261114T041503/attempts.jsonl；paired_records.csv保存source_file、两方法源行号、筛选后的sample_id/L/N、score、答案和各项计时。冻结规则与验证顺序取同次运行的rule.json和snapshot/datasets/frozen-main80/frozen.json；证据完整性标签及生成停止计数引用前置审核construction_checks.csv和runtime_generation_limits.csv。\n\n'
    report+='## 2 实际参数映射与相同输出\n\n'
    report+=md(['输入长度','N','映射格','固定T/S','动态T/S','相同参数配对','相同token配对'],[[r['L'],r['N'],f"{r['mapped_L']}/{r['mapped_N']}",f"{r['fixed_T']}/{r['fixed_S']}",f"{r['dynamic_T']}/{r['dynamic_S']}",f"{r['n_parameters_same']}/{r['n_pairs']}",f"{r['n_tokens_same']}/{r['n_pairs']}"] for r in mapping])
    report+='\n规则在log2(L)、log2(N)空间取最近格，6144映射8192，12288映射16384。6K/4与6K/16采用不同于固定值的参数；6K/8与12K三格共40组配对的参数、答案及token序列相同。12K这一长度下动态规则全部沿用固定0.7/1.0。\n\n'
    report+='## 3 质量结果与证据完整性\n\n'
    report+=md(['口径','配对数','固定F1','动态F1','动态−固定/百分点','95%配对区间','动态胜/平/负'],[[s,r['n_pairs'],f"{r['fixed_f1_percent']:.4f}",f"{r['dynamic_f1_percent']:.4f}",f"{r['difference_pp']:+.4f}",f"[{r['difference_ci_low_pp']:.3f}, {r['difference_ci_high_pp']:.3f}]",f"{r['win_dynamic']}/{r['tie']}/{r['loss_dynamic']}"] for s,r in summaries.items()])
    report+='\n全量六格分别如下；每格为相同10题。\n\n'
    report+=md(['L','N','固定F1','动态F1','差/百分点','95%配对区间','动态胜/平/负'],[[r['L'],r['N'],f"{r['fixed_f1_percent']:.3f}",f"{r['dynamic_f1_percent']:.3f}",f"{r['difference_pp']:+.3f}",f"[{r['difference_ci_low_pp']:.2f}, {r['difference_ci_high_pp']:.2f}]",f"{r['win_dynamic']}/{r['tie']}/{r['loss_dynamic']}"] for r in quality if r['scope']=='all10' and r['L']!='ALL'])
    report+='\n![长度质量比较](../figures/length_quality.png)\n\n图中误差线为各方法宏F1的95%配对证据组Bootstrap区间；方法差值的区间见上表。\n\n'
    report+='退化集中在6K/4与6K/16，12K及6K/8持平。证据缺失记录来自一个MuSiQue问题的两种长度、三种N和两方法，共12条生成。该问题仅在6K/16贡献一次损失（固定0.666667、动态0），另外五种条件持平。证据完整9题、54组配对中仍有四次损失、零次改善；该口径数据集分母为4/3/2，结果另表保留。\n\n'
    report+='## 4 冷请求效率及输出长度\n\n'
    def e(m,f):return next(r for r in eff if r['scope']=='all10' and r['L']=='ALL' and r['method']==m and r['metric']==f)
    report+=md(['方法','请求数','TTFT均值/秒','总响应均值/秒','总响应中位数/秒','总响应P90/秒','输出token均值'],[[m,60,f"{e(m,'ttft_seconds')['mean']:.4f}",f"{e(m,'total_seconds')['mean']:.4f}",f"{e(m,'total_seconds')['median']:.4f}",f"{e(m,'total_seconds')['p90']:.4f}",f"{e(m,'output_tokens')['mean']:.3f}"] for m in METHODS])
    report+='\n'+md(['指标','动态−固定/秒','95%配对区间','逐对加速比均值','均值耗时比'],[[r['metric'],f"{r['difference_dynamic_minus_fixed_s']:+.5f}",f"[{r['difference_ci_low_s']:.5f}, {r['difference_ci_high_s']:.5f}]",f"{r['paired_speedup_mean_fixed_over_dynamic']:.4f}",f"{r['ratio_of_means_fixed_over_dynamic']:.4f}"] for r in effpairs if r['scope']=='all10' and r['L']=='ALL'])
    report+='\n动态总响应均值减少17.72毫秒，95%配对区间为[-48.64, 2.56]毫秒；TTFT均值增加0.24毫秒。6K/4的平均输出由9.1降至4.2 token，总响应由1.259降至1.006秒；6K/16的平均输出由6.6增至9.1 token，总响应由1.108增至1.246秒。两个改参分支呈现相反的生成长度和时延变化。动态选参计时均值23.36微秒，固定参数分支为1.62微秒。\n\n'
    report+='生成上限为Qasper 128 token、HotpotQA及MuSiQue 32 token；120条记录中119条以EOS结束，1条达到上限（0.83%）。唯一上限记录为6K/4的证据缺失MuSiQue问题，固定方法重复输出并达到32 token，动态方法输出6 token，两者F1均为0；总响应分别为2.316与1.010秒。该次配对节省1.306秒，全60组净节省1.063秒，其余59组动态合计慢0.243秒。记录见generation_cap_records.csv，分组计数见generation_limits.csv。输出token数包含实际生成的EOS。\n\n'
    report+='按整个缺失证据问题移除六组条件后，完整9题的总响应均值仍由1.4131降至1.3949秒，差值-18.22毫秒，95%区间[-52.58, 3.37]毫秒，平均输出6.519降至6.130 token。该问题在6K/16还存在动态输出变长、耗时增加的反向变化。全量及完整子集均呈现小幅平均时间缩短，两个时间差区间均跨0，单题输出长度变化对均值贡献明显。\n\n'
    report+='\n加速比定义为固定耗时/动态耗时，大于1表示动态更快。总响应随输出长度变化，参数一致的40组保留相同输出，可单独查看时间记录；其余20组对应不同T/S，生成长度与质量一并记录。缓存构建/获取、选参、输出长度及峰值allocated/reserved的均值、中位数、P90、均值CI保存在efficiency_descriptive.csv。显存单位GiB=bytes/2^30，包含模型和存活缓存；模型加载另列，排除于请求时间之外。\n\n![长度时延比较](../figures/length_latency.png)\n\n'
    report+='## 5 统计、案例与本阶段结论\n\n'
    report+='使用seed20261116和5000次配对证据组Bootstrap，在每个数据集内抽原论文/证据组，保留同源问题的全部六种输入及两方法，参数与规则在抽样中固定。区间反映所选验证问题/证据组构成变化；独立单位数见bootstrap_units.csv。六十种输入条件来自十道题，各表明确配对数与独立问题数。\n\n'
    report+='我们按最大正差、最大负差及零差最小ID/L/N选案例。本次改善类型为0，退化和持平案例保存原答案、参考、参数及源行号，见selected_cases.csv；全部60组原答案在paired_records.csv。\n\n'
    report+=f"结论：冻结动态规则在本次新长度检查中没有出现质量提升的配对，全量宏F1较固定低{abs(summaries['all10']['difference_pp']):.3f}个百分点，证据完整子集低{abs(summaries['evidence_complete9']['difference_pp']):.3f}个百分点。主要问题出在6K输入映射到8K格后采用的两个改参分支；12K映射到16K格后参数与固定相同，答案持平。后续独立测试继续使用原冻结规则并按实际结果报告，当前不据此改参。\n\n"
    report+='后续补实验建议：修复所列MuSiQue问题的证据保留构造，按原冻结参数重跑两长度×三块数×两方法的12条记录，补齐完整10题口径；另在独立问题上增加6K/4和6K/16的固定/动态配对，检查两个退化分支的迁移表现。本轮仅登记建议。\n\n'
    report+='研究状态保留SUBMITTED，本次停在实验第四阶段。后续阶段先给简短规划，待用户审阅后执行。未安装依赖、运行GPU、重新生成答案或修改原始证据。\n'
    (OUT/'reports/LENGTH_CHECK_REPORT.md').write_text(report,encoding='utf-8')
    index='# 实验第四阶段：长度检查材料\n\n- [分析报告](reports/LENGTH_CHECK_REPORT.md)\n- [样本范围与分母](tables/scope_counts.csv)\n- [60组配对原分数、答案、参数与计时](tables/paired_records.csv)\n- [映射及相同输出检查](tables/mapping_and_output_checks.csv)\n- [宏平均、六格质量与95%区间](tables/quality_paired.csv)\n- [数据集分层质量](tables/dataset_cell_quality.csv)\n- [效率均值、中位数、P90及区间](tables/efficiency_descriptive.csv)\n- [配对耗时与加速比](tables/efficiency_paired.csv)\n- [证据缺失的六组配对](tables/evidence_missing_pairs.csv)\n- [原答案案例](tables/selected_cases.csv)\n- [重采样单位与簇数](tables/bootstrap_units.csv)\n- [模型加载记录](tables/model_load_workers.csv)\n\n图表均保存PNG/SVG；原冻结规则副本位于sources/。environment.json记录环境、随机种子和统计权重，STAGE_STATUS.json记录停止点。\n\n复现：\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_length_check.py\n```\n\n使用已有NumPy/Matplotlib。所有新文件限于pc-analysis-output；输入为length_check保存score/timing、前置审核标签、冻结规则和固定验证ID顺序。评分重算0次。\n'
    index=index.replace('- [模型加载记录]','- [生成上限及EOS计数](tables/generation_limits.csv)\n- [命中生成上限的源记录](tables/generation_cap_records.csv)\n- [模型加载记录]')
    (OUT/'README.md').write_text(index,encoding='utf-8')
    (BASE/'README.md').write_text('# APE MAIN80 分实验分析\n\n当前已完成实验第四阶段：6K/12K长度检查。研究状态SUBMITTED。\n\n- [实验4：长度检查](stages/04_length_check/README.md)\n- [实验3：验证与规则](stages/03_validation_rule/README.md)\n- [实验2：参数探索](stages/02_parameter_exploration/README.md)\n- [实验1：三方法基线及官方对照](stages/01_longbench_baseline/README.md)\n- [前置审核](stages/01_audit/README.md)\n\n本轮已停在实验4；后续阶段先提交简短规划，用户审阅后执行。原始证据、候选和规则保留。\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
