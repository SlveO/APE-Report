"""Independent CPU audit of frozen protocol, search, rules and applicability.
Run: python -B pc-analysis-output/scripts/analyze_protocol.py
Only writes within pc-analysis-output. No model imports/downloads or GPU work.
"""
from pathlib import Path
import csv, json, math, re, string
from collections import Counter, defaultdict
from itertools import product
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'pc-analysis-output'
REST=ROOT/'restored'
RUN=REST/'output/runs/main80-20261114T041503'
FROZEN=REST/'datasets/frozen-main80'
SNAP=RUN/'snapshot/datasets/frozen-main80'
SEED=20261116
BOOTSTRAPS=5000
DS=('qasper','hotpotqa','musique')
LENGTHS=(4096,8192,16384)
NS=(4,8,16)
PARAMS=list(product((.5,.7,.9,1.),repeat=2))
CELLS=list(product(LENGTHS,NS))
DEFAULT=(.9,.9)

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write_csv(name,rows):
    rows=list(rows)
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def norm(s):
    s=s.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def f1(answer,refs):
    p=Counter(norm(answer).split())
    def one(ref):
        g=Counter(norm(ref).split()); shared=sum((p & g).values())
        return 2*shared/(sum(p.values())+sum(g.values())) if shared else 0.
    return max((one(r) for r in refs),default=0.)
def tie(p): return ((p[0]-.9)**2+(p[1]-.9)**2,p[0],p[1])
def nearest(l,n): return min(CELLS,key=lambda c:((math.log2(l)-math.log2(c[0]))**2+(math.log2(n)-math.log2(c[1]))**2,*c))
def gid(r): return json.dumps(r['group'],ensure_ascii=False,sort_keys=True)
def stats_mean(v): return float(np.mean(v))


def clustered_macro_ci(rows,filters):
    # Resample evidence groups within each dataset; retain all lengths/N per sampled group.
    rows=[r for r in rows if filters(r)]
    strata=defaultdict(lambda:defaultdict(list))
    for r in rows: strata[r['dataset']][r['group']].append(r)
    rng=np.random.default_rng(SEED)
    values=[]
    for b in range(BOOTSTRAPS):
        means=[]
        for d,groups in sorted(strata.items()):
            keys=sorted(groups); selected=rng.integers(0,len(keys),size=len(keys)); cells=defaultdict(list)
            for ix in selected:
                for r in groups[keys[ix]]: cells[(r['L'],r['N'])].append(r['difference'])
            means.extend(float(np.mean(v)) for v in cells.values())
        values.append(float(np.mean(means)))
    return np.quantile(values,[.025,.975]).tolist()


def main():
    for p in ('tables','figures','reports'): (OUT/p).mkdir(parents=True,exist_ok=True)
    records=[]
    with (RUN/'attempts.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))
    success=[r for r in records if r['status']=='success']
    for r in success: r['score_recomputed']=f1(r['answer'],r['reference_answers']) if r['dataset'] in DS else None
    exploration=[r for r in success if r['key']['stage']=='explore']
    validation=[r for r in success if r['key']['stage']=='validation']
    candidates=read(RUN/'candidates.json'); frozen_rule=read(RUN/'rule.json')
    app=read(SNAP/'applicability.json')
    lb=read(SNAP/'longbench.json')
    frozen=read(SNAP/'frozen.json')
    split=read(SNAP/'split-report.json')
    index={r['sample_id']:r for r in lb}
    coverage=[]
    score_groups={stage:defaultdict(list) for stage in ('explore','validation')}
    id_groups={stage:defaultdict(set) for stage in ('explore','validation')}
    for stage,rows in (('explore',exploration),('validation',validation)):
        for r in rows:
            key=(r['dataset'],r['L'],r['N'],r['T'],r['S'])
            score_groups[stage][key].append(r['score_recomputed']); id_groups[stage][key].add(r['sample_id'])
        for key,values in sorted(score_groups[stage].items()):
            d,l,n,t,s=key
            coverage.append(dict(stage=stage,dataset=d,L=l,N=n,T=t,S=s,n_generations=len(values),n_unique_ids=len(id_groups[stage][key]),mean_f1_0_1=np.mean(values),unit='F1 0-1',source='attempts.jsonl: status=success; key.stage; dataset,L,N,T,S,answer,reference_answers'))
    for d,l,n in product(DS,LENGTHS,NS):
        expected_ids={r['sample_id'] for r in app if r['split']=='explore' and r['dataset']==d and r['L']==l and r['applicable']}
        assert all(id_groups['explore'][(d,l,n,*p)]==expected_ids for p in PARAMS), ('explore pairing',d,l,n)
        choices={DEFAULT,tuple(candidates['global']),tuple(candidates['cells'][f'{l}/{n}'])}
        validation_ids={r['sample_id'] for r in app if r['split']=='validation' and r['dataset']==d and r['L']==l and r['applicable']}
        assert all(id_groups['validation'][(d,l,n,*p)]==validation_ids for p in choices), ('validation pairing',d,l,n)
    write_csv('protocol_coverage.csv',coverage)
    def balanced(stage,p,cell=None):
        cells=[cell] if cell else CELLS
        return stats_mean([stats_mean(score_groups[stage][(d,l,n,*p)]) for d in DS for l,n in cells])
    global_best=min(PARAMS,key=lambda p:(-balanced('explore',p),tie(p)))
    cell_best={c:min(PARAMS,key=lambda p:(-balanced('explore',p,c),tie(p))) for c in CELLS}
    fixed=min({DEFAULT,global_best},key=lambda p:(-balanced('validation',p),tie(p)))
    assert tuple(candidates['global'])==global_best
    assert all(tuple(candidates['cells'][f'{l}/{n}'])==cell_best[(l,n)] for l,n in CELLS)
    assert tuple(frozen_rule['fixed'])==fixed
    grid=[]
    for l,n in CELLS:
        for t,s in PARAMS:
            grid.append(dict(L=l,N=n,T=t,S=s,dataset='dataset_equal_macro',mean_f1_0_1=balanced('explore',(t,s),(l,n)),mean_f1_percent=100*balanced('explore',(t,s),(l,n)),n_generations=sum(len(score_groups['explore'][(d,l,n,t,s)]) for d in DS),n_dataset_strata=3,unit='F1 0-1 / percent',aggregation='mean of three dataset means within L/N cell'))
    write_csv('protocol_explore_grid.csv',grid)
    write_csv('protocol_explore_global.csv',[dict(T=t,S=s,mean_f1_0_1=balanced('explore',(t,s)),mean_f1_percent=100*balanced('explore',(t,s)),selected=(t,s)==global_best,n_dataset_cells=27,unit='F1 0-1 / percent',aggregation='equal 3 datasets and equal 9 cells') for t,s in PARAMS])
    write_csv('protocol_validation_global.csv',[dict(label=label,T=p[0],S=p[1],mean_f1_0_1=balanced('validation',p),mean_f1_percent=100*balanced('validation',p),selected=p==fixed,n_dataset_cells=27,unit='F1 0-1 / percent',aggregation='equal 3 datasets and equal 9 cells') for label,p in [('default',DEFAULT),('exploration_global',global_best)]])
    rules=[]
    for l,n in CELLS:
        p=cell_best[(l,n)]; candidate_score=balanced('validation',p,(l,n)); fixed_score=balanced('validation',fixed,(l,n)); chosen=p if candidate_score>fixed_score else fixed
        assert tuple(frozen_rule['cells'][f'{l}/{n}'])==chosen
        rules.append(dict(L=l,N=n,exploration_candidate_T=p[0],exploration_candidate_S=p[1],exploration_candidate_f1=balanced('explore',p,(l,n)),validation_candidate_f1=candidate_score,validation_fixed_f1=fixed_score,validation_difference_f1=candidate_score-fixed_score,chosen_T=chosen[0],chosen_S=chosen[1],fallback=candidate_score<=fixed_score,n_explore_qasper=len(score_groups['explore'][('qasper',l,n,*p)]),n_explore_hotpotqa=len(score_groups['explore'][('hotpotqa',l,n,*p)]),n_explore_musique=len(score_groups['explore'][('musique',l,n,*p)]),n_validation_qasper=len(score_groups['validation'][('qasper',l,n,*p)]),n_validation_hotpotqa=len(score_groups['validation'][('hotpotqa',l,n,*p)]),n_validation_musique=len(score_groups['validation'][('musique',l,n,*p)]),unit='F1 0-1',aggregation='equal dataset weight within cell',source='snapshot rules.py protocol independently recomputed from successful attempts'))
    write_csv('protocol_rule.csv',rules)
    write_csv('protocol_applicability.csv',[dict(sample_id=r['sample_id'],dataset=r['dataset'],split=r['split'],L=r['L'],applicable=r['applicable'],reason=r.get('reason',''),available_tokens=r.get('available_tokens',''),source='snapshot/datasets/frozen-main80/applicability.json') for r in app])
    appcounts=[]
    for key,n in sorted(Counter((r['split'],r['dataset'],r['L'],r['applicable'],r.get('reason','')) for r in app).items()):
        spl,d,l,applicable,reason=key; appcounts.append(dict(split=spl,dataset=d,L=l,applicable=applicable,reason=reason,n_inputs=n,unit='sample-length input'))
    write_csv('protocol_applicability_counts.csv',appcounts)
    labels=Counter(tuple(r['logical_labels']) for r in validation)
    write_csv('protocol_validation_reuse.csv',[dict(logical_labels='|'.join(k),n_generations=v,n_logical_labels=len(k),n_logical_evaluations=v*len(k),unit='generations / evaluations') for k,v in sorted(labels.items())])
    counts=Counter(r['key']['stage'] for r in records)
    logical_validation=sum(len(r['logical_labels']) for r in validation)
    scope_rows=[dict(stage=k,n_generations=v,n_planned_full={'baseline':1800,'explore':7200,'validation':1080,'length_check':120,'attention':12,'longbench_final':1020,'litm':4500,'doc2dial':3000}[k],unit='generation',explanation={'explore':'139 applicable sample-length inputs x 3 N x 16 T/S; 11 inapplicable x 3 x16 =528 omitted before generation','validation':f'119 applicable sample-length inputs x 3 N x 3 logical candidates = {logical_validation}; actual parameter reuse saves {logical_validation-v}; 1 inapplicable sample-length input removes9 logical evaluations','attention':'12 dedicated diagnostics; not included in formal timing quality comparisons','length_check':'first10 fixed validation IDs x 2 lengths x 3 N x 2 methods'}.get(k,'complete frozen matrix')) for k,v in counts.items()]
    write_csv('protocol_stage_counts.csv',scope_rows)
    overlap_rows=[]; overlap_samples=[]
    for a,b in (('explore','validation'),('explore','test'),('validation','test')):
        left=[r for r in lb if r['split']==a]; right=[r for r in lb if r['split']==b]
        for kind,field in (('single_document','input_documents'),('support_document','support_documents')):
            aset={d for r in left for d in r[field]}; bset={d for r in right for d in r[field]}; shared=aset & bset
            assert shared==set(split[f'{a}/{b}'][kind+'_overlap'])
            affected=[r for r in right if set(r[field]) & shared]
            overlap_rows.append(dict(left_split=a,right_split=b,overlap_type=kind,n_unique_documents=len(shared),n_affected_right_ids=len(affected),denominator_right_ids=len(right),affected_fraction=len(affected)/len(right),unit='documents / samples / fraction',source='snapshot longbench.json + split-report.json'))
            for r in affected:
                overlap_samples.append(dict(left_split=a,right_split=b,overlap_type=kind,dataset=r['dataset'],sample_id=r['sample_id'],shared_documents=json.dumps(sorted(set(r[field])&shared),ensure_ascii=False)))
        group_overlap={gid(r) for r in left}&{gid(r) for r in right}; question_overlap={' '.join(r['question'].casefold().split()) for r in left}&{' '.join(r['question'].casefold().split()) for r in right}
        assert not group_overlap and not question_overlap
        overlap_rows.extend([dict(left_split=a,right_split=b,overlap_type='evidence_group',n_unique_documents=0,n_affected_right_ids=0,denominator_right_ids=len(right),unit='group'),dict(left_split=a,right_split=b,overlap_type='normalized_question',n_unique_documents=0,n_affected_right_ids=0,denominator_right_ids=len(right),unit='question')])
    write_csv('protocol_split_overlap.csv',overlap_rows); write_csv('protocol_overlap_samples.csv',overlap_samples)
    # Unique request identity, excluding method and Doc2Dial hot/cold repetitions.
    mapgroups=defaultdict(list); mapindividual=[]
    for stage,rows in [('baseline',[r for r in success if r['key']['stage']=='baseline' and r['method']=='sequential']),('longbench_common510',[r for r in success if r['key']['stage']=='longbench_final' and r['method']=='ape_dynamic']),('litm',[r for r in success if r['key']['stage']=='litm' and r['method']=='ape_dynamic']),('doc2dial',[r for r in success if r['key']['stage']=='doc2dial' and r['method']=='ape_dynamic' and r['cache_mode']=='cold'])]:
        for r in rows:
            l,n=nearest(r['L'],r['N']); chosen=tuple(frozen_rule['cells'][f'{l}/{n}'])
            key=(stage,r['dataset'],r['N'],l,n,*chosen)
            mapgroups[key].append(r)
            mapindividual.append(dict(scope=stage,dataset=r['dataset'],sample_id=r['sample_id'],position=r.get('position'),L=r['L'],actual_N=r['N'],original_document_count=r['original_document_count'],mapped_L=l,mapped_N=n,rule_T=chosen[0],rule_S=chosen[1],out_of_N_range=r['N']<4 or r['N']>16,out_of_L_range=r['L']<4096 or r['L']>16384))
    write_csv('protocol_mapping_samples.csv',mapindividual)
    write_csv('protocol_mapping.csv',[dict(scope=k[0],dataset=k[1],actual_N=k[2],mapped_L=k[3],mapped_N=k[4],T=k[5],S=k[6],n_inputs=len(v),L_min=min(r['L'] for r in v),L_max=max(r['L'] for r in v),n_outside_N_range=sum(r['N']<4 or r['N']>16 for r in v),n_outside_L_range=sum(r['L']<4096 or r['L']>16384 for r in v),unit='sample-position/request inputs',source='attempts.jsonl top-level L,N and frozen rule; one method and cold Doc2Dial') for k,v in sorted(mapgroups.items())])
    # Length checks retain six correlated constructed inputs per source question.
    paired=defaultdict(dict)
    for r in success:
        if r['key']['stage']=='length_check': paired[(r['sample_id'],r['L'],r['N'])][r['method']]=r
    lc=[]
    for key,methods in sorted(paired.items()):
        assert set(methods)=={'ape_fixed','ape_dynamic'}
        fixed_r=methods['ape_fixed']; dynamic_r=methods['ape_dynamic']; l,n=nearest(dynamic_r['L'],dynamic_r['N'])
        assert (dynamic_r['T'],dynamic_r['S'])==tuple(frozen_rule['cells'][f'{l}/{n}'])
        lc.append(dict(sample_id=key[0],dataset=fixed_r['dataset'],L=key[1],N=key[2],group=gid(fixed_r),fixed_f1=fixed_r['score_recomputed'],dynamic_f1=dynamic_r['score_recomputed'],difference=dynamic_r['score_recomputed']-fixed_r['score_recomputed'],fixed_T=fixed_r['T'],fixed_S=fixed_r['S'],dynamic_T=dynamic_r['T'],dynamic_S=dynamic_r['S']))
    write_csv('protocol_length_check_samples.csv',lc)
    lcsummary=[]
    for d in ('all',*DS):
        for l in ('all',6144,12288):
            for n in ('all',4,8,16):
                filt=lambda r,d=d,l=l,n=n:(d=='all' or r['dataset']==d) and (l=='all' or r['L']==l) and (n=='all' or r['N']==n)
                rows=[r for r in lc if filt(r)]
                if not rows: continue
                ag=defaultdict(list)
                for r in rows: ag[(r['dataset'],r['L'],r['N'])].append(r)
                fixed_mean=stats_mean([stats_mean([r['fixed_f1'] for r in rr]) for rr in ag.values()]); dynamic_mean=stats_mean([stats_mean([r['dynamic_f1'] for r in rr]) for rr in ag.values()])
                lo,hi=clustered_macro_ci(lc,filt)
                lcsummary.append(dict(dataset=d,L=l,N=n,n_pairs=len(rows),n_source_ids=len({r['sample_id'] for r in rows}),n_evidence_groups=len({r['group'] for r in rows}),fixed_f1_0_1=fixed_mean,dynamic_f1_0_1=dynamic_mean,difference_f1_0_1=dynamic_mean-fixed_mean,ci95_low=lo,ci95_high=hi,wins=sum(r['difference']>1e-12 for r in rows),ties=sum(abs(r['difference'])<=1e-12 for r in rows),losses=sum(r['difference']<-1e-12 for r in rows),unit='F1 0-1',aggregation='equal dataset and available length/N cell means',resampling='paired evidence groups within dataset; all length/N variants retained',seed=SEED,bootstrap_replicates=BOOTSTRAPS,source='length_check success paired by sample_id,L,N; independent LongBench F1'))
    write_csv('protocol_length_check.csv',lcsummary)
    # 9 heatmaps use a common color scale.
    fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
    values=[r['mean_f1_percent'] for r in grid]; vmin=min(values); vmax=max(values)
    for ax,(l,n) in zip(axes.flat,CELLS):
        data=np.array([[100*balanced('explore',(t,s),(l,n)) for s in (.5,.7,.9,1.)] for t in (.5,.7,.9,1.)])
        im=ax.imshow(data,vmin=vmin,vmax=vmax,cmap='viridis')
        ax.set_xticks(range(4),['0.5','0.7','0.9','1.0']); ax.set_yticks(range(4),['0.5','0.7','0.9','1.0']); ax.set_xlabel('S'); ax.set_ylabel('T'); ax.set_title(f'L={l}, N={n}')
        for i,j in product(range(4),repeat=2): ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if data[i,j]<(vmin+vmax)/2 else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Dataset-equal macro F1 (%)',shrink=.8)
    fig.suptitle('Exploration: 16 T/S settings in each frozen L/N cell')
    for ext in ('png','svg'): fig.savefig(OUT/'figures'/f'ts_heatmaps.{ext}',dpi=180)
    plt.close(fig)
    # Human readable audit fragment, carrying the exact table-level findings.
    summary_lc=next(r for r in lcsummary if r['dataset']=='all' and r['L']=='all' and r['N']=='all')
    lines=['# 协议、数据划分与动态规则审核','',f'分析来源为 campaign `{RUN.name}` 的启动快照、逐题成功记录与冻结产物。CPU复算随机种子 `{SEED}`，长度检查采用 {BOOTSTRAPS} 次证据组配对重采样。','', '## 1 数量与候选复用','',f'八阶段共 {len(records):,} 次生成。探索阶段有效139个题目×长度输入，乘3种N与16种T/S得到6,672次；11个不适用输入移除528次原计划生成。验证阶段119个有效题长输入，三候选逻辑评价共{logical_validation}次，实际参数复用后{len(validation)}次，节省{logical_validation-len(validation)}次生成；一个16K不适用输入移除9个逻辑评价。','', '全部90个开发问题×5种长度共450个构造中，432个有效、18个不适用；其中网格内探索11项、验证1项，额外6K/12K探索6项。适用性在生成前冻结，按构造失败原因排除。各格16种参数覆盖完整，参数间使用相同有效ID；各数据集有效样本数详见 `tables/protocol_rule.csv`。','', '## 2 探索候选、验证回退与冻结','',f'独立按3个数据集、9个格点等权复算，探索全局最佳为T={global_best[0]}、S={global_best[1]}；验证从默认与全局最佳选出的固定参数为T={fixed[0]}、S={fixed[1]}。独立候选与九格规则逐项吻合已保存JSON。','', '| L | N | 探索格点候选T/S | 验证候选F1×100 | 固定F1×100 | 冻结T/S | 回退 |','|---:|---:|---|---:|---:|---|---|']
    for r in rules: lines.append(f"|{r['L']}|{r['N']}|{r['exploration_candidate_T']}/{r['exploration_candidate_S']}|{r['validation_candidate_f1']*100:.4f}|{r['validation_fixed_f1']*100:.4f}|{r['chosen_T']}/{r['chosen_S']}|{'是' if r['fallback'] else '否'}|")
    lines+=['',f"候选冻结时间为 {candidates.get('frozen_at')}；规则冻结时间为 {frozen_rule.get('frozen_at')}，产物声明 `frozen_before={frozen_rule.get('frozen_before')}`。后续长度检查和独立测试沿用该规则。",'', '## 3 资料重合与影响范围','', '| 分割对 | 重合类型 | 重合资料数 | 右侧受影响ID/总ID |','|---|---|---:|---:|']
    for r in overlap_rows:
        if r['overlap_type'] in ('single_document','support_document'): lines.append(f"|{r['left_split']}/{r['right_split']}|{r['overlap_type']}|{r['n_unique_documents']}|{r['n_affected_right_ids']}/{r['denominator_right_ids']}|")
    lines+=['','完整证据组和规范化问题的跨分割重合均为0；单篇资料及支撑资料存在交集。当前分割隔离了完整证据组合，部分Wikipedia资料仍跨开发与测试出现，测试结果应结合这些复用资料阅读。`protocol_overlap_samples.csv`保留受影响的具体ID与共享资料。','', '## 4 实际编码块数及网格映射','', 'N按真正送入编码的token块统计；`original_document_count`记录原始资料数量。正式LongBench完整文档按7500 tokens分块；LITM20篇短资料拼接后多数仍构成一个编码块。规则在log2(L)、log2(N)空间选最近格，N小于4的输入全部使用N=4列。','', '正式外测各输入的实际N和映射统计见 `protocol_mapping.csv`；原始资料数、L与每条映射见 `protocol_mapping_samples.csv`。开发网格的N=8/16列在正式LongBench、LITM和Doc2Dial的实际块数下未被使用，正式质量结果主要评价N=4列的边界映射行为。后续若要覆盖九格规则，应追加真正具有4/8/16个编码块且与开发证据组分开的正式配对输入。','', '## 5 冻结后的长度检查','',f"固定首10个验证ID×2个额外长度×3个N形成60个配对输入，共120次生成。按数据集和长度/N格等权，固定F1×100={summary_lc['fixed_f1_0_1']*100:.4f}，动态={summary_lc['dynamic_f1_0_1']*100:.4f}，差={summary_lc['difference_f1_0_1']*100:+.4f}个百分点，95%区间[{summary_lc['ci95_low']*100:.4f}, {summary_lc['ci95_high']*100:.4f}]。以证据组为重采样单位，同一来源问题的6种构造共同保留；该区间反映这10个已用验证问题之间的样本变动。",'', '长度检查沿用验证来源问题，用于观察冻结规则对6K/12K插值长度的行为。独立测试质量由LongBench test、LITM和Doc2Dial分别评价。','', '## 6 复现','', '```powershell','python -B pc-analysis-output/scripts/analyze_protocol.py','```','', '输出使用恢复后的启动快照作为协议与数据来源；候选和冻结规则由本脚本独立实现的等权聚合与平分顺序复算。未读取测试得分来选参。']
    (OUT/'reports/PROTOCOL_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(records),explore=len(exploration),validation=len(validation),logical_validation=logical_validation,global_best=global_best,fixed=fixed,length_check=summary_lc),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
