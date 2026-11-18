"""Experiment 3: saved validation scores, candidate reuse, selection and frozen rule.

No F1 re-scoring, model generation or test-score use. Sensitivity rule replays are
written only into this analysis stage and never replace the frozen campaign rule.
"""
from pathlib import Path
from collections import defaultdict,Counter
from itertools import product
from statistics import mean
from datetime import datetime
import csv,json,os,shutil,sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/03_validation_rule'
RUN=ROOT/'restored/output/runs/main80-20261114T041503'
AUDIT=BASE/'stages/01_audit'
for folder in ('tables','reports','figures','sources','cache','tmp','logs'):(OUT/folder).mkdir(parents=True,exist_ok=True)
for key in ('MPLCONFIGDIR','XDG_CACHE_HOME','TEMP','TMP'):os.environ[key]=str(OUT/('cache' if key in ('MPLCONFIGDIR','XDG_CACHE_HOME') else 'tmp'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DS=('qasper','hotpotqa','musique')
CELLS=list(product((4096,8192,16384),(4,8,16)))
DEFAULT=(.9,.9)
SEED=20261116
BOOT=5000
POLICIES=('default','global_candidate','cell_candidates','selected_fixed','original_frozen_rule','scope_replay_rule')
LABEL={'default':'默认0.9/0.9','global_candidate':'探索全局0.7/1.0','cell_candidates':'全部逐格候选','selected_fixed':'所选固定参数','original_frozen_rule':'原冻结九格规则','scope_replay_rule':'本口径回放规则'}
SCOPE_LABEL={'all_27':'全部验证输入','complete_27':'证据完整子集','common_complete_27':'三长度共同完整22题'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def csvout(name,rows):
    rows=list(rows)
    if not rows:return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def tie(p):return ((p[0]-.9)**2+(p[1]-.9)**2,p[0],p[1])
def ci(a):return np.quantile(a,[.025,.975],axis=0)
def md(cols,rows):return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)+'\n'

def bootstrap_scope(inputs,policy_values,scope_index):
    """Dataset-stratified evidence-group resampling; every L/N and role retained.

    Replicates require nonempty denominators for each predeclared dataset/cell.
    Rejected empty-cell draws are counted. Identical shared generations remain
    identical across all policy views; no role is sampled independently.
    """
    rng=np.random.default_rng(np.random.SeedSequence([SEED,scope_index]))
    dataset_draws=[];diagnostics=[]
    for d in DS:
        keys=[k for k in inputs if k[0]==d]
        gids={k:json.dumps(inputs[k]['group'],sort_keys=True,ensure_ascii=False) for k in keys}
        groups=sorted(set(gids.values()));lookup={g:i for i,g in enumerate(groups)}
        counts=np.zeros((len(groups),9));sums=np.zeros((len(groups),9,len(POLICIES)))
        for k in keys:
            g=lookup[gids[k]];c=CELLS.index((k[2],k[3]));counts[g,c]+=1;sums[g,c]+=policy_values[k]
        assert (counts.sum(axis=0)>0).all()
        accepted=[];naccepted=0;proposed=0;rejected=0
        while naccepted<BOOT:
            size=min(100,BOOT-naccepted)
            idx=rng.integers(0,len(groups),size=(size,len(groups)))
            den=counts[idx].sum(axis=1);num=sums[idx].sum(axis=1)
            valid=(den>0).all(axis=1);rejected+=int((~valid).sum());proposed+=size
            if valid.any():
                draw=num[valid]/den[valid,:,None];accepted.append(draw);naccepted+=len(draw)
        dataset_draws.append(np.concatenate(accepted))
        diagnostics.append({'scope':list(SCOPE_LABEL)[scope_index],'dataset':d,'unique_ids':len({k[1] for k in keys}),'clusters':len(groups),'input_conditions':len(keys),'min_ids_per_cell':int(counts.sum(axis=0).min()),'accepted_replicates':BOOT,'proposed_replicates':proposed,'rejected_empty_cell_draws':rejected,'seed':SEED,'rng_stream':'SeedSequence([seed, scope_index]); datasets drawn in qasper/hotpotqa/musique order','unit':'ID / evidence cluster / draw count'})
    return np.mean(dataset_draws,axis=0),diagnostics

def main():
    candidates=read(RUN/'candidates.json');original_rule=read(RUN/'rule.json');global_param=tuple(candidates['global'])
    cell_param={c:tuple(candidates['cells'][f'{c[0]}/{c[1]}']) for c in CELLS}
    for name in ('candidates.json','rule.json'):
        dest=OUT/'sources'/('original_'+name)
        if not dest.exists():shutil.copyfile(RUN/name,dest)
    evidence={(r['sample_id'],int(r['L'])):r for r in rcsv(AUDIT/'tables/construction_checks.csv') if r['split']=='validation' and int(r['L']) in (4096,8192,16384)}
    physical=[];by_input=defaultdict(dict)
    with (RUN/'attempts.jsonl').open(encoding='utf-8') as stream:
        for line,text in enumerate(stream,1):
            r=json.loads(text)
            if r['key']['stage']!='validation':continue
            assert r['status']=='success' and r['split']=='validation'
            r['_line']=line;r['_complete']=int(evidence[(r['sample_id'],r['L'])]['n_missing_required_tokens'])==0
            k=(r['dataset'],r['sample_id'],r['L'],r['N']);p=(r['T'],r['S'])
            assert p not in by_input[k]
            by_input[k][p]=r;physical.append(r)
    assert len(physical)==992 and len(by_input)==357
    logical=[];reuse=Counter()
    for r in physical:
        labels=tuple(sorted(r['logical_labels']));reuse['|'.join(labels)]+=1
        for role in labels:logical.append({'role':role,'physical_source_line':r['_line'],'dataset':r['dataset'],'sample_id':r['sample_id'],'L':r['L'],'N':r['N'],'T':r['T'],'S':r['S'],'saved_score_0_1':r['score'],'evidence_complete':r['_complete'],'unit':'logical role referencing one physical generation'})
    assert len(logical)==1071 and Counter(r['role'] for r in logical)=={'default':357,'global':357,'cell':357}
    views={}
    for k,params in by_input.items():
        want={'default':DEFAULT,'global':global_param,'cell':cell_param[(k[2],k[3])]}
        assert set(params)==set(want.values())
        for role,p in want.items():assert role in params[p]['logical_labels']
        sample=params[DEFAULT]
        views[k]={'complete':sample['_complete'],'group':sample['group'],'params':params}
    common={d:set.intersection(*({k[1] for k,v in views.items() if k[0]==d and k[2]==l and v['complete']} for l in (4096,8192,16384))) for d in DS}
    scopes={'all_27':views,'complete_27':{k:v for k,v in views.items() if v['complete']},'common_complete_27':{k:v for k,v in views.items() if k[1] in common[k[0]]}}
    global_rows=[];rule_rows=[];point_rows=[];scope_rows=[];coverage=[];bootdiag=[];rules={};policy_by_scope={};boot_by_scope={};point_by_scope={};fixed_by_scope={};paired_counts=[]
    for scope_index,(scope,inputs) in enumerate(scopes.items()):
        grouped=defaultdict(list)
        for k in inputs:grouped[(k[0],k[2],k[3])].append(k)
        assert len(grouped)==27
        def pmean(p,cell=None):
            cells=[cell] if cell is not None else CELLS
            return mean(mean(by_input[k][p]['score'] for k in grouped[(d,l,n)]) for d,(l,n) in product(DS,cells))
        fixed=min({DEFAULT,global_param},key=lambda p:(-pmean(p),tie(p)))
        fixed_by_scope[scope]=fixed
        selected_cells={}
        for cell in CELLS:
            cp=cell_param[cell];cv=pmean(cp,cell);fv=pmean(fixed,cell)
            selected_cells[cell]=cp if cv>fv else fixed
            dcounts=[len(grouped[(d,*cell)]) for d in DS]
            vals=[]
            for d in DS:
                for k in grouped[(d,*cell)]:vals.append(by_input[k][cp]['score']-by_input[k][fixed]['score'])
            decision='same_as_fixed' if cp==fixed else 'use_cell_candidate' if cv>fv else 'tie_fallback' if cv==fv else 'fallback_to_fixed'
            rule_rows.append({'scope':scope,'L':cell[0],'N':cell[1],'candidate_T':cp[0],'candidate_S':cp[1],'candidate_f1_percent':cv*100,'fixed_T':fixed[0],'fixed_S':fixed[1],'fixed_f1_percent':fv*100,'delta_candidate_minus_fixed_pp':(cv-fv)*100,'decision':decision,'selected_T':selected_cells[cell][0],'selected_S':selected_cells[cell][1],'n_ids':sum(dcounts),'n_qasper':dcounts[0],'n_hotpotqa':dcounts[1],'n_musique':dcounts[2],'win_candidate':sum(x>1e-12 for x in vals),'tie':sum(abs(x)<=1e-12 for x in vals),'loss_candidate':sum(x< -1e-12 for x in vals),'unit':'F1 x100; difference in percentage points; paired ID counts'})
            for d,count in zip(DS,dcounts):coverage.append({'scope':scope,'dataset':d,'L':cell[0],'N':cell[1],'n_ids_per_candidate':count,'roles':3,'physical_configs':len({DEFAULT,global_param,cp}),'paired_ids_same_for_candidates':True,'unit':'unique ID / logical role / generated configuration'})
        rules[scope]={'fixed':list(fixed),'cells':{f'{l}/{n}':list(p) for (l,n),p in selected_cells.items()},'purpose':'validation replay; original campaign rule retained'}
        values={}
        for k in inputs:
            cell=(k[2],k[3]);params=(DEFAULT,global_param,cell_param[cell],fixed,tuple(original_rule['cells'][f'{cell[0]}/{cell[1]}']),selected_cells[cell])
            values[k]=np.array([by_input[k][p]['score'] for p in params])
            point_rows.append({'scope':scope,'dataset':k[0],'sample_id':k[1],'L':k[2],'N':k[3],'evidence_complete':views[k]['complete'],'group':views[k]['group'],**{policy:float(v) for policy,v in zip(POLICIES,values[k])},**{policy+'_source_line':by_input[k][p]['_line'] for policy,p in zip(POLICIES,params)},'unit':'saved F1 0-1; policy views retain shared source-line covariance'})
        policy_by_scope[scope]=values
        point=np.array([[mean(values[k][j] for k in grouped[(d,l,n)]) for j in range(len(POLICIES))] for d,(l,n) in product(DS,CELLS)]).reshape(3,9,len(POLICIES)).mean(axis=0)
        draws,diag=bootstrap_scope(inputs,values,scope_index);bootdiag.extend(diag);boot_by_scope[scope]=draws;point_by_scope[scope]=point
        for j,policy in enumerate(POLICIES):
            lo,hi=ci(draws[:,:,j].mean(axis=1));delta=draws[:,:,j].mean(axis=1)-draws[:,:,POLICIES.index('selected_fixed')].mean(axis=1);dl,dh=ci(delta)
            global_rows.append({'scope':scope,'scope_label':SCOPE_LABEL[scope],'policy':policy,'policy_label':LABEL[policy],'f1_0_1':float(point[:,j].mean()),'f1_percent':float(point[:,j].mean()*100),'ci_low_percent':float(lo*100),'ci_high_percent':float(hi*100),'delta_vs_selected_fixed_pp':float((point[:,j]-point[:,POLICIES.index('selected_fixed')]).mean()*100),'delta_ci_low_pp':float(dl*100),'delta_ci_high_pp':float(dh*100),'n_input_conditions':len(inputs),'n_dataset_cells':27,'n_unique_ids':len({k[1] for k in inputs}),'unit':'F1 x100; paired difference percentage points','seed':SEED,'replicates':BOOT,'aggregation':'equal 3 datasets and equal 9 cells'})
        for rr in [r for r in rule_rows if r['scope']==scope]:
            ix=CELLS.index((rr['L'],rr['N']));delta=draws[:,ix,POLICIES.index('cell_candidates')]-draws[:,ix,POLICIES.index('selected_fixed')];lo,hi=ci(delta);rr.update(delta_ci_low_pp=float(lo*100),delta_ci_high_pp=float(hi*100))
        for pa,pb in [('default','global_candidate'),('selected_fixed','cell_candidates'),('selected_fixed','original_frozen_rule'),('selected_fixed','scope_replay_rule')]:
            ia,ib=POLICIES.index(pa),POLICIES.index(pb);delta=draws[:,:,ib].mean(axis=1)-draws[:,:,ia].mean(axis=1);lo,hi=ci(delta)
            diffs=[values[k][ib]-values[k][ia] for k in inputs]
            paired_counts.append({'scope':scope,'policy_a':pa,'policy_b':pb,'n_condition_pairs':len(inputs),'n_unique_ids':len({k[1] for k in inputs}),'delta_b_minus_a_pp':float((point[:,ib]-point[:,ia]).mean()*100),'ci_low_pp':float(lo*100),'ci_high_pp':float(hi*100),'win_b':int(sum(x>1e-12 for x in diffs)),'tie':int(sum(abs(x)<=1e-12 for x in diffs)),'loss_b':int(sum(x< -1e-12 for x in diffs)),'unit':'macro F1 percentage points; paired input-condition counts','seed':SEED,'replicates':BOOT})
        phys=[r for r in physical if (r['dataset'],r['sample_id'],r['L'],r['N']) in inputs]
        scope_rows.append({'scope':scope,'label':SCOPE_LABEL[scope],'physical_generations':len(phys),'input_conditions':len(inputs),'query_length_inputs':len({(k[1],k[2]) for k in inputs}),'unique_ids':len({k[1] for k in inputs}),'n_qasper_ids':len({k[1] for k in inputs if k[0]=='qasper'}),'n_hotpotqa_ids':len({k[1] for k in inputs if k[0]=='hotpotqa'}),'n_musique_ids':len({k[1] for k in inputs if k[0]=='musique'}),'logical_evaluations':len(inputs)*3,'reuse_savings':len(inputs)*3-len(phys),'unit':'physical generation / input condition / ID / logical evaluation count'})
    assert rules['all_27']['fixed']==original_rule['fixed'] and rules['all_27']['cells']==original_rule['cells']
    saved_checks=[]
    for r in rule_rows:
        if r['scope']=='all_27':
            saved=original_rule['validation_scores'][f"{r['L']}/{r['N']}"]
            for label,field in [('candidate','candidate_f1_percent'),('fixed','fixed_f1_percent')]:
                difference=r[field]/100-saved[label];assert abs(difference)<1e-12
                saved_checks.append({'L':r['L'],'N':r['N'],'role':label,'saved_0_1':saved[label],'aggregated_saved_scores_0_1':r[field]/100,'difference':difference,'rule_matches':True})
    changes=[]
    for cell in CELLS:
        raw=tuple(original_rule['cells'][f'{cell[0]}/{cell[1]}'])
        clean=tuple(rules['complete_27']['cells'][f'{cell[0]}/{cell[1]}']);commonp=tuple(rules['common_complete_27']['cells'][f'{cell[0]}/{cell[1]}'])
        changes.append({'L':cell[0],'N':cell[1],'original_T':raw[0],'original_S':raw[1],'complete_T':clean[0],'complete_S':clean[1],'complete_changes_original':clean!=raw,'common_T':commonp[0],'common_S':commonp[1],'common_changes_original':commonp!=raw})
    new_candidates=[]
    for r in rcsv(BASE/'stages/02_parameter_exploration/tables/candidate_sensitivity.csv'):
        l,n=int(r['L']),int(r['N']);p=(float(r['complete_T']),float(r['complete_S']))
        kk=[k for k in views if (k[2],k[3])==(l,n)];clean=[k for k in kk if views[k]['complete']]
        new_candidates.append({'L':l,'N':n,'complete_exploration_T':p[0],'complete_exploration_S':p[1],'original_exploration_T':float(r['original_T']),'original_exploration_S':float(r['original_S']),'n_validation_ids':len(kk),'n_available_scores':sum(p in by_input[k] for k in kk),'n_missing_generation_configs':sum(p not in by_input[k] for k in kk),'n_complete_ids':len(clean),'n_complete_missing_configs':sum(p not in by_input[k] for k in clean),'unit':'validation ID/configuration pairs','scope':'coverage of stage2 complete-subset candidates only; no score imputation'})
    timeline=[r for r in rcsv(AUDIT/'tables/runtime_stage_timeline.csv') if r['event'] in ('exploration_candidates_frozen','dynamic_rule_frozen') or (r.get('stage') in ('validation','length_check','attention','longbench_final') and r['event']=='worker_started')]
    dt=lambda s:datetime.fromisoformat(s)
    validation_start=min(dt(r['started_at']) for r in physical);validation_end=max(dt(r['ended_at']) for r in physical)
    assert dt(candidates['frozen_at'])<validation_start<=validation_end<dt(original_rule['frozen_at'])
    after_starts=[dt(r['at_utc']) for r in timeline if r['event']=='worker_started' and r['stage'] in ('length_check','attention','longbench_final')]
    assert all(dt(original_rule['frozen_at'])<t for t in after_starts)
    for name,data in [('scope_counts.csv',scope_rows),('logical_candidate_views.csv',logical),('generation_reuse.csv',[{'labels':k,'physical_generations':v,'logical_evaluations':v*len(k.split('|')),'saved_generations':v*(len(k.split('|'))-1)} for k,v in sorted(reuse.items())]),('candidate_pairing_coverage.csv',coverage),('validation_policy_scores.csv',global_rows),('validation_policy_paired.csv',paired_counts),('validation_cell_decisions.csv',rule_rows),('input_policy_scores.csv',point_rows),('saved_rule_checks.csv',saved_checks),('rule_sensitivity.csv',changes),('new_exploration_candidate_coverage.csv',new_candidates),('bootstrap_diagnostics.csv',bootdiag),('freeze_timeline.csv',timeline)]:csvout(name,data)
    (OUT/'sensitivity_rule_replays.json').write_text(json.dumps(rules,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    show=('default','global_candidate','cell_candidates','scope_replay_rule')
    fig,axs=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
    for ax,scope in zip(axs,scopes):
        rr=[next(r for r in global_rows if r['scope']==scope and r['policy']==p) for p in show]
        y=np.array([r['f1_percent'] for r in rr]);err=np.array([[r['f1_percent']-r['ci_low_percent'] for r in rr],[r['ci_high_percent']-r['f1_percent'] for r in rr]])
        ax.bar(range(4),y,color=['#4c78a8','#72b7b2','#eeca3b','#f58518']);ax.errorbar(range(4),y,yerr=err,fmt='none',ecolor='black',capsize=3)
        ax.set_xticks(range(4),['默认','全局候选','全部格点候选','回放规则'],rotation=25);ax.set_title(SCOPE_LABEL[scope]);ax.set_ylabel('F1 ×100')
    fig.suptitle('验证集候选与规则：均值及95%配对聚类区间');save(fig,'validation_policy_comparison')
    fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    bound=max(abs(r['delta_candidate_minus_fixed_pp']) for r in rule_rows)
    for ax,scope in zip(axs,scopes):
        rr=[r for r in rule_rows if r['scope']==scope];lookup={(r['L'],r['N']):r for r in rr}
        mat=np.array([[lookup[(l,n)]['delta_candidate_minus_fixed_pp'] for n in (4,8,16)] for l in (4096,8192,16384)])
        im=ax.imshow(mat,cmap='RdBu_r',vmin=-bound,vmax=bound)
        for i,l in enumerate((4096,8192,16384)):
            for j,n in enumerate((4,8,16)):
                ax.text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',color='white' if abs(mat[i,j])>bound*.55 else 'black')
                if lookup[(l,n)]['decision']=='use_cell_candidate':ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='#ffaa00',linewidth=2.5))
        ax.set_xticks(range(3),[4,8,16]);ax.set_yticks(range(3),['4K','8K','16K']);ax.set_xlabel('实际块数N');ax.set_ylabel('文档长度L');ax.set_title(SCOPE_LABEL[scope])
    fig.suptitle('格点候选−所选固定参数（百分点）；橙框为严格提升并采用的格');fig.colorbar(im,ax=axs.ravel().tolist(),label='F1百分点',fraction=.025,pad=.03)
    # Shared colorbar needs fixed margins rather than tight_layout.
    fig.subplots_adjust(left=.055,right=.87,bottom=.16,top=.82,wspace=.35)
    for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'validation_cell_gains.{ext}',dpi=180)
    plt.close(fig)

    def score(scope,policy):return next(r for r in global_rows if r['scope']==scope and r['policy']==policy)
    def paired_score(scope,a,b):return next(r for r in paired_counts if r['scope']==scope and r['policy_a']==a and r['policy_b']==b)
    fixed_gain=paired_score('all_27','default','global_candidate')
    main_cells=[r for r in rule_rows if r['scope']=='all_27']
    result={'experiment_stage':3,'status':'COMPLETED','research_status':'SUBMITTED','score_recomputed_count':0,'physical_generations':992,'logical_evaluations':1071,'reused_generations_saved':79,'scope_counts':scope_rows,'fixed_by_scope':{s:list(v) for s,v in fixed_by_scope.items()},'global_scores':global_rows,'paired_scores':paired_counts,'original_rule_matches':True,'original_rule_modified':False,'original_adopted_cells':[f"{r['L']}/{r['N']}" for r in main_cells if r['decision']=='use_cell_candidate'],'complete_rule_changes':sum(r['complete_changes_original'] for r in changes),'common_rule_changes':sum(r['common_changes_original'] for r in changes),'common_complete_ids_per_dataset':{d:len(ids) for d,ids in common.items()},'new_complete_candidates_missing_all_validation_generations':sum(r['n_missing_generation_configs'] for r in new_candidates),'new_complete_candidates_missing_complete_generations':sum(r['n_complete_missing_configs'] for r in new_candidates),'validation_scores_used':True,'test_scores_used':False,'later_stage_scores_used':False,'next_stage_executed':False,'bootstrap_replicates':BOOT,'seed':SEED}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'STAGE_STATUS.json').write_text(json.dumps({k:result[k] for k in ('experiment_stage','status','research_status','score_recomputed_count','physical_generations','logical_evaluations','reused_generations_saved','original_rule_matches','original_rule_modified','test_scores_used','next_stage_executed')},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'seed':SEED,'bootstrap_replicates':BOOT,'bootstrap_method':'paired evidence-group within dataset; all group lengths/N/candidate views retained; percentile CI; draws missing a predeclared cell rejected and counted','rng':'SeedSequence([20261116,scope_index])','analysis_commit':'227116e4a004153272640cfe31087a033861b8e9','new_dependencies':[],'scope':'validation scores only; prior-stage tags, candidates and timeline metadata'},ensure_ascii=False,indent=2),encoding='utf-8')
    report='# 实验第三阶段：验证候选参数与冻结规则\n\n'
    report+='## 1 数据与比较方法\n\n我们直接引用validation阶段992条已核对score，覆盖40题、119个题长输入、357个L/N输入条件。三类候选为默认0.9/0.9、探索全局0.7/1.0和探索各格最佳；logical_labels共1071次逻辑评价，各类357次。40次default/cell、39次global/cell共享生成，共节省79次。所有候选在同一数据集/格点使用相同ID。评分重算次数为0。\n\n'
    report+='先在每个数据集/格点内平均，再对三个数据集及九格等权。文中F1按0–100展示，差值单位为百分点。固定参数从默认与探索全局候选中选均值较高者；平分按距默认、T和S排序。格点候选严格高于固定参数时采用，其余回退。置信区间描述样本/证据组组成变化，选择准则沿用预定的均值比较。\n\n'
    report+=md(['口径','实际生成','逻辑评价','题长输入','独立题','Qasper/HotpotQA/MuSiQue题数'],[[r['label'],r['physical_generations'],r['logical_evaluations'],r['query_length_inputs'],r['unique_ids'],f"{r['n_qasper_ids']}/{r['n_hotpotqa_ids']}/{r['n_musique_ids']}"] for r in scope_rows])
    report+='\n证据完整子集按前置审核的源证据缺失token数为0筛选。筛选单位为题目×长度，同一问题在不同长度的完整状态可不同，因此完整记录仍覆盖40题，可用题长从119减为96。三长度共同完整口径固定22题，MuSiQue仅3题；三种口径均保持27个数据集/格点单元等权，各单元分母见candidate_pairing_coverage.csv。\n\n'
    report+='## 2 固定参数验证结果\n\n'
    report+=md(['口径','默认F1','全局候选F1','候选−默认/百分点','95%配对区间','所选固定T/S'],[[SCOPE_LABEL[s],f"{score(s,'default')['f1_percent']:.4f}",f"{score(s,'global_candidate')['f1_percent']:.4f}",f"{paired_score(s,'default','global_candidate')['delta_b_minus_a_pp']:+.4f}",f"[{paired_score(s,'default','global_candidate')['ci_low_pp']:.3f}, {paired_score(s,'default','global_candidate')['ci_high_pp']:.3f}]",'/'.join(map(str,fixed_by_scope[s]))] for s in scopes])
    report+=f"\n全量验证中，全局候选较默认高{fixed_gain['delta_b_minus_a_pp']:.4f}个百分点，按原准则选用0.7/1.0。完整子集及共同完整ID口径的固定参数选择相同。\n\n"
    report+='## 3 九格选择、回退与原规则核对\n\n'
    decisions={'use_cell_candidate':'采用格点候选','fallback_to_fixed':'回退固定','same_as_fixed':'候选即固定','tie_fallback':'平分回退'}
    report+=md(['L','N','格点候选T/S','候选F1','固定F1','候选−固定/百分点','95%配对区间','决定','最终T/S'],[[r['L'],r['N'],f"{r['candidate_T']}/{r['candidate_S']}",f"{r['candidate_f1_percent']:.3f}",f"{r['fixed_f1_percent']:.3f}",f"{r['delta_candidate_minus_fixed_pp']:+.3f}",f"[{r['delta_ci_low_pp']:.2f}, {r['delta_ci_high_pp']:.2f}]",decisions[r['decision']],f"{r['selected_T']}/{r['selected_S']}"] for r in main_cells])
    report+='\n全量回放的固定参数、九格最终参数及18个候选/固定分数均与保存rule.json一致。严格提升并采用的三格为4K/4、8K/4、8K/16；16K/4的候选本来就是固定0.7/1.0。\n\n'
    report+=md(['验证口径','默认','统一固定','全部逐格候选','原冻结规则','本口径回放规则'],[[SCOPE_LABEL[s],*[f"{score(s,p)['f1_percent']:.3f}" for p in ('default','selected_fixed','cell_candidates','original_frozen_rule','scope_replay_rule')]] for s in scopes])
    report+='\n全部逐格候选直接采用的验证均值低于统一固定参数；回退步骤保留有正提升的格点。规则选择和本表评价共用验证数据，正式测试阶段继续评价已冻结版本。\n\n![验证分数与区间](../figures/validation_policy_comparison.png)\n\n'
    report+='## 4 证据完整性与共同ID敏感性\n\n'
    report+=md(['L','N','原冻结T/S','完整子集回放T/S','共同22题回放T/S'],[[r['L'],r['N'],f"{r['original_T']}/{r['original_S']}",f"{r['complete_T']}/{r['complete_S']}",f"{r['common_T']}/{r['common_S']}"] for r in changes])
    rowall=next(r for r in rule_rows if r['scope']=='all_27' and (r['L'],r['N'])==(8192,4));rowclean=next(r for r in rule_rows if r['scope']=='complete_27' and (r['L'],r['N'])==(8192,4))
    report+=f"\n完整子集仅8K/4改变：候选0.7/0.7相对固定的差从{rowall['delta_candidate_minus_fixed_pp']:+.3f}变为{rowclean['delta_candidate_minus_fixed_pp']:+.3f}个百分点，因而回退0.7/1.0。共同22题口径有{result['common_rule_changes']}格与原规则不同，其中4K/8的采用差仅约0.039个百分点。候选的点估计选择对样本构成有敏感性，具体分母与区间在表中一并报告。\n\n![逐格候选相对固定的差异](../figures/validation_cell_gains.png)\n\n"
    report+='这两项敏感性回放沿用原探索提名的候选；阶段2证据完整探索中新出现的候选另查验证覆盖。\n\n'
    report+=md(['L','N','阶段2完整子集候选','缺少验证生成/全量','缺少验证生成/完整子集'],[[r['L'],r['N'],f"{r['complete_exploration_T']}/{r['complete_exploration_S']}",r['n_missing_generation_configs'],r['n_complete_missing_configs']] for r in new_candidates if r['n_missing_generation_configs']])
    report+=f"\n上述三个新参数组合合计缺{result['new_complete_candidates_missing_all_validation_generations']}次全量验证生成，限定已有完整输入则缺{result['new_complete_candidates_missing_complete_generations']}次。若要验证这些新组合，需要另行授权生成；本次未补填或借用其他格点分数。敏感性规则只另存于本阶段，原候选和冻结规则保持原值。\n\n"
    report+='## 5 冻结顺序、统计与交付\n\n'
    report+='候选于北京时间2026-11-14 14:03:50冻结，随后验证worker于14:03:55启动；规则于14:17:49冻结，早于长度检查、注意力与正式测试worker启动。完整时间及源位置见freeze_timeline.csv。规则按log2文档长度、log2实际编码块数的最近格查表，平距选较小L/N。\n\n'
    clean_diag=next(r for r in bootdiag if r['scope']=='complete_27' and r['dataset']=='musique')
    report+=f"置信区间采用seed20261116、5000次有效配对证据组Bootstrap。每数据集内有放回抽组，保留该组的全部长度、N和候选视图；共享生成使用同一源行和同一重采样权重。抽样时固定已选参数与规则，区间反映验证问题/证据组构成变化。每次按原27单元等权；抽样造成空单元时重新抽取并登记计数。完整子集MuSiQue的最少单格为{clean_diag['min_ids_per_cell']}题，提出{clean_diag['proposed_replicates']}次抽样，其中{clean_diag['rejected_empty_cell_draws']}次产生空单元并重抽，得到5000次有效抽样；其他口径/数据集的空单元次数为0。详见bootstrap_diagnostics.csv。\n\n"
    report+='本阶段结论：固定参数仍选0.7/1.0；原九格回退规则符合协议及保存分数；完整子集改变一格回退，共同完整ID口径改变四格；阶段2部分新敏感性候选尚缺对应验证生成。原记录、候选及规则均保留，研究状态SUBMITTED。当前停在实验第三阶段，后续长度检查等实验待用户审阅规划后授权执行。\n'
    (OUT/'reports/VALIDATION_REPORT.md').write_text(report,encoding='utf-8')
    index='# 实验第三阶段材料：验证与冻结规则\n\n- [分析报告](reports/VALIDATION_REPORT.md)\n- [口径与分母](tables/scope_counts.csv)\n- [候选复用计数](tables/generation_reuse.csv)\n- [1071个逻辑候选视图与源行](tables/logical_candidate_views.csv)\n- [全局分数及区间](tables/validation_policy_scores.csv)\n- [主要配对差及区间](tables/validation_policy_paired.csv)\n- [九格决定、配对区间及分母](tables/validation_cell_decisions.csv)\n- [逐输入原分数引用](tables/input_policy_scores.csv)\n- [保存规则分数核对](tables/saved_rule_checks.csv)\n- [完整性敏感性规则](tables/rule_sensitivity.csv)\n- [新探索候选缺少的验证条件](tables/new_exploration_candidate_coverage.csv)\n- [Bootstrap诊断与簇数](tables/bootstrap_diagnostics.csv)\n- [冻结时间线](tables/freeze_timeline.csv)\n\n原候选与规则副本位于sources/，敏感性规则另存sensitivity_rule_replays.json。图表在figures/，含PNG和SVG。环境、版本、随机种子及统计方法见environment.json。\n\n复现（仓库根目录）：\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_validation.py\n```\n\n依赖均为已有NumPy/Matplotlib。本次直接复用validation保存score，评分重算0次；全部新文件位于本阶段目录。未使用正式测试分数，未运行GPU或修改冻结规则。\n'
    (OUT/'README.md').write_text(index,encoding='utf-8')
    (BASE/'README.md').write_text('# APE MAIN80 分实验分析\n\n当前已完成实验第三阶段：验证候选与冻结规则。研究状态SUBMITTED。\n\n- [实验3：验证与规则](stages/03_validation_rule/README.md)\n- [实验2：参数探索](stages/02_parameter_exploration/README.md)\n- [实验1：600题三方法基线及官方对照](stages/01_longbench_baseline/README.md)\n- [前置审核](stages/01_audit/README.md)\n\n本轮已停在实验3；后续每阶段先提交简短规划，由用户审阅后执行。原始证据、候选与规则保留。各阶段的脚本、CSV、图表、环境和复现命令见对应索引。\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('physical_generations','logical_evaluations','reused_generations_saved','fixed_by_scope','original_rule_matches','original_adopted_cells','complete_rule_changes','common_rule_changes','common_complete_ids_per_dataset','new_complete_candidates_missing_all_validation_generations')},ensure_ascii=False,indent=2))
    print(json.dumps([r for r in paired_counts if r['scope']=='all_27'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
