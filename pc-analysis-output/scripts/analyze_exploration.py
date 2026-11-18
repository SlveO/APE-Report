"""Experiment 2 only. Aggregate saved explore scores; no answer re-scoring.

Reads explore records, frozen candidates, applicability metadata and the completed
pre-audit's evidence-retention tags. Does not select using validation/test scores.
"""
from pathlib import Path
from collections import defaultdict,Counter
from itertools import product
from statistics import mean
import csv,json,os,sys,shutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/02_parameter_exploration'
RUN=ROOT/'restored/output/runs/main80-20261114T041503'
AUDIT=BASE/'stages/01_audit'
for name in ('tables','reports','figures','sources','cache','tmp','logs'):
    (OUT/name).mkdir(parents=True,exist_ok=True)
for name in ('MPLCONFIGDIR','XDG_CACHE_HOME','TEMP','TMP'):
    os.environ[name]=str(OUT/('cache' if name in ('MPLCONFIGDIR','XDG_CACHE_HOME') else 'tmp'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DS=('qasper','hotpotqa','musique')
LENGTHS=(4096,8192,16384)
COUNTS=(4,8,16)
PARAMS=list(product((.5,.7,.9,1.),repeat=2))
CELLS=list(product(LENGTHS,COUNTS))
DEFAULT=(.9,.9)
SHARED=[c for c in CELLS if c[0] in (4096,8192)]

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def readcsv(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def writecsv(name, rows):
    rows=list(rows)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)
def tie(p):return ((p[0]-.9)**2+(p[1]-.9)**2,p[0],p[1])
def md(headers, rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

def main():
    tags={(r['sample_id'],int(r['L'])):r for r in readcsv(AUDIT/'tables/construction_checks.csv') if r['split']=='explore' and int(r['L']) in LENGTHS}
    candidates=load(RUN/'candidates.json')
    original_copy=OUT/'sources/candidates_original.json'
    if not original_copy.exists():shutil.copyfile(RUN/'candidates.json',original_copy)
    records=[]
    with (RUN/'attempts.jsonl').open(encoding='utf-8') as stream:
        for line,text in enumerate(stream,1):
            r=json.loads(text)
            if r['key']['stage']!='explore':continue
            assert r['status']=='success' and r['split']=='explore'
            tag=tags[(r['sample_id'],r['L'])]
            r['_line']=line
            r['_missing']=int(tag['n_missing_required_tokens'])
            r['_complete']=r['_missing']==0
            records.append(r)
    assert len(records)==6672
    common={}
    for d in DS:
        common[d]=set.intersection(*({r['sample_id'] for r in records if r['dataset']==d and r['L']==l and r['_complete']} for l in LENGTHS))
    scopes={
        'all_27':{'cells':CELLS,'rows':records,'label':'全部实际输入，27层'},
        'complete_27':{'cells':CELLS,'rows':[r for r in records if r['_complete']],'label':'证据完整子集，27层'},
        'all_shared18':{'cells':SHARED,'rows':[r for r in records if r['L'] in (4096,8192)],'label':'全部输入，仅4K/8K共同18层'},
        'complete_shared18':{'cells':SHARED,'rows':[r for r in records if r['_complete'] and r['L'] in (4096,8192)],'label':'证据完整，仅4K/8K共同18层'},
        'missing_shared18':{'cells':SHARED,'rows':[r for r in records if not r['_complete']],'label':'证据缺失，仅4K/8K共同18层'},
        'complete_common_ids27':{'cells':CELLS,'rows':[r for r in records if r['sample_id'] in common[r['dataset']]],'label':'三长度共同完整ID，27层'},
    }
    coverage=[];grid=[];global_ranks=[];dataset_cells=[];selections=[];paired=[];marginal=[];scope_sizes=[];scores={};best={};global_best={};groups_by_scope={};pairing=[]
    for name,spec in scopes.items():
        rows=spec['rows'];cells=spec['cells'];groups=defaultdict(list)
        for r in rows:groups[(r['dataset'],r['L'],r['N'],r['T'],r['S'])].append(r)
        groups_by_scope[name]=groups
        local_scores={}
        for d,(l,n) in product(DS,cells):
            sets=[{r['sample_id'] for r in groups[(d,l,n,*p)]} for p in PARAMS]
            assert sets[0] and all(ids==sets[0] for ids in sets)
            assert all(len(groups[(d,l,n,*p)])==len(sets[0]) for p in PARAMS)
            pairing.append({'scope':name,'dataset':d,'L':l,'N':n,'n_ids_per_parameter':len(sets[0]),'n_parameters':16,'same_ids_all_16':True,'unit':'unique IDs per dataset/cell/parameter'})
            for p in PARAMS:
                rr=groups[(d,l,n,*p)]
                value=mean(r['score'] for r in rr)
                local_scores[(d,l,n,*p)]=value
                dataset_cells.append({'scope':name,'dataset':d,'L':l,'N':n,'T':p[0],'S':p[1],'n':len(rr),'saved_f1_mean_0_1':value,'f1_percent':value*100,'unit':'F1 x100; score reused','source':'attempts.jsonl:score; key.stage=explore'})
        scores[name]={p:mean(local_scores[(d,l,n,*p)] for d,(l,n) in product(DS,cells)) for p in PARAMS}
        ordered=sorted(PARAMS,key=lambda p:(-scores[name][p],tie(p)))
        global_best[name]=ordered[0]
        for rank,p in enumerate(ordered,1):
            global_ranks.append({'scope':name,'scope_label':spec['label'],'rank':rank,'T':p[0],'S':p[1],'f1_0_1':scores[name][p],'f1_percent':scores[name][p]*100,'delta_vs_default_pp':(scores[name][p]-scores[name][DEFAULT])*100,'n_dataset_cells':3*len(cells),'n_generations_per_parameter':len(rows)//16,'n_query_length_inputs':len({(r['sample_id'],r['L']) for r in rows}),'unit':'F1 x100; percentage-point difference','aggregation':'mean of dataset/cell means; datasets and included cells equally weighted'})
        best[name]={}
        for l,n in cells:
            cell_score={p:mean(local_scores[(d,l,n,*p)] for d in DS) for p in PARAMS}
            order=sorted(PARAMS,key=lambda p:(-cell_score[p],tie(p)))
            bp=order[0];best[name][(l,n)]=bp
            nids=[len(groups[(d,l,n,*DEFAULT)]) for d in DS]
            for rank,p in enumerate(order,1):
                grid.append({'scope':name,'L':l,'N':n,'T':p[0],'S':p[1],'rank_in_cell':rank,'f1_0_1':cell_score[p],'f1_percent':cell_score[p]*100,'n_ids_total':sum(nids),'n_qasper':nids[0],'n_hotpotqa':nids[1],'n_musique':nids[2],'n_dataset_cells':3,'unit':'F1 x100','aggregation':'equal mean of three dataset means'})
            selections.append({'scope':name,'L':l,'N':n,'best_T':bp[0],'best_S':bp[1],'best_f1_percent':cell_score[bp]*100,'second_T':order[1][0],'second_S':order[1][1],'margin_vs_second_pp':(cell_score[bp]-cell_score[order[1]])*100,'default_f1_percent':cell_score[DEFAULT]*100,'gain_vs_default_pp':(cell_score[bp]-cell_score[DEFAULT])*100,'scope_global_T':ordered[0][0],'scope_global_S':ordered[0][1],'gain_vs_scope_global_pp':(cell_score[bp]-cell_score[ordered[0]])*100,'n_ids':sum(nids),'n_qasper':nids[0],'n_hotpotqa':nids[1],'n_musique':nids[2],'unit':'F1 x100; percentage points'})
            diffs=[]
            for d in DS:
                ref={r['sample_id']:r for r in groups[(d,l,n,*DEFAULT)]}
                selected={r['sample_id']:r for r in groups[(d,l,n,*bp)]}
                diffs.extend(selected[s]['score']-ref[s]['score'] for s in ref)
            paired.append({'scope':name,'L':l,'N':n,'candidate_T':bp[0],'candidate_S':bp[1],'reference_T':.9,'reference_S':.9,'n_pairs':len(diffs),'delta_equal_dataset_pp':(cell_score[bp]-cell_score[DEFAULT])*100,'win':sum(x>1e-12 for x in diffs),'tie':sum(abs(x)<=1e-12 for x in diffs),'loss':sum(x< -1e-12 for x in diffs),'unit':'paired ID count; macro F1 percentage points','source':'saved score; same IDs within dataset/cell'})
        for axis in ('T','S'):
            for v in (.5,.7,.9,1.):
                values=[scores[name][p] for p in PARAMS if p[0 if axis=='T' else 1]==v]
                marginal.append({'scope':name,'axis':axis,'value':v,'f1_percent':mean(values)*100,'n_other_parameter_values':4,'n_dataset_cells':3*len(cells),'unit':'F1 x100','aggregation':'equal over other parameter, datasets and included cells'})
        for axis,vals in [('L',sorted({l for l,n in cells})),('N',COUNTS)]:
            for v in vals:
                cc=[c for c in cells if c[0 if axis=='L' else 1]==v]
                for p in (DEFAULT,ordered[0]):
                    marginal.append({'scope':name,'axis':axis,'value':v,'T':p[0],'S':p[1],'f1_percent':mean(local_scores[(d,l,n,*p)] for d,(l,n) in product(DS,cc))*100,'n_dataset_cells':3*len(cc),'unit':'F1 x100','aggregation':'fixed parameter; equal over datasets/included cells'})
        scope_sizes.append({'scope':name,'label':spec['label'],'n_generations':len(rows),'n_unique_ids':len({r['sample_id'] for r in rows}),'n_query_length_inputs':len({(r['sample_id'],r['L']) for r in rows}),'n_cells':len(cells),'n_dataset_cells':3*len(cells),'n_generations_per_parameter':len(rows)//16,'n_qasper_unique_ids':len({r['sample_id'] for r in rows if r['dataset']=='qasper'}),'n_hotpotqa_unique_ids':len({r['sample_id'] for r in rows if r['dataset']=='hotpotqa'}),'n_musique_unique_ids':len({r['sample_id'] for r in rows if r['dataset']=='musique'}),'unit':'generation / ID / ID-length / cell count'})
    assert tuple(candidates['global'])==global_best['all_27']
    assert all(tuple(candidates['cells'][f'{l}/{n}'])==best['all_27'][(l,n)] for l,n in CELLS)
    observed={r['scope']:r for r in scope_sizes}
    assert observed['complete_27']['n_generations']==5280 and observed['missing_shared18']['n_generations']==1392
    for l in LENGTHS:
        rr=[r for r in records if r['L']==l and r['N']==4 and (r['T'],r['S'])==DEFAULT]
        coverage.append({'L':l,'n_applicable_question_lengths':len(rr),'n_complete':sum(r['_complete'] for r in rr),'n_missing':sum(not r['_complete'] for r in rr),'n_qasper':sum(r['dataset']=='qasper' for r in rr),'n_hotpotqa':sum(r['dataset']=='hotpotqa' for r in rr),'n_musique':sum(r['dataset']=='musique' for r in rr),'n_N_values':3,'n_parameter_values':16,'n_physical_generations':len(rr)*48,'unit':'unique question-length / generation count'})
    changes=[]
    for l,n in CELLS:
        orig=best['all_27'][(l,n)];clean=best['complete_27'][(l,n)];commonp=best['complete_common_ids27'][(l,n)]
        changes.append({'L':l,'N':n,'original_T':orig[0],'original_S':orig[1],'complete_T':clean[0],'complete_S':clean[1],'complete_changes_original':clean!=orig,'common_ids_T':commonp[0],'common_ids_S':commonp[1],'common_ids_changes_original':commonp!=orig,'unit':'parameter pair; boolean'})
    provenance=[{'source_line':r['_line'],'sample_id':r['sample_id'],'dataset':r['dataset'],'L':r['L'],'N':r['N'],'T':r['T'],'S':r['S'],'score_0_1':r['score'],'evidence_complete':r['_complete'],'missing_evidence_token_count':r['_missing'],'three_lengths_common_complete_id':r['sample_id'] in common[r['dataset']],'source':'attempts.jsonl:score; 01_audit/construction_checks.csv:n_missing_required_tokens'} for r in records]
    for name,data in [('scope_counts.csv',scope_sizes),('coverage_by_length.csv',coverage),('parameter_pairing.csv',pairing),('dataset_cell_scores.csv',dataset_cells),('cell_parameter_rankings.csv',grid),('global_parameter_rankings.csv',global_ranks),('cell_candidates.csv',selections),('candidate_vs_default_paired.csv',paired),('marginal_effects.csv',marginal),('candidate_sensitivity.csv',changes),('exploration_records.csv',provenance)]:writecsv(name,data)
    writecsv('common_complete_ids.csv',[{'dataset':d,'sample_id':sid,'scope':'complete_common_ids27','all_L':list(LENGTHS),'all_N':list(COUNTS),'evidence_definition':'pre-audit missing source evidence token count=0 at all three lengths'} for d in DS for sid in sorted(common[d])])

    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False})
    values=[r['f1_percent'] for r in grid if r['scope'] in ('all_27','complete_27')]
    vmin,vmax=min(values),max(values)
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    for scope,filename,title in [('all_27','heatmaps_all9','全部实际输入'),('complete_27','heatmaps_complete9','证据完整子集')]:
        fig,axs=plt.subplots(3,3,figsize=(11.5,10.5));fig.subplots_adjust(left=.065,right=.89,bottom=.065,top=.91,wspace=.28,hspace=.38)
        for ax,(l,n) in zip(axs.ravel(),CELLS):
            rr=[r for r in grid if r['scope']==scope and (r['L'],r['N'])==(l,n)]
            lookup={(r['T'],r['S']):r for r in rr}
            mat=np.array([[lookup[(t,s)]['f1_percent'] for s in (.5,.7,.9,1.)] for t in (.5,.7,.9,1.)])
            im=ax.imshow(mat,vmin=vmin,vmax=vmax,cmap='viridis',aspect='auto')
            for i,j in product(range(4),repeat=2):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]<(vmin+vmax)/2 else 'black')
            bp=best[scope][(l,n)];ix=(.5,.7,.9,1.).index(bp[1]);iy=(.5,.7,.9,1.).index(bp[0])
            ax.add_patch(plt.Rectangle((ix-.5,iy-.5),1,1,fill=False,edgecolor='#ff4848',linewidth=2.4))
            ax.plot(2.31,1.69,'o',ms=4,color='white',markeredgecolor='gray')
            ax.set_xticks(range(4),[.5,.7,.9,1.]);ax.set_yticks(range(4),[.5,.7,.9,1.]);ax.set_xlabel('S');ax.set_ylabel('T');ax.set_title(f'L={l}, N={n}, n={rr[0]["n_ids_total"]}')
        cax=fig.add_axes([.915,.15,.02,.65]);fig.colorbar(im,cax=cax,label='F1 ×100（三数据集等权）')
        fig.suptitle(f'实验2 T/S九格热图：{title}\n红框：本口径最佳；白圈：默认0.9/0.9；两图统一色标',fontsize=12)
        save(fig,filename)
    fig,ax=plt.subplots(figsize=(11,4.8));pp=sorted(PARAMS,key=lambda p:(-scores['all_27'][p],tie(p)));xx=np.arange(16)
    for dx,scope,label in [(-.2,'all_27','全部27层'),(0,'complete_27','完整27层'),(.2,'complete_common_ids27','三长度共同完整ID')]:
        ax.bar(xx+dx,[100*scores[scope][p] for p in pp],.2,label=label)
    ax.set_xticks(xx,[f'{p[0]}/{p[1]}' for p in pp],rotation=40);ax.set_xlabel('T / S（按全部输入排名排列）');ax.set_ylabel('F1 ×100');ax.set_title('全局参数表现与证据完整性敏感性');ax.legend(ncol=3);fig.tight_layout();save(fig,'global_parameter_comparison')
    top={name:next(r for r in global_ranks if r['scope']==name and r['rank']==1) for name in scopes}
    default={name:next(r for r in global_ranks if r['scope']==name and (r['T'],r['S'])==DEFAULT) for name in scopes}
    r2=next(r for r in global_ranks if r['scope']=='all_27' and r['rank']==2)
    policy=[]
    for name,spec in scopes.items():
        selected=[r for r in selections if r['scope']==name]
        cell_mean=mean(r['best_f1_percent'] for r in selected)
        policy.append({'scope':name,'n_dataset_cells':3*len(spec['cells']),'default_f1_percent':default[name]['f1_percent'],'global_candidate_T':top[name]['T'],'global_candidate_S':top[name]['S'],'global_candidate_f1_percent':top[name]['f1_percent'],'exploration_cell_candidates_f1_percent':cell_mean,'cell_candidates_minus_global_pp':cell_mean-top[name]['f1_percent'],'unit':'F1 x100; percentage points','evaluation_data':'same exploration subset used to choose candidates; handoff to validation'})
    writecsv('exploration_candidate_summary.csv',policy)
    result={'experiment_stage':2,'status':'COMPLETED','research_status':'SUBMITTED','score_recomputed_count':0,'input_generation_count':len(records),'configuration_count':144,'all_dataset_cells':27,'saved_candidates_match':True,'original_candidates_modified':False,'validation_scores_used':False,'test_scores_used':False,'top_by_scope':top,'default_by_scope':default,'common_complete_ids_per_dataset':{d:len(common[d]) for d in DS},'complete_cell_candidate_changes':sum(r['complete_changes_original'] for r in changes),'common_id_cell_candidate_changes':sum(r['common_ids_changes_original'] for r in changes),'exploration_candidate_summary':policy,'next_stage_executed':False}
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'STAGE_STATUS.json').write_text(json.dumps({k:result[k] for k in ('experiment_stage','status','research_status','score_recomputed_count','saved_candidates_match','original_candidates_modified','validation_scores_used','test_scores_used','next_stage_executed')},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'randomness':'none; saved scores grouped deterministically; no bootstrap in this exploratory-ranking stage','metric':'saved LongBench F1 0–1; presentation x100','source_score_audit':'01_audit/reports/STAGE1_SCORING.md','analysis_commit':'227116e4a004153272640cfe31087a033861b8e9','new_dependencies':[],'scope':'explore only'},ensure_ascii=False,indent=2),encoding='utf-8')
    text='# 实验第二阶段：九格参数探索\n\n'
    text+='## 1 本阶段分析什么\n\n我们直接使用已审核的explore阶段6,672条保存score，按L=4096/8192/16384、N=4/8/16及16组T/S汇总。全部144种配置覆盖完整；每个数据集×L×N中，16组参数使用相同ID。本阶段评分重算次数为0，也未使用验证或正式测试成绩。\n\n'
    text+='单格分数=三个数据集各自题目平均F1的等权平均；全局分数=九格单格分数的等权平均，即27个数据集×格点单元等权。展示为F1×100。平分时依次按距0.9/0.9最近、T升序、S升序处理。\n\n'
    text+=md(['文档长度','有效题长','证据完整','证据缺失','实际生成'],[[r['L'],r['n_applicable_question_lengths'],r['n_complete'],r['n_missing'],r['n_physical_generations']] for r in coverage])
    text+='\n合计139个有效题长输入，乘3种N和16组参数得到6,672次生成。证据完整110个题长/5,280次生成，证据缺失29个题长/1,392次生成。完整性依据前置审核的n_missing_required_tokens=0，按源字符证据及保守支撑文档范围定位；证据位置元数据差异单独保留。\n\n'
    text+='## 2 原探索候选：全局0.7/1.0，九格排序与原产物一致\n\n'
    text+=f"全局第一为T={top['all_27']['T']}、S={top['all_27']['S']}，F1={top['all_27']['f1_percent']:.4f}；默认为{default['all_27']['f1_percent']:.4f}，相差{top['all_27']['delta_vs_default_pp']:.4f}个百分点。第二名T={r2['T']}、S={r2['S']}为{r2['f1_percent']:.4f}，与第一名相差{top['all_27']['f1_percent']-r2['f1_percent']:.4f}个百分点。全局及九格候选均吻合保存的candidates.json。\n\n"
    ss=[r for r in selections if r['scope']=='all_27']
    text+=md(['L','N','最佳T/S','候选F1','默认F1','候选−默认（百分点）','与第二名差距','题数'],[[r['L'],r['N'],f"{r['best_T']}/{r['best_S']}",f"{r['best_f1_percent']:.3f}",f"{r['default_f1_percent']:.3f}",f"{r['gain_vs_default_pp']:+.3f}",f"{r['margin_vs_second_pp']:.3f}",r['n_ids']] for r in ss])
    original_policy=next(r for r in policy if r['scope']=='all_27')
    text+=f"\n逐格分别选参后，这批探索输入的等权均值为{original_policy['exploration_cell_candidates_f1_percent']:.3f}，较统一全局参数高{original_policy['cell_candidates_minus_global_pp']:.3f}个百分点。该值由同一批探索样本选参并评价，下一阶段按既定验证集评估这些候选。\n"
    text+='\n![全部九格热图](../figures/heatmaps_all9.png)\n\n'
    text+='原始探索的N=16三格均选择T=0.5；N=4/8的T随L和N变化。逐格候选与默认的差异、同题胜平负见candidate_vs_default_paired.csv。这里的候选由同批探索分数选出，后续验证阶段评价其表现并决定回退。\n\n'
    text+='## 3 证据完整性是否改变排名\n\n'
    text+=f"完整子集保持全部27个单元，全局第一仍为0.7/1.0，F1={top['complete_27']['f1_percent']:.4f}；默认为{default['complete_27']['f1_percent']:.4f}。九格最佳有{result['complete_cell_candidate_changes']}/9格变化。\n\n"
    text+=md(['L','N','原候选','完整子集候选','改变'],[[r['L'],r['N'],f"{r['original_T']}/{r['original_S']}",f"{r['complete_T']}/{r['complete_S']}",'是' if r['complete_changes_original'] else '否'] for r in changes])
    raw_t=[best['all_27'][(4096,n)][0] for n in COUNTS]
    clean_t=[best['complete_27'][(4096,n)][0] for n in COUNTS]
    text+=f"\n4K下，原最优T随N=4/8/16依次为{raw_t}，完整子集则为{clean_t}。温度随块数的走势在两类输入口径中发生反转，逐格选参对输入完整性及题目构成敏感。\n"
    text+='\n![完整子集九格热图](../figures/heatmaps_complete9.png)\n\n'
    text+='证据缺失组只存在于4K/8K。为对齐格点范围，我们另外把全部输入和完整子集都限制到这两个长度，三者均按18个数据集×格点单元等权。\n\n'
    text+=md(['口径','单元数','最佳T/S','最佳F1','默认F1','每参数生成数'],[[scopes[name]['label'],top[name]['n_dataset_cells'],f"{top[name]['T']}/{top[name]['S']}",f"{top[name]['f1_percent']:.4f}",f"{default[name]['f1_percent']:.4f}",top[name]['n_generations_per_parameter']] for name in ('all_27','complete_27','all_shared18','complete_shared18','missing_shared18','complete_common_ids27')])
    text+='\n完整组与缺失组在共同18单元下的最佳均为0.5/0.9。比较完整27层和缺失18层会同时改变长度构成；这项共同范围比较表明，直接把两种口径的最佳参数变化归到证据缺失会混入长度权重变化。完整/缺失组仍来自不同题目子集，分数差描述它们各自的表现。修复输入后的收益需另用修正输入的生成结果评价。\n\n'
    text+=f"我们还固定三长度均适用且证据完整的共同ID：{result['common_complete_ids_per_dataset']}，共{sum(result['common_complete_ids_per_dataset'].values())}题；同一批题跨九格比较，继续对三个数据集和九格等权。其全局最佳为{top['complete_common_ids27']['T']}/{top['complete_common_ids27']['S']}，F1={top['complete_common_ids27']['f1_percent']:.4f}。完整ID清单和按L/N的参数表现分别保存在common_complete_ids.csv、marginal_effects.csv，用于检查长度与块数走势是否伴随样本构成变化。\n\n"
    text+='## 4 参数变化的具体规律\n\n'
    def marginal_value(scope,axis,value):
        return next(r['f1_percent'] for r in marginal if r['scope']==scope and r['axis']==axis and r['value']==value and 'T' not in r)
    text+='先对另一个参数的四个取值等权平均，再对数据集和格点等权，得到以下边际表现。\n\n'
    text+=md(['参数取值','T取该值（全部输入）','S取该值（全部输入）','T取该值（共同完整ID）','S取该值（共同完整ID）'],[[v,f"{marginal_value('all_27','T',v):.3f}",f"{marginal_value('all_27','S',v):.3f}",f"{marginal_value('complete_common_ids27','T',v):.3f}",f"{marginal_value('complete_common_ids27','S',v):.3f}"] for v in (.5,.7,.9,1.)])
    text+='\n在本取值网格内，较小S=0.5的平均表现较低，S=0.9/1.0较高。单独平均S时T=0.5的边际均值最高，而最佳联合配置为T=0.7/S=1.0，体现T与S的联合作用，候选应按组合比较。\n\n'
    fixed_n=[next(r for r in marginal if r['scope']=='complete_common_ids27' and r['axis']=='N' and r['value']==n and r.get('T')==.7 and r.get('S')==1.) for n in COUNTS]
    text+=md(['实际块数N','共同完整27题、固定0.7/1.0的F1'],[[r['value'],f"{r['f1_percent']:.3f}"] for r in fixed_n])
    text+='\n同一批完整问题、相同长度集合和固定参数下，N增大时分数下降。每块变短会缩小资料独立编码时可见的范围，后续按输入条件选参需同时考虑这种分块影响。共同ID的L=4K/8K/16K对应F1约41.04/43.77/40.89，长度走势也未呈单调增加。各项均按实际构造和固定参数统计。\n\n'
    text+='## 5 本阶段得到的结论\n\n'
    text+=f"1. 原探索候选已解释并核对：全局0.7/1.0较默认高{top['all_27']['delta_vs_default_pp']:.3f}个百分点，领先第二名{top['all_27']['f1_percent']-r2['f1_percent']:.3f}个百分点。\n"
    text+=f"2. 完整子集全局最佳仍为0.7/1.0，逐格最佳变化{result['complete_cell_candidate_changes']}格；格点规则对输入完整性的敏感性需要进入验证解释。\n"
    text+='3. 完整/缺失组的参数排序应在共同4K/8K范围比较，同时列出每层题数；三长度共同ID分析另外控制跨长度的样本变化。\n'
    text+='4. 原候选文件保持冻结版本，敏感性候选单独保存在本阶段表格，交下一阶段验证及补正讨论。\n\n'
    text+='## 6 材料与复现\n\n本阶段仅汇总已有score，未重新评分、生成答案、运行GPU、安装依赖或改动原始证据及冻结规则。脚本为pc-analysis-output/scripts/analyze_exploration.py；所有CSV包含口径、分母、单位或来源行号。研究状态保留SUBMITTED，当前停在参数探索阶段。\n'
    (OUT/'reports/PARAMETER_EXPLORATION.md').write_text(text,encoding='utf-8')
    index='# 实验第二阶段材料：九格参数探索\n\n- [分析报告](reports/PARAMETER_EXPLORATION.md)\n- [各口径分母](tables/scope_counts.csv)\n- [16组全局排名](tables/global_parameter_rankings.csv)\n- [九格×16参数排名及热图数据](tables/cell_parameter_rankings.csv)\n- [各格候选与默认差值](tables/cell_candidates.csv)\n- [完整性敏感性候选](tables/candidate_sensitivity.csv)\n- [完整/缺失配对与样本范围](tables/parameter_pairing.csv)\n- [逐题已保存分数及证据标签](tables/exploration_records.csv)\n- [共同完整ID](tables/common_complete_ids.csv)\n- [T/S及L/N分层走势](tables/marginal_effects.csv)\n- [全局参数对照图](figures/global_parameter_comparison.png)\n\n全部图另存SVG。完整性定义和源字段见报告，原候选副本位于sources/candidates_original.json。\n\n复现（仓库根目录）：\n\n```powershell\n$env:PYTHONUTF8="1"\n$env:PYTHONNOUSERSITE="1"\n& "C:/Users/Osheepne/anaconda3/python.exe" -s -B pc-analysis-output/scripts/analyze_exploration.py\n```\n\n依赖为已有NumPy/Matplotlib，环境见environment.json；本阶段无随机抽样，未执行Bootstrap。输入包含explore保存分数及前置审核标签，评分重算次数0。当前阶段完成后停止，验证阶段由用户安排。\n'
    (OUT/'README.md').write_text(index,encoding='utf-8')
    (BASE/'README.md').write_text('# APE MAIN80 分实验分析\n\n当前已完成实验第二阶段：九格参数探索，直接使用已核对保存分数。研究状态SUBMITTED。\n\n- [实验2：参数探索](stages/02_parameter_exploration/README.md)\n- [实验1：600题三方法基线与官方对照](stages/01_longbench_baseline/README.md)\n- [前置数据审核](stages/01_audit/README.md)\n\n本轮已停在实验2；下一步为验证候选、回退及冻结规则的分析，待用户安排。原候选及规则保留。各阶段的脚本、CSV、图表、环境和命令见对应索引。\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('input_generation_count','configuration_count','saved_candidates_match','score_recomputed_count','complete_cell_candidate_changes','common_complete_ids_per_dataset')},ensure_ascii=False,indent=2))
    print(json.dumps({name:{k:r[k] for k in ('T','S','f1_percent','delta_vs_default_pp','n_dataset_cells')} for name,r in top.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
