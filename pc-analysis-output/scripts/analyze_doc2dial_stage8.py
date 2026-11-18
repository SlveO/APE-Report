"""Stage 8 only: Doc2Dial corpus BLEU and paired cold/hot CPU analysis.

The standard-library scorer helpers were verified in the pre-audit against
SacreBLEU2.4.3. No scorer installation, model generation or rule changes.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from importlib.metadata import version
import csv
import importlib.util
import json
import math
import os
import re
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'pc-analysis-output'
OUT=BASE/'stages/08_doc2dial'
RUN=ROOT/'restored/output/runs/main80-20261114T041503'
FROZEN=RUN/'snapshot/datasets/frozen-main80'
AUDIT=BASE/'stages/01_audit/tables'
REFERENCE='227116e4a004153272640cfe31087a033861b8e9'
for name in ['tables','reports','figures','sources','logs','cache','tmp']:(OUT/name).mkdir(parents=True,exist_ok=True)
for key in ['MPLCONFIGDIR','XDG_CACHE_HOME','TEMP','TMP']:os.environ[key]=str(OUT/('tmp' if key in ['TEMP','TMP'] else 'cache'))
import numpy as np

HELPERS=runpy.run_path(str(BASE/'scripts/audit_stage1_scoring.py'),run_name='doc2dial_bleu_helpers')
bleu_stats=HELPERS['bleu_stats'];bleu_scalar=HELPERS['bleu_from_stats']
text2line=HELPERS['text2line'];tag=HELPERS['tag'];SIGNATURE=HELPERS['SIGNATURE']
METHODS=['sequential','ape_fixed','ape_dynamic'];CACHES=['cold','hot']
CONDS=[(m,c) for m in METHODS for c in CACHES]
COMPARISONS=[('method',(a,c),(b,c)) for a,b in combinations(METHODS,2) for c in CACHES]+[('cache',(m,'cold'),(m,'hot')) for m in METHODS]
METRICS={'ttft_seconds':'s','total_seconds':'s','cache_build_seconds':'s','cache_get_seconds':'s','dynamic_selection_seconds':'s','output_tokens':'tokens including EOS','peak_allocated_bytes':'GiB','peak_reserved_bytes':'GiB','after_ttft_seconds':'s','resume_cache_rebuild_seconds':'s'}
FIELDS=list(METRICS)
RATIOS=[(a,b,f) for _,a,b in COMPARISONS for f in ['ttft_seconds','total_seconds']]
SEED=20261116;BOOT=5000
SPANISH_REFERENCE_IDS={'1085067334266d41636126e5b9ee9519_5'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def serial(value):return json.dumps(value,ensure_ascii=False,sort_keys=True)
def csvout(name,rows,empty_fields=('issue','sample_id','source_line')):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or list(empty_fields)
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:serial(v) if isinstance(v,(list,tuple,dict)) else v for k,v in r.items()})
def ci(values):return np.quantile(values,[.025,.975],axis=0)
def nearest(length,n):
    return min([(l,k) for l in [4096,8192,16384] for k in [4,8,16]],key=lambda c:((math.log2(length)-math.log2(c[0]))**2+(math.log2(n)-math.log2(c[1]))**2,c[0],c[1]))


def bleu_batch(vectors):
    """Vectorized BLEU default corpus computation, including exp zero-match smoothing."""
    v=np.asarray(vectors,dtype=float);syslen,ref=v[...,0],v[...,1]
    hits,total=v[...,2:6],v[...,6:10]
    smooth=np.ones(syslen.shape);prec=[]
    for n in range(4):
        smooth=np.where((hits[...,n]==0)&(total[...,n]>0),smooth*2,smooth)
        with np.errstate(divide='ignore',invalid='ignore'):
            p=np.where(total[...,n]>0,np.where(hits[...,n]>0,100*hits[...,n]/total[...,n],100/(smooth*total[...,n])),0)
        prec.append(p)
    with np.errstate(divide='ignore',invalid='ignore'):
        bp=np.where(syslen>0,np.exp(np.minimum(0,1-ref/syslen)),0)
        value=bp*np.exp(np.log(np.stack(prec,axis=-1)).mean(axis=-1))
    return np.where((syslen>0)&(hits.sum(axis=-1)>0),value,0)


def main():
    manifest=read(FROZEN/'frozen.json');ids=sorted(manifest['orders']['doc2dial'])
    assert len(ids)==len(set(ids))==500
    source={r['sample_id']:r for r in read(FROZEN/'doc2dial.json')}
    assert set(ids)==set(source)
    input_checks=readcsv(AUDIT/'scoring_doc2dial_input_checks.csv')
    hot_proof=readcsv(AUDIT/'scoring_doc2dial_hot_generation_evidence.csv')
    assert len(input_checks)==500 and all(r['status']=='PASS' for r in input_checks)
    assert len(hot_proof)==1500 and all(r['status']=='PASS' for r in hot_proof)
    assert {r['sample_id'] for r in input_checks}==set(ids)
    proof={(r['sample_id'],r['method']):r for r in hot_proof}
    audit_inputs={r['sample_id']:r for r in input_checks}
    csvout('preaudit_input_checks.csv',input_checks);csvout('hot_generation_evidence.csv',hot_proof)
    csvout('history_condition_evidence.csv',readcsv(AUDIT/'scoring_doc2dial_history_condition_evidence.csv'))
    rule=read(RUN/'rule.json')
    stop=read(RUN/'snapshot/models/Llama-3.1-8B-Instruct/generation_config.json')['eos_token_id']
    stop=set(stop if isinstance(stop,list) else [stop])
    records=[];lookup={};issues=[]
    with (RUN/'attempts.jsonl').open(encoding='utf-8') as f:
        for line,text in enumerate(f,1):
            r=json.loads(text)
            if r['key']['stage']!='doc2dial':continue
            key=(r['sample_id'],r['method'],r['cache_mode'])
            assert key not in lookup and key[0] in source and key[1] in METHODS and key[2] in CACHES
            s=source[key[0]];assert r['status']=='success' and r['attempt']==1
            assert r['reference_answers']==s['answers'] and len(s['answers'])==1 and s['answers'][0].strip()
            assert r['source']==s['source'] and r['group']==s['group']
            assert r['L']==s['_length'] and r['N']==math.ceil(r['L']/7500) and r['original_document_count']==1
            assert r['block_lengths']==[min(7500,r['L']-k) for k in range(0,r['L'],7500)]
            assert 0<=r['score']<=100 and math.isfinite(r['score'])
            for field in FIELDS:
                if field=='after_ttft_seconds':continue
                assert isinstance(r[field],(int,float)) and math.isfinite(r[field]) and r[field]>=0
            assert r['total_seconds']>=r['ttft_seconds']>0
            assert len(r['output_token_ids'])==r['output_tokens']<=s['max_new_tokens']==128
            assert r['cache_hit']==(r['cache_mode']=='hot') and r['resume_cache_rebuild_seconds']==0
            if r['cache_mode']=='hot':assert r['cache_build_seconds']==0 and r['timing_condition']=='application_GPU_KV'
            else:assert r['timing_condition']=='cold'
            mapped=nearest(r['L'],r['N'])
            if r['method']=='ape_fixed':assert (r['T'],r['S'])==tuple(rule['fixed'])
            if r['method']=='ape_dynamic':assert (r['T'],r['S'])==tuple(rule['cells'][f'{mapped[0]}/{mapped[1]}'])
            if r['method'].startswith('ape_'):
                assert r['stage_calls']['query']==r['output_tokens']
                assert r['stage_calls']['prefix']==(1 if r['cache_mode']=='cold' else 0)
                assert r['stage_calls']['context']==(r['N'] if r['cache_mode']=='cold' else 0)
            if not r['answer'].strip():issues.append({'issue':'empty answer retained','sample_id':r['sample_id'],'method':r['method'],'cache':r['cache_mode'],'source_line':line})
            r['_line']=line;r['after_ttft_seconds']=r['total_seconds']-r['ttft_seconds']
            r['_cap']=r['output_tokens']==128;r['_eos']=r['output_token_ids'][-1] in stop
            records.append(r);lookup[key]=r
    assert len(records)==3000 and all((sid,m,c) in lookup for sid in ids for m,c in CONDS)
    assert all(r['started_at']>rule['frozen_at'] for r in records)
    membership=[];inputs=[];meta={};cold_hot=[]
    for sid in ids:
        s=source[sid];src=s['source'];hist=s['history'];prev=hist[-1]
        assert s['history_turn_ids']==[v['turn_id'] for v in hist]
        assert all(v['turn_id']<src['target_turn'] for v in hist)
        assert sid==f"{src['dial_id']}_{src['target_turn']-1}" and prev['turn_id']==src['target_turn']-1
        assert s['prefix']==HELPERS['CHAT_START']+tag('title',src['doc_id'])+'\t<doc_context>\t'
        expected='\t'+'\t'.join([tag('last_turn',prev['utterance'])]+[tag(v['role'],v['utterance']) for v in reversed(hist)])+HELPERS['CHAT_END']
        assert s['suffix']==expected
        users=[v for v in hist if v['role']=='user'];assert users
        repeated=[v['turn_id'] for v in hist if text2line(v['utterance'])==s['answers'][0]]
        ref=lookup[(sid,'sequential','cold')]
        mapped=nearest(ref['L'],ref['N'])
        mr={'sample_id':sid,'domain':src['domain'],'doc_id':src['doc_id'],'dial_id':src['dial_id'],'target_agent_turn_id':src['target_turn'],'reference_answer_id':sid,'preceding_turn_id':prev['turn_id'],'preceding_role':prev['role'],'last_turn_utterance':prev['utterance'],'latest_user_turn_id':users[-1]['turn_id'],'latest_user_utterance':users[-1]['utterance'],'history_turns':len(hist),'consecutive_agent_target':prev['role']=='agent','reference_repeated_in_history':bool(repeated),'matching_history_turn_ids':repeated,'dialogue_cluster':serial([src['domain'],src['doc_id'],src['dial_id']]),'document_cluster':serial([src['domain'],src['doc_id']]),'L':ref['L'],'actual_N':ref['N'],'mapped_L':mapped[0],'mapped_N':mapped[1],'dynamic_T':lookup[(sid,'ape_dynamic','cold')]['T'],'dynamic_S':lookup[(sid,'ape_dynamic','cold')]['S'],'original_document_count':1,'prefix_tokens':ref['prefix_tokens'],'request_tokens':ref['request_tokens'],'ungrounded_history_occurrences':int(audit_inputs[sid]['ungrounded_history_turn_occurrences']),'reference_answers':s['answers'],'source_file':'snapshot/datasets/frozen-main80/doc2dial.json','unit':'one target; six paired requests; L document tokens; current last_turn may be agent'}
        mr['confirmed_spanish_reference']=sid in SPANISH_REFERENCE_IDS
        meta[sid]=mr;membership.append(mr)
        inputs.append({**mr,'history_turn_ids':s['history_turn_ids'],'complete_history':hist,'prefix':s['prefix'],'suffix':s['suffix'],'document_characters':len(s['document']),'template_adaptation':s['template_adaptation']})
        for m in METHODS:
            a,b=[lookup[(sid,m,c)] for c in CACHES]
            assert a['output_token_ids']==b['output_token_ids'] and a['answer']==b['answer'] and a['reference_answers']==b['reference_answers']
            assert all(a[k]==b[k] for k in ['L','N','T','S','block_lengths','prefix_tokens','request_tokens','query_start_position'])
            assert b['started_at']>=a['ended_at']
            p=proof[(sid,m)];assert (int(p['cold_attempts_line']),int(p['hot_attempts_line']))==(a['_line'],b['_line'])
            assert abs(a['score']-b['score'])<1e-8
            cold_hot.append({'sample_id':sid,'method':m,'cold_line':a['_line'],'hot_line':b['_line'],'same_tokens':True,'same_answer':True,'cold_miss_hot_hit':True,'hot_build_seconds':b['cache_build_seconds'],'hot_rebuild_seconds':b['resume_cache_rebuild_seconds'],'hot_query_calls':b['stage_calls']['query'],'hot_output_tokens':b['output_tokens'],'cold_start_event_line':p['cold_start_event_line'],'hot_start_event_line':p['hot_start_event_line'],'status':'PASS','unit':'same-target repeated request; independent hot generation; APE-only query hooks'})
    assert len({r['dialogue_cluster'] for r in membership})==366 and len({r['document_cluster'] for r in membership})==172
    csvout('membership.csv',membership);csvout('target_inputs.csv',inputs);csvout('cold_hot_pairs.csv',cold_hot);csvout('record_issues.csv',issues)
    language_pattern=r'\b(?:puede|obtener|ayuda|pagar|escuela|capacitación|usted|necesita|beneficios|también|dependientes)\b|[áéíóúñ¿¡]'
    language_candidates=[{'sample_id':s,'reference_answer':source[s]['answers'][0],'latest_user_utterance':meta[s]['latest_user_utterance'],'confirmed_spanish':s in SPANISH_REFERENCE_IDS,'candidate_pattern':language_pattern,'unit':'targeted reference-language review, original reference retained'} for s in ids if re.search(language_pattern,source[s]['answers'][0],flags=re.IGNORECASE)]
    csvout('reference_language_candidates.csv',language_candidates)
    raw_dialogues=read(ROOT/'restored/datasets/raw-main80/doc2dial/doc2dial_dial_validation.json')
    language_evidence=[]
    for sid in sorted(SPANISH_REFERENCE_IDS):
        src=source[sid]['source'];dialogue=next(d for d in raw_dialogues['dial_data'][src['domain']][src['doc_id']] if d['dial_id']==src['dial_id'])
        target=next(t for t in dialogue['turns'] if t['turn_id']==src['target_turn'])
        assert text2line(target['utterance'])==source[sid]['answers'][0]
        language_evidence.append({'sample_id':sid,'raw_file':'restored/datasets/raw-main80/doc2dial/doc2dial_dial_validation.json','source':src,'raw_target':target,'frozen_reference':source[sid]['answers'][0],'complete_previous_history':source[sid]['history'],'latest_user_utterance':meta[sid]['latest_user_utterance'],'review':'English request and generated answers; Spanish reference confirmed from raw target; original scores retained'})
    (OUT/'sources/reference_language_evidence.json').write_text(json.dumps(language_evidence,ensure_ascii=False,indent=2),encoding='utf-8')

    # Extract 1–4gram statistics once per target/method; identical hot answers reuse them.
    stat_arr=np.zeros((500,6,10));metric_arr=np.zeros((500,6,len(FIELDS)));ratio_arr=np.zeros((500,len(RATIOS)))
    stat_rows=[];requests=[]
    for i,sid in enumerate(ids):
        for m in METHODS:
            cold=lookup[(sid,m,'cold')];v=np.array(bleu_stats(cold['answer'],source[sid]['answers']),float)
            assert np.isfinite(v).all() and (v>=0).all()
            stat_rows.append({'sample_id':sid,'method':m,'system_length_13a':int(v[0]),'reference_length_13a':int(v[1]),**{f'correct_{n+1}':int(v[2+n]) for n in range(4)},**{f'total_{n+1}':int(v[6+n]) for n in range(4)},'cold_source_line':cold['_line'],'hot_source_line':lookup[(sid,m,'hot')]['_line'],'unit':'integer sufficient statistics; same target cold/hot text identical; corpus sum before scoring'})
            for c in CACHES:
                j=CONDS.index((m,c));r=lookup[(sid,m,c)];stat_arr[i,j]=v
                metric_arr[i,j]=[r[f]/2**30 if METRICS[f]=='GiB' else r[f] for f in FIELDS]
                requests.append({**{k:meta[sid][k] for k in ['sample_id','domain','doc_id','dial_id','dialogue_cluster','document_cluster','target_agent_turn_id','reference_answer_id','consecutive_agent_target','reference_repeated_in_history','history_turns','last_turn_utterance','latest_user_utterance','mapped_L','mapped_N']},'method':m,'cache_mode':c,'task_key':r['key'],'attempt':r['attempt'],'status':r['status'],'L':r['L'],'actual_N':r['N'],'T':r['T'],'S':r['S'],'answer':r['answer'],'reference_answers':r['reference_answers'],'saved_single_target_default_corpus_bleu':r['score'],'output_token_ids':r['output_token_ids'],'cap_hit':r['_cap'],'eos_last':r['_eos'],'has_newline':'\n' in r['answer'],**{f:r[f] for f in FIELDS},**{f:r[f] for f in ['cache_hit','timing_condition','prefix_tokens','request_tokens','query_start_position','block_lengths','stage_calls','started_at','ended_at']},'source_file':'restored/output/runs/main80-20261114T041503/attempts.jsonl','source_line':r['_line'],'unit':'saved per-target BLEU0-100 for cases only; seconds; raw memory bytes; tokens including EOS'})
        ratio_arr[i]=[lookup[(sid,*a)][f]/lookup[(sid,*b)][f] for a,b,f in RATIOS]
    csvout('bleu_sufficient_statistics.csv',stat_rows);csvout('per_request_records.csv',requests)

    fixture_vectors=[[6,7,6,4,2,1,6,5,4,3],[10,10,10,9,8,7,10,9,8,7],[10,10,0,0,0,0,10,9,8,7],[2,2,2,1,0,0,2,1,0,0],[6,7,2,0,0,0,6,5,4,3]]
    validation=[]
    for name,v in zip(['upstream example component','exact corpus100','zero matches','absent4gram effective-order-off','exp smoothing'],fixture_vectors):
        expected=bleu_scalar(v);actual=float(bleu_batch(v));assert abs(actual-expected)<1e-10
        validation.append({'check':name,'statistics':v,'scalar_expected':expected,'batch_actual':actual,'absolute_difference':abs(actual-expected),'status':'PASS','unit':'BLEU0-100; scorer validation fixture'})
    summary=read(RUN/'summary.json')['doc2dial_paired']
    assert summary['paired_n']==summary['selected_n']==500 and set(summary['sample_ids'])==set(ids)
    consistency=[]
    audit_corpus={(r['method'],r['cache_mode']):r for r in readcsv(AUDIT/'scoring_doc2dial_corpus_consistency.csv')}
    for row in summary['conditions']:
        cond=(row['method'],row['cache_mode']);j=CONDS.index(cond);v=stat_arr[:,j].sum(axis=0)
        expected_stats=audit_corpus[cond]
        assert v.tolist()==[int(expected_stats['system_length']),int(expected_stats['reference_length']),*json.loads(expected_stats['correct_1_4']),*json.loads(expected_stats['total_1_4'])]
        actual=float(bleu_batch(v));original=row['corpus_bleu'];assert original['signature']==SIGNATURE and abs(actual-original['score'])<1e-8
        consistency.append({'method':cond[0],'cache_mode':cond[1],'n_targets':500,'server_corpus_bleu':original['score'],'current_corpus_bleu':actual,'absolute_difference':abs(actual-original['score']),'server_signature':original['signature'],'stats_match_preaudit':True,'unit':'BLEU0-100, six server aggregate/statistic checks'})
    csvout('scorer_validation.csv',validation);csvout('original_summary_crosscheck.csv',consistency)

    scopes=[('all500',ids),('actual_N_1',[s for s in ids if meta[s]['actual_N']==1]),('actual_N_2',[s for s in ids if meta[s]['actual_N']==2])]
    scopes += [(name,[s for s in ids if predicate(meta[s])]) for name,predicate in [
        ('previous_user',lambda x:not x['consecutive_agent_target']),('consecutive_agent',lambda x:x['consecutive_agent_target']),
        ('reference_repeat',lambda x:x['reference_repeated_in_history']),('reference_not_repeat',lambda x:not x['reference_repeated_in_history']),
        ('clean_user_no_repeat',lambda x:not x['consecutive_agent_target'] and not x['reference_repeated_in_history']),
        ('history_flagged_union',lambda x:x['consecutive_agent_target'] or x['reference_repeated_in_history']),
        ('history_1_4',lambda x:1<=x['history_turns']<=4),('history_5_8',lambda x:5<=x['history_turns']<=8),('history_9_15',lambda x:9<=x['history_turns']<=15),
    ]]
    scopes += [(f'domain_{d}',[s for s in ids if meta[s]['domain']==d]) for d in sorted({x['domain'] for x in meta.values()})]
    scopes.append(('without_confirmed_spanish_reference',[s for s in ids if s not in SPANISH_REFERENCE_IDS]))
    index={sid:i for i,sid in enumerate(ids)}
    quality=[];quality_pairs=[];efficiency=[];performance=[];diagnostics=[];scope_counts=[]
    for scope_index,(scope,scope_ids) in enumerate(scopes):
        if not scope_ids:continue
        ix=[index[s] for s in scope_ids];ss=stat_arr[ix];mm=metric_arr[ix];rr=ratio_arr[ix]
        point_stats=ss.sum(axis=0);point_bleu=bleu_batch(point_stats)
        for unit_index,unit in enumerate(['dialogue','document']):
            group_keys=[meta[s][unit+'_cluster'] for s in scope_ids];unique=sorted(set(group_keys));gindex={g:i for i,g in enumerate(unique)}
            n_groups=len(unique);sizes=np.zeros(n_groups);sums=np.zeros((n_groups,60+(6*len(FIELDS)+len(RATIOS) if scope=='all500' else 0)))
            for i,g in enumerate(group_keys):
                j=gindex[g];sizes[j]+=1
                sums[j,:60]+=ss[i].reshape(-1)
                if scope=='all500':sums[j,60:]+=np.concatenate([mm[i].reshape(-1),rr[i]])
            bscore=None;bmetric=None;bratio=None;target_counts=[]
            if n_groups>1:
                rng=np.random.default_rng(np.random.SeedSequence([SEED,scope_index,unit_index]));chunks=[]
                for start in range(0,BOOT,100):
                    draw=rng.integers(0,n_groups,size=(min(100,BOOT-start),n_groups));den=sizes[draw].sum(axis=1);summed=sums[draw].sum(axis=1)
                    target_counts.extend(den.tolist());chunks.append((summed,den))
                summed=np.concatenate([x[0] for x in chunks]);den=np.concatenate([x[1] for x in chunks])
                bscore=bleu_batch(summed[:,:60].reshape(BOOT,6,10))
                if scope=='all500':
                    bmetric=summed[:,60:60+6*len(FIELDS)].reshape(BOOT,6,len(FIELDS))/den[:,None,None]
                    bratio=summed[:,60+6*len(FIELDS):]/den[:,None]
            diag={'scope':scope,'cluster_unit':unit,'n_targets':len(scope_ids),'n_clusters':n_groups,'largest_cluster_targets':int(sizes.max()),'replicates':BOOT if n_groups>1 else 0,'seed_key':[SEED,scope_index,unit_index],'sampled_target_count_min':min(target_counts) if target_counts else '', 'sampled_target_count_max':max(target_counts) if target_counts else '', 'ci_status':'paired cluster percentile' if n_groups>1 else 'one cluster; between-source variation unavailable','unit':'all six method/cache outcomes retained; corpus stats summed, efficiency sums divided by sampled targets'}
            diagnostics.append(diag);scope_counts.append({k:diag[k] for k in ['scope','cluster_unit','n_targets','n_clusters','ci_status']})
            common={'scope':scope,'cluster_unit':unit,'n_targets':len(scope_ids),'n_clusters':n_groups,'replicates':diag['replicates'],'seed':SEED,'ci_status':diag['ci_status']}
            for j,(m,c) in enumerate(CONDS):
                lo,hi=ci(bscore[:,j]) if bscore is not None else ('','');v=point_stats[j]
                quality.append({**common,'method':m,'cache_mode':c,'corpus_bleu':float(point_bleu[j]),'ci_low':float(lo) if lo!='' else '', 'ci_high':float(hi) if hi!='' else '', 'system_length_13a':int(v[0]),'reference_length_13a':int(v[1]),'length_ratio':float(v[0]/v[1]),'brevity_penalty':float(math.exp(min(0,1-v[1]/v[0]))),'correct_1_4':v[2:6].astype(int).tolist(),'total_1_4':v[6:10].astype(int).tolist(),'precision_1_4_percent':[float(100*h/t) if t else 0 for h,t in zip(v[2:6],v[6:10])],'server_setting_signature':SIGNATURE,'local_engine':'independent standard-library/NumPy implementation; SacreBLEU package not installed','unit':'corpus BLEU0-100; all ngram counts summed before scoring'})
            for kind,a,b in COMPARISONS:
                ja,jb=CONDS.index(a),CONDS.index(b);lo,hi=ci(bscore[:,jb]-bscore[:,ja]) if bscore is not None else ('','')
                quality_pairs.append({**common,'comparison_type':kind,'reference_method':a[0],'reference_cache':a[1],'comparison_method':b[0],'comparison_cache':b[1],'delta_bleu':float(point_bleu[jb]-point_bleu[ja]),'ci_low':float(lo) if lo!='' else '', 'ci_high':float(hi) if hi!='' else '', 'unit':'paired difference of re-aggregated corpus BLEU; comparison minus reference'})
            if scope!='all500':continue
            for j,(m,c) in enumerate(CONDS):
                for k,field in enumerate(FIELDS):
                    lo,hi=ci(bmetric[:,j,k]);vv=mm[:,j,k]
                    efficiency.append({**common,'method':m,'cache_mode':c,'metric':field,'mean':float(vv.mean()),'median':float(np.median(vv)),'p90':float(np.quantile(vv,.9)),'mean_ci_low':float(lo),'mean_ci_high':float(hi),'unit':METRICS[field],'cache_condition':'fresh document encode' if c=='cold' else 'same-process GPU document KV; history/query/decode recomputed'})
            for kind,a,b in COMPARISONS:
                ja,jb=CONDS.index(a),CONDS.index(b)
                for field in ['ttft_seconds','total_seconds']:
                    k=FIELDS.index(field);ir=RATIOS.index((a,b,field));dl,dh=ci(bmetric[:,jb,k]-bmetric[:,ja,k]);rl,rh=ci(bratio[:,ir]);ml,mh=ci(bmetric[:,ja,k]/bmetric[:,jb,k])
                    performance.append({**common,'comparison_type':kind,'reference_method':a[0],'reference_cache':a[1],'comparison_method':b[0],'comparison_cache':b[1],'metric':field,'reference_mean_s':float(mm[:,ja,k].mean()),'comparison_mean_s':float(mm[:,jb,k].mean()),'delta_comparison_minus_reference_s':float((mm[:,jb,k]-mm[:,ja,k]).mean()),'delta_ci_low_s':float(dl),'delta_ci_high_s':float(dh),'paired_speedup_mean':float(rr[:,ir].mean()),'paired_speedup_median':float(np.median(rr[:,ir])),'paired_speedup_p90':float(np.quantile(rr[:,ir],.9)),'speedup_mean_ci_low':float(rl),'speedup_mean_ci_high':float(rh),'ratio_of_means':float(mm[:,ja,k].mean()/mm[:,jb,k].mean()),'ratio_of_means_ci_low':float(ml),'ratio_of_means_ci_high':float(mh),'unit':'seconds; speedup reference/comparison (>1 comparison faster)'})
    for name,rows in [('quality.csv',quality),('quality_paired.csv',quality_pairs),('efficiency_descriptive.csv',efficiency),('efficiency_paired.csv',performance),('bootstrap_diagnostics.csv',diagnostics),('scope_counts.csv',scope_counts)]:csvout(name,rows)

    limits=[];conditioned=[];equivalence=[];length_groups=[];target_pairs=[]
    for m,c in CONDS:
        rows=[r for r in requests if r['method']==m and r['cache_mode']==c]
        limits.append({'method':m,'cache_mode':c,'n_requests':500,'max_new_tokens':128,'cap_hits':sum(r['cap_hit'] for r in rows),'cap_fraction':np.mean([r['cap_hit'] for r in rows]),'eos_last_count':sum(r['eos_last'] for r in rows),'eos_at_cap_count':sum(r['cap_hit'] and r['eos_last'] for r in rows),'multiline_count':sum(r['has_newline'] for r in rows),'output_tokens_mean':np.mean([r['output_tokens'] for r in rows]),'unit':'actual requests; token counts include EOS'})
        for cap in [False,True]:
            chosen=[r for r in rows if r['cap_hit']==cap]
            if not chosen:continue
            st=np.sum([stat_arr[index[r['sample_id']],CONDS.index((m,c))] for r in chosen],axis=0)
            conditioned.append({'method':m,'cache_mode':c,'cap_hit':cap,'n_targets':len(chosen),'corpus_bleu':float(bleu_batch(st)),'mean_output_tokens':np.mean([r['output_tokens'] for r in chosen]),'mean_total_s':np.mean([r['total_seconds'] for r in chosen]),'mean_ttft_s':np.mean([r['ttft_seconds'] for r in chosen]),'system_reference_length_ratio':float(st[0]/st[1]),'unit':'descriptive generation-length subgroup corpus BLEU0-100/s/tokens'})
    for a,b in combinations(METHODS,2):
        pairs=[(lookup[(s,a,'cold')],lookup[(s,b,'cold')]) for s in ids]
        equivalence.append({'reference_method':a,'comparison_method':b,'n_targets':500,'same_tokens':sum(x['output_token_ids']==y['output_token_ids'] for x,y in pairs),'same_auxiliary_scores':sum(abs(x['score']-y['score'])<1e-12 for x,y in pairs),'unit':'target counts; cold/hot tokens already identical'})
        for label,predicate in [('all',lambda x,y:True),('same_tokens',lambda x,y:x['output_token_ids']==y['output_token_ids']),('different_tokens',lambda x,y:x['output_token_ids']!=y['output_token_ids']),('comparison_longer',lambda x,y:y['output_tokens']>x['output_tokens']),('comparison_shorter',lambda x,y:y['output_tokens']<x['output_tokens'])]:
            chosen=[(x,y) for x,y in pairs if predicate(x,y)]
            if chosen:length_groups.append({'reference_method':a,'comparison_method':b,'group':label,'n_targets':len(chosen),'mean_reference_tokens':np.mean([x['output_tokens'] for x,y in chosen]),'mean_comparison_tokens':np.mean([y['output_tokens'] for x,y in chosen]),'mean_cold_total_delta_s':np.mean([y['total_seconds']-x['total_seconds'] for x,y in chosen]),'mean_hot_total_delta_s':np.mean([lookup[(x['sample_id'],b,'hot')]['total_seconds']-lookup[(x['sample_id'],a,'hot')]['total_seconds'] for x,y in chosen]),'unit':'descriptive target groups; timescomparison-reference; corpus BLEU not averaged here'})
    for sid in ids:
        a,b=[lookup[(sid,m,'cold')] for m in ['ape_fixed','ape_dynamic']]
        target_pairs.append({**meta[sid],'fixed_T':a['T'],'fixed_S':a['S'],'dynamic_T':b['T'],'dynamic_S':b['S'],'fixed_auxiliary_bleu':a['score'],'dynamic_auxiliary_bleu':b['score'],'auxiliary_delta_bleu':b['score']-a['score'],'fixed_original_answer':a['answer'],'dynamic_original_answer':b['answer'],'reference_answer':source[sid]['answers'][0],'fixed_output_tokens':a['output_tokens'],'dynamic_output_tokens':b['output_tokens'],'fixed_cold_line':a['_line'],'dynamic_cold_line':b['_line'],'fixed_hot_line':lookup[(sid,'ape_fixed','hot')]['_line'],'dynamic_hot_line':lookup[(sid,'ape_dynamic','hot')]['_line'],'fixed_cold_total_s':a['total_seconds'],'dynamic_cold_total_s':b['total_seconds'],'fixed_hot_total_s':lookup[(sid,'ape_fixed','hot')]['total_seconds'],'dynamic_hot_total_s':lookup[(sid,'ape_dynamic','hot')]['total_seconds'],'same_tokens':a['output_token_ids']==b['output_token_ids'],'unit':'auxiliary one-target corpus-default BLEU for selection; primary result is aggregate corpus BLEU'})
    selected=[];case_counts=[]
    for label,predicate,sign in [('改善',lambda x:x>1e-12,-1),('退化',lambda x:x< -1e-12,1),('持平',lambda x:abs(x)<=1e-12,1)]:
        chosen=sorted([r for r in target_pairs if predicate(r['auxiliary_delta_bleu'])],key=lambda r:(sign*r['auxiliary_delta_bleu'],r['sample_id']))
        case_counts.append({'case_type':label,'n_targets':len(chosen),'denominator':500,'unit':'auxiliary one-target scores; not corpus BLEU win/loss'})
        if chosen:selected.append({'case_type':label,**chosen[0],'selection_rule':'maximum positive/negative saved one-target default-corpus BLEU delta; zero/ties sample_id ascending'})
    for name,rows in [('generation_limits.csv',limits),('generation_cap_conditioned.csv',conditioned),('output_equivalence.csv',equivalence),('paired_output_length_groups.csv',length_groups),('dynamic_fixed_per_target.csv',target_pairs),('selected_cases.csv',selected),('case_type_counts.csv',case_counts)]:csvout(name,rows)
    csvout('generation_cap_records.csv',[r for r in requests if r['cap_hit']])
    csvout('model_load_workers.csv',[r for r in readcsv(AUDIT/'runtime_workers.csv') if r['stage']=='doc2dial'])
    csvout('server_environment.csv',readcsv(AUDIT/'runtime_environment.csv'))
    (OUT/'sources/original_frozen_rule.json').write_text(json.dumps(rule,ensure_ascii=False,indent=2),encoding='utf-8')
    main_quality=[r for r in quality if r['scope']=='all500' and r['cluster_unit']=='dialogue']
    primary_pairs=[r for r in quality_pairs if r['scope']=='all500' and r['reference_method']=='ape_fixed' and r['comparison_method']=='ape_dynamic']
    result={'experiment_stage':8,'status':'COMPLETED','research_status':'SUBMITTED','n_targets':500,'n_records':3000,'n_dialogues':366,'n_documents':172,'n_conditions':6,'main_cluster_unit':'dialogue','sensitivity_cluster_unit':'document','main_quality':main_quality,'primary_dynamic_fixed':primary_pairs,'N1_targets':sum(r['actual_N']==1 for r in membership),'N2_targets':sum(r['actual_N']==2 for r in membership),'consecutive_agent_targets':sum(r['consecutive_agent_target'] for r in membership),'reference_repeated_targets':sum(r['reference_repeated_in_history'] for r in membership),'both_history_flags_targets':sum(r['consecutive_agent_target'] and r['reference_repeated_in_history'] for r in membership),'clean_user_no_repeat_targets':sum(not r['consecutive_agent_target'] and not r['reference_repeated_in_history'] for r in membership),'ungrounded_history_occurrences':sum(r['ungrounded_history_occurrences'] for r in membership),'hot_hits':1500,'hot_regeneration_evidence_pass':1500,'cold_hot_identical_tokens_pairs':1500,'resume_cache_rebuild_requests':0,'scorer_engine':'independent pre-audit helpers and vectorized NumPy sufficient statistics','scorer_package_installed':importlib.util.find_spec('sacrebleu') is not None,'server_signature':SIGNATURE,'sufficient_statistics_extracted_unique_answers':1500,'single_target_scores_recomputed':0,'seed':SEED,'replicates':BOOT,'rule_modified':False,'next_stage_executed':False}
    result['confirmed_spanish_reference_targets']=len(SPANISH_REFERENCE_IDS)
    result['reference_language_candidate_count']=len(language_candidates)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'STAGE_STATUS.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ['main_quality','primary_dynamic_fixed']},ensure_ascii=False,indent=2),encoding='utf-8')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env={'analysis_commit':commit,'reference_commit':REFERENCE,'changes_from_reference':subprocess.check_output(['git','diff','--name-only',f'{REFERENCE}..HEAD'],cwd=ROOT,text=True).splitlines(),'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'matplotlib':version('matplotlib'),'new_dependencies':[],'local_sacrebleu_installed':result['scorer_package_installed'],'scorer_engine':result['scorer_engine'],'comparison_server_signature':SIGNATURE,'scorer_helpers':'pc-analysis-output/scripts/audit_stage1_scoring.py pure functions loaded with main disabled','seed':SEED,'replicates':BOOT,'rng':'SeedSequence([20261116, scope_index, unit_index]); unit0 dialogue,1 document; sorted cluster keys;5000 draws of same cluster count with replacement','bootstrap_quality':'sum sampled cluster1–4gram counts/lengths then recompute corpus BLEU each replicate; six outcomes retained','bootstrap_efficiency':'sum metrics within sampled clusters divided by sampled target count; paired ratios sampled jointly','one_cluster_subgroup':'CI left empty: between-source variation not observed; point corpus still reported','all_outputs_cache_temp':str(OUT)}
    (OUT/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'stage':8,'main_quality':[{k:r[k] for k in ['method','cache_mode','corpus_bleu','ci_low','ci_high']} for r in main_quality],'primary_dynamic_fixed':primary_pairs,'history_flags':{k:result[k] for k in ['consecutive_agent_targets','reference_repeated_targets','clean_user_no_repeat_targets']},'status':'SUBMITTED'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
