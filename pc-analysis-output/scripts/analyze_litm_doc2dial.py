"""Recompute LITM/Doc2Dial on CPU; input evidence is read-only.
SacreBLEU 2.4.3 13a/exp corpus-BLEU defaults independently implemented.
Source: mjpost/sacrebleu v2.4.3 tokenizers/tokenizer_{13a,re}.py and metrics/bleu.py.
"""
from __future__ import annotations
import argparse, collections, csv, gzip, json, math, re, string, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=20261116
METHODS=('sequential','ape_fixed','ape_dynamic')
POSITIONS=(0,9,19)
OFFICIAL_SIGNATURE='nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3'
RE_TOKEN=[(re.compile(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])'),r' \1 '),
 (re.compile(r'([^0-9])([\.,])'),r'\1 \2 '),
 (re.compile(r'([\.,])([^0-9])'),r' \1 \2'),(re.compile(r'([0-9])(-)'),r'\1 \2 ')]
def norm(s):
 s=''.join(c for c in s.lower() if c not in string.punctuation)
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def em(p,refs): return float(any(norm(r) in norm(p) for r in refs))
def tokenize13a(s):
 s=s.rstrip().replace('<skipped>','').replace('-\n','').replace('\n',' ')
 for a,b in [('&quot;','"'),('&amp;','&'),('&lt;','<'),('&gt;','>')]: s=s.replace(a,b)
 s=' '+s+' '
 for rx,repl in RE_TOKEN:s=rx.sub(repl,s)
 return s.split()
def bleu_stats(pred,ref):
 p,r=tokenize13a(pred),tokenize13a(ref)
 v=[len(p),len(r)]
 cs=[];ts=[]
 for n in range(1,5):
  a=collections.Counter(tuple(p[i:i+n]) for i in range(len(p)-n+1))
  b=collections.Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1))
  cs.append(sum((a&b).values()));ts.append(sum(a.values()))
 return np.array(v+cs+ts,dtype=np.float64)
def bleu_from_stats(v):
 v=np.asarray(v);p,r=v[:2];c=v[2:6];t=v[6:10]
 if p==0 or not np.any(c):return 0.0
 bp=math.exp(1-r/p) if p<r else 1.0
 ps=[];smooth=1.0
 for a,b in zip(c,t):
  if b==0:return 0.0
  if a==0:smooth*=2;ps.append(100/(smooth*b))
  else:ps.append(100*a/b)
 return bp*math.exp(sum(math.log(x) for x in ps)/4)
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def text2line(s):return s.replace('\n','\t').replace('\r','\t').strip()
def tag(role,s):return f'<{role}>\t{text2line(s)}'
def dump_csv(path,rows):
 rows=list(rows)
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
def pct(v):return [float(x) for x in np.percentile(v,[2.5,97.5])]
def bootstrap_means(arr,rng,reps):
 n=len(arr);out=np.empty((reps,)+arr.shape[1:])
 for k in range(reps):out[k]=arr[rng.integers(0,n,n)].mean(axis=0)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--restored',type=Path,default=Path(__file__).resolve().parents[2]/'restored');ap.add_argument('--replicates',type=int,default=2000)
 args=ap.parse_args();base=args.restored.resolve();out=Path(__file__).resolve().parents[1]
 for p in ('tables','reports','figures'):(out/p).mkdir(exist_ok=True)
 run=base/'output/runs/main80-20261114T041503';tables=out/'tables';reports=out/'reports';figs=out/'figures'
 rng=np.random.default_rng(SEED);issues=[];rows=[];allkeys=collections.defaultdict(list)
 with (run/'attempts.jsonl').open(encoding='utf-8') as f:
  for line_no,line in enumerate(f,1):
   r=json.loads(line)
   if r.get('dataset') not in ('litm','doc2dial'):continue
   r['_line']=line_no;allkeys[json.dumps(r['key'],sort_keys=True)].append(r)
 for key,attempts in allkeys.items():
  valid=[r for r in attempts if r['status']=='success']
  if len(valid)>1:issues.append({'kind':'duplicate_success','key':key,'lines':[r['_line'] for r in valid]})
  if valid:rows.append(valid[-1])
  else:issues.append({'kind':'no_success','key':key,'lines':[r['_line'] for r in attempts]})
 litm=js(base/'datasets/frozen-main80/litm.json');doc=js(base/'datasets/frozen-main80/doc2dial.json')
 lf={(r['sample_id'],r['position']):r for r in litm};df={r['sample_id']:r for r in doc}
 lr=[r for r in rows if r['dataset']=='litm'];dr=[r for r in rows if r['dataset']=='doc2dial']
 evidence=[];data_checks=[]
 for r in rows:
  frozen=lf[(r['sample_id'],r['position'])] if r['dataset']=='litm' else df[r['sample_id']]
  if r['reference_answers']!=frozen['answers']:issues.append({'kind':'reference_mismatch','line':r['_line'],'sample_id':r['sample_id']})
  for field in ('answer','reference_answers'):
   if not r.get(field):issues.append({'kind':'empty_'+field,'line':r['_line']})
 # Audit actual LITM raw document contents and position conditions.
 cfg=js(base/'data_sources.json');raw_litm={}
 selected_questions={' '.join(r['question'].casefold().split()) for r in litm}
 for pos,path in cfg['litm'].items():
  q={}
  with gzip.open(base/path,'rt',encoding='utf-8') as f:
   for line in f:
    x=json.loads(line);key=' '.join(x['question'].casefold().split())
    if key in selected_questions:q[key]=x
  raw_litm[int(pos)]=q
 questions=sorted({r['sample_id'] for r in litm});litm_raw_bad=[]
 for sid in questions:
  fro=[lf[(sid,p)] for p in POSITIONS];q=' '.join(fro[0]['question'].casefold().split());raw=[raw_litm[p][q] for p in POSITIONS]
  sets=[sorted((x['title'],x['text']) for x in r['ctxs']) for r in raw]
  if any(len(x['ctxs'])!=20 or not x['ctxs'][p].get('isgold') or sum(bool(d.get('isgold')) for d in x['ctxs'])!=1 for p,x in zip(POSITIONS,raw)) or any(s!=sets[0] for s in sets):litm_raw_bad.append(sid)
  for p,f,x in zip(POSITIONS,fro,raw):
   expected='\n'.join(f'Document [{j+1}](Title: {d["title"]}) {d["text"]}' for j,d in enumerate(x['ctxs']))
   if f['document']!=expected or f['answers']!=x['answers']:litm_raw_bad.append(f'{sid}@{p}')
 data_checks.append({'check':'LITM raw questions/documents/gold positions/frozen input/answers','n':len(questions),'issues':len(litm_raw_bad),'detail':litm_raw_bad[:10]})
 # Audit complete Doc2Dial histories, current question and corpus references by ID.
 raw_dial=js(base/cfg['doc2dial_dialogues']);raw_docs=js(base/cfg['doc2dial_documents']);raw_targets={}
 for domain,docs in raw_dial['dial_data'].items():
  for docid,dials in docs.items():
   for d in dials:
    for i,t in enumerate(d['turns']):
     if t['role']=='agent':raw_targets[f'{d["dial_id"]}_{t["turn_id"]-1}']=(domain,docid,d,t,i)
 hist_bad=[];template_bad=[];ref_bad=[];history_extra=0
 for r in doc:
  domain,docid,d,t,i=raw_targets[r['sample_id']];previous=d['turns'][:i]
  expected_hist=[{'turn_id':x['turn_id'],'role':x['role'],'utterance':x['utterance']} for x in previous]
  if r['history']!=expected_hist or r['history_turn_ids']!=[x['turn_id'] for x in previous] or previous[-1]['role']!='user' or previous[-1]['turn_id']!=t['turn_id']-1:hist_bad.append(r['sample_id'])
  history_extra+=sum(not x['references'] for x in previous)
  expected_suffix='\t'+'\t'.join([tag('last_turn',previous[-1]['utterance'])]+[tag(x['role'],x['utterance']) for x in reversed(previous)])+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  expected_prefix='<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'+tag('title',docid)+'\t<doc_context>\t'
  if r['suffix']!=expected_suffix or r['prefix']!=expected_prefix or r['document']!=text2line(raw_docs['doc_data'][domain][docid]['doc_text']):template_bad.append(r['sample_id'])
  if r['answers']!=[text2line(t['utterance'])] or not t['references']:ref_bad.append(r['sample_id'])
 data_checks += [{'check':'Doc2Dial complete earlier history','n':len(doc),'issues':len(hist_bad),'detail':hist_bad[:10]},
 {'check':'Doc2Dial common adapted full document/template','n':len(doc),'issues':len(template_bad),'detail':template_bad[:10]},
 {'check':'Doc2Dial official target/reference ID and answer','n':len(doc),'issues':len(ref_bad),'detail':ref_bad[:10]}]
 dump_csv(tables/'litm_doc2dial_data_checks.csv',data_checks)
 # Question-level paired LITM: three positions are retained in every replicate.
 lb={(r['method'],r['position'],r['sample_id']):r for r in lr};ids=sorted(set.intersection(*({r['sample_id'] for r in lr if r['method']==m and r['position']==p} for m in METHODS for p in POSITIONS)))
 arr=np.zeros((len(ids),3,3,2));litm_scores=[]
 for i,sid in enumerate(ids):
  for j,m in enumerate(METHODS):
   for k,p in enumerate(POSITIONS):
    r=lb[m,p,sid];frozen=lf[sid,p];full=em(r['answer'],frozen['answers']);first=em(r['answer'].split('\n')[0].strip(),frozen['answers']);arr[i,j,k]=[full,first]
    litm_scores.append({'sample_id':sid,'method':m,'position':p,'attempts_line':r['_line'],'recorded_score':r['score'],'full_em':full,'official_first_line_em':first,'changed':full!=first,'N':r['N'],'L':r['L'],'T':r['T'],'S':r['S']})
    if full!=first:evidence.append({'dataset':'litm','sample_id':sid,'method':m,'position':p,'attempts_line':r['_line'],'answer':r['answer'],'references':frozen['answers'],'full_em':full,'official_first_line_em':first})
 boot=bootstrap_means(arr,rng,args.replicates);litm_result=[];litm_diff=[]
 for z,metric in enumerate(('campaign_full_answer_em','official_first_line_em')):
  for j,m in enumerate(METHODS):
   for k,p in list(enumerate(POSITIONS))+[(None,'overall')]:
    a=arr[:,j,:,z].mean(axis=1) if k is None else arr[:,j,k,z];b=boot[:,j,:,z].mean(axis=1) if k is None else boot[:,j,k,z];ci=pct(b)
    litm_result.append({'metric':metric,'method':m,'position':p,'n_questions':len(ids),'n_position_generations':len(ids)*3 if k is None else len(ids),'score_raw':float(a.mean()),'score_percent':100*float(a.mean()),'ci_low':100*ci[0],'ci_high':100*ci[1],'unit':'percent','bootstrap_unit':'question; keep all three positions','seed':SEED,'replicates':args.replicates})
  for a,b in ((1,0),(2,0),(2,1)):
   for k,p in list(enumerate(POSITIONS))+[(None,'overall')]:
    d=arr[:,a,:,z]-arr[:,b,:,z];bd=boot[:,a,:,z]-boot[:,b,:,z]
    dv=d.mean(axis=1) if k is None else d[:,k];bv=bd.mean(axis=1) if k is None else bd[:,k];ci=pct(bv)
    litm_diff.append({'metric':metric,'method':METHODS[a],'reference_method':METHODS[b],'position':p,'n_questions':len(ids),'difference':100*float(dv.mean()),'ci_low':100*ci[0],'ci_high':100*ci[1],'wins':int((dv>1e-12).sum()),'ties':int((abs(dv)<1e-12).sum()),'losses':int((dv < -1e-12).sum()),'unit':'percentage_point','bootstrap_unit':'question; keep all three positions','seed':SEED,'replicates':args.replicates})
 dump_csv(tables/'litm_scores.csv',litm_scores);dump_csv(tables/'litm_quality.csv',litm_result);dump_csv(tables/'litm_paired.csv',litm_diff);dump_csv(tables/'litm_scoring_difference_evidence.csv',evidence)
 # Doc2Dial complete six-condition pairing and sufficient statistics.
 conditions=[(m,c) for m in METHODS for c in ('cold','hot')];db={(r['method'],r['cache_mode'],r['sample_id']):r for r in dr};dids=sorted(set.intersection(*({r['sample_id'] for r in dr if r['method']==m and r['cache_mode']==c} for m,c in conditions)))
 stats=np.zeros((len(dids),6,10));doc_scores=[];cold_hot=[];score_bad=[]
 for i,sid in enumerate(dids):
  for j,(m,c) in enumerate(conditions):
   r=db[m,c,sid];f=df[sid];stats[i,j]=bleu_stats(r['answer'],f['answers'][0]);score=bleu_from_stats(stats[i,j])
   if abs(score-r['score'])>1e-8:score_bad.append({'sample_id':sid,'method':m,'cache':c,'line':r['_line'],'recorded':r['score'],'recomputed':score})
   doc_scores.append({'sample_id':sid,'dial_id':f['source']['dial_id'],'doc_id':f['source']['doc_id'],'method':m,'cache_mode':c,'attempts_line':r['_line'],'sample_bleu_diagnostic':score,**{f'stat_{k}':float(v) for k,v in enumerate(stats[i,j])}})
  for m in METHODS:
   a,b=db[m,'cold',sid],db[m,'hot',sid]
   cold_hot.append({'sample_id':sid,'method':m,'tokens_equal':a['output_token_ids']==b['output_token_ids'],'answers_equal':a['answer']==b['answer'],'hot_cache_hit':b['cache_hit'],'hot_query_calls':b['stage_calls']['query'],'hot_timing_condition':b['timing_condition'],'cold_ttft_seconds':a['ttft_seconds'],'hot_ttft_seconds':b['ttft_seconds'],'cold_total_seconds':a['total_seconds'],'hot_total_seconds':b['total_seconds'],'ttft_speedup':a['ttft_seconds']/b['ttft_seconds'],'total_speedup':a['total_seconds']/b['total_seconds']})
 dump_csv(tables/'doc2dial_sample_scores.csv',doc_scores);dump_csv(tables/'doc2dial_cold_hot_pairs.csv',cold_hot);dump_csv(tables/'doc2dial_score_mismatch.csv',score_bad)
 doc_result=[];doc_diff=[]
 actual=np.array([bleu_from_stats(v) for v in stats.sum(axis=0)])
 for grouping in ('dialogue','document'):
  keys=[df[sid]['source']['dial_id'] if grouping=='dialogue' else (df[sid]['source']['domain'],df[sid]['source']['doc_id']) for sid in dids]
  groups=sorted(set(keys));gb=np.stack([stats[np.array([k==g for k in keys])].sum(axis=0) for g in groups]);bs=np.zeros((args.replicates,6))
  for b in range(args.replicates):
   st=gb[rng.integers(0,len(groups),len(groups))].sum(axis=0);bs[b]=[bleu_from_stats(v) for v in st]
  for j,(m,c) in enumerate(conditions):
   ci=pct(bs[:,j]);v=stats[:,j].sum(axis=0)
   doc_result.append({'metric':'corpus_BLEU','method':m,'cache_mode':c,'n_requests':len(dids),'n_clusters':len(groups),'bootstrap_unit':grouping,'score':float(actual[j]),'ci_low':ci[0],'ci_high':ci[1],'unit':'BLEU_0_100','signature':OFFICIAL_SIGNATURE,'implementation':'independent 13a+exp sufficient statistics; compared to all saved per-request BLEU and summary corpus BLEU','sys_len':v[0],'ref_len':v[1],'seed':SEED,'replicates':args.replicates})
  for c in ('cold','hot'):
   for ma,mb in (('ape_fixed','sequential'),('ape_dynamic','sequential'),('ape_dynamic','ape_fixed')):
    a,b=conditions.index((ma,c)),conditions.index((mb,c));ci=pct(bs[:,a]-bs[:,b]);doc_diff.append({'metric':'corpus_BLEU','method':ma,'reference_method':mb,'cache_mode':c,'n_requests':len(dids),'n_clusters':len(groups),'bootstrap_unit':grouping,'difference':float(actual[a]-actual[b]),'ci_low':ci[0],'ci_high':ci[1],'unit':'BLEU_point','seed':SEED,'replicates':args.replicates})
 dump_csv(tables/'doc2dial_quality.csv',doc_result);dump_csv(tables/'doc2dial_paired.csv',doc_diff)
 # Cross-check independent corpus recomputation against saved summary fields.
 summary=js(run/'summary.json');summary_checks=[]
 for r in summary['doc2dial_paired']['conditions']:
  j=conditions.index((r['method'],r['cache_mode']));v=r['corpus_bleu']['score'];summary_checks.append({'dataset':'doc2dial','method':r['method'],'cache_mode':r['cache_mode'],'recorded':v,'recomputed':float(actual[j]),'difference':float(actual[j]-v),'recorded_signature':r['corpus_bleu']['signature']})
 dump_csv(tables/'doc2dial_summary_checks.csv',summary_checks)
 # Deterministic cases: largest gain/loss; no-change smallest stable sample ID.
 cases=[]
 for dataset in ('litm','doc2dial'):
  for ma,mb in (('ape_fixed','sequential'),('ape_dynamic','sequential'),('ape_dynamic','ape_fixed')):
   items=[]
   if dataset=='litm':
    for sid in ids:
     for p in POSITIONS:
      a,b=lb[ma,p,sid],lb[mb,p,sid];ref=lf[sid,p]['answers'];delta=em(a['answer'].split('\n')[0].strip(),ref)-em(b['answer'].split('\n')[0].strip(),ref);items.append((delta,sid,p,a,b,ref))
   else:
    for sid in dids:
     a,b=db[ma,'cold',sid],db[mb,'cold',sid];ref=df[sid]['answers'];delta=bleu_from_stats(bleu_stats(a['answer'],ref[0]))-bleu_from_stats(bleu_stats(b['answer'],ref[0]));items.append((delta,sid,None,a,b,ref))
   for label,sign in [('improvement',1),('degradation',-1),('unchanged',0)]:
    eligible=[v for v in items if (v[0]>1e-12 if sign==1 else v[0]<-1e-12 if sign==-1 else abs(v[0])<=1e-12)]
    if not eligible:continue
    v=sorted(eligible,key=lambda v:(-sign*v[0],v[1],v[2] or 0))[0];d,sid,p,a,b,ref=v
    cases.append({'dataset':dataset,'case_class':label,'method':ma,'reference_method':mb,'sample_id':sid,'position':p,'selection':'largest signed metric delta; tie sample_id then position; unchanged smallest sample_id','case_metric':'official_first_line_EM' if dataset=='litm' else 'single_request_BLEU_diagnostic','difference':d,'T':a['T'],'S':a['S'],'reference_T':b['T'],'reference_S':b['S'],'answer':a['answer'],'reference_method_answer':b['answer'],'gold_answers':ref,'attempts_line':a['_line'],'reference_attempts_line':b['_line']})
 dump_csv(tables/'litm_doc2dial_cases.csv',cases)
 # Figures, with exactly the CSV values used in the report.
 plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
 fig,ax=plt.subplots(figsize=(7.2,4.2))
 for j,m in enumerate(METHODS):
  y=100*arr[:,j,:,1].mean(axis=0);low=np.percentile(100*boot[:,j,:,1],2.5,axis=0);high=np.percentile(100*boot[:,j,:,1],97.5,axis=0);ax.errorbar(POSITIONS,y,yerr=[y-low,high-y],marker='o',capsize=4,label=m)
 ax.set(xlabel='Gold document position (zero-based)',ylabel='Official first-line EM (%)',title=f'LITM paired {len(ids)} questions; 95% question bootstrap CI');ax.set_xticks(POSITIONS);ax.legend();ax.grid(alpha=.2);fig.tight_layout()
 for ext in ('png','svg'):fig.savefig(figs/f'litm_official_em.{ext}',dpi=180)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.2,4.2));rr=[r for r in doc_result if r['bootstrap_unit']=='dialogue' and r['cache_mode']=='cold'];ys=[r['score'] for r in rr];ax.bar(range(3),ys,color=['#446c9c','#73a7a1','#d6a058']);ax.errorbar(range(3),ys,yerr=[[r['score']-r['ci_low'] for r in rr],[r['ci_high']-r['score'] for r in rr]],fmt='none',ecolor='black',capsize=4);ax.set_xticks(range(3),METHODS);ax.set(ylabel='Corpus BLEU (0–100)',title=f'Doc2Dial {len(dids)} targets; 95% dialogue bootstrap CI');fig.tight_layout()
 for ext in ('png','svg'):fig.savefig(figs/f'doc2dial_bleu.{ext}',dpi=180)
 plt.close(fig)
 metadata={'seed':SEED,'replicates':args.replicates,'litm_generations':len(lr),'litm_paired_questions':len(ids),'litm_raw_check_failures':litm_raw_bad,'litm_changed_generations':len(evidence),'litm_changed_questions':len({r['sample_id'] for r in evidence}),'doc2dial_generations':len(dr),'doc2dial_paired_targets':len(dids),'doc2dial_unique_dialogues':len({r['source']['dial_id'] for r in doc}),'doc2dial_unique_documents':len({(r['source']['domain'],r['source']['doc_id']) for r in doc}),'doc2dial_history_ungrounded_turn_occurrences_retained':history_extra,'doc2dial_history_failures':hist_bad,'doc2dial_template_failures':template_bad,'doc2dial_reference_failures':ref_bad,'doc2dial_per_request_bleu_mismatches':score_bad,'summary_max_bleu_difference':max(abs(r['difference']) for r in summary_checks),'cold_hot_token_mismatches':sum(not r['tokens_equal'] for r in cold_hot),'hot_cache_misses':sum(not r['hot_cache_hit'] for r in cold_hot),'issues':issues,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'python':sys.version,'signature':OFFICIAL_SIGNATURE}
 (reports/'litm_doc2dial_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# LITM 与 Doc2Dial 评分及输入审核','',f'CPU 分析随机种子 {SEED}，配对聚类 Bootstrap {args.replicates} 次。输入路径 `{base}`。','', '## 1 LITM 官方评分差异','',f'我们复核 {len(lr)} 条生成，得到 {len(ids)} 个完整配对问题，每题保留位置 0、9、19 与三方法。官方评测入口先取第一行，再应用 best_subspan_em；campaign 使用完整答案。重新评分改变 {len(evidence)} 条生成、涉及 {metadata["litm_changed_questions"]} 个问题。原始记录保持原样，正式表以官方首行 EM 为主，全文 EM 另列对照。','', '|方法|位置0 EM %|位置9 EM %|位置19 EM %|总体 EM %|','|---|---:|---:|---:|---:|']
 for j,m in enumerate(METHODS):lines.append('|'+m+'|'+'|'.join(f'{100*v:.4f}' for v in arr[:,j,:,1].mean(axis=0))+f'|{100*arr[:,j,:,1].mean():.4f}|')
 lines+=['',f'原始文档集合、金标准位置、冻结文本和参考答案核对异常 {len(litm_raw_bad)} 项。最小受影响记录及行号见 `tables/litm_scoring_difference_evidence.csv`。','', '## 2 Doc2Dial 语料 BLEU 与共同模板','',f'我们复核 {len(dr)} 条生成、{len(dids)} 个六条件完整配对目标，来自 {metadata["doc2dial_unique_dialogues"]} 个对话、{metadata["doc2dial_unique_documents"]} 篇资料。完整历史、冻结输入模板及参考 ID 的异常数量依次为 {len(hist_bad)}、{len(template_bad)}、{len(ref_bad)}。历史中保留了 {history_extra} 次无 grounding 的前序轮次出现。','', '|方法|冷请求 BLEU|热请求 BLEU|','|---|---:|---:|']
 for m in METHODS:lines.append(f'|{m}|{actual[conditions.index((m,"cold"))]:.6f}|{actual[conditions.index((m,"hot"))]:.6f}|')
 lines+=['',f'独立 13a 分词和 exp 平滑充分统计实现逐条对照 {len(dr)} 条保存 BLEU，差异超过 1e-8 的记录 {len(score_bad)} 条；与保存 corpus BLEU 最大绝对差 {metadata["summary_max_bleu_difference"]:.3g}。评分签名 `{OFFICIAL_SIGNATURE}`。本地复算使用独立实现，环境未安装 SacreBLEU；服务器原库版本为 2.4.3，官方历史依赖为 1.4.3。','', '我们采用目标 ID 连接参考：`{dial_id}_{agent_turn_id-1}`。共同模板保留官方 text2line、角色标签、last_turn 及逆序历史，将全文资料移到历史前面，加入统一 Llama chat envelope，并保留全部前序历史。官方历史模板在累积历史前跳过无 references 轮次，且将全文资料追加到历史后面。此次比较三种方法共享本次适配模板。','',f'热请求的缓存键包括方法、资料 prefix、实际编码块及参数，历史和当前请求随 suffix 重新 forward，再逐 token 贪心生成。{len(cold_hot)} 组冷/热输出 token 对照差异 {metadata["cold_hot_token_mismatches"]} 组，热缓存未命中 {metadata["hot_cache_misses"]} 组。','', '## 3 统计与案例','', 'LITM 按问题重采样，三个位置和各方法共同保留；总体先对每问题三个位置等权平均。Doc2Dial 主区间按对话抽取相同数量的对话簇、簇内所有选中目标共同保留，资料聚类作为敏感性结果；每次重采样都汇总 n-gram 匹配/总数、系统/参考长度后重新计算 corpus BLEU。区间反映本次选中问题或对话/资料样本的变动。冷/热两个请求保留在同一目标与方法内。','', '改善和退化案例依相同指标差绝对方向排序、ID/位置打破平局；无变化案例取排序首个 ID。LITM 使用官方首行 EM；Doc2Dial 案例使用单请求 BLEU 诊断，以完整语料 BLEU 评估总体结果。原答案、参考答案、参数及源行号见 `tables/litm_doc2dial_cases.csv`。','', '## 4 官方来源','', '- LongBench 英文 F1：https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py 与 LongBench/eval.py','- LITM 指标：https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py','- LITM 首行处理：https://github.com/nelson-liu/lost-in-the-middle/blob/main/scripts/evaluate_qa_responses.py','- Doc2Dial 参考 ID / 语料计分：https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/sharedtask_utils.py','- Doc2Dial 原模板：https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/subtask2/seq2seq_utils.py','- Doc2Dial 历史依赖：https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/requirements.txt','- 13a 规则：https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/tokenizers/tokenizer_13a.py 与 tokenizer_re.py','', '## 5 后续实验','', 'LITM 首行评分通过现有答案即可修正。需要评估共同模板顺序及完整历史的具体效果时，固定现有500目标并补充原官方顺序/历史筛选对照。扩展动态规则到正式实际块数需在 N=1–3 的独立开发构造上另设候选与验证后重新冻结。']
 (reports/'LITM_DOC2DIAL_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(metadata,ensure_ascii=False,indent=2))
 print('LITM official means',arr[:,:,:,1].mean(axis=(0,2)).tolist());print('Doc2Dial BLEU',actual.tolist())
if __name__=='__main__':main()
