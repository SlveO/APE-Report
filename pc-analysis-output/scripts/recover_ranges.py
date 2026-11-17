"""Range-based recovery after observed transport EOFs. Accept-Encoding identity keeps
manifest byte offsets stable. Existing evidence is retained; no extra digests.
"""
from pathlib import Path
import concurrent.futures,json,shutil,threading,time
import requests
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'pc-analysis-output';BASE='https://raw.githubusercontent.com/SlveO/APE-Report/227116e4a004153272640cfe31087a033861b8e9/'
m=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));es={e['path']:e for e in m['files']};local=threading.local();lock=threading.Lock()
def resolve(p):
 e=es[p];return resolve(e['alias_of']) if e['storage']=='alias' else p
aliases={}
for p in es:aliases.setdefault(resolve(p),[]).append(p)
jobs={};canon_for={}
for p,e in es.items():
 if e['storage']=='file':jobs[e['stored_path']]=e['bytes'];canon_for[e['stored_path']]=p
 if e['storage']=='parts':
  for part in e['parts']:jobs[part['path']]=part['bytes'];canon_for[part['path']]=p

def restore(p):
 e=es[p]
 if e['storage']=='file':src=ROOT/e['stored_path']
 else:
  if not all((ROOT[x['path']]).exists() and (ROOT[x['path']]).stat().st_size==x['bytes'] for x in e['parts']):return
  src=OUT/'acquisition/reconstructed'/p;src.parent.mkdir(parents=True,exist_ok=True)
  if not src.exists():
   with src.open('wb') as w:
    for x in e['parts']:
     with (ROOT/x['path']).open('rb') as r:shutil.copyfileobj(r,w,1024*1024)
 if not src.exists():return
 for rel in aliases[p]:
  dst=ROOT/'restored'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copyfile(src,dst)
def priority(p):
 c=canon_for[p]
 if any(x in c for x in ('attempts.jsonl','events.jsonl','longbench.json','generation_config.json','doc2dial_dial_validation.json','doc2dial_doc.json')):return 0
 if 'datasets/raw-main80/litm/' in c:return 1
 if 'development.json' in c:return 2
 if 'attention.jsonl' in c:return 3
 if c.startswith('datasets/raw-main80') and c.endswith('.zip'):return 10
 return 4

def fetch(p):
 dst=ROOT/p;expected=jobs[p]
 if dst.exists():
  if dst.stat().st_size!=expected:return {'path':p,'status':'existing_size_conflict','bytes':dst.stat().st_size,'expected':expected}
  with lock:restore(canon_for[p])
  return {'path':p,'status':'reused','bytes':expected}
 tmp=OUT/'acquisition/range-downloads'/p;tmp.parent.mkdir(parents=True,exist_ok=True)
 if not hasattr(local,'session'):
  local.session=requests.Session();local.session.headers['Accept-Encoding']='identity'
 start=tmp.stat().st_size if tmp.exists() else 0
 if start>expected:return {'path':p,'status':'temporary_size_conflict','bytes':start}
 errors=[]
 with tmp.open('ab') as w:
  while start<expected:
   end=min(expected-1,start+4*1024*1024-1)
   for trial in range(4):
    try:
     r=local.session.get(BASE+p,headers={'Range':f'bytes={start}-{end}'},timeout=(15,60));r.raise_for_status();b=r.content
     if r.status_code==200 and start==0 and len(b)==expected:
      w.write(b);start=expected;break
     want=f'bytes {start}-{end}/{expected}'
     if r.headers.get('Content-Range')!=want or len(b)!=(end-start+1):raise ValueError((r.status_code,r.headers.get('Content-Range'),len(b),want))
     w.write(b);w.flush();start=end+1;break
    except Exception as exc:
     errors.append(str(exc)[:300])
     if trial==3:
      print('FAILED',p,start,errors[-1],flush=True);return {'path':p,'status':'failed','offset':start,'errors':errors}
     time.sleep(1)
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(tmp,dst)
 with lock:restore(canon_for[p])
 print('OK',p,expected,flush=True)
 return {'path':p,'status':'downloaded','bytes':expected,'recovered_errors':errors}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 fs=[pool.submit(fetch,p) for p in sorted(jobs,key=lambda p:(priority(p),jobs[p]))]
 for f in concurrent.futures.as_completed(fs):results.append(f.result())
(OUT/'acquisition/range_recovery.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('done',len(results),'failed',sum(x['status'] not in ('reused','downloaded') for x in results),flush=True)
