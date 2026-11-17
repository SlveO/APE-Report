from pathlib import Path
import json, concurrent.futures, requests, threading, shutil, time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'pc-analysis-output/acquisition'
BASE='https://raw.githubusercontent.com/SlveO/APE-Report/227116e4a004153272640cfe31087a033861b8e9/'
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
jobs={}
for e in manifest['files']:
    if e['storage']=='file': jobs[e['stored_path']]=e['bytes']
    elif e['storage']=='parts':
        jobs.update({part['path']:part['bytes'] for part in e['parts']})
selected=[(p,n) for p,n in jobs.items() if n>1048576 and (not (ROOT/p).exists() or (ROOT/p).stat().st_size!=n)]
local=threading.local()
progress={'files':0,'bytes':0,'total':sum(n for p,n in selected)}
lock=threading.Lock()
(OUT/'download-only').mkdir(parents=True,exist_ok=True)
def fetch(item):
    rel,expected=item
    dst=ROOT/rel
    tmp=OUT/'download-only'/rel
    tmp.parent.mkdir(parents=True,exist_ok=True)
    # Retain prior failed destination bytes, then acquire into a separate temporary.
    if dst.exists():
        saved=OUT/'preserved-before-download'/rel
        saved.parent.mkdir(parents=True,exist_ok=True)
        if not saved.exists(): shutil.copyfile(dst,saved)
    prior=OUT/'range-downloads'/rel
    if not tmp.exists() and prior.exists() and prior.stat().st_size<=expected:
        shutil.copyfile(prior,tmp)
    start=tmp.stat().st_size if tmp.exists() else 0
    if start>expected: raise ValueError(('temporary too large',rel,start,expected))
    if not hasattr(local,'session'):
        local.session=requests.Session()
        local.session.headers.update({'Accept-Encoding':'identity'})
    errors=[]
    with tmp.open('ab') as w:
        while start<expected:
            end=min(expected-1,start+4*1024*1024-1)
            for attempt in range(6):
                try:
                    response=local.session.get(BASE+rel,headers={'Range':f'bytes={start}-{end}'},timeout=(15,45))
                    response.raise_for_status()
                    content=response.content
                    if response.status_code==200 and start==0 and len(content)==expected:
                        w.write(content); w.flush();start=expected;break
                    wanted=f'bytes {start}-{end}/{expected}'
                    if response.headers.get('Content-Range')!=wanted or len(content)!=end-start+1:
                        raise ValueError((response.status_code,response.headers.get('Content-Range'),len(content),wanted))
                    w.write(content);w.flush();start=end+1
                    break
                except Exception as exc:
                    errors.append(str(exc)[:400])
                    if attempt==5:
                        print('FAILED',rel,'offset',start,errors[-1],flush=True)
                        return {'path':rel,'status':'failed','offset':start,'expected':expected,'errors':errors}
                    time.sleep(min(attempt+1,3))
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(tmp,dst)
    with lock:
        progress['files']+=1;progress['bytes']+=expected
        (OUT/'download-only-progress.json').write_text(json.dumps(progress),encoding='utf-8')
    print('DOWNLOADED',rel,expected,flush=True)
    return {'path':rel,'status':'downloaded','bytes':expected,'retries':errors}
print('START',len(selected),'files',progress['total'],'bytes',flush=True)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    fs=[pool.submit(fetch,item) for item in sorted(selected,key=lambda x:x[1])]
    for f in concurrent.futures.as_completed(fs):
        results.append(f.result())
(OUT/'download-only-result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE','files',len(results),'failed',sum(r['status']=='failed' for r in results),flush=True)
