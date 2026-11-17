"""Recover interrupted Git blob transfer through manifest-sized raw files.
Only copies bytes; no additional digests. Existing correct-sized evidence is reused.
"""
from pathlib import Path
import concurrent.futures, json, shutil, subprocess
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'pc-analysis-output';BASE='https://raw.githubusercontent.com/SlveO/APE-Report/227116e4a004153272640cfe31087a033861b8e9/'
m=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));es={e['path']:e for e in m['files']}
def resolve(p):
    e=es[p];return resolve(e['alias_of']) if e['storage']=='alias' else p
aliases={}
for p in es:aliases.setdefault(resolve(p),[]).append(p)
jobs={}
for e in es.values():
    if e['storage']=='file':jobs[e['stored_path']]=e['bytes']
    if e['storage']=='parts':
        for part in e['parts']:jobs[part['path']]=part['bytes']
# Reuse byte-complete pre-restored evidence; preserve failed transfers for the log.
for p,e in es.items():
    if e['storage']=='alias':continue
    candidates=[OUT/'acquisition/canonical'/p,ROOT/'restored'/p]
    src=next((x for x in candidates if x.exists() and x.stat().st_size==e['bytes']),None)
    if src is None:continue
    if e['storage']=='file':
        dst=ROOT/e['stored_path'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(src,dst)
    else:
        with src.open('rb') as r:
            for part in e['parts']:
                dst=ROOT/part['path'];dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():r.seek(part['bytes'],1);continue
                remaining=part['bytes']
                with dst.open('wb') as w:
                    while remaining:
                        b=r.read(min(1024*1024,remaining));w.write(b);remaining-=len(b)
def priority(p):
    if any(s in p for s in ['attempts.jsonl','summary.json','events.jsonl','longbench.json','development.json','generation_config','ASSET_MANIFEST','doc2dial_dial_validation','doc2dial_doc.json','qa.prompt']):return 0
    if 'litm/' in p:return 1
    if 'frozen-main80' in p:return 2
    if 'parts/' in p:return 3
    return 4
file_canon={e['stored_path']:p for p,e in es.items() if e['storage']=='file'}
def restore_single(p,storage):
    src=ROOT/storage
    for rel in aliases[p]:
        dst=ROOT/'restored'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(src,dst)
def fetch(p):
    expected=jobs[p];dst=ROOT/p
    if dst.exists() and dst.stat().st_size==expected:
        if p in file_canon:restore_single(file_canon[p],p)
        return {'path':p,'bytes':expected,'status':'reused'}
    tmp=OUT/'acquisition/downloads'/p;tmp.parent.mkdir(parents=True,exist_ok=True)
    errors=[]
    for attempt in range(4):
        cmd=['curl.exe','--http1.1','--fail','--location','--retry','2','--retry-all-errors','--connect-timeout','15','--max-time','240','--silent','--show-error','--output',str(tmp),BASE+p+'?pc_analysis_recovery='+str(attempt)]
        r=subprocess.run(cmd,capture_output=True,text=True)
        size=tmp.stat().st_size if tmp.exists() else -1
        if r.returncode==0 and size==expected:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(tmp,dst)
            if p in file_canon:restore_single(file_canon[p],p)
            print('OK',p,expected,flush=True)
            return {'path':p,'bytes':expected,'status':'downloaded','attempts':attempt+1}
        errors.append({'attempt':attempt+1,'exit':r.returncode,'bytes':size,'expected':expected,'stderr':r.stderr[-600:]})
    print('FAILED',p,errors[-1],flush=True)
    return {'path':p,'status':'failed','errors':errors}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    fs=[pool.submit(fetch,p) for p in sorted(jobs,key=lambda p:(priority(p),jobs[p]))]
    for f in concurrent.futures.as_completed(fs):results.append(f.result())
(OUT/'acquisition/storage_recovery.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('storage total',len(results),'failed',sum(x['status']=='failed' for x in results),flush=True)
