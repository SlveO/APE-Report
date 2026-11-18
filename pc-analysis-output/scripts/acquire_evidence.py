"""Restore analysis-relevant evidence by the supplied byte manifest while Git transfers archives.
No digests. Sources remain unchanged; destination existing files are retained.
"""
import concurrent.futures, json, pathlib, urllib.request, shutil
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'pc-analysis-output'
BASE = 'https://raw.githubusercontent.com/SlveO/APE-Report/227116e4a004153272640cfe31087a033861b8e9/'
manifest = json.load(urllib.request.urlopen(BASE+'manifest.json', timeout=60))
(OUT/'acquisition').mkdir(parents=True, exist_ok=True)
(OUT/'acquisition'/'manifest_source.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
entries = {e['path']: e for e in manifest['files']}
selected = [p for p in entries if p.startswith(('APE/', 'docs/', 'datasets/frozen-main80/', 'output/runs/main80-20261114T041503/', 'logs/runs/20261114-main80-methods-full/')) or ('environment' in p.lower()) or p in ('config.json','data_sources.json')]
canonical = {}
def resolve(p):
    e=entries[p]
    return resolve(e['alias_of']) if e['storage']=='alias' else p
for p in selected: canonical.setdefault(resolve(p),[]).append(p)
def fetch(p):
    e=entries[p]
    dst=OUT/'acquisition'/'canonical'/p
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        pieces=e['parts'] if e['storage']=='parts' else [{'path':e['stored_path'],'bytes':e['bytes']}]
        with dst.with_name(dst.name+'.downloading').open('wb') as w:
            for piece in pieces:
                with urllib.request.urlopen(BASE+piece['path'],timeout=180) as r:
                    shutil.copyfileobj(r,w,1024*1024)
        dst.with_name(dst.name+'.downloading').replace(dst)
    if dst.stat().st_size != e['bytes']: raise ValueError(p)
    for rel in canonical[p]:
        target=ROOT/'restored'/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copyfile(dst,target)
    print('restored',p,e['bytes'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures={pool.submit(fetch,p):p for p in canonical}
    errors=[]
    for f in concurrent.futures.as_completed(futures):
        try: f.result()
        except Exception as exc:
            errors.append((futures[f],repr(exc)))
            print('ERROR',futures[f],repr(exc),flush=True)
(OUT/'acquisition'/'selective_restore.json').write_text(json.dumps({'reference_commit':'227116e4a004153272640cfe31087a033861b8e9','selected_logical_paths':selected,'errors':errors},ensure_ascii=False,indent=2),encoding='utf-8')
if errors: raise SystemExit(1)
