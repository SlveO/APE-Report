"""Minimal CPU evidence for the Stage 1 construction discrepancy; no score analysis."""
from pathlib import Path
import json, os
from importlib.metadata import version
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'pc-analysis-output/stages/01_audit'
REST=ROOT/'restored'
os.environ['HF_HOME']=str(OUT/'tmp/hf')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from transformers import AutoTokenizer
source=REST/'output/runs/main80-20261114T041503/snapshot/datasets/frozen-main80'
model=REST/'output/runs/main80-20261114T041503/snapshot/models/Llama-3.1-8B-Instruct'
base={x['sample_id']:x for x in json.loads((source/'longbench.json').read_text(encoding='utf-8'))}
dev=json.loads((source/'development.json').read_text(encoding='utf-8'))
native=Tokenizer.from_file(str(model/'tokenizer.json'))
auto=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
cases=[]
for sid,L in [('hotpotqa:de89026c970fbb6dff04176107b9cf20407d155bc1ae2eec',4096),('hotpotqa:de89026c970fbb6dff04176107b9cf20407d155bc1ae2eec',6144),('qasper:e419f2bff9d2ab7c3b60b3250caccd2d9ae1285ec3e8e818',4096)]:
    row=next(r for r in dev if r['sample_id']==sid and r['L']==L)
    b=base[sid]; n=native.encode(b['document'],add_special_tokens=False); a=auto(b['document'],add_special_tokens=False,return_offsets_mapping=True)
    ev={i for i,(s,e) in enumerate(n.offsets) if e>s and any(s<y and e>x for x,y in b['evidence_char_spans'])}
    keep={i for x,y in row['retained_source_intervals'] for i in range(x,y)}
    missing=sorted(ev-keep)
    offsets=[n.offsets[i] for i in missing]
    cases.append(dict(sample_id=sid,L=L,question=b['question'],reference_answers=b['answers'],evidence_mapping_policy=b['evidence_mapping_policy'],evidence_char_spans=b['evidence_char_spans'],evidence_source_text=[b['document'][x:min(y,x+1500)] for x,y in b['evidence_char_spans']],n_source_tokens=len(n.ids),native_auto_token_IDs_equal=n.ids==a['input_ids'],native_auto_offsets_equal=n.offsets==a['offset_mapping'],retained_source_intervals=row['retained_source_intervals'],deleted_source_intervals=row['deleted_source_intervals'],n_required_evidence_tokens=len(ev),n_missing_required_tokens=len(missing),missing_source_token_positions=missing,missing_source_token_IDs=[n.ids[i] for i in missing],missing_source_offsets=offsets,missing_source_text=native.decode([n.ids[i] for i in missing],skip_special_tokens=False),saved_evidence_intervals=row['evidence_token_intervals'],saved_evidence_interval_texts=[native.decode(row['document_tokens'][x:y],skip_special_tokens=False) for x,y in row['evidence_token_intervals']],source=str(source.relative_to(ROOT))+'/{longbench,development}.json'))
report=dict(server_environment=json.loads((REST/'logs/runs/20261113-main80-prep/environment.json').read_text(encoding='utf-8')),local_environment=dict(transformers=version('transformers'),tokenizers=version('tokenizers')),cases=cases,upstream_sources=[dict(url='https://github.com/huggingface/tokenizers/blob/v0.19.1/tokenizers/src/models/bpe/model.rs',observation='ignore_merges direct-vocabulary branch returns zero-length token offsets'),dict(url='https://github.com/huggingface/tokenizers/pull/1640',observation='merged fix changes ignore_merges offsets to token length'),dict(url='https://github.com/huggingface/tokenizers/issues/1517',observation='Llama3 zero-length offset_mapping report')],remaining_root_cause_evidence='Server tokenizers0.19.1 per-token offset output is not saved in the supplied preparation JSON; reproduce this minimal CPU example in that environment for root-cause archival.')
(OUT/'reports/STAGE1_EVIDENCE_DELETION_EXAMPLE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(sample_id=c['sample_id'],L=c['L'],native_auto_token_IDs_equal=c['native_auto_token_IDs_equal'],native_auto_offsets_equal=c['native_auto_offsets_equal'],n_required_evidence_tokens=c['n_required_evidence_tokens'],n_missing_required_tokens=c['n_missing_required_tokens'],missing_source_text=c['missing_source_text']) for c in cases],ensure_ascii=False,indent=2))
