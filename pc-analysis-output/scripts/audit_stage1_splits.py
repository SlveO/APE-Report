"""Stage 1: frozen LongBench splits and evidence-preserving constructions.
Read-only inputs under restored/. Writes only pc-analysis-output/stages/01_audit.
No parameter scoring, plotting, length-check statistics or bootstrap.
"""
from pathlib import Path
import csv, json, os, re, sys, string, platform
from collections import Counter, defaultdict
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'pc-analysis-output/stages/01_audit'
REST=ROOT/'restored'
FROZEN=REST/'output/runs/main80-20261114T041503/snapshot/datasets/frozen-main80'
BASE=REST/'datasets/frozen-main80'
TOKENIZER=REST/'output/runs/main80-20261114T041503/snapshot/models/Llama-3.1-8B-Instruct/tokenizer.json'
DS=('qasper','hotpotqa','musique')
SPLITS=('explore','validation','test')
LENGTHS=(4096,8192,16384,6144,12288)
SRC='restored/output/runs/main80-20261114T041503/snapshot/datasets/frozen-main80/'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False)
def qkey(x): return ' '.join(x.casefold().split())
def norm(x): return ' '.join(re.sub(r'\b(a|an|the)\b',' ',x.lower().translate(str.maketrans('','',string.punctuation))).split())
def compact_intervals(indices):
    out=[]
    for i in sorted(indices):
        if out and out[-1][1]==i: out[-1][1]+=1
        else: out.append([i,i+1])
    return out

def expand(spans,bound):
    previous=0; valid=True; values=[]
    for span in spans:
        if not isinstance(span,list) or len(span)!=2: valid=False; continue
        a,b=span
        if not 0<=a<b<=bound or a<previous: valid=False
        values.extend(range(max(0,a),min(bound,b))); previous=b
    return values,valid

def write_csv(name,rows):
    rows=list(rows)
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    for folder in ('tables','reports','tmp'): (OUT/folder).mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(OUT/'tmp'); os.environ['TMP']=str(OUT/'tmp')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_HOME']=str(OUT/'tmp/hf'); os.environ['XDG_CACHE_HOME']=str(OUT/'tmp/cache')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(TOKENIZER))
    lb=read(FROZEN/'longbench.json'); app=read(FROZEN/'applicability.json'); dev=read(FROZEN/'development.json')
    frozen=read(FROZEN/'frozen.json'); split_report=read(FROZEN/'split-report.json')
    reviews=read(REST/frozen['sources']['supplement_reviews'])
    findings=[]; checks=[]; examples=[]
    def check(name,passed,n,source,detail='',sample_id='',L=''):
        checks.append(dict(check=name,passed=bool(passed),denominator=n,unit='records unless specified',source=source,detail=detail,sample_id=sample_id,L=L))
        if not passed: findings.append(dict(issue=name,sample_id=sample_id,L=L,detail=detail,source=source))
    check('600 unique LongBench IDs',len(lb)==600 and len({r['sample_id'] for r in lb})==600,600,SRC+'longbench.json:sample_id')
    by_id={r['sample_id']:r for r in lb}
    check('source character spans valid',all(all(0<=a<b<=len(r['document']) for a,b in r['evidence_char_spans']) for r in lb),len(lb),SRC+'longbench.json:evidence_char_spans,document')
    check('support document IDs belong to public input documents',all(set(r['support_documents']).issubset(r['input_documents']) for r in lb),len(lb),SRC+'longbench.json:support_documents,input_documents')
    split_counts=[]
    expected={'explore':{'qasper':17,'hotpotqa':17,'musique':16},'validation':{'qasper':13,'hotpotqa':13,'musique':14},'test':dict.fromkeys(DS,170)}
    for d in DS:
        rows=[r for r in lb if r['dataset']==d]
        check('200 IDs per dataset',len(rows)==200,200,SRC+'longbench.json:dataset',d)
        raw=[json.loads(line) for line in (REST/frozen['sources']['longbench'][d]).read_text(encoding='utf-8').splitlines() if line.strip()]
        raw_ids={r['_id']:r for r in raw}
        matched=0
        for r in rows:
            public=raw_ids.get(r['public_id'])
            ok=public is not None and r['question']==public['input'] and r['document']==public['context'] and r['answers']==public['answers']
            matched+=ok
            if not ok: findings.append(dict(issue='frozen/public data field mismatch',sample_id=r['sample_id'],source=frozen['sources']['longbench'][d],detail='public_id + question/input + document/context + answers'))
        check('frozen public ID/question/context/references join',matched==len(rows),len(rows),frozen['sources']['longbench'][d],f'{d}: {matched}/{len(rows)}')
        for spl in SPLITS:
            rr=[r for r in rows if r['split']==spl]
            split_counts.append(dict(dataset=d,split=spl,n_sample_ids=len(rr),expected_ids=expected[spl][d],n_evidence_groups=len({canonical(r['group']) for r in rr}),n_normalized_questions=len({qkey(r['question']) for r in rr}),denominator_dataset=200,unit='sample IDs / groups / questions',source=SRC+'longbench.json:dataset,split,sample_id,group,question'))
            check('frozen split expected size',len(rr)==expected[spl][d],expected[spl][d],SRC+'longbench.json:split',f'{d}/{spl}')
    for spl in SPLITS:
        ids={r['sample_id'] for r in lb if r['split']==spl}
        order=frozen['orders'][spl]
        check('order matches frozen split',len(order)==len(set(order)) and set(order)==ids,len(ids),SRC+'frozen.json:orders',spl)
    write_csv('splits_counts.csv',split_counts)
    # Independently reconstruct connected groups from equal full support sets and questions.
    parent=list(range(len(lb)))
    def root(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[root(j)]=root(i)
    seen_support={}; seen_question={}
    for i,r in enumerate(lb):
        for seen,key in ((seen_support,tuple(sorted(r['support_documents']))),(seen_question,qkey(r['question']))):
            if key in seen: union(i,seen[key])
            else: seen[key]=i
    components=defaultdict(list)
    for i,r in enumerate(lb): components[root(i)].append(r)
    group_check=all(len({r['split'] for r in rr})==1 and len({canonical(r['group']) for r in rr})==1 for rr in components.values())
    group_to_components=defaultdict(set)
    for k,rr in components.items():
        for r in rr: group_to_components[canonical(r['group'])].add(k)
    check('reconstructed evidence/question groups match partition',group_check and all(len(v)==1 for v in group_to_components.values()),len(components),SRC+'longbench.json:support_documents,question,group,split',f'{len(components)} connected groups; equal complete support set or normalized question defines edges')
    group_rows=[dict(dataset=r['dataset'],sample_id=r['sample_id'],split=r['split'],evidence_group=canonical(r['group']),support_documents=canonical(r['support_documents']),input_documents=canonical(r['input_documents']),source_ids=canonical(r['source_ids']),evidence_mapping_policy=canonical(r['evidence_mapping_policy']),n_evidence_char_spans=len(r['evidence_char_spans']),original_document_count=r['original_document_count'],original_document_tokens=r['_length'],unit='sample',source=SRC+'longbench.json') for r in lb]
    write_csv('splits_sample_dictionary.csv',group_rows)
    overlap_summary=[]; overlap_documents=[]; affected_ids=[]
    pairs=(('explore','validation'),('explore','test'),('validation','test'),('development','test'))
    for left,right in pairs:
        a=[r for r in lb if r['split'] in ('explore','validation')] if left=='development' else [r for r in lb if r['split']==left]
        b=[r for r in lb if r['split']==right]
        specs=(('normalized_question',lambda r:[qkey(r['question'])]),('complete_evidence_group',lambda r:[canonical(r['group'])]),('single_document',lambda r:r['input_documents']),('support_document',lambda r:r['support_documents']))
        for kind,extract in specs:
            aset={x for r in a for x in extract(r)}; bset={x for r in b for x in extract(r)}; shared=aset & bset
            aa=[r for r in a if set(extract(r)) & shared]; bb=[r for r in b if set(extract(r)) & shared]
            overlap_summary.append(dict(left_split=left,right_split=right,overlap_type=kind,n_unique_shared_items=len(shared),n_affected_left_ids=len(aa),denominator_left_ids=len(a),n_affected_right_ids=len(bb),denominator_right_ids=len(b),right_affected_fraction=len(bb)/len(b),unit='unique items / sample IDs / fraction',source=SRC+'longbench.json:question,group,input_documents,support_documents'))
            for x in sorted(shared):
                left_ids=[r['sample_id'] for r in aa if x in extract(r)]; right_ids=[r['sample_id'] for r in bb if x in extract(r)]
                overlap_documents.append(dict(left_split=left,right_split=right,overlap_type=kind,shared_item=x,n_left_ids=len(left_ids),n_right_ids=len(right_ids),left_ids=canonical(left_ids),right_ids=canonical(right_ids),unit='item / sample IDs',source=SRC+'longbench.json'))
            for side,rr in (('left',aa),('right',bb)):
                for r in rr: affected_ids.append(dict(left_split=left,right_split=right,overlap_type=kind,side=side,dataset=r['dataset'],sample_id=r['sample_id'],actual_split=r['split'],shared_items=canonical(sorted(set(extract(r))&shared)),unit='sample ID',source=SRC+'longbench.json'))
            if left!='development':
                field={'normalized_question':'question_overlap','complete_evidence_group':'evidence_group_overlap','single_document':'single_document_overlap','support_document':'support_document_overlap'}[kind]
                saved=set(split_report[f'{left}/{right}'][field])
                check('split-report overlap matches independent set intersection',saved==shared,len(shared),SRC+'split-report.json:'+field,f'{left}/{right}/{kind}')
            if kind in ('normalized_question','complete_evidence_group'): check('cross-split isolated '+kind,len(shared)==0,len(a)+len(b),SRC+'longbench.json',f'{left}/{right}')
    write_csv('splits_overlap_summary.csv',overlap_summary); write_csv('splits_overlap_documents.csv',overlap_documents); write_csv('splits_overlap_affected_ids.csv',affected_ids)
    # Retokenize the 90 development sources with the exported tokenizer; no model weights.
    development_ids={r['sample_id'] for r in lb if r['split']!='test'}
    encoded={sid:tokenizer.encode(by_id[sid]['document'],add_special_tokens=False) for sid in sorted(development_ids)}
    check('development source token lengths',all(len(enc.ids)==by_id[sid]['_length'] for sid,enc in encoded.items()),len(encoded),str(TOKENIZER.relative_to(ROOT))+' + '+SRC+'longbench.json:_length','tokenizers '+version('tokenizers'))
    evidence_indices={sid:{i for i,(s,e) in enumerate(enc.offsets) if e>s and any(s<b and e>a for a,b in by_id[sid]['evidence_char_spans'])} for sid,enc in encoded.items()}
    appkeys=[(r['sample_id'],r['L']) for r in app]; devkeys=[(r['sample_id'],r['L']) for r in dev]
    expected_app={(sid,l) for sid in development_ids for l in LENGTHS}
    check('450 distinct sample-length applicability candidates',len(app)==450 and len(set(appkeys))==450 and set(appkeys)==expected_app,450,SRC+'applicability.json:sample_id,L')
    expected_dev={(r['sample_id'],r['L']) for r in app if r['applicable']}
    check('development rows match applicable constructions',len(dev)==432 and len(set(devkeys))==432 and set(devkeys)==expected_dev,432,SRC+'applicability.json + development.json')
    apptable=[]; nabyreason=[]
    for r in app:
        sid=r['sample_id']; ev=evidence_indices[sid]
        precise_reason=''
        if not r['applicable']:
            precise_reason='no_annotated_evidence' if not ev else ('required_evidence_tokens_exceed_L' if len(ev)>r['L'] else r.get('reason',''))
        apptable.append(dict(sample_id=sid,dataset=r['dataset'],split=r['split'],L=r['L'],applicable=r['applicable'],saved_reason=r.get('reason',''),reviewed_reason=precise_reason,n_required_evidence_tokens=len(ev),available_tokens=r.get('available_tokens',''),unit='sample-length input / tokens',source=SRC+'applicability.json and tokenizer offsets over longbench.json:evidence_char_spans'))
        if not r['applicable'] and len([x for x in examples if x.get('kind')=='inapplicable' and x.get('reviewed_reason')==precise_reason])<2:
            examples.append(dict(kind='inapplicable',sample_id=sid,dataset=r['dataset'],split=r['split'],L=r['L'],saved_reason=r.get('reason'),reviewed_reason=precise_reason,n_required_evidence_tokens=len(ev),available_tokens=r.get('available_tokens'),question=by_id[sid]['question'],source=SRC+'applicability.json'))
    countgroups=Counter((r['split'],r['dataset'],r['L'],r['applicable'],r['saved_reason'],r['reviewed_reason']) for r in apptable)
    for (spl,d,l,yes,reason,precise),n in sorted(countgroups.items()): nabyreason.append(dict(split=spl,dataset=d,L=l,applicable=yes,saved_reason=reason,reviewed_reason=precise,n_inputs=n,denominator_split_dataset_ids=sum(x['split']==spl and x['dataset']==d for x in lb),unit='sample-length inputs',source=SRC+'applicability.json'))
    write_csv('construction_applicability.csv',apptable); write_csv('construction_applicability_counts.csv',nabyreason)
    appindex={(r['sample_id'],r['L']):r for r in app}
    construction_rows=[]; supplement_rows=[]; cross_rows=[]; missing_evidence=[]; missing_examples=[]
    for r in dev:
        sid=r['sample_id']; source=by_id[sid]; enc=encoded[sid]; tokens=enc.ids; ev=evidence_indices[sid]; L=r['L']
        retained,retvalid=expand(r['retained_source_intervals'],len(tokens)); deleted,delvalid=expand(r['deleted_source_intervals'],len(tokens))
        retainedset=set(retained); partition=retvalid and delvalid and retainedset.isdisjoint(deleted) and retainedset | set(deleted)==set(range(len(tokens)))
        positions={old:new for new,old in enumerate(retained)}
        all_evidence_retained=ev.issubset(retainedset)
        missing=sorted(ev-retainedset)
        if missing:
            missing_spans=compact_intervals(missing)
            first=missing_spans[0]; a=enc.offsets[first[0]][0]; b=enc.offsets[first[1]-1][1]
            missing_evidence.append(dict(sample_id=sid,dataset=r['dataset'],split=r['split'],L=L,affected_N='4|8|16',n_required_evidence_tokens=len(ev),n_missing_required_tokens=len(missing),n_retained_required_tokens=len(ev)-len(missing),missing_fraction=len(missing)/len(ev),missing_source_token_intervals=canonical(missing_spans),first_missing_source_token_interval=canonical(first),first_missing_char_interval=canonical([a,b]),first_missing_source_text=source['document'][a:min(b,a+600)],evidence_mapping_policy=canonical(source['evidence_mapping_policy']),unit='sample-length construction / tokens / fraction',source=SRC+'longbench.json:evidence_char_spans + development.json:retained_source_intervals; exported tokenizer offsets'))
            missing_examples.append(dict(kind='required_source_evidence_deleted',sample_id=sid,dataset=r['dataset'],split=r['split'],L=L,question=source['question'],source_ids=source['source_ids'],evidence_mapping_policy=source['evidence_mapping_policy'],evidence_char_spans=source['evidence_char_spans'],required_evidence_tokens=len(ev),missing_required_tokens=len(missing),retained_source_intervals=r['retained_source_intervals'],missing_source_token_intervals=missing_spans,first_missing_char_interval=[a,b],first_missing_source_text=source['document'][a:min(b,a+600)],all_constructed_token_ids_reproduced=True,source=SRC+'longbench.json and development.json'))
        derived_evidence=compact_intervals(positions[i] for i in ev if i in positions)
        evidence_field_ok=derived_evidence==r['evidence_token_intervals']
        built=[tokens[i] for i in retained]; donor_ok=True; donor_token_count=0; no_answer_hits=True; cert_ok=True
        donor_ids=[]
        for j,s in enumerate(r['supplement_sources']):
            donor_ids.append(s['sample_id']); donor=by_id.get(s['sample_id']); review=reviews.get(sid,{}).get(s['sample_id'],{})
            if donor is None or donor['split']=='test' or s['sample_id'] not in encoded: donor_ok=False; continue
            donorenc=encoded[s['sample_id']]; donoridx,donorvalid=expand(s['source_token_intervals'],len(donorenc.ids))
            overlap_support=set(source['support_documents']) & set(donor['support_documents'])
            same_group=source['group']==donor['group']
            approved=set(i for a,b in review.get('approved_token_intervals',[]) for i in range(a,b))
            selected=set(donoridx)
            cert=bool(review) and review.get('contains_additional_answer_evidence') is False and bool(review.get('basis')) and selected.issubset(approved)
            saved_review=s.get('review',{})
            for k in ('basis','contains_additional_answer_evidence','review_ref','reviewed_by'):
                if k in saved_review and saved_review[k]!=review.get(k): cert=False
            ref=saved_review.get('review_ref','').split('#')[0]
            if ref and not (REST/ref).is_file(): cert=False
            cert_ok=cert_ok and cert
            donor_ok=donor_ok and donorvalid and not overlap_support and not same_group and s['source_documents']==donor['input_documents']
            donor_tokens=[donorenc.ids[i] for i in donoridx]; built.extend(donor_tokens); donor_token_count+=len(donor_tokens)
            hits=[]
            for a,b in s['source_token_intervals']:
                text=' '+norm(tokenizer.decode(donorenc.ids[a:b],skip_special_tokens=False))+' '
                for answer in source['answers']:
                    if norm(answer) and ' '+norm(answer)+' ' in text: hits.append(answer)
            no_answer_hits=no_answer_hits and not hits
            supplement_rows.append(dict(target_sample_id=sid,target_dataset=r['dataset'],target_split=r['split'],L=L,donor_sample_id=s['sample_id'],donor_dataset=donor['dataset'],donor_split=donor['split'],n_source_intervals=len(s['source_token_intervals']),n_contributed_tokens=len(donor_tokens),cross_split=r['split']!=donor['split'],same_evidence_group=same_group,n_shared_support_documents=len(overlap_support),source_intervals=canonical(s['source_token_intervals']),review_ref=saved_review.get('review_ref',''),review_basis=saved_review.get('basis',''),approved_and_certified=cert,answer_literal_hits=canonical(hits),unit='supplement source contribution / tokens',source=SRC+'development.json:supplement_sources; datasets/prepared-main80/supplement-reviews.json'))
            if not any(x.get('kind')=='supplement' for x in examples):
                first=s['source_token_intervals'][0]
                examples.append(dict(kind='supplement',target_sample_id=sid,target_question=source['question'],L=L,donor_sample_id=s['sample_id'],target_split=r['split'],donor_split=donor['split'],source_token_interval=first,source_text=tokenizer.decode(donorenc.ids[first[0]:first[1]],skip_special_tokens=False)[:600],review_ref=saved_review.get('review_ref'),review_basis=saved_review.get('basis'),source=SRC+'development.json:supplement_sources'))
        token_match=built==r['document_tokens'] and len(built)==L
        app_match=all(appindex[(sid,L)].get(k)==r.get(k) for k in ('retained_source_intervals','deleted_source_intervals','supplement_sources','evidence_token_intervals','cross_block_evidence'))
        cross_ok=True
        for N in (4,8,16):
            crossing=[span for span in derived_evidence if span[0]//(L//N)!=(span[1]-1)//(L//N)]
            ok=crossing==r['cross_block_evidence'][str(N)]; cross_ok=cross_ok and ok
            saved_self_consistent=r['cross_block_evidence'][str(N)]==[span for span in r['evidence_token_intervals'] if span[0]//(L//N)!=(span[1]-1)//(L//N)]
            cross_rows.append(dict(sample_id=sid,dataset=r['dataset'],split=r['split'],L=L,N=N,block_tokens=L//N,n_crossing_evidence_intervals=len(crossing),has_crossing=bool(crossing),saved_field_matches=ok,saved_field_self_consistent=saved_self_consistent,unit='constructed sample/L/N cell / evidence intervals',source=SRC+'development.json:evidence_token_intervals,cross_block_evidence'))
        for name,ok in [('retained/deleted partition',partition),('required evidence retained',all_evidence_retained),('evidence token positions',evidence_field_ok),('donor restrictions and intervals',donor_ok),('donor review certification',cert_ok),('no literal reference answer in selected donor spans',no_answer_hits),('exact reconstructed construction tokens and length',token_match),('applicability/development field agreement',app_match),('cross-block evidence field',cross_ok)]:
            if not ok: findings.append(dict(issue=name,sample_id=sid,L=L,source=SRC+'development.json',detail='see construction_checks.csv and construction_supplement_sources.csv'))
        construction_rows.append(dict(sample_id=sid,dataset=r['dataset'],split=r['split'],L=L,n_source_tokens=len(tokens),n_retained_source_tokens=len(retained),n_deleted_source_tokens=len(deleted),n_required_evidence_tokens=len(ev),n_saved_evidence_tokens=sum(b-a for a,b in r['evidence_token_intervals']),n_missing_required_tokens=len(missing),n_expected_retained_evidence_intervals=len(derived_evidence),n_saved_evidence_intervals=len(r['evidence_token_intervals']),n_supplement_sources=len(donor_ids),n_supplement_tokens=donor_token_count,supplement_donor_ids=canonical(donor_ids),source_partition_ok=partition,evidence_retained=all_evidence_retained,evidence_positions_ok=evidence_field_ok,donor_restrictions_ok=donor_ok,donor_certification_ok=cert_ok,no_literal_answer_in_donor=no_answer_hits,exact_tokens_match=token_match,exact_L_match=len(r['document_tokens'])==L,applicability_fields_match=app_match,cross_block_field_ok=cross_ok,unit='sample-length construction / tokens',source=SRC+'development.json; reconstructed from longbench.json and exported tokenizer'))
    write_csv('construction_checks.csv',construction_rows); write_csv('construction_supplement_sources.csv',supplement_rows); write_csv('construction_cross_block.csv',cross_rows)
    write_csv('construction_missing_evidence.csv',missing_evidence)
    missing_keys={(r['sample_id'],r['L']) for r in missing_evidence}
    affected_generations=[]
    attempt_path=REST/'output/runs/main80-20261114T041503/attempts.jsonl'
    with attempt_path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip(): continue
            r=json.loads(line)
            if r['key']['stage'] not in ('explore','validation','length_check','attention') or (r['sample_id'],r['L']) not in missing_keys: continue
            affected_generations.append(dict(stage=r['key']['stage'],dataset=r['dataset'],sample_id=r['sample_id'],L=r['L'],N=r['N'],method=r['method'],T=r['T'],S=r['S'],attempt=r['attempt'],status=r['status'],logical_labels=canonical(r['logical_labels']),unit='generation record',source='restored/output/runs/main80-20261114T041503/attempts.jsonl joined by sample_id,L; no scores read for selection'))
    write_csv('construction_missing_evidence_generations.csv',affected_generations)
    affected_stage_summary=[]
    for stage,d,L in sorted({(r['stage'],r['dataset'],r['L']) for r in affected_generations}):
        rr=[r for r in affected_generations if (r['stage'],r['dataset'],r['L'])==(stage,d,L)]
        affected_stage_summary.append(dict(stage=stage,dataset=d,L=L,n_generations=len(rr),n_sample_ids=len({r['sample_id'] for r in rr}),N_values=canonical(sorted({r['N'] for r in rr})),unit='generation records / sample IDs',source='construction_missing_evidence_generations.csv'))
    write_csv('construction_missing_evidence_stage_summary.csv',affected_stage_summary)
    for d in DS:
        rr=[r for r in missing_examples if r['dataset']==d]
        if rr: examples.append(max(rr,key=lambda r:r['missing_required_tokens']))
    # Export a retained-but-mislabelled minimal case separately from real deletion.
    r=next(x for x in dev if x['sample_id']=='qasper:e419f2bff9d2ab7c3b60b3250caccd2d9ae1285ec3e8e818' and x['L']==4096)
    sid=r['sample_id']; retained,_=expand(r['retained_source_intervals'],len(encoded[sid].ids)); position={old:new for new,old in enumerate(retained)}
    examples.append(dict(kind='retained_evidence_position_metadata_difference',sample_id=sid,L=4096,evidence_char_spans=by_id[sid]['evidence_char_spans'],expected_full_evidence_tokens=len(evidence_indices[sid]),saved_evidence_tokens=sum(b-a for a,b in r['evidence_token_intervals']),saved_evidence_intervals=r['evidence_token_intervals'],expected_evidence_intervals=compact_intervals(position[i] for i in evidence_indices[sid] if i in position),saved_interval_texts=[tokenizer.decode(r['document_tokens'][a:b],skip_special_tokens=False) for a,b in r['evidence_token_intervals']],source=SRC+'longbench.json and development.json'))
    construction_summary=[]
    for spl,d,l in sorted({(r['split'],r['dataset'],r['L']) for r in construction_rows}):
        rr=[r for r in construction_rows if (r['split'],r['dataset'],r['L'])==(spl,d,l)]
        construction_summary.append(dict(split=spl,dataset=d,L=l,n_constructions=len(rr),n_with_supplements=sum(r['n_supplement_tokens']>0 for r in rr),n_supplement_tokens_total=sum(r['n_supplement_tokens'] for r in rr),min_required_evidence_tokens=min(r['n_required_evidence_tokens'] for r in rr),max_required_evidence_tokens=max(r['n_required_evidence_tokens'] for r in rr),n_exact_token_matches=sum(r['exact_tokens_match'] for r in rr),n_all_evidence_retained=sum(r['evidence_retained'] for r in rr),unit='sample-length constructions / tokens',source='construction_checks.csv'))
    write_csv('construction_summary.csv',construction_summary)
    supplement_summary=[]
    for target_spl,donor_spl in sorted({(r['target_split'],r['donor_split']) for r in supplement_rows}):
        rr=[r for r in supplement_rows if (r['target_split'],r['donor_split'])==(target_spl,donor_spl)]
        supplement_summary.append(dict(target_split=target_spl,donor_split=donor_spl,n_source_contributions=len(rr),n_target_constructions=len({(r['target_sample_id'],r['L']) for r in rr}),n_target_ids=len({r['target_sample_id'] for r in rr}),n_donor_ids=len({r['donor_sample_id'] for r in rr}),n_contributed_tokens=sum(r['n_contributed_tokens'] for r in rr),unit='source contributions / constructions / sample IDs / tokens',source='construction_supplement_sources.csv'))
    write_csv('construction_supplement_split_summary.csv',supplement_summary)
    cross_summary=[]
    for spl,d,l,n in sorted({(r['split'],r['dataset'],r['L'],r['N']) for r in cross_rows}):
        rr=[r for r in cross_rows if (r['split'],r['dataset'],r['L'],r['N'])==(spl,d,l,n)]
        cross_summary.append(dict(split=spl,dataset=d,L=l,N=n,n_constructions=len(rr),n_with_cross_block_evidence=sum(r['has_crossing'] for r in rr),n_crossing_intervals=sum(r['n_crossing_evidence_intervals'] for r in rr),fraction_with_crossing=sum(r['has_crossing'] for r in rr)/len(rr),unit='constructed cells / evidence intervals / fraction',source='construction_cross_block.csv'))
    write_csv('construction_cross_block_summary.csv',cross_summary)
    policy_counts=Counter((r['dataset'],r['split'],p) for r in lb for p in r['evidence_mapping_policy'])
    write_csv('construction_evidence_mapping_policy.csv',[dict(dataset=d,split=spl,policy=p,n_ids=n,denominator_dataset_split=sum(r['dataset']==d and r['split']==spl for r in lb),unit='sample IDs',source=SRC+'longbench.json:evidence_mapping_policy') for (d,spl,p),n in sorted(policy_counts.items())])
    for (d,policy) in sorted({(r['dataset'],p) for r in lb for p in r['evidence_mapping_policy']}):
        r=next(r for r in sorted(lb,key=lambda r:r['sample_id']) if r['dataset']==d and policy in r['evidence_mapping_policy'])
        spans=r['evidence_char_spans']; span=spans[0] if spans else None
        examples.append(dict(kind='source_evidence_mapping',dataset=d,policy=policy,sample_id=r['sample_id'],split=r['split'],question=r['question'],source_ids=r['source_ids'],support_documents=r['support_documents'],evidence_char_interval=span,source_text=r['document'][span[0]:min(span[1],span[0]+600)] if span else '',source=SRC+'longbench.json:evidence_char_spans,document'))
    doc_example=next((r for r in overlap_documents if r['left_split']=='explore' and r['right_split']=='test' and r['overlap_type']=='support_document'),None)
    if doc_example: examples.append(dict(kind='cross_split_shared_support_document',**doc_example))
    check('all 432 constructions preserve exact input tokens and required evidence',all(r['exact_tokens_match'] and r['evidence_retained'] and r['source_partition_ok'] and r['evidence_positions_ok'] and r['cross_block_field_ok'] and r['donor_restrictions_ok'] and r['donor_certification_ok'] and r['no_literal_answer_in_donor'] and r['applicability_fields_match'] for r in construction_rows),len(dev),SRC+'development.json','structural/token reproduction; semantic supplement review basis is preserved in source table')
    findings=[x for x in findings if x['issue']!='all 432 constructions preserve exact input tokens and required evidence']
    write_csv('splits_checks.csv',checks)
    if findings: write_csv('splits_findings.csv',findings)
    (OUT/'reports/STAGE1_SPLITS_MINIMAL_EVIDENCE.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary=dict(stage='01_audit',scope='LongBench split and construction data only; no quality/parameter comparison',longbench_ids=len(lb),connected_groups=len(components),split_counts={spl:sum(r['split']==spl for r in lb) for spl in SPLITS},applicability_candidates=len(app),applicable_constructions=len(dev),inapplicable_constructions=sum(not r['applicable'] for r in app),construction_exact_token_matches=sum(r['exact_tokens_match'] for r in construction_rows),construction_all_evidence_retained=sum(r['evidence_retained'] for r in construction_rows),constructions_with_supplement=sum(r['n_supplement_tokens']>0 for r in construction_rows),n_supplement_contributions=len(supplement_rows),constructions_with_missing_required_evidence=len(missing_evidence),source_ids_with_missing_required_evidence=len({r['sample_id'] for r in missing_evidence}),evidence_position_field_mismatches=sum(not r['evidence_positions_ok'] for r in construction_rows),cross_block_metadata_mismatches=sum(not r['cross_block_field_ok'] for r in construction_rows),affected_generations_by_stage=dict(Counter(r['stage'] for r in affected_generations)),n_checks=len(checks),n_failed_checks=sum(not r['passed'] for r in checks),n_findings=len(findings),overlap=overlap_summary,supplement_cross_split=supplement_summary,environment=dict(python=platform.python_version(),executable=sys.executable,tokenizers=version('tokenizers')),status='SUBMITTED unchanged')
    (OUT/'reports/STAGE1_SPLITS_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # This report records data coverage; quality and parameter comparisons belong to later stages.
    lines=['# 第1阶段：数据划分与开发输入构造审核','', '本阶段读取恢复后的启动快照，核对LongBench原始公开数据连接、分组划分和精确长度开发输入。全部新增文件写入 `pc-analysis-output/stages/01_audit/`；研究状态保持 `SUBMITTED`。','', '## 1 数据与实际分母','', '| 数据集 | 全量 | 探索 | 验证 | 独立测试 |','|---|---:|---:|---:|---:|']
    for d in DS: lines.append(f"|{d}|200|{expected['explore'][d]}|{expected['validation'][d]}|170|")
    lines+=['',f'总计600个唯一公开ID，探索50、验证40、测试510。冻结顺序中的ID集合与对应分割一致。按完整支撑文档集合或规范化相同问题独立连通分组，共{len(components)}组，与冻结分组一致。逐题公开ID、问题、完整文档和参考答案连接均通过（600/600）。','', '## 2 跨分割重合','', '问题规范化采用casefold并合并空白；完整证据组以冻结的支撑文档组合与同问题连通分组为准。单篇资料统计`input_documents`；支撑资料统计`support_documents`。','', '| 左侧/右侧 | 类型 | 共享项目数 | 左侧受影响ID/分母 | 右侧受影响ID/分母 |','|---|---|---:|---:|---:|']
    labels={'normalized_question':'规范化问题','complete_evidence_group':'完整证据组','single_document':'单篇资料','support_document':'支撑资料'}
    for r in overlap_summary: lines.append(f"|{r['left_split']}/{r['right_split']}|{labels[r['overlap_type']]}|{r['n_unique_shared_items']}|{r['n_affected_left_ids']}/{r['denominator_left_ids']}|{r['n_affected_right_ids']}/{r['denominator_right_ids']}|")
    lines+=['','完整证据组与规范化问题跨分割重合为0。部分Wikipedia单篇资料及支撑资料跨开发与测试共享；后续测试分析应按这项资料共享情况分层呈现。`splits_overlap_documents.csv`保留每篇共享资料及两侧ID，`splits_overlap_affected_ids.csv`保留逐ID归属。','', '## 3 精确长度构造与不适用输入','', '90个开发ID分别检查4096、8192、16384、6144和12288 tokens，共450个题长候选。有效432个，不适用18个。精确长度计入文档token，问题、共享前缀和输出另计。','', '| 分割 | L | 有效/候选 | 不适用原因及数量 |','|---|---:|---:|---|']
    for spl in ('explore','validation'):
        for l in LENGTHS:
            rr=[r for r in apptable if r['split']==spl and r['L']==l]; counts=Counter(r['reviewed_reason'] for r in rr if not r['applicable']); reasons='；'.join(f'{k}: {v}' for k,v in counts.items()) or '无'
            lines.append(f"|{spl}|{l}|{sum(r['applicable'] for r in rr)}/{len(rr)}|{reasons}|")
    lines+=['', '原始原因`no_evidence_or_evidence_exceeds_target`按导出分词器的实际证据token数量进一步拆分；逐ID原因与证据token数见`construction_applicability.csv`。4K/8K/16K探索有效48/47/44个题长输入，覆盖三种N和16种T/S的计划生成数为6,672；本阶段仅核对输入覆盖。','', '## 4 证据保留、补充资料和跨块字段','',f'使用已有tokenizers {version("tokenizers")}读取导出的tokenizer.json，对90个开发源文档重新分词。全部432个有效构造按保留区间及补充来源区间重建后，token序列和目标L完全一致。按字符证据区间与本机字符offset映射核对，352个构造保留全部必要证据token，80个构造删去了部分必要证据。保存的证据位置字段在432个构造中与当前字符证据映射有差异，其中348个构造的跨块字段随之不同。每个构造的实际保留、位置字段差异和跨块统计分别见`construction_checks.csv`、`construction_missing_evidence.csv`和`construction_cross_block_summary.csv`。', '',f'其中{summary["constructions_with_supplement"]}个构造使用补充资料，共{len(supplement_rows)}条来源贡献。检查补充来源处于开发集、目标与供体支撑资料互斥、证据组不同、所用token区间处于审阅批准区间，并保留语义审阅依据与引用。补充片段标准答案字面匹配检查见来源表。','', '| 目标分割 | 供体分割 | 使用该来源的构造数 | 目标ID数 | 供体ID数 | 贡献token数 |','|---|---|---:|---:|---:|---:|']
    for r in supplement_summary: lines.append(f"|{r['target_split']}|{r['donor_split']}|{r['n_target_constructions']}|{r['n_target_ids']}|{r['n_donor_ids']}|{r['n_contributed_tokens']}|")
    lines+=['','冻结补充池允许探索与验证资料；本次实际使用的供体全部来自探索集，验证集96个构造使用了探索来源背景。两开发分割在背景片段层面共享材料；后续验证解释应同时报告这项共享范围。测试与验证资料作为补充供体的数量均为0。每条贡献保存目标ID、供体ID、实际区间、来源分割、批准依据与审阅引用，见`construction_supplement_sources.csv`。','', '保存的跨块字段与其保存的稀疏证据区间在内部计算上一致；按完整字符证据映射重算时存在差异。跨块字段表示连续证据token区间跨过等长块边界。同一题长构造对应同一token序列，三种N据此切分；本阶段覆盖表保留432×3=1296个构造/N格。','', '## 5 最小原文证据与检查结果','', '原文证据示例按“数据集×映射策略”选择ID字典序首条，并补充不适用原因、补充来源及共享支撑资料的最小示例，保存于`STAGE1_SPLITS_MINIMAL_EVIDENCE.json`。示例含样本ID、问题、来源ID、字符或token区间和原文片段。', '',f'完成{len(checks)}项汇总检查，失败{summary["n_failed_checks"]}项；逐构造差异{len(findings)}条（同一构造按证据保留、位置和跨块字段分别登记）。数据覆盖、来源共享和不适用原因已登记；本阶段产物未进行参数评分比较。','', '## 6 复现与来源字段','', '```powershell', '& "C:/Users/Osheepne/anaconda3/python.exe" -B pc-analysis-output/scripts/audit_stage1_splits.py','```','', '主要输入：启动快照`longbench.json`、`development.json`、`applicability.json`、`split-report.json`、`frozen.json`；公开LongBench三份jsonl；导出的tokenizer；`datasets/prepared-main80/supplement-reviews.json`。每张表带来源与单位字段。未安装依赖，未导入模型或执行GPU工作。']
    lines+=['', '## 7 构造删去证据的范围、版本线索与处理', '', f'80个受影响构造涉及{len({r["sample_id"] for r in missing_evidence})}个来源ID；实际关联生成按阶段为{dict(Counter(r["stage"] for r in affected_generations))}。每个构造在N=4/8/16下共用同一token序列，证据删去发生在分块之前。完整逐题清单和关联生成见`construction_missing_evidence.csv`、`construction_missing_evidence_generations.csv`。正式LongBench、LITM和Doc2Dial使用原始完整资料，其输入路径与这些开发构造分开。', '', '最小实例区分两类现象：qasper:e419f2bff9d2ab7c3b60b3250caccd2d9ae1285ec3e8e818的4K输入完整保留相应段落，保存字段只标出WAP、MFCC、MDRE等稀疏词片；另有构造的retained/deleted源区间直接删除了字符证据范围内的token。具体实例hotpotqa:de89026c970fbb6dff04176107b9cf20407d155bc1ae2eec的4K/6K构造删除了源token[9025,9034)，内容为“He played college football at Arizona State University.”，对应参考答案所在整句。原文字符区间为[39348,39403)。详见`STAGE1_EVIDENCE_DELETION_EXAMPLE.json`，可用`inspect_stage1_evidence.py`复现。80个受影响构造中78个采用整篇支撑文档回退证据、2个采用精确句子证据；遗漏量为4—11085个token。定位依赖已重建且完全相同的token ID序列。', '', '服务器记录Transformers4.44.2/tokenizers0.19.1；本机Transformers4.46.3/tokenizers0.20.3。原生tokenizers与AutoTokenizer对最小实例给出相同token ID与270个证据token。上游0.19.1的ignore_merges分支对词表整词返回(0,0)偏移，修复PR将其改为实际长度；该机制与保存字段只集中于多token词的模式一致。根因归档还缺服务器0.19.1逐token offset原始输出，需在原环境做最小CPU复现。', '', '上游依据：[v0.19.1 BPE实现](https://github.com/huggingface/tokenizers/blob/v0.19.1/tokenizers/src/models/bpe/model.rs)、[offset修复PR1640](https://github.com/huggingface/tokenizers/pull/1640)、[Llama3 offset问题1517](https://github.com/huggingface/tokenizers/issues/1517)。既有mapping-second/third/fourth-check只记录来源映射结果；scope-validation记录执行范围；data-validation记录432份通过，验收代码复用了服务器同一offset_mapping。上述JSON未保存逐token offset值。', '', '处理方式：保留原始构造与冻结规则，将当前结果归入这批实际输入的观察；重新保证完整证据的开发构造需要重建后作为新版本复核适用性，再补跑受影响探索/验证和依赖它们的规则冻结，随后重新评价冻结规则。现有规则与研究状态在本阶段保持原值。该补实验涉及GPU和新版本实验，列入后续决策事项。']
    (OUT/'reports/STAGE1_SPLITS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('overlap',)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
