import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
started = datetime.now(timezone.utc).isoformat()
snapshot = json.loads((BASE / 'input.snapshot.json').read_text())
current = json.loads(Path('logs/runs/20261113-main80-prep/supplement-proposals.json').read_text())
review = json.loads((BASE / 'review.json').read_text())
assert current == snapshot, 'Source changed after the reviewed snapshot.'
assert len(review) == len(snapshot['proposals']) == 88
assert len(snapshot['catalog']) == 395
assert len({r['target_sample'] for r in review}) == len(review)
approved_pairs = rejected_pairs = 0
overlaps = []
source_splits = []
all_used = []
for p, r in zip(snapshot['proposals'], review):
    assert p['target_sample'] == r['target_sample']
    assert r['question'] == p['question'] and r['reference_answers'] == p['reference_answers']
    approved = r['approved_catalog_ids']
    rejected = [x['catalog_id'] for x in r['rejected_catalog_ids']]
    assert len(approved) == len(set(approved))
    assert len(rejected) == len(set(rejected))
    assert all(cid in p['catalog_ids'] for cid in approved + rejected)
    assert [cid for cid in p['catalog_ids'] if cid in approved] == approved
    assert [cid for cid in p['catalog_ids'] if cid in rejected] == rejected
    assert len(approved) + len(rejected) == len(p['catalog_ids'])
    assert not any(cid in rejected for cid in approved)
    assert r['status'] == 'SUBMITTED' and r['reviewer_role'] == 'EXECUTOR'
    assert r['contains_additional_answer_evidence'] is (False if approved else None)
    assert r['basis']
    assert all(x['reason'] for x in r['rejected_catalog_ids'])
    assert r['approved_token_count'] == sum(snapshot['catalog'][cid]['token_interval'][1] - snapshot['catalog'][cid]['token_interval'][0] for cid in approved)
    approved_pairs += len(approved)
    rejected_pairs += len(rejected)
    normalized_texts = []
    for cid in p['catalog_ids']:
        donor = snapshot['catalog'][cid]
        text = ' '.join(donor['text'].split())
        assert text not in normalized_texts, ('Repeated proposed text', p['target_sample'], cid)
        normalized_texts.append(text)
        if donor['source_split'] not in source_splits:
            source_splits.append(donor['source_split'])
        shared = [doc for doc in donor['source_documents'] if doc in p['target_documents']]
        if shared:
            overlaps.append({'target': p['target_sample'], 'catalog_id': cid, 'shared_documents': shared})
        if cid not in all_used:
            all_used.append(cid)
    assert p['split'] in ['explore', 'validation']
assert not overlaps
assert all(s in ['explore', 'validation'] for s in source_splits)
assert len(all_used) == 395
summary = {
    'status': 'validation_passed_for_executor_submission_only',
    'source_unchanged_by_direct_json_content_comparison': True,
    'targets': len(review),
    'catalog_paragraphs': len(snapshot['catalog']),
    'catalog_paragraphs_selected_by_at_least_one_proposal': len(all_used),
    'source_splits_in_metadata': source_splits,
    'proposed_target_paragraph_pairs': approved_pairs + rejected_pairs,
    'approved_target_paragraph_pairs': approved_pairs,
    'rejected_target_paragraph_pairs': rejected_pairs,
    'source_target_document_overlap_count_from_supplied_metadata': len(overlaps),
    'order_and_complete_partition_preserved': True,
    'no_repeated_normalized_paragraph_text_within_any_proposal': True,
    'limitations': [
        'Document/split checks use supplied proposal metadata; this review did not reconstruct or recertify the upstream raw-data mapping.',
        'Token counts are sums of supplied token interval lengths, not a new tokenization or a proof of final exact-length feasibility.',
        'Semantic decisions are review judgments against supplied questions/references; no model output or performance was consulted.',
        'No final data freeze, acceptance, model execution or campaign launch is performed by this EXECUTOR.',
    ],
}
path = BASE / 'validation.json'
assert not path.exists()
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
record_path = BASE / 'execution.json'
record = json.loads(record_path.read_text())
record['commands'].append({
    'cmd': 'bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B logs/runs/20261114-main80-supplements-review/validate_review.py',
    'started_at': started,
    'ended_at': datetime.now(timezone.utc).isoformat(),
    'exit_code': 0,
    'purpose': 'Check IDs, subset/order/partition, retained-only evidence flag, supplied split/document metadata and direct input-content equality.',
})
record['validation_delivery'] = str(path.relative_to(Path.cwd()))
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
for r in review:
    if r['rejected_catalog_ids']:
        print('REJECTION_SUMMARY', r['target_index'], r['target_sample'], r['approved_paragraph_count'], r['rejected_paragraph_count'], r['approved_token_count'])
