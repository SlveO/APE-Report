"""Describe frozen correction/natural selections; paired support-group inference.

Only existing answers and saved selections are read. Historical code is not run.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parent
PKG = REPO / "restored/followup2604/FOLLOWUP2604_ACCEPTED"
os.environ["MPLCONFIGDIR"] = str(OUT / "cache/matplotlib")
os.environ["TMP"] = os.environ["TEMP"] = str(OUT / "tmp")
for directory in ["tables", "figures", "reports", "logs", "cache/matplotlib", "tmp"]:
    (OUT / directory).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261116
B = 5000
DATASETS = ["qasper", "hotpotqa", "musique"]
CELLS = [(l, n) for l in [4096, 8192, 16384] for n in [4, 8, 16]]
PARAMS = [(t, s) for t in [.5, .7, .9, 1.] for s in [.5, .7, .9, 1.]]
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "svg.fonttype": "none", "figure.dpi": 130})


def read_json(name):
    return json.loads((PKG / name).read_text(encoding="utf-8"))


def read_lines(path):
    with path.open(encoding="utf-8") as f:
        return [dict(json.loads(line), _source_line=i) for i, line in enumerate(f, 1) if line.strip()]


def csvout(name, rows):
    assert rows, name
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "tables" / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row["sample_id"], int(row["L"]), int(row["N"]), float(row["T"]), float(row["S"])


def group(row):
    return json.dumps(row["group"], sort_keys=True, ensure_ascii=False)


def balanced(rows, field="score"):
    bins = defaultdict(list)
    for row in rows:
        bins[(row["dataset"], row["L"], row["N"])].append(float(row[field]))
    assert set(bins) == {(d, l, n) for d in DATASETS for l, n in CELLS}
    return float(np.mean([np.mean(v) for v in bins.values()]))


def ci(values):
    return tuple(float(x) for x in np.quantile(np.asarray(values) * 100, [.025, .975]))


def support_bootstrap(items, policies, tag):
    """Each item has dataset, frozen support group, L/N and paired policy scores.

    Equal dataset/cell means; all views in a support group have one multiplicity.
    Resample a dataset draw when any of its nine cells is empty.
    """
    result = np.empty((B, len(DATASETS), len(CELLS), len(policies)))
    diagnostics = []
    for di, dataset in enumerate(DATASETS):
        rows = [r for r in items if r["dataset"] == dataset]
        groups = sorted({r["group_key"] for r in rows})
        sums = np.zeros((len(groups), 9, len(policies)))
        counts = np.zeros((len(groups), 9))
        gindex = {g: i for i, g in enumerate(groups)}
        for r in rows:
            gi, cell = gindex[r["group_key"]], CELLS.index((r["L"], r["N"]))
            sums[gi, cell] += [r[p] for p in policies]
            counts[gi, cell] += 1
        rng = np.random.default_rng(np.random.SeedSequence([SEED, tag, di]))
        accepted, proposals, rejects = 0, 0, 0
        while accepted < B:
            weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
            den = weights @ counts
            proposals += 1
            if np.any(den == 0):
                rejects += 1
                continue
            result[accepted, di] = np.einsum("g,gcp->cp", weights, sums) / den[:, None]
            accepted += 1
        diagnostics.append({"scope": tag, "scope_name": {110:"exploration_fixed_parameter_old_corrected_input",111:"A_frozen_policy_validation",112:"validation_default_parameter_old_corrected_input"}[tag], "dataset": dataset, "frozen_support_groups": len(groups),
                            "unique_questions": len({r["sample_id"] for r in rows}),
                            "input_conditions": len(rows), "effective_replicates": B,
                            "proposals": proposals, "empty_cell_redraws": rejects,
                            "seed": SEED, "rng": f"SeedSequence([{SEED},{tag},{di}])",
                            "unit": "frozen support group, all question/cell/policy views retained"})
    return result, diagnostics


def savefig(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    olddev = read_json("inputs/datasets/frozen-main80/development.json")
    newdev = read_json("inputs/datasets/frozen-main80-corrected-v1/development.json")
    oldmap = {(r["sample_id"], r["L"]): r for r in olddev}
    newmap = {(r["sample_id"], r["L"]): r for r in newdev}
    assert len(oldmap) == len(olddev) == 432 and len(newmap) == len(newdev) == 359
    assert not (newmap.keys() - oldmap.keys())
    fields = ["document_tokens", "prefix", "suffix", "answers", "max_new_tokens", "scorer"]
    identity = {}
    input_rows = []
    for sample_key in sorted(oldmap):
        old, new = oldmap[sample_key], newmap.get(sample_key)
        diffs = [f for f in fields if new is not None and old[f] != new[f]]
        status = "removed_from_applicable" if new is None else "changed_generation_input" if diffs else "identical_generation_input"
        identity[sample_key] = status
        input_rows.append({"sample_id": old["sample_id"], "dataset": old["dataset"], "split": old["split"], "L": old["L"],
                           "status": status, "changed_fields": json.dumps(diffs),
                           "old_evidence_token_count": sum(b-a for a,b in old["evidence_token_intervals"]),
                           "new_evidence_token_count": sum(b-a for a,b in new["evidence_token_intervals"]) if new else "",
                           "saved_evidence_metadata_changed": old["evidence_token_intervals"] != new["evidence_token_intervals"] if new else "",
                           "unit": "sample_id × document length; exact token/prefix/suffix/reference/generation-control comparison",
                           "source": "inputs/datasets/frozen-main80{,-corrected-v1}/development.json"})
    csvout("a_input_conditions.csv", input_rows)
    bins = defaultdict(Counter)
    for r in input_rows:
        bins[(r["split"], r["L"])][r["status"]] += 1
    count_rows = [{"split": split, "L": l, "old_applicable": sum(c.values()),
                   "corrected_applicable": c["identical_generation_input"] + c["changed_generation_input"],
                   "identical_inputs": c["identical_generation_input"], "changed_inputs": c["changed_generation_input"],
                   "removed_inputs": c["removed_from_applicable"], "unit": "sample-length construction"}
                  for (split,l),c in sorted(bins.items())]
    csvout("a_input_counts.csv", count_rows)

    ledger = REPO / "restored/output/runs/main80-20261114T041503/attempts.jsonl"
    old_all = read_lines(ledger)
    oldexp = [r for r in old_all if r["key"]["stage"] == "explore"]
    oldval = [r for r in old_all if r["key"]["stage"] == "validation"]
    del old_all
    newexp = read_lines(PKG / "provenance/corrected-exploration-for-candidate-reconstruction.jsonl")
    validation = read_lines(PKG / "server/corrected-validation-results.jsonl")
    assert (len(oldexp),len(newexp),len(validation)) == (6672,5328,839)
    oldexpmap, oldvalmap = {key(r):r for r in oldexp}, {key(r):r for r in oldval}
    assert len(oldexpmap) == len(oldexp)
    assert {key(r) for r in newexp}.issubset(oldexpmap)
    nom_old = json.loads((REPO / "restored/output/runs/main80-20261114T041503/candidates.json").read_text(encoding="utf-8"))
    nom_new = read_json("inputs/logs/runs/20261117-main80-correction/corrected-candidates.json")
    oldrule = read_json("inputs/output/runs/main80-20261114T041503/rule.json")
    newrule = read_json("server/corrected-rule.json")
    assert newrule["status"] == "FROZEN"

    matched = []
    for new in newexp:
        old = oldexpmap[key(new)]
        sameinput = identity[(new["sample_id"],new["L"])] == "identical_generation_input"
        reused = old["started_at"] == new["started_at"] and old["output_token_ids"] == new["output_token_ids"]
        if reused:
            assert sameinput and old["score"] == new["score"]
        matched.append({"sample_id": new["sample_id"], "dataset": new["dataset"], "L":new["L"], "N":new["N"], "T":new["T"], "S":new["S"],
                        "group_key":group(new), "input_status":identity[(new["sample_id"],new["L"])],
                        "historical_generation_reused": reused, "old_score":old["score"], "corrected_score":new["score"],
                        "delta_pp":(new["score"]-old["score"])*100,
                        "old_source_line":old["_source_line"], "corrected_source_line":new["_source_line"],
                        "old_answer":old["answer"], "corrected_answer":new["answer"], "reference_answers":json.dumps(new["reference_answers"],ensure_ascii=False),
                        "unit":"same sample/L/N/T/S; input identity explicitly classified; F1 raw0-1"})
    assert sum(r["historical_generation_reused"] for r in matched) == 4080
    csvout("a_exploration_matched_records.csv", matched)
    newkeys = {key(r) for r in newexp}
    oldrestricted = [r for r in oldexp if key(r) in newkeys]
    exploration_scores = []
    for t,s in PARAMS:
        oldmean=balanced([r for r in oldexp if (r["T"],r["S"])==(t,s)])
        commonold=balanced([r for r in oldrestricted if (r["T"],r["S"])==(t,s)])
        newmean=balanced([r for r in newexp if (r["T"],r["S"])==(t,s)])
        assert abs(newmean-nom_new["balanced_scores"]["global"][f"{t}/{s}"])<1e-12
        exploration_scores.append({"T":t,"S":s,"old_all_f1_percent":oldmean*100,"old_common_f1_percent":commonold*100,
                                   "corrected_common_f1_percent":newmean*100,"sample_applicability_shift_pp":(commonold-oldmean)*100,
                                   "same_parameter_input_shift_pp":(newmean-commonold)*100,
                                   "old_all_conditions":417,"common_conditions":333,
                                   "saved_old_global":list((t,s))==nom_old["global"],"saved_corrected_global":list((t,s))==nom_new["global"],
                                   "unit":"F1×100; equal three datasets and nine cells; descriptive decomposition, no new selection"})
    csvout("a_exploration_scores.csv", exploration_scores)
    held_items = defaultdict(dict)
    for r in matched:
        if (r["T"],r["S"]) not in [(.7,1.),(.5,1.)]:continue
        k=r["sample_id"],r["L"],r["N"]
        x=held_items[k]
        x.update({f"{prefix}_{r['T']}":r[source] for prefix,source in [("old","old_score"),("corrected","corrected_score")]})
        x.update({k:r[k] for k in ["sample_id","dataset","L","N","group_key"]})
    held_policies=["old_0.7","corrected_0.7","old_0.5","corrected_0.5"]
    exp_draws, diag_exp=support_bootstrap(list(held_items.values()),held_policies,110)
    exp_macro=exp_draws.mean(axis=(1,2))
    exp_comparisons=[]
    for oldi,newi,t in [(0,1,.7),(2,3,.5)]:
        means=[r for r in exploration_scores if r["T"]==t and r["S"]==1][0]
        lo,hi=ci(exp_macro[:,newi]-exp_macro[:,oldi])
        exp_comparisons.append({"scope":"matched corrected-applicable exploration","T":t,"S":1.,"n_input_conditions":333,
                                "old_common_f1_percent":means["old_common_f1_percent"],"corrected_f1_percent":means["corrected_common_f1_percent"],
                                "delta_corrected_minus_old_pp":means["same_parameter_input_shift_pp"],"ci_low_pp":lo,"ci_high_pp":hi,
                                "unit":"fixed parameter across old/corrected inputs; equal dataset/cell means","replicates":B,"seed":SEED})
    csvout("a_exploration_frozen_parameter_pairs.csv",exp_comparisons)

    val_by_condition=defaultdict(dict)
    physical=[]
    for r in validation:
        assert r["status"]=="success" and r["attempt"]==1
        ident=identity[(r["sample_id"],r["L"])]
        hist=bool(r.get("historical_reuse",False))
        if hist:assert ident=="identical_generation_input"
        val_by_condition[(r["sample_id"],r["L"],r["N"])][(r["T"],r["S"])]=r
        physical.append({"sample_id":r["sample_id"],"dataset":r["dataset"],"L":r["L"],"N":r["N"],"T":r["T"],"S":r["S"],
                         "score":r["score"],"historical_reuse":hist,"input_status":ident,"logical_labels":json.dumps(r["logical_labels"]),
                         "source_line":r["_source_line"],"historical_source_line":r.get("source_attempts_line",""),"answer":r["answer"],
                         "reference_answers":json.dumps(r["reference_answers"],ensure_ascii=False),"group_key":group(r),
                         "unit":"one actual answer; multiple labels share this row; not repeated runtime"})
    assert len(val_by_condition)==300 and sum(r["historical_reuse"] for r in physical)==377
    csvout("a_validation_records.csv",physical)
    origins=Counter((r['historical_reuse'],r['input_status']) for r in physical)
    origin_rows=[{'stage':'prior_corrected_exploration','generation_origin':'historical_identical_input','n_physical_answers':4080,'unit':'actual prior answers; outside FOLLOWUP2604 new2430'},
                 {'stage':'prior_corrected_exploration','generation_origin':'corrected_input_answer_from_prior_correction_campaign','n_physical_answers':1248,'unit':'actual prior answers; outside FOLLOWUP2604 new2430'}]
    for (historical,status),count in sorted(origins.items()):
        origin_rows.append({'stage':'A_corrected_validation','generation_origin':'historical_identical_input' if historical else 'new_corrected_input' if status=='changed_generation_input' else 'new_candidate_key_on_identical_input',
                            'n_physical_answers':count,'unit':'actual answers; label reuse excluded'})
    csvout('a_generation_origin_counts.csv',origin_rows)
    policies=["default","global","cell","fixed","dynamic"]
    logical=[]; val_items=[]; old_new_defaults=[]
    for k,records in sorted(val_by_condition.items()):
        sid,l,n=k; ref=next(iter(records.values()))
        cell=f"{l}/{n}"
        param={"default":(.9,.9),"global":tuple(nom_new["global"]),"cell":tuple(nom_new["cells"][cell]),
               "fixed":tuple(newrule["fixed"]),"dynamic":tuple(newrule["cells"][cell])}
        item={"sample_id":sid,"dataset":ref["dataset"],"L":l,"N":n,"group_key":group(ref)}
        for policy in policies:
            r=records[param[policy]]
            item[policy]=r["score"]
            if policy in ["default","global","cell"]:
                assert policy in r["logical_labels"]
                logical.append({"sample_id":sid,"dataset":ref["dataset"],"L":l,"N":n,"label":policy,"T":param[policy][0],"S":param[policy][1],
                                "score":r["score"],"physical_source_line":r["_source_line"],"historical_reuse":bool(r.get("historical_reuse",False)),
                                "unit":"logical view; physical source line defines actual generation"})
        val_items.append(item)
        old=oldvalmap.get((sid,l,n,.9,.9))
        assert old is not None
        old_new_defaults.append({**item,"old_default":old["score"],"new_default":item["default"]})
    assert len(logical)==900
    csvout("a_validation_logical_views.csv",logical)
    draws,diag_val=support_bootstrap(val_items,policies,111)
    means=np.array([[[np.mean([r[p] for r in val_items if r['dataset']==d and (r['L'],r['N'])==(l,n)]) for p in policies] for l,n in CELLS] for d in DATASETS])
    qrows=[]; prows=[]
    for label,indices in [("macro_equal_dataset_cell",list(range(3)))]+[(d,[i]) for i,d in enumerate(DATASETS)]:
        point=means[indices].mean(axis=(0,1)); boots=draws[:,indices].mean(axis=(1,2))
        selected=[r for r in val_items if r["dataset"] in [DATASETS[i] for i in indices]]
        for pi,p in enumerate(policies):
            lo,hi=ci(boots[:,pi]);qrows.append({"scope":label,"policy":p,"f1_percent":point[pi]*100,"ci_low_percent":lo,"ci_high_percent":hi,
                                               "n_questions":len({r['sample_id'] for r in selected}),"n_conditions":len(selected),"unit":"F1×100; nine cells equal","seed":SEED,"replicates":B})
        for reference,comparison in [("default","fixed"),("fixed","dynamic"),("fixed","cell")]:
            ai,bi=policies.index(reference),policies.index(comparison)
            lo,hi=ci(boots[:,bi]-boots[:,ai]);delta=point[bi]-point[ai]
            diffs=np.array([r[comparison]-r[reference] for r in selected])
            prows.append({"scope":label,"reference":reference,"comparison":comparison,"delta_pp":delta*100,"ci_low_pp":lo,"ci_high_pp":hi,
                          "wins":int((diffs>1e-12).sum()),"ties":int((abs(diffs)<=1e-12).sum()),"losses":int((diffs<-1e-12).sum()),
                          "n_questions":len({r['sample_id'] for r in selected}),"n_conditions":len(selected),"unit":"F1 percentage points; counts are input-condition pairs","seed":SEED,"replicates":B})
    for ci_index,(l,n) in enumerate(CELLS):
        for reference,comparison in [("fixed","dynamic"),("fixed","cell")]:
            ai,bi=policies.index(reference),policies.index(comparison)
            point=(means[:,ci_index,bi]-means[:,ci_index,ai]).mean();boots=(draws[:,:,ci_index,bi]-draws[:,:,ci_index,ai]).mean(axis=1)
            lo,hi=ci(boots)
            selected=[r for r in val_items if (r["L"],r["N"])==(l,n)]
            diffs=np.array([r[comparison]-r[reference] for r in selected])
            prows.append({"scope":f"{l}/{n}","reference":reference,"comparison":comparison,"delta_pp":point*100,"ci_low_pp":lo,"ci_high_pp":hi,
                          "wins":int((diffs>1e-12).sum()),"ties":int((abs(diffs)<=1e-12).sum()),"losses":int((diffs<-1e-12).sum()),
                          "n_questions":len(selected),"n_conditions":len(selected),"unit":"F1 percentage points; three datasets equal","seed":SEED,"replicates":B})
    # Verify stored cell means using saved labels/parameters, without running selection.
    for name,scores in newrule["validation_scores"].items():
        idx=CELLS.index(tuple(map(int,name.split('/'))))
        assert abs(means[:,idx,2].mean()-scores["candidate"])<1e-12
        assert abs(means[:,idx,3].mean()-scores["fixed"])<1e-12
    csvout("a_validation_policies.csv",qrows);csvout("a_validation_paired.csv",prows)
    default_draws,diag_default=support_bootstrap(old_new_defaults,["old_default","new_default"],112)
    default_point=np.mean([np.mean([r['new_default']-r['old_default'] for r in old_new_defaults if r['dataset']==d and (r['L'],r['N'])==(l,n)]) for d in DATASETS for l,n in CELLS])
    lo,hi=ci((default_draws[:,:,:,1]-default_draws[:,:,:,0]).mean(axis=(1,2)))
    csvout("a_validation_matched_input_comparison.csv",[{"parameter":"0.9/0.9","n_conditions":300,"delta_corrected_minus_old_pp":default_point*100,"ci_low_pp":lo,"ci_high_pp":hi,
                                                         "unit":"same sample/L/N/default; corrected applicability set, input change classified in a_input_conditions","replicates":B,"seed":SEED}])
    csvout("a_bootstrap_diagnostics.csv",diag_exp+diag_val+diag_default)

    rule_rows=[]
    for l,n in CELLS:
        name=f"{l}/{n}"
        rule_rows.append({"L":l,"N":n,"old_candidate":json.dumps(nom_old['cells'][name]),"corrected_candidate":json.dumps(nom_new['cells'][name]),
                          "candidate_changed":nom_old['cells'][name]!=nom_new['cells'][name],"old_frozen_rule":json.dumps(oldrule['cells'][name]),
                          "corrected_frozen_rule":json.dumps(newrule['cells'][name]),"frozen_rule_changed":oldrule['cells'][name]!=newrule['cells'][name],
                          "corrected_candidate_f1_percent":newrule['validation_scores'][name]['candidate']*100,"corrected_fixed_f1_percent":newrule['validation_scores'][name]['fixed']*100,
                          "unit":"saved candidates/frozen rules; no parameter selection performed"})
    csvout("a_rule_changes.csv",rule_rows)

    natural_candidates=read_json("server/natural-candidates.json");natural_rule=read_json("server/natural-rule.json")
    natural_explore=read_lines(PKG/'server/natural-exploration-results.jsonl')
    natural_val=read_lines(PKG/'server/natural-validation-results.jsonl')
    assert len(natural_explore)==864 and len(natural_val)==104
    natural_scores=[]
    for scope,dscores in [("global",natural_candidates['weighted_scores']['global'])]+[(f"N{n}",v) for n,v in natural_candidates['weighted_scores']['by_N'].items()]:
        selected=natural_candidates['global'] if scope=='global' else natural_candidates['by_N'][scope[1:]]
        for params,value in dscores.items():
            t,s=map(float,params.split('/'))
            rows=[r for r in natural_explore if r['T']==t and r['S']==s and (scope=='global' or r['N']==int(scope[1:]))]
            dataset_values=[]
            for dataset in DATASETS:
                sub=[r for r in rows if r['dataset']==dataset]
                if not sub:continue
                if scope=='global' and dataset=='qasper':dataset_values.append(np.mean([np.mean([r['score'] for r in sub if r['N']==n]) for n in [1,2,3]]))
                else:dataset_values.append(np.mean([r['score'] for r in sub]))
            calculated=float(np.mean(dataset_values));assert abs(calculated-value)<1e-12
            natural_scores.append({'scope':scope,'T':t,'S':s,'f1_percent':calculated*100,'saved_candidate':list((t,s))==selected,
                                   'n_questions':len(rows),'unit':'saved C1 weighting; global dataset-equal, Qasper N-equal; N2/N3 Qasper only'})
    csvout('a_natural_development.csv',natural_scores)
    natural_validation=[]
    for param,value in natural_rule['fixed_validation_scores'].items():
        t,s=map(float,param.split('/'));r=[x for x in natural_val if x['T']==t and x['S']==s]
        dvals=[]
        for d in DATASETS:
            sub=[x for x in r if x['dataset']==d]
            dvals.append(np.mean([np.mean([x['score'] for x in sub if x['N']==n]) for n in [1,2,3]]) if d=='qasper' else np.mean([x['score'] for x in sub]))
        assert abs(np.mean(dvals)-value)<1e-12
        natural_validation.append({'scope':'global_fixed_choice','T':t,'S':s,'f1_percent':value*100,'selected_parameters':json.dumps(natural_rule['fixed']),
                                   'n_questions':36,'unit':'C2 weighting dataset-equal, Qasper N-equal; selection already frozen'})
    for n,decision in natural_rule['conditional_validation'].items():
        for parameter,stored in [('candidate','candidate_score'),('fixed','fixed_score')]:
            ts=tuple(decision[parameter]);sub=[r for r in natural_val if r['N']==int(n) and (r['T'],r['S'])==ts]
            calculated=np.mean([np.mean([r['score'] for r in sub if r['dataset']==d]) for d in DATASETS if any(r['dataset']==d for r in sub)])
            assert abs(calculated-decision[stored])<1e-12
        natural_validation.append({'scope':f'N{n}','candidate':json.dumps(decision['candidate']),'candidate_f1_percent':decision['candidate_score']*100,
                                   'fixed':json.dumps(decision['fixed']),'fixed_f1_percent':decision['fixed_score']*100,'delta_candidate_minus_fixed_pp':(decision['candidate_score']-decision['fixed_score'])*100,
                                   'selected_parameters':json.dumps(decision['selected']),'decision':decision['decision'],'n_questions':28 if n=='1' else 4,
                                   'unit':'N1 three-dataset equal; N2/N3 Qasper only; saved decision'})
    csvout('a_natural_validation.csv',natural_validation)

    labels=[f'{t}/{s}' for t,s in PARAMS];x=np.arange(16)
    fig,ax=plt.subplots(figsize=(12,5))
    for col,label in [('old_all_f1_percent','旧探索全部适用'),('old_common_f1_percent','旧答案限定修正适用ID'),('corrected_common_f1_percent','修正输入、同适用ID')]:
        ax.plot(x,[r[col] for r in exploration_scores],marker='o',label=label)
    ax.set_xticks(x,labels,rotation=45);ax.set_ylabel('F1 ×100（27单元等权）');ax.set_xlabel('T/S');ax.set_title('探索变化分解：适用样本变化与同参数输入修正分别呈现');ax.legend();ax.grid(alpha=.2)
    savefig(fig,'a_exploration_decomposition')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,rule,title in zip(axes,[oldrule,newrule],['旧 MAIN80 冻结规则','修正 corrected-nine-cell-v1']):
        arr=np.array([[rule['cells'][f'{l}/{n}'][0] for n in [4,8,16]] for l in [4096,8192,16384]])
        image=ax.imshow(arr,vmin=.5,vmax=1,cmap='YlOrRd')
        for i,l in enumerate([4096,8192,16384]):
            for j,n in enumerate([4,8,16]):ax.text(j,i,' / '.join(map(str,rule['cells'][f'{l}/{n}'])),ha='center',va='center',fontsize=12)
        ax.set_xticks([0,1,2],[4,8,16]);ax.set_yticks([0,1,2],['4K','8K','16K']);ax.set_xlabel('实际N');ax.set_ylabel('开发长度L');ax.set_title(title+'\n格内为T/S；底色为T')
    fig.colorbar(image,ax=axes,label='T',fraction=.03,pad=.03);savefig(fig,'a_rule_changes')
    fig,ax=plt.subplots(figsize=(9,5));mainrows=[r for r in qrows if r['scope']=='macro_equal_dataset_cell']
    heights=[r['f1_percent'] for r in mainrows];low=[r['ci_low_percent'] for r in mainrows];high=[r['ci_high_percent'] for r in mainrows]
    ax.bar(np.arange(5),heights,color=['#777','#4c78a8','#a9b6c4','#59a14f','#f28e2b']);ax.errorbar(np.arange(5),heights,yerr=[np.array(heights)-low,np.array(high)-heights],fmt='none',color='black',capsize=4)
    ax.set_xticks(np.arange(5),['默认','全局候选','逐格候选','修正版固定','修正版动态']);ax.set_ylabel('F1 ×100');ax.set_title('A验证：冻结策略在同输入上的成绩\n95%冻结支撑组配对Bootstrap区间；三数据集×九格等权');ax.grid(axis='y',alpha=.2)
    for i,v in enumerate(heights):ax.text(i,high[i]+.8,f'{v:.3f}',ha='center')
    savefig(fig,'a_validation_policies')
    fig,ax=plt.subplots(figsize=(8,3));ax.axis('off')
    table=ax.table(cellText=[[n,'/'.join(map(str,v['candidate'])),'/'.join(map(str,v['fixed'])),f"{(v['candidate_score']-v['fixed_score'])*100:+.3f}",'/'.join(map(str,v['selected']))] for n,v in natural_rule['conditional_validation'].items()],
                   colLabels=['实际N','C1候选T/S','C2固定T/S','C2候选−固定/pp','冻结动态T/S'],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.8);ax.set_title('自然分块轨道单独冻结：natural-chunk-v2\nN2/N3候选验证仅来自Qasper')
    savefig(fig,'a_natural_rule')

    macroq={r['policy']:r for r in qrows if r['scope']=='macro_equal_dataset_cell'}
    macrop=[r for r in prows if r['scope']=='macro_equal_dataset_cell']
    result={'stage':'A corrected selection audit','records':{'old_exploration':len(oldexp),'corrected_exploration':len(newexp),'corrected_exploration_new':1248,'corrected_exploration_historical_reuse':4080,
            'A_physical_results':839,'A_new':462,'A_historical_reuse':377,'A_logical_labels':900,'A_input_conditions':300,'A_sample_lengths':100},
            'input_classification':dict(Counter(r['status'] for r in input_rows)),'old_fixed':oldrule['fixed'],'corrected_fixed':newrule['fixed'],
            'candidate_changed_cells':sum(r['candidate_changed'] for r in rule_rows),'frozen_rule_changed_cells':sum(r['frozen_rule_changed'] for r in rule_rows),
            'validation_macro':macroq,'validation_paired':macrop,'same_parameter_exploration_pairs':exp_comparisons,
            'natural_fixed':natural_rule['fixed'],'natural_by_N':natural_rule['by_N'],'natural_validation':natural_validation,
            'bootstrap_seed':SEED,'bootstrap_replicates':B,'bootstrap_unit':'frozen evidence/support group within dataset; retain all question/L/N/policy observations; nine cells equal',
            'new_parameter_selection_performed':False,'new_model_answers':0,'historical_scripts_executed':False}
    (OUT/'a_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    d=next(r for r in macrop if r['reference']=='fixed' and r['comparison']=='dynamic')
    f=next(r for r in macrop if r['reference']=='default' and r['comparison']=='fixed')
    report=f'''# A阶段：修正探索、验证与冻结规则核对

我们仅连接已有回答和保存候选/冻结规则，未重新选参。旧探索6672条、修正探索5328条；A验证839个实际回答对应900个逻辑候选视图，包含462个新回答和377个历史复用。复用视图保留同一物理源行，未增加独立样本或计时。

## 1 输入修复与适用样本变化

冻结开发构造由432份变为359份：73份退出适用清单，98份生成输入改变，261份生成输入相同。直接比较document_tokens、prefix、suffix、answers、max_new_tokens、scorer。修正冻结文件明确将token IDs与ByteLevel token bytes直接映射到原文UTF-8 bytes，弃用native tokenizers0.19.1 offset_mapping。一般构造原则为保留已验证证据、缺负证据补充审定则不适用；旧/修正开发文件只有适用构造，73份退出项的逐题原因字段未随包保存，逐ID清单保留于a_input_conditions.csv。

上述一般政策与字符映射的准确来源为包内inputs/datasets/frozen-main80-corrected-v1/frozen.json的construction_policy、character_mapping字段；逐ID存在性与输入比较来源为同目录development.json和inputs/datasets/frozen-main80/development.json。

实际探索由139题长缩至111，4K/8K/16K为32/35/44，覆盖48题（Qasper15、HotpotQA17、MuSiQue16）；A验证由119题长缩至100，分别27/34/39，仍覆盖40题。新旧沿用原50/40开发问题的冻结名册。跨版本分数分开呈现全部旧样本、旧答案限定共同适用ID及修正输入共同ID；这些差包含输入与样本构成变化。

修正探索5328条=1248条此前修正campaign产生的回答+4080条严格同输入MAIN80历史回答，全部16组参数配对。这1248条作为provenance带入，未计入FOLLOWUP2604新增2430。a_exploration_scores.csv逐参数列出样本退出的均值变化及同样本/同参数输入修正对应变化；a_exploration_frozen_parameter_pairs.csv另给两个已冻结全局参数的配对组Bootstrap区间。

以旧全局0.7/1.0为固定参照：旧全量F1=41.1629，限定修正适用ID后的旧答案=42.1221，修正输入同参数=42.0641，最后改用保存新版全局0.5/1.0为42.1856。三段变化分别约+0.9592、-0.0580、+0.1215个百分点。同参数修正输入差-0.0580的95%区间为[-1.6027,1.5288]。新旧总均值变化据此分为样本适用性、输入和参数变化，保留其中不利的一段。A新增462条进一步分为240条输入变化和222条同输入新增候选键，a_generation_origin_counts.csv保存计数。

## 2 已冻结的参数变化

探索全局候选由0.7/1.0改为0.5/1.0，逐格提名4/9格改变。A按预定默认/全局候选比较冻结固定0.5/1.0；九格最终6/9格改变。旧MAIN80、corrected-nine-cell-v1、natural-chunk-v2分别保存。

A默认F1={macroq['default']['f1_percent']:.4f}，修正版固定={macroq['fixed']['f1_percent']:.4f}，修正版动态={macroq['dynamic']['f1_percent']:.4f}。固定−默认={f['delta_pp']:+.4f}个百分点，95%CI[{f['ci_low_pp']:.4f},{f['ci_high_pp']:.4f}]；动态−固定={d['delta_pp']:+.4f}个百分点，95%CI[{d['ci_low_pp']:.4f},{d['ci_high_pp']:.4f}]。这些是冻结策略在选择用验证集上的评价；B迁移与C独立确认分别承担后续评价。

![探索分解](../figures/a_exploration_decomposition.png)

![冻结规则变化](../figures/a_rule_changes.png)

![A验证](../figures/a_validation_policies.png)

## 3 自然分块轨道的保存选择

C1为54题×16参数=864回答；C2为36题、104个物理回答及108逻辑视图。保存全局候选/固定为1.0/0.9，动态N1/N3回退固定，N2取0.9/1.0。C1/C2全局权重为数据集等权、Qasper内N等权；N1候选三数据集等权，N2/N3仅Qasper。C3主指标仍按每数据集题均值后等权汇总。

![自然规则](../figures/a_natural_rule.png)

## 4 统计和证据

A冻结支撑组数Qasper/HotpotQA/MuSiQue为10/13/14，同一组全部题、长度、块数和策略共同抽入。seed={SEED}，5000次有效配对组Bootstrap；每次先计算各数据集×九格均值再等权。缺失格的该数据集抽样重抽，次数在a_bootstrap_diagnostics.csv；抽样中参数与规则固定。胜/平/负按输入条件计数，独立来源数另列。

三个独立随机流的scope为：110是探索同参数旧/修正输入比较（固定0.7/1.0与0.5/1.0），各数据集5000提出/5000有效、重抽0；111是A验证五个冻结策略主分析，MuSiQue5002提出、重抽2、5000有效；112是默认0.9/0.9不变、限定修正适用ID的旧/修正验证输入比较，MuSiQue5003提出、重抽3、5000有效。111/112其余数据集均重抽0。不同scope区间分别对应各自明确的比较，使用各自SeedSequence，全部诊断均保留。

数据来源为包内provenance/corrected-exploration-for-candidate-reconstruction.jsonl、server/corrected-validation-results.jsonl、inputs/冻结构造与候选、server/corrected-rule.json，以及原MAIN80 attempts.jsonl。所有服务器绝对路径仅作原始文字，实际读取包内相对路径。源行、原答案、参考、输入类别、物理生成及逻辑视图均在a_前缀CSV。修改输入后的得分变化与改用参数的同输入差分别报告，保留退化及跨零区间。
'''
    (OUT/'reports/A_CORRECTION.md').write_text(report,encoding='utf-8')
    (OUT/'logs/a_analysis.json').write_text(json.dumps({'status':'PASS','seed':SEED,'replicates':B,'diagnostics':diag_exp+diag_val+diag_default,'new_selection':False,'new_answers':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','A_macro_fixed':macroq['fixed']['f1_percent'],'A_macro_dynamic':macroq['dynamic']['f1_percent'],'A_dynamic_minus_fixed':d,'inputs':result['input_classification']},ensure_ascii=False))


if __name__=='__main__':
    main()
