"""Paired support-component analysis of frozen natural-rule confirmation answers.

Only Python/NumPy and existing answers are used. No model, historical program,
parameter selection or test-dependent rule adjustment is executed.
"""
import csv
import itertools
import json
import math
import os
import platform
import re
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parent
PKG = REPO / "restored/followup2604/FOLLOWUP2604_ACCEPTED"
SEED = 20261116
REPLICATES = 5000
DATASETS = ["qasper", "hotpotqa", "musique"]
METHODS = ["sequential", "pcw", "ape_default", "ape_main80_fixed", "ape_main80_dynamic",
           "ape_natural_fixed", "ape_natural_dynamic"]
FIXED, DYNAMIC = METHODS.index("ape_natural_fixed"), METHODS.index("ape_natural_dynamic")


def jload(rel):
    return json.loads((PKG / rel).read_text(encoding="utf-8-sig"))


def jsonl(rel):
    with (PKG / rel).open(encoding="utf-8-sig") as f:
        return [(line_no, json.loads(line)) for line_no, line in enumerate(f, 1) if line.strip()]


def csvout(name, rows):
    assert rows, name
    with (OUT / "tables" / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_f1(answer, refs):
    pred = normalized(answer).split()
    best = 0.0
    for ref in refs:
        target = normalized(ref).split()
        overlap = sum((Counter(pred) & Counter(target)).values())
        score = 2 * overlap / (len(pred) + len(target)) if overlap else 0.0
        best = max(best, score)
    return best


def bounds(values):
    return np.quantile(values, [.025, .975], axis=0)


def summarize_quality(scope, point, boot, n, clusters, rows, paired, note):
    lo, hi = bounds(boot)
    for m, method in enumerate(METHODS):
        rows.append({"scope": scope, "method": method, "n_questions": n, "n_components": clusters,
                     "f1_0_1": float(point[m]), "f1_percent": float(point[m] * 100),
                     "ci_low_percent": float(lo[m] * 100), "ci_high_percent": float(hi[m] * 100),
                     "seed": SEED, "replicates": REPLICATES, "unit": "F1 x100; raw F1 also retained",
                     "sampling": note})
    for a, b in itertools.combinations(range(len(METHODS)), 2):
        ci = bounds((boot[:, b] - boot[:, a]) * 100)
        paired.append({"scope": scope, "reference_method": METHODS[a], "comparison_method": METHODS[b],
                       "n_questions": n, "n_components": clusters,
                       "delta_0_1": float(point[b] - point[a]), "delta_pp": float((point[b] - point[a]) * 100),
                       "ci_low_pp": float(ci[0]), "ci_high_pp": float(ci[1]),
                       "seed": SEED, "replicates": REPLICATES,
                       "unit": "F1 percentage points; comparison minus reference", "sampling": note})


def main():
    for folder in ["tables", "logs", "reports", "figures", "cache", "tmp"]:
        (OUT / folder).mkdir(exist_ok=True, parents=True)
    all_samples = jload("samples.json")
    samples = {s["sample_id"]: s for s in all_samples if s["split"] == "confirmation"}
    assert len(samples) == 150
    answer_lines = jsonl("server/confirmation-results.jsonl")
    assert len(answer_lines) == 1050
    actual = {}
    for line, r in jsonl("server/attempts.jsonl"):
        key = encoded(r["key"])
        assert key not in actual, "duplicate successful key"
        assert r["status"] == "success" and r["attempt"] == 1
        actual[key] = (line, r)
    assert len(actual) == 2430
    rules = jload("server/natural-rule.json")
    old_rule = jload("inputs/output/runs/main80-20261114T041503/rule.json")
    assert rules["fixed"] == [1.0, .9] and rules["by_N"]["2"] == [.9, 1.0]
    eos = jload("code/001/models/Llama-3.1-8B-Instruct/generation_config.json")["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    indexed, score_recheck, logical, physical_c = {}, [], [], {}
    for line, r in answer_lines:
        sid, method = r["sample_id"], r["method"]
        assert sid in samples and method in METHODS and (sid, method) not in indexed
        s = samples[sid]
        assert r["dataset"] == s["dataset"] and r["split"] == "confirmation"
        assert r["L"] == s["L"] and r["natural_N"] == s["natural_N"]
        assert r["reference_answers"] == s["answers"]
        assert s["natural_N"] == math.ceil(s["L"] / 7500)
        assert r["answer"].strip() and r["output_token_ids"]
        if method == "ape_natural_fixed":
            assert [r["T"], r["S"]] == rules["fixed"]
        if method == "ape_natural_dynamic":
            assert [r["T"], r["S"]] == rules["by_N"][str(s["natural_N"]) ]
        pkey = encoded(r["source_attempt_key"])
        actual_line, physical = actual[pkey]
        assert physical["sample_id"] == sid and physical["key"]["stage"] == "independent_confirmation"
        assert r["answer"] == physical["answer"]
        assert r["output_token_ids"] == physical["output_token_ids"]
        assert r["reference_answers"] == physical["reference_answers"]
        assert r["score"] == physical["score"] and r["total_seconds"] == physical["total_seconds"]
        rechecked = qa_f1(r["answer"], r["reference_answers"])
        assert abs(rechecked - r["score"]) < 1e-12
        indexed[sid, method] = (line, r, actual_line, physical)
        physical_c[pkey] = (actual_line, physical)
        score_recheck.append({"sample_id": sid, "method": method, "saved_F1": r["score"],
                              "recalculated_F1": rechecked, "absolute_difference": abs(rechecked-r["score"]),
                              "source_file": "server/confirmation-results.jsonl", "source_line": line,
                              "unit": "F1 0-1; maximum over frozen references"})
        logical.append({"sample_id": sid, "dataset": s["dataset"], "support_component": s["support_component"],
                        "method": method, "natural_N": s["natural_N"], "L": s["L"], "T": r["T"], "S": r["S"],
                        "answer": r["answer"], "reference_answers": encoded(s["answers"]),
                        "score_0_1": r["score"], "output_tokens": len(r["output_token_ids"]),
                        "cap_hit": len(r["output_token_ids"]) >= s["max_new_tokens"],
                        "eos_last": r["output_token_ids"][-1] in eos,
                        "physical_generation_shared_by": encoded(r["physical_generation_shared_by"]),
                        "parameter_reuse": r["parameter_reuse"],
                        "physical_generation_key": pkey, "logical_source_line": line,
                        "physical_source_line": actual_line,
                        "total_seconds": r["total_seconds"], "ttft_seconds": physical["ttft_seconds"],
                        "cache_build_seconds": physical["cache_build_seconds"],
                        "dynamic_selection_seconds": physical["dynamic_selection_seconds"],
                        "peak_allocated_bytes": physical["peak_allocated_bytes"],
                        "peak_reserved_bytes": physical["peak_reserved_bytes"],
                        "timing_condition": physical.get("timing_condition", ""),
                        "source_file": "server/confirmation-results.jsonl;server/attempts.jsonl",
                        "note": "shared logical methods use the same physical observation; no independent rerun"})
    assert len(physical_c) == 897 and 1050-len(physical_c) == 153
    for sid in samples:
        assert all((sid, m) in indexed for m in METHODS)

    quality, paired, wtl, membership, component_rows, diagnostics = [], [], [], [], [], []
    arrays, ds_boot, ds_points, ds_draws, q_context = {}, {}, {}, {}, None
    rng = np.random.default_rng(SEED)
    for ds, expected in zip(DATASETS, [15, 46, 7]):
        ids = sorted(sid for sid, s in samples.items() if s["dataset"] == ds)
        assert len(ids) == 50
        components = sorted({samples[sid]["support_component"] for sid in ids})
        assert len(components) == expected, (ds, len(components))
        matrix = np.array([[indexed[sid, m][1]["score"] for m in METHODS] for sid in ids], dtype=float)
        groups = np.array([components.index(samples[sid]["support_component"]) for sid in ids])
        counts = np.bincount(groups, minlength=len(components))
        sums = np.zeros((len(components), len(METHODS)))
        np.add.at(sums, groups, matrix)
        draws = rng.integers(0, len(components), size=(REPLICATES, len(components)))
        denominator = counts[draws].sum(axis=1)
        boot = sums[draws].sum(axis=1) / denominator[:, None]
        point = matrix.mean(axis=0)
        arrays[ds] = (ids, matrix, groups)
        ds_points[ds], ds_boot[ds], ds_draws[ds] = point, boot, draws
        diagnostics.append({"scope": ds, "stream": "primary default_rng(seed), dataset order Q/H/M",
                            "components": len(components), "proposed": REPLICATES,
                            "rejected_missing_N": 0, "effective": REPLICATES,
                            "sampled_question_count_min": int(denominator.min()),
                            "sampled_question_count_max": int(denominator.max()),
                            "seed": SEED, "note": "primary draws never rejected for N-strata coverage"})
        summarize_quality(ds, point, boot, 50, len(components), quality, paired,
                          "within-dataset support_component sampling; all questions/methods retained; divide by sampled questions")
        for sid in ids:
            s = samples[sid]
            f, d = indexed[sid, METHODS[FIXED]][1], indexed[sid, METHODS[DYNAMIC]][1]
            membership.append({"sample_id": sid, "dataset": ds, "support_component": s["support_component"],
                               "support_documents": encoded(s["support_documents"]), "L": s["L"],
                               "natural_N": s["natural_N"], "source_file": s["source_file"],
                               "source_container_id": s["source_container_id"], "public_id": s["public_id"],
                               "fixed_F1": f["score"], "dynamic_F1": d["score"],
                               "delta_F1": d["score"]-f["score"],
                               "same_physical_generation": encoded(f["source_attempt_key"]) == encoded(d["source_attempt_key"]),
                               "same_answer": f["answer"] == d["answer"],
                               "same_tokens": f["output_token_ids"] == d["output_token_ids"],
                               "unit": "one confirmation question; all seven methods paired"})
        for c, component in enumerate(components):
            mask = groups == c
            sub = matrix[mask]
            delta = sub[:, DYNAMIC] - sub[:, FIXED]
            component_rows.append({"dataset": ds, "support_component": component,
                                   "n_questions": int(mask.sum()), "dataset_fraction": float(mask.mean()),
                                   "N_values": encoded(sorted({samples[ids[i]]["natural_N"] for i in np.flatnonzero(mask)})),
                                   "dynamic_minus_fixed_sum_0_1": float(delta.sum()),
                                   "dynamic_minus_fixed_mean_pp": float(delta.mean()*100),
                                   "wins": int((delta>1e-12).sum()), "ties": int((np.abs(delta)<=1e-12).sum()),
                                   "losses": int((delta < -1e-12).sum()),
                                   "sample_ids": encoded([ids[i] for i in np.flatnonzero(mask)])})
        if ds == "qasper":
            q_context = (ids, matrix, groups, counts, sums, components)

    primary_point = np.mean([ds_points[d] for d in DATASETS], axis=0)
    primary_boot = np.mean([ds_boot[d] for d in DATASETS], axis=0)
    summarize_quality("macro_equal_dataset", primary_point, primary_boot, 150, 68, quality, paired,
                      "independent within-dataset component draws; question-weighted dataset means then equal mean of three datasets")
    ids, qmat, qgroups, qcounts, qsums, qcomponents = q_context
    n_values = np.array([samples[sid]["natural_N"] for sid in ids])
    strat_counts = np.zeros((15, 3), dtype=int)
    strat_sums = np.zeros((15, 3, len(METHODS)))
    for i in range(len(ids)):
        strat_counts[qgroups[i], n_values[i]-1] += 1
        strat_sums[qgroups[i], n_values[i]-1] += qmat[i]
    # Separate conditional stream for N reporting; main bootstrap above is unchanged.
    strng = np.random.default_rng(np.random.SeedSequence([SEED, 1]))
    accepted, proposed, rejected = [], 0, 0
    while sum(len(x) for x in accepted) < REPLICATES:
        need = REPLICATES - sum(len(x) for x in accepted)
        draws = strng.integers(0, 15, size=(need, 15))
        denominators = strat_counts[draws].sum(axis=1)
        valid = (denominators > 0).all(axis=1)
        proposed += need
        rejected += int((~valid).sum())
        if valid.any():
            accepted.append(strat_sums[draws[valid]].sum(axis=1) / denominators[valid, :, None])
    strat_boot = np.concatenate(accepted, axis=0)
    assert len(strat_boot) == REPLICATES
    for n in [1, 2, 3]:
        mask = n_values == n
        point = qmat[mask].mean(axis=0)
        n_comp = len(set(qgroups[mask]))
        summarize_quality(f"qasper_N{n}", point, strat_boot[:, n-1, :], int(mask.sum()), n_comp,
                          quality, paired, "exploratory; Qasper15 components jointly drawn; reject if any N missing; separate stream")
    equal_n_point = np.mean([qmat[n_values == n].mean(axis=0) for n in [1, 2, 3]], axis=0)
    summarize_quality("qasper_equal_N_supplement", equal_n_point, strat_boot.mean(axis=1), 50, 15,
                      quality, paired, "supplement only; equal mean of N1/N2/N3; same conditional paired draws as N strata")
    diagnostics.append({"scope": "qasper_N_strata_and_equal_N", "stream": "SeedSequence([20261116,1])",
                        "components": 15, "proposed": proposed, "rejected_missing_N": rejected,
                        "effective": REPLICATES, "sampled_question_count_min": "",
                        "sampled_question_count_max": "", "seed": SEED,
                        "note": "condition on every N present; never replaces the primary overall draw distribution"})

    for scope in DATASETS + ["macro_equal_dataset", "qasper_N1", "qasper_N2", "qasper_N3"]:
        if scope == "macro_equal_dataset":
            selected = sorted(samples)
        elif scope.startswith("qasper_N"):
            selected = [sid for sid in samples if samples[sid]["dataset"] == "qasper" and samples[sid]["natural_N"] == int(scope[-1])]
        else:
            selected = [sid for sid in samples if samples[sid]["dataset"] == scope]
        for a, b in itertools.combinations(METHODS, 2):
            delta = np.array([indexed[sid,b][1]["score"]-indexed[sid,a][1]["score"] for sid in selected])
            wtl.append({"scope": scope, "reference_method": a, "comparison_method": b,
                        "n_questions": len(selected), "wins": int((delta > 1e-12).sum()),
                        "ties": int((np.abs(delta) <= 1e-12).sum()), "losses": int((delta < -1e-12).sum()),
                        "unit": "paired question counts; tolerance 1e-12"})
    for row in paired:
        match = next((r for r in wtl if all(r[k] == row[k] for k in ["scope", "reference_method", "comparison_method"])), None)
        row.update({"wins": match["wins"] if match else "", "ties": match["ties"] if match else "", "losses": match["losses"] if match else ""})

    crosscheck = []
    with (PKG / "review/confirmation-means.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            m = METHODS.index(r["method"])
            for scope in DATASETS + ["equal_dataset_macro"]:
                ours = primary_point[m] if scope == "equal_dataset_macro" else ds_points[scope][m]
                difference = abs(ours-float(r[scope]))
                assert difference < 1e-12
                crosscheck.append({"method": r["method"], "scope": scope,
                                   "accepted_mean_0_1": r[scope], "current_mean_0_1": ours,
                                   "absolute_difference": difference, "unit": "F1 0-1"})

    efficiency, limits = [], []
    for method in METHODS:
        rr = [r for r in logical if r["method"] == method]
        assert len(rr) == 150
        for field in ["ttft_seconds", "total_seconds", "cache_build_seconds", "dynamic_selection_seconds", "output_tokens", "peak_allocated_bytes", "peak_reserved_bytes"]:
            values = np.array([r[field] for r in rr], dtype=float)
            assert np.isfinite(values).all() and (values >= 0).all()
            efficiency.append({"method": method, "metric": field, "n_logical_questions": 150,
                               "n_unique_physical_sources_within_method": len({r["physical_generation_key"] for r in rr}),
                               "mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.quantile(values,.9)),
                               "unit": "bytes" if field.endswith("bytes") else "tokens" if field=="output_tokens" else "seconds",
                               "ci_low": "", "ci_high": "", "cache_condition": "cold; model loading excluded",
                               "note": "descriptive only; shared logical labels are one timing observation, not independent repeat runs"})
        limits.append({"method": method, "n_questions": 150,
                       "cap_hits": sum(r["cap_hit"] for r in rr), "cap_fraction": sum(r["cap_hit"] for r in rr)/150,
                       "eos_last": sum(r["eos_last"] for r in rr),
                       "eos_at_cap": sum(r["eos_last"] and r["cap_hit"] for r in rr),
                       "note": "logical-method descriptive counts; common physical sources counted once in physical ledger"})
    case_rows = []
    sorted_members = sorted(membership, key=lambda r:r["sample_id"])
    groups = {"improvement": [r for r in sorted_members if r["delta_F1"]>1e-12],
              "degradation": [r for r in sorted_members if r["delta_F1"] < -1e-12],
              "unchanged": [r for r in sorted_members if abs(r["delta_F1"]) <= 1e-12]}
    for kind, rr in groups.items():
        if not rr:
            continue
        chosen = sorted(rr, key=lambda r: (-r["delta_F1"] if kind=="improvement" else r["delta_F1"] if kind=="degradation" else 0, r["sample_id"]))[0]
        sid = chosen["sample_id"]
        fline, fixed, fphysical, _ = indexed[sid, METHODS[FIXED]]
        dline, dynamic, dphysical, _ = indexed[sid, METHODS[DYNAMIC]]
        case_rows.append({"case_type": kind, **chosen, "question": samples[sid]["question"],
                          "fixed_T": fixed["T"], "fixed_S": fixed["S"], "dynamic_T": dynamic["T"], "dynamic_S": dynamic["S"],
                          "fixed_answer": fixed["answer"], "dynamic_answer": dynamic["answer"],
                          "reference_answers": encoded(samples[sid]["answers"]),
                          "fixed_confirmation_line": fline, "dynamic_confirmation_line": dline,
                          "fixed_physical_line": fphysical, "dynamic_physical_line": dphysical,
                          "selection": "max improvement/min degradation/zero delta, ties sample_id ascending"})

    csvout("c_membership.csv", membership)
    csvout("c_logical_answers.csv", logical)
    csvout("c_score_recheck.csv", score_recheck)
    csvout("c_quality.csv", quality)
    csvout("c_paired.csv", paired)
    csvout("c_win_tie_loss.csv", wtl)
    csvout("c_support_components.csv", component_rows)
    csvout("c_bootstrap_diagnostics.csv", diagnostics)
    csvout("c_accepted_mean_crosscheck.csv", crosscheck)
    csvout("c_efficiency_descriptive.csv", efficiency)
    csvout("c_generation_limits.csv", limits)
    csvout("c_cases.csv", case_rows)
    physical_rows = [{"source_line": line, "sample_id": r["sample_id"], "physical_generation_key": key,
                      "logical_labels": encoded(r["logical_labels"]), "output_tokens": r["output_tokens"],
                      "ttft_seconds": r["ttft_seconds"], "total_seconds": r["total_seconds"],
                      "source_file": "server/attempts.jsonl", "unit": "one actual new generation"}
                     for key,(line,r) in physical_c.items()]
    csvout("c_physical_generations.csv", physical_rows)
    primary = next(r for r in paired if r["scope"] == "macro_equal_dataset" and r["reference_method"] == METHODS[FIXED] and r["comparison_method"] == METHODS[DYNAMIC])
    assert abs(primary["delta_pp"]-.14222164357828228) < 1e-10
    assert [primary[k] for k in ["wins", "ties", "losses"]] == [3,146,1]
    result = {"stage": "C3_INDEPENDENT_CONFIRMATION", "n_questions": 150,
              "n_logical_answers": 1050, "n_physical_generations": 897, "extra_same_parameter_labels": 153,
              "support_components": dict(zip(DATASETS,[15,46,7])),
              "primary_comparison": primary, "method_means": [r for r in quality if r["scope"]=="macro_equal_dataset"],
              "qasper_N_paired": [r for r in paired if r["scope"].startswith("qasper_N") and r["reference_method"]==METHODS[FIXED] and r["comparison_method"]==METHODS[DYNAMIC]],
              "bootstrap_diagnostics": diagnostics, "rule_parameters_reselected": False,
              "new_answers_generated_by_PC": 0, "research_run_status": "SUBMITTED",
              "package_delivery_status": "ACCEPTED", "MAIN80_overall_status": "REVISION_REQUIRED"}
    (OUT/"c_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    environment = {"analysis_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
                   "server_delivery_commit": "ab816f5256f83892c83ea71bcf63a0b618e2f18d",
                   "branch": subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip(),
                   "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "numpy": np.__version__, "seed": SEED, "replicates": REPLICATES,
                   "primary_rng": "default_rng(20261116), sequential dataset draws Qasper,HotpotQA,MuSiQue",
                   "strata_rng": "default_rng(SeedSequence([20261116,1])); separate conditional Qasper stream",
                   "confidence_interval": "2.5%/97.5% percentile of paired support-component replicate values",
                   "new_dependencies": [], "historical_package_scripts_executed": False,
                   "source_scoring": "independent normalize/token-overlap F1, max over existing frozen references",
                   "package_root": "restored/followup2604/FOLLOWUP2604_ACCEPTED"}
    (OUT/"environment.json").write_text(json.dumps(environment,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"logs/c_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"primary":primary,"qasper_N":result["qasper_N_paired"],"stratum_proposals":proposed,"stratum_rejected":rejected},ensure_ascii=False))


if __name__ == "__main__":
    main()
