"""FOLLOWUP2604 B: frozen-rule length transfer, paired support-group bootstrap.

Only reads accepted answers and input metadata. No model or historical script is run.
"""
import csv
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parent
PACKAGE = REPO / "restored/followup2604/FOLLOWUP2604_ACCEPTED"
for variable in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "TEMP", "TMP"]:
    directory = OUT / ("cache/matplotlib" if variable == "MPLCONFIGDIR" else "tmp")
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261116
REPS = 5000
DATASETS = ["qasper", "hotpotqa", "musique"]
METHODS = ["ape_corrected_fixed", "ape_corrected_dynamic"]
CELLS = [(L, N) for L in [6144, 12288] for N in [4, 8, 16]]


def load(rel):
    return json.loads((PACKAGE / rel).read_text(encoding="utf-8"))


def lines(rel):
    return [(i, json.loads(s)) for i, s in enumerate((PACKAGE / rel).read_text(encoding="utf-8").splitlines(), 1) if s.strip()]


def csvread(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csvwrite(name, rows):
    with (OUT / "tables" / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def norm(s):
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s.lower().translate(str.maketrans("", "", string.punctuation))).split())


def f1(answer, refs):
    prediction = norm(answer).split()
    values = []
    for ref in refs:
        gold = norm(ref).split()
        overlap = sum((Counter(prediction) & Counter(gold)).values())
        values.append(2 * overlap / (len(prediction) + len(gold)) if overlap else 0.0)
    return max(values)


def savefig(name):
    plt.savefig(OUT / "figures" / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.savefig(OUT / "figures" / (name + ".svg"), bbox_inches="tight")
    plt.close()


def main():
    for name in ["tables", "reports", "figures", "logs"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    load("PORTABLE-PATHS.json")
    accepted = load("FINAL-REVIEW.json")
    rule = load("server/corrected-rule.json")
    plan = load("server/corrected-length-plan.json")
    sourced = lines("server/corrected-length-results.jsonl")
    assert len(sourced) == 120 and len(plan["jobs"]) == 103 and len(plan["reuse"]) == 17
    ids = {r["sample_id"] for _, r in sourced}
    assert len(ids) == 10
    new_inputs = {(r["sample_id"], int(r["L"])): r for r in load("inputs/datasets/frozen-main80-corrected-v1/development.json") if r["sample_id"] in ids and r["L"] in [6144, 12288]}
    old_inputs = {(r["sample_id"], int(r["L"])): r for r in load("inputs/datasets/frozen-main80/development.json") if r["sample_id"] in ids and r["L"] in [6144, 12288]}
    assert len(new_inputs) == 20
    records, lookup = [], {}
    score_errors = []
    for source_line, raw in sourced:
        sid, L, N = raw["sample_id"], int(raw["L"]), int(raw["N"])
        method = raw["key"]["method"]
        assert method in METHODS and raw["status"] == "success"
        key = (sid, L, N, method)
        assert key not in lookup
        data = new_inputs[(sid, L)]
        assert raw["reference_answers"] == data["answers"]
        group = json.dumps(data["group"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert len(data["document_tokens"]) == L
        expected_param = rule["fixed"] if method == METHODS[0] else rule["cells"][f"{8192 if L == 6144 else 16384}/{N}"]
        assert [raw["T"], raw["S"]] == expected_param
        score_errors.append(abs(f1(raw["answer"], raw["reference_answers"]) - float(raw["score"])))
        r = {"sample_id": sid, "dataset": raw["dataset"], "support_group": group,
             "support_documents": json.dumps(data["support_documents"], ensure_ascii=False),
             "L": L, "N": N, "method": method, "T": raw["T"], "S": raw["S"],
             "score_0_1": float(raw["score"]), "answer": raw["answer"],
             "reference_answers": json.dumps(raw["reference_answers"], ensure_ascii=False),
             "answer_origin": raw["answer_origin"], "output_tokens": len(raw["output_token_ids"]),
             "source_file": "server/corrected-length-results.jsonl", "source_line": source_line,
             "historical_source_line": raw.get("source_line", ""),
             "ttft_seconds": raw.get("ttft_seconds", ""), "total_seconds": raw.get("total_seconds", ""),
             "cache_build_seconds": raw.get("cache_build_seconds", ""),
             "cache_get_seconds": raw.get("cache_get_seconds", ""),
             "selection_seconds": raw.get("dynamic_selection_seconds", ""),
             "peak_allocated_GiB": raw["peak_allocated_bytes"] / 2**30 if "peak_allocated_bytes" in raw else "",
             "peak_reserved_GiB": raw["peak_reserved_bytes"] / 2**30 if "peak_reserved_bytes" in raw else "",
             "timing_scope": "one new cold generation" if raw["answer_origin"] == "new_followup_generation" else "no new timing observation",
             "unit": "F1 0-1; tokens include EOS; seconds; memory GiB"}
        lookup[key] = (r, raw)
        records.append(r)
    assert max(score_errors) < 1e-12
    assert Counter(r["answer_origin"] for r in records) == {"new_followup_generation": 103, "historical_exact_reuse": 17}
    csvwrite("b_records.csv", sorted(records, key=lambda r: (r["sample_id"], r["L"], r["N"], r["method"])))

    membership, pairs = [], []
    for sid in sorted(ids):
        data = new_inputs[(sid, 6144)]
        group = json.dumps(data["group"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert data["group"] == new_inputs[(sid, 12288)]["group"]
        membership.append({"sample_id": sid, "dataset": data["dataset"], "support_group": group,
                           "n_lengths": 2, "n_block_conditions": 3, "n_methods": 2,
                           "unit": "one question; all L/N/methods jointly retained"})
        for L, N in CELLS:
            fixed, fr = lookup[(sid, L, N, METHODS[0])]
            dynamic, dr = lookup[(sid, L, N, METHODS[1])]
            same_param = (fixed["T"], fixed["S"]) == (dynamic["T"], dynamic["S"])
            same_tokens = fr["output_token_ids"] == dr["output_token_ids"]
            assert not same_param or same_tokens
            pairs.append({"sample_id": sid, "dataset": data["dataset"], "support_group": group,
                          "L": L, "N": N, "fixed_T": fixed["T"], "fixed_S": fixed["S"],
                          "dynamic_T": dynamic["T"], "dynamic_S": dynamic["S"],
                          "fixed_F1": fixed["score_0_1"], "dynamic_F1": dynamic["score_0_1"],
                          "delta_pp": (dynamic["score_0_1"]-fixed["score_0_1"]) * 100,
                          "same_parameters": same_param, "same_output_tokens": same_tokens,
                          "fixed_origin": fixed["answer_origin"], "dynamic_origin": dynamic["answer_origin"],
                          "fixed_answer": fixed["answer"], "dynamic_answer": dynamic["answer"],
                          "reference_answers": fixed["reference_answers"],
                          "fixed_source_line": fixed["source_line"], "dynamic_source_line": dynamic["source_line"],
                          "unit": "F1 0-1; delta in percentage points; one question/L/N pair"})
    csvwrite("b_membership.csv", membership)
    csvwrite("b_paired.csv", pairs)

    arrays, counts, dsids = {}, {}, {}
    diagnostics = []
    for ds_index, ds in enumerate(DATASETS):
        dsids[ds] = sorted(r["sample_id"] for r in membership if r["dataset"] == ds)
        groups = [next(r["support_group"] for r in membership if r["sample_id"] == sid) for sid in dsids[ds]]
        unique_groups = sorted(set(groups))
        assignment = np.array([[int(g == group) for g in groups] for group in unique_groups])
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, ds_index]))
        counts[ds] = rng.multinomial(len(unique_groups), np.full(len(unique_groups), 1 / len(unique_groups)), size=REPS) @ assignment
        arrays[ds] = np.array([[[lookup[(sid, L, N, method)][0]["score_0_1"] for method in METHODS] for L, N in CELLS] for sid in dsids[ds]])
        diagnostics.append({"dataset": ds, "n_questions": len(groups), "n_support_groups": len(unique_groups),
                            "replicates_valid": REPS, "proposed": REPS, "empty_strata_rejections": 0,
                            "min_sampled_questions": int(counts[ds].sum(axis=1).min()),
                            "max_sampled_questions": int(counts[ds].sum(axis=1).max()),
                            "rng": f"SeedSequence([{SEED},2,{ds_index}])",
                            "unit": "frozen B support group; cluster size retained; all six conditions paired"})
    assert [x["n_support_groups"] for x in diagnostics] == [3, 3, 3]
    csvwrite("b_bootstrap_diagnostics.csv", diagnostics)
    scopes = [("ALL", list(range(6))), ("L6144", [0, 1, 2]), ("L12288", [3, 4, 5])]
    scopes += [(f"N{N}", [i for i, cell in enumerate(CELLS) if cell[1] == N]) for N in [4, 8, 16]]
    scopes += [(f"{L}/{N}", [i]) for i, (L, N) in enumerate(CELLS)]
    quality = []
    for scope, selected_cells in scopes:
        for dataset in ["macro_equal_dataset", *DATASETS]:
            included = DATASETS if dataset == "macro_equal_dataset" else [dataset]
            means, reps = [], []
            for ds in included:
                qmean = arrays[ds][:, selected_cells, :].mean(axis=1)
                means.append(qmean.mean(axis=0))
                reps.append(counts[ds] @ qmean / counts[ds].sum(axis=1)[:, None])
            point = np.mean(means, axis=0) * 100
            boot = np.mean(reps, axis=0) * 100
            delta = boot[:, 1] - boot[:, 0]
            chosen = [r for r in pairs if r["dataset"] in included and CELLS.index((r["L"], r["N"])) in selected_cells]
            differences = np.array([r["delta_pp"] for r in chosen])
            quality.append({"scope": scope, "dataset": dataset,
                            "n_questions": sum(len(dsids[ds]) for ds in included), "n_input_pairs": len(chosen),
                            "n_support_groups": sum(next(x["n_support_groups"] for x in diagnostics if x["dataset"] == ds) for ds in included),
                            "fixed_F1_percent": float(point[0]), "dynamic_F1_percent": float(point[1]),
                            "delta_pp": float(point[1] - point[0]),
                            "delta_ci_low_pp": float(np.quantile(delta, .025)), "delta_ci_high_pp": float(np.quantile(delta, .975)),
                            "fixed_ci_low_percent": float(np.quantile(boot[:, 0], .025)), "fixed_ci_high_percent": float(np.quantile(boot[:, 0], .975)),
                            "dynamic_ci_low_percent": float(np.quantile(boot[:, 1], .025)), "dynamic_ci_high_percent": float(np.quantile(boot[:, 1], .975)),
                            "wins": int((differences > 1e-10).sum()), "ties": int((abs(differences) <= 1e-10).sum()), "losses": int((differences < -1e-10).sum()),
                            "seed": SEED, "replicates": REPS,
                            "aggregation": "question mean within dataset and selected cells; equal included datasets",
                            "unit": "F1 x100; difference percentage points; W/T/L counts input pairs"})
    csvwrite("b_quality.csv", quality)
    main_result = next(r for r in quality if r["scope"] == "ALL" and r["dataset"] == "macro_equal_dataset")

    # Exact historical comparisons use matched IDs and direct token/prompt equality.
    historical_path = REPO / "pc-analysis-output/stages/04_length_check/tables/paired_records.csv"
    old_pairs = csvread(historical_path)
    old_lookup = {(r["sample_id"], int(r["L"]), int(r["N"])): r for r in old_pairs}
    old_ids = {r["sample_id"] for r in old_pairs}
    historical = []
    input_checks = []
    for (sid, L), data in sorted(new_inputs.items()):
        old = old_inputs.get((sid, L))
        equal = {field: old is not None and old[field] == data[field] for field in ["document_tokens", "prefix", "suffix", "max_new_tokens", "answers"]}
        input_checks.append({"sample_id": sid, "L": L, "has_old_frozen_input": old is not None,
                             "in_old_length_experiment": sid in old_ids,
                             **{field + "_same": v for field, v in equal.items()},
                             "exact_generation_input_same": all(equal.values()),
                             "unit": "one question/length; direct content comparison; no hash"})
    csvwrite("b_historical_input_checks.csv", input_checks)
    for pair in pairs:
        sid, L, N = pair["sample_id"], pair["L"], pair["N"]
        old = old_lookup.get((sid, L, N))
        if old is None:
            continue
        same_input = next(r["exact_generation_input_same"] for r in input_checks if r["sample_id"] == sid and r["L"] == L)
        for label in ["fixed", "dynamic"]:
            current = lookup[(sid, L, N, METHODS[0 if label == "fixed" else 1])][0]
            same_param = float(old[label + "_T"]) == current["T"] and float(old[label + "_S"]) == current["S"]
            historical.append({"sample_id": sid, "dataset": pair["dataset"], "L": L, "N": N, "method_role": label,
                               "old_T": old[label + "_T"], "old_S": old[label + "_S"], "new_T": current["T"], "new_S": current["S"],
                               "same_input": same_input, "same_parameters": same_param,
                               "comparison_condition": f"input_{'same' if same_input else 'changed'}__parameters_{'same' if same_param else 'changed'}",
                               "old_F1": old[label + "_f1_0_1"], "new_F1": current["score_0_1"],
                               "delta_pp": (current["score_0_1"] - float(old[label + "_f1_0_1"])) * 100,
                               "old_answer": old[label + "_answer"], "new_answer": current["answer"],
                               "old_source_line": old[label + "_line"], "new_source_line": current["source_line"],
                               "new_answer_origin": current["answer_origin"], "unit": "matched historical question/L/N/method role; input and parameter changes explicit"})
    csvwrite("b_historical_pairs.csv", historical)
    history_summary = []
    for condition in sorted({r["comparison_condition"] for r in historical}):
        for label in ["fixed", "dynamic"]:
            rows = [r for r in historical if r["comparison_condition"] == condition and r["method_role"] == label]
            if rows:
                history_summary.append({"condition": condition, "method_role": label, "n_input_comparisons": len(rows),
                                        "n_questions": len({r["sample_id"] for r in rows}),
                                        "old_F1_percent": float(np.mean([float(r["old_F1"]) for r in rows]) * 100),
                                        "new_F1_percent": float(np.mean([r["new_F1"] for r in rows]) * 100),
                                        "delta_pp": float(np.mean([r["delta_pp"] for r in rows])),
                                        "unit": "descriptive mixed-request mean; different strata are distinct conditions"})
    csvwrite("b_historical_conditions.csv", history_summary)
    csvwrite("b_old_new_membership.csv", [{"sample_id": sid, "in_old": sid in old_ids, "in_corrected": sid in ids,
                                          "scope": "common" if sid in ids & old_ids else "new_only" if sid in ids else "old_only",
                                          "unit": "one question ID"} for sid in sorted(ids | old_ids)])

    efficiency = []
    for method in METHODS:
        selected = [r for r in records if r["method"] == method and r["answer_origin"] == "new_followup_generation"]
        for metric in ["ttft_seconds", "total_seconds", "cache_build_seconds", "cache_get_seconds", "selection_seconds", "output_tokens", "peak_allocated_GiB", "peak_reserved_GiB"]:
            values = np.array([r[metric] for r in selected], dtype=float)
            efficiency.append({"method": method, "metric": metric, "n_new_requests": len(values),
                               "mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
                               "unit": "GiB" if "GiB" in metric else "tokens" if metric == "output_tokens" else "seconds",
                               "scope": "new cold generations only; fixed/dynamic counts differ; descriptive"})
    csvwrite("b_efficiency_new_generation.csv", efficiency)
    timing_pairs = []
    for p in pairs:
        fr = lookup[(p["sample_id"], p["L"], p["N"], METHODS[0])][0]
        dr = lookup[(p["sample_id"], p["L"], p["N"], METHODS[1])][0]
        if fr["answer_origin"] != "new_followup_generation" or dr["answer_origin"] != "new_followup_generation":
            continue
        timing_pairs.append({"sample_id": p["sample_id"], "dataset": p["dataset"], "L": p["L"], "N": p["N"],
                             "fixed_total_seconds": fr["total_seconds"], "dynamic_total_seconds": dr["total_seconds"],
                             "delta_total_seconds": dr["total_seconds"] - fr["total_seconds"],
                             "fixed_ttft_seconds": fr["ttft_seconds"], "dynamic_ttft_seconds": dr["ttft_seconds"],
                             "paired_total_speedup": fr["total_seconds"] / dr["total_seconds"],
                             "fixed_output_tokens": fr["output_tokens"], "dynamic_output_tokens": dr["output_tokens"],
                             "unit": "same input; both newly generated cold requests; seconds; ratio fixed/dynamic"})
    csvwrite("b_efficiency_new_pairs.csv", timing_pairs)
    case_rows = []
    for category, candidates, sorting in [("improvement", [p for p in pairs if p["delta_pp"] > 1e-10], lambda p: (-p["delta_pp"], p["sample_id"], p["L"], p["N"])),
                                          ("degradation", [p for p in pairs if p["delta_pp"] < -1e-10], lambda p: (p["delta_pp"], p["sample_id"], p["L"], p["N"])),
                                          ("tie", [p for p in pairs if abs(p["delta_pp"]) <= 1e-10], lambda p: (p["sample_id"], p["L"], p["N"]))]:
        if candidates:
            chosen = sorted(candidates, key=sorting)[0]
            case_rows.append({"case_type": category, **chosen,
                              "question": new_inputs[(chosen["sample_id"], chosen["L"])]["question"],
                              "selection_rule": "largest signed F1 change; ties by sample ID,L,N; zero case lexicographic"})
    csvwrite("b_selected_cases.csv", case_rows)

    summary = {"phase": "B_LENGTH_TRANSFER", "research_status": "SUBMITTED",
               "logical_results": 120, "new_generations": 103, "historical_reuse": 17,
               "new_generation_timing_requests": 103, "new_generation_timing_pairs": len(timing_pairs),
               "unique_questions": 10, "n_pairs": 60, "group_counts": {r["dataset"]: r["n_support_groups"] for r in diagnostics},
               "new_fixed_parameters": rule["fixed"], "main_equal_dataset_result": main_result,
               "descriptive_mixed_request_mean": {"fixed_F1_percent": float(np.mean([p["fixed_F1"] for p in pairs]) * 100),
                                                   "dynamic_F1_percent": float(np.mean([p["dynamic_F1"] for p in pairs]) * 100),
                                                   "delta_pp": float(np.mean([p["delta_pp"] for p in pairs]))},
               "same_parameter_pairs": sum(p["same_parameters"] for p in pairs),
               "identical_output_pairs": sum(p["same_output_tokens"] for p in pairs),
               "historical_comparison": {"old_questions": len(old_ids), "new_questions": len(ids), "common_questions": len(ids & old_ids),
                                           "new_only": sorted(ids-old_ids), "old_only": sorted(old_ids-ids),
                                           "matched_method_inputs": len(historical), "strata": history_summary},
               "score_recalculation_max_abs_error": max(score_errors),
               "seed": SEED, "replicates": REPS, "accepted_delivery": accepted["computation_and_data_delivery"],
               "model_run": False, "parameters_reselected": False}
    (OUT / "b_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "logs/b_analysis_config.json").write_text(json.dumps({"seed": SEED, "replicates": REPS, "datasets": DATASETS,
            "python": sys.version, "python_executable": sys.executable, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "command": "python -s -B pc-analysis-followup-output/scripts/analyze_b_length.py",
            "cells": CELLS, "aggregation": main_result["aggregation"], "diagnostics": diagnostics,
            "new_timing_only": True, "historical_scripts_executed": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False})
    cell_rows = [next(r for r in quality if r["dataset"] == "macro_equal_dataset" and r["scope"] == f"{L}/{N}") for L, N in CELLS]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(6)
    for offset, role, color in [(-.19, "fixed", "#4c78a8"), (.19, "dynamic", "#f58518")]:
        y = np.array([r[role + "_F1_percent"] for r in cell_rows])
        lo = np.array([r[role + "_ci_low_percent"] for r in cell_rows])
        hi = np.array([r[role + "_ci_high_percent"] for r in cell_rows])
        ax.bar(x + offset, y, .36, label="修正版固定" if role == "fixed" else "修正版动态", color=color,
               yerr=np.stack([y-lo, hi-y]), capsize=3)
    ax.set_xticks(x, [f"{L//1024}K / N={N}" for L, N in CELLS]); ax.set_ylim(0, 100)
    ax.set_ylabel("F1 ×100；三个数据集等权")
    ax.set_title("B长度迁移：10题、60配对；冻结支撑组5000次配对Bootstrap")
    ax.legend(); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    savefig("b_length_quality")
    forest = [main_result] + [next(r for r in quality if r["dataset"] == "macro_equal_dataset" and r["scope"] == scope) for scope in ["L6144", "L12288"]] + cell_rows
    fig, ax = plt.subplots(figsize=(11, 5.8))
    y = np.arange(len(forest))[::-1]
    points = np.array([r["delta_pp"] for r in forest]); lows = np.array([r["delta_ci_low_pp"] for r in forest]); highs = np.array([r["delta_ci_high_pp"] for r in forest])
    ax.errorbar(points, y, xerr=np.stack([points-lows, highs-points]), fmt="o", color="#f58518", capsize=4)
    ax.axvline(0, color="grey", linestyle="--")
    labels = [r["scope"].replace("ALL", "总体").replace("L6144", "6K全部N").replace("L12288", "12K全部N") for r in forest]
    ax.set_yticks(y, labels); ax.set_xlabel("动态−固定 F1百分点；误差线为95%配对支撑组区间")
    ax.set_title("B长度迁移：改善与退化条件并列保留")
    for yy, r in zip(y, forest): ax.annotate(f"{r['delta_pp']:+.3f} [{r['delta_ci_low_pp']:.3f},{r['delta_ci_high_pp']:.3f}]", (r["delta_ci_high_pp"], yy), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    ax.margins(x=.35); ax.grid(axis="x", alpha=.2)
    savefig("b_length_delta")
    fig, ax = plt.subplots(figsize=(9, 4))
    values = [len(ids & old_ids), len(ids-old_ids), len(old_ids-ids)]
    ax.bar(["旧/新版共同题", "仅新版", "仅旧版"], values, color=["#4c78a8", "#54a24b", "#e45756"])
    for x, value in enumerate(values): ax.text(x, value+.1, str(value), ha="center")
    ax.set_ylim(0, 9); ax.set_ylabel("独立问题数"); ax.set_title("长度检查样本组成：两版各10题，共同7题")
    savefig("b_historical_membership")
    report = "# B阶段：修正版6K/12K长度迁移\n\n"
    report += f"修正版动态宏F1为{main_result['dynamic_F1_percent']:.4f}，固定为{main_result['fixed_F1_percent']:.4f}，差{main_result['delta_pp']:+.4f}个百分点，95%支撑组配对区间[{main_result['delta_ci_low_pp']:.4f},{main_result['delta_ci_high_pp']:.4f}]。60条件中5胜、53平、2负。区间跨0，当前样本中正负方向均在区间内。\n\n"
    report += "## 材料与分母\n\n120逻辑结果来自103次新生成和17次历史精确复用，独立问题10个，Qasper/HotpotQA/MuSiQue为4/3/3。每题两长度、三块数、两方法共同保留；各数据集冻结支撑组均为3个，Qasper其中一组含2题。没有使用自然确认集的组件或把60条件当60题。固定参数0.5/1.0，动态读取corrected-nine-cell-v1，不重新选参。\n\n"
    report += "主指标先对同数据集问题及所列长度/块数条件等权平均，再对三个数据集等权；5000次、seed20261116，在各数据集抽同数冻结组，保留组内题数和全部条件。同一组抽中多次按该次数加权，每次以抽到题数归一。所有题六条件齐全，空分层重抽为0。区间描述这9个来源组的构成变化，小组数下对新来源的推断仍需更多资料。\n\n"
    report += "| 范围 | 固定F1 | 动态F1 | 差/百分点 | 95%区间 | 胜/平/负 |\n|---|---:|---:|---:|---|---|\n"
    for r in [main_result] + cell_rows:
        report += f"| {r['scope']} | {r['fixed_F1_percent']:.3f} | {r['dynamic_F1_percent']:.3f} | {r['delta_pp']:+.3f} | [{r['delta_ci_low_pp']:.3f},{r['delta_ci_high_pp']:.3f}] | {r['wins']}/{r['ties']}/{r['losses']} |\n"
    report += "\n![长度质量](../figures/b_length_quality.png)\n\n![差与区间](../figures/b_length_delta.png)\n\n"
    mixed = summary["descriptive_mixed_request_mean"]
    report += f"验收表以60输入条件直接混合，固定{mixed['fixed_F1_percent']:.4f}、动态{mixed['dynamic_F1_percent']:.4f}、差{mixed['delta_pp']:+.4f}；与主表数据集等权的差异来自Q/H/M=4/3/3题数权重。两口径分别保存。六格及数据集完整值见b_quality.csv。\n\n"
    report += "## 与旧版的关系\n\n旧/新版长度检查各10题，只有7题共同，另3题发生变化；因此旧版−4.396个百分点与新版总体差并列作为各自范围结果。共同42个题长块条件按方法形成84项跨版连接，b_historical_pairs.csv分别标记实际token/prefix/suffix/上限/参考是否相同、T/S是否相同。输入变化、参数变化和样本变化分别保存，不将其混合命名为算法提升。历史复用17行只用于质量，timing空白，未复制旧计时为新请求。\n\n"
    report += "![两版问题组成](../figures/b_historical_membership.png)\n\n"
    report += "## 效率、案例与解释\n\n固定60、动态43条新生成具备本批计时；17条复用缺新计时。描述效率按这两个实际分母列均值、中位数和P90，另列43组两方法都新生成的相同输入配对。缺随机交错重复运行和阶段算子分解，因此本轮保留描述性时延结果。\n\n"
    for c in case_rows:
        report += f"- {c['case_type']}：`{c['sample_id']}`，L={c['L']}、N={c['N']}，差{c['delta_pp']:+.3f}个百分点；固定：{c['fixed_answer']}；动态：{c['dynamic_answer']}；参考：{c['reference_answers']}。\n"
    report += "\n本版6K/8出现改善，6K/4及12K/4有退化，N16两长度和12K/8同参持平。6K/8及6K整体区间在本次样本中为正，这些局部分层作为探索性结果保留；总体区间仍跨0。具体答错机制原因未确定，缺参数与证据位置的控制诊断。结果只评价已冻结修正版九格在这10个既有验证问题上的长度迁移；自然确认集使用另一套冻结自然规则，分别报告。研究状态SUBMITTED。\n\n"
    report += "## 复现与来源\n\n运行`python -s -B pc-analysis-followup-output/scripts/analyze_b_length.py`。输入为已恢复包内server/corrected-length-results.jsonl、corrected-length-plan.json、corrected-rule.json及inputs两版development.json；历史结果连接原pc-analysis-output/stages/04_length_check/tables/paired_records.csv。源行、原答案和参数见b_records.csv/b_paired.csv；抽样规则见logs/b_analysis_config.json与b_bootstrap_diagnostics.csv。仅执行自编CPU脚本，未执行包内历史脚本、模型、选参或摘要计算。\n"
    (OUT / "reports/B_LENGTH.md").write_text(report, encoding="utf-8")
    print(json.dumps({"B_main": main_result, "new_timing_pairs": len(timing_pairs), "historical_strata": history_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
