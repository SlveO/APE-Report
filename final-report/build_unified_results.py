"""从 PC 发布版 0b4f099 的结果表汇总结题报告所用比较。"""

import csv
import json
import math
import subprocess
from collections import defaultdict
from functools import cache
from io import StringIO
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent
REPO = OUTPUT.parent
MAIN = Path("pc-analysis-output/stages")
BASELINE = MAIN / "01_longbench_baseline/tables"
BASELINE_MEMBERSHIP = BASELINE / "baseline_per_sample.csv"
FOLLOWUP = Path("pc-analysis-followup-output/tables")
RELEASE = "0b4f099"
OLD = "main80-nine-cell-v1"
CORRECTED = "corrected-nine-cell-v1"
NATURAL = "natural-chunk-v2"
DATASET_NAMES = {"qasper": "Qasper", "hotpotqa": "HotpotQA", "musique": "MuSiQue"}
THREE = " / ".join(DATASET_NAMES.values())
F1 = "LongBench QA F1（百分点）"
EM = "LITM 官方首行 EM（百分点）"
BLEU = "Doc2Dial corpus BLEU（分）"
F1_CI = ("delta_pp", "ci_low_pp", "ci_high_pp")
MAIN_CI = ("delta_b_minus_a_pp", "ci_low_pp", "ci_high_pp")
TIME_CI = ("delta_comparison_minus_reference_s", "delta_ci_low_s", "delta_ci_high_s")
MAIN_RUN = "main80-20261114T041503"
CORRECTION_RUN = "main80-correction-20261117T060352"
FOLLOWUP_RUN = "followup2604-20261117T085011"
MAIN_CODE = "MAIN80 执行快照（服务器输入导出版 227116e4）"
FOLLOWUP_CODE = "FOLLOWUP2604 code/001（服务器输入包 ab816f5）"
MODEL = "Llama-3.1-8B-Instruct"
MAIN_MODEL = f"{MODEL} | ModelScope LLM-Research/Meta-Llama-3.1-8B-Instruct @ master | 权重文件观察版本 043efdca396f2b6f4b5714d76d70df7840856ed7"
FOLLOWUP_MODEL = f"{MODEL} | 服务器 code/001 模型配置与 tokenizer 快照 | 本包未另记权重版本"
MAIN_RUNTIME = MAIN / "01_audit/reports/STAGE1_RUNTIME.md"
MODEL_SOURCE = MAIN / "01_audit/tables/runtime_model_source_files.csv"
FOLLOWUP_PACKAGE = "followup2604-20261117/FOLLOWUP2604_ACCEPTED.zip"
LB_INPUT = "LongBench 扩展文本（Qasper、HotpotQA、MuSiQue） | files/datasets/frozen-main80/longbench.json"
OLD_DEV_INPUT = "MAIN80 旧裁切／补齐的开发输入 | manifest.json → datasets/frozen-main80/development.json"
CORRECTED_DEV_INPUT = f"修正版开发输入 | {FOLLOWUP_PACKAGE} → inputs/datasets/frozen-main80-corrected-v1/development.json"
NATURAL_INPUT = f"公开原数据的完整来源上下文 | {FOLLOWUP_PACKAGE} → inputs/datasets/frozen-followup-v2/natural.json"
LITM_INPUT = "Lost in the Middle 公开 NQ-open 20 文档位置变体 | files/datasets/frozen-main80/litm.json"
DOC2DIAL_INPUT = "Doc2Dial 原始 validation 与文档 | files/datasets/raw-main80/doc2dial/doc2dial_dial_validation.json | files/datasets/raw-main80/doc2dial/doc2dial_doc.json"
DATASET_WEIGHTS = "各数据集内按题等权，再对纳入的数据集等权"
GRID_WEIGHTS = "每个数据集×L/N格内按题等权，再对数据集及纳入格等权"
REQUEST_WEIGHTS = "均值按实际请求等权，重采样保留原配对层"
BLEU_WEIGHTS = "合并全部目标的 1–4 gram 计数及长度后重算 corpus BLEU（13a、exp、单参考）"


@cache
def released_text(path):
    return subprocess.check_output(
        ["git", "show", f"{RELEASE}:{path.as_posix()}"], cwd=REPO
    ).decode("utf-8-sig")


@cache
def released_table(path):
    with StringIO(released_text(path), newline="") as handle:
        return list(csv.DictReader(handle))


def source(path, **filters):
    matches = [
        (path, index, row)
        for index, row in enumerate(released_table(path), 1)
        if all(row[column] == value for column, value in filters.items())
    ]
    if not matches:
        raise ValueError(f"No source rows for {path}, {filters}")
    return matches


def groups_by_dataset(rows, field):
    return {row["dataset"]: int(row[field]) for _, _, row in rows}


def format_group_counts(counts):
    return " / ".join(
        f"{DATASET_NAMES[dataset]}={counts[dataset]}"
        for dataset in DATASET_NAMES
        if dataset in counts
    )


def point(item, field, *, scale=1):
    path, index, row = item
    value = row[field]
    location = f"{path.as_posix()} / 记录{index} / {field}"
    if scale != 1:
        value = float(value) * scale
        location += f" × {scale}"
    return value, location


def point_from(path, field, **filters):
    result, = source(path, **filters)
    return point(result, field)


def point_pair(path, field, method_column, reference, comparison, **filters):
    return (
        point_from(path, field, **{method_column: reference}, **filters),
        point_from(path, field, **{method_column: comparison}, **filters),
    )


def run_trace(batch, stage, model, code, data, generation, evidence):
    return {
        "run_batch": batch,
        "run_stage": stage,
        "model_checkpoint": model,
        "run_code_version": code,
        "data_source": data,
        "generation_or_reuse": generation,
        "run_evidence": evidence,
    }


def main_trace(stage, data, generation, *tables):
    evidence = " | ".join(p.as_posix() for p in (MAIN_RUNTIME, MODEL_SOURCE, *tables))
    return run_trace(MAIN_RUN, stage, MAIN_MODEL, MAIN_CODE, data, generation, evidence)


def followup_trace(stage, data, generation, *tables, history=False):
    evidence = " | ".join((f"{FOLLOWUP_PACKAGE} → server/code-versions.json 与 server/config.json",
                           *(p.as_posix() for p in tables)))
    batch, model, code = FOLLOWUP_RUN, FOLLOWUP_MODEL, FOLLOWUP_CODE
    if history:
        batch += f" | 复用源 {MAIN_RUN}"
        model += f" | 复用源权重表 {MODEL_SOURCE.as_posix()}"
        code += f" | 复用源 {MAIN_CODE}"
    return run_trace(batch, stage, model, code, data, generation, evidence)


def sampling(group_counts, *sources, group_unit="论文或冻结支撑资料组",
             sampling_unit="数据集内配对抽样的支撑资料组", component_count=None,
             component_breakdown=None):
    if isinstance(group_counts, dict):
        support_count = sum(group_counts.values())
        support_breakdown = format_group_counts(group_counts)
    else:
        support_count = group_counts
        support_breakdown = ""
    if sampling_unit is None:
        sampling_unit = "不适用（无重采样区间）"
    elif component_count is None:
        component_count = support_count
        component_breakdown = support_breakdown
    return {
        "n_support_groups": support_count,
        "support_group_unit": group_unit,
        "support_groups_by_dataset": support_breakdown,
        "statistical_sampling_unit": sampling_unit,
        "n_sampling_components": component_count,
        "sampling_components_by_dataset": component_breakdown,
        "count_source_tables": " / ".join(path.as_posix() for path in sources),
    }


def observation(read, problem, version, dataset, sample_count, pair_count,
                reference, comparison, metric, fields, sampling_metadata,
                interval_basis, scope, role="补充比较", design="同题配对", *,
                trace, weights, points, related_result=""):
    path, index, row = read
    delta, low, high = (row[key] if key else "" for key in fields)
    assert int(sample_count) > 0
    if pair_count is not None:
        assert int(pair_count) > 0
    assert math.isfinite(float(delta))
    if low or high:
        assert math.isfinite(float(low)) and math.isfinite(float(high))
        assert float(low) <= float(high)
        assert int(sampling_metadata["n_sampling_components"]) > 0
    else:
        assert interval_basis.startswith("原表无区间")
    (reference_point, reference_source), (comparison_point, comparison_source) = points
    assert math.isclose(float(comparison_point) - float(reference_point), float(delta),
                        rel_tol=1e-9, abs_tol=1e-10), (path, index, reference_point, comparison_point, delta)
    if metric == F1:
        point_unit = "F1×100"
    elif metric == EM:
        point_unit = "EM×100"
    elif metric.startswith(EM):
        point_unit = "EM百分点"
    elif metric == BLEU:
        point_unit = "BLEU分"
    else:
        point_unit = "秒"
    return {
        "experiment_question": problem,
        "rule_version": version,
        "dataset": dataset,
        "n_unique_questions": int(sample_count),
        "sample_unit": "目标轮次" if dataset == "Doc2Dial" else "题",
        "n_paired_conditions": pair_count,
        **sampling_metadata,
        "comparison_design": design,
        "reference": reference,
        "comparison": comparison,
        "metric": metric,
        "delta_comparison_minus_reference": delta,
        "ci95_low": low,
        "ci95_high": high,
        "ci_basis": interval_basis,
        "scope": scope,
        "role": role,
        "source_release": RELEASE,
        "source_file": path.as_posix(),
        "source_record": index,
        "source_value_fields": " / ".join(key for key in fields if key),
        **trace,
        "score_weighting": weights,
        "reference_point_estimate": reference_point,
        "comparison_point_estimate": comparison_point,
        "point_estimate_unit": point_unit,
        "reference_point_source": reference_source,
        "comparison_point_source": comparison_source,
        "related_result": related_result,
    }


def baseline():
    group_sets = defaultdict(set)
    for _, _, row in source(BASELINE_MEMBERSHIP, method="sequential"):
        group_sets[row["dataset"]].add(row["group"])
    group_counts = {dataset: len(groups) for dataset, groups in group_sets.items()}
    version = "MAIN80 基线（默认 APE 0.9/0.9）"
    trace = main_trace("baseline", LB_INPUT, "MAIN80 原始冷请求，三方法各600条实际生成", BASELINE_MEMBERSHIP)
    for item in source(BASELINE / "baseline_quality_paired.csv"):
        row = item[2]
        name = row["dataset"]
        dataset = DATASET_NAMES.get(name, THREE)
        counts = group_counts if name == "macro_equal_dataset" else {name: group_counts[name]}
        yield observation(item, "完整600题基线的质量差", version,
                          dataset, row["n"], row["n"], row["method_a"], row["method_b"],
                          F1, MAIN_CI, sampling(counts, BASELINE_MEMBERSHIP),
                          "组内题目共同保留，宏平均按数据集等权", "完整基线：各数据集200题",
                          trace=trace, weights=DATASET_WEIGHTS,
                          points=point_pair(BASELINE / "baseline_quality.csv", "f1_percent", "method",
                                            row["method_a"], row["method_b"], dataset=name))
    for item in source(BASELINE / "baseline_efficiency_paired.csv",
                       dataset="pooled600_stratified_CI", method_a="sequential"):
        row = item[2]
        metric = "平均首 token 时间（秒）" if row["metric"] == "ttft_seconds" else "平均总响应时间（秒）"
        yield observation(item, "本机冷请求效率", version,
                          THREE, row["n"], row["n"], row["method_a"], row["method_b"],
                          metric, ("delta_b_minus_a_s", "delta_ci_low_s", "delta_ci_high_s"),
                          sampling(group_counts, BASELINE_MEMBERSHIP),
                          "按数据集分层配对，按实际请求汇总", "完整600题冷请求",
                          trace=trace, weights=REQUEST_WEIGHTS,
                          points=(point(item, "mean_a_s"), point(item, "mean_b_s")))
    paper_trace = {
        **trace,
        "run_batch": f"APE 原论文公开汇总 | {MAIN_RUN}",
        "run_stage": "APE 原论文 Table 2 | baseline",
        "model_checkpoint": f"论文同名 {MODEL}（权重版本未登记） | {MAIN_MODEL}",
        "run_code_version": f"APE 论文 v1 Table 2（执行快照未登记） | {MAIN_CODE}",
        "generation_or_reuse": "论文原有公开汇总与MAIN80本机冷请求生成的描述性对照",
    }
    for item in source(BASELINE / "official_comparison.csv", dataset="macro_equal_dataset"):
        row = item[2]
        yield observation(item, "与同名上游模型三任务公开结果对照", version, THREE,
                          row["local_n"], None, f"APE 论文 Table {row['paper_table']} {row['method']}",
                          f"MAIN80 {row['method']}", F1, ("local_minus_paper_pp", None, None),
                          sampling(group_counts, BASELINE_MEMBERSHIP, sampling_unit=None),
                          "原表无区间（公开汇总的描述性对照）",
                          "唯一题数和组数为 MAIN80 范围，论文和本机输入与题目设置各依原口径",
                          role="公开参照", design="公开汇总对照",
                          trace={**paper_trace, "data_source": f"{LB_INPUT} | 论文输入：{row['paper_input']}"},
                          weights="双方分别依其任务汇总分数，对共同的三个数据集等权",
                          points=(point(item, "paper_f1_percent"), point(item, "local_f1_percent")))


def development():
    stage = MAIN / "02_parameter_exploration/tables"
    rankings = stage / "global_parameter_rankings.csv"
    item, = source(rankings, scope="all_27", T="0.7", S="1.0")
    records = stage / "exploration_records.csv"
    explored_ids = {row["sample_id"] for _, _, row in source(records)}
    explored_groups = defaultdict(set)
    joined_ids = set()
    for _, _, row in source(BASELINE_MEMBERSHIP, method="sequential"):
        if row["sample_id"] in explored_ids:
            joined_ids.add(row["sample_id"])
            explored_groups[row["dataset"]].add(row["group"])
    assert explored_ids == joined_ids
    n_explored = len(explored_ids)
    counts = {dataset: len(groups) for dataset, groups in explored_groups.items()}
    exploration = sampling(counts, records, BASELINE_MEMBERSHIP, sampling_unit=None)
    trace = main_trace("explore", OLD_DEV_INPUT, "MAIN80 原始实验生成 | 选参阶段复用相同输入的资料KV", records)
    yield observation(item, "原探索提名能否改善探索集分数", OLD, THREE,
                      n_explored, item[2]["n_generations_per_parameter"], "默认 0.9/0.9", "全局候选 0.7/1.0",
                      F1, ("delta_vs_default_pp", None, None), exploration, "原表无区间（探索描述）",
                      "50题探索池中实际48题，139份适用题长",
                      trace=trace, weights=GRID_WEIGHTS,
                      points=(point_from(rankings, "f1_percent", scope="all_27", T="0.9", S="0.9"),
                              point(item, "f1_percent")))

    validation = MAIN / "03_validation_rule/tables"
    count_table = validation / "bootstrap_diagnostics.csv"
    counts = groups_by_dataset(source(count_table, scope="all_27"), "clusters")
    validation_trace = main_trace("validation", OLD_DEV_INPUT,
                                  "MAIN80 原始验证生成 | 同输入同参的候选标签共享生成",
                                  validation / "generation_reuse.csv")
    for item in source(validation / "validation_policy_paired.csv", scope="all_27"):
        row = item[2]
        if row["policy_b"] in {"global_candidate", "original_frozen_rule"}:
            yield observation(item, "原验证的固定与逐格参数选择", OLD, THREE,
                              row["n_unique_ids"], row["n_condition_pairs"], row["policy_a"], row["policy_b"],
                              F1, MAIN_CI, sampling(counts, count_table), "数据集与九格等权",
                              "用于选择的验证记录", trace=validation_trace, weights=GRID_WEIGHTS,
                              points=point_pair(validation / "validation_policy_scores.csv", "f1_percent", "policy",
                                                row["policy_a"], row["policy_b"], scope=row["scope"]))

    item, = source(FOLLOWUP / "a_exploration_scores.csv", T="0.7", S="1.0")
    yield observation(item, "适用样本筛选改变同参数探索均值", f"{OLD} / {CORRECTED}", THREE,
                      n_explored, None, "旧全量 T/S=0.7/1.0", "旧输入中保留修正后仍适用题长 T/S=0.7/1.0",
                      F1, ("sample_applicability_shift_pp", None, None), exploration,
                      "原表无区间（样本组成差异）",
                      f"旧输入条件 {item[2]['old_all_conditions']}→{item[2]['common_conditions']}，"
                      "题数和组数为旧探索全集", design="适用范围变化",
                      trace={**trace, "data_source": f"{OLD_DEV_INPUT} | 共同范围由{CORRECTED_DEV_INPUT}确定",
                             "generation_or_reuse": "同一批 MAIN80 旧探索回答，全量与共同适用子集"},
                      weights=GRID_WEIGHTS,
                      points=(point(item, "old_all_f1_percent"), point(item, "old_common_f1_percent")))

    count_table = FOLLOWUP / "a_bootstrap_diagnostics.csv"
    counts_rows = source(count_table, scope_name="exploration_fixed_parameter_old_corrected_input")
    counts = groups_by_dataset(counts_rows, "frozen_support_groups")
    explored_questions = sum(int(row["unique_questions"]) for _, _, row in counts_rows)
    origin_counts = FOLLOWUP / "a_generation_origin_counts.csv"
    correction_archive = f"output/runs/{CORRECTION_RUN}"
    correction_trace = run_trace(
        f"{MAIN_RUN} | {CORRECTION_RUN}", "explore / corrected_exploration",
        f"{MAIN_MODEL} | 修正探索沿用 {MODEL}，包内未另记权重版本",
        f"{MAIN_CODE} | 前置修正探索的 code-version.json 与 snapshot/ 已归档于服务器 {correction_archive}/，"
        "PC 交付包未附这两项，包内逐项来源见 run_evidence",
        f"{OLD_DEV_INPUT} | {CORRECTED_DEV_INPUT}",
        "修正探索全库含前置修正生成1248条及历史同输入复用4080条，当前行按已冻结参数筛选",
        f"{origin_counts.as_posix()} | {FOLLOWUP_PACKAGE} → provenance/corrected-exploration-for-candidate-reconstruction.jsonl"
        f" | 服务器归档（PC 交付包未附）：{correction_archive}/code-version.json 与 {correction_archive}/snapshot/",
    )
    for item in source(FOLLOWUP / "a_exploration_frozen_parameter_pairs.csv"):
        row = item[2]
        yield observation(item, "固定参数时修正输入的影响", f"{OLD} / {CORRECTED}", THREE,
                          explored_questions, row["n_input_conditions"], f"旧输入 T/S={row['T']}/{row['S']}",
                          f"修正输入 T/S={row['T']}/{row['S']}", F1,
                          ("delta_corrected_minus_old_pp", "ci_low_pp", "ci_high_pp"),
                          sampling(counts, count_table), "数据集与九格等权", "仅共同适用题长，同题同L/N及参数",
                          trace=correction_trace, weights=GRID_WEIGHTS,
                          points=(point(item, "old_common_f1_percent"), point(item, "corrected_f1_percent")))

    counts = groups_by_dataset(source(count_table, scope_name="A_frozen_policy_validation"),
                               "frozen_support_groups")
    trace = followup_trace("A_corrected_validation", CORRECTED_DEV_INPUT,
                           "A阶段新生成462条、历史严格复用377条 | 同参候选另共享逻辑标签",
                           origin_counts, history=True)
    for item in source(FOLLOWUP / "a_validation_paired.csv", scope="macro_equal_dataset_cell"):
        row = item[2]
        if row["comparison"] in {"fixed", "dynamic"}:
            yield observation(item, "修正输入后的冻结策略验证", CORRECTED, THREE,
                              row["n_questions"], row["n_conditions"], row["reference"], row["comparison"],
                              F1, F1_CI, sampling(counts, count_table), "数据集与九格等权",
                              "A阶段，参与选择的验证题", trace=trace, weights=GRID_WEIGHTS,
                              points=point_pair(FOLLOWUP / "a_validation_policies.csv", "f1_percent", "policy",
                                                row["reference"], row["comparison"], scope=row["scope"]))

    stage = MAIN / "04_length_check/tables"
    count_table = stage / "bootstrap_units.csv"
    counts = groups_by_dataset(source(count_table, scope="all10"), "evidence_groups")
    trace = main_trace("length_check", OLD_DEV_INPUT,
                       "MAIN80 原始冷请求，固定与动态各60条生成", stage / "paired_records.csv")
    for item in source(stage / "quality_paired.csv", scope="all10"):
        row = item[2]
        yield observation(item, "原规则的6K/12K长度迁移", OLD, THREE,
                          row["n_unique_ids"], row["n_pairs"], "旧固定", "旧动态", F1,
                          ("difference_pp", "difference_ci_low_pp", "difference_ci_high_pp"),
                          sampling(counts, count_table), "同题各构造共同抽入，三个数据集等权",
                          f"旧长度检查 L={row['L']}, N={row['N']}", trace=trace, weights=GRID_WEIGHTS,
                          points=(point(item, "fixed_f1_percent"), point(item, "dynamic_f1_percent")))

    count_table = FOLLOWUP / "b_bootstrap_diagnostics.csv"
    counts = groups_by_dataset(source(count_table), "n_support_groups")
    trace = followup_trace("B_length_transfer", CORRECTED_DEV_INPUT,
                           "B阶段冷请求新生成103条、历史严格复用17条 | 复用项无本次计时",
                           FOLLOWUP / "b_records.csv", history=True)
    summary_path = FOLLOWUP.parent / "b_results.json"
    mixed = json.loads(released_text(summary_path))["descriptive_mixed_request_mean"]
    mixed_result = (f"同批60条件直接等权（数据集占比4/3/3）｜固定{mixed['fixed_F1_percent']:.4f}，"
                    f"动态{mixed['dynamic_F1_percent']:.4f}，差{mixed['delta_pp']:+.4f}个百分点"
                    f"｜{summary_path.as_posix()}#/descriptive_mixed_request_mean")
    for item in source(FOLLOWUP / "b_quality.csv", dataset="macro_equal_dataset"):
        row = item[2]
        scope = {"L6144": "6K，N=4/8/16", "L12288": "12K，N=4/8/16",
                 "ALL": "6K/12K，N=4/8/16"}.get(row["scope"], row["scope"])
        assert int(row["n_support_groups"]) == sum(counts.values())
        yield observation(item, "修正规则的6K/12K长度迁移", CORRECTED, THREE,
                          row["n_questions"], row["n_input_pairs"], "修正固定", "修正动态", F1,
                          ("delta_pp", "delta_ci_low_pp", "delta_ci_high_pp"), sampling(counts, count_table),
                          "同题全部六条件共同抽入，分层内对数据集等权",
                          f"B阶段 {scope}，与旧长度检查仅7题共同",
                          "迁移总体" if row["scope"] == "ALL" else "迁移探索分层",
                          trace=trace, weights="所列条件内先按题及条件等权求各数据集均值，再对三个数据集等权",
                          points=(point(item, "fixed_F1_percent"), point(item, "dynamic_F1_percent")),
                          related_result=mixed_result if row["scope"] == "ALL" else "")


def formal_and_confirmation():
    stage = MAIN / "06_longbench_final/tables"
    count_table = stage / "bootstrap_diagnostics.csv"
    counts = groups_by_dataset(source(count_table, scope="test510"), "n_clusters")
    trace = main_trace(
        "baseline / longbench_final", LB_INPUT,
        "三方法从完整基线复用同题已有冷请求1530条，历史固定与动态新增冷请求1020条（本运行内）",
        stage / "main_quality.csv",
    )
    for item in source(stage / "main_quality_paired.csv", scope="test510"):
        row = item[2]
        name = row["dataset"]
        if name != "macro_equal_present_dataset" and (
            row["reference_method"], row["comparison_method"]
        ) != ("ape_fixed", "ape_dynamic"):
            continue
        use_counts = counts if name == "macro_equal_present_dataset" else {name: counts[name]}
        yield observation(item, "旧规则在LongBench正式测试上的质量", OLD, DATASET_NAMES.get(name, THREE),
                          row["n_questions"], row["n_questions"], row["reference_method"],
                          row["comparison_method"], F1, F1_CI, sampling(use_counts, count_table),
                          "在冻结组内保留所有方法，各集先按题平均", "旧510题（各集170题），完整LongBench扩展文本",
                          "历史测试", trace=trace, weights=DATASET_WEIGHTS,
                          points=point_pair(stage / "main_quality.csv", "f1_percent", "method",
                                            row["reference_method"], row["comparison_method"],
                                            scope=row["scope"], dataset=name))
    for scope, name in (("support_overlap", "musique"), ("support_nonoverlap", "macro_equal_present_dataset")):
        item, = source(stage / "quality_paired.csv", scope=scope, dataset=name,
                       reference_method="ape_fixed", comparison_method="ape_dynamic")
        row = item[2]
        use_counts = groups_by_dataset(source(count_table, scope=scope), "n_clusters")
        yield observation(item, "旧测试与开发资料重合的质量分层", OLD, DATASET_NAMES.get(name, THREE),
                          row["n_questions"], row["n_questions"], row["reference_method"],
                          row["comparison_method"], F1, F1_CI, sampling(use_counts, count_table),
                          "在冻结组内保留所有方法，各集先按题平均",
                          "与开发共享至少一篇支撑资料" if scope == "support_overlap" else "与开发无支撑资料重合",
                          "历史测试分层", trace=trace, weights=DATASET_WEIGHTS,
                          points=point_pair(stage / "quality.csv", "f1_percent", "method",
                                            row["reference_method"], row["comparison_method"],
                                            scope=scope, dataset=name))
    for item in source(stage / "efficiency_paired.csv", scope="test510", dataset="pooled_requests",
                       reference_method="ape_fixed", comparison_method="ape_dynamic"):
        row = item[2]
        metric = "平均首 token 时间（秒）" if row["metric"] == "ttft_seconds" else "平均总响应时间（秒）"
        yield observation(item, "旧510题冷请求效率", OLD, THREE,
                          row["n_questions"], row["n_questions"], row["reference_method"],
                          row["comparison_method"], metric, TIME_CI, sampling(counts, count_table),
                          "在冻结组内保留所有方法，按实际请求汇总", "旧510题，实际请求均值",
                          trace=trace, weights=REQUEST_WEIGHTS,
                          points=(point(item, "reference_mean_s"), point(item, "comparison_mean_s")))

    count_table = FOLLOWUP / "c_bootstrap_diagnostics.csv"
    counts = {row["scope"]: int(row["components"]) for _, _, row in source(count_table)}
    primary_counts = {name: counts[name] for name in DATASET_NAMES}
    c_quality = FOLLOWUP / "c_quality.csv"
    trace = followup_trace(
        "C3_independent_confirmation", NATURAL_INPUT,
        "C3同批新生成897条、1050个逻辑方法结果 | 同参标签共享生成 | 历史规则同批重跑",
        FOLLOWUP / "c_membership.csv", FOLLOWUP / "c_logical_answers.csv",
    )
    for item in source(FOLLOWUP / "c_paired.csv", comparison_method="ape_natural_dynamic"):
        row = item[2]
        name = row["scope"]
        macro = name == "macro_equal_dataset"
        primary = macro and row["reference_method"] == "ape_natural_fixed"
        if not macro and row["reference_method"] != "ape_natural_fixed":
            continue
        group_unit = "共享支撑资料的连通组件"
        strata = name.startswith("qasper_")
        if strata:
            dataset = "Qasper"
            basis = "共同保留N1/N2/N3，独立随机流，任一层缺失则重抽"
            metadata = sampling({"qasper": int(row["n_components"])}, item[0], count_table,
                                group_unit=group_unit, sampling_unit="Qasper 支撑资料连通组件",
                                component_count=counts["qasper_N_strata_and_equal_N"],
                                component_breakdown=f"Qasper={counts['qasper_N_strata_and_equal_N']}")
            description = f"新150题的 Qasper 层 {name.removeprefix('qasper_')}"
            weights = ("Qasper 各N层内按题平均，再对N1/N2/N3等权"
                       if name == "qasper_equal_N_supplement" else "Qasper 本N层内按题等权")
        else:
            dataset = THREE if macro else DATASET_NAMES[name]
            use_counts = primary_counts if macro else {name: counts[name]}
            metadata = sampling(use_counts, item[0], count_table, group_unit=group_unit,
                                sampling_unit="数据集内配对抽样的支撑资料连通组件")
            basis = "抽中组件保留全部题和方法"
            if macro:
                basis += "，先数据集内题均值，再三集等权"
            description = ("新150题（各集50题），完整原始来源上下文" if macro
                           else f"新确认集的 {dataset} 单数据集50题，完整原始来源上下文")
            weights = DATASET_WEIGHTS if macro else "本数据集内50题按题等权"
        version = NATURAL
        if row["reference_method"].startswith("ape_main80"):
            version = f"{NATURAL} / {OLD}"
        yield observation(item, "自然分块动态规则的独立确认", version, dataset,
                          row["n_questions"], row["n_questions"], row["reference_method"],
                          row["comparison_method"], F1, F1_CI, metadata, basis, description,
                          "动态规则主要验证" if primary else "确认集参照或分层",
                          trace=trace, weights=weights,
                          points=point_pair(c_quality, "f1_percent", "method", row["reference_method"],
                                            row["comparison_method"], scope=name))
    item, = source(FOLLOWUP / "c_paired.csv", scope="macro_equal_dataset",
                   reference_method="ape_main80_fixed", comparison_method="ape_natural_fixed")
    yield observation(item, "新版固定与历史固定在同输入上的比较", f"{NATURAL} / {OLD}", THREE,
                      item[2]["n_questions"], item[2]["n_questions"], "ape_main80_fixed",
                      "ape_natural_fixed", F1, F1_CI,
                      sampling(primary_counts, item[0], count_table, group_unit="共享支撑资料的连通组件",
                               sampling_unit="数据集内配对抽样的支撑资料连通组件"),
                      "先数据集内题均值，再三集等权", "同一新150题，历史规则重新执行",
                      trace=trace, weights=DATASET_WEIGHTS,
                      points=point_pair(c_quality, "f1_percent", "method", "ape_main80_fixed",
                                        "ape_natural_fixed", scope="macro_equal_dataset"))


def position():
    attention_pairs = MAIN / "05_attention/tables/answer_pairs.csv"
    trace = main_trace("attention", OLD_DEV_INPUT, "独立 attention 探查生成，3题×2种构造×2组参数",
                       attention_pairs)
    for item in source(attention_pairs):
        row = item[2]
        _, _, membership = source(BASELINE_MEMBERSHIP, sample_id=row["sample_id"], method="sequential")[0]
        assert membership["group"]
        yield observation(item, "探索参数如何变化注意力诊断答案", "MAIN80 探索候选",
                          DATASET_NAMES[row["dataset"]], 1, 1, f"默认 {row['default_T']}/{row['default_S']}",
                          f"候选 {row['best_T']}/{row['best_S']}", F1, ("difference_pp", None, None),
                          sampling({row["dataset"]: 1}, BASELINE_MEMBERSHIP, sampling_unit=None),
                          "原表无区间（单例诊断）", f"L={row['L']}, N={row['N']}, 样本 {row['sample_id']}",
                          trace=trace, weights="单题单构造的答案 F1",
                          points=(point(item, "default_f1_0_1", scale=100), point(item, "best_f1_0_1", scale=100)))

    stage = MAIN / "07_litm/tables"
    questions = len(source(stage / "bootstrap_questions.csv"))
    metadata = sampling(None, stage / "bootstrap_questions.csv", stage / "membership.csv",
                        group_unit="未建立跨题支撑资料组",
                        sampling_unit="题（同题三个位置及全部方法共同保留）", component_count=questions)
    trace = main_trace(
        "litm", LITM_INPUT, "三方法×三个位置×500题的4500条历史冷请求 | PC从原回答重新评分",
        stage.parent / "sources/input_sources.json", stage / "per_question_scores.csv",
    )
    position_weights = "先按题平均指定位置，再按题等权"
    for item in source(stage / "main_quality_paired.csv"):
        row = item[2]
        yield observation(item, "证据位置如何影响首行答题准确率", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_requests_per_method"], row["reference_method"],
                          row["comparison_method"], EM, F1_CI, metadata, "按题配对，同题保留三个位置",
                          f"位置 {row['position']}，20篇资料，全部实际N=1",
                          trace=trace, weights=position_weights,
                          points=point_pair(stage / "main_quality.csv", "em_percent", "method",
                                            row["reference_method"], row["comparison_method"],
                                            position=row["position"], score_metric=row["score_metric"]))
    for contrast, metric in (("middle_minus_ends_mean", "中位减两端均值"),
                             ("last_minus_first", "末位减首位")):
        item, = source(stage / "main_paired_position_contrasts.csv", contrast=contrast,
                       reference_method="ape_fixed", comparison_method="ape_dynamic")
        row = item[2]
        yield observation(item, "历史动态如何改变不同证据位置之间的落差", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_questions"], row["reference_method"], row["comparison_method"],
                          f"{EM}，{metric}的变化", ("delta_contrast_pp", "ci_low_pp", "ci_high_pp"),
                          metadata, "题内位置差共同重采样", "比较预先定义的位置组合",
                          trace=trace, weights=f"每题先算{metric}，再对500题等权",
                          points=point_pair(stage / "main_position_contrasts.csv", "contrast_pp", "method",
                                            row["reference_method"], row["comparison_method"],
                                            contrast=contrast, score_metric=row["score_metric"]))
    for item in source(stage / "scoring_scope_impact.csv", position="ALL"):
        row = item[2]
        yield observation(item, "首行重评分修正了多少原始分数", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_requests_per_method"], f"{row['method']} 原全文",
                          f"{row['method']} 官方首行", EM, ("official_minus_saved_pp", "ci_low_pp", "ci_high_pp"),
                          metadata, "同一生成的两种评分配对", f"三个位置共同汇总，{row['changed_generations']} 次生成分数改变",
                          trace=trace, weights=position_weights,
                          points=point_pair(stage / "quality.csv", "em_percent", "score_metric", "saved_full_answer_em",
                                            "official_first_line_em", position=row["position"], method=row["method"]))
    for item in source(stage / "efficiency_paired.csv", position="ALL", metric="total_seconds"):
        row = item[2]
        yield observation(item, "位置问答的冷请求总响应差", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_requests_per_method"], row["reference_method"],
                          row["comparison_method"], "平均总响应时间（秒）", TIME_CI, metadata,
                          "按题配对，同题保留三个位置", "三个位置共同汇总，冷请求",
                          trace=trace, weights=REQUEST_WEIGHTS,
                          points=(point(item, "reference_mean_s"), point(item, "comparison_mean_s")))


def doc2dial():
    stage = MAIN / "08_doc2dial/tables"
    count_table = stage / "bootstrap_diagnostics.csv"
    counts = {row["cluster_unit"]: int(row["n_clusters"]) for _, _, row in source(count_table, scope="all500")}
    data_scope = "原始 Doc2Dial validation 的500个目标轮次"
    trace = main_trace(
        "doc2dial", DOC2DIAL_INPUT,
        "三方法各500冷、500暖请求，共3000次实际生成 | 暖请求只复用同进程资料KV，重新生成回答",
        stage / "hot_generation_evidence.csv",
    )
    for file, filters in (
        ("main_quality_paired.csv", {}),
        ("main_quality_paired_document_sensitivity.csv",
         {"reference_method": "ape_fixed", "comparison_method": "ape_dynamic", "reference_cache": "cold"}),
    ):
        for item in source(stage / file, **filters):
            row = item[2]
            cluster = "对话" if row["cluster_unit"] == "dialogue" else "资料"
            quality = stage / file.replace("_paired", "")
            metadata = sampling(counts["document"], count_table, group_unit="原始资料",
                                sampling_unit=cluster, component_count=counts[row["cluster_unit"]])
            yield observation(item, "多轮问答质量及冷热一致性", OLD, "Doc2Dial",
                              row["n_targets"], row["n_targets"],
                              f"{row['reference_method']} ({row['reference_cache']})",
                              f"{row['comparison_method']} ({row['comparison_cache']})",
                              BLEU, ("delta_bleu", "ci_low", "ci_high"), metadata,
                              f"按{cluster}配对，每次重算 corpus BLEU", data_scope,
                              "资料抽样敏感性" if cluster == "资料" else "补充比较",
                              trace=trace, weights=BLEU_WEIGHTS,
                              points=(point_from(quality, "corpus_bleu", scope=row["scope"],
                                                 method=row["reference_method"], cache_mode=row["reference_cache"]),
                                      point_from(quality, "corpus_bleu", scope=row["scope"],
                                                 method=row["comparison_method"], cache_mode=row["comparison_cache"])))
    for item in source(stage / "cache_effects.csv"):
        row = item[2]
        metric = "平均首 token 时间（秒）" if row["metric"] == "ttft_seconds" else "平均总响应时间（秒）"
        yield observation(item, "同题资料KV热复用降低多少时延", OLD, "Doc2Dial",
                          row["n_targets"], row["n_targets"], f"{row['reference_method']} (cold)",
                          f"{row['comparison_method']} (hot)", metric, TIME_CI,
                          sampling(counts["document"], count_table, group_unit="原始资料",
                                   sampling_unit="对话", component_count=counts["dialogue"]),
                          "按对话配对，保留同目标轮次的冷与热请求", f"{data_scope}，同进程重新生成",
                          trace=trace, weights=REQUEST_WEIGHTS,
                          points=(point(item, "reference_mean_s"), point(item, "comparison_mean_s")))


def main():
    rows = [row for section in (baseline, development, formal_and_confirmation, position, doc2dial)
            for row in section()]
    primary, = [row for row in rows if row["role"] == "动态规则主要验证"]
    assert primary["n_unique_questions"] == 150
    assert primary["n_support_groups"] == primary["n_sampling_components"] == 68
    assert math.isclose(float(primary["delta_comparison_minus_reference"]), 0.1422216435782886)
    keys = [(row["source_file"], row["source_record"]) for row in rows]
    assert len(keys) == len(set(keys)), "A source record was included twice"
    with (OUTPUT / "统一结果表.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Integrated {len(rows)} comparisons, including one primary confirmation row.")


if __name__ == "__main__":
    main()
