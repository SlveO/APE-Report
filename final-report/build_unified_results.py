"""从 PC 发布版 0b4f099 的结果表汇总结题报告所用比较。"""

import csv
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


@cache
def released_table(path):
    content = subprocess.check_output(
        ["git", "show", f"{RELEASE}:{path.as_posix()}"], cwd=REPO
    )
    with StringIO(content.decode("utf-8-sig"), newline="") as handle:
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
                interval_basis, scope, role="补充比较", design="同题配对"):
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
    }


def baseline():
    group_sets = defaultdict(set)
    for _, _, row in source(BASELINE_MEMBERSHIP, method="sequential"):
        group_sets[row["dataset"]].add(row["group"])
    group_counts = {dataset: len(groups) for dataset, groups in group_sets.items()}
    version = "MAIN80 基线（默认 APE 0.9/0.9）"
    for item in source(BASELINE / "baseline_quality_paired.csv"):
        row = item[2]
        name = row["dataset"]
        dataset = DATASET_NAMES.get(name, THREE)
        counts = group_counts if name == "macro_equal_dataset" else {name: group_counts[name]}
        yield observation(item, "完整600题基线的质量差", version,
                          dataset, row["n"], row["n"], row["method_a"], row["method_b"],
                          F1, MAIN_CI, sampling(counts, BASELINE_MEMBERSHIP),
                          "组内题目共同保留，宏平均按数据集等权", "完整基线：各数据集200题")
    for item in source(BASELINE / "baseline_efficiency_paired.csv",
                       dataset="pooled600_stratified_CI", method_a="sequential"):
        row = item[2]
        metric = "平均首 token 时间（秒）" if row["metric"] == "ttft_seconds" else "平均总响应时间（秒）"
        yield observation(item, "本机冷请求效率", version,
                          THREE, row["n"], row["n"], row["method_a"], row["method_b"],
                          metric, ("delta_b_minus_a_s", "delta_ci_low_s", "delta_ci_high_s"),
                          sampling(group_counts, BASELINE_MEMBERSHIP),
                          "按数据集分层配对，按实际请求汇总", "完整600题冷请求")
    for item in source(BASELINE / "official_comparison.csv", dataset="macro_equal_dataset"):
        row = item[2]
        yield observation(item, "与同名上游模型三任务公开结果对照", version, THREE,
                          row["local_n"], None, f"APE 论文 Table {row['paper_table']} {row['method']}",
                          f"MAIN80 {row['method']}", F1, ("local_minus_paper_pp", None, None),
                          sampling(group_counts, BASELINE_MEMBERSHIP, sampling_unit=None),
                          "原表无区间（公开汇总的描述性对照）",
                          "唯一题数和组数为 MAIN80 范围，论文和本机输入与题目设置各依原口径",
                          role="公开参照", design="公开汇总对照")


def development():
    stage = MAIN / "02_parameter_exploration/tables"
    item, = source(stage / "global_parameter_rankings.csv", scope="all_27", T="0.7", S="1.0")
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
    yield observation(item, "原探索提名能否改善探索集分数", OLD, THREE,
                      n_explored, item[2]["n_generations_per_parameter"], "默认 0.9/0.9", "全局候选 0.7/1.0",
                      F1, ("delta_vs_default_pp", None, None), exploration, "原表无区间（探索描述）",
                      "50题探索池中实际48题，139份适用题长")

    validation = MAIN / "03_validation_rule/tables"
    count_table = validation / "bootstrap_diagnostics.csv"
    counts = groups_by_dataset(source(count_table, scope="all_27"), "clusters")
    for item in source(validation / "validation_policy_paired.csv", scope="all_27"):
        row = item[2]
        if row["policy_b"] in {"global_candidate", "original_frozen_rule"}:
            yield observation(item, "原验证的固定与逐格参数选择", OLD, THREE,
                              row["n_unique_ids"], row["n_condition_pairs"], row["policy_a"], row["policy_b"],
                              F1, MAIN_CI, sampling(counts, count_table), "数据集与九格等权",
                              "用于选择的验证记录")

    item, = source(FOLLOWUP / "a_exploration_scores.csv", T="0.7", S="1.0")
    yield observation(item, "适用样本筛选改变同参数探索均值", f"{OLD} / {CORRECTED}", THREE,
                      n_explored, None, "旧全量 T/S=0.7/1.0", "旧输入中保留修正后仍适用题长 T/S=0.7/1.0",
                      F1, ("sample_applicability_shift_pp", None, None), exploration,
                      "原表无区间（样本组成差异）",
                      f"旧输入条件 {item[2]['old_all_conditions']}→{item[2]['common_conditions']}，"
                      "题数和组数为旧探索全集", design="适用范围变化")

    count_table = FOLLOWUP / "a_bootstrap_diagnostics.csv"
    counts_rows = source(count_table, scope_name="exploration_fixed_parameter_old_corrected_input")
    counts = groups_by_dataset(counts_rows, "frozen_support_groups")
    explored_questions = sum(int(row["unique_questions"]) for _, _, row in counts_rows)
    for item in source(FOLLOWUP / "a_exploration_frozen_parameter_pairs.csv"):
        row = item[2]
        yield observation(item, "固定参数时修正输入的影响", f"{OLD} / {CORRECTED}", THREE,
                          explored_questions, row["n_input_conditions"], f"旧输入 T/S={row['T']}/{row['S']}",
                          f"修正输入 T/S={row['T']}/{row['S']}", F1,
                          ("delta_corrected_minus_old_pp", "ci_low_pp", "ci_high_pp"),
                          sampling(counts, count_table), "数据集与九格等权", "仅共同适用题长，同题同L/N及参数")

    counts = groups_by_dataset(source(count_table, scope_name="A_frozen_policy_validation"),
                               "frozen_support_groups")
    for item in source(FOLLOWUP / "a_validation_paired.csv", scope="macro_equal_dataset_cell"):
        row = item[2]
        if row["comparison"] in {"fixed", "dynamic"}:
            yield observation(item, "修正输入后的冻结策略验证", CORRECTED, THREE,
                              row["n_questions"], row["n_conditions"], row["reference"], row["comparison"],
                              F1, F1_CI, sampling(counts, count_table), "数据集与九格等权",
                              "A阶段，参与选择的验证题")

    stage = MAIN / "04_length_check/tables"
    count_table = stage / "bootstrap_units.csv"
    counts = groups_by_dataset(source(count_table, scope="all10"), "evidence_groups")
    for item in source(stage / "quality_paired.csv", scope="all10"):
        row = item[2]
        yield observation(item, "原规则的6K/12K长度迁移", OLD, THREE,
                          row["n_unique_ids"], row["n_pairs"], "旧固定", "旧动态", F1,
                          ("difference_pp", "difference_ci_low_pp", "difference_ci_high_pp"),
                          sampling(counts, count_table), "同题各构造共同抽入，三个数据集等权",
                          f"旧长度检查 L={row['L']}, N={row['N']}")

    count_table = FOLLOWUP / "b_bootstrap_diagnostics.csv"
    counts = groups_by_dataset(source(count_table), "n_support_groups")
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
                          "迁移总体" if row["scope"] == "ALL" else "迁移探索分层")


def formal_and_confirmation():
    stage = MAIN / "06_longbench_final/tables"
    count_table = stage / "bootstrap_diagnostics.csv"
    counts = groups_by_dataset(source(count_table, scope="test510"), "n_clusters")
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
                          "历史测试")
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
                          "历史测试分层")
    for item in source(stage / "efficiency_paired.csv", scope="test510", dataset="pooled_requests",
                       reference_method="ape_fixed", comparison_method="ape_dynamic"):
        row = item[2]
        metric = "平均首 token 时间（秒）" if row["metric"] == "ttft_seconds" else "平均总响应时间（秒）"
        yield observation(item, "旧510题冷请求效率", OLD, THREE,
                          row["n_questions"], row["n_questions"], row["reference_method"],
                          row["comparison_method"], metric, TIME_CI, sampling(counts, count_table),
                          "在冻结组内保留所有方法，按实际请求汇总", "旧510题，实际请求均值")

    count_table = FOLLOWUP / "c_bootstrap_diagnostics.csv"
    counts = {row["scope"]: int(row["components"]) for _, _, row in source(count_table)}
    primary_counts = {name: counts[name] for name in DATASET_NAMES}
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
        else:
            dataset = THREE if macro else DATASET_NAMES[name]
            use_counts = primary_counts if macro else {name: counts[name]}
            metadata = sampling(use_counts, item[0], count_table, group_unit=group_unit,
                                sampling_unit="数据集内配对抽样的支撑资料连通组件")
            basis = "抽中组件保留全部题和方法"
            if macro:
                basis += "，先数据集内题均值，再三集等权"
            description = "新150题（各集50题），完整原始来源上下文"
        version = NATURAL
        if row["reference_method"].startswith("ape_main80"):
            version = f"{NATURAL} / {OLD}"
        yield observation(item, "自然分块动态规则的独立确认", version, dataset,
                          row["n_questions"], row["n_questions"], row["reference_method"],
                          row["comparison_method"], F1, F1_CI, metadata, basis, description,
                          "动态规则主要验证" if primary else "确认集参照或分层")
    item, = source(FOLLOWUP / "c_paired.csv", scope="macro_equal_dataset",
                   reference_method="ape_main80_fixed", comparison_method="ape_natural_fixed")
    yield observation(item, "新版固定与历史固定在同输入上的比较", f"{NATURAL} / {OLD}", THREE,
                      item[2]["n_questions"], item[2]["n_questions"], "ape_main80_fixed",
                      "ape_natural_fixed", F1, F1_CI,
                      sampling(primary_counts, item[0], count_table, group_unit="共享支撑资料的连通组件",
                               sampling_unit="数据集内配对抽样的支撑资料连通组件"),
                      "先数据集内题均值，再三集等权", "同一新150题，历史规则重新执行")


def position():
    for item in source(MAIN / "05_attention/tables/answer_pairs.csv"):
        row = item[2]
        _, _, membership = source(BASELINE_MEMBERSHIP, sample_id=row["sample_id"], method="sequential")[0]
        assert membership["group"]
        yield observation(item, "探索参数如何变化注意力诊断答案", "MAIN80 探索候选",
                          DATASET_NAMES[row["dataset"]], 1, 1, f"默认 {row['default_T']}/{row['default_S']}",
                          f"候选 {row['best_T']}/{row['best_S']}", F1, ("difference_pp", None, None),
                          sampling({row["dataset"]: 1}, BASELINE_MEMBERSHIP, sampling_unit=None),
                          "原表无区间（单例诊断）", f"L={row['L']}, N={row['N']}, 样本 {row['sample_id']}")

    stage = MAIN / "07_litm/tables"
    questions = len(source(stage / "bootstrap_questions.csv"))
    metadata = sampling(None, stage / "bootstrap_questions.csv", stage / "membership.csv",
                        group_unit="未建立跨题支撑资料组",
                        sampling_unit="题（同题三个位置及全部方法共同保留）", component_count=questions)
    for item in source(stage / "main_quality_paired.csv"):
        row = item[2]
        yield observation(item, "证据位置如何影响首行答题准确率", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_requests_per_method"], row["reference_method"],
                          row["comparison_method"], EM, F1_CI, metadata, "按题配对，同题保留三个位置",
                          f"位置 {row['position']}，20篇资料，全部实际N=1")
    for contrast, metric in (("middle_minus_ends_mean", "中位减两端均值"),
                             ("last_minus_first", "末位减首位")):
        item, = source(stage / "main_paired_position_contrasts.csv", contrast=contrast,
                       reference_method="ape_fixed", comparison_method="ape_dynamic")
        row = item[2]
        yield observation(item, "历史动态如何改变不同证据位置之间的落差", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_questions"], row["reference_method"], row["comparison_method"],
                          f"{EM}，{metric}的变化", ("delta_contrast_pp", "ci_low_pp", "ci_high_pp"),
                          metadata, "题内位置差共同重采样", "比较预先定义的位置组合")
    for item in source(stage / "scoring_scope_impact.csv", position="ALL"):
        row = item[2]
        yield observation(item, "首行重评分修正了多少原始分数", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_requests_per_method"], f"{row['method']} 原全文",
                          f"{row['method']} 官方首行", EM, ("official_minus_saved_pp", "ci_low_pp", "ci_high_pp"),
                          metadata, "同一生成的两种评分配对", f"三个位置共同汇总，{row['changed_generations']} 次生成分数改变")
    for item in source(stage / "efficiency_paired.csv", position="ALL", metric="total_seconds"):
        row = item[2]
        yield observation(item, "位置问答的冷请求总响应差", OLD, "Lost in the Middle",
                          row["n_questions"], row["n_requests_per_method"], row["reference_method"],
                          row["comparison_method"], "平均总响应时间（秒）", TIME_CI, metadata,
                          "按题配对，同题保留三个位置", "三个位置共同汇总，冷请求")


def doc2dial():
    stage = MAIN / "08_doc2dial/tables"
    count_table = stage / "bootstrap_diagnostics.csv"
    counts = {row["cluster_unit"]: int(row["n_clusters"]) for _, _, row in source(count_table, scope="all500")}
    data_scope = "原始 Doc2Dial validation 的500个目标轮次"
    for file, filters in (
        ("main_quality_paired.csv", {}),
        ("main_quality_paired_document_sensitivity.csv",
         {"reference_method": "ape_fixed", "comparison_method": "ape_dynamic", "reference_cache": "cold"}),
    ):
        for item in source(stage / file, **filters):
            row = item[2]
            cluster = "对话" if row["cluster_unit"] == "dialogue" else "资料"
            metadata = sampling(counts["document"], count_table, group_unit="原始资料",
                                sampling_unit=cluster, component_count=counts[row["cluster_unit"]])
            yield observation(item, "多轮问答质量及冷热一致性", OLD, "Doc2Dial",
                              row["n_targets"], row["n_targets"],
                              f"{row['reference_method']} ({row['reference_cache']})",
                              f"{row['comparison_method']} ({row['comparison_cache']})",
                              BLEU, ("delta_bleu", "ci_low", "ci_high"), metadata,
                              f"按{cluster}配对，每次重算 corpus BLEU", data_scope,
                              "资料抽样敏感性" if cluster == "资料" else "补充比较")
    for item in source(stage / "cache_effects.csv"):
        row = item[2]
        metric = "平均首 token 时间（秒）" if row["metric"] == "ttft_seconds" else "平均总响应时间（秒）"
        yield observation(item, "同题资料KV热复用降低多少时延", OLD, "Doc2Dial",
                          row["n_targets"], row["n_targets"], f"{row['reference_method']} (cold)",
                          f"{row['comparison_method']} (hot)", metric, TIME_CI,
                          sampling(counts["document"], count_table, group_unit="原始资料",
                                   sampling_unit="对话", component_count=counts["dialogue"]),
                          "按对话配对，保留同目标轮次的冷与热请求", f"{data_scope}，同进程重新生成")


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
