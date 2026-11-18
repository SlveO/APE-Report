"""Collect existing stage estimates and the proposed follow-up plan, CPU only.

No answers are generated or rescored. Source selectors identify saved table rows.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TABLES = BASE / "tables"
METRICS = []


def read(relative):
    with (BASE / relative).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    TABLES.mkdir(exist_ok=True)
    with (TABLES / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(stage, relative, ids, fields, select=lambda r: True):
    for r in read(relative):
        if not select(r):
            continue
        selector = {k: r[k] for k in ids if k in r}
        n = next((r[k] for k in ["n_targets", "n_questions", "n_unique_ids", "n_query_length_inputs", "n"] if k in r), "1" if stage == 5 else "")
        for field, label, unit, low, high in fields:
            if field not in r:
                continue
            METRICS.append({
                "experiment_stage": stage,
                "source_selector": json.dumps(selector, ensure_ascii=False, sort_keys=True),
                "metric": label, "value": r[field], "unit": unit,
                "n_samples": n,
                "n_input_pairs": r.get("n_pairs", r.get("n_input_conditions", "")),
                "n_generations_per_parameter": r.get("n_generations_per_parameter", ""),
                "n_requests_per_method": r.get("n_requests_per_method", ""),
                "ci_low": r.get(low, "") if low else "",
                "ci_high": r.get(high, "") if high else "",
                "source_file": relative,
                "source_fields": ";".join(k for k in [field, low, high] if k),
                "note": "existing stage aggregate; sample scope in selector; F1 dataset/cell equal weighting follows source stage; no new generation, scoring or bootstrap",
            })


def main():
    s = "stages/"
    collect(1, s + "01_longbench_baseline/tables/baseline_quality.csv",
            ["dataset", "method"], [("f1_percent", "F1", "F1 x100", "ci_low_percent", "ci_high_percent")])
    collect(1, s + "01_longbench_baseline/tables/baseline_quality_paired.csv",
            ["dataset", "method_a", "method_b"], [("delta_b_minus_a_pp", "paired F1 difference", "percentage points", "ci_low_pp", "ci_high_pp")])
    collect(2, s + "02_parameter_exploration/tables/global_parameter_rankings.csv",
            ["scope", "rank", "T", "S"], [("f1_percent", "exploration selected F1", "F1 x100", "", ""), ("delta_vs_default_pp", "selected minus default", "percentage points", "", "")], lambda r: r["rank"] == "1")
    collect(3, s + "03_validation_rule/tables/validation_policy_scores.csv",
            ["scope", "policy"], [("f1_percent", "validation policy F1", "F1 x100", "ci_low_percent", "ci_high_percent"), ("delta_vs_selected_fixed_pp", "policy minus selected fixed", "percentage points", "delta_ci_low_pp", "delta_ci_high_pp")])
    collect(3, s + "03_validation_rule/tables/validation_cell_decisions.csv",
            ["scope", "L", "N", "candidate_T", "candidate_S", "selected_T", "selected_S"], [("delta_candidate_minus_fixed_pp", "cell candidate minus fixed", "percentage points", "delta_ci_low_pp", "delta_ci_high_pp")])
    collect(4, s + "04_length_check/tables/quality_paired.csv",
            ["scope", "L", "N"], [("fixed_f1_percent", "fixed F1", "F1 x100", "fixed_ci_low", "fixed_ci_high"), ("dynamic_f1_percent", "dynamic F1", "F1 x100", "dynamic_ci_low", "dynamic_ci_high"), ("difference_pp", "dynamic minus fixed", "percentage points", "difference_ci_low_pp", "difference_ci_high_pp")])
    collect(5, s + "05_attention/tables/answer_pairs.csv",
            ["dataset", "sample_id", "L", "N", "default_T", "default_S", "best_T", "best_S"], [("default_f1_0_1", "default F1 raw", "F1 0-1", "", ""), ("best_f1_0_1", "exploration candidate F1 raw", "F1 0-1", "", ""), ("difference_pp", "candidate minus default", "percentage points", "", "")])
    collect(6, s + "06_longbench_final/tables/main_quality.csv",
            ["scope", "dataset", "method"], [("f1_percent", "F1", "F1 x100", "ci_low_percent", "ci_high_percent")])
    collect(6, s + "06_longbench_final/tables/main_quality_paired.csv",
            ["scope", "dataset", "reference_method", "comparison_method"], [("delta_pp", "paired F1 difference", "percentage points", "ci_low_pp", "ci_high_pp")])
    collect(7, s + "07_litm/tables/main_quality.csv",
            ["position", "method", "score_metric"], [("em_percent", "official first-line EM", "EM x100", "ci_low_percent", "ci_high_percent")])
    collect(7, s + "07_litm/tables/main_quality_paired.csv",
            ["position", "reference_method", "comparison_method", "score_metric"], [("delta_pp", "paired first-line EM difference", "percentage points", "ci_low_pp", "ci_high_pp")])
    collect(8, s + "08_doc2dial/tables/main_quality.csv",
            ["scope", "cluster_unit", "method", "cache_mode"], [("corpus_bleu", "corpus BLEU", "BLEU 0-100", "ci_low", "ci_high")])
    collect(8, s + "08_doc2dial/tables/main_quality_paired.csv",
            ["scope", "cluster_unit", "reference_method", "reference_cache", "comparison_method", "comparison_cache"], [("delta_bleu", "paired corpus BLEU difference", "BLEU points", "ci_low", "ci_high")])
    for stage, relative in [(1, "01_longbench_baseline/tables/baseline_efficiency.csv"), (6, "06_longbench_final/tables/main_efficiency.csv"), (7, "07_litm/tables/efficiency_descriptive.csv"), (8, "08_doc2dial/tables/main_efficiency.csv")]:
        fields = [("mean", "request metric mean", "see selector and source unit", "ci_mean_low" if stage == 1 else "mean_ci_low", "ci_mean_high" if stage == 1 else "mean_ci_high")]
        collect(stage, s + relative, ["scope", "dataset", "position", "method", "cache_mode", "metric"], fields, lambda r: r["metric"] in {"ttft_seconds", "total_seconds", "output_tokens"})
    write("internal_briefing_metrics.csv", METRICS)

    checks = read(s + "01_audit/tables/construction_checks.csv")
    length_keys = {(r["sample_id"], int(r["L"])) for r in read(s + "04_length_check/tables/paired_records.csv")}
    missing_gen = read(s + "01_audit/tables/construction_missing_evidence_generations.csv")
    gen_counts = Counter((r["sample_id"], int(r["L"])) for r in missing_gen)
    usage = {}
    for r in checks:
        split, length = r["split"], int(r["L"])
        key = (r["sample_id"], length)
        used = length in {4096, 8192, 16384} or key in length_keys
        missing = int(r["n_missing_required_tokens"]) > 0
        x = usage.setdefault((split, length), Counter())
        x["n_frozen"] += 1
        x["n_used"] += used
        x["n_unused"] += not used
        x["n_missing_frozen"] += missing
        x["n_missing_used"] += missing and used
        x["n_missing_unused"] += missing and not used
        x["n_generations_using_missing"] += gen_counts[key]
    columns = ["n_frozen", "n_used", "n_unused", "n_missing_frozen", "n_missing_used", "n_missing_unused", "n_generations_using_missing"]
    usage_rows = [{"split": split, "L": length, **{k: x[k] for k in columns},
                   "unit": "construction=(sample_id,L); generation count separately labelled",
                   "source": "01_audit/tables/construction_checks.csv;04_length_check/tables/paired_records.csv;01_audit/tables/construction_missing_evidence_generations.csv",
                   "usage_rule": "all applicable explore/validation L4096/8192/16384; L6144/12288 only keys present in actual length_check paired_records"}
                  for (split, length), x in sorted(usage.items())]
    totals = {k: sum(x[k] for x in usage.values()) for k in columns}
    assert [totals[k] for k in columns] == [432, 278, 154, 80, 54, 26, 1605], totals
    usage_rows.append({**usage_rows[0], "split": "ALL", "L": "ALL", **totals})
    write("internal_briefing_construction_usage.csv", usage_rows)

    follow = [
        ("P0-1", "修正开发证据构造及候选验证", "54实际构造/1605生成受审核证据范围删除影响;完整筛选改变探索4格与验证1格", "同题同L/N/T/S原/修正输入;修正探索后再提名验证;长度最小12生成", "参数排名、固定/动态规则、长度迁移", "CPU offset及构造修复先行;新候选确定后列任务", "01_audit;02_parameter_exploration;03_validation_rule;04_length_check"),
        ("P0-2", "自然实际N开发及新增独立测试", "正式LongBench510均N1–3,LITM全N1,Doc2Dial493/500 N1", "7500上限自然N分层搜索验证;强制N同token机制实验另列;新规则冻结后新资料隔离测试", "动态规则在实际分布的质量收益", "修正开发构造;现有测试结果仅作旧规则结果", "04_length_check;06_longbench_final;07_litm;08_doc2dial"),
        ("P1-1", "停止及输出长度控制", "LITM1465 cap中1391已换行;Doc2Dial cap70%–76.2%,生成约参考5.6倍", "LITM首换行停止测时,已换行首行EM现有答案已确定;未换行完整配对增上限;Doc2Dial共用简洁模板对照", "冗长/截断的质量及时间贡献,方法排序", "冻结开发样本;记录首换行/逐token时标", "07_litm;08_doc2dial"),
        ("P1-2", "交错重复计时及路径拆分", "同输出APE仍慢;每方法仅连续worker;动态固定差达微秒/毫秒", "同输入同输出子集随机交错多轮;拆prepare/prefill/query/decode/选参;冷热独立", "APE开销来源及动态效率微小差异", "固定ID和缓存条件;下一轮同时记录实际GPU信息", "01_longbench_baseline;06_longbench_final;07_litm;08_doc2dial"),
        ("P2-1", "不同轮次连续热复用", "现有仅同目标立即重提交,热命中与重新生成已验证", "相同资料按原对话更新目标轮次;每轮cold重建vs仅资料KV常驻", "连续应用缓存效率及生命周期", "保持每轮历史/问题更新并重新生成", "08_doc2dial"),
        ("P2-2", "位置×实际N及最小注意力补存", "LITM三点末位最低且全N1;Qasper decode历史分歧;缺QKV/target向量", "同题同文档位置0/9/19强制N1/4/8/16;固定输出轨迹;选层头步补向量", "位置曲线机制及参数/生成历史效应,weighted-V独立复核", "先实际N控制,后密集位置点;复用已完成区域掩码", "05_attention;07_litm"),
        ("P3", "官方条件复现", "已有公开成绩/设置对照;缺表格逐题ID、检索文本及最终T/S", "材料齐后逐项完整文本vs检索块/默认vs验证调参;官方速度配置独立", "官方与本机方法优势差异来源", "先补材料再决定生成清单", "01_longbench_baseline"),
    ]
    follow_rows = [dict(zip(["priority", "experiment", "existing_question", "controlled_comparison", "affected_conclusion", "prerequisite", "evidence_stages"], row), execution_status="PLAN_ONLY_REQUIRES_SEPARATE_AUTHORIZATION") for row in follow]
    write("internal_briefing_follow_up_plan.csv", follow_rows)

    report = BASE / "reports/INTERNAL_BRIEFING_2026-11-17.md"
    content = report.read_text(encoding="utf-8")
    assert not re.search("不能说明|不能证明|不代表|不等于", content)
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
        if not target.startswith(("https://", "http://")):
            assert (report.parent / target).exists(), target
    for stage in range(1, 9):
        assert f"## {stage}." in content
    assert len(follow_rows) == 7
    validation = {"status": "PASS", "metric_rows": len(METRICS),
                  "construction_totals": totals, "follow_up_items": len(follow_rows),
                  "report_local_links": "all exist", "all_eight_stages_present": True,
                  "new_generations": 0, "new_score_computations": 0,
                  "research_status": "SUBMITTED", "follow_up_execution": "not authorized or started"}
    (BASE / "reports/INTERNAL_BRIEFING_VALIDATION.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
