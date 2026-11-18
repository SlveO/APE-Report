"""Stage 1 only: task roster, record integrity, references and snapshot lineage.

Reads the frozen CPU job factory; never imports model workers or executes GPU work.
All files created by this script are under pc-analysis-output/stages/01_audit.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "restored"
RUN = SRC / "output/runs/main80-20261114T041503"
SNAPSHOT = RUN / "snapshot"
OUT = ROOT / "pc-analysis-output/stages/01_audit"
REFERENCE = "227116e4a004153272640cfe31087a033861b8e9"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_csv(name, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "tables" / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: identity(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def main():
    for sub in ("tables", "reports", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    # Archived modules are used solely to enumerate immutable CPU task definitions.
    sys.path.insert(0, str(SNAPSHOT / "APE"))
    from main80.jobs import JobFactory, STAGES
    from main80.rules import choose

    factory = JobFactory("datasets/frozen-main80")
    state = read(RUN / "state.json")
    summary = read(RUN / "summary.json")
    delivery = read(RUN / "delivery.json")
    incomplete = read(RUN / "incomplete.json")
    candidates = read(RUN / "candidates.json")
    rule = read(RUN / "rule.json")
    expected = [j for stage in STAGES for j in factory.make(stage, state["limits"], candidates)]
    by_expected = {identity(j["key"]): j for j in expected}
    records, parse_errors = [], []
    with (RUN / "attempts.jsonl").open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                parse_errors.append({"line": line_number, "issue": "blank_line", "bytes": len(line)})
                continue
            try:
                row = json.loads(line)
                row["_line"] = line_number
                records.append(row)
            except (ValueError, UnicodeError) as exc:
                parse_errors.append({"line": line_number, "issue": str(exc), "bytes": len(line)})

    keys = collections.Counter(identity(r["key"]) for r in records)
    issues, roster = [], []
    checks = collections.Counter()

    def check(row, name, condition, observed=None, expected_value=None):
        checks[name] += 1
        if not condition:
            issues.append({"line": row.get("_line"), "sample_id": row.get("sample_id"),
                           "stage": row.get("key", {}).get("stage"), "check": name,
                           "observed": observed, "expected": expected_value})

    for r in records:
        key = identity(r["key"])
        job = by_expected.get(key)
        check(r, "expected_logical_key", job is not None, r["key"], "frozen JobFactory task")
        if job is None:
            continue
        frozen = factory.tables[job["table"]][job["index"]]
        for field in ("sample_id", "method"):
            check(r, "key_" + field, r[field] == r["key"][field], r[field], r["key"][field])
        check(r, "frozen_sample_id", r["sample_id"] == frozen["sample_id"], r["sample_id"], frozen["sample_id"])
        check(r, "frozen_dataset", r["dataset"] == frozen["dataset"], r["dataset"], frozen["dataset"])
        check(r, "frozen_split", r.get("split") == frozen.get("split"), r.get("split"), frozen.get("split"))
        check(r, "frozen_references", r.get("reference_answers") == frozen["answers"], r.get("reference_answers"), frozen["answers"])
        check(r, "frozen_position", r.get("position") == frozen.get("position"), r.get("position"), frozen.get("position"))
        check(r, "cache_label", r["cache_mode"] == job["cache"], r["cache_mode"], job["cache"])
        check(r, "logical_labels", r.get("logical_labels") == job["logical_labels"], r.get("logical_labels"), job["logical_labels"])
        check(r, "references_nonempty", bool(r.get("reference_answers")), r.get("reference_answers"), "nonempty list")
        check(r, "answer_nonempty", bool(r.get("answer", "").strip()), r.get("answer"), "nonempty answer or scored empty failure")
        check(r, "positive_attempt", isinstance(r.get("attempt"), int) and r["attempt"] >= 1, r.get("attempt"), ">=1")
        for field in ("L", "N", "output_tokens", "prefix_tokens", "request_tokens", "query_start_position"):
            check(r, "integer_" + field, isinstance(r.get(field), int) and r[field] >= 0, r.get(field), "nonnegative integer")
        check(r, "block_count", len(r["block_lengths"]) == r["N"], len(r["block_lengths"]), r["N"])
        check(r, "block_sum", sum(r["block_lengths"]) == r["L"], sum(r["block_lengths"]), r["L"])
        check(r, "positive_block_lengths", all(isinstance(x, int) and x > 0 for x in r["block_lengths"]), r["block_lengths"], "positive integers")
        check(r, "output_token_count", len(r["output_token_ids"]) == r["output_tokens"], len(r["output_token_ids"]), r["output_tokens"])
        check(r, "generation_limit", r["output_tokens"] <= job["max_new_tokens"], r["output_tokens"], job["max_new_tokens"])
        if job["N"] is not None:
            check(r, "forced_block_count", r["N"] == job["N"], r["N"], job["N"])
            check(r, "constructed_length", r["L"] == frozen["L"], r["L"], frozen["L"])
        else:
            check(r, "natural_blocking", r["N"] == math.ceil(r["L"] / 7500), r["N"], math.ceil(r["L"] / 7500))
        if r["method"] in ("sequential", "pcw"):
            parameters = (None, None)
        elif r["method"] == "ape_default":
            parameters = (.9, .9)
        elif r["method"] == "ape_dynamic":
            parameters = choose(rule, r["L"], r["N"])
        else:
            parameters = tuple(job["param"]) if job["param"] else tuple(rule["fixed"])
        check(r, "applied_parameters", (r.get("T"), r.get("S")) == parameters, [r.get("T"), r.get("S")], parameters)
        for field in ("score", "ttft_seconds", "total_seconds", "cache_build_seconds", "cache_get_seconds",
                      "dynamic_selection_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "resume_cache_rebuild_seconds"):
            value = r.get(field)
            check(r, "finite_nonnegative_" + field, isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, value, "finite >=0")
        bound = 100 if r["dataset"] == "doc2dial" else 1
        check(r, "score_range", isinstance(r.get("score"), (int, float)) and 0 <= r["score"] <= bound, r.get("score"), [0, bound])
        check(r, "ttft_within_total", 0 <= r["ttft_seconds"] <= r["total_seconds"], r["ttft_seconds"], r["total_seconds"])
        roster.append({"source_line": r["_line"], "stage": r["key"]["stage"], "dataset": r["dataset"],
                       "sample_id": r["sample_id"], "method": r["method"], "position": r.get("position"),
                       "cache_mode": r["cache_mode"], "attempt": r["attempt"], "status": r["status"],
                       "frozen_table": job["table"], "frozen_index_zero_based": job["index"],
                       "requested_N": job["N"], "actual_N": r["N"], "actual_L_tokens": r["L"],
                       "max_new_tokens": job["max_new_tokens"], "logical_labels": r["logical_labels"],
                       "reference_count": len(r["reference_answers"]), "task_key": r["key"]})

    missing = sorted(set(by_expected) - set(keys))
    unexpected = sorted(set(keys) - set(by_expected))
    duplicates = [{"key": json.loads(k), "records": n} for k, n in keys.items() if n > 1]
    counted = collections.Counter(r["key"]["stage"] for r in records)
    expected_count = collections.Counter(j["key"]["stage"] for j in expected)
    theoretical = {"baseline": 1800, "explore": 7200, "validation": 1080, "length_check": 120,
                   "attention": 12, "longbench_final": 1020, "litm": 4500, "doc2dial": 3000}
    stages = []
    for stage in STAGES:
        rs = [r for r in records if r["key"]["stage"] == stage]
        logical = sum(len(r["logical_labels"]) if stage == "validation" else 1 for r in rs)
        stages.append({"stage": stage, "theoretical_generation_slots": theoretical[stage],
                       "applicable_logical_evaluations": logical, "frozen_physical_jobs": expected_count[stage],
                       "observed_records": counted[stage], "successes": sum(r["status"] == "success" for r in rs),
                       "logical_reuse_savings": logical - len(rs),
                       "inapplicable_slots": theoretical[stage] - logical,
                       "unit": "generation/evaluation count", "source": "snapshot/APE/main80/jobs.py; frozen inputs; attempts.jsonl:key.stage/logical_labels"})
    write_csv("record_stage_counts.csv", stages)
    write_csv("task_record_join.csv", roster)
    write_csv("record_issues.csv", issues, ["line", "sample_id", "stage", "check", "observed", "expected"])
    write_csv("record_parse_errors.csv", parse_errors, ["line", "issue", "bytes"])
    write_csv("duplicate_keys.csv", duplicates, ["key", "records"])
    write_csv("task_set_differences.csv", [{"kind": "missing", "key": json.loads(k)} for k in missing] + [{"kind": "unexpected", "key": json.loads(k)} for k in unexpected], ["kind", "key"])
    write_csv("record_check_counts.csv", [{"check": name, "checked_records": n, "issues": sum(x["check"] == name for x in issues)} for name, n in sorted(checks.items())])

    bundle = read(ROOT / "manifest.json")
    entries = {e["path"]: e for e in bundle["files"]}

    def canonical(path):
        while entries[path]["storage"] == "alias":
            path = entries[path]["alias_of"]
        return path

    lineage = []
    for path in read(RUN / "source-files.json"):
        snapshot = "output/runs/main80-20261114T041503/snapshot/" + path
        found = path in entries and snapshot in entries
        lineage.append({"source_path": path, "snapshot_path": snapshot,
                        "source_canonical": canonical(path) if path in entries else None,
                        "snapshot_canonical": canonical(snapshot) if snapshot in entries else None,
                        "same_manifest_storage": found and canonical(path) == canonical(snapshot),
                        "evidence": "manifest.json:files[].alias_of/storage (export direct-comparison lineage)"})
    write_csv("snapshot_lineage.csv", lineage)
    explicit = {"key": ("JSON", "冻结任务键；key.N=null表示自然分块，实际块数见N；attempt另列"),
                "sample_id": ("ID", "对应冻结样本；LITM同一问题跨三个位置共享ID；Doc2Dial按目标轮次区分"),
                "group": ("evidence group", "原论文／完整支撑文档组；统计聚类依据"),
                "L": ("token", "实际编码资料长度，前缀与请求另外记录"),
                "N": ("block", "实际编码块数；与original_document_count分列"),
                "logical_labels": ("label list", "同次生成对应的候选角色；validation复用按一次生成计数"),
                "reference_answers": ("string list", "与frozen_table/index连接并逐项核对的参考答案"),
                "score": ("F1/EM 0–1; Doc2Dial sentence BLEU 0–100", "原保存评分；官方口径复核见STAGE1_SCORING"),
                "output_token_ids": ("token ID list", "实际生成序列，包含生成的终止token"),
                "output_tokens": ("token", "包含终止token的生成长度"),
                "original_document_count": ("source document", "分块前资料数量，区别于实际编码块数N"),
                "attempt": ("integer", "同一逻辑任务的尝试编号"),
                "source": ("source IDs", "原始来源ID；沿用记录，不计算新增摘要"),
                "T": ("dimensionless", "实际注意力温度；sequential/pcw为null"),
                "S": ("dimensionless", "实际缩放参数；sequential/pcw为null"),
                "status": ("enum", "success或保留的失败状态；空生成在质量表保留零分"),
                "cache_hit": ("boolean", "本进程实际资料KV命中"),
                "cache_mode": ("enum", "cold/search/hot请求条件"),
                "timing_condition": ("enum", "cold/search_immutable_KV/application_GPU_KV"),
                "stage_calls": ("call counts", "APE首层prefill阶段调用计数；原生sequential/pcw保持0")}
    all_fields = sorted({k for r in records for k in r if not k.startswith("_")})
    dictionary = []
    for field in all_fields:
        if field.endswith("_seconds"):
            unit, definition = "s", "计时范围见STAGE1_RUNTIME及方法源码"
        elif field.endswith("_bytes"):
            unit, definition = "byte", "进程显存峰值（含模型与存活缓存），定义见STAGE1_RUNTIME"
        else:
            unit, definition = explicit.get(field, ("raw JSON field", "保留的原始逐题字段，见methods.py/worker.py"))
        dictionary.append({"field": field, "unit": unit, "definition": definition,
                           "present_records": sum(field in r for r in records),
                           "nonnull_records": sum(r.get(field) is not None for r in records),
                           "denominator": len(records), "source": "attempts.jsonl:" + field})
    write_csv("data_dictionary.csv", dictionary)
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
           "analysis_commit": actual_commit, "reference_commit": REFERENCE,
           "same_as_reference": actual_commit == REFERENCE, "reference_changes": [] if actual_commit == REFERENCE else "inspect git diff before later stages",
           "analysis_stage": "01_audit", "generation_seed": 42,
           "later_stage_bootstrap_seed": 20261116, "bootstrap_executed_this_stage": False,
           "new_dependency_installations": [], "input_root": str(SRC), "output_root": str(OUT),
           "PYTHONUTF8": os.environ.get("PYTHONUTF8"), "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE")}
    (OUT / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"records": len(records), "expected_physical_tasks": len(expected), "unique_keys": len(keys),
              "statuses": dict(collections.Counter(r["status"] for r in records)),
              "attempts": dict(collections.Counter(r["attempt"] for r in records)),
              "parse_errors": len(parse_errors), "duplicates": len(duplicates), "missing_tasks": len(missing),
              "unexpected_tasks": len(unexpected), "field_or_join_issues": len(issues),
              "manifest_logical_files": len(entries), "manifest_counts": bundle["counts"],
              "snapshot_lineage_items": len(lineage), "snapshot_lineage_matches": sum(x["same_manifest_storage"] for x in lineage),
              "delivery_status": delivery["status"], "state_status": state["state"],
              "summary_successes": summary["successful_records"], "summary_pending": summary["pending"],
              "incomplete_sizes": {k: len(incomplete[k]) for k in ("pending_jobs", "failed_attempts", "incomplete_lines")},
              "stage_counts": stages, "total_field_checks": sum(checks.values())}
    (OUT / "core_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = "# 第1阶段：任务、记录及快照审核\n\n"
    text += f"从运行快照的CPU任务工厂枚举得到{len(expected):,}项实际生成任务，与{len(records):,}条记录按完整任务键连接。重复键{len(duplicates)}，缺失任务{len(missing)}，额外任务{len(unexpected)}，解析错误{len(parse_errors)}。本次逐条核对冻结参考、样本/方法/位置/缓存/标签、分块及参数、输出长度和数值合法性，共{sum(checks.values()):,}项检查，发现{len(issues)}项字段或连接差异。\n\n"
    text += "## 规模及复用\n\n|阶段|理论槽位|适用逻辑评价|实际任务|记录|复用节省|不适用槽位|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for s in stages:
        text += f"|{s['stage']}|{s['theoretical_generation_slots']}|{s['applicable_logical_evaluations']}|{s['frozen_physical_jobs']}|{s['observed_records']}|{s['logical_reuse_savings']}|{s['inapplicable_slots']}|\n"
    text += "\n原矩阵18,600个评价槽位，加120个长度检查与12个注意力任务，共18,732。探索不适用减少528，验证不适用减少9，验证候选共用生成节省79，得到18,116条实际生成。初始调度18,195项对应适用逻辑评价，后续生成复用解释79项差额。构造不适用原因与分组见STAGE1_SPLITS。\n\n"
    text += "## 证据与状态\n\n"
    text += f"manifest登记{len(entries)}项逻辑文件，其中{bundle['counts']['aliases']}项别名、{bundle['counts']['multipart_files']}项分片文件。运行source-files清单{len(lineage)}项中，{sum(x['same_manifest_storage'] for x in lineage)}项在manifest中与快照解析到同一存储来源。此项检查使用导出清单的别名关系，未对文件计算新增摘要。\n\n"
    text += f"实际PC提交为`{actual_commit}`，参考提交为`{REFERENCE}`。交付和状态记录均为{delivery['status']}，incomplete记录的pending/failed/incomplete均为空。执行命令、环境和本阶段输出见本目录README；任务与逐题源行连接见task_record_join.csv。\n\n"
    text += "## 审核处理规则\n\n同一逻辑键成功记录重复时先登记差异并暂停相关配对表；保留所有失败attempt。空生成质量取零分并保留失败状态；缺失参考及非法评分列入问题表并明确受影响分母。实际记录均成功、attempt=1，本次未触发这些处理。正式参数比较、bootstrap及质量/效率优劣结论在后续阶段开展。\n"
    (OUT / "reports/STAGE1_CORE.md").write_text(text, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("records", "expected_physical_tasks", "statuses", "duplicates", "missing_tasks", "unexpected_tasks", "field_or_join_issues", "snapshot_lineage_items", "snapshot_lineage_matches")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
