"""Check delivered Doc2Dial denominators, scoring agreement and report links."""
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "stages/08_doc2dial"


def read_table(name):
    with (OUT / "tables" / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    checks = {}
    counts = {
        "membership.csv": 500,
        "target_inputs.csv": 500,
        "per_request_records.csv": 3000,
        "bleu_sufficient_statistics.csv": 1500,
        "preaudit_input_checks.csv": 500,
        "hot_generation_evidence.csv": 1500,
        "cold_hot_pairs.csv": 1500,
        "main_quality.csv": 6,
        "main_quality_paired.csv": 9,
        "main_efficiency.csv": 60,
        "generation_cap_records.csv": 2162,
        "selected_cases.csv": 3,
        "scorer_validation.csv": 5,
        "original_summary_crosscheck.csv": 6,
        "record_issues.csv": 0,
        "reference_language_candidates.csv": 1,
    }
    for name, expected in counts.items():
        actual = len(read_table(name))
        assert actual == expected, (name, actual, expected)
        checks[name] = actual
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert result["research_status"] == "SUBMITTED"
    assert not result["rule_modified"] and not result["next_stage_executed"]
    assert result["n_dialogues"] == 366 and result["n_documents"] == 172
    assert result["seed"] == 20261116 and result["replicates"] == 5000
    assert all(r["status"] == "PASS" for r in read_table("preaudit_input_checks.csv"))
    pairs = read_table("cold_hot_pairs.csv")
    assert all(r["status"] == "PASS" and r["same_tokens"] == "True"
               and r["same_answer"] == "True" and r["cold_miss_hot_hit"] == "True"
               and float(r["hot_build_seconds"]) == 0
               and float(r["hot_rebuild_seconds"]) == 0 for r in pairs)
    cross = read_table("original_summary_crosscheck.csv")
    assert all(float(r["absolute_difference"]) < 1e-8
               and r["stats_match_preaudit"] == "True" for r in cross)
    assert all(r["status"] == "PASS" for r in read_table("scorer_validation.csv"))
    quality = read_table("main_quality.csv")
    for method in {r["method"] for r in quality}:
        qs = [r for r in quality if r["method"] == method]
        assert len(qs) == 2 and len({r["corpus_bleu"] for r in qs}) == 1
    n2 = [r for r in read_table("quality.csv")
          if r["scope"] == "actual_N_2" and r["cluster_unit"] == "document"]
    assert len(n2) == 6 and all(r["n_clusters"] == "1"
                              and r["ci_low"] == r["ci_high"] == "" for r in n2)
    caps = read_table("generation_limits.csv")
    assert sum(int(r["cap_hits"]) for r in caps) == 2162
    assert sum(int(r["eos_last_count"]) for r in caps) == 840
    assert sum(int(r["eos_at_cap_count"]) for r in caps) == 2
    assert all(int(r["cap_hits"]) + int(r["eos_last_count"])
               - int(r["eos_at_cap_count"]) == 500 for r in caps)
    improvement = next(r for r in read_table("selected_cases.csv")
                       if r["sample_id"] == "4b3bb867f442586694d2824e9d26ed8b_10")
    assert improvement["fixed_output_tokens"] == "128"
    missing = []
    for path in OUT.rglob("*.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#")[0]
            if target and not re.match(r"https?://", target):
                if not (path.parent / target).exists():
                    missing.append([str(path.relative_to(OUT)), target])
    assert not missing, missing
    pngs = sorted(p.stem for p in (OUT / "figures").glob("*.png"))
    svgs = sorted(p.stem for p in (OUT / "figures").glob("*.svg"))
    assert len(pngs) == 8 and pngs == svgs
    root_readme = (BASE / "README.md").read_text(encoding="utf-8")
    assert root_readme.count("[实验8：Doc2Dial]") == 1
    payload = {
        "status": "PASS", "checked_table_denominators": checks,
        "server_aggregate_and_statistics_agreement": True,
        "cold_hot_generation_pairs_pass": len(pairs),
        "one_document_subgroup_CI_empty": True,
        "report_links_missing": missing, "figure_pairs": pngs,
        "visual_review": "Figures reviewed by plotting agent; corpus BLEU and cache speedups also inspected by primary agent.",
        "independent_review": "Protocol audit agent reviewed target pairing, source grouping and cluster-bootstrap aggregation.",
        "research_status": result["research_status"],
        "rule_modified": False, "next_stage_executed": False,
    }
    (OUT / "logs/validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "tables": len(counts),
                      "figure_pairs": len(pngs), "missing_links": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
