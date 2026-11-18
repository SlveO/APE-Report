"""Targeted checks of delivered denominators, intervals, source labels and links."""
import ast
import csv
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]


def rows(name):
    with (OUT/"tables"/name).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def main():
    expected={"a_validation_records.csv":839,"a_validation_logical_views.csv":900,
              "b_records.csv":120,"b_paired.csv":60,"b_membership.csv":10,
              "c_membership.csv":150,"c_logical_answers.csv":1050,
              "c_physical_generations.csv":897,"c_score_recheck.csv":1050,
              "c_quality.csv":56,"c_paired.csv":168,"c_support_components.csv":68,
              "c_cases.csv":3,"headline_results.csv":3,"figure_index.csv":10}
    observed={name:len(rows(name)) for name in expected}
    assert observed==expected,(observed,expected)
    for r in rows("c_score_recheck.csv"):
        assert float(r["absolute_difference"]) < 1e-12
    members=rows("c_membership.csv")
    for ds,count in zip(["qasper","hotpotqa","musique"],[15,46,7]):
        rr=[r for r in members if r["dataset"]==ds]
        assert len(rr)==50 and len({r["support_component"] for r in rr})==count
    assert sum(r["same_physical_generation"]=="True" for r in members)==133
    q=rows("c_quality.csv")
    for method in {r["method"] for r in q}:
        main=next(float(r["f1_0_1"]) for r in q if r["scope"]=="macro_equal_dataset" and r["method"]==method)
        datasets=[float(r["f1_0_1"]) for r in q if r["scope"] in ["qasper","hotpotqa","musique"] and r["method"]==method]
        assert abs(main-sum(datasets)/3)<1e-12
    p=next(r for r in rows("c_paired.csv") if r["scope"]=="macro_equal_dataset" and r["reference_method"]=="ape_natural_fixed" and r["comparison_method"]=="ape_natural_dynamic")
    assert abs(float(p["delta_pp"])-.14222164357828228)<1e-10
    assert [int(p[k]) for k in ["wins","ties","losses"]]==[3,146,1]
    assert float(p["ci_low_pp"])<0<float(p["ci_high_pp"])
    for table in ["a_validation_policies.csv","a_validation_paired.csv","b_quality.csv","c_quality.csv","c_paired.csv"]:
        for r in rows(table):
            assert int(r["seed"])==20261116 and int(r["replicates"])==5000,table
    diag=next(r for r in rows("c_bootstrap_diagnostics.csv") if r["scope"]=="qasper_N_strata_and_equal_N")
    assert [int(diag[k]) for k in ["proposed","rejected_missing_N","effective"]]==[5069,69,5000]
    new_b=rows("b_records.csv")
    # B exposes its physical source as answer_origin, not a boolean flag.
    assert sum(r["answer_origin"]!="historical_exact_reuse" for r in new_b)==103
    assert sum(r["answer_origin"]=="historical_exact_reuse" for r in new_b)==17
    for r in new_b:
        if r["answer_origin"]=="historical_exact_reuse":
            assert all(r.get(k,"")=="" for k in ["total_seconds","ttft_seconds"])
        else:
            assert float(r["total_seconds"])>0 and float(r["ttft_seconds"])>0
    count=next(r for r in rows("logical_count_reconciliation.csv") if r["phase"]=="ALL")
    assert [int(count[k]) for k in ["physical_source_rows","logical_views","new_answers","historical_answers","extra_labels"]]==[2824,3042,2430,394,218]
    missing=[]
    for file in list((OUT/"reports").glob("*.md"))+[OUT/"README.md"]:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",file.read_text(encoding="utf-8")):
            if not target.startswith(("https://","http://")) and not (file.parent/target).exists():
                missing.append([str(file.relative_to(OUT)),target])
    assert not missing,missing
    png={p.stem for p in (OUT/"figures").glob("*.png")}
    svg={p.stem for p in (OUT/"figures").glob("*.svg")}
    expected_figs={r["figure_stem"] for r in rows("figure_index.csv")}
    assert png==svg==expected_figs and len(png)==10
    for file in (OUT/"scripts").glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"),filename=str(file))
    result={"status":"PASS","table_denominators":observed,"figures_png":len(png),"figures_svg":len(svg),
            "report_links_missing":missing,"primary_delta_pp":float(p["delta_pp"]),
            "primary_CI_pp":[float(p["ci_low_pp"]),float(p["ci_high_pp"])],
            "qasper_stratum_proposals":5069,"qasper_stratum_rejections":69,"qasper_stratum_effective":5000,
            "independent_review":"C main/strata Bootstrap and F1 independently reproduced; A and summary facts reviewed",
            "visual_review":"A/B figures reviewed by their authors; C main figures also inspected by primary and length-analysis agents",
            "physical_generation_sources":2824,"expanded_logical_views":3042,
            "new_PC_model_answers":0,"parameters_reselected":False,
            "historical_package_scripts_executed":False}
    (OUT/"logs/delivery_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status","figures_png","figures_svg","primary_delta_pp","primary_CI_pp"]},ensure_ascii=False))


if __name__=="__main__":main()
