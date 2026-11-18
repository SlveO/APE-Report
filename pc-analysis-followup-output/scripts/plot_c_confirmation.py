"""Static figures of existing paired confirmation statistics; no inference rerun."""
import csv
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(OUT / "cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(OUT / "cache")
os.environ["TEMP"] = os.environ["TMP"] = str(OUT / "tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"sequential": "顺序", "pcw": "PCW", "ape_default": "默认APE",
          "ape_main80_fixed": "旧固定", "ape_main80_dynamic": "旧动态",
          "ape_natural_fixed": "新固定", "ape_natural_dynamic": "新动态"}
METHODS = list(LABELS)
COLORS = ["#4c78a8", "#72b7b2", "#b279a2", "#9d755d", "#bab0ac", "#59a14f", "#f28e2b"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 11,
                     "savefig.dpi": 170})


def read(name):
    with (OUT / "tables" / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    for ext in ["png", "svg"]:
        fig.savefig(OUT / "figures" / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    quality, paired = read("c_quality.csv"), read("c_paired.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, scope, title in zip(axes.flat, ["qasper", "hotpotqa", "musique", "macro_equal_dataset"],
                               ["Qasper：50题 / 15组件", "HotpotQA：50题 / 46组件", "MuSiQue：50题 / 7组件", "主指标：三数据集等权 / 150题"]):
        rr = [next(r for r in quality if r["scope"] == scope and r["method"] == m) for m in METHODS]
        values = np.array([float(r["f1_percent"]) for r in rr])
        lo, hi = np.array([float(r["ci_low_percent"]) for r in rr]), np.array([float(r["ci_high_percent"]) for r in rr])
        x = np.arange(7)
        ax.bar(x, values, color=COLORS, zorder=3)
        ax.errorbar(x, values, yerr=[values-lo, hi-values], fmt="none", ecolor="#222222", capsize=3, lw=1.1, zorder=4)
        ax.set(xticks=x, xticklabels=[LABELS[m] for m in METHODS], ylim=(0, 100), ylabel="F1 × 100", title=title)
        ax.tick_params(axis="x", labelrotation=20)
        for i,v in enumerate(values):
            ax.text(i, max(v,hi[i])+2, f"{v:.2f}", ha="center", fontsize=9)
        ax.grid(axis="y", color="0.9", zorder=0)
    fig.suptitle("独立确认集：七方法质量\n误差线为5,000次支撑资料组件配对Bootstrap的95%区间", fontsize=15)
    save(fig, "c_seven_method_quality")

    scopes = ["macro_equal_dataset", "qasper", "hotpotqa", "musique", "qasper_N1", "qasper_N2", "qasper_N3", "qasper_equal_N_supplement"]
    titles = ["主指标：三数据集等权，150题", "Qasper：50题 / 15组", "HotpotQA：50题 / 46组", "MuSiQue：50题 / 7组",
              "Qasper N1：17题 / 5组，同参共享", "Qasper N2：17题 / 6组，探索性", "Qasper N3：16题 / 4组，同参共享", "补充：Qasper三种N等权"]
    rr = [next(r for r in paired if r["scope"] == s and r["reference_method"] == "ape_natural_fixed" and r["comparison_method"] == "ape_natural_dynamic") for s in scopes]
    fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    y = np.arange(len(rr))[::-1]
    for i,r in enumerate(rr):
        v,lo,hi = [float(r[k]) for k in ["delta_pp", "ci_low_pp", "ci_high_pp"]]
        color = "#d95f02" if i==0 else "#4c78a8"
        ax.plot([lo,hi], [y[i],y[i]], color=color, lw=2)
        ax.scatter([v],[y[i]],color=color,s=50,zorder=4)
        ax.text(6.25,y[i],f"{v:+.3f} [{lo:.3f}, {hi:.3f}]",va="center",fontsize=10)
    ax.axvline(0,color="0.45",lw=1,ls="--")
    ax.set(yticks=y,yticklabels=titles,xlim=(-3,10),xlabel="新版动态 − 新版固定（F1百分点）",
           title="主要比较与Qasper分层\n主抽样不按N缺层拒绝；分层使用独立的完整N抽样流")
    ax.grid(axis="x",color="0.93")
    save(fig,"c_primary_and_strata")

    components = [r for r in read("c_support_components.csv") if r["dataset"] == "qasper" and json.loads(r["N_values"]) == [2]]
    components.sort(key=lambda r:r["support_component"])
    contributions = np.array([float(r["dynamic_minus_fixed_sum_0_1"])*100/150 for r in components])
    fig,ax = plt.subplots(figsize=(11,5.5),constrained_layout=True)
    labels = [f"组件{i+1}（{r['n_questions']}题）" for i,r in enumerate(components)]
    ax.barh(np.arange(len(components)),contributions,color=["#59a14f" if v>=0 else "#e15759" for v in contributions])
    for i,v in enumerate(contributions):
        ax.text(v + (.004 if v>=0 else -.004),i,f"{v:+.4f}",ha="left" if v>=0 else "right",va="center")
    ax.axvline(0,color="0.4",lw=1)
    ax.set(yticks=np.arange(len(components)),yticklabels=labels,xlim=(-.14,.29),
           xlabel="对150题三数据集等权宏F1差的贡献（百分点）",
           title="收益与损失仅来自Qasper N2的三个组件\n六个N2组件合计贡献+0.1422；组件对应ID见图数据表")
    ax.grid(axis="x",color="0.93")
    save(fig,"c_component_contributions")
    outrows = [{**r,"component_plot_label":labels[i],"overall_macro_contribution_pp":contributions[i],
                "formula":"sum(question F1 differences) / 150 * 100; each dataset has50 questions"} for i,r in enumerate(components)]
    with (OUT/"tables/c_component_figure_data.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(outrows[0]));writer.writeheader();writer.writerows(outrows)
    print("C figures: 3 PNG + 3 SVG; component figure source table written.")


if __name__=="__main__":
    main()
