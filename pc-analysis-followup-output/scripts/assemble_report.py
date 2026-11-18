"""Assemble the Chinese follow-up briefing from completed CPU statistics."""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parent
PKG = REPO / "restored/followup2604/FOLLOWUP2604_ACCEPTED"
os.environ["MPLCONFIGDIR"] = str(OUT / "cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(OUT / "cache")
os.environ["TEMP"] = os.environ["TMP"] = str(OUT / "tmp")
METHODS = ["sequential", "pcw", "ape_default", "ape_main80_fixed", "ape_main80_dynamic", "ape_natural_fixed", "ape_natural_dynamic"]
NAMES = dict(zip(METHODS,["顺序编码","PCW","默认APE","MAIN80固定规则","MAIN80动态规则","新版自然固定","新版自然动态"]))


def load(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def table(name):
    with (OUT/"tables"/name).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write_table(name, rows):
    with (OUT/"tables"/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def md(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers)]+["| "+" | ".join(str(x).replace("|","/") for x in row)+" |" for row in rows])+"\n"


def interval(low, high, digits=4):
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"


def main():
    a,b,c=load("a_results.json"),load("b_results.json"),load("c_results.json")
    quality,paired=table("c_quality.csv"),table("c_paired.csv")
    a_primary=next(r for r in a["validation_paired"] if r["reference"]=="fixed" and r["comparison"]=="dynamic")
    b_primary=b["main_equal_dataset_result"]
    c_primary=c["primary_comparison"]
    headline=[
        {"phase":"A corrected validation","n_questions":40,"n_input_pairs":300,"fixed_F1_percent":a["validation_macro"]["fixed"]["f1_percent"],"dynamic_F1_percent":a["validation_macro"]["dynamic"]["f1_percent"],"delta_pp":a_primary["delta_pp"],"ci_low_pp":a_primary["ci_low_pp"],"ci_high_pp":a_primary["ci_high_pp"],"wins":14,"ties":273,"losses":13,"rule_family":"corrected-nine-cell-v1","interpretation":"selection-use validation; dataset/cell equal","source":"a_validation_policies.csv;a_validation_paired.csv"},
        {"phase":"B length transfer","n_questions":10,"n_input_pairs":60,"fixed_F1_percent":b_primary["fixed_F1_percent"],"dynamic_F1_percent":b_primary["dynamic_F1_percent"],"delta_pp":b_primary["delta_pp"],"ci_low_pp":b_primary["delta_ci_low_pp"],"ci_high_pp":b_primary["delta_ci_high_pp"],"wins":5,"ties":53,"losses":2,"rule_family":"corrected-nine-cell-v1","interpretation":"frozen validation questions at new lengths; dataset/cell equal","source":"b_quality.csv"},
        {"phase":"C independent confirmation","n_questions":150,"n_input_pairs":150,"fixed_F1_percent":next(r["f1_percent"] for r in c["method_means"] if r["method"]=="ape_natural_fixed"),"dynamic_F1_percent":next(r["f1_percent"] for r in c["method_means"] if r["method"]=="ape_natural_dynamic"),"delta_pp":c_primary["delta_pp"],"ci_low_pp":c_primary["ci_low_pp"],"ci_high_pp":c_primary["ci_high_pp"],"wins":3,"ties":146,"losses":1,"rule_family":"natural-chunk-v2","interpretation":"primary; question mean in each dataset then equal3 datasets","source":"c_quality.csv;c_paired.csv"},
    ]
    for r in headline:r.update(unit="F1 x100; difference percentage points",seed=20261116,replicates=5000)
    write_table("headline_results.csv",headline)
    phases=defaultdict(lambda:Counter())
    with (PKG/"logical-provenance.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);x=phases[r["phase"]]
            x["physical_source_rows"]+=1;x["logical_views"]+=len(r["logical_labels"])
            x["new_answers"]+=bool(r["new_generation"]);x["historical_answers"]+=not r["new_generation"]
    ledger=[]
    for phase,x in sorted(phases.items()):
        ledger.append({"phase":phase,**dict(x),"extra_labels":x["logical_views"]-x["physical_source_rows"],
                       "source":"logical-provenance.jsonl:phase,new_generation,logical_labels",
                       "unit":"source containers versus expanded logical views; no extra independent timing"})
    totals={k:sum(r[k] for r in ledger) for k in ["physical_source_rows","logical_views","new_answers","historical_answers","extra_labels"]}
    assert totals=={"physical_source_rows":2824,"logical_views":3042,"new_answers":2430,"historical_answers":394,"extra_labels":218},totals
    ledger.append({"phase":"ALL",**totals,"source":"logical-provenance.jsonl; FINAL-REVIEW157 is C2+C3 subset", "unit":"A61+C2 4+C3 153=218 extra labels"})
    write_table("logical_count_reconciliation.csv",ledger)
    figure_sources = [
        ("a_exploration_decomposition", "a_exploration_scores.csv", "old_all/old_common/corrected fixed-parameter means"),
        ("a_rule_changes", "a_rule_changes.csv", "L,N and old/corrected saved candidates/frozen parameters"),
        ("a_validation_policies", "a_validation_policies.csv", "frozen-policy F1 and confidence intervals"),
        ("a_natural_rule", "a_natural_validation.csv;a_natural_development.csv", "saved actual-N candidate/fixed/selected parameters"),
        ("b_length_quality", "b_quality.csv", "dataset-equal F1 in each L/N"),
        ("b_length_delta", "b_quality.csv", "dynamic-fixed paired difference and percentile interval"),
        ("b_historical_membership", "b_old_new_membership.csv", "old/new/shared question membership"),
        ("c_seven_method_quality", "c_quality.csv", "three dataset means and equal-dataset primary macro"),
        ("c_primary_and_strata", "c_paired.csv", "primary/unconditional and Qasper-strata conditional intervals"),
        ("c_component_contributions", "c_component_figure_data.csv", "Qasper N2 component sums / 150 * 100"),
    ]
    write_table("figure_index.csv", [{"figure_stem":s,"formats":"PNG;SVG","table_sources":t,"meaning":m} for s,t,m in figure_sources])

    p=lambda ref,comp,scope="macro_equal_dataset":next(r for r in paired if r["scope"]==scope and r["reference_method"]==ref and r["comparison_method"]==comp)
    adverse=[p("ape_default","ape_natural_dynamic"),p("ape_main80_fixed","ape_natural_dynamic"),p("ape_main80_fixed","ape_natural_fixed")]
    write_table("c_adverse_reference_comparisons.csv",adverse)
    macro_rows=[]
    for method in METHODS:
        r=next(r for r in quality if r["scope"]=="macro_equal_dataset" and r["method"]==method)
        vals=[next(float(x["f1_percent"]) for x in quality if x["scope"]==ds and x["method"]==method) for ds in ["qasper","hotpotqa","musique"]]
        macro_rows.append([NAMES[method]]+[f"{v:.4f}" for v in vals]+[f"{float(r['f1_percent']):.4f}",interval(r['ci_low_percent'],r['ci_high_percent'])])
    report="""# FOLLOWUP2604 中文分析总结（供服务器端复核）

## 本次结论

我们已完成验收包恢复、修正选参核对、6K/12K迁移和150题独立确认的CPU分析。修正输入后，九格轨道固定参数从0.7/1.0变为0.5/1.0，最终九格6/9改变；自然轨道另冻结固定1.0/0.9，只有N2采用动态0.9/1.0。

长度迁移主差为+2.0688个F1百分点，95%区间[-2.3551,7.0006]，仍有退化条件。独立确认主差为+0.1422个百分点，区间[-0.2731,0.6848]，3胜146平1负。现有结果是小幅正向点估计，总体可靠收益尚未确立，跨零区间也未构成等效结论。

需要保留的不利结果：在同一150题上，新版自然动态比默认APE低3.3786个百分点、比MAIN80固定规则低3.6188个百分点，两项逐比较区间均为负。修复与重新冻结改变了参数，未带来全面优于参照方法的确认结果。

## 1 材料、规则与分母

本次分析使用服务器交付版本 `ab816f5`（`analysis/main80-pc-results`）。166项验收文件按清单恢复到 `restored/followup2604/FOLLOWUP2604_ACCEPTED/`。服务器计算与数据交付状态为 ACCEPTED，35 项检查全部通过。原始运行状态为 SUBMITTED，包内 MAIN80 整体状态为 REVISION_REQUIRED。

本轮FOLLOWUP物理来源2824项=2430新增+394历史复用。完全展开共有3042逻辑视图，额外标签218=A61+C2 4+C3 153。验收记录的157覆盖C2+C3；A的61个额外候选标签另在其839个容器的logical_labels内。同参标签共享一次生成，样本和计时均按物理源连接。更早修正探索的1248条新回答由provenance带入，未计入FOLLOWUP新增2430。

三套规则分别为旧main80-nine-cell-v1、修正corrected-nine-cell-v1、自然natural-chunk-v2。A/B评价修正九格，C评价自然规则，并在同一确认集重新执行旧规则作参照。旧510题均值及旧LITM/Doc2Dial结果保留为历史范围。

"""
    report+=md(["范围","独立题数","输入配对数","固定F1","动态F1","差/百分点","95%区间","胜/平/负"],[[r['phase'],r['n_questions'],r['n_input_pairs'],f"{r['fixed_F1_percent']:.4f}",f"{r['dynamic_F1_percent']:.4f}",f"{r['delta_pp']:+.4f}",interval(r['ci_low_pp'],r['ci_high_pp']),f"{r['wins']}/{r['ties']}/{r['losses']}"] for r in headline])
    report+="""
A为选参所用验证集，B使用固定验证题的新长度，C为独立确认。三行分别评价不同范围，分数差只在各自行内配对。

## 2 修正是否改变选参

冻结开发构造432→359：73份退出适用清单、98份实际生成输入变化、261份输入一致。实际探索139→111题长，验证119→100题长。修正字符定位改为token ID与ByteLevel字节还原后连接原UTF-8字节，替代原native offset定位；构造按已核实证据和负证据背景审核条件判断适用。退出项逐ID清单已保存，但逐项退出原因字段缺失；73份各自属于哪种退出原因仍待服务器补字段。

我们读取保存候选与规则，未重选参数。探索提名4/9改变，固定0.7/1.0→0.5/1.0，最终九格6/9改变。为分开样本、输入和参数变化，固定旧全局参数0.7/1.0的探索得分依次为：旧全量41.1629；限定修正适用共同ID42.1221；修正输入同参数42.0641。前一变化来自适用样本构成，后一同参数输入差为-0.0580个百分点，95%区间[-1.6027,1.5288]。在同一修正输入改用已保存新候选0.5/1.0后为42.1856，点估计比0.7/1.0高0.1215。

因此，跨版总体分数变化同时包含样本退出、输入变化和参数变化。修正后的参数选择确有变化；把版本均值差全部计为算法收益会混入前两项。

"""
    report+=md(["L","N","旧冻结T/S","修正冻结T/S"],[[r['L'],r['N'],r['old_frozen_rule'],r['corrected_frozen_rule']] for r in table('a_rule_changes.csv')])
    report+="""
A验证覆盖40题/300条件，固定42.4017、动态43.1724，差+0.7706，区间[-0.9926,2.5739]。这些记录参加了冻结选择，区间对应固定策略下来源构成变化。自然开发另用54探索题/864回答、36验证题/104物理回答；固定1.0/0.9，N1/N3回退固定，N2候选0.9/1.0在4个验证题上高0.8164个百分点。该保存选择早于C3生成。

![探索分解](../figures/a_exploration_decomposition.png)

![规则变化](../figures/a_rule_changes.png)

## 3 长度迁移是否仍退化

新版B：10题，Qasper/HotpotQA/MuSiQue=4/3/3，各3个冻结支撑组；60个题长块条件形成120逻辑结果，103新增、17历史精确复用。主指标对数据集及六条件等权，固定39.7678、动态41.8366，差+2.0688，区间[-2.3551,7.0006]。

"""
    br=[r for r in table("b_quality.csv") if r["dataset"]=="macro_equal_dataset" and r["scope"]!="ALL"]
    report+=md(["L/N","固定F1","动态F1","差/百分点","95%区间","胜/平/负"],[[r['scope'],f"{float(r['fixed_F1_percent']):.3f}",f"{float(r['dynamic_F1_percent']):.3f}",f"{float(r['delta_pp']):+.3f}",interval(r['delta_ci_low_pp'],r['delta_ci_high_pp'],3),f"{r['wins']}/{r['ties']}/{r['losses']}"] for r in br])
    report+="""
6K/8有明显局部改善，6K/4及12K/4下降；两种长度的N16及12K/8同参持平。六格属于探索性分层，逐格95%区间未做多重比较校正。总体可靠改善尚未确立。

验收描述按60输入条件直接混合为固定41.5251、动态43.5730、差+2.0478；与主口径差别来自4/3/3题数权重，两个口径分别保存。旧长度检查与本次各10题，仅7题共同；旧-4.3959与新+2.0688分别属于各自样本和规则。共同7题/42条件已按输入与参数身份连接，17条同输入同参数复用分数完全相同；其余按改输入、改参数单列。新旧总均值差未记为修复的配对提升。

![长度迁移](../figures/b_length_delta.png)

## 4 新版动态是否有可靠独立收益

确认集各数据集50题，共150题，七方法1050逻辑结果来自897次实际生成。新版固定42.3108、新版动态42.4530，主差+0.1422，95%区间[-0.2731,0.6848]。同一批抽样共同保留所有方法，3胜146平1负。

"""
    report+=md(["方法","Qasper/50","HotpotQA/50","MuSiQue/50","等权宏F1","宏F1的95%区间"],macro_rows)
    report+="""
MAIN80两行是在本次150题上应用旧规则所得的服务器回答。与新版的差使用同样本配对，区别于旧510题的历史均值。

"""
    report+=md(["比较−参照","宏F1差/百分点","95%区间"],[[NAMES[r['comparison_method']]+' − '+NAMES[r['reference_method']],f"{float(r['delta_pp']):+.4f}",interval(r['ci_low_pp'],r['ci_high_pp'])] for r in adverse])
    report+="""
上述不利参照比较属于七方法逐比较区间，未做21对的多重校正；主要检验仍为预定的新版动态−新版固定。新版固定与新版动态在确认集均低于默认和MAIN80固定的点估计，验证优势未在这批独立确认中保留。具体哪些输入属性或参数机制导致该变化，原因未确定，缺独立机制对照；本次保留原冻结决策。

![七方法确认](../figures/c_seven_method_quality.png)

### Qasper分层与差异集中度

"""
    nrows=[]
    for n in [1,2,3]:
        scope=f"qasper_N{n}"
        r=p('ape_natural_fixed','ape_natural_dynamic',scope)
        f=next(x for x in quality if x['scope']==scope and x['method']=='ape_natural_fixed')
        d=next(x for x in quality if x['scope']==scope and x['method']=='ape_natural_dynamic')
        nrows.append([n,r['n_questions'],r['n_components'],f"{float(f['f1_percent']):.4f}",f"{float(d['f1_percent']):.4f}",f"{float(r['delta_pp']):+.4f}",interval(r['ci_low_pp'],r['ci_high_pp']),f"{r['wins']}/{r['ties']}/{r['losses']}"])
    report+=md(['N','题数','组件','固定F1','动态F1','差/百分点','95%区间','胜/平/负'],nrows)
    report+="""
Qasper整体差+0.4267，区间[-0.8193,2.0543]。N2局部差+1.2549，区间[-2.3605,5.7588]；仅6个组件，4个非零差题集中于3个组件，标为探索性。N1/N3及全部HotpotQA/MuSiQue的新版参数相同，共133题共享一次实际生成，质量逐token相同，零差来自这个执行事实。

Qasper N等权补充差+0.4183，区间[-0.7868,1.9196]，单独保存；主结果继续按Qasper50题的17/17/16题数权重。多块材料仅来自Qasper，未取得HotpotQA/MuSiQue新版多块结果。完整原始来源上下文与旧LongBench扩展文本分别报告。

![主比较与分层](../figures/c_primary_and_strata.png)

![来源组件贡献](../figures/c_component_contributions.png)

## 5 统计做法与推断范围

固定seed20261116、5000次有效配对聚类Bootstrap，百分位2.5%/97.5%，F1展示乘100，差值为百分点。

- A主验证按冻结支撑组Q/H/M=10/13/14抽样，保留同组全部题、L/N和策略，数据集×九格等权。主验证MuSiQue空单元重抽2次；同默认参数的跨版验证输入比较重抽3次，探索输入比较重抽0，各分析有效5000。scope110/111/112及其随机流分别保存在a_bootstrap_diagnostics.csv。
- B按自身冻结支撑组3/3/3抽样，全部六条件和两方法共同保留，按抽到题数归一，分层完整、重抽0。
- C在每数据集抽15/46/7个support_component，组件全部题和七方法共同纳入，每次先算数据集题均值，再对三数据集等权。主5000次不因Qasper缺N分层而拒绝。
- Qasper分层使用独立抽样流SeedSequence([20261116,1])，从其15组件整体抽取；缺任一N则重抽，提出5069、拒绝69、有效5000。该条件分布不替换总体主区间。

MuSiQue最大组件25/50题，Qasper N3仅4组件、N2仅6组件；绝对成绩及参照差的区间应随来源数阅读。新版动态−固定在HotpotQA/MuSiQue差恒0，主差的不确定性实际来自Qasper组件。区间反映这批冻结来源的组成变化，未包括重新选参、模型变化和重复运行的随机性；等效结论需要预先界定等效范围及相应设计。

## 6 效率和案例

效率仅描述现有物理生成。C的新版固定/动态平均总响应为1.2730/1.2550秒，平均输出11.873/11.533 token；133题共享同一次生成。每方法单一连续worker时段，当前资料缺交错重复运行。B只有固定60和动态43条新计时、43组成对新计时；17历史复用未补填时间。所有计时来源均可追溯物理键。

C全部逻辑结果均EOS结束，生成上限命中0/1050；对应897次物理生成。新版在这批确认材料上的不利质量结果伴随完整EOS结束，当前记录未显示上限截断。

典型C案例按主差最大改善、最大退化、零差最小ID选择，原答案/参考/配置/源行保存在c_cases.csv：

"""
    for r in table("c_cases.csv"):
        report+=f"- **{r['case_type']}**：`{r['sample_id']}`，N={r['natural_N']}，F1 {float(r['fixed_F1'])*100:.3f}→{float(r['dynamic_F1'])*100:.3f}。问题：{r['question']}\n\n  固定：{r['fixed_answer']}\n\n  动态：{r['dynamic_answer']}\n\n  参考：{r['reference_answers']}\n\n"
    report+="""
改善例中动态删除了参考未列出的额外方法和语料描述，token F1上升。退化例两份答案都未回答参考的“co-occurrence matrix低秩近似”；固定20分中的重合包含功能词，动态0分。我们保留该词面指标的真实分数，并连同原答案解释。

## 7 适合写入结题报告的结论

可以写：我们修正开发输入并补齐冻结候选验证，原全局/固定参数及六格规则发生改变；完整自然来源按实际N=1/2/3建立了独立开发与确认；新版动态在150题上相对新版固定点估计+0.1422个百分点，区间跨0，收益未达到可靠确认；同确认集上的默认APE和MAIN80固定规则表现更高，保留该不利结果。

长度迁移可以写：本次修正版本总体点估计转正，6K/8改善，6K/4和12K/4仍有退化；新旧题目和规则同时变化，分别报告其范围。A验证与B迁移的组件数、C的Qasper专属多块范围一起记录。

原MAIN80的开发证据问题已通过修正输入/退出不适用条件进入新分析，旧任务、原答案及旧冻结规则保留。旧LITM/Doc2Dial继续作为旧规则结果，新版在这两项任务的质量和热复用数据缺失。本批效率保留描述，稳定加速缺少重复计时证据。若继续研究，优先增加独立多块支撑资料、跨任务新版评价和交错重复运行；本次仅使用现有回答，未为正向结论追加生成或改规则。

## 8 复现与交付

复现命令和数据入口见[README](../README.md)。主表与分层表记录分母、单位、置信区间、种子与来源，可按阶段报告和独立复核记录追溯。[修正分析](A_CORRECTION.md) · [长度迁移](B_LENGTH.md) · [协议审核](PROTOCOL_AUDIT.md)
"""
    (OUT/"reports/FOLLOWUP_REPORT.md").write_text(report,encoding="utf-8")
    # Short separate C entry for server-side review.
    c_report=report[report.index("## 4 "):report.index("## 7 ")]
    (OUT/"reports/C_CONFIRMATION.md").write_text("# C阶段独立确认与统计\n\n"+c_report,encoding="utf-8")
    config={"seed":20261116,"replicates":5000,"interval":"paired cluster percentile 2.5%/97.5%",
            "A":{"unit":"frozen support group within dataset","groups":{"qasper":10,"hotpotqa":13,"musique":14},"aggregation":"equal dataset and nine L/N cells; same question all conditions retained","invalid_cell_redraws":{"musique":2}},
            "B":{"unit":"B frozen support group within dataset","groups":{"qasper":3,"hotpotqa":3,"musique":3},"aggregation":"question-weighted within dataset; six conditions together; equal datasets"},
            "C":{"unit":"samples.json support_component within dataset","groups":{"qasper":15,"hotpotqa":46,"musique":7},"aggregation":"question means within dataset then equal datasets","primary_rng":"default_rng(20261116), dataset order Q/H/M","primary_invalid_N_redraws":0},
            "Qasper_strata":{"rng":"SeedSequence([20261116,1])","proposed":5069,"missing_N_rejected":69,"effective":5000,"condition":"all N1/N2/N3 present","primary_distribution_unchanged":True},
            "efficiency":"descriptive physical-source timings only; no confidence interval or stable-speedup claim",
            "parameters_reselected":False,"new_model_answers":0,"historical_package_scripts_executed":False}
    config["A"]["all_scope_diagnostics"] = table("a_bootstrap_diagnostics.csv")
    config["A"]["scope_codes"] = {"110":"matched old/corrected exploration inputs at frozen parameters", "111":"A validation frozen-policy main analysis", "112":"matched old/corrected validation inputs at unchanged default parameters"}
    config["B"]["all_scope_diagnostics"] = table("b_bootstrap_diagnostics.csv")
    (OUT/"STATISTICAL_CONFIG.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    env=load("environment.json")
    import matplotlib
    env["matplotlib"]=matplotlib.__version__
    env["server_environment"] = json.loads((PKG/"server/environment.json").read_text(encoding="utf-8"))
    env["package_delivery_status"]="ACCEPTED";env["raw_run_status"]="SUBMITTED";env["MAIN80_overall_status"]="REVISION_REQUIRED"
    (OUT/"environment.json").write_text(json.dumps(env,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    index="""# FOLLOWUP2604 PC分析交付

本目录保存修正选参核对、6K/12K迁移和150题独立确认的CPU统计。入口为[中文总结](reports/FOLLOWUP_REPORT.md)；服务器端可先核对[主要结果](tables/headline_results.csv)、[C确认配对](tables/c_paired.csv)及[统计配置](STATISTICAL_CONFIG.json)。

主结论：修正九格固定从0.7/1.0改为0.5/1.0，6/9格冻结规则改变；自然固定1.0/0.9，动态仅N2改为0.9/1.0。B总体差+2.0688pp，95%CI[-2.3551,7.0006]；C主要比较+0.1422pp，CI[-0.2731,0.6848]，3胜146平1负。C新版动态低于默认及MAIN80固定的同题成绩，详见总结。不将跨零区间解释为等效。

## 文件结构

```text
pc-analysis-followup-output/
├── README.md                    # 本交付入口与复现
├── STATISTICAL_CONFIG.json      # 权重、组件、随机流、CI和重抽规则
├── environment.json             # 实际提交、环境及状态
├── a_results.json               # 修正探索/A验证与自然开发规则摘要
├── b_results.json               # B长度迁移摘要
├── c_results.json               # C独立确认摘要
├── scripts/                     # 自编CPU脚本，包内历史脚本不执行
├── tables/                      # a_/b_/c_主/分层CSV、来源、案例、诊断
├── figures/                     # 10组PNG/SVG及对应CSV数据
├── reports/                     # 中文总报告、A/B/C阶段及协议审核
├── logs/                        # 包恢复、分析配置与检查记录
├── cache/                       # 本地绘图缓存，Git忽略
└── tmp/                         # 本地临时目录，Git忽略
```

## 阅读与核对

- [A修正分析](reports/A_CORRECTION.md)：输入变化/适用变化/参数变化分开；[规则表](tables/a_rule_changes.csv)。
- [B长度迁移](reports/B_LENGTH.md)：10题60配对，自身冻结支撑组；[六格与数据集表](tables/b_quality.csv)。
- [C独立确认](reports/C_CONFIRMATION.md)：七方法；[主/分层质量](tables/c_quality.csv)、[配对CI](tables/c_paired.csv)、[组件成员](tables/c_support_components.csv)。
- [协议与独立复核](reports/PROTOCOL_AUDIT.md)：组件、来源、冻结顺序与标签分母。
- [逻辑计数对照](tables/logical_count_reconciliation.csv)：2824来源容器、3042视图；218额外标签中A61，C2+C3为157。
- [C重抽诊断](tables/c_bootstrap_diagnostics.csv)：主分析独立，Qasper分层5069提出/69拒绝/5000有效。
- [C不利参照比较](tables/c_adverse_reference_comparisons.csv)、[典型案例](tables/c_cases.csv)、[实际897次生成](tables/c_physical_generations.csv)。
- [交付检查](logs/delivery_validation.json)；[执行失败与处理记录](logs/failures.json)。检查脚本的B字段适配错误已修正，统计估计未受影响。
- [图表数据索引](tables/figure_index.csv)：10组PNG/SVG与源CSV对应。

## 输入与范围

本次分析使用服务器交付版本 **`ab816f5`**（`analysis/main80-pc-results`），该提交包含验收包和任务说明。166 项验收文件解压至 `restored/followup2604/FOLLOWUP2604_ACCEPTED/`，通过 `PORTABLE-PATHS.json` 读取包内相对路径。A 阶段和历史对照还读取原 MAIN80 的恢复证据及 `pc-analysis-output/`。[PC_TASK](../followup2604-20261117/PC_TASK.md)

分析以冻结规则和已有回答为输入，供服务器端复核。服务器计算与数据交付状态为 ACCEPTED，35 项验收检查全部通过。原始运行状态为 SUBMITTED，包内 MAIN80 整体状态为 REVISION_REQUIRED。

## 复现命令

从仓库根目录运行，使用既有Python3.12.7、NumPy1.26.4、Matplotlib3.9.2，无新增依赖。完整配置见environment.json、STATISTICAL_CONFIG.json及各分析日志。若旧MAIN80证据未恢复，先按仓库原有说明恢复后再执行A及B历史连接。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -s -B pc-analysis-followup-output/scripts/prepare_inputs.py
python -s -B pc-analysis-followup-output/scripts/analyze_a_correction.py
python -s -B pc-analysis-followup-output/scripts/analyze_b_length.py
python -s -B pc-analysis-followup-output/scripts/analyze_c_confirmation.py
python -s -B pc-analysis-followup-output/scripts/plot_c_confirmation.py
python -s -B pc-analysis-followup-output/scripts/assemble_report.py
python -s -B pc-analysis-followup-output/scripts/validate_delivery.py
```

A/B脚本同时生成其图表与阶段报告；所有新增脚本、表、图、缓存和日志均位于本目录。恢复脚本遇已有不同证据会保留现场并登记，不覆盖。
"""
    (OUT/"README.md").write_text(index,encoding="utf-8")
    commands=["prepare_inputs.py","analyze_a_correction.py","analyze_b_length.py","analyze_c_confirmation.py","plot_c_confirmation.py","assemble_report.py","validate_delivery.py"]
    (OUT/"logs/run_commands.json").write_text(json.dumps({"working_directory":str(REPO),"commands":["python -s -B pc-analysis-followup-output/scripts/"+x for x in commands],"environment":{"PYTHONUTF8":"1","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1"},"mode":"CPU only; own scripts","server_historical_scripts_executed":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Assembled main/C reports, README, statistics config, headline/count reconciliation tables.")


if __name__=="__main__":main()
