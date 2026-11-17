"""Collect the completed stage-1 audit material. No later-stage analysis."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "pc-analysis-output"
STAGE = BASE / "stages/01_audit"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    core = load(STAGE / "core_result.json")
    splits = load(STAGE / "reports/STAGE1_SPLITS_SUMMARY.json")
    scoring = load(STAGE / "reports/STAGE1_SCORING_METADATA.json")
    runtime = load(STAGE / "tables/runtime_audit_summary.json")
    environment = load(STAGE / "environment.json")
    affected = splits["affected_generations_by_stage"]
    total_affected = sum(affected.values())
    shared = {x["overlap_type"]: x for x in splits["overlap"]
              if x["left_split"] == "development" and x["right_split"] == "test"}
    findings = [
        {"id": "A01", "finding": "开发构造删去源证据token",
         "observed": f"{splits['constructions_with_missing_required_evidence']}/{splits['applicable_constructions']}构造；{splits['source_ids_with_missing_required_evidence']}独立ID；{total_affected}次实际生成",
         "affected_scope": json.dumps(affected, ensure_ascii=False),
         "impact": "探索、验证、长度检查的输入与完整证据保留要求有差异；冻结参数来自这些实际输入",
         "action": "保留原数据及答案，先设计独立修正构造；复核服务器offset，再决定修正开发实验及新规则的GPU补实验",
         "evidence": "construction_missing_evidence.csv; construction_missing_evidence_generations.csv; reports/STAGE1_EVIDENCE_DELETION_EXAMPLE.json"},
        {"id": "A02", "finding": "证据位置与跨块元数据差异",
         "observed": f"证据位置{splits['evidence_position_field_mismatches']}/432；跨块字段{splits['cross_block_metadata_mismatches']}/432",
         "affected_scope": "开发构造的证据位置与跨块字段；注意力图的证据标注",
         "impact": "按源字符证据和实际token序列定位的区域与原字段不同；注意力解释需使用重新定位的区域",
         "action": "后续CPU生成单独修正元数据；保存原字段；定位服务器offset根因所需材料",
         "evidence": "construction_checks.csv; construction_cross_block.csv; reports/STAGE1_SPLITS.md"},
        {"id": "A03", "finding": "LITM全文评分与官方首行入口不同",
         "observed": f"{scoring['litm_changed_generations']}/4500次生成；{scoring['litm_changed_questions']}/500个问题",
         "affected_scope": "LITM逐题EM、位置及总体汇总、方法配对",
         "impact": "保存分数符合campaign全文定义；官方入口对首行进行best_subspan_em",
         "action": "后续质量主表采用官方首行EM；全文口径保留为审核对照；使用既有答案CPU重评分",
         "evidence": "scoring_litm_first_line_changed_IDs.csv; scoring_litm_first_line_minimal_evidence.csv"},
        {"id": "A04", "finding": "正式输入实际N全部低于开发网格最小值",
         "observed": "LongBench共同510题N=1/2/3为177/104/229；LITM1500输入全部N=1；Doc2Dial500目标N=1/2为493/7",
         "affected_scope": "正式LongBench、LITM、Doc2Dial动态规则映射",
         "impact": "正式输入100%处于开发L/N矩形外，最近格映射的N均为4；N=8/16列用于开发实验",
         "action": "后续按实际映射格呈现结果；若要评价N=8/16正式迁移，列入另行授权的匹配块数补实验",
         "evidence": "runtime_unique_inputs.csv; runtime_input_scope_summary.csv; runtime_input_distributions.csv"},
        {"id": "A05", "finding": "开发与测试共享部分单篇资料和支撑资料",
         "observed": f"共享单篇资料{shared['single_document']['n_unique_shared_items']}，影响测试{shared['single_document']['n_affected_right_ids']}/510；共享支撑资料{shared['support_document']['n_unique_shared_items']}，影响测试{shared['support_document']['n_affected_right_ids']}/510",
         "affected_scope": "LongBench测试分层及组相关性",
         "impact": "问题和完整证据组跨分割重合为0；单篇资料重合需要进入结果解释",
         "action": "保留冻结510主表，另列共享/未共享开发支撑资料分层",
         "evidence": "splits_overlap_documents.csv; splits_overlap_affected_ids.csv; splits_overlap_summary.csv"},
        {"id": "A06", "finding": "开发验证输入复用探索背景片段",
         "observed": "96个验证题长构造使用探索供体，涉及40个验证ID；测试供体为0",
         "affected_scope": "开发验证背景资料；199个补充构造及298条贡献",
         "impact": "目标问题及完整证据组划分独立，背景片段在两个开发分割间共享",
         "action": "参数验证解释中记录共享背景来源，保留来源及批准区间",
         "evidence": "construction_supplement_sources.csv; construction_supplement_split_summary.csv"},
        {"id": "A07", "finding": "开发生成上限与早期配置文字不同",
         "observed": "Qasper128；HotpotQA/MuSiQue32；总体3731次达到上限，其中25次EOS恰在上限",
         "affected_scope": "explore/validation/length_check/attention及长度时延解释",
         "impact": "早期APE实验配置_纯文本写参数比较统一128；仓库v3未单列开发上限，冻结执行继承任务上限",
         "action": "按实际128/32设置报告；后续联合分析输出长度、EOS、质量和时延",
         "evidence": "runtime_generation_limits.csv; snapshot/APE/main80/freeze.py; APE实验配置_纯文本.txt"},
        {"id": "A08", "finding": "Doc2Dial来源与目标轮次的实际条件",
         "observed": f"官方validation抽取500目标，{scoring['doc2dial_unique_dialogues']}对话/{scoring['doc2dial_unique_documents']}资料；连续agent目标{scoring['doc2dial_consecutive_agent_targets']}；参考在历史重复{scoring['doc2dial_reference_repeated_in_history_targets']}",
         "affected_scope": "Doc2Dial应用解释、目标条件及聚类",
         "impact": "实际来源为DialDoc原始validation；28个last_turn为agent；7目标参考已在完整历史出现",
         "action": "保留原始500目标和官方ID，后续标记这些条件，按对话/资料聚类；说明资料前置及chat封装",
         "evidence": "scoring_doc2dial_input_checks.csv; scoring_doc2dial_history_condition_evidence.csv; data_sources.json"},
        {"id": "A09", "finding": "实际硬件采集字段待补充",
         "observed": "GPU0执行配置明确；RTX4090 48GB为方案登记；设备名/UUID/总显存/驱动等未采集",
         "affected_scope": "实验设置、跨设备效率复现",
         "impact": "当前硬件栏可填写执行编号和方案登记型号，实际设备/驱动依据需要补充",
         "action": "向服务器已有记录补取设备和驱动元数据；PC端保持CPU分析范围",
         "evidence": "runtime_environment.csv; reports/STAGE1_RUNTIME.md"},
    ]
    with (STAGE / "tables/audit_findings.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(findings[0]))
        writer.writeheader()
        writer.writerows(findings)
    followups = [
        {"priority": 1, "item": "开发证据缺失及位置元数据补正方案", "work": "CPU重建证据定位、构造及新旧差异清单；核对服务器tokenizers0.19.1的offset原始行为", "authorization": "本轮仅登记需求；新依赖或GPU任务另行授权", "affected_findings": "A01,A02"},
        {"priority": 2, "item": "修正开发实验与冻结规则的补实验", "work": "修正输入另存版本，重跑必要探索/验证，冻结新规则后安排质量和效率评价；现campaign保留", "authorization": "GPU补实验及新campaign须另行授权", "affected_findings": "A01"},
        {"priority": 3, "item": "LITM官方首行重评分", "work": "后续LITM阶段用既有答案CPU复算并形成正式EM配对表", "authorization": "属于原CPU分析范围，待用户安排对应阶段", "affected_findings": "A03"},
        {"priority": 4, "item": "实际N匹配与硬件资料", "work": "补取既有硬件元数据；需研究N=8/16迁移时设计匹配块数评价", "authorization": "GPU补实验另行授权", "affected_findings": "A04,A09"},
    ]
    with (STAGE / "tables/follow_up_requirements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(followups[0]))
        writer.writeheader()
        writer.writerows(followups)

    report = """# 第1阶段审核报告：数据、配置、划分与评分

## 1 本阶段结论

我们完成了已恢复证据的内容审核，发现需要补正的开发构造和LITM评分入口问题。逐题记录、冻结任务、参考答案和运行事件连接完整；开发构造的证据保留及其元数据另有实质差异。当前分析阶段为`COMPLETED_WITH_FINDINGS`，研究状态保留`SUBMITTED`。

本阶段整理原始事实、问题范围和后续需求；正式参数优劣、质量/效率配对区间及图表在对应阶段开展。

## 2 记录规模与来源

"""
    report += f"实际PC分析提交为`{environment['analysis_commit']}`，与参考提交相同。运行来源清单36项与快照按manifest别名关系对应；manifest登记507项逻辑文件。我们通过快照CPU任务工厂枚举{core['expected_physical_tasks']:,}项任务，并与{core['records']:,}条逐题记录完整连接。重复、缺失、额外任务、解析错误和冻结参考连接异常均为0；全部记录status=success、attempt=1。16个worker退出码均为0，逐请求事件对为18,116组。\n\n"
    report += "原矩阵18,600个槽位，加120个长度检查和12个注意力任务，为18,732。探索不适用减少528，验证不适用减少9，候选共用生成节省79，得到18,116条实际生成。\n\n"
    report += "|阶段|实际生成|\n|---|---:|\n" + "\n".join(f"|{x['stage']}|{x['observed_records']:,}|" for x in core["stage_counts"]) + "\n\n"
    report += "## 3 优先处理：开发证据保留问题\n\n"
    report += f"432份构造均能由源token保留区间与补充来源精确重建，说明保存的token内容可追溯。其中{splits['construction_all_evidence_retained']}份保留全部按源字符标注定位的证据token，另有{splits['constructions_with_missing_required_evidence']}份删去部分证据，涉及{splits['source_ids_with_missing_required_evidence']}个独立问题。80份包含4K34份、8K18份、6K22份、12K6份，16K为0；78份采用完整公共支撑文档保守映射，2份采用精确证据映射。\n\n"
    report += "最小实例为`hotpotqa:de89026c970fbb6dff04176107b9cf20407d155bc1ae2eec`。问题询问Dallas Cowboys punter的大学，参考为Arizona State University；源文字符区间[39348,39403)为“He played college football at Arizona State University.”，对应源token[9025,9034)。4K构造只保留源token[3791,7887)，上述完整9-token证据句被删除。6K构造也受影响。精确原文及区间见`STAGE1_EVIDENCE_DELETION_EXAMPLE.json`。\n\n"
    report += f"这些构造关联的实际生成为探索{affected.get('explore', 0):,}、验证{affected.get('validation', 0):,}、长度检查{affected.get('length_check', 0):,}，共{total_affected:,}条；注意力任务真实证据删除关联数为0。受影响ID、长度、实际N、T/S与原记录行号均已列入`construction_missing_evidence_generations.csv`。正式LongBench保持完整公开文本；LITM和Doc2Dial输入也未使用这些开发裁剪。冻结参数选择受到这些开发输入条件影响。\n\n"
    report += "另外，本机按源字符证据重新定位的证据位置与保存字段不符432/432份，跨块字段不符348/432份；保存字段内部对其稀疏区间的计算保持一致。服务器为Transformers4.44.2/tokenizers0.19.1，本机tokenizers0.20.3；定向AutoTokenizer核对使用本机Transformers4.46.3，三例的token ID和offset与原生tokenizers一致。全部432份源token重建一致，且最小实例能直接定位原文删除。服务器当时完整offset_mapping未保存在当前证据中，根因需要补取该版本的offset原始行为；本次将已确认的删除事实与待核对的根因分别登记。\n\n"
    report += "## 4 评分、划分和运行条件\n\n"
    report += f"保存的18,116条分数按campaign定义均能复现，最大浮点差为{scoring['score_max_abs_difference']:.3g}。LITM campaign按全文计算best_subspan_em，而[官方入口](https://github.com/nelson-liu/lost-in-the-middle/blob/main/scripts/evaluate_qa_responses.py)先取首行；该处理改变{scoring['litm_changed_generations']}条生成，涉及{scoring['litm_changed_questions']}个问题。后续LITM主表应对既有答案按官方首行重评分，原分数和全文口径保留为审核对照。\n\n"
    report += "LongBench为600题、536个连通证据组，探索50、验证40、测试510，测试每数据集170。跨开发/测试的问题和完整证据组重合为0；共享单篇资料96篇，影响59/510个测试ID；共享支撑资料24篇，影响29/510个测试ID。另有96份验证题长构造使用探索供体背景，涉及全部40个验证ID；测试资料供体为0。\n\n"
    report += "正式实际N：共同LongBench510题的1/2/3块为177/104/229；LITM500问题×3位置共1500输入全部1块、原始20篇；Doc2Dial500目标中493个1块、7个2块、原始均1篇。正式输入均因N<4处于开发网格矩形之外，最近格映射的N均为4。\n\n"
    report += "Doc2Dial实际读取官方`doc2dial_dial_validation.json`抽取500目标，来自366个对话、172篇资料。早期配置文字写ChatRAG-Bench公开测试集，实际冻结来源和模板依据为DialDoc原始validation；报告按实际来源命名。28个目标紧邻前一轮为agent，7个目标参考在完整历史中重复，均按官方ID保留并标记。所有目标的已选前序历史均带references；实际模板适配为全篇资料前置与统一chat envelope。\n\n"
    report += "Doc2Dial1500组冷/热请求完整，hot在cold后命中资料GPU KV并重新执行请求和生成，输出token一致。顺序路径的stage_calls保持0；APE热请求query调用与生成token数逐题对应。BLEU独立实现通过官方例子、全部逐题分数及六组服务器corpus结果核对；服务器签名中的版本为2.4.3，本机复算使用标准库实现并明确标识。\n\n"
    report += "实际开发生成上限为Qasper128、HotpotQA/MuSiQue32，早期配置文字写统一128，仓库v3未单独明确开发上限。共有3731条生成达到上限，其中25条EOS恰在上限；LITM为1465/4500，Doc2Dial为2162/3000。后续需联合输出长度与质量/时延解释。实际编码为同一GPU逐块forward后合并KV；TTFT与total含准备、选参和KV工作，加载单列，峰值显存含模型与存活缓存。\n\n"
    report += "执行配置明确使用physical GPU0和BF16/batch1；RTX4090 48GB来自方案登记。当前证据缺少该campaign设备名、UUID、实际总显存及驱动等采集，实验设置中列为待补充。\n\n"
    report += "## 5 后续处理与阶段停止点\n\n"
    report += "优先整理开发构造补正方案：用已定位的源字符/token信息重建证据与位置元数据，另存新版本，并补取服务器offset行为。对修正开发输入产生新答案、重做参数选择及新规则评价需要GPU补实验授权，具体范围由下一阶段安排确定。当前campaign、冻结规则、原始答案与分数全部保留。LITM官方首行重评分、共享资料分层、对话/资料聚类和生成长度联合分析已列入对应后续阶段。\n\n"
    report += "第1阶段至此完成，后续阶段尚未启动。完整发现与处理需求分别见`audit_findings.csv`和`follow_up_requirements.csv`。本阶段未安装依赖、未运行GPU、未计算新增文件摘要、未提交或推送Git。\n"
    (STAGE / "reports/AUDIT.md").write_text(report, encoding="utf-8")

    inputs = ["数据字典与任务连接", "划分及开发构造", "评分及参考连接", "配置、事件与生成停止"]
    report_files = ["STAGE1_CORE.md", "STAGE1_SPLITS.md", "STAGE1_SCORING.md", "STAGE1_RUNTIME.md"]
    readme = "# 第1阶段材料索引\n\n当前阶段：数据、配置、划分及评分审核。交付状态：完成并登记发现。研究状态：SUBMITTED。\n\n"
    readme += "## 1 阅读顺序\n\n- [综合审核与下一步需求](reports/AUDIT.md)\n"
    readme += "\n".join(f"- [{label}](reports/{name})" for label, name in zip(inputs, report_files))
    readme += "\n\n## 2 表格索引\n\n|类别|主要文件|用途|\n|---|---|---|\n"
    indexes = [
        ("总发现", "audit_findings.csv / follow_up_requirements.csv", "现象、影响、处理方式和证据"),
        ("任务和字段", "record_stage_counts.csv / task_record_join.csv / data_dictionary.csv", "数量、源行和冻结输入连接"),
        ("快照", "snapshot_lineage.csv", "36项来源的manifest别名关系"),
        ("真实证据删除", "construction_missing_evidence.csv / construction_missing_evidence_generations.csv", "80份构造与1605条实际生成"),
        ("位置与构造", "construction_checks.csv / construction_cross_block.csv", "实际源字符/token证据与保存元数据的差异"),
        ("划分共享", "splits_overlap_documents.csv / splits_overlap_affected_ids.csv", "共享篇目、支撑资料及受影响ID"),
        ("补充背景", "construction_supplement_sources.csv", "供体区间、批准依据及分割"),
        ("LITM入口", "scoring_litm_first_line_changed_IDs.csv / scoring_litm_first_line_minimal_evidence.csv", "501条评分差异及最小原文"),
        ("Doc2Dial条件", "scoring_doc2dial_history_condition_evidence.csv", "连续agent和历史参考重复条件"),
        ("输入范围", "runtime_unique_inputs.csv / runtime_input_scope_summary.csv", "以唯一输入为分母的L/N与原篇数"),
        ("生成与事件", "runtime_generation_limits.csv / runtime_workers.csv / runtime_hot_pairs.csv", "上限、退出和热请求重生成"),
        ("评分一致性", "scoring_definition_consistency.csv / scoring_doc2dial_corpus_consistency.csv", "逐条及语料充分统计核对"),
    ]
    readme += "\n".join(f"|{a}|" + " / ".join(f"[{name}](tables/{name})" for name in b.split(" / ")) + f"|{c}|" for a, b, c in indexes)
    readme += "\n\nCSV均为UTF-8，包含样本ID、来源、计数分母或单位；原文件行号为一基，frozen_index_zero_based为零基。不同异常类别可能覆盖同一构造，请分别使用各表分母。\n\n"
    readme += "## 3 复现\n\n输入为仓库restored及manifest，输出仅此阶段目录。核心、评分和运行审核仅使用Python标准库；构造审核使用已有tokenizers0.20.3。本阶段确定性执行，无Bootstrap和随机重采样；生成实验seed=42，后续统计规划seed=20261116、5000次。\n\n"
    readme += "```powershell\nSet-Location 'D:\\_Greatest_programmer\\_Projects\\APEReport'\n$env:PYTHONUTF8='1'\n$env:PYTHONNOUSERSITE='1'\n$env:PYTHONDONTWRITEBYTECODE='1'\n$env:TEMP=(Join-Path (Get-Location) 'pc-analysis-output\\tmp')\n$env:TMP=$env:TEMP\nforeach ($part in @('core','splits','scoring','runtime')) {\n    & 'C:\\Users\\Osheepne\\anaconda3\\python.exe' -s -B ('pc-analysis-output/scripts/audit_stage1_' + $part + '.py')\n    if ($LASTEXITCODE -ne 0) { throw ('audit script failed: ' + $part) }\n}\n& 'C:\\Users\\Osheepne\\anaconda3\\python.exe' -s -B pc-analysis-output/scripts/summarize_stage1.py\n```\n\n"
    readme += "构造审核保留已发现的问题，具体failed检查见splits_checks.csv；脚本执行成功与研究验收状态分别登记。首次定向排查记录保存在reports/，运行环境见environment.json。本次无需新增依赖。后续阶段由用户安排。\n"
    (STAGE / "README.md").write_text(readme, encoding="utf-8")
    status = {"stage": "01_audit", "stage_delivery": "COMPLETED_WITH_FINDINGS", "research_status": "SUBMITTED",
              "later_stages_started_this_turn": False, "formal_bootstrap_executed": False,
              "gpu_executed": False, "new_dependencies_installed": [], "analysis_commit": environment["analysis_commit"],
              "material_findings": len(findings), "construction_missing_evidence_count": splits["constructions_with_missing_required_evidence"],
              "construction_affected_generations": total_affected,
              "litm_changed_generations": scoring["litm_changed_generations"], "litm_changed_questions": scoring["litm_changed_questions"],
              "next_step": "用户安排后，优先讨论开发构造补正及参数分析范围"}
    (STAGE / "STAGE_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (BASE / "README.md").write_text("# APE MAIN80 PC端分析\n\n当前已交付第1阶段：数据、配置、划分及评分审核，含需补正的发现。研究状态保留SUBMITTED。后续阶段待用户安排。\n\n- [第1阶段材料索引](stages/01_audit/README.md)\n- [综合审核报告](stages/01_audit/reports/AUDIT.md)\n- [阶段状态](stages/01_audit/STAGE_STATUS.json)\n\n实际分析提交：`" + environment["analysis_commit"] + "`，与参考提交相同。输入只读使用restored/及manifest；新脚本保存在scripts/，本阶段表格、最小证据和报告集中在stages/01_audit/。既有acquisition/为仓库获取记录，早期tables/figures/与MAIN_STATISTICS.md为本次分阶段交付前的初算材料。\n\n环境、依赖、复现命令及下一步需求均在第1阶段索引中。原始文件及冻结规则保持原样。\n", encoding="utf-8")
    (BASE / "reports").mkdir(exist_ok=True)
    (BASE / "reports/AUDIT.md").write_text("# APE MAIN80 分阶段审核\n\n第1阶段已完成并登记发现。研究状态：SUBMITTED。\n\n[第1阶段完整审核报告](../stages/01_audit/reports/AUDIT.md)\n\n后续阶段待用户安排，正式中文实验报告在对应阶段结果完成后形成。\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
