# 第五阶段审核记录

范围为12条attention生成、231条注意力捕获及六份开发构造，独立问题3个。所有捕获流式读取，无结构/非有限值/负权重/保存容差异常。以下问题影响区域解释、decode配对或PC复核所需数据。

| 现象 | 范围 | 证据 | 影响与处理 | 后续所需 |
|---|---|---|---|---|
| 保存证据区间稀疏 | 6/6构造; 12项诊断 | 完整证据token q43/h1468/m1833; 保存区间q10/h235/m494 | 直接按保存区间会将部分证据计入背景；按原字符证据和现有tokenizer重建掩码、投影retained区间; 原始输入及字段保留 | 本阶段区域统计已修正; 关联上游映射问题见前置审核 |
| decode历史分歧 | Qasper4K从step5; Qasper16K从step1 | decode_history_alignment.csv逐步列出两历史token ID | 后续注意力同时随参数和已生成内容变化；同输入step0为主; decode逐步保留same_history标签及未配对尾部数 | 需要隔离参数效应时补固定同一token序列的teacher-forced诊断 |
| 未保存weighted-V独立重建输入 | 231条捕获 | 原始字段保存weights/branch/error/tolerance; 缺Q/K/V和实际target向量 | PC可核对保存误差、容差和质量一致性; weighted-V误差数值来自服务器采集；明确误差的来源和检验步骤 | 若需PC独立重建,为选定layer/head/step补存Q/K/V/target及实际分支输出 |
| 探索候选与冻结动态规则的角色不同 | 16K/16的三个参数比较 | 候选0.5/0.7; rule.json冻结0.7/1.0 | 本阶段解释探索候选的注意力与答案；表中同时列探索参数与冻结参数; 本阶段保持冻结规则 | 冻结规则收益在后续正式配对实验评价 |

具体最小证据见sources/evidence_mapping_examples.json、tables/decode_history_alignment.csv、tables/capture_audit.csv和tables/answer_pairs.csv。稀疏证据位置的根因及全campaign受影响范围沿用前置审核报告，本阶段只处理六份实际诊断输入。

代码依据：运行快照main80/jobs.py第55–64行定义诊断选题与候选；main80/diagnostics.py第25–49行定义权重重建、容差和存储顺序；ape/ape_llama.py第170–197行定义T/S、分支混合与o_proj前观察；main80/methods.py第203–223行定义请求与生成循环。这些文件位于restored/APE/，运行快照对应副本在该次snapshot/APE/。数据依据为同次snapshot/datasets/frozen-main80/和attention.jsonl。

复核结论：231条层级捕获、924条head-step，36项配置×层覆盖完整；保存重建最大误差0.006171465，最大误差/容差0.112259。六组step0输入匹配；两个Qasper轨迹的历史分歧另标；保存原始质量的归一偏差与BF16分支质量相符。原始记录、候选、冻结规则和研究状态保留。
