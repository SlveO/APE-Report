# 第1阶段：数据划分与开发输入构造审核

本阶段读取恢复后的启动快照，核对LongBench原始公开数据连接、分组划分和精确长度开发输入。全部新增文件写入 `pc-analysis-output/stages/01_audit/`；研究状态保持 `SUBMITTED`。

## 1 数据与实际分母

| 数据集 | 全量 | 探索 | 验证 | 独立测试 |
|---|---:|---:|---:|---:|
|qasper|200|17|13|170|
|hotpotqa|200|17|13|170|
|musique|200|16|14|170|

总计600个唯一公开ID，探索50、验证40、测试510。冻结顺序中的ID集合与对应分割一致。按完整支撑文档集合或规范化相同问题独立连通分组，共536组，与冻结分组一致。逐题公开ID、问题、完整文档和参考答案连接均通过（600/600）。

## 2 跨分割重合

问题规范化采用casefold并合并空白；完整证据组以冻结的支撑文档组合与同问题连通分组为准。单篇资料统计`input_documents`；支撑资料统计`support_documents`。

| 左侧/右侧 | 类型 | 共享项目数 | 左侧受影响ID/分母 | 右侧受影响ID/分母 |
|---|---|---:|---:|---:|
|explore/validation|规范化问题|0|0/50|0/40|
|explore/validation|完整证据组|0|0/50|0/40|
|explore/validation|单篇资料|4|3/50|3/40|
|explore/validation|支撑资料|2|2/50|2/40|
|explore/test|规范化问题|0|0/50|0/510|
|explore/test|完整证据组|0|0/50|0/510|
|explore/test|单篇资料|32|11/50|30/510|
|explore/test|支撑资料|8|7/50|10/510|
|validation/test|规范化问题|0|0/40|0/510|
|validation/test|完整证据组|0|0/40|0/510|
|validation/test|单篇资料|67|14/40|44/510|
|validation/test|支撑资料|17|8/40|21/510|
|development/test|规范化问题|0|0/90|0/510|
|development/test|完整证据组|0|0/90|0/510|
|development/test|单篇资料|96|25/90|59/510|
|development/test|支撑资料|24|15/90|29/510|

完整证据组与规范化问题跨分割重合为0。部分Wikipedia单篇资料及支撑资料跨开发与测试共享；后续测试分析应按这项资料共享情况分层呈现。`splits_overlap_documents.csv`保留每篇共享资料及两侧ID，`splits_overlap_affected_ids.csv`保留逐ID归属。

## 3 精确长度构造与不适用输入

90个开发ID分别检查4096、8192、16384、6144和12288 tokens，共450个题长候选。有效432个，不适用18个。精确长度计入文档token，问题、共享前缀和输出另计。

| 分割 | L | 有效/候选 | 不适用原因及数量 |
|---|---:|---:|---|
|explore|4096|48/50|no_annotated_evidence: 2|
|explore|8192|47/50|no_annotated_evidence: 2；insufficient_verified_development_supplement: 1|
|explore|16384|44/50|no_annotated_evidence: 2；insufficient_verified_development_supplement: 4|
|explore|6144|47/50|no_annotated_evidence: 2；insufficient_verified_development_supplement: 1|
|explore|12288|47/50|no_annotated_evidence: 2；insufficient_verified_development_supplement: 1|
|validation|4096|40/40|无|
|validation|8192|40/40|无|
|validation|16384|39/40|insufficient_verified_development_supplement: 1|
|validation|6144|40/40|无|
|validation|12288|40/40|无|

原始原因`no_evidence_or_evidence_exceeds_target`按导出分词器的实际证据token数量进一步拆分；逐ID原因与证据token数见`construction_applicability.csv`。4K/8K/16K探索有效48/47/44个题长输入，覆盖三种N和16种T/S的计划生成数为6,672；本阶段仅核对输入覆盖。

## 4 证据保留、补充资料和跨块字段

使用已有tokenizers 0.20.3读取导出的tokenizer.json，对90个开发源文档重新分词。全部432个有效构造按保留区间及补充来源区间重建后，token序列和目标L完全一致。按字符证据区间与本机字符offset映射核对，352个构造保留全部必要证据token，80个构造删去了部分必要证据。保存的证据位置字段在432个构造中与当前字符证据映射有差异，其中348个构造的跨块字段随之不同。每个构造的实际保留、位置字段差异和跨块统计分别见`construction_checks.csv`、`construction_missing_evidence.csv`和`construction_cross_block_summary.csv`。

其中199个构造使用补充资料，共298条来源贡献。检查补充来源处于开发集、目标与供体支撑资料互斥、证据组不同、所用token区间处于审阅批准区间，并保留语义审阅依据与引用。补充片段标准答案字面匹配检查见来源表。

| 目标分割 | 供体分割 | 使用该来源的构造数 | 目标ID数 | 供体ID数 | 贡献token数 |
|---|---|---:|---:|---:|---:|
|explore|explore|103|47|6|451324|
|validation|explore|96|40|7|442852|

冻结补充池允许探索与验证资料；本次实际使用的供体全部来自探索集，验证集96个构造使用了探索来源背景。两开发分割在背景片段层面共享材料；后续验证解释应同时报告这项共享范围。测试与验证资料作为补充供体的数量均为0。每条贡献保存目标ID、供体ID、实际区间、来源分割、批准依据与审阅引用，见`construction_supplement_sources.csv`。

保存的跨块字段与其保存的稀疏证据区间在内部计算上一致；按完整字符证据映射重算时存在差异。跨块字段表示连续证据token区间跨过等长块边界。同一题长构造对应同一token序列，三种N据此切分；本阶段覆盖表保留432×3=1296个构造/N格。

## 5 最小原文证据与检查结果

原文证据示例按“数据集×映射策略”选择ID字典序首条，并补充不适用原因、补充来源及共享支撑资料的最小示例，保存于`STAGE1_SPLITS_MINIMAL_EVIDENCE.json`。示例含样本ID、问题、来源ID、字符或token区间和原文片段。

完成46项汇总检查，失败1项；逐构造差异860条（同一构造按证据保留、位置和跨块字段分别登记）。数据覆盖、来源共享和不适用原因已登记；本阶段产物未进行参数评分比较。

## 6 复现与来源字段

```powershell
& "C:/Users/Osheepne/anaconda3/python.exe" -B pc-analysis-output/scripts/audit_stage1_splits.py
```

主要输入：启动快照`longbench.json`、`development.json`、`applicability.json`、`split-report.json`、`frozen.json`；公开LongBench三份jsonl；导出的tokenizer；`datasets/prepared-main80/supplement-reviews.json`。每张表带来源与单位字段。未安装依赖，未导入模型或执行GPU工作。

## 7 构造删去证据的范围、版本线索与处理

80个受影响构造涉及34个来源ID；实际关联生成按阶段为{'explore': 1392, 'validation': 201, 'length_check': 12}。每个构造在N=4/8/16下共用同一token序列，证据删去发生在分块之前。完整逐题清单和关联生成见`construction_missing_evidence.csv`、`construction_missing_evidence_generations.csv`。正式LongBench、LITM和Doc2Dial使用原始完整资料，其输入路径与这些开发构造分开。

最小实例区分两类现象：qasper:e419f2bff9d2ab7c3b60b3250caccd2d9ae1285ec3e8e818的4K输入完整保留相应段落，保存字段只标出WAP、MFCC、MDRE等稀疏词片；另有构造的retained/deleted源区间直接删除了字符证据范围内的token。具体实例hotpotqa:de89026c970fbb6dff04176107b9cf20407d155bc1ae2eec的4K/6K构造删除了源token[9025,9034)，内容为“He played college football at Arizona State University.”，对应参考答案所在整句。原文字符区间为[39348,39403)。详见`STAGE1_EVIDENCE_DELETION_EXAMPLE.json`，可用`inspect_stage1_evidence.py`复现。80个受影响构造中78个采用整篇支撑文档回退证据、2个采用精确句子证据；遗漏量为4—11085个token。定位依赖已重建且完全相同的token ID序列。

服务器记录Transformers4.44.2/tokenizers0.19.1；本机Transformers4.46.3/tokenizers0.20.3。原生tokenizers与AutoTokenizer对最小实例给出相同token ID与270个证据token。上游0.19.1的ignore_merges分支对词表整词返回(0,0)偏移，修复PR将其改为实际长度；该机制与保存字段只集中于多token词的模式一致。根因归档还缺服务器0.19.1逐token offset原始输出，需在原环境做最小CPU复现。

上游依据：[v0.19.1 BPE实现](https://github.com/huggingface/tokenizers/blob/v0.19.1/tokenizers/src/models/bpe/model.rs)、[offset修复PR1640](https://github.com/huggingface/tokenizers/pull/1640)、[Llama3 offset问题1517](https://github.com/huggingface/tokenizers/issues/1517)。既有mapping-second/third/fourth-check只记录来源映射结果；scope-validation记录执行范围；data-validation记录432份通过，验收代码复用了服务器同一offset_mapping。上述JSON未保存逐token offset值。

处理方式：保留原始构造与冻结规则，将当前结果归入这批实际输入的观察；重新保证完整证据的开发构造需要重建后作为新版本复核适用性，再补跑受影响探索/验证和依赖它们的规则冻结，随后重新评价冻结规则。现有规则与研究状态在本阶段保持原值。该补实验涉及GPU和新版本实验，列入后续决策事项。
