# 第1阶段：LITM、Doc2Dial 与评分实现审核

研究状态：SUBMITTED。

本次仅审核输入、配对、参考连接、评分定义及热请求生成记录。正式方法质量比较、时延比较、置信区间及图表留待后续阶段。

## 1 执行与证据

运行环境：`C:\Users\Osheepne\anaconda3\python.exe`，Python `3.12.7`；仅用标准库，SacreBLEU 未安装或导入。本阶段无随机抽样。

```powershell
C:\Users\Osheepne\anaconda3\python.exe -B pc-analysis-output/scripts/audit_stage1_scoring.py
```

输入：`D:\_Greatest_programmer\_Projects\APEReport\restored`，campaign `main80-20261114T041503`。我们读取 18,116 条物理生成记录，使用 `snapshot/APE/main80/` 解释实际执行，按冻结 ID 连接参考。`logical_labels` 保留为生成标签，同一物理生成只审核一次。

## 2 已发现的 LITM 评分入口差异

campaign 将完整答案传入 `best_subspan_em`；官方评测入口先取第一行并去首尾空白，再做规范化子串匹配。专项复算发现 501/4,500 条生成的评分发生变化，涉及 147 个问题。受影响范围为 LITM 的逐题 EM、位置汇总、总体汇总以及后续方法配对比较。原始答案及保存分数保留；后续质量分析需另列官方首行 EM，并保留全文口径作为审核对照。

|方法|位置|生成数|首行规则改变评分的生成数|
|---|---:|---:|---:|
|ape_dynamic|0|500|55|
|ape_dynamic|9|500|53|
|ape_dynamic|19|500|54|
|ape_fixed|0|500|53|
|ape_fixed|9|500|47|
|ape_fixed|19|500|81|
|sequential|0|500|53|
|sequential|9|500|49|
|sequential|19|500|56|

本表描述评分入口的影响范围。全部受影响 ID 和源行号见 `tables/scoring_litm_first_line_changed_IDs.csv`；每个方法／位置取最小样本 ID 形成一条原答案证据，共 9 条，见 `tables/scoring_litm_first_line_minimal_evidence.csv`。

## 3 输入、配对与参考连接

LITM 读取 500 个问题、三种位置和三方法，共 4,500 条成功生成。我们从原始压缩文件流式读取选中问题，核对20篇文档的多重集合、唯一 gold 及位置、冻结文档全文、问题与参考答案，以及共同输入模板。输入核对异常 0 条；方法／位置配对异常 0 项。

Doc2Dial 读取 500 个目标及 3,000 条成功生成，目标分布于 366 个对话和 172 篇资料。我们以官方 `{dial_id}_{agent_turn_id-1}` 连接原始目标，核对grounded agent角色、紧邻前一轮的ID与角色、每个前序历史轮次、完整资料及参考答案。输入／参考异常 0 条；六条件配对异常 0 项。

实际对话条件：28/500 个目标的紧邻前一轮也是agent，因而模板last_turn在这些目标中是前一个agent发言；它们共对应 168 条生成。另有 7/500 个目标参考在前序历史中出现完全相同的text2line文本，涉及 42 条生成。官方参考ID连接及原模板按agent轮次处理，均保留这些原始目标。后续分析须标记连续agent与历史重复条件，并按对话／资料处理相关性。最小原文及匹配轮次见 `scoring_doc2dial_history_condition_evidence.csv`，其中样本 `22198e14bb6cd2a93ec1f6540101934b_2` 的原始目标和上一agent文本均为`gh`、最近user文本为`hg`。

选中目标历史中无grounding前序轮次的出现次数为 0，同一历史轮次随多个目标出现时分别计次。这500个目标的所有前序轮次均带references，完整历史策略与旧脚本筛选后的历史内容一致。官方 `text2line`、标签、last_turn 与逆序历史继续采用；本次将全文资料放在历史前面、保留全部前序历史并加入统一 Llama chat envelope。官方历史模板先跳过无references轮次，再累计历史，全文资料追加在历史后面。上述适配写入冻结记录，三方法共享。目标参考经 text2line 转换，其13a分词与原始目标 utterance 逐项对照。

## 4 保存评分与官方定义

LongBench：小写、去ASCII标点、去冠词a/an/the、合并空白，按词袋重叠计算F1，多参考取最大。我们以独立实现复算，同时从包内官方源仅加载三个纯评分函数作逐条对照。F1／EM原值范围为0–1，Doc2Dial BLEU为0–100。

保存逐条分数与其campaign定义的复算差异超过 1e-08 的记录共 0 条；最大绝对数值差 2.13e-14。冻结参考连接异常 0 条，非法分数／缺失输入检查异常 0 条。明细按数据集和阶段记录于 `scoring_definition_consistency.csv`，只展示一致性和差值。

Doc2Dial逐条 `score` 使用单条输入的 corpus_bleu 默认设置；汇总另调用 corpus_score。因此审核对逐条和语料聚合分别核对。独立实现采用13a分词、1–4gram截断匹配、exp平滑、mixed case和effective order off，先累加充分统计量，再计算语料BLEU及长度惩罚。我们验证官方HF例子与SacreBLEU例子中单句组件的独立手算，并对照全部保存逐条分数和六个服务器语料聚合。

服务器签名为 `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3`；当前实现是独立标准库复算实现，签名用于对照服务器设置。官方历史环境记录SacreBLEU1.4.3、datasets1.3.0，本campaign记录SacreBLEU2.4.3。实际设置及六组充分统计／绝对差保存在 `scoring_doc2dial_corpus_consistency.csv`。

## 5 热请求重新生成证据

我们核对 1,500 个目标／方法冷热请求对，联合检查独立request_started/ended事件及时间顺序、cold miss/hot hit、相同资料与配置、输出token一致性和APE query调用。未通过联合检查的请求对 0 个。Doc2Dial缓存恢复事件 0 个。

实际执行代码每次将history/current request的suffix重新forward，从当前logits重新逐token生成；热缓存只复用资料KV。APE热请求prefix/context调用为0，query调用与本次输出token数对应。顺序方法的stage hook按代码保持0，其重复生成链路由统一run循环和独立请求事件登记。冷热输出token相同另行作为相同贪心请求的一致性检查，证据表保留源记录及事件行号。恢复重建成本的字段及代码已登记，计时比较在后续阶段进行。

## 6 本阶段结论与交接

本阶段交付评分差异和数据连接审核。LITM后续质量分析须采用官方首行处理，并明确原campaign全文计分造成的影响范围。输入及评分一致性检查结果均列入 `scoring_checks.csv`；存在异常时按具体ID／源行号复核相关范围。研究状态保持SUBMITTED。

## 7 官方来源与本地定位

Doc2Dial实际数据源为 `data_sources.json:doc2dial_dialogues` 指向的 `datasets/raw-main80/doc2dial/doc2dial_dial_validation.json`，本campaign从官方validation分割中冻结抽取500个目标轮次。早期PC端配置《APE实验配置_纯文本.txt》第19行规划使用ChatRAG-Bench中的Doc2Dial公开测试集；该文本记录早期方案，本次执行来源按冻结清单和实际源文件登记为官方validation分割，两者按方案与执行版本分别保留。

- [longbench_metrics](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py)
- [longbench_eval](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py)
- [litm_metrics](https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py)
- [litm_first_line](https://github.com/nelson-liu/lost-in-the-middle/blob/main/scripts/evaluate_qa_responses.py)
- [doc2dial_eval](https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/sharedtask_utils.py)
- [doc2dial_template](https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/subtask2/seq2seq_utils.py)
- [doc2dial_historical_versions](https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/requirements.txt)
- [historical_hf_defaults_and_example](https://github.com/huggingface/datasets/blob/1.3.0/metrics/sacrebleu/sacrebleu.py)
- [bleu_243](https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/metrics/bleu.py)
- [tokenizer_13a_243](https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/tokenizers/tokenizer_13a.py)
- [tokenizer_regexp_243](https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/tokenizers/tokenizer_re.py)
- [sacrebleu_example_component](https://github.com/mjpost/sacrebleu/blob/v2.4.3/README.md#using-sacrebleu-from-python)

本地执行链路：`snapshot/APE/main80/metrics.py`、`freeze.py`、`methods.py`、`worker.py`、`report.py`；官方保存源：`datasets/raw-main80/references/longbench_metrics.py.txt` 与 `litm_metrics.py.txt`。表中 `attempts_line`、事件行号均为UTF-8原文件一基行号。
