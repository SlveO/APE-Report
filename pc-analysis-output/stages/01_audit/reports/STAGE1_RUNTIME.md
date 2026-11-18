# 第1阶段：运行配置、计时与生成记录审核

我们按 worker bundle、运行快照和逐请求事件审核了 18,116 条生成记录。记录与任务上限全部连接，输入/输出长度、EOS、缓存条件和计时字段检查共记录 0 项异常。第1阶段仅整理配置与数据合法性，正式质量复算、效率比较和注意力图在后续阶段开展。研究状态保持 `SUBMITTED`。

## 1 实际模型、环境与硬件登记

本 campaign 从 `models/Llama-3.1-8B-Instruct` 离线读取模型和 tokenizer，实际来源为 ModelScope 用户镜像 `LLM-Research/Meta-Llama-3.1-8B-Instruct` 的 `master` 分支。获取清单逐文件记录观察到的 Revision；获取前后 Path/Size/Revision 元数据比较结果为 `True`。模型权重保留在服务器，本 PC 包保存配置、索引和 tokenizer。模型架构为 32 层、32 个注意力头、8 个 KV 头，上下文配置 131,072 tokens。

准备阶段记录 Python 3.10.20、PyTorch 2.6.0+cu124、Transformers 4.44.2、tokenizers 0.19.1、FlashAttention 2.7.4.post1；冻结审核另记录 SacreBLEU 2.4.3。with_env 使用 core-tf44 和 main80-scoring overlay，模型加载指定 FlashAttention 2、BF16、batch=1，种子42。PyTorch `cu124` 是其构建标签。

硬件记录分为执行配置和方案登记：执行配置为 physical_gpu=0、`CUDA_VISIBLE_DEVICES=0`、进程内 `cuda:0`；方案登记“RTX 4090 48GB”。本包缺少该 campaign 的设备名称、GPU UUID、实际总显存、驱动版本与功耗上限采集。报告的实际硬件栏应补充这些字段后填定；历史 R2 的 L20 注释对应另一入口。来源：`restored/output/runs/main80-20261114T041503/snapshot/scripts/with_env.sh:18`、`restored/docs/EXPERIMENT_PLAN.md:9`、`restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:86`。

完整版本和来源见 `runtime_environment.csv`，模型获取元数据见 `restored/logs/runs/20261113-main80-assets/model-manifest.json`。

## 2 输入模板和编码方法

任务 bundle 用 table/index 指向同一冻结行，再注入 N、方法和参数。我们检查 18,116 个任务与冻结行的 sample_id/max_new_tokens，并按 table、sample_id、L、N、position 检查各方法所用索引，共 3,434 个输入条件组。具体异常见 `runtime_integrity_issues.csv`。

LongBench 官方提示在 context 处分割，LITM 保留公开问题和20篇资料文本，三者共同使用 Llama user/assistant envelope。Doc2Dial 的全篇资料前置，history/current question 保留在 suffix，`<last_turn>` 与倒序角色历史共同组成请求；所有方法使用同一适配。`runtime_templates.csv` 列出四张冻结表的字段与上限。Doc2Dial 的 last_turn 取历史最后一轮，可为 user 或 agent；该轮在 last_turn 与角色历史中均出现，完整角色历史保留，最近用户轮次可能更早。500个目标中28个紧邻前一轮为agent，按官方 turn_id-1 构造合法ID，具体计数见评分审核，来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/freeze.py:63`。

顺序方法先编码公共 prefix+全部资料。PCW 对每个资料块加入独立 BOS 并在 position=0 编码，最长窗口稳定排首位，合并 KV 时保留一个 BOS，再以 prefix[1:]+suffix 发起请求。APE 每块共享公共 prefix，分别编码资料，再合并 prefix+各块 KV；query 使用 T/S 对上下文和其他分支进行调整。PCW 使用 stock Llama，APE 使用前缀/上下文/query 补丁。三种路径的 RoPE/缓存位置组织随方法定义变化。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:129`、`restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:42`。

本次 PCW/APE 均通过 Python 逐块循环调用 forward，各块在同一 GPU 上顺序执行，之后合并独立编码的 KV，batch=1。常规输入按每块最多7500 document tokens 分块；开发网格显式指定 N。原始篇数与实际块数分别保存于 original_document_count 和 N。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:25`。

### 2.1 以唯一输入为分母的 L/N 盘点

LongBench完整基线每题取一条输入（3方法为条件）；共同测试取5方法交集ID（每题5个条件）；LITM按问题×位置去重，三个位置仍隶属于同一问题；Doc2Dial按目标轮次去重（3方法×冷/热为6个条件）。下表中的 unique_inputs 是输入数量，unique_question_or_target_ids 保留问题/目标ID数量。各方法/冷热条件之间仅核对L、N、原始资料篇数及逐块长度的一致性，时延与质量留待后续分析。

| scope | dataset | unique_inputs | unique_question_or_target_ids | conditions_per_input | L_min | L_max | N_values | original_document_count_values | complete_condition_count | input_metadata_equal_count | outside_grid_rectangle_count | denominator |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longbench_full600 | hotpotqa | 200 | 200 | 3 | 1726 | 16322 | 1,2,3 | 2,3,4,5,6,7,8,9,10 | 200 | 200 | 200 | 200 |
| longbench_full600 | musique | 200 | 200 | 3 | 6482 | 16335 | 1,2,3 | 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 | 200 | 200 | 200 | 200 |
| longbench_full600 | qasper | 200 | 200 | 3 | 1846 | 21110 | 1,2,3 | 1 | 200 | 200 | 200 | 200 |
| longbench_full600 | ALL | 600 | 600 | 3 | 1726 | 21110 | 1,2,3 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 | 600 | 600 | 600 | 600 |
| longbench_common510 | hotpotqa | 170 | 170 | 5 | 1726 | 16322 | 1,2,3 | 2,3,4,5,6,7,8,9,10 | 170 | 170 | 170 | 170 |
| longbench_common510 | musique | 170 | 170 | 5 | 6482 | 16335 | 1,2,3 | 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 | 170 | 170 | 170 | 170 |
| longbench_common510 | qasper | 170 | 170 | 5 | 1846 | 21110 | 1,2,3 | 1 | 170 | 170 | 170 | 170 |
| longbench_common510 | ALL | 510 | 510 | 5 | 1726 | 21110 | 1,2,3 | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 | 510 | 510 | 510 | 510 |
| litm_question_positions1500 | litm | 1500 | 500 | 3 | 2532 | 3573 | 1 | 20 | 1500 | 1500 | 1500 | 1500 |
| litm_question_positions1500 | ALL | 1500 | 500 | 3 | 2532 | 3573 | 1 | 20 | 1500 | 1500 | 1500 | 1500 |
| doc2dial_targets500 | doc2dial | 500 | 500 | 6 | 223 | 8611 | 1,2 | 1 | 500 | 500 | 500 | 500 |
| doc2dial_targets500 | ALL | 500 | 500 | 6 | 223 | 8611 | 1,2 | 1 | 500 | 500 | 500 | 500 |

网格范围定义为4096≤L≤16384且4≤实际N≤16；精确九节点定义为L∈{4096,8192,16384}且N∈{4,8,16}。`runtime_input_scope_summary.csv` 同时保存范围外比例、单独L/N越界计数和精确节点之外计数；`runtime_input_distributions.csv` 给出每个实际L、N及原始篇数的频数，`runtime_input_LN_joint.csv` 给出联合分布，均以唯一输入为分母。`runtime_unique_inputs.csv` 保存逐输入源行和完整配对检查结果。

## 3 生成配置与合法性

config 声明 do_sample=False、num_beams=1、seed=42；Engine 实际手工 `logits.argmax` 逐 token 贪心解码。模型 generation_config 的 do_sample=True、temperature=0.6、top_p=0.9 来自模型文件，执行入口仅使用其中 EOS 集 `[128001, 128008, 128009]`。终止条件为 EOS 或达到任务 max_new_tokens；输出 token 数包含最终 EOS，decode 文本去除特殊 token。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:216`、`restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:212`。

有效 token ID 数组 18,116/18,116，输出长度与数组/上限一致 18,116/18,116。总体命中上限 3,731 条，其中 EOS 恰在上限 25 条。最长输出 128 tokens。我们同时检查了正块长、L等于块长和、N等于块数、完整原始输入加生成上限落在模型窗口内；PCW 计算时将已附加在 request_tokens 中的 prefix[1:] 只计一次。

开发阶段实际沿用每数据集生成上限：Qasper为128，HotpotQA/MuSiQue为32。这个设置覆盖explore、validation、length_check和attention，来源为 `snapshot/APE/main80/freeze.py:21` 的行构造及随后开发行复制。后续参数比较应按这些实际执行上限解释各数据集结果。

各阶段数量、上限和上限命中如下。命中数在本阶段用作数据审核，长度与质量/时延的联合分析留给后续阶段。

| stage | records | cache_modes | max_new_tokens | cap_hits |
| --- | --- | --- | --- | --- |
| baseline | 1800 | cold | 32,128 | 31 |
| explore | 6672 | search | 32,128 | 41 |
| validation | 992 | search | 32,128 | 18 |
| length_check | 120 | cold | 32,128 | 1 |
| attention | 12 | cold | 32,128 | 0 |
| longbench_final | 1020 | cold | 32,128 | 13 |
| litm | 4500 | cold | 128 | 1465 |
| doc2dial | 3000 | cold,hot | 128 | 2162 |

按数据集/方法/冷热细分见 `runtime_generation_limits.csv`，包含分母、EOS交叉计数和输出长度最小/最大值。

## 4 计时与显存字段定义

Engine.run 在 CUDA synchronize 后重置峰值统计，计时起点位于 prepare 前。TTFT 包括 prepare 分词、动态或固定选参分支、KV 获取/构建、请求 forward、首 token argmax 与同步；total 继续包含完整 decode 和文本解码。每个 token 的 argmax 后同步。`cache_build_seconds` 为缓存未命中时的资料编码区间（末端同步），`cache_get_seconds` 是 CPU 命中判断区间；`dynamic_selection_seconds` 对所有方法均记录参数分支的 CPU 时间，动态规则 choose 的耗时已经计入 TTFT/total。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:167`、`restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:226`。

`peak_allocated_bytes` 和 `peak_reserved_bytes` 为该方法进程在重置峰值后的总峰值，含模型及存活缓存。重置峰值保留分配器 reserved 池；请求开始时还可能保留上一资料缓存，cold 分支在编码前清空 self.encoded。bytes 后续可按 bytes/2^30 展示为 GiB。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:235`。

全部 18,116 条记录保存输入/输出长度、TTFT、total、cache_build/get、selection、resume rebuild、allocated/reserved；所检查数值字段均非负且有限的结果为 `True`。T/S 在 sequential/PCW 中为 null，APE 中有值；实际字段可用性按阶段列于 `runtime_field_availability.csv`。

## 5 搜索、冷、热、诊断和加载分层

search 的 KV 缓存身份忽略 T/S，复用同一资料编码，timing_condition=`search_immutable_KV`；正式 cold 每次编码，timing_condition=`cold`；Doc2Dial hot 要求同进程、同方法/配置/文档的 GPU KV 命中，timing_condition=`application_GPU_KV`。资料缓存不含 suffix，因此历史/当前问题每次进入 query。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:188`、`restored/output/runs/main80-20261114T041503/snapshot/APE/main80/methods.py:203`。

我们连接了 1,500 个 cold/hot 对，输出 token 一致 1,500 对，hot 在对应 cold 结束后启动 1,500 对，hot 均命中且本请求 cache_build=0 的对数 1,500。APE 的 1,000 条 hot 记录有 query 调用，调用数与输出 token 数逐题相同。sequential/PCW 的 stage_calls hook 未设置 query 阶段，query计数固定0；其新请求由代码完整 forward/decode 路径及独立 request_started/request_ended 事件核对。worker 每次 hot 都调用 engine.run，再检查 cold/hot token 列表。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/worker.py:86`。

attention 的12项配置在独立 worker/stage 中启用 observer，采样层0/15/31、头0/8/16/24、请求尾查询和实际decode步，其计时与常规请求分开。第1阶段仅审核该路径的配置和事件；完整注意力内容留到后续阶段。模型加载独立记录 model_loaded，含加载后 CUDA 同步，排除于请求计时之外。恢复资料 KV 的路径会把重建时间加回 TTFT/total，并另记 resume_cache_rebuild_seconds；本 campaign 的 resume_cache_rebuilt 事件数为 0。来源 `restored/output/runs/main80-20261114T041503/snapshot/APE/main80/worker.py:79`、`restored/output/runs/main80-20261114T041503/snapshot/APE/main80/worker.py:34`、`restored/output/runs/main80-20261114T041503/snapshot/APE/main80/worker.py:13`。

按 stage/method/cache/timing 的记录数、命中和重建分层见 `runtime_cache_layers.csv`；model_load_seconds 仅原值保存在 `runtime_workers.csv`。

## 6 启动、冻结、退出与提交顺序

事件包含 worker_started/model_loaded/worker_ended 各 16/16/16 条，stage_submitted 8 条。worker退出码全部为0的结果为 `True`。逐请求 started/ended 事件各 18,116/18,116 条，记录与唯一事件对连接 18,116/18,116。

准备审定时间为北京时间 2026-11-14T12:06:41+08:00；campaign 于 2026-11-14T12:15:05+08:00 启动；探索候选于 2026-11-14T14:03:50+08:00 固定；动态规则于 2026-11-14T14:17:49+08:00 冻结；独立长度检查/注意力/测试首个 worker 启动于 2026-11-14T14:17:54+08:00；最终提交时间为 2026-11-14T23:43:25+08:00。候选和规则的冻结时间来自相应 JSON，events 另记录各阶段提交。timeline 保存 UTC 与北京时间，便于跨来源连接。

`delivery.json.status=SUBMITTED`，`state.json.state=SUBMITTED`，delivery 保留 requires_PLANNER_review=True。启动快照和早期 launch observation 记录的是当时进度；当前提交状态据 delivery/state 读取。

## 7 第1阶段待补充与产物

- 硬件遥测待补充：GPU实际名称/UUID、总显存、驱动版本、功耗上限；现有方案和执行设备编号分别列示。
- 后续评分与统计阶段应继续按已冻结参数开展，保留上述生成长度和缓存分层；本阶段未运行 GPU、重新生成答案或修改证据。
- `runtime_integrity_issues.csv` 保存本脚本发现的最小异常证据；当前 0 行。
- 复现命令：`C:\Users\Osheepne\anaconda3\python.exe -B pc-analysis-output/scripts/audit_stage1_runtime.py`。
- 当前解释器 `C:\Users\Osheepne\anaconda3\python.exe`，Python 3.12.7；仅使用标准库，临时/缓存指向本阶段输出目录。全部输出在 `pc-analysis-output/stages/01_audit/`，本次未读取 attention 权重内容。
