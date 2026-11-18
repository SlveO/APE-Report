# APE 结题报告勘误关闭复核

2026-11-18，服务器 PLANNER 复核 `SlveO/APE-Report` 的 `analysis/main80-pc-results` 分支提交 `feb682807cdfc93e215e5f91afb4d3c6d88fa445`。结题报告正文、统一结果表的数值和统计口径及运行来源登记均为 `ACCEPTED`。第 25–26 行来源勘误关闭，报告与附表可用于结题。服务器侧技术终审完成，后续结题材料整理、学校模板、签章及正式提交由 PC 端接管。[勘误提交](https://github.com/SlveO/APE-Report/commit/feb682807cdfc93e215e5f91afb4d3c6d88fa445) · [上轮终审意见](https://github.com/SlveO/APE-Report/blob/cdd74e123b08da4071d8d6ef97171032d03c5616/final-report/server-reviews/20261118/recheck-45d0d7f/PLANNER_REVIEW.md)

## 变更与数值

相对 `cdd74e123b08da4071d8d6ef97171032d03c5616`，提交只修改统一结果表和生成器。逐列比对表头及 119 行 × 41 列，共 6 个单元格有改动，均位于 CSV 第 25、26 行的 `run_batch`、`run_code_version` 和 `run_evidence`。基准提交的附表内容与上一轮审定的 `45d0d7f` 一致。原审定的 119 项差值、238 个点估计及其来源、108 对 95% 区间端点、样本数量、记录顺序和统计口径均逐项一致。报告正文也与基准提交逐字节一致。[统一结果表](https://github.com/SlveO/APE-Report/blob/feb682807cdfc93e215e5f91afb4d3c6d88fa445/final-report/统一结果表.csv) · [生成器](https://github.com/SlveO/APE-Report/blob/feb682807cdfc93e215e5f91afb4d3c6d88fa445/final-report/build_unified_results.py)

11 条区间缺项与原表一致。第 18–20 行为公开汇总的描述性对照，第 21 行为探索描述，第 24 行为样本组成差异，第 79–84 行为单例诊断。CSV 第 73 行的 C3 主比较覆盖 150 题、68 个抽样组件，新版固定与动态的 F1 分别为 42.3108 与 42.4530，差 +0.1422 个百分点，95% 区间 [−0.2731，+0.6848]，3 胜、146 平、1 负。总体收益尚未确立，多块证据限于本批 Qasper。[上轮数值核验](https://github.com/SlveO/APE-Report/blob/cdd74e123b08da4071d8d6ef97171032d03c5616/final-report/server-reviews/20261118/recheck-45d0d7f/TABLE.md)

## 运行来源

第 25 行对应 T/S=0.7/1.0，第 26 行对应 T/S=0.5/1.0，每行均为 333 个配对条件。两行登记了历史批次 `main80-20261114T041503` 和实际前置修正探索批次 `main80-correction-20261117T060352`。修正探索的 5,328 条记录再次与服务器原始运行 `attempts.jsonl` 逐项核对，其中 1,248 条与修正批次实际生成完全一致，4,080 条与历史批次的同输入复用记录完全一致。第 24 行的历史批次登记亦复核一致。

附表两行准确指向服务器归档 `output/runs/main80-correction-20261117T060352/code-version.json` 及同目录 `snapshot/`。版本清单中所列 39 个快照文件均可定位，含实际 `APE/main80/worker.py`。PC 输入包 `followup2604-20261117/FOLLOWUP2604_ACCEPTED.zip` 与发布输入包逐字节一致，包内逐项来源可定位至 `provenance/corrected-exploration-for-candidate-reconstruction.jsonl`。[PC 输入包](https://github.com/SlveO/APE-Report/blob/ab816f5256f83892c83ea71bcf63a0b618e2f18d/followup2604-20261117/FOLLOWUP2604_ACCEPTED.zip) · [勘误行](https://github.com/SlveO/APE-Report/blob/feb682807cdfc93e215e5f91afb4d3c6d88fa445/final-report/统一结果表.csv#L25-L26)

## 重建和既有裁决

将目标提交的生成器放入服务器工作区隔离目录，用现有 Conda Python 和 `.overlays/core-tf44` 执行。生成器从固定发布 `0b4f099` 读取已发布源表，退出码为 0，stderr 为空，输出 119 项比较。重建 CSV 与目标提交逐字节一致。服务器原件目录 `logs/runs/20261118-final-report-review/recheck-feb6828/` 保存 `GENERATOR-attempt-001.stdout.txt`、`ERRATUM_CHECK-attempt-001.stdout.json`、对应 stderr 和重建 CSV。

FOLLOWUP2604 计算与数据验收、PC 统计验收为 `ACCEPTED`，历史 MAIN80 研究审核为 `REVISION_REQUIRED`。本轮裁决范围为结题报告与附表的来源勘误。
