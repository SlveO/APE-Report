# APE 结题报告修订稿终审

2026-11-18，复查 `SlveO/APE-Report` 的 `analysis/main80-pc-results` 分支修订提交 `45d0d7fab6c72523eaf749821b6297be7233c87e`。相对首稿 `4d0131e22a5e9bec07565de48f37e63dfbdd6d66`，该提交更新中文正文、统一结果表及生成脚本。[修订稿](../../../../exports/main80-pc-analysis/final-report/结题报告.md) · [首轮终审意见](../PLANNER_REVIEW.md)

## 裁决

正文和统一结果表的 119 项数值及统计口径均为 `ACCEPTED`。附表第 25–26 行的运行来源登记为 `REVISION_REQUIRED`。本修订稿可附下列勘误用于结题。历史 MAIN80 研究审核仍为 `REVISION_REQUIRED`，FOLLOWUP2604 计算与数据验收、PC 统计验收仍为 `ACCEPTED`。本轮裁决限于研究报告及其附表，学校正式模板、签章和提交手续未核验。[正文复查](TEXT.md) · [附表复查](TABLE.md) · [阶段与授权](../../../../docs/REPRODUCTION_STAGE.md)

正文的五项主要反馈已实质落地。C3 主比较为新版动态减新版固定 +0.1422 个 F1 百分点，95% 区间 [−0.2731，0.6848]，3 胜、146 平、1 负。总体收益尚未确立，多块证据限于本批 Qasper。附表逐行核对了 119 项差值、108 对区间端点及 238 个双方点值，来源涉及 33 张表。其余 11 项保留原区间缺项原因，生成脚本的输出与提交版 CSV 逐字节一致。[正文复查](TEXT.md) · [附表检查结果](TABLE_CHECK-attempt-002.stdout.json) · [审定事实](../../../../docs/RESULTS.md)

## 附表勘误

在 `final-report/统一结果表.csv` 第 25–26 行及 `final-report/build_unified_results.py` 相应生成逻辑中，把修正探索登记为服务器实际批次 `main80-correction-20261117T060352`。该批代码见 `output/runs/main80-correction-20261117T060352/code-version.json` 与同目录 `snapshot/`。两行的历史输入侧来自 `main80-20261114T041503`，修正探索汇总中含 1,248 条前置修正生成和 4,080 条历史同输入复用。以上记录已与服务器两批原始记录逐条比对。[修正探索验收](../../20261117-main80-correction/EXPLORATION_REVIEW.md) · [代码版本记录](../../../../output/runs/main80-correction-20261117T060352/code-version.json) · [运行快照](../../../../output/runs/main80-correction-20261117T060352/snapshot/)

PC 输入包的 `provenance/corrected-exploration-for-candidate-reconstruction.jsonl` 提供上述 5,328 条构建记录。实际批次名、运行代码版本和快照来自服务器留存。把这两处证据的可用位置写入来源字段并由脚本重生 CSV，即可闭合勘误。勘误仅涉及来源登记。[附表](../../../../exports/main80-pc-analysis/final-report/统一结果表.csv) · [生成脚本](../../../../exports/main80-pc-analysis/final-report/build_unified_results.py) · [PC 输入包](../../../../exports/main80-pc-analysis/followup2604-20261117/FOLLOWUP2604_ACCEPTED.zip)
