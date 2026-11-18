# APE 结题报告服务器终审与 PC 交接

2026-11-18，服务器复核通过 `analysis/main80-pc-results` 分支提交 `feb6828` 的来源勘误。中文正文、统一结果表的 119 项数值及统计口径，以及第 25–26 行的运行来源登记均为 `ACCEPTED`。历史 MAIN80 研究审核仍为 `REVISION_REQUIRED`，FOLLOWUP2604 计算与数据、PC 统计均为 `ACCEPTED`。学校模板、签章和提交手续由 PC 端按学校实际要求办理。[勘误复核终审](recheck-feb6828/PLANNER_REVIEW.md)

后续由 PC 端统筹、实施、审核和提交。GitHub `SlveO/APE-Report` 的 `analysis/main80-pc-results` 分支用于 PC 端正式交付和留档，后续交付无需回传服务器复核。[PC 端接管提示词](PC_TAKEOVER.md)

## 审核原件

| 轮次 | 原件 | 用途 |
| --- | --- | --- |
| 勘误复核，`feb6828` | [PLANNER 终审](recheck-feb6828/PLANNER_REVIEW.md) | 最终裁决与运行来源闭环 |
| 第二轮，`45d0d7f` | [PLANNER 终审](recheck-45d0d7f/PLANNER_REVIEW.md) | 当轮裁决、附表勘误及服务器证据 |
| 第二轮，`45d0d7f` | [正文复查](recheck-45d0d7f/TEXT.md) | 五项主要反馈的落实与正文通过依据 |
| 第二轮，`45d0d7f` | [附表复查](recheck-45d0d7f/TABLE.md) | 119 项数值的复核、生成一致性和第 25–26 行字段要求 |
| 第一轮，`4d0131e` | [PLANNER 审核](PLANNER_REVIEW.md) | 首稿裁决与整体补正要求 |
| 第一轮，`4d0131e` | [正文审核](TEXT_REVIEW.md) | 首稿表述、方法口径和申报目标的逐项意见 |
| 第一轮，`4d0131e` | [附表审核](TABLE_REVIEW.md) | 首稿的数值、分母、范围及证据追溯核对 |

七件审核原件按原目录布局逐字节保留。原件中指向服务器证据的绝对路径和本地相对链接适用于审核时的服务器目录。PC 端可从下列仓库入口查阅现有交付。[结题正文](../../结题报告.md) · [统一结果表](../../统一结果表.csv) · [表格生成器](../../build_unified_results.py) · [FOLLOWUP2604 输入包](../../../followup2604-20261117/FOLLOWUP2604_ACCEPTED.zip)

## 提交与运行版本

| 提交 | 含义 |
| --- | --- |
| [`227116e`](https://github.com/SlveO/APE-Report/commit/227116e4a004153272640cfe31087a033861b8e9) | 历史 MAIN80 服务器输入导出版 |
| [`209b7d1`](https://github.com/SlveO/APE-Report/commit/209b7d1755bf38cfa87f517260503153e26a5641) | 历史 MAIN80 PC 分析交付 |
| [`ab816f5`](https://github.com/SlveO/APE-Report/commit/ab816f5256f83892c83ea71bcf63a0b618e2f18d) | FOLLOWUP2604 服务器输入包 |
| [`0b4f099`](https://github.com/SlveO/APE-Report/commit/0b4f099d94ec6198fc8603cc21cb8b638f89054f) | FOLLOWUP2604 PC 统计交付 |
| [`4d0131e`](https://github.com/SlveO/APE-Report/commit/4d0131e22a5e9bec07565de48f37e63dfbdd6d66) | 第一轮审核的结题初稿 |
| [`45d0d7f`](https://github.com/SlveO/APE-Report/commit/45d0d7fab6c72523eaf749821b6297be7233c87e) | 第二轮审核的结题修订稿 |
| [`feb6828`](https://github.com/SlveO/APE-Report/commit/feb682807cdfc93e215e5f91afb4d3c6d88fa445) | 第 25–26 行来源勘误及复核通过的报告版本 |

服务器实际执行使用各批次留存的代码记录。FOLLOWUP2604 批次为 `followup2604-20261117T085011`，worker 实际运行快照为 `001`。第 25–26 行涉及历史批次 `main80-20261114T041503` 和前置修正探索批次 `main80-correction-20261117T060352`。

## 历史 PC 补正提示词

以下是第二轮审核后的补正指令，现已执行并归档。当前工作按本页的 [PC 端接管提示词](PC_TAKEOVER.md) 推进。

```text
请同步 GitHub 仓库 SlveO/APE-Report 的 analysis/main80-pc-results 分支，阅读 final-report/server-reviews/20261118/README.md 及其中的第二轮 PLANNER 终审和附表复查。4d0131e 是首稿初审，45d0d7f 是修订稿终审。正文和统一结果表的 119 项数值及统计口径均已通过。服务器输入、PC 分析、正文提交和实际运行快照的映射见 README。

本次文书任务仅补正 final-report/统一结果表.csv 第 25–26 行的运行来源。先检查当前分支是否已完成勘误。已完成时只验证并回传对应提交和链接。仍待补正时，先改 final-report/build_unified_results.py 对应逻辑，再由生成器重生 CSV。

在 run_batch 中登记历史 main80-20261114T041503 和实际前置修正探索 main80-correction-20261117T060352。在 run_code_version 中注明修正探索的服务器执行快照已归档、PC 交付包未附。run_evidence 保留 PC 输入包 followup2604-20261117/FOLLOWUP2604_ACCEPTED.zip 中的 provenance/corrected-exploration-for-candidate-reconstruction.jsonl，并补记服务器归档 output/runs/main80-correction-20261117T060352/code-version.json 和同目录 snapshot/。PC 输入包中的 5,328 条逐项记录含 1,248 条前置修正生成及 4,080 条历史同输入复用。

验收范围为生成器重生、CSV 内容对照和运行来源勘误。确保差异限于第 25–26 行的来源字段，保留 119 项数值、样本与统计口径；本次不重跑实验、模型或统计。完成补正后，在 PC 提交并推送到原分支 analysis/main80-pc-results，回传提交 ID 和 GitHub 链接供服务器复核。后续项目交接也使用 GitHub。
```
