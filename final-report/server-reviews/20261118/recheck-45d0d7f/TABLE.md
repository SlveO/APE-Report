# 结题稿统一结果表复核

2026-11-18。复核对象为报告仓库 `analysis/main80-pc-results` 分支的 `45d0d7fab6c72523eaf749821b6297be7233c87e`。PC 分析发布为 `0b4f099d94ec6198fc8603cc21cb8b638f89054f`，FOLLOWUP2604 服务器输入包为 `ab816f5256f83892c83ea71bcf63a0b618e2f18d`。工作树干净。从初稿 `4d0131e` 到本次提交仅变动正文、统一表和生成器。建议通过附表数值，补正第 25–26 行的运行来源。[统一结果表](../../../../exports/main80-pc-analysis/final-report/统一结果表.csv) · [初审记录](../TABLE_REVIEW.md)

## 数值和重建

- 119 个源表记录键及其顺序与初稿一致，119 项差值和 108 对区间端点与发布源表逐项一致。其余 11 项在发布源表中只有点差，见 CSV 第 18–21、24、79–84 行。初稿已有字段仅第 65–67 行的 `scope` 调整为各自数据集 50 题。
- 对 119 项的 238 个绝对点估计分别核对了源文件、记录、字段和单位换算，连同差值来源共涉及 33 张发布表。按各源表结构核对了方法、策略、实验范围、证据位置、评分规则和冷热请求的对应关系。逐行验证比较端减参照端等于登记差值，最大浮点绝对差为 `1.38e-14`。隔离重建的 CSV 与目标提交逐字节一致。[核对脚本](check_table.py) · [成功输出](TABLE_CHECK-attempt-002.stdout.json) · [隔离重建](generated/final-report/统一结果表.csv)
- 将第 28、36–47、65–78 行的 54 个绝对点估计分别对照既有服务器独立复算，最大绝对差为 `1.43e-14` 个百分点。第 73 行 C3 主比较固定和动态 F1 分别为 42.3108 和 42.4530，差 +0.1422 个百分点，95% 区间 [−0.2731，+0.6848]。第 70、71、78 行均为同一批确认题上的参照，绝对分数与七方法复算一致。第 36 行 B 主结果约为 39.7678 和 41.8366，差 +2.0688。同一行附带的 +2.0478 使用 60 个配对条件直接等权的描述口径。[服务器独立复算](../../20261118-followup-pc-review/numeric-recompute.json)
- 按数据集 × 长度 × 块数格内题均值再等权，用修正探索逐条配对表单独汇总第 25–26 行。每行 333 个配对条件均分布于 27 个数据集与格点组合。第 25 行 T/S=0.7/1.0 为 42.1221 和 42.0641，第 26 行 T/S=0.5/1.0 为 41.7097 和 42.1856，均与登记值在 `1e-12` 个百分点内一致。[修正探索逐条配对](../../../../exports/main80-pc-analysis/pc-analysis-followup-output/tables/a_exploration_matched_records.csv)

## 第 25–26 行的来源勘误

`run_batch` 应登记历史 `main80-20261114T041503` 和前置修正探索 `main80-correction-20261117T060352`。服务器输入包保留合并后的 5,328 条逐项结果，包内缺该独立批次 ID、对应执行快照和服务器审核记录。逐项与服务器原始输出比对，1,248 条匹配前置修正批次实际生成，4,080 条匹配旧批次同输入生成复用。第 24 行使用历史批次生成，修正版仅定义共同适用筛选范围。[CSV 第 25 行](../../../../exports/main80-pc-analysis/final-report/统一结果表.csv#L25) · [服务器修正探索审核](../../20261117-main80-correction/EXPLORATION_REVIEW.md)

`run_code_version` 应将“前置修正探索执行快照未单独登记”改为“前置修正探索服务器执行快照已归档，本交付包未附”。`run_evidence` 可补列该批次的 `output/runs/main80-correction-20261117T060352/code-version.json` 和 `output/runs/main80-correction-20261117T060352/snapshot/`，并保留交付内的逐项记录定位。修正点估计来源、样本、结果和区间可沿用。生成器对应追溯字段在第 336–342 行。[代码版本清单](../../../../output/runs/main80-correction-20261117T060352/code-version.json) · [服务器 worker 快照](../../../../output/runs/main80-correction-20261117T060352/snapshot/APE/main80/worker.py) · [表格生成器](../../../../exports/main80-pc-analysis/final-report/build_unified_results.py#L336)

## 复核留证

首次脚本在读取不含 `scope` 的比较表时触发 `KeyError`。校正来源表结构的读取方式后，第二次执行退出码 0，stderr 为空。本轮检查和补充汇总使用服务器现有 Conda Python、项目内 `.overlays/core-tf44` 和已存结果。[首次失败记录](TABLE_CHECK-attempt-001.stderr.txt) · [第二次 stderr](TABLE_CHECK-attempt-002.stderr.txt)
