# 结题稿统一结果表复核

本次复核的提交为 `4d0131e22a5e9bec07565de48f37e63dfbdd6d66`。在独立报告仓库 `exports/main80-pc-analysis` 检查工作树干净后，以 SSH 获取 `SlveO/APE-Report` 的 `analysis/main80-pc-results` 分支，确认远端跟踪引用为指定提交，并从 `0b4f099` 仅快进至该提交。复核前后工作树干净。两个提交间涉及的变动只有 `final-report/` 下的三个新文件，证据源表未变动。

## 逐行核对

- 已解析 119 条比较，覆盖 22 张 `0b4f099d94ec6198fc8603cc21cb8b638f89054f` 的源表。119 个源定位均唯一，逐表按表头后首条数据为记录 1 定位。119 个差值与 108 对区间端点逐字匹配各行指定的源字段。其余 11 行为公开总量比较、探索描述或单例诊断，源表只有点差。数值格式、区间顺序和来源引用通过核对。[统一表](../../../exports/main80-pc-analysis/final-report/统一结果表.csv)
- 逐行核对了唯一题数或目标数、成对条件数、规则版、支撑组和统计抽样组件。基线按逐题成员聚类复算为 142／200／194 组，旧正式测试为 116／170／164 组；旧 29 题的重合分层来自 MuSiQue。C3 主分析为 150 题／150 对、15／46／7 个组件，Qasper N2 为 17 题、6 个实际覆盖组件，Bootstrap 从 Qasper 全部 15 个组件共同抽样。LITM 为 500 个题级单位，三位置汇总 1,500 个条件；Doc2Dial 的 500 个目标轮次来自 366 个对话、172 篇资料，主区间按对话，敏感性按资料。[独立核对脚本](check_unified_table.py) · [执行输出](TABLE_CHECK-attempt-001.stdout.json)
- 24 行与既有服务器独立复算的差值和区间在 `1e-12` 个百分点内一致。覆盖 A 主比较、B 12 个等权宏平均范围、C3 主比较、各数据集、Qasper 三种 N 与 N 等权，以及默认 APE／历史固定关键参照。C3 新版动态减新版固定为 +0.1422［−0.2731，0.6848］，同批对默认 APE 为 −3.3786［−6.7809，−0.4737］，对历史固定为 −3.6188［−8.4072，−0.2035］；后两项区间未做多重比较校正。[服务器独立复算](../20261118-followup-pc-review/numeric-recompute.json)

## 需补正的口径和可追溯性

1. 摘要或第 2.2 节列明历史 MAIN80 的研究审核状态 `REVISION_REQUIRED`，第 2.2 节标题使用“原 MAIN80 冻结测试”。明确 29 个测试 ID 与开发共享单篇支撑资料；仅 C3 使用按资料连通组件隔离的独立确认设计。附表最好直接标示历史批次和审核状态。[初稿第 39–41 行](../../../exports/main80-pc-analysis/final-report/结题报告.md#L39) · [阶段记录](../../../docs/REPRODUCTION_STAGE.md#L526)
2. 第 45 行明确 432→359 属于 90 题、五种长度的 450 个候选，包含 6K／12K 预审。4K／8K／16K 的主九格适用题长为 258→211，探索 139→111、验证 119→100。补写已确认的 BPE `ignore_merges` 零长度 offset 根因，73 项退出的合并原因继续按现存粒度陈述。[服务器审定](../../../docs/RESULTS.md#L49) · [前次交接勘误](../20261118-followup-pc-review/PLANNER_REVIEW.md)
3. 自然选参段应报告 C2 的 36 题用途和结果。C2 分数按数据集等权、Qasper 内按 N 等权，候选固定相对默认为 +2.4403 个 F1 百分点。Qasper N2 为 4 题，候选减固定 +0.8164 个百分点。两项为选参点估计。随后单独列 C3 预冻结参数的确认结果，注明 C3 按数据集内题均值再等权。[C2 原表](../../../exports/main80-pc-analysis/pc-analysis-followup-output/tables/a_natural_validation.csv) · [冻结配额](../../../docs/REPRODUCTION_STAGE.md#L29)
4. B 阶段正文和统一表的三数据集等权总差为 +2.0688。首次验收直接合并 60 条成对条件得到 +2.0478；两种汇总因数据集 4／3／3 题数不同而具有不同权重。正文可简短明示两种口径。6K 和 6K／8 的分层区间为未经多重比较校正的探索性区间。[B 分析](../../../exports/main80-pc-analysis/pc-analysis-followup-output/reports/B_LENGTH.md)
5. 统一表当前可用于定位比较。按服务器交接补齐每个主张关联的实验批次、模型、实际运行代码版本字段、数据来源、生成与复用口径、明确权重和参照／比较的绝对点估计。保留完整服务器输入提交、PC 分析发布提交及对应运行记录的独立字段。统一表第 65–67 行分别为一个数据集的 50 题，`scope` 目前都显示“新150题（各集50题）”，可写成“新150题确认批次，本行仅［数据集］50题”。[统一表](../../../exports/main80-pc-analysis/final-report/统一结果表.csv#L65) · [结题交接字段](../20261118-followup-pc-review/PLANNER_REVIEW.md)

## 核对记录

执行者使用项目指定 Conda Python、只读项目 overlay 和标准库。脚本从指定 Git 提交读取数据，按源列、各成员表和冻结组统计对照分母；使用既有服务器复算产物验证覆盖的 24 行。旧批次的源表区间按已发布结果逐项核对。脚本未复跑旧分析、模型或 GPU。首次执行退出码 0，stderr 为空。[命令输出](TABLE_CHECK-attempt-001.stdout.json) · [stderr](TABLE_CHECK-attempt-001.stderr.txt)

```bash
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/.overlays/core-tf44" \
  XDG_CACHE_HOME="$PWD/logs/runs/20261118-final-report-review" \
  TMPDIR="$PWD/logs/runs/20261118-final-report-review" \
  /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B \
  logs/runs/20261118-final-report-review/check_unified_table.py
```
