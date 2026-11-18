# FOLLOWUP2604 PC分析交付

本目录保存修正选参核对、6K/12K迁移和150题独立确认的CPU统计。入口为[中文总结](reports/FOLLOWUP_REPORT.md)；服务器端可先核对[主要结果](tables/headline_results.csv)、[C确认配对](tables/c_paired.csv)及[统计配置](STATISTICAL_CONFIG.json)。

主结论：修正九格固定从0.7/1.0改为0.5/1.0，6/9格冻结规则改变；自然固定1.0/0.9，动态仅N2改为0.9/1.0。B总体差+2.0688pp，95%CI[-2.3551,7.0006]；C主要比较+0.1422pp，CI[-0.2731,0.6848]，3胜146平1负。C新版动态低于默认及MAIN80固定的同题成绩，详见总结。不将跨零区间解释为等效。

## 文件结构

```text
pc-analysis-followup-output/
├── README.md                    # 本交付入口与复现
├── STATISTICAL_CONFIG.json      # 权重、组件、随机流、CI和重抽规则
├── environment.json             # 实际提交、环境及状态
├── a_results.json               # 修正探索/A验证与自然开发规则摘要
├── b_results.json               # B长度迁移摘要
├── c_results.json               # C独立确认摘要
├── scripts/                     # 自编CPU脚本，包内历史脚本不执行
├── tables/                      # a_/b_/c_主/分层CSV、来源、案例、诊断
├── figures/                     # 10组PNG/SVG及对应CSV数据
├── reports/                     # 中文总报告、A/B/C阶段及协议审核
├── logs/                        # 包恢复、分析配置与检查记录
├── cache/                       # 本地绘图缓存，Git忽略
└── tmp/                         # 本地临时目录，Git忽略
```

## 阅读与核对

- [A修正分析](reports/A_CORRECTION.md)：输入变化/适用变化/参数变化分开；[规则表](tables/a_rule_changes.csv)。
- [B长度迁移](reports/B_LENGTH.md)：10题60配对，自身冻结支撑组；[六格与数据集表](tables/b_quality.csv)。
- [C独立确认](reports/C_CONFIRMATION.md)：七方法；[主/分层质量](tables/c_quality.csv)、[配对CI](tables/c_paired.csv)、[组件成员](tables/c_support_components.csv)。
- [协议与独立复核](reports/PROTOCOL_AUDIT.md)：组件、来源、冻结顺序与标签分母。
- [逻辑计数对照](tables/logical_count_reconciliation.csv)：2824来源容器、3042视图；218额外标签中A61，C2+C3为157。
- [C重抽诊断](tables/c_bootstrap_diagnostics.csv)：主分析独立，Qasper分层5069提出/69拒绝/5000有效。
- [C不利参照比较](tables/c_adverse_reference_comparisons.csv)、[典型案例](tables/c_cases.csv)、[实际897次生成](tables/c_physical_generations.csv)。
- [交付检查](logs/delivery_validation.json)；[执行失败与处理记录](logs/failures.json)。检查脚本的B字段适配错误已修正，统计估计未受影响。
- [图表数据索引](tables/figure_index.csv)：10组PNG/SVG与源CSV对应。

## 输入与范围

本次分析使用服务器交付版本 **`ab816f5`**（`analysis/main80-pc-results`），该提交包含验收包和任务说明。166 项验收文件解压至 `restored/followup2604/FOLLOWUP2604_ACCEPTED/`，通过 `PORTABLE-PATHS.json` 读取包内相对路径。A 阶段和历史对照还读取原 MAIN80 的恢复证据及 `pc-analysis-output/`。[PC_TASK](../followup2604-20261117/PC_TASK.md)

分析以冻结规则和已有回答为输入，供服务器端复核。服务器计算与数据交付状态为 ACCEPTED，35 项验收检查全部通过。原始运行状态为 SUBMITTED，包内 MAIN80 整体状态为 REVISION_REQUIRED。

## 复现命令

从仓库根目录运行，使用既有Python3.12.7、NumPy1.26.4、Matplotlib3.9.2，无新增依赖。完整配置见environment.json、STATISTICAL_CONFIG.json及各分析日志。若旧MAIN80证据未恢复，先按仓库原有说明恢复后再执行A及B历史连接。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONNOUSERSITE="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -s -B pc-analysis-followup-output/scripts/prepare_inputs.py
python -s -B pc-analysis-followup-output/scripts/analyze_a_correction.py
python -s -B pc-analysis-followup-output/scripts/analyze_b_length.py
python -s -B pc-analysis-followup-output/scripts/analyze_c_confirmation.py
python -s -B pc-analysis-followup-output/scripts/plot_c_confirmation.py
python -s -B pc-analysis-followup-output/scripts/assemble_report.py
python -s -B pc-analysis-followup-output/scripts/validate_delivery.py
```

A/B脚本同时生成其图表与阶段报告；所有新增脚本、表、图、缓存和日志均位于本目录。恢复脚本遇已有不同证据会保留现场并登记，不覆盖。
