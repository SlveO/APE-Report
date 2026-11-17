# APE MAIN80 实验分析产物

本文件夹集中保存campaign `main80-20261114T041503`的本地CPU审核与分析产物：复现脚本、报告、统计表、图、来源摘录、环境及检查记录。八个实验阶段和前置审核均已完成，研究状态保持 **SUBMITTED**；补实验当前仅形成计划。

建议先读[中文内部汇报](reports/INTERNAL_BRIEFING_2026-11-17.md)，了解原计划、实际结果、差异原因和补实验优先级，再按下方索引查看阶段证据。

## 文件结构与作用

```text
pc-analysis-output/
├── README.md                  # 总说明与阅读入口
├── .gitignore                 # 本地获取副本、缓存和临时文件的排除规则
├── scripts/                   # CPU审核、统计、绘图及报告整理脚本
├── reports/                   # 跨阶段内部汇报、早期综合审核和统计说明
├── tables/                    # 跨阶段CSV：质量、效率、来源与补实验清单
├── figures/                   # 早期综合图表，PNG及SVG
├── environment.json           # 初期综合分析环境记录
├── main_results.json          # 初期综合统计的机器可读结果
├── stages/                    # 按实验阶段归档的完整材料
│   ├── 01_audit/              # 前置审核：数量、构造、划分、评分、运行
│   ├── 01_longbench_baseline/  # 实验1：完整600题三方法基线及官方对照
│   ├── 02_parameter_exploration/ # 实验2：九格×16组T/S探索
│   ├── 03_validation_rule/     # 实验3：验证、候选复用、回退与冻结
│   ├── 04_length_check/        # 实验4：6K/12K长度迁移检查
│   ├── 05_attention/           # 实验5：注意力权重、区域与答案案例
│   ├── 06_longbench_final/     # 实验6：共同510题五方法正式比较
│   ├── 07_litm/                # 实验7：LITM三位置及官方首行EM
│   └── 08_doc2dial/            # 实验8：Doc2Dial质量与冷/热资料KV
├── acquisition/               # 本地下载分片、获取记录和恢复副本，Git忽略
├── runtime-cache/             # 本地运行缓存，Git忽略
└── tmp/                       # 本地临时文件，Git忽略
```

各阶段按实际需要包含以下内容；具体文件以该阶段`README.md`为索引：

| 内容 | 作用 |
|---|---|
| `reports/` | 阶段结果、审核发现、处理方式及讨论 |
| `tables/` | 可读取CSV，保留样本范围、分母、单位、区间和来源字段 |
| `figures/` | 报告用静态图，主要为PNG及SVG |
| `sources/` | 规则副本、官方来源摘录和定向问题的最小证据 |
| `logs/` | 分析执行与交付检查记录 |
| `results.json`、`environment.json` | 阶段摘要、依赖版本、随机种子与统计设置 |
| `cache/`、`tmp/` | 本机缓存与临时文件，Git忽略 |

## 主要材料入口

| 材料 | 内容 |
|---|---|
| [中文内部汇报](reports/INTERNAL_BRIEFING_2026-11-17.md) | 八阶段结论、原计划与实际差异、优先补实验 |
| [关键数字及源表筛选](tables/internal_briefing_metrics.csv) | 汇报中的主要质量、效率估计及来源字段 |
| [开发构造使用范围](tables/internal_briefing_construction_usage.csv) | 冻结构造、实际使用与证据范围删除的数量对应 |
| [补实验清单](tables/internal_briefing_follow_up_plan.csv) | 优先级、待回答问题、对照设计及影响的结论 |
| [前置审核](stages/01_audit/README.md) | 18,116条记录、输入构造、划分、评分和运行核对 |
| [实验1：三方法基线](stages/01_longbench_baseline/README.md) | 完整600题质量、效率与官方成绩对照 |
| [实验2：参数探索](stages/02_parameter_exploration/README.md) | 144种配置、T/S热图和候选敏感性 |
| [实验3：验证与规则](stages/03_validation_rule/README.md) | 固定参数、九格回退和冻结时间链 |
| [实验4：长度检查](stages/04_length_check/README.md) | 10题、60组配对的新长度表现 |
| [实验5：注意力诊断](stages/05_attention/README.md) | 层/头/生成步骤、区域权重及原答案 |
| [实验6：LongBench正式比较](stages/06_longbench_final/README.md) | 共同510题五方法、自然块数与规则分支 |
| [实验7：Lost in the Middle](stages/07_litm/README.md) | 500题三位置、首行评分修正和效率 |
| [实验8：Doc2Dial](stages/08_doc2dial/README.md) | 500目标、corpus BLEU、冷/热GPU资料KV |

## 输入与复现

实际分析提交及参考提交均为`227116e4a004153272640cfe31087a033861b8e9`。分析输入来自仓库根目录的`restored/`恢复证据；按根目录`pc_analysis.py`和manifest恢复，原始`files/`、`parts/`及恢复证据保持只读。`acquisition/`保存历史获取过程的本地副本，阶段分析使用`restored/`。

在仓库根目录执行各阶段README列出的命令。分析使用既有Python 3.12.7，主要依赖NumPy、Matplotlib及部分审核使用的tokenizers、regex，精确版本见相应`environment.json`。配对Bootstrap通常为seed `20261116`、5000次，聚类单位和分母随任务记录；Doc2Dial本地使用独立充分统计实现核对服务器SacreBLEU 2.4.3设置。

根目录`reports/`、`tables/`、`figures/`和`main_results.json`保留早期综合分析材料；八阶段详细结果在`stages/`，跨阶段结论在中文内部汇报。Git跟踪正式分析材料，本地获取副本、缓存及临时文件保留在本机。
