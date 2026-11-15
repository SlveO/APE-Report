# APE MAIN80 PC分析包

Campaign: `main80-20261114T041503`。计算已提交，研究结果仍待正式审核。原始文件未移动、删改或重跑。

包含本次全部逐题答案/评分/时延/显存、事件/命令/退出/原始stdout及stderr、搜索和冻结规则、完整token注意力及12幅SVG、启动快照、准备与审定证据、冻结及公开原始数据、当前代码/脚本、模型来源/配置/索引/分词器、环境版本/锁。模型权重、下载partial、运行环境二进制、浏览器/账户配置、原申报书和无关历史实验未纳入。

## PC端使用

目标仓库：[SlveO/APE-Report](https://github.com/SlveO/APE-Report)。Python3.10+，基础分析无第三方依赖、无需GPU。在PC终端执行：

```text
git clone https://github.com/SlveO/APE-Report.git
cd APE-Report
python -B pc_analysis.py verify
python -B pc_analysis.py summary
python -B pc_analysis.py restore --output restored
```

`summary`直接读取包内逐题数据和自动汇总；`restore`按manifest恢复原始相对目录，之后可用pandas、Jupyter或Excel分析。已有非一致目标文件会报错，不覆盖PC修改。

GitHub常规Git单文件上限为100MiB（[官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)）。这里超过45MiB的文件按字节拆分；相同内容经直接比较后存为别名。`restore`会重建全部原文件，包含200MB完整注意力文件。没有删减或压缩实验数值，没有计算文件摘要。`.gitattributes`禁止换行转换以保留原始字节。

恢复后的主要入口：

- `output/runs/main80-20261114T041503/attempts.jsonl`：18,116条逐题记录。
- 同目录`summary.json`、`results.csv`、`events.jsonl`、`rule.json`、`candidates.json`、`attention.jsonl`及`attention-figures/`。
- `datasets/frozen-main80/`：样本、分组、构造区间及嵌套顺序。
- `docs/EXPERIMENT_PLAN.md`、`docs/REPRODUCTION_STAGE.md`和`APE/main80/SOURCES.md`：协议、授权、来源与模板适配。

原始命令保留服务器路径作为证据；PC分析不执行这些GPU命令。快照是启动时的历史内容，当前计算状态以campaign的`delivery.json`为准。自动汇总不等同最终验收。原始数据来源及许可位于`datasets/raw-main80/ASSET_MANIFEST.json`，模型为ModelScope用户镜像master，非Meta官方发布或不可变revision。

用户已授权向上述仓库提交和上传本分析包。Git使用协议必需的内部对象标识和校验；不额外计算实验文件摘要。传输状态以仓库实际分支及工作区的独立上传记录为准。
