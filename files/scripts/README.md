# MAIN80 入口

在工作区根执行：

```bash
bash scripts/run_campaign.sh check --config campaign.json
bash scripts/run_campaign.sh start --config campaign.json
bash scripts/run_campaign.sh resume --campaign <id>
bash scripts/run_campaign.sh status --campaign <id>
```

`check` 不启动模型；完整准备证据经PLANNER审定前，`start` 拒绝运行。启动后自动脱离终端、单实例串行运行GPU worker、本地HTTP监控和独立浏览器。物理GPU0，BF16，seed42，greedy，连续80小时含加载/故障/汇总。方法切换用新进程，不做GPU或磁盘预检。

`with_env.sh` 将HOME、Python、HF/Torch/CUDA、浏览器、临时及XDG目录限定在工作区；解释器及外部环境只读。core及另行授权评分overlay通过PYTHONPATH使用，不安装外部依赖。

数据准备：`python -B -m main80.freeze --sources data_sources.json --output datasets/frozen-main80`（同样必须通过with_env）。先解决600个来源映射、证据及补充资料审核；无法确认有效性的长度构造记N/A。`build_supplement_candidates.py` 仅生成待语义审核提案，不能自行认证补充内容。

`APE/experiments/Longbench/pred_parallel9.py` 转发同一campaign接口，默认数据由冻结清单的Qasper/HotpotQA/MuSiQue各200题确定。历史 `APE/demo_ape.py` 仅旧R2说明，不是MAIN80运行入口。历史脚本和命令只作文本证据。

下载工具见 [main80-assets](main80-assets/README.md)；准备状态与授权见[任务卡](../docs/REPRODUCTION_STAGE.md)。
