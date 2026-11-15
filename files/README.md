# APE 项目

唯一研究主线：在 Llama-3.1-8B-Instruct 上公平比较顺序编码、PCW、APE，研究按文档长度与块数选择 T/S 的九格规则，并评价位置问答及 Doc2Dial。完整方案授权实施，正式运行受连续80小时预算约束。

| 入口 | 内容 |
|---|---|
| [项目规则](AGENTS.md) | PLANNER/EXECUTOR、工作区和证据边界 |
| [研究协议 v3.0](docs/EXPERIMENT_PLAN.md) | 四类实验、数据冻结、参数与预算 |
| [任务卡 r26](docs/REPRODUCTION_STAGE.md) | PREP、MAIN80状态和授权 |
| [审定事实](docs/RESULTS.md) | 历史基础及新验证的实际范围 |
| [脚本](scripts/README.md) | 当前运行入口 |

代码在 `APE/`，解释器只读，core overlay 在 `.overlays/core-tf44/`。新模型目标 `models/Llama-3.1-8B-Instruct`；所有缓存、临时、日志和结果都在工作区内。GPU0；不做资源预检或摘要计算。

准备包已审定；MAIN80于北京时间2026-11-14 12:15启动、23:43完成全部8阶段，18,116条记录成功、运行失败0，未缩样。当前为SUBMITTED，正在结果审核与结题整理阶段；80小时为执行上限。见 [交付](output/runs/main80-20261114T041503/delivery.json)、[自动汇总](output/runs/main80-20261114T041503/summary.json)及[阶段记录](docs/REPRODUCTION_STAGE.md)。正式质量、效率及动态规则收益尚待验收。

旧R2只代表旧模型有限功能。日志精简先保存精确清单；[原版申报书](docs/share/[参考]ape项目申报书.doc)保持原件，前次[整理记录](logs/runs/20261113-workspace-cleanup/CHANGES.md)及[本次准备变更](logs/runs/20261113-main80-prep/CHANGES.md)保留。
