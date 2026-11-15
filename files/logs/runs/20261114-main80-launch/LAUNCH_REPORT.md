# MAIN80 启动汇报 / RUNNING

角色：EXECUTOR `/root/method_cpu`。授权：U_MAIN80；PREP已由PLANNER ACCEPTED，本次只领取并一次启动READY的完整执行包。此报告不是实验提交或最终验收。

- Campaign：`main80-20261114T041503`；controller PID `4187806`。
- 准确启动命令：`bash scripts/run_campaign.sh start --config campaign.json`，实际退出码0。
- 计时起点：2026-11-14T04:15:05.134575+00:00（北京时间2026-11-14T12:15:05.134575+08:00）。
- 硬截止：2026-11-17T12:15:05.134575+00:00（北京时间2026-11-17T20:15:05.134575+08:00）。未重启计时。
- 本机监控：http://127.0.0.1:42501/campaign/main80-20261114T041503/；已核对本campaign真实Chrome rendered receipt，不是伪造请求。
- 唯一观察命令：`bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B -u scripts/observe_campaign.py --campaign main80-20261114T041503`。
- 实际观察179.003973秒，退出码0；未为更多样本延长，观察器明确未停止后台。
- 窗口最后观测：2026-11-14T04:18:35.399824+00:00，RUNNING/baseline，成功159/18195、失败attempts 0。后续进度由后台持续更新，本报告不延长观察取样。

观测窗口内的速度快照（2026-11-14T04:17:22.825610+00:00）：

| 方法/条件 | 文档长度桶 | 观测数 | 平均秒/回答 | 实测范围秒 |
|---|---:|---:|---:|---:|
| sequential/cold | 4096 | 80 | 1.043 | 0.307–3.774 |
| sequential/cold | 8192 | 16 | 1.536 | 0.974–2.118 |
| sequential/cold | 16384 | 4 | 3.357 | 2.195–4.099 |

全程ETA暂不可可靠估计：PCW、APE及未来缓存条件没有足够正式观测。未将仅覆盖已知层的预算forecast当作全程预计结束时间。

证据：本目录claim、start-command/start-result、start.stdout/stderr、observation-command/observation-result、observation.stdout/stderr、observation-status-snapshot、campaign-start及launch-summary；campaign目录自动保留代码/输入快照、launcher/controller/worker命令与PID、原始逐题记录、日志、状态、浏览器凭据和后续交付。

所有准备源码、配置、模型、冻结数据保持未修改；未执行任何GPU或资源占用预检、额外模型运行、安装或摘要计算。仅按任务卡在MAIN80.execution_record登记领取及真实启动/观察，未改PLANNER验收区。后台实验及页面继续运行，任务保持RUNNING。
