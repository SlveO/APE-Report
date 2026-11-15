# 最终调度代码只读复核 / SUBMITTED

角色：EXECUTOR `/root/method_cpu`。依据：PLANNER追加最终READONLY审核任务。范围：campaign、jobs、worker、common、report、monitor及所调用的rules/plots关键路径。本轮只读源码并写本报告，未执行模型/GPU/下载/安装/额外实验，未重跑已通过的CPU或整模型检查，未修改源码，不作ACCEPTED结论。

## 复核结论

本轮发现的空回答处理问题已反馈PLANNER，读取到的当前代码已将成功回答与完整空生成分开建索引：空生成仍是失败请求，以官方0分参与可评分outcome。`selected_records`及结果配对均改为使用同一outcome，纯空白回答也纳入；Doc2Dial热请求允许跟在可评分空cold之后重新生成。预算估计已改用`ledger.finished`排除不会重跑的终态数据错误，避免虚构剩余GPU成本。

除PLANNER已知、正在补齐的异常恢复计时及PREP签核到首发前的直接内容核对外，本次静态复核未发现新的会使正常start/resume立即失败、绕过规则冻结或依赖、或取消自有worker硬截止的P1。此为只读控制流结论，不替代PLANNER正在补充的非GPU回归证据与完整准备验收。

## 已落实的前次问题

| 核对路径 | 当前实现 | 结论范围 |
|---|---|---|
| `campaign.estimate/reduce_budget` | 独立显示真实ETA与预算forecast；未测hot/search可用已测cold代理上界；未知层不再抹掉已知部分成本；终态失败不再当待运行 | 缩样不再因末阶段hot未观测而全程失效；forecast仍如实标识代理与未知 |
| `campaign.reduce_budget` | 仅Doc2Dial完成后锁定LITM/Doc共同分析范围 | LITM完成后仍可共同缩样并保留已产出记录 |
| `campaign.snapshot/seal_generated` | candidates/rule保存完整内容副本，resume逐字比较 | 变化内容不能沿用旧成功键冒充同次恢复 |
| `campaign.control/launch` | ledger有system_error即拒绝继续/恢复；sample_error终态单列，成功才是done | 系统错误不能因resume跳过失败键后自动推进依赖 |
| `worker.execute` | 新进程遇hot而无encoded时只重建资料KV；不重新生成已完成cold答案 | 与整模型短测中已通过的restore_cache路径衔接 |
| `common.Ledger` | attempt最大值同时读取request_started事件 | 中断于ledger完成写入前的请求不会复用attempt号 |
| `campaign.control` | 每个自有worker设独立Timer，先终止再记事件；wait后先检查deadline，再状态IO | JSON/status/export不会阻塞终止信号的调度路径 |
| `campaign.selected_records` | 可评分失败与成功统一进入成对参数选择；不可评分数据错误显式排除并留事件 | 保留官方0分，不按模型空输出整题删掉参数比较 |
| `report.export` | 单一scored索引去重；LITM按三方法×三位置、Doc按三方法×冷热共同ID汇总，明确排除项；LITM有整体正确率 | 方法质量比较不再使用不同成功题集 |
| `campaign.confirm_browser/monitor.current_status` | 真实浏览器render receipt；过期controller状态为STALE，预算按当前时间显示 | Popen成功不再被当作页面已加载；旧RUNNING不再无限沿用 |
| `jobs`与冻结数据准备 | 自动阶段顺序不变；144参数配置保留，固定前缀缩样；PLANNER已提供全部允许nested级别覆盖实测 | 不重复运行该已完成的覆盖验证 |

调参公式仍按实际T/S去重、数据集及九格等权汇总，验证固定候选选择与每格严格优于才启用的逻辑没有发现回归。整模型方法/core/诊断路径沿用已交付短测证据，本轮不重测。

## 已反馈的最后补正点

1. `worker.execute`的恢复构建加时原先只在`Engine.run`正常返回后合并。若run以完整空生成抛SampleError，异常evidence中的TTFT/total尚未加`recovery_build`；同一异常分支还应核对hot/cold实际token一致性。PLANNER已说明正在补异常路径，保持失败状态并单列恢复构建及纯热时间即可，不需要额外GPU运行。
2. `seal_generated`先写artifact再复制sealed，二者之间极窄窗口中断时，resume会因一侧存在而拒绝。该行为是fail-safe，不会混用参数；可在限定补正中从同一冻结ledger重建并逐项比较后补足封存，保留原始中断证据。未据此要求重跑实验或扩大授权。
3. PREP审核后的源码/输入到首发前直接比较是PLANNER已明确正在增加的检查，本报告不重复列为新发现。

交付：本报告SUBMITTED。后续源码补正、非GPU验证范围及MAIN80是否READY由PLANNER维护。
