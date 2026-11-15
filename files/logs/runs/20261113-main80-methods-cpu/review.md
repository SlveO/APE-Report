# PREP 只读代码审核 / SUBMITTED

角色：EXECUTOR `/root/method_cpu`。依据：PLANNER追加只读审核任务，研究协议v3.0。范围：`APE/main80/{campaign,jobs,worker,rules,report,freeze,data,common,monitor}.py`。本次只读源码与协议，未执行模型、GPU、下载、安装、生成研究记录或修改源码；已提交的3项CPU测试不重跑。以下为静态控制流推演，行号对应本次读取版本，PLANNER正在并行维护源码，修正后的状态由PLANNER复核。

## P1 / 预算估计与缩样在主体阶段不可用

- `campaign.py:166-177`：任意未来job的速度层缺少3个观测，整个estimate就返回None；`campaign.py:218-219`随后跳过缩样。预计任务包含所有未来Doc2Dial hot请求，而hot的实际观测直到最后阶段才出现，因此此前无法得到全程ETA，也无法根据已测主体速度触发预授权缩样。
- `jobs.py:23`读取正式样本的`_length`；`freeze.py:96,116-121`只为开发构造保存L，没有给正式LongBench/LITM/Doc2Dial写`_length`。这些job.key.L为None，被`campaign.py:147-148`预测成4K桶，破坏按实际长度分层。
- `campaign.py:227-228`在LITM完成后锁住litm和doc2dial共同缩样；此时热请求才可能取得速度，Doc2Dial即使允许500→100也无法缩量，可能直接预算BLOCKED。

最简修复：冻结全部正式输入的实际token长度；区分已测部分估计和未知部分，不因未来未测层丢弃已知超预算信号；根据可验证的正式观测建立剩余阶段估计，并明确未知覆盖。允许按预先冻结的共同前缀缩减LITM与Doc2Dial分析范围，已完成而被排除的LITM结果保留。只有影响后续参数冻结的探索/验证候选需要锁定。

## P1 / Doc2Dial冷、热之间中断无法恢复

位置：`campaign.py:283-287`、`worker.py:18,46-51`、`methods.py:172-175`。

冷请求成功后、热请求写入前中断，resume会跳过已成功cold。新worker的Engine没有encoded缓存，第一条pending hot必然抛出ValueError，并被登记为system_error；无法完成原来的同题热请求。

最简修复：恢复热请求时只重建匹配的资料KV，不重复成功答案；将恢复构建单列记录，随后重新生成热答案，并保留其真实cache条件与计时边界。

## P1 / 系统错误在恢复时被当作已完成而绕过

位置：`campaign.py:284-287,322-335`。

pending排除ledger中的所有已attempted键，包含system_error。系统错误导致本轮停止后，resume将跳过失败键并推进其余阶段；baseline等阶段没有完整性门槛，可能使依赖在系统错误未解决时继续。此行为也不符合“只跳过成功记录”的恢复契约。

最简修复：成功键才作为已完成。sample_error可按明确策略保留为终态并在交付中单列；system_error及未解决依赖必须保持停止，只有明确的补正/恢复策略才能新增attempt，不可通过一次resume隐式绕过。

## P1 / 缩样前缀没有验证144格与数据集覆盖

位置：`freeze.py:110-115`、`jobs.py:35-47,85-96`、`rules.py:33-35`。

冻结仅保证完整开发集合中每个数据集/长度至少有一题适用；构造N/A的样本仍在原始order中。35/20/10/6的前缀可能移除某数据集唯一适用的16K题，lower仍接受该缩样，执行后才在规则选择中因缺格失败。prefix包含三数据集不等于三数据集在全部九格均有有效输入。

最简修复：正式运行前为每个允许的嵌套级别验证全部144配置及三数据集有效覆盖，并冻结对应ID；不足以覆盖的级别不能下发为有效缩样。不得根据成绩换题，也不得从测试补开发题。

## P1 / 缺少独立硬截止，状态IO可推迟终止worker

位置：`campaign.py:306-320,337,345-346`。

worker的终止检查仅在controller轮询循环开头。wait抵达deadline后，controller先同步读取完整账本、重建全任务、写status等（315行），下一轮才检查时间；这些操作和阶段导出不受剩余时间限制。worker自身只在每题开始检查deadline，正在运行的一题也不能保证在80h停止。

最简修复：给本campaign自有worker设置独立的绝对截止watchdog/timeout，状态计算和导出不得阻塞终止信号；wait返回后先检查deadline，再执行状态IO。到点保留现场与部分结果，只终止自有worker。

## P1 / 热缓存恢复与动态规则的身份没有冻结到结果键

位置：`campaign.py:54-60,262,331,334,365-366`、`jobs.py:22-24`、`worker.py:21`。

resume只比较初始source-files快照；运行中生成的candidates.json和rule.json没有被复核。ape_fixed/ape_dynamic结果键的parameters为None，也没有规则内容标识。若这些文件内容变化，resume会保留旧规则下的成功键、以新规则生成其余键，混合为同一campaign。此处应直接比较完整内容，不使用摘要。

最简修复：候选和规则冻结后保存不可变内容副本，并在resume时逐项比较；或从相同已冻结选择范围的原始ledger重建并比较实际参数内容。规则变化必须显式形成新配置/新campaign，不可继续沿用旧成功键。

## P1 / 完全相同支撑文档集合可跨数据集拆组

位置：`data.py:163,201-203,224-230`。

证据组键包含dataset。HotpotQA与MuSiQue若共享同一完整support_documents集合但问题不同，会被强制分成两组并可能分至开发/测试；overlap_report又比较带dataset的group，完整证据组交叉无法被发现，虽然单篇文档重合会列出。本项与正在修正的标题匹配无关。

最简修复：对已经规范化且带来源命名空间的完整资料ID集合建立跨数据集证据组身份，再与相同规范化问题的约束求连通组；分层抽样仍可用dataset计数。分别保留完整集合与单篇资料的重合报告。

## P1 / LITM与Doc2Dial部分失败时没有共同有效ID比较

位置：`report.py:17-34`，尤其`29-31`。

quality按每个方法/位置/cache各自成功记录汇总。Doc2Dial再从该成功集合反向构造references，ID核对只能确认这个被裁小的集合，无法发现与计划500题或其他方法不一致。LongBench另有共同ID的paired表（43-48行），LITM和Doc2Dial没有；样本错误继续后，方法/位置/冷热得分可能来自不同题集。

最简修复：单方法部分统计可保留，但必须标PARTIAL及缺失ID；成对比较按固定已选ID与三方法/三位置或冷热条件共同有效的交集显式join，并报告排除项和覆盖。LITM还应输出三个位置以及共同题集的整体正确率，当前只输出分位置组。

## P2 / 中断请求重用attempt编号

位置：`worker.py:39-43,60-64`、`common.py:85-97`。

request_started落盘后、ledger.add之前进程退出，原attempt仅存在events；Ledger.attempts只读取完成记录，resume会再次使用同一个attempt号。原始事件虽保留，两次实际生成不能被唯一标识。

最简修复：恢复时将未匹配结束事件的开始记录登记为interrupted attempt（保留原始内容），再递增编号；或在请求开始就持久分配唯一attempt，并让完成记录引用它。

## P2 / 浏览器加载和失联状态未纳入监控确认

位置：`campaign.py:99-113,267-269`、`monitor.py:75-85`、`campaign.py:410-412`。

start_browser只确认Popen，controller从不检查进程结果或browser-seen receipt，浏览器失败仍会直接执行worker。controller意外退出时，独立HTTP服务和status命令继续显示持久文件中的RUNNING，页面只重复读取同一旧状态，没有心跳失效标记。

最简修复：以真实浏览器render receipt确认campaign页面已加载（不得伪造receipt），未确认则明确记录；UI/CLI对过期updated_at显示状态失联，不把旧RUNNING当当前运行事实。启动观察仍遵守最多180秒。

## 已核对的正确路径

- `jobs.py:44-47`按实际T/S合并default/global/cell逻辑标签，一次生成可合法复用。
- `rules.py:39-70`按实际T/S、数据集和九格等权选探索候选；验证的固定候选选择、每格严格优于才启用、否则回退的公式与协议一致。对不成对数据显式报错。
- `freeze.py:53-65`保存目标turn之前的完整历史；未沿用参考脚本过滤无grounding历史的行为，并在模板适配字段明示。
- 当前读取的`campaign.py:59`已包含tokenizer.json快照；此前缺失项已由PLANNER修正，本报告不重复登记为未修正问题。

本报告状态为SUBMITTED，未作ACCEPTED结论；是否修正、有限验证范围及准备验收由PLANNER维护。
