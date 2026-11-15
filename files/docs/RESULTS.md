# APE 已审定结果

> 更新：2026-11-13。研究设计见[实验方案](EXPERIMENT_PLAN.md)，交接见[阶段文档](REPRODUCTION_STAGE.md)。

## 环境与模型基础

| 项目 | 已知事实 | 证据及限制 |
|---|---|---|
| APE代码 | 官方观察提交 `52513bcb0c3145aedec26636eb59f7c3deabb856`，保留工作区修改 | 实际Git仓库为 `APE/` |
| 基础环境 | Python 3.10.20、Torch 2.6.0+cu124、Transformers 4.57.6、FlashAttention 2.7.4.post1 | 既有APE Conda环境只读 |
| 官方core overlay | Transformers 4.44.2、tokenizers 0.19.1 | `.overlays/core-tf44/`及[冻结文件](../environment/locks/requirements-core-overlay-lock.txt) |
| 官方core与基础TF兼容性 | 原始core在4.57.6出现内部cache/mask接口错误；4.44.2 overlay路径实测通过 | 不为此修改外部环境 |
| 依赖检查 | 基础pip check通过；overlay全局检查报告继承的vLLM版本冲突 | core与vLLM不在同一进程混用 |
| 主模型 | `/home/shenxiaoyang/project/models/Llama-3-8B-Instruct`；2026-09-25恢复验收通过 | 工作区外只读；ModelScope master为获取分支，不声称不可变revision |
| GPU微型验证 | BF16、FlashAttention及官方三阶段调用曾实测通过 | [环境结果](../logs/runs/20260926-r1env-a1/ENV_RESULT.md.txt)；技术结果附历史过程偏离接收 |

## 官方8B功能短测

2026-09-26，R2两次独立进程均正常退出，使用同一本地Llama模型，BF16、2个context、87个输入token、恰好8个输出token，APE参数0.9/0.9。

| 观测 | 第一轮 | 第二轮 |
|---|---:|---:|
| prefix/context/query首层成功调用次数 | 1 / 1 / 8 | 1 / 1 / 8 |
| 加载调用墙钟（秒） | 3.949851 | 3.821084 |
| 请求墙钟（秒） | 1.567110 | 1.524208 |
| 请求峰值allocated（bytes） | 16,146,713,600 | 16,146,713,600 |
| 请求峰值reserved（bytes） | 16,311,648,256 | 16,311,648,256 |
| 退出码 | 0 | 0 |

生成文本一致，为截断片段 `A social with a large group of`。证据：[第一轮输出](../logs/runs/20260926-r2-a3/model.stdout)、[第二轮输出](../logs/runs/20260926-r2-a4/model.stdout)、两轮相邻stderr及退出记录。

这些时间包含对应历史入口的CPU准备、同步及插桩，不能作为正式加速比；短输出功能验证不代表问答质量。生成配置、mask/PAD等警告原样留在stderr。用户对首轮工具审批等待的解释已接收，不要求补跑。

## 当前研究状态

2026-11-14，新主模型准备阶段已审定的有限事实：

- ModelScope指定用户镜像master的13项Llama-3.1文件齐全，291个索引tensor、四权重头部结构及获取前后来源元数据一致；master仍是可变分支。下载失败及续传记录保留，外部环境未修改。
- 新模型在GPU0/BF16下四独立串行worker完成16次synthetic短生成，进程均退出0；每请求54个标准内容token、3–4个输出token（上限8）。冷/热/重建KV后的输出一致，可复用KV逐张量未变；APE首层实际阶段调用1/2/3，T/S改变实际logits。PCW单窗argmax相同并满足预置BF16容差，最大logit差0.34375，不声称逐位一致。
- 注意力9条记录覆盖层0/15/31、头0/8/16/24和实际query/decode步，token权重重建最大输出误差0.003183，小于记录容差。
- 不利事实：把synthetic问题从blue lake改为green river后，APE两标签仍答Azure Lake，sequential/PCW答Emerald River。新请求forward已执行；不以答案质量作为短测放行断言，不据此修改算法或补跑。

上述证据见[完整交付](../logs/runs/20261114-main80-methods-full/delivery.md)。其范围是有限正确性，不能替代正式长输入、质量和速度对照。

- 官方8B有限功能路径已有通过证据。
- 当前demo经工作区维护后移除资源限额逻辑；本轮只做静态核验，未运行GPU，维护后的入口没有新增运行成绩。
- 主线方案v3.0完整准备包已审定，MAIN80已下发；新模型有限功能事实见上文，正式成对质量和性能仍待完整交付审核。
- 2026-11-16核对：MAIN80于北京时间2026-11-14 12:15启动、同日23:43计算提交，当前SUBMITTED；8阶段共18,116条、运行失败0、未完成0，未触发预算缩样。正式成对质量、效率、搜索规则及结题结论仍未完成审核。[交付与实际状态](../output/runs/main80-20261114T041503/delivery.json)和逐题原始记录独立保存，自动汇总尚未转为审定质量成绩。历史Llama-3模型R2不作为Llama-3.1的运行证据。

历史原始日志保留实际内容；脚本与命令按文本证据保存。工作区整理及删除范围见[变更清单](../logs/runs/20261113-workspace-cleanup/CHANGES.md)。
