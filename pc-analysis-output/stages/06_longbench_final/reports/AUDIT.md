# 第六阶段审核记录

已核对冻结510题/2550条、五方法逐ID连接、actual L/N与块长、参考答案、冷缓存、attempt、T/S和summary均值。保存score已在前置审核核对官方实现；本阶段复用原分数进行统计。

| 审核项 | 实际结果 | 证据表 |
|---|---|---|
| 样本及逻辑生成 | 每DS170题; baseline1530+final1020; 无复用logical_labels合并 | membership.csv; per_request_records.csv |
| 缺失/异常 | 失败0、空答案0、缺参考0、非有限/负值0; 全attempt1 | record_issues.csv; 本阶段脚本断言 |
| 原汇总对照 | 两个测试汇总部分30项F1、ID与分母一致 | original_summary_crosscheck.csv |
| 冻结和映射 | 全部T/S吻合rule; 正式worker在冻结后 | rule_mapping_counts.csv; model_load_workers.csv |
| Bootstrap | 450冻结组;413支撑连通组件;5法共同抽样 | bootstrap_membership.csv; bootstrap_diagnostics.csv |

| 现象 | 影响 | 证据 | 处理与补充 |
|---|---|---|---|
| 开发与正式实际N分布不同 | 全部动态规则评价 | 开发N4/8/16; 测试N1/2/3=177/104/229; 510/510网格外 | 冻结最近格映射; 单列实际N、原资料数和三个使用分支；在开发数据用自然分块N1–3搜索/验证后重新冻结, 用新增独立测试评价 |
| 正式动态仅使用三个分支 | 动态相对固定的收益来源 | 4K/4=146;8K/4=76;16K/4=288; 同参288条答案一致 | 按分支和实际N统计; 对全510的贡献逐项保存；针对8K/4的退化在证据完整开发集复核; 当前保持原规则 |
| 开发/测试单篇支撑资料重合 | 独立测试样本解释及源资料相关性 | 24篇支撑资料,29个MuSiQue测试ID; 单篇输入资料涉及59测试ID | 保留冻结510主表; 支撑重合29/不重合141的MuSiQue分层;450组与413连通组件CI；新增按单篇支撑资料隔离的独立测试 |
| 毫秒级计时差存在于连续worker运行 | 动态TTFT平均缩短2.33ms的归因 | 五方法各一个worker时段; 同参/同token288题也有计时差 | 报告时间差、输出长度、选参µs和worker顺序; CI反映样本来源变化；同参同输出请求随机交错重复测时, 分离选参及执行时序影响 |
| 短回答任务达到生成上限 | 质量与总响应时间 | 41/2550; MuSiQue5–8/170/方法; Qasper0 | 保留原分数; EOS/cap分组; 保存全部41条原答案与来源；在开发集对相同问题成对增加生成上限, 检查截断与重复输出 |
| 硬件登记缺少campaign遥测 | 跨硬件效率对照 | 方案RTX4090 48GB; 记录物理GPU0/BF16/batch1及软件版本; manifest无GPU型号遥测 | 分别列方案登记与实际执行配置, 保留单campaign效率；未来实验保存实际GPU型号/显存/驱动运行元数据 |

最小证据：membership.csv逐ID保存网格外标志、映射、资料交集和同参输出标签；generation_cap_records.csv保存41条上限原答案；dynamic_fixed_per_sample.csv保存510对分数、原答案、参数及源行。冻结候选/验证阶段的构造问题沿用前置审核清单，本阶段保留正式完整文档输入。

代码及数据来源：restored/output/runs/main80-20261114T041503/下的attempts.jsonl、summary.json、rule.json、snapshot/datasets/frozen-main80/；方法及评分依据同次snapshot/APE/main80/methods.py、rules.py、jobs.py、metrics.py。原始资料不写入，CPU分析产物位于本阶段目录。研究状态SUBMITTED，后续阶段未执行。
