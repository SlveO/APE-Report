# 第八阶段审核记录

已核对500目标/3000生成、六条件配对、原始参考ID、完整历史、资料KV命中与热重新生成、规则L/N及独立评分充分统计。主聚类为366对话,资料172作为敏感性；N2资料CI因仅一簇留空。

| 项目 | 实际证据 | 产物 |
|---|---|---|
| 参考/历史/模板 | 500前置检查全部PASS,28连续agent、7参考历史重复、交集2 | preaudit_input_checks;target_inputs;history_condition_counts |
| 热请求 | 1500 hot全部命中、build/rebuild0,输出token/答案/参考一致;APE query调用完整 | cold_hot_pairs;hot_generation_evidence |
| 语料BLEU | 六组服务器分数及十项统计一致;独立向量化通过5fixture | original_summary_crosscheck;scorer_validation |
| 参考语言 | 1条英语请求/西语参考来自原始agent目标,主表保留并单列499敏感性 | reference_language_candidates;sources/reference_language_evidence.json |

| 发现 | 影响 | 证据 | 处理及补充 |
|---|---|---|---|
| 历史相关性与目标角色 | 参考连接、last_turn解释及CI独立单位 | 500目标来自366对话/172资料;28上一轮为agent,7参考历史重复,交集2 | 500主表保留;467普通条件/33标记条件分层;对话主CI/资料敏感性；原始角色条件分别复核,新增规范user→agent目标对照 |
| 热请求为相同目标再次提交 | 热缓存效率条件 | 1500热请求同worker GPU KV命中,build/rebuild0,答案token全同 | 资料KV与历史/请求/答案重新计算分开解释;保留事件行和阶段调用；增加同资料下不同目标轮次的连续应用热测 |
| N2仅一篇资料 | 规则映射和多块子组质量解释 | 493目标N1取1/1;7目标N2来自5对话/同一studentaid资料,取.7/.7 | 子语料单独累计计分;N2资料CI留空,5对话CI注明同资料内变化；增加独立N2及更大N资料的开发/验证/冻结后测试 |
| 高生成上限命中率且输出较参考长 | 语料precision和decode总响应 | 每法70%/76.2%/70%;13a系统长度为参考约5.64/5.75/5.64倍 | 记录cap/EOS,按实际长度分组,原始答案/参考保留；统一简洁模板及输出长度控制对照,保存结束原因/逐token时标 |
| 英文请求/西语参考 | 该目标词面匹配和案例解释 | 确认1085067334266d41636126e5b9ee9519_5原target6为西语,连接正确 | 原500保留;原始target/历史存最小证据;499目标敏感性另列；原标注人工语言核对后,另版本定义语言匹配的参考条件 |
| 毫秒差与连续worker顺序 | 小幅方法差的时延归因及峰值reserved历史 | 每方法一个worker,cold后立即hot;模型加载与请求分开 | 报告源簇CI、顺序、输出长度、两种加速比及显存口径；同资料生命周期内交错重复测时,分解prefill/query/decode |

原始证据路径为restored/output/runs/main80-20261114T041503/attempts.jsonl、summary.json、snapshot/datasets/frozen-main80/doc2dial.json和同次snapshot/APE/main80/{freeze,methods,worker,metrics,rules}.py。原始对话/资料路径由restored/data_sources.json登记。源行保存在逐请求和案例CSV,热事件行保存在热生成证据CSV；新增产物仅在pc-analysis-output。原始数据、规则及研究状态保留。
