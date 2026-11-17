# 官方评分及模板来源

本阶段于2026-11-17查阅[DialDoc21评分入口](https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/sharedtask_utils.py)、[seq2seq模板](https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/subtask2/seq2seq_utils.py)和[SacreBLEU2.4.3](https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/metrics/bleu.py)。评分入口按agent目标构造dial_id与target_turn−1连接ID,调用语料级sacrebleu；模板保留last_turn标签、倒序历史、title和doc_context。

本campaign将完整资料置于历史前,包装为共用Llama聊天输入,逐方法保持同一适配。服务器SacreBLEU2.4.3签名为`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3`。PC未安装SacreBLEU,使用前置独立标准库实现和NumPy向量化充分统计计分,对照服务器六组数值/充分统计和边界fixture。

官方历史脚本会跳过未grounding轮次,当前实现保留全部历史；本批500目标的历史中该类轮次为0。原始参考/模板的前置500项检查全部PASS。
