# 官方评分来源查阅

本阶段于2026-11-16查阅作者main分支的[评测入口](https://github.com/nelson-liu/lost-in-the-middle/blob/main/scripts/evaluate_qa_responses.py)与[指标实现](https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py)。入口get_metrics_for_example在第57行先保留回答首行并去首尾空白，再调用best_subspan_em；指标按SQuAD式规范化后检验任一参考是否为回答子串。

保存的作者指标文本位于restored/datasets/raw-main80/references/litm_metrics.py.txt。本次主分数来自冻结生成及前置审核修正表，九个原答案实例分别复核全文和首行两次函数结果。在线main作为本次观察的作者入口，原始campaign证据保持只读。
