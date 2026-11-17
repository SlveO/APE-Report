# 实验第一阶段：LongBench三方法基线与官方结果对照

## 1 本阶段做了什么

我们从baseline阶段1800条记录复算完整600题（Qasper、HotpotQA、MuSiQue各200题）的顺序编码、PCW和默认APE。三方法逐题使用相同原始完整资料、问题和参考，均为冷请求，默认APE T/S=0.9/0.9。F1保留0–1原值，本文乘100展示。前置审核另存于01_audit；本阶段直接交付基线质量、效率及官方对照。参数探索等后续实验尚待逐阶段开展。

## 2 本次质量结果

| 数据集 | 题数 | 顺序编码 | PCW | 默认APE |
|---|---|---|---|---|
| qasper | 200 | 46.308 | 45.366 | 46.091 |
| hotpotqa | 200 | 58.032 | 54.087 | 53.012 |
| musique | 200 | 31.955 | 30.561 | 30.875 |
| macro_equal_dataset | 600 | 45.432 | 43.338 | 43.326 |

默认APE相对顺序编码的宏平均下降2.106个百分点；PCW下降2.094个百分点；默认APE与PCW相差−0.012个百分点。HotpotQA的APE下降5.021个百分点，是宏平均下降的主要来源。Qasper下降0.217、MuSiQue下降1.081个百分点。

| dataset | method_a | method_b | delta_b_minus_a_pp | ci_low_pp | ci_high_pp | win_b | tie | loss_b |
|---|---|---|---|---|---|---|---|---|
| qasper | sequential | pcw | -0.941 | -4.242 | 2.232 | 43 | 100 | 57 |
| qasper | sequential | ape_default | -0.217 | -2.417 | 2.035 | 32 | 125 | 43 |
| qasper | pcw | ape_default | 0.725 | -2.322 | 3.737 | 54 | 105 | 41 |
| hotpotqa | sequential | pcw | -3.945 | -8.205 | 0.091 | 18 | 159 | 23 |
| hotpotqa | sequential | ape_default | -5.021 | -8.618 | -1.661 | 7 | 173 | 20 |
| hotpotqa | pcw | ape_default | -1.075 | -4.801 | 2.589 | 16 | 160 | 24 |
| musique | sequential | pcw | -1.395 | -6.630 | 3.947 | 26 | 145 | 29 |
| musique | sequential | ape_default | -1.081 | -5.788 | 3.694 | 23 | 154 | 23 |
| musique | pcw | ape_default | 0.314 | -3.437 | 3.980 | 21 | 161 | 18 |
| macro_equal_dataset | sequential | pcw | -2.094 | -4.624 | 0.353 | 87 | 404 | 109 |
| macro_equal_dataset | sequential | ape_default | -2.106 | -4.237 | -0.001 | 62 | 452 | 86 |
| macro_equal_dataset | pcw | ape_default | -0.012 | -2.035 | 2.041 | 91 | 426 | 83 |

差值统一B−A，胜/平/负按同一题的F1比较。区间使用seed20261116、5000次配对证据组Bootstrap；各数据集组数为142/200/194，宏平均对三个数据集等权。区间描述样本/证据组构成变化。宏平均APE−顺序的区间上界非常接近0，报告同时呈现差值、完整区间与逐题胜平负。

## 3 与官方论文的数值对照

官方来源固定为[APE论文arXiv v1表2](https://arxiv.org/html/2502.05431v1#S5.T2)。主表选同名Llama-3.1-8B模型：顺序取128K完整上下文行，PCW和APE取Contriever检索20个4000-word块的行。下表差值为本次−论文表值；两组使用各自公布的输入设置。

| dataset | method | local_f1_percent | paper_f1_percent | local_minus_paper_pp |
|---|---|---|---|---|
| qasper | sequential | 46.308 | 47.200 | -0.892 |
| hotpotqa | sequential | 58.032 | 53.460 | 4.572 |
| musique | sequential | 31.955 | 28.350 | 3.605 |
| macro_equal_dataset | sequential | 45.432 | 43.003 | 2.428 |
| qasper | pcw | 45.366 | 41.520 | 3.846 |
| hotpotqa | pcw | 54.087 | 49.470 | 4.617 |
| musique | pcw | 30.561 | 21.230 | 9.331 |
| macro_equal_dataset | pcw | 43.338 | 37.407 | 5.931 |
| qasper | ape_default | 46.091 | 43.030 | 3.061 |
| hotpotqa | ape_default | 53.012 | 55.410 | -2.398 |
| musique | ape_default | 30.875 | 26.880 | 3.995 |
| macro_equal_dataset | ape_default | 43.326 | 41.773 | 1.552 |

官方这三任务重算宏平均为顺序43.003、PCW37.407、APE41.773；本次分别高2.428、5.931、1.552个百分点。官方APE相对PCW在这三任务的宏平均高4.367个百分点，本次两者相近；官方APE相对128K顺序的三任务宏平均低1.230个百分点，本次低2.106。论文表内原Avg覆盖八个任务，这里统一重算三个任务平均。

差异解释先从输入组织展开：本次三方法共用完整文本，官方PCW/APE采用检索结果；本次全部实际N为1–3。前者保留全部原文和干扰信息，后者改变证据与背景的组成。其次，本次固定默认0.9/0.9，论文按验证集搜索参数，逐任务最终值未列出。本阶段先登记这些实际条件，参数与输入机制的进一步分析进入对应后续阶段。

[论文表6](https://arxiv.org/html/2502.05431v1#S6.T6)另报告完整长文本分段，但采用Llama-3-8B；三个任务的顺序/APE分别为Qasper32.83/39.14、HotpotQA44.24/50.11、MuSiQue20.47/25.79。模型和顺序上下文窗口与本次不同，作为补充来源列入official_reference.csv。

官方复现入口在观察提交52513bcb的pred_parallel9.py:99按公开length字段过滤<8000样本，:121分7500-token块，:183的query调用仅传model；同期定义要求model/temperature/scale。该入口默认任务和本次三任务也不同。代码筛选的本次子集范围另见official_code_filter_sensitivity.csv；主基线保留全部600题。论文表2/6的逐题ID和精确分母尚缺公开对应记录。下载来源与原文件保存在sources/。

## 4 本次冷请求效率

以下总体均值、中位数/P90在600题请求上计算；均值CI在数据集内按证据组重采样后等权聚合。

| method | metric | n | mean | median | p90 | ci_mean_low | ci_mean_high | unit |
|---|---|---|---|---|---|---|---|---|
| sequential | ttft_seconds | 600 | 1.509 | 1.818 | 2.268 | 1.475 | 1.542 | s |
| sequential | total_seconds | 600 | 1.813 | 2.044 | 2.435 | 1.769 | 1.855 | s |
| sequential | output_tokens | 600 | 11.075 | 5.000 | 31.000 | 10.251 | 11.883 | token_including_EOS |
| pcw | ttft_seconds | 600 | 1.389 | 1.678 | 2.030 | 1.360 | 1.418 | s |
| pcw | total_seconds | 600 | 1.635 | 1.874 | 2.251 | 1.598 | 1.673 | s |
| pcw | output_tokens | 600 | 8.905 | 5.000 | 21.000 | 8.239 | 9.559 | token_including_EOS |
| ape_default | ttft_seconds | 600 | 1.452 | 1.744 | 2.098 | 1.423 | 1.481 | s |
| ape_default | total_seconds | 600 | 1.978 | 2.153 | 2.764 | 1.925 | 2.031 | s |
| ape_default | output_tokens | 600 | 11.033 | 5.000 | 29.000 | 10.203 | 11.819 | token_including_EOS |

逐题加速比=A耗时/B耗时，大于1表示B更快；均值比另列。

| method_a | method_b | metric | ratio_of_means_a_over_b | paired_speedup_mean_a_over_b | paired_speedup_median_a_over_b | delta_b_minus_a_s | delta_ci_low_s | delta_ci_high_s |
|---|---|---|---|---|---|---|---|---|
| sequential | pcw | ttft_seconds | 1.086 | 1.063 | 1.080 | -0.119 | -0.125 | -0.115 |
| sequential | pcw | total_seconds | 1.108 | 1.136 | 1.087 | -0.177 | -0.196 | -0.158 |
| sequential | ape_default | ttft_seconds | 1.039 | 1.003 | 1.040 | -0.057 | -0.061 | -0.052 |
| sequential | ape_default | total_seconds | 0.916 | 0.917 | 0.968 | 0.166 | 0.138 | 0.193 |
| pcw | ape_default | ttft_seconds | 0.957 | 0.942 | 0.962 | 0.063 | 0.062 | 0.064 |
| pcw | ape_default | total_seconds | 0.827 | 0.832 | 0.907 | 0.343 | 0.311 | 0.375 |

TTFT计入预处理、参数分支和资料KV构建，总响应继续计入生成及文本解码；模型加载另表。峰值显存含模型与存活缓存，以GiB=bytes/2^30展示。本次PCW/APE在一张GPU逐块forward后合并KV。各数据集输出长度、EOS、上限命中见baseline_generation_limits.csv，实际N分层见baseline_quality_by_N.csv。

官方速度实验使用H100、vLLM、2K–128K上下文、固定256-token问题和生成、batch1/4；本次为实际问答、batch1、128/32输出上限及冷资料编码。论文4.5倍取自该长上下文速度实验；本次速度结论使用上面的600题实测配对值。硬件执行编号为GPU0，方案登记RTX4090 48GB，设备实际遥测缺项沿用前置审核记录。

## 5 案例、结论与阶段停止点

我们按默认APE−顺序的最大正差、最大负差及零差最小ID选取每数据集案例，原答案、参考、配置和源行号保存在baseline_cases.csv。质量和时延变化结合输出长度及实际N讨论。

本阶段得到的基线是：顺序编码宏平均F1最高；默认APE与PCW质量接近，默认APE在HotpotQA退化较大。官方对照把同名模型、输入方式、参数与速度设置分别列示；本次方法比较使用同600题配对。600题基线包括探索/验证/测试分割，下一正式五方法比较采用独立510题范围。

本阶段至此停止，研究状态保留SUBMITTED。前置审核的开发裁剪问题记录在调参材料中，本阶段输入为原始完整资料。后续各实验由用户逐阶段安排。
