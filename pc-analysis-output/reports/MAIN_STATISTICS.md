# 主统计复算记录

本次从逐题答案独立复算LongBench F1，并按冻结测试ID连接方法。所有置信区间采用固定种子20261116、5000次配对聚类百分位重采样。LongBench在每个数据集内按原论文/证据组重采样，再对三个数据集等权；LITM时延以问题为簇共同保留三位置，Doc2Dial时延以对话为簇保留目标轮及缓存条件。区间描述该campaign样本组构成变化，硬件重复运行变动未采样。

原始记录18116，状态{'success': 18116}，重复逻辑键0，非法/缺失字段项0。LongBench逐题评分最大绝对差2.22e-16。

## LongBench

| scope | dataset | method | n | f1_percent | ci_low_percent | ci_high_percent |
|---|---|---|---|---|---|---|
| baseline600 | qasper | sequential | 200 | 46.308 | 41.576 | 51.089 |
| baseline600 | qasper | pcw | 200 | 45.366 | 40.205 | 50.458 |
| baseline600 | qasper | ape_default | 200 | 46.091 | 41.359 | 50.808 |
| baseline600 | hotpotqa | sequential | 200 | 58.032 | 51.956 | 64.114 |
| baseline600 | hotpotqa | pcw | 200 | 54.087 | 47.997 | 60.309 |
| baseline600 | hotpotqa | ape_default | 200 | 53.012 | 46.948 | 59.138 |
| baseline600 | musique | sequential | 200 | 31.955 | 25.888 | 38.102 |
| baseline600 | musique | pcw | 200 | 30.561 | 24.632 | 36.871 |
| baseline600 | musique | ape_default | 200 | 30.875 | 25.109 | 37.019 |
| baseline600 | macro_equal_dataset | sequential | 600 | 45.432 | 42.205 | 48.665 |
| baseline600 | macro_equal_dataset | pcw | 600 | 43.338 | 39.962 | 46.764 |
| baseline600 | macro_equal_dataset | ape_default | 600 | 43.326 | 40.102 | 46.644 |
| common510 | qasper | sequential | 170 | 46.819 | 41.650 | 51.828 |
| common510 | qasper | pcw | 170 | 45.335 | 39.712 | 50.973 |
| common510 | qasper | ape_default | 170 | 46.181 | 41.053 | 51.247 |
| common510 | qasper | ape_fixed | 170 | 43.720 | 38.661 | 48.795 |
| common510 | qasper | ape_dynamic | 170 | 44.344 | 39.018 | 49.506 |
| common510 | hotpotqa | sequential | 170 | 59.530 | 52.769 | 65.923 |
| common510 | hotpotqa | pcw | 170 | 56.121 | 49.484 | 62.795 |
| common510 | hotpotqa | ape_default | 170 | 55.462 | 48.634 | 61.996 |
| common510 | hotpotqa | ape_fixed | 170 | 54.878 | 48.166 | 61.466 |
| common510 | hotpotqa | ape_dynamic | 170 | 53.345 | 46.524 | 59.931 |
| common510 | musique | sequential | 170 | 32.273 | 25.512 | 39.157 |
| common510 | musique | pcw | 170 | 29.568 | 22.991 | 36.406 |
| common510 | musique | ape_default | 170 | 29.948 | 23.411 | 36.641 |
| common510 | musique | ape_fixed | 170 | 30.672 | 24.342 | 37.381 |
| common510 | musique | ape_dynamic | 170 | 30.731 | 24.399 | 37.450 |
| common510 | macro_equal_dataset | sequential | 510 | 46.207 | 42.663 | 49.765 |
| common510 | macro_equal_dataset | pcw | 510 | 43.675 | 40.037 | 47.394 |
| common510 | macro_equal_dataset | ape_default | 510 | 43.864 | 40.382 | 47.454 |
| common510 | macro_equal_dataset | ape_fixed | 510 | 43.090 | 39.624 | 46.615 |
| common510 | macro_equal_dataset | ape_dynamic | 510 | 42.807 | 39.360 | 46.465 |

## 配对差异

| scope | dataset | method_a | method_b | n | delta_percentage_points | ci_low_pp | ci_high_pp | win_b | tie | loss_b |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline600 | qasper | sequential | pcw | 200 | -0.941 | -4.242 | 2.232 | 43 | 100 | 57 |
| baseline600 | qasper | sequential | ape_default | 200 | -0.217 | -2.417 | 2.035 | 32 | 125 | 43 |
| baseline600 | qasper | pcw | ape_default | 200 | 0.725 | -2.322 | 3.737 | 54 | 105 | 41 |
| baseline600 | hotpotqa | sequential | pcw | 200 | -3.945 | -8.205 | 0.091 | 18 | 159 | 23 |
| baseline600 | hotpotqa | sequential | ape_default | 200 | -5.021 | -8.618 | -1.661 | 7 | 173 | 20 |
| baseline600 | hotpotqa | pcw | ape_default | 200 | -1.075 | -4.801 | 2.589 | 16 | 160 | 24 |
| baseline600 | musique | sequential | pcw | 200 | -1.395 | -6.630 | 3.947 | 26 | 145 | 29 |
| baseline600 | musique | sequential | ape_default | 200 | -1.081 | -5.788 | 3.694 | 23 | 154 | 23 |
| baseline600 | musique | pcw | ape_default | 200 | 0.314 | -3.437 | 3.980 | 21 | 161 | 18 |
| baseline600 | macro_equal_dataset | sequential | pcw | 600 | -2.094 | -4.624 | 0.353 | 87 | 404 | 109 |
| baseline600 | macro_equal_dataset | sequential | ape_default | 600 | -2.106 | -4.237 | -0.001 | 62 | 452 | 86 |
| baseline600 | macro_equal_dataset | pcw | ape_default | 600 | -0.012 | -2.035 | 2.041 | 91 | 426 | 83 |
| common510 | qasper | sequential | pcw | 170 | -1.484 | -5.175 | 2.002 | 38 | 80 | 52 |
| common510 | qasper | sequential | ape_default | 170 | -0.638 | -3.071 | 1.648 | 28 | 104 | 38 |
| common510 | qasper | sequential | ape_fixed | 170 | -3.099 | -5.982 | -0.436 | 35 | 85 | 50 |
| common510 | qasper | sequential | ape_dynamic | 170 | -2.475 | -4.686 | -0.502 | 9 | 135 | 26 |
| common510 | qasper | pcw | ape_default | 170 | 0.847 | -2.795 | 4.189 | 48 | 86 | 36 |
| common510 | qasper | pcw | ape_fixed | 170 | -1.615 | -5.767 | 2.256 | 42 | 88 | 40 |
| common510 | qasper | pcw | ape_dynamic | 170 | -0.991 | -5.340 | 3.169 | 46 | 79 | 45 |
| common510 | qasper | ape_default | ape_fixed | 170 | -2.461 | -5.345 | 0.370 | 28 | 95 | 47 |
| common510 | qasper | ape_default | ape_dynamic | 170 | -1.837 | -4.555 | 0.835 | 25 | 104 | 41 |
| common510 | qasper | ape_fixed | ape_dynamic | 170 | 0.624 | -2.416 | 3.509 | 41 | 89 | 40 |
| common510 | hotpotqa | sequential | pcw | 170 | -3.409 | -7.766 | 0.833 | 16 | 135 | 19 |
| common510 | hotpotqa | sequential | ape_default | 170 | -4.068 | -7.908 | -0.612 | 7 | 147 | 16 |
| common510 | hotpotqa | sequential | ape_fixed | 170 | -4.653 | -9.306 | -0.192 | 15 | 134 | 21 |
| common510 | hotpotqa | sequential | ape_dynamic | 170 | -6.185 | -11.170 | -1.314 | 15 | 132 | 23 |
| common510 | hotpotqa | pcw | ape_default | 170 | -0.659 | -4.690 | 3.132 | 15 | 135 | 20 |
| common510 | hotpotqa | pcw | ape_fixed | 170 | -1.244 | -6.052 | 3.628 | 20 | 128 | 22 |
| common510 | hotpotqa | pcw | ape_dynamic | 170 | -2.776 | -7.738 | 2.352 | 20 | 125 | 25 |
| common510 | hotpotqa | ape_default | ape_fixed | 170 | -0.584 | -3.817 | 2.708 | 14 | 145 | 11 |
| common510 | hotpotqa | ape_default | ape_dynamic | 170 | -2.117 | -5.751 | 1.348 | 13 | 144 | 13 |
| common510 | hotpotqa | ape_fixed | ape_dynamic | 170 | -1.532 | -4.577 | 1.213 | 5 | 157 | 8 |
| common510 | musique | sequential | pcw | 170 | -2.704 | -8.363 | 2.911 | 19 | 125 | 26 |
| common510 | musique | sequential | ape_default | 170 | -2.325 | -7.345 | 2.861 | 18 | 131 | 21 |
| common510 | musique | sequential | ape_fixed | 170 | -1.601 | -7.386 | 4.397 | 24 | 121 | 25 |
| common510 | musique | sequential | ape_dynamic | 170 | -1.542 | -7.329 | 4.452 | 24 | 121 | 25 |
| common510 | musique | pcw | ape_default | 170 | 0.380 | -3.247 | 4.008 | 17 | 139 | 14 |
| common510 | musique | pcw | ape_fixed | 170 | 1.103 | -3.636 | 5.738 | 24 | 129 | 17 |
| common510 | musique | pcw | ape_dynamic | 170 | 1.162 | -3.573 | 5.806 | 24 | 129 | 17 |
| common510 | musique | ape_default | ape_fixed | 170 | 0.724 | -3.633 | 5.239 | 18 | 139 | 13 |
| common510 | musique | ape_default | ape_dynamic | 170 | 0.783 | -3.583 | 5.281 | 18 | 139 | 13 |
| common510 | musique | ape_fixed | ape_dynamic | 170 | 0.059 | 0.000 | 0.181 | 1 | 169 | 0 |
| common510 | macro_equal_dataset | sequential | pcw | 510 | -2.533 | -5.159 | 0.028 | 73 | 340 | 97 |
| common510 | macro_equal_dataset | sequential | ape_default | 510 | -2.344 | -4.597 | -0.050 | 53 | 382 | 75 |
| common510 | macro_equal_dataset | sequential | ape_fixed | 510 | -3.118 | -5.764 | -0.438 | 74 | 340 | 96 |
| common510 | macro_equal_dataset | sequential | ape_dynamic | 510 | -3.401 | -6.100 | -0.796 | 48 | 388 | 74 |
| common510 | macro_equal_dataset | pcw | ape_default | 510 | 0.189 | -1.929 | 2.237 | 80 | 360 | 70 |
| common510 | macro_equal_dataset | pcw | ape_fixed | 510 | -0.585 | -3.194 | 1.982 | 86 | 345 | 79 |
| common510 | macro_equal_dataset | pcw | ape_dynamic | 510 | -0.868 | -3.572 | 1.816 | 90 | 333 | 87 |
| common510 | macro_equal_dataset | ape_default | ape_fixed | 510 | -0.774 | -2.783 | 1.311 | 60 | 379 | 71 |
| common510 | macro_equal_dataset | ape_default | ape_dynamic | 510 | -1.057 | -3.157 | 1.076 | 56 | 387 | 67 |
| common510 | macro_equal_dataset | ape_fixed | ape_dynamic | 510 | -0.283 | -1.698 | 1.036 | 47 | 415 | 48 |

## 效率口径

均值=Σx/n；P90使用NumPy线性分位数；配对加速比=同一请求A耗时/B耗时，表内分别给均值、中位数与均值比。差值统一B−A。所有秒字段保留原始单位；显存以GiB=bytes/2^30展示，含模型与存活缓存。描述表按stage/dataset/method/cache_mode/cache_hit分层，搜索复用、正式冷请求、应用热请求、注意力诊断分别统计。模型加载表由注意力审计脚本从events独立生成。

具体样本成员、逐题评分、字段字典及全部指标见tables CSV；表中source、line、scope和denominator提供追溯。
