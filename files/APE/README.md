# APE 官方核心工作副本

方法来源：[APE论文](https://arxiv.org/abs/2502.05431)、[官方仓库](https://github.com/Infini-AI-Lab/APE)、[项目页面](https://infini-ai-lab.github.io/APE-Page)。

`ape/`包含官方Llama、Mistral和Gemma核心适配；`demo_ape.py`是本项目维护的Llama本地功能短测入口。已有8B功能结果见[项目结果](../docs/RESULTS.md)。

实验任务和运行范围由[主线方案v3.0](../docs/EXPERIMENT_PLAN.md)及[阶段任务r26](../docs/REPRODUCTION_STAGE.md)确定。MAIN80使用物理GPU0；不进行GPU情况审查、资源占用检查或文件摘要计算。所有写入位于项目工作区内。

`main80/`提供统一数据、生成、评分、恢复及预算控制；入口为工作区的 `scripts/run_campaign.sh`。`experiments/Longbench/pred_parallel9.py`转发同一受预算约束的入口，不再筛除短样本或默认多GPU。`experiments/lm-eval-harness/`保留上游参考用途。方法来源和模板适配见[说明](main80/SOURCES.md)，环境见[环境说明](../environment/README.md)。

引用：

```bibtex
@inproceedings{yang2025ape,
  title={APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding},
  author={Yang, Xinyu and Chen, Tianqi and Chen, Beidi},
  booktitle={ICLR 2025},
  year={2025}
}
```
