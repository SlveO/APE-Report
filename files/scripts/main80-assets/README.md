# PREP-ASSETS

角色：EXECUTOR；授权：U_MAIN80；不执行 GPU、依赖安装、文件摘要或资源预检。所有目标、partial、元数据、缓存及临时文件均位于 APE 工作区。

模型来源是 ModelScope `LLM-Research/Meta-Llama-3.1-8B-Instruct` 用户镜像，获取分支为可变 `master`，不声称 Meta 官方发布或不可变 revision。仅取研究协议规定13文件。文件大小来自远端清单，权重按索引和 safetensors 头部结构检查。

工作区根运行：

```bash
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/fetch_assets.py model-small
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/run_asset_attempt.py --attempt-id new_unique_attempt
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/fetch_assets.py data
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/fetch_assets.py refs
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/inspect_assets.py data
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/inspect_assets.py model
```

网络命令需要可访问公开 HTTPS。`fetch_model_ranges.py` 使用8个并行请求；对原始流下载的 partial 只读取前缀并保留原文件；每段记录字节区间、状态及失败，完成后改名正式权重。不获取其他模型格式。其他成功文件按路径/大小复用，失败 partial 保留。不要启动两个相同目标下载器。

中断续传必须使用新的attempt ID，`run_asset_attempt.py`记录真实命令/起止/退出，跳过全部已完成段，并沿失败事件中已写入的cursor补足未完成段。原始失败及partial不会移除。随后finalizer只检查本次attempt的终态，不能因历史失败退出；仅在13项结构和可变来源前后元数据一致后提交。首次2026-11-13的37段URLError及finalizer错误保留；2026-11-14按用户“在中断处继续”授权修正辅助脚本并续传。

数据使用作者公开来源：LongBench v1 archive；Qasper v0.3 作者S3（地址见作者HF loader）；HotpotQA作者HF `hotpotqa/hotpot_qa` reformatted dev distractor parquet；MuSiQue v1.0 作者Google Drive；LITM作者GitHub20文档gold0/9/19；Doc2Dial作者GitHub v1.0.1 docs及validation。可变分支不声称不可变。来源/许可及变换记录见 `datasets/raw-main80/ASSET_MANIFEST.json`。

`inspect_assets.py` 只解出三份LongBench、Qasper JSON及MuSiQue answerable dev。gzip/zip格式用原始DEFLATE解码，不额外计算文件CRC或摘要。Hotpot parquet至JSON的记录在资产清单；原始parquet保留。

作者源码以 `.py.txt` 保存为只读参考，不直接运行。PCW为Apache2.0，LongBench/LITM仓库代码为MIT；数据分别遵守原始来源许可。Doc2Dial模板中的空references历史过滤不会自动带入主实验，主入口由PLANNER依v3协议适配。

日志位于 `logs/runs/20261113-main80-assets/`。`execution.json` 只记录EXECUTOR领取及交付；最终技术验收由PLANNER完成。

## 独立评分依赖

`U_MAIN80_SCORING` 单独授权向 `.overlays/main80-scoring/` 展开SacreBLEU2.4.3、portalocker2.10.1、colorama0.4.6、tabulate0.9.0和lxml5.3.0预编译wheel。来源为PyPI版本JSON及其公开wheelURL；按解释器兼容tags选择，不运行pip、编译或重算RECORD摘要。现有NumPy/regex只读复用。回退从本任务PYTHONPATH移除该目录，文件保留。

```bash
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/prepare_scoring_overlay.py
bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B scripts/main80-assets/validate_scoring_overlay.py
```

验证包含官方README API例子48.53、精确匹配100、空输出0、逆序参考降分、全部导入版本/路径及lxml native parse。SacreBLEU本身不拒绝hyp/ref数量不齐，调用方必须先检查样本ID、数量及顺序；首个错误预期assertion及修正记录保留于日志。
