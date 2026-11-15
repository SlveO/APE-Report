# EXECUTOR / PREP-ASSETS / SUBMITTED

授权：U_MAIN80；评分overlay另见U_MAIN80_SCORING。最终技术验收由PLANNER完成。

- 指定ModelScope用户镜像master分支的13项必要Llama-3.1文件已取得；master可变，不声称Meta官方或不可变revision。
- model-manifest.json记录获取前来源/大小/观察revision；model-source-stability.json记录获取后逐项元数据比较；model-structure.json验证13项大小、291个索引tensor及四权重头部形状/offset连续结构。
- offline-config-tokenizer.json为CPU离线AutoConfig/AutoTokenizer实际证据；没有整模型或GPU运行。
- LongBench三子集各200唯一ID、Qasper原始train/dev/test、HotpotQA原始dev distractor及无损JSON、MuSiQue answerable dev、LITM20三位置和Doc2Dial文档/validation全部取得；datasets/raw-main80/ASSET_MANIFEST.json及data-structure.json列出来源/许可/变换/结构。
- 作者评分、prompt和PCW参考源码保存在references，以文本只读，不直接作为运行入口。
- 所有获取错误、初始流下载退出143、原始partial、Range分段状态保留。无外部修改、资源审查或摘要计算。未启动正式80小时计时。
- 评分依赖交付见scoring-execution.json、scoring-overlay-manifest.json及scoring-overlay-validation.json。首轮辅助断言失败保留；实际库不拒绝hyp/ref数量不齐，调用方必须先核对ID/数量/顺序。
