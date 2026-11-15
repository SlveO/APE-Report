# APE 项目代理协议

适用工作区：`/home/shenxiaoyang/project/ape`。遵守父级规则；用户批准的[研究协议 v3.0](docs/EXPERIMENT_PLAN.md)及本项目例外优先。实际 Git 仓库为 `APE/`。

## 角色与任务

- `PLANNER` MUST 维护研究范围、授权、任务依赖、验收及 `review_record`；可直接执行用户授权的文档整理、清理和代码维护。
- `EXECUTOR` MUST 只领取 `READY` 任务，记录领取、实际命令、起止、退出码和交付；只编辑自己的 `execution_record` 及交付区域，写入前读取最新内容并使用局部补丁。
- 状态为 `PLANNED → READY → RUNNING → SUBMITTED → ACCEPTED`，补正用 `REVISION_REQUIRED`，不可推进用 `BLOCKED`。任务状态与操作授权 MUST 分别记录。
- `EXECUTOR` MUST_NOT 自行验收或扩展任务。`MAIN80` 是完整执行包；领取后 MAY 按冻结依赖推进内部阶段，无须逐阶段人工放行。启动汇报不是提交；结束对话后后台任务仍为 `RUNNING`。
- `docs/EXPERIMENT_PLAN.md` MUST 维护研究协议，`docs/REPRODUCTION_STAGE.md` MUST 维护任务及授权，`docs/RESULTS.md` MUST 只登记审定事实。开始工作时 MUST 按需读取相关规则、文档及代码。

## 工作区与证据

- 所有创建、修改、移动、删除、缓存和临时文件 MUST 位于工作区内；MUST_NOT 经符号链接或相对路径向外写入。外部 Conda 环境、共享模型及父级规则只读。
- MUST 保护无关 dirty 变更及原版申报书；MUST_NOT 使用破坏性 reset、checkout、clean 或未经授权的 stash。
- 删除、清理、迁移已有资产 MUST 有具体任务授权，先保存精确路径清单，执行后记录内容和可恢复性。`U_MAIN80` 授权范围见阶段文档。
- Git 提交、推送、远程发布、上传及对外消息 MUST 单独授权。
- MUST_NOT 计算文件哈希或校验和，包括 SHA、MD5、Git blob 摘要；MUST 用清单、结构、样本 ID、数量及必要的逐项内容比较验证。
- MUST 保留真实原始输出和失败 attempt；历史脚本及命令只作文本证据，MUST_NOT 作为运行入口。MUST_NOT 记录令牌、密码、私有代理或无关个人信息。

## 环境与运行

- 使用物理 GPU 0（用户指定 RTX 4090 48GB），MUST_NOT 执行 GPU 情况审查、资源占用预检或磁盘/内存检查。运行中时延、吞吐及显存 MAY 作为观测，MUST_NOT 作为放行条件。
- MUST 使用已有 `/home/shenxiaoyang/miniconda3/envs/ape/bin/python` 及项目内 `.overlays/core-tf44`；MUST_NOT 修改工作区外依赖、驱动、系统 CUDA、编译器、服务或其他用户进程，MUST_NOT 使用 sudo。
- Python MUST 关闭用户 site 和字节码写入，显式设置工作区内缓存、临时目录及输出；离线模型 MUST 使用本地路径、`local_files_only=True` 和 `trust_remote_code=False`。
- 模型/数据获取 MUST 记录来源、获取分支或 revision、格式、许可、目标和失败处置。镜像 MUST_NOT 标成官方发布，可变分支 MUST_NOT 标成不可变 revision。
- 环境变更 MUST 先明确目标、准确版本、兼容性、验证和回退并获授权；本轮仅另行授权 `U_MAIN80_SCORING` 所列独立工作区评分overlay，其他依赖变更未授权。
- `MAIN80` MUST 从首个正式任务启动计连续 80 小时，包含加载、搜索、故障及汇总；准备仅含代码、获取和有限正确性短测，MUST_NOT 把研究运行归入准备。

## 实验与验收

- MUST 保存实际命令、已观察代码版本及 diff/代码快照、环境、模型路径、数据/组/样本 ID、参数、种子、起止、退出码、答案和评分。
- 比较 MUST 共用模型、任务内容与顺序、生成和评分规则；PCW/APE 的位置、前缀、编码和缓存差异 MUST 明示。
- 空输出、异常退出、输入/评分错位 MUST_NOT 标通过；import 成功 MUST_NOT 代替整模型证据。
- 普通辅助脚本错误 MAY 在授权范围内修正并留证；系统性 CUDA、接口、模型错误 MUST 保存现场并停止依赖任务，MUST_NOT 擅改目标、模型、精度、算法、144 配置或扩大运行次数。
- 官方旧模型 R2 已审定；MUST_NOT 将其作为新 Llama-3.1 或新公平对照证据。当前状态以阶段任务卡为准。
