# PREP-METHODS-FULL / SUBMITTED

角色：EXECUTOR `/root/method_cpu`。授权：`U_MAIN80`及READY任务卡。状态为SUBMITTED，待PLANNER审核；没有自行ACCEPTED。

准确入口：`bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B -u scripts/run_preparation_gpu.py`。实际执行通过已授权require_escalated访问GPU；未执行设备、占用、内存或磁盘预检。

实际起止：2026-11-14 03:28:27–03:29:04 UTC（北京时间11:28:27–11:29:04）。launcher与sequential、pcw、ape_default、ape_fixed四个独立严格串行worker全部退出0。仅执行一次任务，无重试、benchmark、正式研究计时、源码修改、依赖安装或摘要计算。

## 有限运行事实

- 模型为工作区本地Llama-3.1-8B-Instruct，离线只读加载、BF16、物理GPU0。
- 四方法各4次短synthetic生成，共16次；每请求54个标准内容输入token，输出3–4 token（上限8），不超过256输入。
- 四方法的标准任务内容token清单一致。PCW固有的独立BOS/合并KV/问题位置组织仍不同，不声称物理token位置或缓存组织完全相同。
- 四方法cold、hot及仅重建资料KV后的recovered_hot输出token逐项一致；可复用资料KV在query前后逐张量`torch.equal`；变更问题的请求重新执行query生成。
- PCW单窗与完整顺序forward的argmax相同，预置BF16 `assert_close(atol=0.3,rtol=0.01)`通过；实际最大logit绝对差为0.34375，不声称逐位相同。单BOS与最长窗口后起位断言通过。
- APE两个标签首轮第一层实际prefix/context/query调用为1/2/3，公共T/S绑定值核对通过；0.9/0.9与0.5/0.7的完整模型query logits最大差为3.046875。附加核对未生成额外答案。
- ape_fixed第一条cold请求采集9条完整token权重重建记录：层0/15/31，头0/8/16/24，查询末token及实际存在的decode步骤。最大输出重建误差0.00318300724029541，小于记录中的最小容差0.0310888671875；数值原文完整保留。

## 原始答案与限制

| 方法 | cold/hot/recovered_hot：blue lake问题 | changed_question：green river问题 |
|---|---|---|
| sequential | Azure Lake | Emerald River |
| pcw | Azure Lake | Emerald River |
| ape_default | Azure Lake | Azure Lake |
| ape_fixed | Azure Lake | Azure Lake |

APE变更问题后仍给出Azure Lake这一不利结果原样保留，未按质量换题或补跑。该任务证明有限调用、缓存与接口路径；不作为正式问答质量、长输入、速度或提升结论。changed_question_fresh_generation字段仅表示实际新请求执行，不表示答案正确。

## 证据

- `claim.json`、`execution.json`：领取与脚本SUBMITTED记录。
- `launcher-command.json`、`launcher-start.txt`、`launcher-end.txt`、`launcher-exit.txt`、`launcher.stdout`、`launcher.stderr`：真实入口及外层执行记录。
- `commands.jsonl`：四独立worker准确命令、工作目录、起止与退出码。
- `{sequential,pcw,ape_default,ape_fixed}.stdout/.stderr`：原始输出和警告。
- 各方法目录的`result.json`：标准内容ID、全部4条答案与token、原始时延/显存观测及检查值。
- `ape_fixed/token-attention.jsonl`：完整token权重、分支质量及重建误差。
- `code/`与`run_preparation_gpu.py.txt`：本次运行实际源码快照；未计算Git或文件摘要。

正式MAIN80未由本任务启动，80小时正式计时未开始。
