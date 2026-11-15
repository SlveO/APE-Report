# APE 任务与授权

版本：r26，2026-11-13。协议：[v3.0](EXPERIMENT_PLAN.md)。代理消息及任务记录 MUST 使用结构化 Markdown/YAML 和 `PLANNER`/`EXECUTOR` 约束。

## 授权

`U_MAIN80`：本轮用户要求实施《APE单主线80小时执行方案》。授权规则/文档/代码维护，删除 `docs/AI_INFRA_LAB.md` 及所有副线引用，按先列清单方式清理方案指定冗余日志；保留有效环境、模型恢复、R2原始证据、清理记录、失败简述、其他dirty变更和原版申报书。授权工作区内获取方案指定新模型及四类实验/原始映射数据，有限正确性短测、完整 MAIN80 无人值守执行包及本机监控/浏览器。未授权外部依赖变更、提交、推送、发布、上传或外部消息。GPU0，不做资源审查，不计算摘要。

历史 `U_MAINTENANCE_20261113`：此前已授权工作区整理，证据见 [清单](../logs/runs/20261113-workspace-cleanup/CHANGES.md)。历史旧模型 R2 不扩展为新模型证据。

`U_MAIN80_SCORING`：本轮用户明确回复“授权工作区内独立评分依赖（推荐）”。仅向 `.overlays/main80-scoring/` 展开 PyPI 预编译 wheel：SacreBLEU 2.4.3、portalocker 2.10.1、colorama 0.4.6、tabulate 0.9.0、lxml 5.3.0；复用现有 NumPy/regex，不修改外部环境，不重算摘要。验证 BLEU 官方样例及 ID 对齐；回退从本任务 PYTHONPATH 移除独立目录并保留文件。该子任务由 PREP-ASSETS EXECUTOR 执行，来源及命令记录于其独立交付区。

## 准备任务

```yaml
protocol: ape-plan-exec/v1
task_id: PREP
plan_version: v3.0
owner: PLANNER
status: ACCEPTED
depends_on: []
authorization_ref: U_MAIN80
inputs: [docs/EXPERIMENT_PLAN.md, docs/RESULTS.md, APE/ape/ape_llama.py]
allowed_actions:
  - rewrite_rules_docs_and_maintain_code
  - manifest_then_remove_authorized_learning_file_and_redundant_logs
  - acquire_named_assets_into_workspace_with_source_and_license_records
  - prepare_frozen_data_and_campaign
  - delegate_finite_correctness_checks_to_EXECUTOR
outputs:
  - logs/runs/20261113-main80-prep
  - APE/main80
  - scripts/run_campaign.sh
  - campaign.json
acceptance:
  - all_v3_preparation_checks_pass_with_real_evidence
  - data_mapping_and_sources_resolved_before_freeze
failure_policy:
  asset_failure: preserve_partial_and_log_no_substitution
  system_failure: preserve_scene_stop_dependent_work
execution_record:
  actor: PLANNER
  started: 2026-11-13
  delivery: pending
review_record:
  conclusion: ACCEPTED_20261114
  evidence: logs/runs/20261113-main80-prep/review.json
```

## 完整执行包

### PREP-ASSETS / EXECUTOR

```yaml
protocol: ape-plan-exec/v1
task_id: PREP-ASSETS
plan_version: v3.0
owner: EXECUTOR
status: ACCEPTED
depends_on: []
authorization_ref: U_MAIN80
inputs: [logs/runs/20261113-main80-prep/model_files.remote.txt]
allowed_actions:
  - HTTPS_download_13_required_model_files_from_named_ModelScope_master_mirror
  - HTTPS_download_LongBench_v1_Qasper_HotpotQA_MuSiQue_and_original_mapping_data
  - HTTPS_download_LITM20_gold0_9_19_and_Doc2Dial_validation_documents
  - record_sources_branches_formats_licenses_and_keep_partial_on_failure
  - CPU_only_structure_and_offline_config_tokenizer_validation
outputs:
  - models/Llama-3.1-8B-Instruct
  - datasets/raw-main80
  - scripts/main80-assets
  - logs/runs/20261113-main80-assets
acceptance: [source_metadata_and_license, model_13_files_and_index_structure, actual_download_evidence]
failure_policy: {network_or_asset_failure: preserve_partial_record_no_substitution}
execution_record: {claimed_by: null, delivery: pending}
review_record: {planner: PLANNER, conclusion: pending}
```

该任务输出位于实际 Git 仓库之外的独占目录；PLANNER 不编辑这些目录。EXECUTOR MUST_NOT 执行 GPU、安装 `U_MAIN80_SCORING` 之外的依赖、修改任务范围或编写最终验收。领取及交付写入其独立日志，不编辑文档其他区域。

```yaml
protocol: ape-plan-exec/v1
task_id: MAIN80
plan_version: v3.0
owner: EXECUTOR
status: SUBMITTED
depends_on: [PREP_ACCEPTED]
authorization_ref: U_MAIN80
inputs: [campaign.json, frozen_data, frozen_code, prep_review]
allowed_actions:
  - single_physical_gpu_0_serial_workers_BF16_seed42_greedy
  - automatically_advance_frozen_internal_dependencies
  - authorized_nested_sample_reduction_with_all_144_configs
  - local_detached_monitor_and_browser
outputs: [raw_attempts, rules, diagnostics, sampling_changes, summary, incomplete_manifest]
acceptance: [four_experiment_families, fair_paired_inputs, complete_truthful_evidence, wall_clock_at_most_80_hours]
failure_policy:
  sample_data_error: record_and_continue_no_replacement
  systematic_cuda_interface_model_error: stop_dependents_preserve_scene
  minimum_scale_over_budget: BLOCKED_budget
  deadline: stop_only_owned_workers_save_partial
execution_record:
  claimed_by: EXECUTOR_/root/method_cpu
  status: RUNNING
  claimed_at_record: logs/runs/20261114-main80-launch/claim.json
  campaign: main80-20261114T041503
  command: bash scripts/run_campaign.sh start --config campaign.json
  started_at: '2026-11-14T04:15:05.134575+00:00'
  started_at_beijing: '2026-11-14T12:15:05.134575+08:00'
  deadline_utc: '2026-11-17T12:15:05.134575+00:00'
  deadline_beijing: '2026-11-17T20:15:05.134575+08:00'
  controller_pid: 4187806
  monitor_url: http://127.0.0.1:42501/campaign/main80-20261114T041503/
  browser_rendered_receipt: output/runs/main80-20261114T041503/browser-seen.jsonl
  start_exit_code: 0
  observation:
    elapsed_seconds: 179.00397308985703
    exit_code: 0
    last_observed_at_utc: '2026-11-14T04:18:35.399824+00:00'
    last_observed_stage: baseline
    successful_records: 159
    planned_records: 18195
    failed_attempts: 0
    ETA: unknown_insufficient_formal_method_length_cache_observations
    background_stopped: false
    report: logs/runs/20261114-main80-launch/LAUNCH_REPORT.md
  delivery: logs/runs/20261114-main80-launch
review_record:
  planner: PLANNER
  conclusion: COMPUTATION_SUBMITTED_QUALITY_AND_RESEARCH_REVIEW_PENDING
  evidence: logs/runs/20261113-main80-prep/review.json
  execution_delivery: output/runs/main80-20261114T041503/delivery.json
  status_checked_date_beijing: '2026-11-16'
```

`EXECUTOR` MUST 在领取处记录实际命令、模型/数据/参数/版本、起止和退出及交付链接。MUST_NOT 把启动汇报当 SUBMITTED；后台正常时保持 RUNNING。内部阶段由冻结依赖自动推进，最终由 PLANNER 审核。

## 已审定基础

| 事实 | 证据 | 范围 |
|---|---|---|
| 既有环境、TF4.44.2 overlay | [环境原始结果](../logs/runs/20260926-r1env-a1/ENV_RESULT.md.txt)及有效锁 | 历史已审定 |
| 旧Llama-3官方core两轮R2 | [第一轮](../logs/runs/20260926-r2-a3/model.stdout)、[第二轮](../logs/runs/20260926-r2-a4/model.stdout) | 有限功能，不代表新主线 |
| 前次整理 | [变更](../logs/runs/20261113-workspace-cleanup/CHANGES.md) | 保留原始证据 |

## PLANNER 审核区

### PREP-METHODS-CPU / ACCEPTED

`EXECUTOR` MUST 仅执行有限CPU方法测试（随机2层Llama，GQA及Llama3 RoPE，禁止GPU），命令 `bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B -m main80.selftest --suite methods_cpu --output logs/runs/20261113-main80-methods-cpu/tests.txt`。授权 `U_MAIN80`；输出独占 `logs/runs/20261113-main80-methods-cpu/`，记录领取/命令/起止/退出并 SUBMITTED。不得修改源码或自行验收。测试覆盖PCW单窗等价、不等长与完整block mask参照、唯一BOS/最长窗位置、缓存不可变及新请求重新生成。失败保留现场交PLANNER修正。

PLANNER审核：该CPU任务已ACCEPTED（2026-11-14）；3项测试退出0，原始命令/输出与代码快照齐全。后续只读代码审核已SUBMITTED，问题正在准备阶段补正，不属于新的GPU实验。

### PREP-ASSETS / ACCEPTED

PLANNER审核（2026-11-14）：13项文件齐全，291个索引tensor与四权重头部结构一致；来源清单在传输前后逐项一致，离线config/tokenizer读取通过；四类数据及原始映射、来源/许可清单齐全。评分overlay指定5版本导入和官方48.53例子通过，ID对齐由统一评分器补充测试通过。验收限于获取和结构，不代表整模型运行。证据见 `logs/runs/20261113-main80-assets/DELIVERY.md` 及相邻JSON；旧失败/partial保留。

### PREP-METHODS-FULL / ACCEPTED

```yaml
protocol: ape-plan-exec/v1
task_id: PREP-METHODS-FULL
plan_version: v3.0
owner: EXECUTOR
status: ACCEPTED
depends_on: [PREP_ASSETS_ACCEPTED]
authorization_ref: U_MAIN80
inputs: [models/Llama-3.1-8B-Instruct, APE/main80/prep_gpu.py]
allowed_actions:
  - four_independent_serial_workers_on_physical_GPU_0
  - exactly_four_short_synthetic_generations_per_method
  - additional_query_logit_and_KV_only_correctness_forwards
outputs: [logs/runs/20261114-main80-methods-full]
acceptance: [full_model_offline_BF16, common_content, PCW_single_window, official_APE_stages_T_S, immutable_cold_hot_resume_KV, token_attention_output_agreement]
failure_policy: {any_worker_error: preserve_stop_remaining_workers_no_automatic_retry}
execution_record: {claimed_by: null, delivery: pending}
review_record: {planner: PLANNER, conclusion: ACCEPTED_functional_scope_only, evidence: logs/runs/20261114-main80-methods-full/delivery.md}
```

准确命令：`bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B -u scripts/run_preparation_gpu.py`。方法sequential/pcw/ape_default/ape_fixed，每次输入≤256 tokens、输出≤8 tokens，共16次短生成，外加无生成的单窗/T/S核对。每worker≤240秒；不做设备/资源审查、不计算摘要、不安装。不得用此任务运行benchmark或测正式研究速度。执行者只写独占交付区，源码补正交PLANNER。

PLANNER完整准备审核（2026-11-14）：ACCEPTED。数据600/50/40/510、432个有效开发长度输入与18个N/A、LITM500三版本、Doc2Dial500完整历史、全部缩样144配置覆盖逐项通过；3项CPU方法、16次整模型短生成、12项协议/恢复/单实例回归及实际独立浏览器/5秒刷新/后台持久/独立截止测试通过。源码、冻结数据和模型小文件40项已保存接受副本，首发及恢复必须逐字比较，依赖版本和13项模型清单另核对；没有摘要计算或资源审查。签核见 `logs/runs/20261113-main80-prep/review.json`。

MAIN80已于北京时间2026-11-14 12:15:05启动，并于同日23:43完成全部8阶段、生成交付。campaign `main80-20261114T041503` 当前为SUBMITTED，待PLANNER正式验收。实际约11小时28分，18,116条记录成功、失败0、未完成0，未触发预算缩样。80小时为上限，不要求跑满；没有自动扩展新模型或额外实验。

计算提交证据见 [delivery.json](../output/runs/main80-20261114T041503/delivery.json)、[state.json](../output/runs/main80-20261114T041503/state.json)、[summary.json](../output/runs/main80-20261114T041503/summary.json)及 [incomplete.json](../output/runs/main80-20261114T041503/incomplete.json)。启动/有限观察见 [launch目录](../logs/runs/20261114-main80-launch)。EXECUTOR领取区中的RUNNING为启动时记录，当前任务状态以本次PLANNER核对和机器交付为准。

2026-11-16状态核对：已检查原始记录数量、唯一key、退出与阶段事件，未发现运行失败；结果尚未正式验收。下一步核对评分/生成质量、配对统计、数据与规则适用域及效率解释，再形成结题汇总。特别记录：正式LongBench实际N为1—3，LITM全部N=1，Doc2Dial为N=1/2，因此正式外测未覆盖规则的N=8/16列。不得将18,116次程序成功等同于动态方法取得收益，或据此改动冻结规则/测试集。
