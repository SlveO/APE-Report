# 第七阶段审核记录

本阶段读取4500条LITM生成，连接500题×3位置×3方法。全部成功、attempt1、cold miss；参考、实际L/N、prefix/request和query位置一致。源全文分数保留，主表另列官方首行分数。

| 审核项 | 实际结果 | 证据 |
|---|---|---|
| 输入与位置 | 1500前置title/text、gold、模板检查PASS;500题删除gold后干扰顺序一致 | preaudit_input_checks.csv;membership.csv |
| 来源字段 | 生成source为null;gold公开ID为None;通过question和position连接原作者输入 | membership.csv;sources/input_sources.json |
| 评分入口 | 501条修正147题;9fixture/18函数调用通过;12个原全文汇总均值匹配 | official_entrypoint_fixtures.csv;original_summary_crosscheck.csv |
| 统计单位 | 500问题;三位置及三方法共同保留;总体W/T/L按题均值 | bootstrap_questions.csv;main_quality_paired.csv |
| 实际配置 | 原资料20篇;实际N1;L2532–3573;动态1/1;固定.7/1 | input_distribution.csv;per_request_records.csv |

| 现象 | 影响 | 证据 | 处理及补充 |
|---|---|---|---|
| 原保存评分采用全文,官方入口采用首行 | 位置EM、总体EM及所有配对质量结论 | 501/4500标签由1变0,涉及147问题;末位动态−固定由全文-1.0变为首行+4.4pp | 复用前置501条修正;9原答案实例复核;全文分数另表保留；未来生成与评分入口统一使用作者首行约定 |
| 全部输入为单编码块且低于开发长度下界 | 动态规则迁移和位置效应解释 | 原20篇;L2532–3573,N1;全部映4K/4,动态1/1 | 明确资料位置在一个编码块内变化;固定.7/1与动态1/1实际参数另列；同20篇完整资料控制实际N=1/4/8/16,分开研究资料位置与编码块位置 |
| 当前最低点在末位 | 位置敏感性的结论 | 三方法位置19均低于位置9;末−首为-9.2/-16.0/-8.2pp | 同时报告中间−两端和末−首;保留三点原曲线；扩展预先冻结的位置点,检查位置曲线形状 |
| 高比例达到生成上限,多数首行已完成 | 长输出的质量与总响应 | 1465/4500达到128;1391已换行;69无EOS且首行未换行;393次评分修正在cap组 | 保留全部原始生成,将EOS、上限、首行状态和质量一并统计；统一比较EOS/128与首个换行停止;对69类未完成首行条件成对提高上限 |
| 缺逐token时标 | 首行停止规则的实际加速量 | 仅保存TTFT和总响应;首行换行时刻缺失 | 按已有完整生成计时报告;输出状态单列；补首个换行时刻或逐token时标,实测停止规则的时延 |
| 单块APE同输出路径耗时较高 | 单块效率与输出长度的归因 | 顺序/动态1044对token相同,动态平均总响应仍多1.406秒 | 单列同输出组;记录APE两分支query及完整冷请求实现；同输入同输出随机交错重复测时,拆分prefill/query与后续decode |

最小证据保留在official_entrypoint_fixtures.csv、scoring_changed_records.csv、selected_case_positions.csv和generation_cap_records.csv；每条含sample_id/method/position、源行与原答案。原作者入口来源见sources/official_scoring_notes.md。关键数字与公式见provenance.csv，原始证据只读，研究状态SUBMITTED，下一阶段未执行。
