# 本地监控设计锁

设计对象：APE实验操作者使用的本地只读网页；目标是判断当前进度、失败和预算，所有数据来自campaign实际记录。直接实现，不引入前端依赖或外部字体。

参考：refero-design 无可用实时工具，使用本地 [typography.md](/home/shenxiaoyang/.agents/skills/refero-design/references/typography.md) 的工作工具原则（紧凑、中性、一套字体、表格数值等宽）作为主要方向；[craft-details.md](/home/shenxiaoyang/.agents/skills/refero-design/references/craft-details.md) 的语义HTML、可见焦点、异步状态播报作为交互补充；用户方案确定展示字段和5秒刷新。

| 决定 | 来源与角色 | 实现约束 |
|---|---|---|
| 单列总览+分层速度表 | 用户监控字段、typography工作工具扫描原则 | 当前任务先于历史结果 |
| 白色背景、深灰文字、细灰分割线 | typography的可读性及3级文字原则 | 不使用装饰渐变/图片 |
| 系统sans字体、14px正文、等宽数字 | typography紧凑UI及数值角色 | 不联网加载字体 |
| 状态同时有文字和颜色 | craft语义及异步更新规则 | 红色仅失败、绿色仅运行成功，未知ETA明确显示 |
| 固定80小时进度尺 | 用户预算、原生progress | 完成数与预算进度分别显示 |
| 表格与记录链接 | 用户结果入口、craft原生链接 | 移动端横向滚动，键盘可达 |

验收：实际浏览器确认本次campaign标识与加载请求，检查800px与窄屏表格、连接失败状态。数据缺失时不得填造速度、ETA或完成数。
