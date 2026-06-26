# Inner-Only Method Evolution

## 1. 目标

当前版本不再先做外层框架进化，而是直接把 [tsp_response.py](D:\\A_Shaw\\Academic\\Files\\program\\multi_op\\tsp_response.py) 中的 `response` 解析成一个 `AlgorithmFrame`，然后只在内层对该框架中的各个 method 做 EoH 进化。

核心思路：

- 固定整个 `Algorithm` 框架
- `run` 不参与进化，只作为主流程上下文
- 其余 method 按预算调度逐步进化
- 每次 method 进化完成后，立即替换回整个类并重新评估整框架分数

## 2. 入口与运行

当前入口在 [main.py](D:\A_Shaw\Academic\Files\program\multi_op\main.py)：

- `mode = "inner_only"`：只执行 inner-only 实验
- `mode = "outer"`：保留原来的外层流程

默认 inner-only 配置在 `run_inner_only(...)` 中：

- `max_sample_nums=1000`
- `pop_size=10`
- `budget_mode="average"`
- `benefit_mode="absolute_gain"`
- `per_call_budget_cap=50`
- `use_adj_prev_operator=True`
- `use_adj_next_operator=True`

运行步骤：

1. 在 [tsp_response.py](D:\A_Shaw\Academic\Files\program\multi_op\tsp_response.py) 中准备好当前要改进的 `response`
2. 在 [main.py](D:\A_Shaw\Academic\Files\program\multi_op\main.py) 中确认 `mode = "inner_only"`
3. 根据实验目标修改 `run_inner_only(...)` 中的预算和策略参数
4. 运行 `python multi_op/main.py`
5. 到 `logs_inner_only/...` 查看 method 调度日志、各 method 的 EoH 日志和收敛图

## 3. 预算与调度策略

### 基线实验

- `budget_mode="average"`
- 总预算按 method 数平均分配
- 按代码中的 method 顺序依次进化
- 若预算不能整除，多出的预算按顺序补给前几个 method

### 自适应调度实验

- `budget_mode="adaptive"`
- 每次 method 进化最多用 `per_call_budget_cap` 个 sample
- 每轮结束后根据 method 的效益值选择下一次进化哪个 method
- 预算耗尽或所有 method 都没有正收益时停止

## 4. 效益值策略

`benefit_mode` 当前支持三种：

- `absolute_gain`
  - 本轮 method 进化后，整框架分数的绝对提升量

- `gain_per_sample`
  - 本轮分数提升量 / 该 method 已消耗预算

- `hybrid`
  - `w1 * absolute_gain + w2 * gain_per_sample + w3 * recent_success_rate`
  - 权重由 `hybrid_weights` 控制
  - `recent_success_rate` 由最近 `recent_window` 次该 method 调用中正收益比例计算

统一口径：

- `gain = new_score - old_score`
- `score` 越大越好
- `None / inf / -inf` 都视为无效分数，不参与正收益计算

## 5. Prompt 与新算子

当前 EoH prompt 除了原有信息，还新增了：

- 当前 `Algorithm` 的 `run` 主流程代码
- 当前 method 之外其他 method 的 introduction

新增两个邻接算子：

- `ap`
  - 给前一个 method 的代码和 thought
  - 给当前 method 的父代代码和 thought
  - 适用于当前 method 不是第一个 method

- `an`
  - 给当前 method 的父代代码和 thought
  - 给后一个 method 的代码和 thought
  - 适用于当前 method 不是最后一个 method

这两个算子分别由：

- `use_adj_prev_operator`
- `use_adj_next_operator`

控制开关。

## 6. 日志与结果

inner-only 运行结果保存在 `logs_inner_only/...` 下，主要包括：

- `run_log.txt`
  - 每次 method 调度的摘要

- `method_schedule.json`
  - 每轮调度的详细记录
  - 包含 method 名称、预算、收益、分数变化、剩余预算、使用的算子等

- `method_best.json`
  - 当前历史最优 method 调度记录和最终框架摘要

- `convergence_inner.png`
  - 以总消耗 sample 为横轴、整框架历史最优 score 为纵轴的收敛图

- `logs_inner_only/.../<frame_id>/<method>/...`
  - 每个 method 自己的一套 EoH 日志

## 7. 推荐实验组合

### 实验 1：平均预算基线

- `budget_mode="average"`
- `benefit_mode="absolute_gain"`

### 实验 2：固定上限 + 绝对提分调度

- `budget_mode="adaptive"`
- `benefit_mode="absolute_gain"`
- `per_call_budget_cap=50`

### 实验 3：固定上限 + 单位预算提分调度

- `budget_mode="adaptive"`
- `benefit_mode="gain_per_sample"`
- `per_call_budget_cap=50`

### 实验 4：固定上限 + 混合效益调度

- `budget_mode="adaptive"`
- `benefit_mode="hybrid"`
- `per_call_budget_cap=50`
- 视情况修改 `hybrid_weights`

## 8. 实现说明

当前实现中的关键点：

- method 顺序以 `Algorithm` 类中的源码定义顺序为准
- `run` 不参与演化
- 每次 method 演化结束后，都会把最优个体替换回整个 `AlgorithmFrame`
- 替换后重新评估的是整个框架，不是局部 method
- 邻接算子中的 thought 当前默认使用 `method_introduction`
