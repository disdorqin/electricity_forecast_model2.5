# 24点价差预测：

> **重要说明：** 本包是从历史 Cycle88 思路独立重构出的可审查、可独立运行版本，不是对原 Cycle88 训练器的逐位复刻。当前独立运行结果为 raw **55.84%**、balanced **50.31%**、MAE **81.58**；历史 Cycle88 segmented 参考为 raw **56.35%**、balanced **53.34%**、MAE **84.09**。两者应分别理解，不应宣称本包精确复现了历史指标。

## 我们预测什么？
在 D-1 14:00 预测 D 日 24 个小时：`Spread = 日前电价 - 实时电价（DA - RT）`。模型输出连续价差，正负号给出方向。

## 预测时能知道什么？
允许：D-2及更早完整历史、D-1 1–14点已发生信息、业务上提前发布的 D 日 forecast-type 基本面和日历信息。
禁止：D-1 15–24点实际、D日实时电价、D日实际运行量、target-day actual、D-1完整价差标签，以及任何未来 actual 进入训练/特征筛选/校准。

## 运行
```bash
pip install -r requirements.txt
python run.py
pytest
```
默认使用 `data/frozen_repro`，运行后生成 `outputs/predictions.csv`、`outputs/metrics.csv`、`outputs/leakage_audit.csv`。`python run.py --smoke` 可快速检查两天；`--rebuild-from-raw` 仅显示 raw_reference 说明，不改变默认固定复现入口。

## 主审查版本
Cycle88 `numeric_v2 / full_existing_F0_F9 + lgbm_segmented`，三个时段分别为 1–8、9–16、17–24，180日滚动训练，训练标签截止 D-2。历史跨月参考：raw 56.35%、+recall 68.00%、-recall 38.67%、balanced 53.34%、MAE 84.09%。

本包不做 SHA 校验，不依赖原项目 scripts、utils、outputs 或绝对路径。完整背景见 `docs/`。
