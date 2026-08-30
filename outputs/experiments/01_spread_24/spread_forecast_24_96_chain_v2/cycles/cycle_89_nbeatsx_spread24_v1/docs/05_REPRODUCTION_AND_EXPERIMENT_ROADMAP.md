# NBEATSx 复现、业务适配与实验路线图

status: active  
date: 2026-08-29  
responsibility: convert the paper reproduction and H34 business design into an executable, claim-driven experiment roadmap

## 1. 两条严格隔离的路线

### Track P — Paper Reproduction

目标：证明我们的 NBEATSx core 真正复现了论文结构和训练行为。

禁止混入：DA-RT spread、H34、D-1 14:00、business directional loss、CORE5/B208、rolling 9m business protocol。

### Track B — Business Adaptation

目标：把已验证 paper core 迁移到：

```text
D-1 14:00 -> D 24:00
DA-RT spread
H34 direct MIMO
rolling 9m
strict D-2 labels
```

任何 Business 结果都不能称为“论文复现成绩”。

---

## 2. Paper reproduction 的证据层级

### P0 — Source lock

固定 DOI/arXiv、official repo commit/hash、environment、reference configs、public EPF dataset snapshot。

### P1 — Structural parity

单元测试：identity/trend/seasonality basis、TCN/WaveNet exogenous encoder、double residual、forecast aggregation、decomposition sum、initialization、shape/mask behavior。

### P2 — Numerical parity

固定 tiny tensor 和 matching weights，对官方实现与本实现比较 backcast、forecast、decomposition、MAE、gradients。

### P3 — Training-behavior parity

固定小数据集、相同 seed、optimizer/scheduler，比较 loss 下降、best step、lr decay、early stopping、checkpoint reload。

### P4 — Public EPF reproduction

优先 Nord Pool reference path：`L=168, H=24, MAE, Adam`。先冻结 paper-like config，再 limited hyperparameter reproduction。

### P5 — Full search reproduction（高算力可选）

官方 README 的 Nord Pool 入口：

```text
space=nbeats_x
n_val_weeks=52
hyperopt_iters=1500
data_augmentation=0
random_validation=0
```

只有完成这一层才称 full search reproduction。

---

## 3. Paper-derived business anchor

Business v1 不重新做 1500 次搜索，先冻结一个位于论文搜索空间内部的中型 anchor：

```text
stack order: Identity -> Exogenous TCN
blocks: [1,1]
FC layers: 2
hidden: 256
TCN channels: 8
kernel: 3
activation: Softplus
batch norm: false
dropout: 0.05
```

这不是宣称论文最优参数，而是 paper-valid anchor。

只有 H34 baseline 有正信号后，再最小比较：hidden128、TCN->Identity、Identity->WaveNet。

---

## 4. Business v1 实验顺序

### B0 — Contract smoke

验证 H34 mapping、CORE5 future availability、D-2 cutoff、train-only scaler、one-batch overfit、checkpoint、first10/last24 masks 和 metric scope。

### B1 — Paper-loss H34 baseline

```text
L=168
H=34
CORE5
rolling 9m
validation 28d
batch 32
Identity->TCN
hidden 256
MAE
Adam
```

Pilot：2026-06 + 2026-07。

### B2 — Robust magnitude

唯一变化：`MAE -> pseudo-Huber`。

### B3 — Balanced direction

在 B2 上加入 `0.20 balanced positive/negative sign surrogate`。

### B4 — Raw direction supplement

在 B3 上加入 `0.10 raw sign surrogate`，形成候选：

```text
0.70 magnitude + 0.20 balanced sign + 0.10 raw sign
```

如果 B3 比 B4 更好，则 B3 为最终 loss。

---

## 5. 输入长度研究

loss candidate 冻结后：

```text
I0: L168
I1: L336
```

若 I1 在 Jun+Jul 和扩展月均一致改善，再考虑 `L672`。

同步报告 parameter count、train time、best step、overfit gap、raw/balanced、MAE、horizon error profile。

---

## 6. Training window 研究

输入长度与 calibration history 独立：

```text
input length = 单个模型样本看到多少 recent hours
training window = 参数从多少 historical daily-origin samples 学习
```

第一版 `L168 + 9m`，后续单独比较 `6m/9m/12m`，每个窗口仍保留最后 28 天 chronological validation。

---

## 7. 更新频率研究

Business reference：`daily cold retrain`。

通过后再研究：

```text
U1: every 3 days cold retrain
U2: weekly cold retrain
U3: weekly cold reset + daily replay warm update
```

评价精度、GPU minutes/day、regime staleness。warm-start 必须重做 state/leakage audit。

---

## 8. Feature roadmap

### F0 — CORE5 raw trajectory

主线：direct load、interconnection、wind、solar、bidding space、calendar。

### F1 — CORE5 + renewable total

验证新能源总加在 wind/solar 之外是否仍提供增量。

### F2 — CORE5 + causal uncertainty/error states

只从 Cycle88 F5/F6 选择严格 causal 的历史 forecast-error / uncertainty summary。

### F3 — B208_COMPAT

将特征筛选 AI 最终 LightGBM shortlist 作为显式 compatibility ablation，回答 engineered tabular features 在 raw temporal encoder 之外是否仍有增量。

### F4 — no-exogenous NBEATS

必须保留 raw spread history only baseline，用于测量 exogenous branch 的净贡献。

---

## 9. H34 horizon profile 决定后续“分时段”研究

v1 不分旧的 1-8/9-16/17-24 模型。

每次 Business run 必须输出 `metric_by_forecast_offset.csv`，offset 1..34 分别统计 MAE、direction、positive recall、negative recall、bias、sample count，并标记 bridge 1..10 与 D-day 11..34。

只有 error profile 证明存在稳定连续失败区间，才研究 horizon-weighted loss、specialist head、block-specific residual 或 mixture-of-experts。时段边界由 empirical forecast-offset behavior 决定。

---

## 10. 业务比较基线

至少比较：

```text
C88-LGBM-B208
NBEATS-history-only
NBEATSx-CORE5-MAE
NBEATSx-CORE5-best-business-loss
```

其他历史模型只有在 target/sign/cutoff 完全一致时才能并列。

---

## 11. 评价与晋级

Headline 只算 D-day last24。

必报 raw Direction、Positive Recall、Negative Recall、Balanced、all-positive/all-negative baseline、MAE、RMSE、month-level delta、horizon-level delta。

### Jun+Jul pilot gate

candidate 必须：

1. 两个月 raw 不低于父 baseline；
2. 两个月 balanced 不低于父 baseline；
3. 任一 sign recall 不明显 collapse；
4. macro MAE 不恶化 >3%；
5. leakage status = STRICT/PASS。

满足才扩展 Jan/May/Aug strict blocks。

### Strong signal

建议将以下视为值得继续投入：

```text
macro raw >= LGBM + 2pp
macro balanced >= LGBM + 2pp
positive recall >= 45%
negative recall >= 45%
MAE ratio <= 1.03
```

这不是最终 70% 目标，而是判断模型族是否打开新平台。

---

## 12. Seeds

快速 pilot：seed42。

候选确认只对最多 Top2 配置跑 41/42/43，报告 mean/std。禁止 seed hunting。

---

## 13. 计算预算策略

严格停止式执行：

```text
P1/P2 parity
 -> B0 smoke
 -> B1 Jun+Jul
 -> B2/B3/B4 loss ladder
 -> only best loss enters input/window/architecture ablation
 -> cross-month
 -> seeds
 -> final lockbox
```

任何阶段失败就停止该分支。

---

## 14. 程序模块与实验块对应

```text
src/nbeatsx_spread/data/
  origin_index.py
  dataset.py
  covariates.py
  scalers.py

src/nbeatsx_spread/model/
  basis.py
  exogenous_tcn.py
  exogenous_wavenet.py
  block.py
  model.py

src/nbeatsx_spread/losses/
  mae.py
  pseudo_huber.py
  stable_directional.py

src/nbeatsx_spread/training/
  trainer.py
  scheduler.py
  early_stopping.py
  checkpoint.py

src/nbeatsx_spread/evaluation/
  metrics.py
  horizon_profile.py
  decomposition.py

src/nbeatsx_spread/audits/
  origin.py
  availability.py
  cutoff.py
  counterfactual.py
```

CLI：

```text
scripts/run_paper_repro.py
scripts/run_business_backtest.py
scripts/run_loss_ablation.py
scripts/run_input_ablation.py
scripts/run_window_ablation.py
scripts/run_recalibration_ablation.py
scripts/inspect_decomposition.py
```

---

## 15. 冻结顺序

fresh holdout 前依次冻结：paper core、H34 alignment、feature profile、input length、architecture、loss、training window、validation length、retraining frequency、optimization budget、seeds、metric definitions。全部冻结后才允许一次性 lockbox。
