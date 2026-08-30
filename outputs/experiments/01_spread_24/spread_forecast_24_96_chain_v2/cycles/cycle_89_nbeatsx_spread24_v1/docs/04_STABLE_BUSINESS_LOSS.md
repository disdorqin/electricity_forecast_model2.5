# Stable Business Loss for H34 NBEATSx

status: active  
date: 2026-08-29  
responsibility: define a numerically stable loss that preserves point-forecast quality while directly improving positive/negative direction performance  
validation basis: NBEATSx paper MAE baseline, Cycle 88 direction-aware loss evidence, robust regression / multi-task optimization principles

## 1. 设计目标

业务模型同时关心：

1. 价差数值误差；
2. 总体方向准确率；
3. 正价差预测能力；
4. 负价差预测能力；
5. 跨月稳定性；
6. 训练不能因尖峰、类别不平衡或方向项导致梯度爆炸。

但是不能直接把离散 `accuracy` 放进反向传播，因为：

```text
I(sign(y_hat) == sign(y))
```

不可导。

因此训练使用 differentiable surrogate；真实 Direction / Positive Recall / Negative Recall 只用于 validation 和最终评价。

---

## 2. 必须先保留 paper MAE baseline

论文复现 profile 继续严格使用：

```text
L_paper = MAE
```

业务实验必须首先有：

```text
B0 = H34 + CORE5 + paper MAE
```

再运行：

```text
B1 = H34 + CORE5 + StableDirectionalPseudoHuber
```

这样如果 B1 失败，可以判断问题来自 loss 而不是模型/数据。

---

## 3. 为什么业务主损失不直接用 MSE

电价价差存在尖峰。MSE 梯度：

```text
dL/dy_hat = 2 * error
```

误差越大梯度越大，少数极端样本可能主导更新。

MAE 的梯度有界，但在 0 处不可微且所有大误差梯度幅度基本相同。

业务候选采用 pseudo-Huber：

```text
PH(r; delta) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
```

其梯度：

```text
dPH/dr = r / sqrt(1 + (r/delta)^2)
```

当误差很大时梯度趋近 `±delta`，天然有界；在 0 附近平滑、二阶可导。

因此比 MSE 更抗尖峰，同时比纯 MAE 更利于平滑优化。

---

## 4. Target 归一化：只缩放，不平移

为了让不同 target day 的 loss 尺度稳定，同时保持 0 为业务方向分界点：

```text
scale_y = max(median(abs(y_train)), 10.0)
z_true = y_true / scale_y
z_pred = y_pred / scale_y
```

或者模型直接预测 normalized `z_pred`，最后反缩放。

禁止对 target 做：

```text
(y - median) / IQR
```

然后直接使用 0 判断方向，因为 centering 会移动正负分界点。

v1：

```text
pseudo_huber_delta = 1.0
```

---

## 5. H34 的 bridge 与 business target 必须区别对待

34 个输出中：

```text
bridge = first 10
scored = last 24
```

Bridge 的价值是辅助学习连续未来状态，不是业务 headline。

定义：

```text
L_mag_scored = mean(PH(z_pred-z_true) on last24)
L_mag_bridge = mean(PH(z_pred-z_true) on first10)
```

组合：

```text
L_magnitude = L_mag_scored + alpha_bridge * L_mag_bridge
alpha_bridge = 0.25
```

因此单个 scored hour 的梯度权重大于 bridge hour。

### 后续 bridge ablation

```text
alpha_bridge = 0.0 / 0.25 / 1.0
```

v1 冻结 `0.25`，不与其他结构同时搜索。

---

## 6. 方向损失必须优化“正负两类平衡”，而不只是 raw accuracy

Cycle 88 已经证明 raw accuracy 容易被类别比例欺骗。

因此主要方向 surrogate 不是简单地对全部样本平均 BCE，而是分别计算正负类。

对 scored 24 点，设温度：

```text
tau = 0.35
```

### Positive sign loss

真实 `z_true > 0`：

```text
L_pos = mean(softplus(-z_pred / tau))
```

预测越负，惩罚越高。

### Negative sign loss

真实 `z_true < 0`：

```text
L_neg = mean(softplus(z_pred / tau))
```

预测越正，惩罚越高。

### Balanced direction surrogate

```text
L_bal_sign = 0.5 * L_pos + 0.5 * L_neg
```

这和业务上的：

```text
BalancedAccuracy = 0.5*(PositiveRecall + NegativeRecall)
```

目标结构直接对应。

若一个 batch 暂时缺少某一类，则该类 loss 使用 0 并通过 batch-level valid flag 记录；DataLoader 优先采用 sign-stratified sampler 降低这种情况发生概率。

---

## 7. 总体 raw direction 也保留一个小权重

只优化 balanced direction 有可能牺牲总体类别分布下的 raw accuracy。

定义 scored points 的普通 sign surrogate：

```text
s = sign(z_true)
L_raw_sign = mean(softplus(-s * z_pred / tau))
```

它按自然样本比例计算。

最终同时拥有：

- `L_bal_sign`：保护 positive/negative recall；
- `L_raw_sign`：照顾实际总体 direction accuracy。

---

## 8. 近零价差不应该获得和大价差同样的方向梯度

当真实 spread 非常接近 0 时，正负号可能被微小价格噪声翻转。

如果对 `+0.1` 和 `+100` 使用同样强的方向惩罚，模型会浪费容量追逐边界噪声。

定义可靠性权重：

```text
m = abs(z_true)
w_dir = clamp(m / 0.25, 0, 1)
```

因此：

```text
|z| >= 0.25  -> full directional weight
|z| near 0   -> reduced directional weight
```

`L_pos/L_neg/L_raw_sign` 均乘 `w_dir` 后归一化。

最终 evaluation 不使用这个权重，真实 direction metric 仍按项目现行公式报告。

---

## 9. 最终业务 loss

### 9.1 magnitude phase

训练前 20% optimization steps：

```text
lambda_mag = 1.00
lambda_bal = 0.00
lambda_raw = 0.00
```

即先学会“数值大致在哪里”。

### 9.2 direction ramp

在 20%~40% steps 内线性 ramp：

```text
lambda_mag: 1.00 -> 0.70
lambda_bal: 0.00 -> 0.20
lambda_raw: 0.00 -> 0.10
```

### 9.3 mature phase

40% steps 以后：

```text
L_total =
  0.70 * L_magnitude
+ 0.20 * L_bal_sign
+ 0.10 * L_raw_sign
```

注意：

```text
L_bal_sign = 0.5*positive surrogate + 0.5*negative surrogate
```

因此 loss 已经显式兼顾正负方向，不需要再叠加三个高度重复的 direction terms。

---

## 10. 为什么不直接照搬 Cycle88 λ=0.4

Cycle 88 已出现：

```text
某方向 loss 在一个月显著提高，但另一个月反向下降
```

说明方向项过强或过早参与可能把模型推向短期类别 regime。

新设计通过四层保护降低这种风险：

1. pseudo-Huber bounded gradient；
2. target scale normalization；
3. direction warm-up/ramp；
4. balanced + raw 两个 surrogate 分工。

因此不是简单把 direction 权重从 0.4 改成另一个拍脑袋数字。

---

## 11. 梯度稳定策略

业务 v1 强制：

```text
optimizer = Adam
learning_rate = 5e-4
global_grad_clip_norm = 1.0
NaN/Inf gradient check = true
```

### 11.1 梯度检查

每个 optimization step：

```text
if loss is non-finite -> abort
backward
if any gradient non-finite -> abort
clip_grad_norm_(1.0)
optimizer.step()
```

记录：

```text
pre_clip_grad_norm
post_clip_grad_norm
clip_fraction
```

若大量 step 都被 clip：

```text
clip_fraction > 0.25
```

必须标记 `GRADIENT_INSTABILITY_WARNING`，不能仅靠 clipping 掩盖问题。

### 11.2 Mixed precision

GPU 可使用 BF16 autocast；若硬件不支持则 FP16 + GradScaler。

Loss 聚合、metric 和 scale 计算保持 float32。

Paper parity profile 可另行固定 float32，以减少版本/精度差异。

---

## 12. Learning rate schedule

论文使用 Adam + lr halving，多次衰减。

Business v1 保留同类 schedule，而不是同时改成完全不同优化器：

```text
initial_lr = 5e-4
max_steps = 1200
lr_decay_gamma = 0.5
scheduled_decays = 3
nominal decay steps = 300 / 600 / 900
```

如果 early stopping 在 decay 前结束，不强制继续训练。

不在 v1 使用 OneCycle、ReduceLROnPlateau、CosineWarmRestarts 等额外变量。

---

## 13. Early stopping 不能直接监控离散 Accuracy

Direction Accuracy 会跳变，不适合直接作为反向优化和唯一 early-stop metric。

### B0 MAE baseline

```text
valid_loss = scored24 normalized MAE
```

### B1 directional model

定义 validation business score：

```text
V =
  0.55 * normalized_MAE
+ 0.15 * (1 - direction_accuracy)
+ 0.15 * (1 - positive_recall)
+ 0.15 * (1 - negative_recall)
```

只在 scored 24 点上计算。

为什么不是训练 loss 本身：

- validation 应反映真实业务指标；
- 可以发现 surrogate 虽下降但真实 sign 并未改善；
- positive/negative 单独出现，防止多数类掩盖。

`normalized_MAE = MAE / scale_y_validation_reference`，scale 只来源于训练/validation 合法历史，不能由 target day label 决定。

### Early stop

```text
val_check_steps = 25
patience_checks = 8
min_delta = 1e-4 relative score
```

同时保存：

```text
best_business_score
best_MAE
best_direction
best_pos_recall
best_neg_recall
```

---

## 14. Sign-stratified batch sampler

如果自然 batch 经常出现正/负比例极端，balanced sign loss 方差会变大。

v1 可实现一个**只基于训练 label**的 sampler：

```text
50% sample origins from days containing enough positive scored slots
50% from remaining pool
```

但不要直接按单个小时拆散 day sample；一个样本仍然必须保持完整 34-step target vector。

更保守的 v1 默认：普通 shuffled daily samples。

只有审计发现 `>10%` batch 缺失某一 sign class 时才启用 sign-stratified sampler，且必须作为显式配置和 ablation 记录。

---

## 15. Horizon weighting

第一版 scored 24 点全部等权：

```text
w_h = 1 for h in D h1-h24
```

不再直接套旧：

```text
1-8 / 9-16 / 17-24
```

如果统一 H34 模型跑完后发现 error profile 存在连续的特定 horizon 区间失败，例如：

```text
forecast offsets 18..26 consistently collapse
```

后续才按**forecast offset / empirical error regime**设计 specialist 或 horizon weights。

必须先看：

```text
metric_by_forecast_offset.csv
```

再决定，不先验指定三个时段。

---

## 16. Business loss 实验顺序

只做可归因实验：

```text
L0: paper MAE
L1: pseudo-Huber only
L2: pseudo-Huber + balanced sign
L3: pseudo-Huber + balanced sign + raw sign (final candidate)
```

每次只增加一部分。

### 晋级最低要求

相对 L0：

- Jun raw direction 不下降；
- Jul raw direction 不下降；
- macro balanced 至少 +1pp 才视为 meaningful；
- positive/negative recall 任何一侧不得崩到 <40%；
- MAE 恶化不得 >3%；
- 不能靠某一个月的大提升抵消另一个月明显倒退。

如果 L2 已优于 L3，则不因为“L3更完整”强行选 L3。

---

## 17. 必须输出的训练诊断

每个 target day：

```text
loss_components.csv
  step
  magnitude
  bridge_magnitude
  balanced_sign
  raw_sign
  total

optimizer_trace.csv
  step
  lr
  grad_norm_pre_clip
  grad_norm_post_clip
  clipped

validation_trace.csv
  step
  normalized_mae
  direction_accuracy
  positive_recall
  negative_recall
  balanced_accuracy
  business_score
```

汇总：

```text
loss_stability_summary.json
```

包含：

- non_finite_count；
- gradient_clip_fraction；
- best_step 分布；
- max_steps_reached_fraction；
- seed variance。

---

## 18. 核心结论

第一版不是追求一个复杂的“创新 loss”，而是建立一条稳定可证伪路线：

```text
MAE paper baseline
   -> pseudo-Huber bounded-gradient magnitude
   -> balanced positive/negative sign surrogate
   -> small raw-direction term
   -> warm-up/ramp
   -> gradient clipping + chronological early stopping
```

业务主候选公式：

```text
L = 0.70 * [PH(scored) + 0.25*PH(bridge)]
  + 0.20 * 0.5*(positive_sign_loss + negative_sign_loss)
  + 0.10 * raw_sign_loss
```

方向部分只在 scored D-day 24 点上计算，且近零 spread 自动降权。
