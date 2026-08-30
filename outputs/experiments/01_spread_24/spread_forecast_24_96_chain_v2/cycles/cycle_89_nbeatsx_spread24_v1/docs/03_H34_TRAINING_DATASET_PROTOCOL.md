# H34 训练数据与滚动更新协议

status: active  
date: 2026-08-29  
responsibility: define the business dataset, input length, rolling split, retraining cadence and training sample semantics for NBEATSx H=34  
validation basis: NBEATSx IJF paper, official `cchallu/nbeatsx`, multi-output forecasting literature, Cycle 88 strict D-1 14:00 contract

## 1. 研究问题

真实业务不是“每天任意时刻预测未来 24 小时”，而是固定：

```text
forecast origin = D-1 14:00
business target = D h1-h24
```

因此 D-1 14:00 之后到 D 24:00 共存在 34 个未知小时：

```text
D-1 h15-h24 = 10 bridge hours
D   h01-h24 = 24 scored hours
```

v1 采用 direct multiple-output：

```text
168 observed hours -> 34 future hours
```

不递归回填 D-1 h15-h24，不把任何 synthetic target 当成 observed history。

---

## 2. 为什么 v1 输入长度固定 L=168

NBEATSx 原论文在电价预测任务中固定：

```text
backcast L = 168 hours
forecast H = 24 hours
```

168 小时同时覆盖：

- 7 个完整日周期；
- 一个完整周周期；
- 当前 D-1 h1-h14 的已发生价差状态；
- lag1d / lag2d / lag3d / lag7d 等 Cycle 88 已证明有价值的信息，不需要全部再手工展开成表格特征。

H 从 24 增加到 34 并不自动意味着必须把 L 同比例放大。NBEATSx/N-BEATS 是 fixed-horizon MIMO 模型，forecast length 与 backcast length 可以独立设计。

### v1 冻结

```text
L = 168
H = 34
```

### 后续受控 ablation

只在 v1 模型与 loss 稳定后比较：

```text
A0: L=168
A1: L=336
A2: L=672   # optional, only if A1 gives consistent gains
```

不把 72h/96h 作为第一轮主候选，因为它们无法完整覆盖周周期；不一开始采用 672h，因为 NBEATSx 的首层参数量、旧 regime 噪声和计算量都会明显增加。

**晋级规则**：更长输入必须在相同 H34、相同 feature/loss/model size 下，Jun+Jul 两个月 raw 与 balanced 均不下降，并在扩展月上保持收益，才能替换 168。

---

## 3. 一个训练样本到底是什么

对历史 target day `d`，构造唯一合法样本：

```text
origin(d) = d-1 14:00
```

### 3.1 Target backcast

```text
y_backcast:
  [origin-167h, ..., origin]
  shape = [168]
```

只允许真实、在该历史 origin 已经发生的 DA-RT spread。

### 3.2 Historical covariates

CORE5 + calendar 在 backcast 区间的轨迹：

```text
X_hist shape = [168, 9]
```

9 channels：

1. fcast_直调负荷
2. fcast_联络线受电负荷
3. fcast_风电总加
4. fcast_光伏总加
5. fcast_竞价空间
6. hour_sin
7. hour_cos
8. dow_sin
9. dow_cos

### 3.3 Future covariates

```text
X_future:
  d-1 h15-h24 + d h1-h24
  shape = [34, 9]
```

所有 future covariates 必须在 `d-1 14:00` 已知。训练前必须逐列、逐时间点运行 availability audit；不能只因为字段名叫 `fcast_*` 就默认合法。

### 3.4 Targets

```text
y_future shape = [34]
```

其中：

```text
[0:10]  bridge auxiliary target
[10:34] D-day scored target
```

最终 headline metrics 只使用 `[10:34]`。

---

## 4. 预测 D 时允许使用哪些历史样本

严格监督边界：

```text
latest complete target day <= D-2
```

注意这里约束的是**每个训练样本的 scored target day**。

例如预测 `2026-08-27`：

```text
latest supervised historical target day = 2026-08-25
```

`2026-08-26` 虽然 h1-h14 已发生，但其完整 H34 label 仍未成熟，不能成为 supervised training sample。

D-1 h1-h14 只能作为当前预测样本的 observed context，绝不能把 D-1 整日 label 拼入训练。

---

## 5. Rolling 9-month calibration window

第一版业务模型冻结：

```text
calibration horizon = 9 calendar months
```

对于目标日 D：

```text
latest_label_day = D-2
calibration_start = latest_label_day - 9 calendar months + 1 day
```

然后按**样本 target day**选取 9 个月内的 daily-origin samples。

### 5.1 Context extension 不等于扩大训练标签

为了构造 calibration_start 附近第一个样本的 168h backcast，可以读取 calibration_start 之前最多 7 天的历史数据作为 context。

这只是输入上下文，不代表这些更早日被算入 calibration sample count。

---

## 6. 每个目标日的 Train / Validation 划分

v1 不使用随机 validation，因为电价存在明显 regime drift，随机切分会把未来状态混入 early stopping。

对每个目标 D 的 9 个月 calibration samples：

```text
Validation = 最近 28 个完整历史 target days
Train      = 之前剩余的 calibration target days
```

典型规模约为：

```text
9 months ≈ 270 daily samples
validation = 28
train ≈ 235-245
```

已直接用 Cycle88 `numeric_v2_latest_20260828/slot_table.parquet` 核验：

```text
预测 2026-06-01: calibration 273, train 245, validation 28
预测 2026-07-01: calibration 273, train 245, validation 28
预测 2026-08-26: calibration 273, train 245, validation 28
```

所以 batch=32 时每个完整训练 epoch 约 8 个梯度 step，数据规模与本协议假设一致。具体 target day 仍按完整日和数据质量审计决定。

### 6.1 为什么 28 天

- 足够覆盖 4 个周周期；
- 可以同时观察 weekday/weekend 和新能源状态变化；
- 不至于从仅约 9 个月的数据中拿走过多训练样本；
- 比随机 validation 更接近真实未来分布。

### 6.2 最低数据门槛

```text
min_train_samples = 180 daily origins
min_validation_samples = 21 daily origins
```

不足则该目标日不允许训练正式 v1，而不是自动缩短到一个不可比窗口。

---

## 7. 为什么 v1 每天重新训练

NBEATSx 原论文在 hyperparameter 选定后进行 daily recalibration：为每个测试日重新训练/校准，使最新可用信息进入模型。

Cycle 89 v1 采用同样的研究逻辑：

```text
recalibration_frequency = 1 target day
initialization = cold start from deterministic seed
```

也就是预测每个 D 时：

1. 重新构造 D 对应的严格 9m train/28d validation；
2. 从相同初始化规则重新训练；
3. early stopping；
4. 只预测该目标 D；
5. 保存 checkpoint / split / training curve / audit；
6. 下一天重新开始。

这样最慢，但科学归因最干净：没有前一天模型状态穿越到下一天，也不会因为 warm-start 引入无法解释的路径依赖。

### 7.1 后续效率研究，不属于 v1

v1 通过后再测试：

```text
R1: daily cold retrain          # reference
R2: 3-day full retrain
R3: 7-day full retrain
R4: weekly cold reset + daily warm update
```

warm update 必须使用 replay mini-batches，而不能只拿新增的 1 个 day sample 连续微调，否则极易 catastrophic forgetting。

---

## 8. Batch 设计

论文原始数据跨多年，搜索 batch size 为 256/512；我们的 9m daily-origin 数据只有约 200 多个 train samples，机械使用 256/512 会退化成接近 full-batch。

业务 v1：

```text
batch_size = 32 daily-origin samples
shuffle = true within training set only
validation_shuffle = false
```

理由：

- 一个 epoch 约 7-8 个 gradient steps；
- 有足够 stochasticity；
- 不会像 batch=8/16 一样噪声过大；
- 不会像 batch=128/256 一样几乎没有 minibatch regularization。

后续只需受控比较：

```text
16 / 32 / 64
```

不在第一轮做大网格搜索。

---

## 9. Epoch/step 语义

因为不同目标日 train sample 数略有变化，代码以 `optimization_step` 为统一训练预算，而不是固定 epoch 数。

业务 baseline 推荐：

```text
max_steps = 1200
min_steps = 200
val_check_steps = 25
early_stop_patience_checks = 8
```

相当于最多约 150 个小数据 epoch，但通常应由 early stopping 在更早位置停止。

保存：

```text
best_step
best_validation_loss
steps_since_best
train_loss_curve
validation_loss_curve
```

任何 target day 如果跑满 1200 steps 仍持续改善，应标记 `MAX_STEPS_REACHED`，后续单独研究是否欠拟合，而不是静默扩大训练预算。

---

## 10. 模型大小的 v1 约束

论文官方搜索 hidden units 50..500，2 FC layers。

Cycle 89 business v1 的目标不是最大网络，而是让约 200-250 daily samples 能稳定训练。

冻结第一候选：

```text
stacks = [Identity, Exogenous-TCN]
blocks = [1, 1]
FC layers per block = 2
hidden_width = 256
exogenous_encoder_channels = 8
TCN kernel = 3
activation = Softplus
batch_norm = false
dropout_theta = 0.05
dropout_exog = 0.05
```

程序必须输出 `parameter_count`。v1 推荐模型总参数量控制在约 `0.3M~1.5M`，若超过 `2M` 必须在 manifest 中显式告警。

### 后续结构对照

只在 baseline 跑通后比较：

```text
128 vs 256 hidden
Identity->TCN vs TCN->Identity
TCN vs WaveNet
```

不同时改变 input length / loss / feature set。

---

## 11. 标准化

价差可能存在尖峰、正负混合且零点具有业务意义，因此禁止对 target 做会改变 0 位置的 center transform 后直接拿 sign 评价。

### Target

只做**纯尺度缩放，不平移**：

```text
scale_y = max(median(abs(y_train)), scale_floor)
z = y / scale_y
```

建议：

```text
scale_floor = 10.0
```

因此：

```text
sign(z) == sign(y)
```

### Exogenous

每个 covariate 的 scaler 只在 train split 上拟合；推荐 RobustScaler：

```text
(x - median_train) / IQR_train
```

validation/test 只应用 train scaler。

calendar sin/cos 不再缩放。

所有 scaler 参数随 target day checkpoint 保存。

---

## 12. Dataset / DataLoader 必须输出的内容

每个 batch 至少包含：

```text
sample_target_day
origin_timestamp
y_backcast          [B,168]
x_backcast          [B,168,C]
x_future            [B,34,C]
y_future            [B,34]
bridge_mask          [B,34]
score_mask           [B,34]
positive_mask        [B,34]
negative_mask        [B,34]
feature_availability_mask
```

这使 loss、metric 和 leakage audit 都不需要依赖隐式时间位置。

---

## 13. Pilot / 扩展评价

服从 parent chain 的 Cycle88+ 快速纪律：

### First pilot

```text
2026-06
2026-07
```

两个月首先只运行预注册配置，不根据其中某一天临时调参。

### 扩展

只有 Jun+Jul 均不低于父 baseline、balanced 不下降、正负 recall 不明显崩坏后，才扩展：

```text
2026-01
2026-05
2026-08 allowed strict block
```

### Final holdout

保持 untouched，直到：

- input length；
- feature profile；
- architecture；
- loss；
- training window；
- retraining cadence；
- random seeds；

全部冻结。

---

## 14. 必须记录的未来研究分支

不在 v1 同时执行：

1. `GAP_FILL_H24`: 先预测/补 D-1 h15-h24，再做 H24；
2. `INPUT_336`: 两周 backcast；
3. `INPUT_672`: 四周 backcast；
4. `WINDOW_6M/12M`: rolling calibration window；
5. `RECALIBRATION_3D/7D/WARM`: 更新频率；
6. `HORIZON_SPECIALIST`: 根据 horizon error profile 而不是旧的固定 1-8/9-16/17-24 做 specialist；
7. `PERIOD_LOSS_WEIGHTING`: 仅在统一模型已证明某些 horizon 区间稳定失败后再研究。

---

## 15. 核心结论

v1 的训练问题被固定为：

```text
For each target D:
  calibration target days = rolling 9 months, latest <= D-2
  validation = latest 28 complete target days
  train = preceding calibration days
  one legal origin per historical target day
  input = 168 observed hours
  output = 34 direct future hours
  cold retrain every target day
  batch = 32
  early stop chronologically
```

这比把 hourly 序列按 step=1 任意切窗更符合真实业务，也比递归 gap filling 更容易保持严格信息边界。
