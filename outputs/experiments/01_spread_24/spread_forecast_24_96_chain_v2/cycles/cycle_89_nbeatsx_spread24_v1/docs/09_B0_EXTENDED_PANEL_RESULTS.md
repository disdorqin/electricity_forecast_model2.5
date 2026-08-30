---
status: active
date: 2026-08-29
scope: Cycle89 B0 readiness repair and pre-registered extended MAE panel
validation_basis: pytest 42, project preflight 14/14, fixed panel artifacts under runs/b0_extended_panel_14d
---

# Cycle89 B0 Extended Panel Results

## 1. Scope and frozen contract

本轮只运行冻结的 `business_strict34_core` MAE 配置：DA-RT、D-1 14:00 origin、D-2
training cutoff、L=168、H=34、CORE5 + calendar、Identity -> Exogenous TCN、9-month
rolling calibration、最后28个 validation days、daily cold retrain、float32。没有运行
方向损失、Pseudo-Huber、B208、长回看窗口、specialist、warm start 或 AMP。

预注册的14个日期未替换：2026-06-01/05/10/15/20/25/30 和
2026-07-01/05/10/15/20/25/31。每个日期独立训练并只产生24个 D-day scored rows；
总 scored sample_count=336。

## 2. Readiness repairs and gates

- seed-before-model-init：PASS；fresh-process 2026-06-01 两次的 initial state hash 相同，
  best_step 相同（175），target prediction SHA256 相同。
- scaler：PASS；CORE5 数值通道 train-only robust median/IQR，calendar 四通道 identity；
  inference sample 不包含 `y_future`，label 在 forward 后单独 join。
- config execution：PASS；`config_execution_audit.json` 对实际 train_count=245、
  validation_count=28 使用 `>=` 关系，而不是回填配置下限。
- provenance：PASS；每个日期记录 source-data/config/source-code、initial/best state、
  checkpoint、split、environment hash。
- H34：PASS；输出使用 `h34_offset=1..34`，bridge=1..10 / D-day=11..34，
  `business_hour` 分别为15..24 / 1..24。
- holdout/leakage：PASS；holdout registry 未与请求日期重叠，14个日期的 audit 均为
  `STRICT/PASS`，target-day labels 只在 forward 后用于评估。
- project preflight：14/14 PASS。
- Cycle89 pytest：42 passed，5个 warning（Torch weight_norm deprecation）；未见 test failure。
- Git/source：`GIT_DISCOVERABLE_UNTRACKED`。精确 gitignore exception 使源码可发现，
  但 `git ls-files` 仍未列出 Cycle89 文件，因此不宣称已提交或已追踪。

## 3. Reproducibility and device

- Python 3.11.14；Torch 2.6.0+cu124；CUDA 12.4。
- selected device: NVIDIA GeForce RTX 4060 Laptop GPU (`cuda`)；float32，AMP=false。
- deterministic device gate：PASS；same-device tiny forward/backward exact tensor match。
- convergence sanity：200 steps，schedule budget 固定为1200，milestones=300/600/900；
  median grad=4.5400，p90=7.6339，max=14.3495，nonfinite=0，clip_fraction=1.0000。
  training loss 3.6417 -> 1.6270，validation MAE 2.8337 -> 2.6044，均下降，故为
  `PASS + GRADIENT_INSTABILITY_WARNING`，未触发“高 clip 且无下降趋势”阻断。

Fresh-process evidence:

```text
process A/B initial_state_sha256 = b5c143227365af1fe5142dbad885ecab3c36bf1877b2d528129bf54b5344b6fc
process A/B best_step            = 175
process A/B prediction_sha256    = d2adeb85843b9a7d54c789bcf753e5a3e762a65045fdd4681272be3dc147dff6
row_count per process            = 24
```

## 4. NBEATSx 14-day results

### 4.1 Daily rows

| target day | raw | positive recall | negative recall | balanced | MAE |
|---|---:|---:|---:|---:|---:|
| 2026-06-01 | 0.6250 | 0.4444 | 0.7333 | 0.5889 | 51.27 |
| 2026-06-05 | 0.4583 | 0.9000 | 0.1429 | 0.5214 | 164.84 |
| 2026-06-10 | 0.7500 | 0.0000 | 1.0000 | 0.5000 | 214.37 |
| 2026-06-15 | 0.8750 | 1.0000 | 0.0000 | 0.5000 | 65.84 |
| 2026-06-20 | 0.7500 | 0.7778 | 0.6667 | 0.7222 | 84.80 |
| 2026-06-25 | 0.4583 | 0.5789 | 0.0000 | 0.2895 | 100.24 |
| 2026-06-30 | 0.5000 | 0.9000 | 0.2143 | 0.5571 | 94.29 |
| 2026-07-01 | 0.8750 | 1.0000 | 0.0000 | 0.5000 | 125.04 |
| 2026-07-05 | 0.4583 | 0.8462 | 0.0000 | 0.4231 | 112.17 |
| 2026-07-10 | 0.3333 | 0.5000 | 0.2500 | 0.3750 | 68.02 |
| 2026-07-15 | 0.5417 | 0.5417 | N/A | 0.5417 | 58.44 |
| 2026-07-20 | 0.6667 | 1.0000 | 0.5789 | 0.7895 | 100.72 |
| 2026-07-25 | 0.2917 | 0.1875 | 0.5000 | 0.3438 | 47.73 |
| 2026-07-31 | 0.4167 | 0.5714 | 0.2000 | 0.3857 | 66.13 |

### 4.2 Micro and daily macro

| aggregation | raw | positive recall | negative recall | balanced | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| micro hourly, n=336 | 0.5714 | 0.6856 | 0.4155 | 0.5505 | 96.71 | 141.66 |
| daily macro mean | 0.5714 | 0.6606 | 0.3297 | 0.5027 | 96.71 | — |
| daily macro median | 0.5208 | 0.6784 | 0.2143 | 0.5000 | 89.54 | — |

all-positive baseline=0.5774，all-negative baseline=0.4226。结果文件：
`runs/b0_extended_panel_14d/micro_hourly_metrics.json`、`macro_daily_metrics.json`、
`daily_metrics.csv`。

按月的 NBEATSx micro：June raw=0.631、balanced=0.619、MAE=110.81；July
raw=0.512、balanced=0.478、MAE=82.61。

## 5. Same-date Cycle88 comparison

主比较器为同14日期的 Cycle88 `full_existing_F0_F9` LGBM：micro raw=0.6190、balanced=0.5908、
MAE=84.74；daily-macro raw=0.6190、balanced=0.5679、MAE=84.74。

NBEATSx - primary paired delta：

```text
micro raw       -0.0476
micro balanced  -0.0403
micro MAE       +11.97
macro raw       -0.0476
macro balanced  -0.0652
macro MAE       +11.97
```

次比较器 Cycle88 `baseline_2026_lgbm`：micro raw=0.5774、balanced=0.5944、MAE=84.94；
NBEATSx 的 raw=-0.0060、balanced=-0.0439、MAE=+11.77。June-only `trees_220` 也已保留
在 `baseline_comparison_daily.csv`，只使用7个 June 日期。

所有 paired delta 均先按 target day 对齐后计算；没有使用不同日期集合。

## 6. Majority / sign-transition diagnostics

`majority_collapse_diagnostics.csv` 标记3/14天（21.43%）：2026-06-10、2026-06-15、
2026-07-01。三天预测只输出单一符号并达到当日 majority baseline；这部分 raw 不能视为
模型提升。其余日期也存在 minority recall 不稳定：daily macro negative recall=0.3297，
低于 positive recall=0.6606。

实际/预测 sign-switch count 已逐日记录。14天合计 actual=75、predicted=69；模型并非完全
没有切换，但在部分强 majority 日丢失了目标日内转折结构。

## 7. H34 offset diagnostics

完整逐 offset 结果在 `metric_by_h34_offset.csv`，包含1..34、section、business_hour、n、
MAE、bias、raw、positive/negative recall、balanced。聚合上表现最差的 balanced offsets
为 H34=20 (0.3571)、22 (0.3571)、32 (0.3667)、27 (0.3750)；最高为 H34=33 (0.8542)、
H34=1 bridge (0.7917)、H34=31 (0.7778)。这只作为后续证据，不启动 specialist，也不重新
硬编码1-8/9-16/17-24分段。

## 8. Interpretation and recommendation

结论：`WEAK_SIGNAL`。NBEATSx 未在主 Cycle88 comparator 上同时改善 micro raw 与 daily-macro
balanced，MAE 也更差，且有21.43%的 majority-collapse days。14-day panel 是固定开发集，
不是完整 June+July B0，也不是 final holdout；不能把它包装为跨月正式月度成绩。

软件 readiness hard gates 已满足并记录为 `B0_READY=true`，但基于本轮弱信号，**不建议下一轮
直接启动完整 June+July MAE B0**。本轮不因结果追分、不启动 directional loss/feature
expansion；是否进入下一实验必须由新的、明确批准的研究协议决定。

上一轮3-step smoke 的28-day validation `sample_count=672` 仍仅为
`LEGACY_SMOKE_VALIDATION_ONLY`，与本轮336个真实 target-day OOS scored rows无关。