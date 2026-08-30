# Cycle 89 — NBEATSx Spread24 v1

status: active  
date: 2026-08-29  
responsibility: strict NBEATSx paper reproduction + DA-RT business adaptation  
validation basis: NBEATSx IJF paper / official implementation + Cycle 88 strict-origin contract

## Purpose

This cycle creates a complete deep-learning research program rather than a one-off API call. It has two stages:

1. reproduce the published NBEATSx electricity-price architecture/training protocol;
2. adapt that verified core to the project's D-1 14:00 -> D-day spread problem without relaxing any leakage boundary.

## Frozen v1 decisions

- target sign: `DA - RT`
- no period-split models in v1
- input target backcast: 168 hourly points ending at D-1 14:00
- business forecast horizon: 34 hourly points
  - D-1 h15-h24 = 10 auxiliary bridge outputs
  - D h1-h24 = 24 scored outputs
- v1 forecasting strategy: direct multiple-output, no synthetic gap fill
- first business training history: fixed rolling 9 months; 6/12 months are later ablations
- first business feature profile: compact raw primitive forecast trajectories, not B208
- first business baseline loss: paper MAE; the business candidate is a separately tested pseudo-Huber + balanced sign loss
- per-target-day split: rolling 9 months, latest 28 complete historical target days for chronological validation
- v1 recalibration: daily cold retrain; warm-start / 3-day / 7-day updates are later efficiency studies
- v1 batch/model anchor: 32 daily-origin samples, Identity→Exogenous-TCN, 2×256 hidden layers

## Why horizon 34

At the real origin, target history stops at D-1 14:00. Filling D-1 h15-h24 would create synthetic target observations and a second forecasting model inside the pipeline. Instead, v1 asks NBEATSx to predict the entire unknown path in one forward pass. NBEATSx is naturally a fixed-horizon multiple-output model and can consume known future exogenous trajectories across the forecast horizon.

The first 10 outputs are useful auxiliary supervision but are not part of the final D-day headline score.

## Main feature profile

`CORE5` temporal forecast covariates:

1. direct load forecast
2. interconnection received-load forecast
3. wind forecast
4. solar forecast
5. official bidding-space forecast

plus known calendar variables.

The raw historical spread is supplied directly as the autoregressive backcast. This intentionally avoids duplicating the same information through dozens of lag/rolling/profile features in the first NBEATSx test.

## Program modes

- `paper_repro_generic`: NBEATSx-G paper reproduction
- `paper_repro_interpretable`: NBEATSx-I paper reproduction/decomposition
- `business_strict34_core`: verified NBEATSx core, H=34, CORE5, MAE
- later: `business_strict34_directional`, `B208_COMPAT`, 6m/12m window ablations, gap-fill research branch

## Directory layout

```text
cycle_89_nbeatsx_spread24_v1/
├─ AGENTS.md
├─ README.md
├─ experiment_manifest.json
├─ configs/
│  ├─ paper_repro_generic.json
│  ├─ business_strict34_core.json
│  └─ business_strict34_directional.json
├─ docs/
│  ├─ 00_RESEARCH_DECISIONS.md
│  ├─ 01_PAPER_REPRODUCTION_PROTOCOL.md
│  ├─ 02_PROGRAM_ARCHITECTURE.md
│  ├─ 03_H34_TRAINING_DATASET_PROTOCOL.md
│  ├─ 04_STABLE_BUSINESS_LOSS.md
│  └─ 05_REPRODUCTION_AND_EXPERIMENT_ROADMAP.md
├─ src/nbeatsx_spread/
│  ├─ contracts.py
│  ├─ data/
│  ├─ model/
│  ├─ losses/
│  ├─ training/
│  └─ evaluation/
├─ scripts/
├─ tests/
├─ third_party/           # pinned official reference source
├─ share/                 # small collaboration package
│  ├─ monthly/             # monthly comparison tables
│  ├─ daily/               # selected scored predictions
│  └─ metadata/            # hashes and navigation catalog
└─ runs/                  # local generated results, not source
```

## Reading order for collaborators

1. `README.md` and `experiment_manifest.json`: scientific contract and status.
2. `share/monthly/`: the five-month FULLDEV5 comparison and the RACE25 L1 table.
3. `share/daily/`: selected 24-point scored predictions for discussion.
4. `src/nbeatsx_spread/`: the complete Cycle89 core program.
5. `scripts/` and `configs/`: executable entry points and frozen experiment settings.
6. `tests/` and `docs/`: verification and research decisions.

The top-level directories already separate source, configuration, tests,
protocols, reference code and results.  The local `runs/` directory is further
grouped by study (`FULLDEV5`, `history_window_study`, `feature_study`,
`forecast_strategy_stage1`, `C3_DIRMO_10_12_12`, `loss_objective`, etc.).
Its navigation list is exported to `share/metadata/cycle89_results_catalog.json`;
large checkpoints and raw data remain local by design.

## Runtime boundary

This is a complete Cycle89 research program, but not a standalone data bundle.
Run it from the repository root with the root `utils/resolution.py` and the
project environment from `requirements.txt`.  Supply the private canonical
input with `--data`; the collaboration branch intentionally does not contain
raw data, checkpoints, or the sibling Cycle88 run tree.

## Paper sources

- Olivares et al., *Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx*, International Journal of Forecasting, DOI 10.1016/j.ijforecast.2022.03.001
- official implementation: https://github.com/cchallu/nbeatsx
- arXiv: https://arxiv.org/abs/2104.05522

No performance claim is valid until the paper-reproduction parity checks and project leakage audits pass.

## Implementation status (2026-08-29, final readiness round)

`B0_EXTENDED_PANEL_COMPLETE / STRICT-PASS`：已完成本轮 readiness repair，并按预注册日期
运行 14 个独立 cold-retrain 的 NBEATSx MAE H34 OOS 日样本。模型输入与标签读取已分离，
训练只使用 D-2 及更早完整标签；headline 只使用 D-day 24 点。

本轮修复包括：seed-before-model-init、CORE5 numeric-only robust scaling + calendar
identity、runtime-fact config audit、provenance/device identity、H34 offset 命名以及
固定 300/600/900 学习率节点。AMP、directional loss、B208、长回看窗口和 specialist
均未启动。

已验证：

```text
project preflight                         14/14 PASS
Cycle89 pytest                            50 passed
200-step convergence sanity               PASS + GRADIENT_INSTABILITY_WARNING
fresh-process 2026-06-01 reproducibility PASS
fixed extended panel                     14/14 days, 336 scored rows
```

Paper parity 当前为 `SOURCE-EQUATION PASS`（official commit
`fe116d21785fca55670d258756e7c35fcb613eca`，mapped basis/TCN）；不宣称 full official
training/block numerical parity。源码当前状态按协议如实记录为 `GIT_DISCOVERABLE_UNTRACKED`
（未被 `git ls-files` 证明已追踪）。

14-day panel classification: `WEAK_SIGNAL`。本轮结果是固定开发集证据，不是完整
June+July B0，也不是 final holdout；基于该信号，当前不建议下一轮直接扩展完整
June+July MAE B0，应先由下一轮明确选择是否进入受控后续研究。

上一轮 3-step validation 的 `sample_count=672` 仅为 `LEGACY_SMOKE_VALIDATION_ONLY`，
不得与本轮 target-day OOS 数字混用。

## History window study（2026-08-29，Phase A DEV14）

本轮严格执行 `docs/10_LONG_HISTORY_FEATURE_ROLLOUT_RESEARCH_PROTOCOL.md` 与
`configs/next_stage_history_feature_rollout_matrix.json`，只比较 A0=9m、A1=24m、
A2=36m；CORE5、L168/H34、Identity→TCN、MAE、seed=42、验证28日及其他训练参数全部冻结。
未运行 feature stage、rollout stage、directional loss、AMP、warm start 或额外调参。

三候选均为 14 个预注册 DEV14 日期的独立 cold retrain，headline 为每个 target day 的
24 个 D-day 点（每候选 336 点），且审计状态为 `STRICT/PASS`：

| 窗口 | micro raw | + recall | - recall | balanced | MAE | macro balanced | collapse days |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 9m | 57.14% | 68.56% | 41.55% | 55.05% | 96.71 | 50.27% | 3 |
| A1 24m | 61.61% | 77.84% | 39.44% | 58.64% | 86.75 | 51.24% | 5 |
| A2 36m | 67.86% | 80.41% | 50.70% | 65.56% | 84.85 | 52.01% | 7 |

A2 的 pooled raw / MAE 最好，但其 daily-macro balanced 仅较 A0 高 1.74 个百分点，且
majority-collapse 从 3 天增至 7 天；A1/A2 均未通过预注册的稳健性晋级门。因此按协议
推荐 **A0_HISTORY_9M** 作为 feature stage 的冻结窗口，不启动 A3/A4，也不把 DEV14
结果当作 final holdout 成绩。相对同日 Cycle88 LightGBM，NBEATSx 的 paired mean
Δ(raw/balanced/MAE) 分别为：A0 `-4.76pp/-6.52pp/+11.97`，A1
`-0.30pp/-5.56pp/+2.01`，A2 `+5.95pp/-4.79pp/+0.11`。

完整机器产物位于 `runs/history_window_study/comparison/`，包括 daily/micro/macro、
paired deltas、训练样本、梯度、majority-collapse、transition 指标及
`history_window_review.md`。CONFIRM21 与 2026-09 lockbox 保持未触碰。

## A3 validation-robustness follow-up（2026-08-29）

按 `docs/11_HISTORY_REVIEW_AND_A3_VALIDATION_ROBUSTNESS_PROTOCOL.md`，仅运行
`A3_HISTORY36_VAL84`：36 个月历史、84 天 chronological validation、DEV14 14 日，
其余设置与 A2 完全一致。A3 生成 14 个独立 cold-retrain，headline 共 336 个 D-day 点，
审计状态 `STRICT/PASS`。

结果：Raw `65.48%`，Balanced `64.16%`，MAE `80.41`，Daily Macro Balanced
`49.23%`，positive recall `72.68%`，negative recall `55.63%`，majority-collapse
`9/14` 天；transition precision/recall/F1 均为 `0`。5 个科学 gate 通过 `2/5`
（仅 MAE 与 negative/minority 条件通过），最终状态为
**`LONG_HISTORY_REGRESSION`**。A3 未证明 84 日 validation 能修复 36m 的稳健性问题，
因此停止于本轮，不启动 feature stage、rollout 或 directional loss。

A3 产物位于 `runs/history_window_study/36m_val84/` 与
`runs/history_window_study/comparison_a3/`；其中含 A0/A2/Cycle88 同日 paired delta、
transition 与 H34 offset 诊断。

## Compact feature rescue（2026-08-29，DEV14）

本轮严格执行 `docs/12_A3_DIAGNOSIS_AND_FEATURE_RESCUE_PROTOCOL.md` 与
`configs/compact_feature_rescue_matrix.json`，只在 A2 的 36m/28d chassis 上运行
F0、F1_PHYSICAL_SHAPE、F2_CAUSAL_PRICE_STATE 三个独立 package。没有运行 F1+F2、
forecast-error features、B208、rollout、directional loss、history/validation search、
CONFIRM21 或 September lockbox。设备为 deterministic CUDA，precision 为 float32。

执行验收：project preflight `14/14 PASS`；Cycle89 pytest `65 passed`；42/42 个 target-day
run 的 strict leakage audit `STRICT/PASS`；每个 target day 均为独立 cold-retrain，headline
均严格为 D-day 24 点。F1/F2 增加输入通道导致 parameter guard 分别产生可审计的
`MODEL_CAPACITY_WARNING`（并未改变 hidden/TCN/优化器配置）。

| package | micro raw | + recall | - recall | micro balanced | MAE | daily-macro balanced | macro minority recall | collapse | transition F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 A2 CORE5 | 67.86% | 80.41% | 50.70% | 65.56% | 84.85 | 52.01% | 14.54% | 7/14 | 0.1677 |
| F1 physical shape | 57.44% | 50.52% | 66.90% | 58.71% | 160.17 | 49.68% | 5.56% | 8/14 | 0.0260 |
| F2 causal price state | 61.90% | 94.33% | 17.61% | 55.97% | 81.34 | 53.57% | 10.71% | 9/14 | 0.0918 |

F2 是本轮相对 A2 **最强的部分改善候选**：daily-macro balanced `+1.56pp`、MAE
`-3.52`，但 raw `-5.95pp`、macro minority recall `-3.83pp`、collapse `+2` 天、
transition F1 `-0.0759`，未达到预注册的综合 rescue gate。F1 同时损害 Raw、MAE、
daily-macro balanced、minority recall、collapse 与 transition。因此本轮结论是
**没有 feature package 通过 promotion；不要把 F2 作为已验证晋级方案**。详细 paired
delta、H34 offset、逐日 collapse/transition 与 Cycle88 同日对照位于
`runs/feature_study/comparison/`，总清单为 `runs/feature_study/feature_study_manifest.json`。

当前状态：`COMPACT_FEATURE_STUDY_COMPLETE / STRICT-PASS`，feature rescue `REJECTED_NO_PROMOTION`。
CONFIRM21、September lockbox、feature cross-history check 与后续 rollout 均保持未启动。


## Forecast Strategy Stage-1（2026-08-29，DEV14）

本轮严格执行 `docs/13_FORECAST_STRATEGY_SURVEY_AND_EXPERIMENT_PROTOCOL.md` 与
`configs/forecast_strategy_stage1_matrix.json`，仅运行 C0_DIRECT_H34、C1_GAP_DIRECT_D24、
C2A_BRIDGE_TF。仅使用预注册 DEV14 的 14 个日期；禁止 C2B/C2C、DIRMO、RecMO、
Recursive H1、scheduled sampling、新特征、directional loss、history search、CONFIRM21
和 September lockbox。C0 使用已冻结的 A2/F0 精确参考产物；C1 完成 14 次独立 cold retrain；
C2A 完成 14 次 Stage-1 + 14 次 Stage-2 独立 cold retrain。设备为 deterministic CUDA，
float32。

42/42 个 target-day run 的 strict leakage audit 均为 `STRICT/PASS`；每个候选的 OOS headline
均严格为 14×24=336 个 D-day 点，C2A 推理只使用 Stage-1 预测 bridge，teacher-forcing 合同
记录为 `LEGAL_TF_TRAIN_PREDICTED_BRIDGE_INFERENCE`。

| strategy | micro raw | + recall | - recall | micro balanced | MAE | daily-macro balanced | minority recall | collapse | transition F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_DIRECT_H34 | 67.86% | 80.41% | 50.70% | 65.56% | 84.85 | 52.01% | 14.54% | 7/14 | 0.1677 |
| C1_GAP_DIRECT_D24 | 57.74% | 66.49% | 45.77% | 56.13% | 89.13 | 45.64% | 8.96% | 5/14 | 0.0595 |
| C2A_BRIDGE_TF | 61.01% | 66.49% | 53.52% | 60.01% | 90.12 | 49.38% | 7/14 | 0.0574 |

C2A Stage-1 bridge MAE 均值为 97.77，bridge balanced 为 68.43%；bridge error 与 D24 MAE
相关系数为 0.2908，最高 bridge-error 四分位的 D24 MAE 均值为 126.89。相对 C0 的同日
paired mean Δ(raw/balanced/MAE)：C1 为 -10.12pp/-6.37pp/+4.28，C2A 为
-6.85pp/-2.63pp/+5.26。相对 Cycle88 同日基准：C0 为 +5.95pp/-4.79pp/+0.11，
C1 为 -4.17pp/-11.16pp/+4.39，C2A 为 -0.89pp/-7.42pp/+5.38。

本轮结论：**`NO_FORMULATION_SIGNAL_YET`**。C1 虽将 collapse 从 7 降至 5 天，但 Raw、
macro balanced、minority recall、transition F1 和 MAE 均未形成综合改进；C2A 也未达到预
注册晋级条件。完整机器产物与诊断位于 `runs/forecast_strategy_stage1/comparison/`。
本轮不启动任何 Stage-2 策略，也不启动完整 June+July B0。


## C3 DIRMO 10+12+12（2026-08-29，DEV14）

本轮严格执行 `docs/14_C3_DIRMO_BLOCK_STUDY_PROTOCOL.md` 与
`configs/c3_dirmo_10_12_12.json`，只运行固定的独立 direct-block 分解：B0=Bridge10、
B1=D-day 前12点、B2=D-day 后12点。三块均从同一 D-1 14:00 合法输入独立预测，预测不互相
回灌；仅使用 DEV14 的14个日期、36个月训练历史、28日 chronological validation、CORE5、
MAE、Identity→TCN、seed42、float32。共完成42个独立 block cold retrain。

42/42 个 target-day strict leakage audit 均为 `STRICT/PASS`，每个 target day 均输出24个
D-day headline rows；bridge 仅作诊断。未运行 C2B/C2C、其他 block size、RecMO、Recursive H1、
新特征、directional loss、history/validation/model-size search、CONFIRM21 或 September lockbox。

| 指标 | C3 DIRMO | C0 DIRECT_H34 |
|---|---:|---:|
| micro Raw | 63.39% | 67.86% |
| positive recall | 73.71% | 80.41% |
| negative recall | 49.30% | 50.70% |
| micro Balanced | 61.50% | 65.56% |
| MAE | 84.32 | 84.85 |
| daily macro Balanced | 51.31% | 52.01% |
| majority collapse | 5/14 | 7/14 |
| transition F1 | 0.1000 | 0.1677 |

C3 相对 C0 的同日 paired mean Δ(raw/balanced/MAE) 为 **-4.46pp/-0.70pp/-0.53**；相对
Cycle88 为 **+1.49pp/-5.48pp/-0.42**。C3 系统平均参数量约 3,289,346，平均每目标日
训练约126.35秒、推理为3次 forward pass；额外容量成本已单独记录，不能将收益归因于结构本身。

最终判断：**`DIRMO_MIXED_SIGNAL`**。C3 改善了 MAE 和 collapse 天数，但 Raw、Balanced、
daily-macro Balanced、minority recall 与 transition F1 未形成稳健综合提升，因此本轮停止，
不自动启动 RecMO 或其他 block 研究。完整产物位于 `runs/C3_DIRMO_10_12_12/comparison/`。

## FULLDEV5 full-month cross-month validation（2026-08-30）

本轮严格按 `docs/15_FULL_MONTH_CROSS_MONTH_VALIDATION_PROTOCOL.md` 与
`configs/full_month_cross_month_dev5.json` 执行，只比较 C0_DIRECT_H34、
C3_DIRMO_10_12_12 和冻结的 Cycle88 strict comparator。完整月份为 2026-01、02、04、06、07，
共 150 个 target days、每策略 3,600 个 D-day scored points；36m history、VAL28、CORE5、MAE、
float32、daily cold retrain 全部冻结。没有运行 RecMO、Recursive H1、其他 block size、新特征、
directional loss、CONFIRM21 或 September lockbox。

验收：project preflight `14/14 PASS`；Cycle89 pytest `83 passed, 5 warnings`；150/150 日期的
strict pre-training leakage audit 与模型输入/标签分离均通过，所有 target-day prediction 均严格为
24 行，状态 `STRICT/PASS`。设备为 deterministic CUDA（NVIDIA GeForce RTX 4060 Laptop GPU），
precision `float32`，AMP `false`。C0/C3 的 daily artifacts 采用 `COLD_RETRAIN_NEW`；仅本轮中已
完成且身份哈希一致的前两日 artifacts 标记 `REUSED_HASH_VERIFIED`，未复用旧 DEV14 结果。

| strategy | micro raw | + recall | - recall | balanced | MAE | daily-macro balanced | collapse days | transition F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_DIRECT_H34 | 52.53% | 55.49% | 47.84% | 51.66% | 99.18 | 50.22% | 46/150 | 0.1227 |
| C3_DIRMO_10_12_12 | 52.75% | 57.62% | 45.04% | 51.33% | 103.29 | 50.93% | 33/150 | 0.1136 |
| Cycle88 strict comparator | 56.12% | 70.44% | 33.38% | 51.91% | 87.24 | 51.60% | 37/150 | 0.1342 |

C3 相对 C0 的 paired daily mean Δ(raw/balanced/MAE) 为 `+0.23pp/+0.71pp/+4.11`；相对 Cycle88
为 `-3.36pp/-0.67pp/+16.05`。C3 的 collapse 从 46 天降至 33 天，但 transition F1、MAE 和
跨月稳健性门均未达标；6 个预注册 promotion gate 仅 `2/6` 通过（collapse 与 raw 宽松门）。
未见/已见月份分层、paired bootstrap 95% CI、minority recall、transition、collapse、H34 offset
和 runtime/capacity 明细均位于 `runs/FULLDEV5/`，其中 `horizon_by_month.csv` 保持 bridge
offset1-10 与 D-day offset11-34 分离，bridge 不进入 headline。

本轮结论：**`DIRMO_FULLMONTH_MIXED`**，不是晋级信号。按协议停止，不启动下一阶段结构、loss 或
feature 实验；当前不建议把 C3 作为完整 June+July MAE B0 的替代方案。

## L1 loss objective race（2026-08-30）

本轮严格按 `docs/17_LOSS_OBJECTIVE_DEEP_SURVEY_AND_BUDGETED_EXPERIMENT_PROTOCOL.md` 与
`configs/loss_objective_budgeted_matrix.json` 执行，仅运行 `L1_D24_MAE_BRIDGE025`：
`D-day MAE + 0.25 × bridge MAE`。C0_DIRECT_H34、36m、VAL28、CORE5、float32、seed42、
daily cold retrain 全部冻结；C0 仅复用 `runs/FULLDEV5/C0/` 的 hash-verified artifacts，未重新训练。
没有运行 L2/L3/L4、RACE50、CONFIRM21、September、完整150日回测、新特征、结构/历史/验证/模型规模搜索。

unit tests `86 passed, 5 warnings`，project preflight `14/14 PASS`；RACE5 与 RACE25 的候选日期
均通过 strict leakage audit，设备为 deterministic CUDA，precision `float32`，AMP `false`。

RACE25（25日、600个 D-day scored points）结果：

| objective | Raw | + recall | - recall | Balanced | MAE | daily-macro Balanced | minority recall | collapse | transition F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 49.83% | 54.62% | 42.80% | 48.71% | 100.71 | 51.29% | 14.89% | 6/25 | 0.1637 |
| C0 | 56.17% | 72.83% | 31.69% | 52.26% | 86.70 | 52.21% | 14.10% | 7/25 | 0.1065 |

L1 相对 C0 的 pooled paired delta 为 Raw `-6.33pp`、+ recall `-13.21pp`、- recall `+11.11pp`、
Balanced `-3.55pp`、MAE `+14.00`；daily-macro Balanced `-0.92pp`，minority recall `+0.78pp`，
collapse `-1` 天，transition F1 `+0.0573`。五个月均无同时满足“daily-macro Balanced 改善且 MAE 不超过
105%”的月份（breadth `0/5`）。RACE25 明确信号门槛 `2/7`，未达到，结论为 **REJECT / no promotion**。

完整 L1 产物位于 `runs/loss_objective/L1_D24_MAE_BRIDGE025/`；RACE5/RACE25 结束后按协议停止，
不启动后续 loss 或完整月度 B0。

## 协作分享包

为便于通过 Git 与师兄讨论，`share/` 按用途分成三个子目录，仅保留轻量、可审计结果：

- `share/monthly/cycle89_monthly_results.csv`：FULLDEV5 五个月完整月度结果 + RACE25 L1 五个月结果；
- `share/daily/cycle89_race25_daily_predictions.csv`：RACE25 的 L1 与冻结 C0 同日 24 点预测；
- `share/metadata/cycle89_results_catalog.json`：阅读顺序和本地结果目录索引；
- `share/metadata/cycle89_share_manifest.json`：来源、哈希、泄露状态及排除项。

本协作包不包含 `runs/`、checkpoint、原始数据或大型缓存；完整运行产物继续留在本机。
