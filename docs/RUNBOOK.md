# Runbook

## Prerequisites

- Conda environment `epf-2` with dependencies installed (`pip install -r requirements.txt`)
- Input data file at `data/24/canonical/shandong_pmos_hourly.xlsx`
- CUDA-capable GPU for TimeMixer and RT916 models
- For GPU TimeMixer, keep `--deterministic` off: strict CUDA determinism is
  rejected because the configured Torch/CUDA build has no deterministic
  `upsample` backward kernel. Use CPU for strict deterministic experiments.
- The legacy scheduler serializes CPU/GPU queues by default to isolate this
  Torch runtime policy. The 96-point `split_process` scheduler instead starts
  isolated CPU/GPU children together: CPU is DAG-aware with two workers and
  GPU is strictly serial with one worker. Its task graph, not model-name
  filtering, controls DA→RT prerequisites.

> **Self-contained:** LightGBM and TimesFMBackend are bundled in this repository.
> No external EPF v1.0 repository is required. The `--epf-v1-root` option is
> retained only for legacy compatibility.

## 0.1 Formal 96-point façade

```powershell
python main.py --96 YYYY-MM-DD
python main.py --96 YYYY-MM-DD --predict both
python main.py --96 YYYY-MM-DD --predict dayahead
python main.py --96 YYYY-MM-DD --predict realtime
python main.py --96 YYYY-MM-DD --finish
```

`--predict` only materializes the requested prediction ledger scope;
`--finish` reuses that day's strict prediction provenance and runs
`ledger_weight → ledger_fuse → final_outputs → postflight`. Formal 96 uses
CPU DAG workers=2, GPU serial=1, RT916 stride=24 and `smape_reg/SLSQP`.
`ledger_classifier` is recorded as `disabled_by_production_policy`; its source,
cache and replay entry points remain available outside the formal façade.

Formal96 prediction cache reuse is fail-closed: cached CSVs must carry the current
`formal96_dynamic_snapshot_v1` contract and match resolution/resource mode,
snapshot/FeatureView/model/task;
RT916 additionally proves stride=24 and SGDFNet proves D-1 decision-day anchor with
`fallback_used=false`. A failed full attempt preserves the latest valid prediction
provenance, so a later `--finish` can recover Stage1 and proceed to the strict-history gate.

## 1. Environment Check

```powershell
conda run -n epf-2 python scripts/env_check.py
```

Expected output:
```
ENV_CHECK
python: 3.11.x OK
cuda: available OK
dependencies: all OK
local_paths: all OK
status: PASS
```

## 2. Smoke Test (Quick Validation)

Run a minimal end-to-end test with reduced training scope:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_smoke --date 2026-02-24 ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx ^
    --smoke-training-months 3 --smoke-timemixer-epochs 3 --smoke-timemixer-patience 1 ^
    --max-cpu-workers 2 --max-gpu-workers 1 ^
    --seed 42 --deterministic --force
```

Expected: ~10-15 minutes. All production model legs produce 24-hour predictions.

## 3. 30-Day Backfill

Populate the prediction ledger with historical data:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_backfill ^
    --start YYYY-MM-DD --end YYYY-MM-DD ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx ^
    --max-cpu-workers 2 --max-gpu-workers 1 ^
    --seed 42 --deterministic
```

Example: 2026-01-25 through 2026-02-23 (30 days).

Expected runtime: overnight (~10-12 hours depending on GPU).

Progress tracking: each completed day logs to `outputs/runs/backfill_manifest.json`.

## 4. Full Pipeline (Single Day, legacy/24 compatibility entry)

This advanced `--pipeline` example documents the legacy 24-point five-stage
delivery.  The supported 96-point production façade is the four-stage path in
§0.1 (`--96 DATE`); it records the classifier as
`disabled_by_production_policy` and does not consume corrected outputs.

Run all 5 stages for a target date:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_full ^
    --date YYYY-MM-DD ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx ^
    --max-cpu-workers 2 --max-gpu-workers 1 ^
    --seed 42 --deterministic
```

Expected runtime: ~30-40 minutes (first run; subsequent runs are faster due to prediction cache).

## 5. Verify Existing Outputs

Validate a completed run without re-running models:

```powershell
conda run -n epf-2 python scripts/tests/verify_final_pipeline.py --date YYYY-MM-DD --runs-root outputs/runs
```

Expected output:
```
FINAL_VERIFY: YYYY-MM-DD
ledger_predict: complete
ledger_weight: complete
ledger_fuse: complete
ledger_classifier: complete
final_outputs: complete
errors: 0
warnings: 0
FINAL_STATUS: PASS
```

## 6. Reproducibility Check

Verify that two runs with the same seed produce identical outputs:

```powershell
conda run -n epf-2 python scripts/tests/check_reproducibility.py YYYY-MM-DD ^
    --seed 42 --deterministic --epf-v1-root "path\to\epf-v1" --keep-tmp
```

Expected: `=== Result: PASS (all outputs identical) ===`

## 7. TimeMixer Alignment Check

Verify TimeMixer timestamp alignment (hour 24 = D+1 00:00 rule):

```powershell
conda run -n epf-2 python scripts/tests/check_timemixer_alignment.py --date YYYY-MM-DD
```

Expected: `ALL OK` for both dayahead and realtime.

## 8. Stage-by-Stage Commands (Debug; 24 legacy/advanced compatibility)

Run individual stages if rerunning from scratch:

```powershell
# Stage 1: Predict
conda run -n epf-2 python main.py --pipeline ledger_predict --date YYYY-MM-DD

# Stage 2: Weight learning
conda run -n epf-2 python main.py --pipeline ledger_weight --date YYYY-MM-DD

# Stage 3: Fusion
conda run -n epf-2 python main.py --pipeline ledger_fuse --date YYYY-MM-DD

# Stage 4: Classifier
conda run -n epf-2 python main.py --pipeline ledger_classifier --date YYYY-MM-DD
```

以上分类器命令仅适用于 24 legacy 或 shadow/replay。formal 96 production 不执行
ExtremePriceClf，manifest 标记 `disabled_by_production_policy`，RT fuse 直接进入 final。

分类器 bridge 使用可复用的 `range_runner`，不再通过生产 bridge 启动旧的
`run_daily.py` 子进程。24 点和 96 点均先规范化到分类器的小时语义；96 点输入按小时
聚合，最终校正结果再由 bridge 广播回 96 个 15 分钟槽位。classifier/shadow 缓存按调用 profile 分域。当前正式布局的可复用 cache 位于：

```text
outputs/24/cache/classifier/realtime/<source-spec-hash>/
outputs/96/cache/classifier/realtime/<source-spec-hash>/
```

只有显式 compatibility `feature_store` profile 才使用 `outputs/{24,96}/feature_store/cache/...`；当前 formal96 磁盘上该 tree 已归档并不存在。

缓存包括规范化输入、Stage1/Stage2 特征、p1 概率和 manifest；扩展日期范围时只补齐
缺失时间戳，不重复执行历史预热。旧入口仍保留用于兼容和对照，不是正式 bridge 的
默认执行路径。实验区可直接运行：

```powershell
python scripts/experiments/classifier_range/run_range.py `
  --source data/24/canonical/shandong_pmos_hourly.xlsx `
  --start 2026-01-01 --end 2026-01-07 `
  --resolution hourly --task realtime
```

切换前的 24 点单日逐点回归已通过：决策列一致，概率最大绝对误差约
`5.6e-17`；7 日增量回放使用同一缓存完成。分类器只改变执行组织和缓存，不改变
原有特征、模型、阈值、滚动训练和校正规则。

## 9. Force Re-run

To force a full rerun (bypass prediction cache):

```powershell
conda run -n epf-2 python main.py --pipeline ledger_full --date YYYY-MM-DD ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx --force
```

To force a specific stage, add `--force` after the cleaned outputs.

## Common CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--date` | — | Target date YYYY-MM-DD |
| `--start` / `--end` | — | Date range for backfill |
| `--epf-v1-root` | None | [Optional legacy compatibility] External EPF v1.0 root |
| `--seed` | 42 | Global random seed |
| `--deterministic` | False | Strict deterministic mode; CPU-only for TimeMixer |
| `--max-cpu-workers` | 2 | CPU parallel workers |
| `--max-gpu-workers` | 1 | GPU serial workers |
| `--realtime-cutoff-hour` | 14 (24 legacy) | Legacy/advanced compatibility parameter; formal `--96` serving visibility comes from the immutable snapshot/FeatureView |
| `--force` | False | Force rerun, bypass cache |
| `--data-path` | `data/24/canonical/shandong_pmos_hourly.xlsx` | Input data file path |
| `--ledger-root` | resolution-dependent | Override ledger storage root |
| `--runs-root` | resolution-dependent | Override daily run output root |
| `--output-profile` | `production` | Production roots under `outputs/{24,96}`; `legacy` / `feature_store` are compatibility profiles |
| `--feature-store-root` | profile-derived | Candidate raw/materialized feature cache root |

To run the isolated FeatureStore candidate chain without touching the legacy
ledger:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_full --date YYYY-MM-DD ^
    --resolution 15min --output-profile feature_store --feature-store-mode raw
```

The business model pool is fixed in `fusion/model_pool.py`: Dayahead uses
`lightgbm + timesfm + timemixer`; Realtime uses
`timesfm + sgdfnet + timemixer + rt916`. Do not repeat or override these
lists in pipeline scripts.

## Verification Scripts Reference

```powershell
# Full environment check
python scripts/env_check.py

# Verify final pipeline outputs
python scripts/tests/verify_final_pipeline.py --date 2026-02-24 --runs-root outputs/runs

# Reproducibility
python scripts/tests/check_reproducibility.py 2026-02-24 --seed 42 --deterministic

# TimeMixer alignment
python scripts/tests/check_timemixer_alignment.py --date 2026-02-24

# Smoke verification
python scripts/tests/verify_smoke.py
```

## 10. 数据同步、范围运行与部署职责

本文件统一承接数据同步、范围回测和服务器运行的操作入口；旧版专项手册已归档，后续只更新本节。

### 10.1 数据同步

```powershell
python main.py --pipeline sync_dataset --resolution hourly --sync-source db
python main.py --pipeline sync_dataset --resolution 15min --sync-source db
```

同步前后必须检查源表、分辨率、最新时间、重复键、每日 96 点完整性和 manifest。同步失败不得覆盖上一份有效快照。

### 10.2 多日范围运行

```powershell
python main.py --pipeline ledger_full_range --start YYYY-MM-DD --end YYYY-MM-DD --resolution hourly --seed 42 --deterministic
```

范围运行按正式 profile 执行：`ledger_predict → ledger_weight → ledger_fuse → final_outputs`；24 点 legacy 仍可执行 classifier 第四阶段。formal 96 完整运行前必须确认每个任务具备最近 30 个严格 production 完整训练日；不满足时在模型启动前 fail-fast，写入 `INSUFFICIENT_STRICT_HISTORY` / `FAILED_NO_DELIVERY`，不允许 degraded fallback 或生成新 submission。仅 `--predict ...` 任务入口可继续运行以积累正式 ledger。24 legacy 的降级语义保持不变。

### 10.3 GPU/CPU环境

- Python、Torch、JAX 和 CUDA 版本以根目录 `requirements.txt` 为准；
- LightGBM、TimesFM 的轻量验证可在 CPU 执行；TimeMixer/RT916 完整训练按 GPU 环境执行；
- Windows 下训练脚本必须遵守 DataLoader worker 约束；
- 运行 manifest 记录 `resolution`、`cutoff`、`seed`、数据快照、git commit 和权重门控阈值。

### 10.4 回归顺序

1. 环境和数据预检；
2. CLI/编译检查；
3. 单模型或 smoke 检查；
4. ledger、权重门控、融合和最终输出检查；
5. 需要正式交付时再执行完整链路。

### 10.5 冠军门控学习器（实验候选）

`champion_short` 是显式实验选项，不改变当前生产默认 `smape_reg`，当前只允许用于已标记为
`historical-invalid-features` 的旧 96 点预测账本做相对融合验证。协议固定为
14 日窗口（7 日拟合 + 7 日验证）、DA/RT 半衰期均 7 日；负权只在有界冠军锚定
候选中出现。关闭融合权重门控必须显式传 `--weight-prune-threshold 0`，不得作为
生产默认值。

```powershell
python main.py --date YYYY-MM-DD --pipeline ledger_weight --resolution 15min `
  --validation-days 14 --weight-learner champion_short `
  --ledger-root outputs/ledger_96 `
  --runs-root outputs/experiments/03_fusion_weighting/champion_weight_ab/integration_runs `
  --weight-prune-threshold 0

python main.py --date YYYY-MM-DD --pipeline ledger_fuse --resolution 15min `
  --ledger-root outputs/ledger_96 `
  --runs-root outputs/experiments/03_fusion_weighting/champion_weight_ab/integration_runs `
  --weight-prune-threshold 0
```

验证重点是 `weight/candidate_metrics.csv`、`weights.csv`、`fuse/fused_debug.csv`
和 manifest；新鲜有效预测集建立后，必须在独立账本上重新回测，满足稳定性门槛后
才能讨论替换生产学习器。目标日预测必须包含模型池中的全部模型；缺失任一模型时
`champion_short` 应拒绝学习并由融合阶段报错，不得用残缺模型集合交付。

### 10.6 服务器 96 点预测账本（先预测、后回放）

96 点服务器预测统一走正式 runner。生产模型层长期只维护 `model_input_full.parquet` 一份；闭合历史按需从 full 逻辑筛选，不再每日重建 clean。formal96 不再把 as-of/model scratch 平铺到 `outputs/96/runtime/`：每次 invocation 在 resolved `runs_root` 的 sibling `runtime/attempt_<date>_<attempt>/` 下建立一个 sandbox，masked parquet 与五模型 scratch 全部归属该 attempt；NORMAL 后整目录删除，失败时单目录暂留给诊断/TTL。用户不需要手工准备 masked 数据。

先同步并刷新单一 full 模型仓：

```bash
python main.py --pipeline sync_dataset --resolution 15min \
  --sync-source db --sync-mode full --force-sync
```

单日真实场景 smoke（Dynamic-v1；无需显式 `--data-path`）：

```bash
export OPTIM_NUM_WORKERS=0
python main.py --pipeline ledger_predict \
  --target realtime --models timesfm \
  --date 2026-09-16 --resolution 15min --force
```

manifest 必须显示：
- `model_input_source = data/96/model_input/shandong_pmos_96_model_input_full.parquet`；
- `serving_protocol = formal96_dynamic_snapshot_v1`；
- `dynamic_snapshot.protocol = formal96_dynamic_snapshot_v1`、`snapshot_id` 与 values/manifest 路径；
- `feature_view.status = PASS`、`feature_view.target_truth_mask = true`；
- target-day 10 个 canonical forecast 字段均为 96/96。

正式 96 DA/RT 四阶段使用 `--96 DATE`（旧 `--pipeline ledger_full` 仍可作 advanced/legacy compatibility）。当前目标的 `--96` façade 先强制 DB sync，再生成 immutable D/T snapshot 和 FeatureView；已 closed 的历史目标优先复用成功 LIVE snapshot，没有时生成显式 `HISTORICAL_PROXY_V1`，两者均汇合到同一 FeatureView。sync/source 失败以 `DATABASE_SYNC_FAILED` 或 `MISSING_CRITICAL_SOURCE` fail-closed，不使用 stale/degraded fallback。`--finish` 只复用 Stage1 snapshot/provenance，不 sync、不新建 snapshot。随后 façade 强制 production profile，并 fail-closed 拒绝已知 legacy roots：`outputs/ledger_96`、`outputs/runs_96`、旧 `outputs/cache` 或 `outputs/96/feature_store/*`；甲方若确需自定义状态目录，可使用其他独立 deployment root。目标日 actual 在 live 模式并非必需；仅做历史结算/
验收时显式增加 `--require-target-actual`。若目标日 D 的 96 点预测型基本面尚未进入
`epf_pmos_96_full`，runner 必须以 `TARGET_FORECAST_NOT_READY` fail closed，禁止用 D-1
预测值代替。

legacy/shadow replay 若从 2026-08-15 继续 ExtremePriceClf，旧 `p1_cache` 截止
2026-08-14 23:00。v5 cache 会先验证历史预测特征和训练标签前缀；语义一致才自动继承，
否则安全失效并重算。文件路径、size、mtime 变化本身不再导致昂贵 p1 cache 冷启动。

正式连续 prediction replay 使用 `scripts/server/run_96_prediction_backtest.py`；legacy 对照可显式
`--resource-mode legacy`，formal96 production 固定 `split_process`（CPU=2、DAG-aware、GPU=1），
并强制 `--require-target-actual`。resume 不以“账本已有96点”作为
唯一依据：还必须存在同日 run manifest，证明 Dynamic-v1 snapshot/FeatureView、target truth mask、
完整生产模型池以及相同 resource mode。正式 façade 的 classifier
仍保持 `disabled_by_production_policy`。服务器 A/B 对照中 legacy 保持 CPU=1；split_process
正式候选固定 CPU=2、DAG-aware，GPU 两种模式均为
严格串行 1 worker；两种模式的 worker 配置会写入 range manifest 并由 artifact audit 校验。

```bash
python scripts/server/benchmark_96_resource_modes.py \
  --date 2026-08-16
```

该命令分别写入隔离的 `legacy/` 与 `split_process/` root，并记录总 wall time、每模型 wall time、
进程树 RAM/CPU 峰值、GPU VRAM/利用率、OOM、7模型完整性、prediction/ledger 数值差异。
没有下游对比时 promotion gate 必定保持 `KEEP_LEGACY_DEFAULT`。连续严格账本满30天后，再用同一份
seed state 比较 weight/fuse/classifier/final：

```bash
python scripts/server/benchmark_96_resource_modes.py \
  --date YYYY-MM-DD --run-downstream \
  --seed-ledger-root outputs/96/ledger \
  --seed-runs-root outputs/96/runs \
  --seed-cache-root outputs/96/cache \
  --seed-resource-mode legacy \
  --require-promotion-gate
```

A/B seed 的前30天必须逐日通过严格 production provenance + prediction/actual ledger 审计，否则
harness 以 `STRICT_SEED_HISTORY_NOT_READY` fail closed。正式 post-run 机械链路验收使用：

```bash
python scripts/server/audit_96_artifacts.py \
  --output-root outputs/96 --phase prediction \
  --start YYYY-MM-DD --end YYYY-MM-DD --resource-mode split_process
```

默认 prediction audit 按 **live serving** 语义验收：目标日 actual 可以尚未完整，但已落地的 actual 行必须槽位合法、唯一且 finite；如果是历史回放/结算，要求目标日 actual 也严格 96/96，则追加：

```bash
python scripts/server/audit_96_artifacts.py \
  --output-root outputs/96 --phase prediction \
  --start YYYY-MM-DD --end YYYY-MM-DD --resource-mode split_process \
  --require-target-actual
```

formal96 的 `NEXT DAY LEDGER READINESS` 与实际 learner 使用**同一个 adaptive selector**：对 T+1 从 `T-1`（lag=2）向前最多回看90个日历日，选择最近30个完整 DA3/RT4 + actual 日；不再要求最近30个日历日连续无缺口。该 readiness 是下一日可启动性预警，不改变当天已经通过的 NORMAL 交付。

### 10.6.1 Dynamic-v1 生产验收基线（2026-09-20）

标准入口：

```bash
python main.py --96 2026-09-20
```

已完成真实 DB-backed 验收：DB full sync 成功；LightGBM DA、TimesFM DA/RT、TimeMixer DA/RT、SGDFNet RT、RT916 RT 七个正式模型腿均真实执行并各产出96点；SGDFNet 决策日前锚为 `2026-09-19`、96点、`fallback_used=false`。下游 `ledger_weight`、`ledger_fuse`、`final_outputs` 全部完成，DA/RT weights 分别 9/12 行，融合各96点，`submission_ready.csv` 96点；最终 `delivery_status=NORMAL`、`exit_code=0`、postflight PASS、next-day readiness PASS、fallback=false。之后同 snapshot 的标准入口重跑会通过严格 cache contract 复用模型结果；cache identity 不一致时必须拒绝并重算。

生产放行标准因此固定为：`main.py --96 T` 正常退出0；DA3/RT4 每腿96点；同一 `snapshot_id` / `formal96_dynamic_snapshot_v1`；SGDFNet anchor contract PASS；weight/fuse/final 完整；postflight PASS；fallback=false；prediction artifact audit PASS。`ledger_predict=complete_with_warnings` 仅允许 live target actual 尚未闭合这一类非模型完整性警告，且 `--finish` 只有在其余严格 provenance 全部通过时才允许复用。

注意这只证明 Dynamic-v1 snapshot/FeatureView、96点完整性、模型池、账本和运行协议正确，
不证明 latest-state 历史 ForecastData 就是当时实际预测时点保存的原始 publication vintage。旧 fixed-hour cutoff 只作兼容审计事实，不是当前 formal serving 规则。当前 `epf_pmos_96_full` 是 latest-state upsert 表；
2026-08-15..2026-09-15 的本机 vintage audit 为 **0/32 strict、32/32
`UNVERIFIED_LEGACY_VINTAGE`**。证据文件：
`outputs/96/sync/forecast_vintage_audit_20260815_20260915.json`。
如果某次评估必须宣称“严格历史 forecast vintage”，使用：

```bash
python scripts/server/audit_96_artifacts.py \
  --output-root outputs/96 --phase prediction \
  --start YYYY-MM-DD --end YYYY-MM-DD --resource-mode split_process \
  --require-strict-forecast-vintage
```

没有当日成功 canonical LIVE Snapshot 或独立 publication-vintage 证据时，该门禁应当 FAIL。当前旧历史 proxy/latest-state 区间只用于生产机械链路、warm history、调度/恢复验收，不应作为“严格无 forecast 修订泄漏”的最终模型效果证据。未来真实 LIVE 日通过持久 canonical Snapshot 逐日积累严格 replay 证据。

### 10.7 生产部署与可恢复状态

服务器部署分成不可变应用资产和可恢复运行状态，两者禁止混在 `outputs/experiments/`：

- **应用资产**：Git 仓库代码、Python/CUDA 环境、配置、外部 secret、TimesFM 等固定模型权重；
- **数据**：正常情况下不随部署包传输，由数据库同步后 bootstrap/增量刷新唯一 `model_input_full.parquet`；
- **可恢复状态**：`outputs/96/ledger/` 是权重学习器的核心长期状态。当前 formal96 预测 T 时只用到 T-2 完整真值；完整 `--96 T` 会先从 authoritative actual source 幂等结算 T-2，再做30日 learner readiness。ledger 可以随换服务器迁移避免重新暖机；若历史不是当前 formal96 直接生成，禁止裸复制，必须用 `scripts/server/bootstrap_96_production_ledger.py` 做有界窗口审计、staging/readiness、原子 promote，并保留 `bootstrap_manifest.json`。`outputs/96/cache/classifier/` 仅为 classifier legacy/shadow 可恢复状态，formal96 当前不依赖它完成交付；
- **业务历史**：`outputs/96/runs/<date>/final/` 与 `run_manifest.json`/delivery report 长期保留；
- **禁止作为部署依赖**：`outputs/experiments/`、旧 `ledger_96/runs_96`、`unified_runs`、`RT916_SpikeMarketLab`、diagnostics、历史 replay、任务级 as-of/scratch。历史实验 ledger 只能作为 warm-start 的来源候选，必须通过迁移审计后才能进入正式 `outputs/96/ledger/`。

新服务器 warm-start 推荐先 dry-run 再 apply：

```powershell
python scripts/server/bootstrap_96_production_ledger.py `
  --source-ledger "<old-server-ledger>" `
  --target-date YYYY-MM-DD `
  --days 30

python scripts/server/bootstrap_96_production_ledger.py `
  --source-ledger "<old-server-ledger>" `
  --target-date YYYY-MM-DD `
  --days 30 `
  --apply
```

迁移脚本与正式 learner 使用**同一个 adaptive selector**：从 T-2 向前最多回看90个日历日，选择最近30个 DA3/RT4 + actual96 完整日；不要求最近30个日历日连续完整。可选的 T-1 prediction 只有在整池96槽且 cutoff 合法时才附带，缺失不会阻断30日 warm-start。脚本拒绝晚于当前 serving boundary 的历史 cutoff，源 cutoff/provenance 原样保留。apply 先在 `outputs/96/runtime` staging 用正式 `history_lag_days=2` selector 验证 learner readiness，再原子 promote；成功后 `outputs/96/ledger/bootstrap_manifest.json` 记录来源和 SHA256。随后直接运行 `python main.py --96 YYYY-MM-DD`。

长期运行 retention：runtime attempt 成功即整目录清理，普通日志目标保留 30 天、失败诊断目标保留 90 天；成功 run 的 prediction/weight/fuse 等大中间产物目标保留 30 天。正式 root manifest 会持久保存 `decision_snapshot`（最终权重 + model-quality gate），因此中间 CSV 过期后仍能解释最终融合决策；maintenance 只有在 DA/RT `decision_snapshot` 都完整时才把这些中间件列入候选，否则标记 `blocked_missing_decision_snapshot`。对同一天重复回测只保留一个 `stale_delivery_previous` 槽并覆盖，不再无限创建 `stale_delivery_<attempt>`。自动 destructive retention 在服务器验收通过前保持关闭。

Crawler 与 predictor 输出严格分域：源码 crawler 运行时写 `outputs/crawl/runtime_96/`；部署 EXE 写 `<exe目录>/output_96/`。两者都不是 formal96 predictor 的输入目录，正式预测只读取 `data/96/...` 与 `outputs/96/...` 状态。项目根 `output_96/` 已于 2026-09-19 归档并禁止重生。

#### 10.7.1 最小 predictor release 与部署 doctor

甲方 predictor 不再通过“整个开发仓打包”交付。使用白名单 release builder：

```powershell
# 只看计划
python scripts/server/build_predictor_release.py --skip-hash

# 计算完整 release hash，但不写包
python scripts/server/build_predictor_release.py

# 物化到一个全新空目录
python scripts/server/build_predictor_release.py --apply --output-dir <NEW_PREDICTOR_DIR>
```

当前 `formal96_predictor_release_v1` 为187文件、约0.866GiB；包含应用代码与 LightGBM/TimesFM 静态模型，排除 `outputs/`、`data/`、crawler、experiments、tests、build、Agent tooling、ExtremePriceClf 与 secrets。`release_manifest.json` 保存逐文件 size/SHA256，builder 拒绝覆盖非空旧 release。

新机先迁 state，再 doctor：

```powershell
python scripts/server/bootstrap_96_production_ledger.py `
  --source-ledger <OLD_LEDGER> `
  --target-ledger outputs/96/ledger `
  --target-date YYYY-MM-DD --days 30 --apply

python scripts/server/doctor_96_deployment.py `
  --root <PREDICTOR_ROOT> --strict-release `
  --require-cuda --check-db --check-writable `
  --target-date YYYY-MM-DD
```

strict doctor 会验证 release manifest、187个文件 hash、生产 import、静态模型、TimesFM 模型路径是否解析到 candidate 自身、CUDA、DB、runtime write/delete 与 adaptive ledger readiness。2026-09-20 已在全新系统临时目录做过真实 clean-deployment acceptance：candidate 自身 DB full sync 后运行 `python main.py --96 2026-09-20`，七模型全部真实执行，最终 `NORMAL / postflight PASS / fallback=false`，artifact audit 与 `--check-final` 均 PASS。因此部署链已经由“开发机可跑”升级为“最小 release + 可迁 state + 外部 DB 可独立运行”。

验收前只运行非破坏性计划器：

```bash
python scripts/server/maintenance_96.py \
  --report outputs/96/sync/retention_plan_preacceptance.json
```

它只列出超期候选，不删除文件；`--apply` 当前硬拒绝。`ledger/`、`cache/classifier/`、
每日日终 `final/` 与 `run_manifest.json` 永久排除在该候选计划之外，`experiments/` 继续人工科研归档。

### Full-source historical ledger audit and three-way replay (2026-09-20)

Use the explicit read-only audit before any migration:

```powershell
python scripts/server/bootstrap_96_production_ledger.py `
  --source-ledger outputs/archive/server_backtest_96/original_server_prediction_20251218_20260814/ledger `
  --target-ledger outputs/96/ledger `
  --target-date 2026-08-16 `
  --history-scope full-source `
  --runtime-root <OS-temp-root>
```

The audit proves the 240-day source, overlap/new-day accounting, staging readiness and current-wins rule. The verified source is `2025-12-18..2026-08-14` (240 complete days); against the current ledger it identifies 210 missing days and 30 overlap days. Production import is only allowed after the dry-run passes, the current ledger is backed up/hashed, and the operator explicitly adds `--apply`; current production always wins overlap keys. Historical `--96 DATE` calls resolve stored LIVE snapshots first, then the p56 historical proxy, while the current target remains LIVE_DYNAMIC; `--finish` reuses Stage1 provenance and never re-resolves.

### 10.8 Historical Proxy 实机验收与服务器接续

`python main.py --96 2026-08-17` 已完成真实七模型单日验收：route=`HISTORICAL_PROXY_V1`；D=2026-08-16 的 actual/RT 仅 p1..p56 可见，RealityTmp 全空，DA 96/96；FeatureView 将 actual 尾部40格路由到 forecast、RT尾部40格路由到 same-day DA，remaining NaN=0；DA3/RT4 全部96点；SGDFNet anchor=D DA96/fallback=false；RT916 stride=24；learner 使用30个完整日且最大训练日期严格为 T-2=2026-08-15；weight/fuse/final/postflight/artifact audit 全部 PASS，delivery=NORMAL。

该机本次耗时：DB full sync 约277秒（4分37秒），正式四阶段约617秒（10分17秒），端到端约896秒（14分56秒）。主要瓶颈是 GPU 串行的 TimeMixer DA、TimeMixer RT、RT916；weight/fuse/final 仅约1秒量级。服务器批量历史接续按约15分钟/天做保守容量规划，实际以服务器 GPU/DB 网络实测为准。

服务器部署、full-source ledger 合并、从 2026-08-17 向最近闭合日逐日接续、验收与切换到每日生产的完整步骤由 `SERVER_96_DEPLOYMENT_BACKFILL.md` 负责；本 RUNBOOK 只保留通用命令和契约。
