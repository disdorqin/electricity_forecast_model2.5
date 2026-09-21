---
status: active
date: 2026-09-21
owner: formal96 server deployment / historical catch-up / daily production
entrypoint: python main.py --96 TARGET_DATE
validated_baseline:
  python: 3.11.x
  torch: 2.6.0+cu124
  production_profile: split_process CPU=2 GPU=1
---

# 96点服务器部署、历史接续与每日生产 Runbook

> Codex/服务器操作员必须先执行 `docs/SERVER_96_STANDARD_SOP.md`。本文仅作为详细补充 Runbook，处理 release/state、异常、验收和恢复细节；2026-08-17..2026-09-19 首轮历史接续已完成并逐日 audit 34/34 PASS，后续新服务器按同一 SOP 复用，不再重新探索环境方案。

## 1. 文档职责

本文只负责一件事：在一台新服务器上完成 formal96 predictor 部署、环境配置、production ledger 恢复、从 2026-08-17 向最近闭合日顺序补跑，并切换到每天正式预测。

通用链路契约仍以 README.md、docs/RUNBOOK.md、docs/DATA_CONTRACT_96.md、docs/LEAKAGE_AUDIT_96.md、docs/OUTPUT_CONVENTION.md、docs/PROJECT_LAYOUT.md 为准。服务器 Codex 不得根据旧 Phase 文档重新设计模型。

## 2. 当前已验收生产基线

唯一正式单日入口：

    python main.py --96 YYYY-MM-DD

固定正式模型池：

    DA: lightgbm, timesfm, timemixer
    RT: timesfm, sgdfnet, timemixer, rt916

固定生产参数：

    resolution = 15min
    output_profile = production
    resource_mode = split_process
    CPU workers = 2
    GPU workers = 1
    RT916 stride = 24
    learner = smape_reg / SLSQP
    required history = 30 complete days
    history lag = 2
    max lookback = 90
    classifier = disabled_by_production_policy

Snapshot 三态：

    A. closed historical day + valid stored LIVE snapshot
       -> STORED_LIVE_SNAPSHOT_REPLAY

    B. closed historical day + no valid LIVE snapshot
       -> HISTORICAL_PROXY_V1
       -> D actual/RT only p1..p56
       -> tail uses normal FeatureView fallback

    C. current/unclosed target
       -> LIVE_DYNAMIC
       -> DB at forecast origin determines visibility

三条路之后全部复用同一个 FeatureViewBuilder 和七个模型腿。

2026-09-20 LIVE Dynamic 已 clean-deployment 验收；2026-08-17 Historical Proxy 已本地真实七模型验收。

## 3. 服务器需要准备的四类资产

### 3.1 Application

优先从开发机生成最小 predictor release：

    python scripts/server/build_predictor_release.py --skip-hash
    python scripts/server/build_predictor_release.py
    python scripts/server/build_predictor_release.py --apply --output-dir <NEW_PREDICTOR_DIR>

不要把整个研究仓、experiments、crawler、legacy outputs 直接压到生产服务器。

### 3.2 Static model assets

release 必须包含：

    models/LightGBM/best_model_日前电价.pkl
    models/timesFM/model.safetensors
    models/timesFM/config.json

TimesFM 必须解析到部署根自己的 models/timesFM，禁止回跳开发机路径或联网下载。

### 3.3 Secrets / DB access

数据库凭据通过服务器环境变量或服务器本地 .env 注入。不要把真实 secret 写入 release manifest、Git 或文档。formal96 full/predict 在每次运行前都要求 DB sync 成功；DB 失败直接 fail-closed，不运行模型。

### 3.4 Mutable production state

至少需要：

    outputs/96/ledger/

推荐同时保留/迁移：

    outputs/96/runs/<successful-live-days>/snapshot/
    outputs/96/runs/<successful-live-days>/run_manifest.json

这样未来历史 replay 可以直接复用真实 canonical LIVE Snapshot。

此外准备一份只读旧服务器历史源：

    outputs/archive/server_backtest_96/
      original_server_prediction_20251218_20260814/
        ledger/

该 source 已审计为 2025-12-18..2026-08-14 共240个完整日，DA3x96、RT4x96、actual96 均完整。

## 4. 新服务器环境配置

生产基线 Python 3.11.x。建议使用独立 conda/venv，不与系统 Python 混用。

安装：

    pip install -r requirements.txt

TimeMixer 和 RT916 正式生产需要 CUDA GPU。已验证软件基线为 torch 2.6.0+cu124。允许服务器使用不同型号 GPU，但必须通过 deployment doctor。

基础检查：

    python scripts/env_check.py

在 predictor 根运行：

    python scripts/server/doctor_96_deployment.py \
      --root <PREDICTOR_ROOT> \
      --strict-release \
      --require-cuda \
      --check-db \
      --check-writable

要求 failures=0。若此时 ledger 还没恢复，不要传依赖 ledger readiness 的 target-date；先完成下一节 state 恢复。

## 5. Production ledger 恢复与旧服务器 240 天并入

目标：current production state + old server 240-day source -> one canonical outputs/96/ledger。

规则：current production wins overlap；old source only fills missing history；source provenance remains unchanged；no raw parquet overwrite；staging -> readiness -> atomic promote。

### 5.1 先放入当前 production ledger

将当前已验收的 outputs/96/ledger 恢复到服务器 predictor 的同一路径。不要把 outputs/ledger_96 或 outputs/runs_96 当 production state。

### 5.2 准备 old-server source

将旧服务器只读 source ledger 放到独立路径，例如：

    /srv/formal96_seed/original_server_20251218_20260814/ledger

不要直接复制到 outputs/96/ledger。

### 5.3 full-source dry-run

先只审计：

    python scripts/server/bootstrap_96_production_ledger.py \
      --source-ledger /srv/formal96_seed/original_server_20251218_20260814/ledger \
      --target-ledger outputs/96/ledger \
      --target-date 2026-09-20 \
      --days 30 \
      --history-scope full-source \
      --runtime-root outputs/96/runtime

当前已验证的预期审计结果：

    source_days = 240
    source_range = 2025-12-18 .. 2026-08-14
    missing/import candidate = 210 days
    overlap = 30 days
    current production wins
    readiness = PASS
    applied = false

如果服务器上 current ledger 与本地状态更新过，import/overlap 数量可以变化；但 source 240日完整性、current-wins、readiness 必须 PASS。

### 5.4 apply

dry-run 无异常后，先记录当前 ledger hash/备份，再显式执行：

    python scripts/server/bootstrap_96_production_ledger.py \
      --source-ledger /srv/formal96_seed/original_server_20251218_20260814/ledger \
      --target-ledger outputs/96/ledger \
      --target-date 2026-09-20 \
      --days 30 \
      --history-scope full-source \
      --runtime-root outputs/96/runtime \
      --apply

成功必须得到 status=COMPLETE、applied=true、final_readiness=PASS，并生成 outputs/96/ledger/bootstrap_manifest.json。不要手工拼 parquet。

## 6. 历史接续：从 2026-08-17 跑到最近闭合日

### 6.1 接续起点

旧服务器正式历史到 2026-08-14。当前本地 state 已包含后续部分日期，并且 2026-08-17 Historical Proxy 已经做过真实单日验收。服务器接续任务仍把 START=2026-08-17 作为业务接续起点。

如果迁移过来的 production state 已经包含某些日期，Codex 必须先审计后 skip；不要因为区间从 8/17 开始就强制覆盖已有合法 prediction/Snapshot。

### 6.2 确定 END

先运行一次同步：

    python main.py --pipeline sync_dataset \
      --resolution 15min \
      --sync-source db \
      --sync-mode full \
      --force-sync

读取 outputs/96/sync/sync_manifest.json 的 latest_closed_day，并定义 END=latest_closed_day。历史补跑不得越过 END。

当前 2026-09-20 本地证据中 latest_closed_day=2026-09-19。服务器实际执行时必须重新读取，不要硬编码 9/19。

### 6.3 canonical catch-up range

formal96 façade 已原生支持日期区间，不需要另写历史循环脚本。推荐：

    python main.py --96 \
      --start 2026-08-17 \
      --end <LATEST_CLOSED_DAY> \
      --require-target-actual \
      --skip-existing-final

等价短写：

    python main.py --96 2026-08-17 <LATEST_CLOSED_DAY> \
      --require-target-actual \
      --skip-existing-final

该命令解析为 `ledger_full_range + resolution=15min + production + split_process`。范围入口在批次开始先做一次 formal DB sync（已有镜像默认最近重叠增量，冷启动自动 full），取得同一份 authoritative/model store 与 `latest_closed_day`，之后按日期顺序调用同一个 `ledger_full`；每一天仍独立执行三态 Snapshot → FeatureView → 七模型 → ledger → learner → fuse → final。它不是新的预测协议。

因此 END 必须是批次开始时已经闭合的 `latest_closed_day`，不要把尚未闭合的 LIVE target 放进一个可能运行数小时的 historical range；当天正式 LIVE 预测仍单独使用 `python main.py --96 TARGET_DATE`，以获得预测时点的新鲜 DB Snapshot。

不要把 `scripts/server/run_96_prediction_backtest.py` 当作生产接续主入口。它是 prediction-only / advanced backtest runner，内部调用 ledger_predict，不负责完整 weight/fuse/final。

### 6.4 顺序执行与停止策略

`ledger_full_range` 内部已经按 START..END 严格串行，不并行写多个 target day。默认某一天失败就停止整个区间；服务器正式接续不要加 `--continue-on-error`。

`--skip-existing-final` 用于 resume：只有现有 final 结构、96 slots、价格、run manifest、formal96 四阶段和 manifest error 检查都通过时才跳过该日。区间完成后仍必须再跑全区间 artifact audit。

### 6.5 resume

中断后直接重跑同一条 range 命令并保留 `--skip-existing-final`，无需自己维护日期循环。range 内部的 skip gate 会先验证已有 final/manifest；最终人工/机械验收仍按以下更严格条件确认：

    delivery_report.json: delivery_status=NORMAL, exit_code=0
    run_manifest.json: status=complete, postflight.status=PASS, fallback=false
    DA lightgbm/timesfm/timemixer each 96
    RT timesfm/sgdfnet/timemixer/rt916 each 96
    artifact audit = PASS

否则从该日恢复。

### 6.6 每日历史验收

    python scripts/server/audit_96_artifacts.py \
      --output-root outputs/96 \
      --phase prediction \
      --start YYYY-MM-DD \
      --end YYYY-MM-DD \
      --resource-mode split_process \
      --require-target-actual

历史闭合日要求 target actual 96/96，因此 catch-up 使用 --require-target-actual。

同时检查 outputs/96/runs/YYYY-MM-DD/delivery_report.json、run_manifest.json、final/submission_ready.csv。要求 NORMAL、postflight PASS、fallback=false、submission 96行、无NaN。

### 6.7 Historical Proxy 日期额外检查

若 route=HISTORICAL_PROXY_V1：

    snapshot_kind = historical_proxy
    proxy_cutoff_period = 56
    historical_vintage = UNVERIFIED_LEGACY_VINTAGE
    strict_historical_vintage_proven = false

Snapshot D 日要求 DA non-null=96、actual final non-null=56、RealityTmp non-null=0、RT non-null=56。FeatureView 要求 target_truth_mask=true、remaining_nan=0。

SGDFNet 要求 anchor_source_day=D、anchor_source_type=decision_day_da、anchor_rows=96、fallback_used=false。RT916 要求 production_rt916_train_steps=24。

这些 proxy 日用于 operational history/learner，不得宣称 strict historical publication-vintage accuracy。

## 7. 时间规划

2026-08-17 本地真实运行实测：

    DB full sync       ~277 s  = 4m37s
    formal four-stage  ~617 s  = 10m17s
    total              ~896 s  = 14m56s

瓶颈主要是 TimeMixer DA、TimeMixer RT、RT916 RT；weight/fuse/final 约1秒量级。

容量规划先按约15 min / historical day。若从 2026-08-17 到 2026-09-19 共34天全部重算，约8.5小时。实际服务器 GPU 与 DB 网络会改变耗时。Codex 完成第一天后必须读取实际 wall time 并重算剩余 ETA。

## 8. 补齐后切换到每日正式生产

历史 catch-up 到 END 完成后：

1. 运行区间/日级 artifact audit，确认没有失败日；
2. 确认 learner 对下一目标可以选择最近30个完整日；
3. 确认 outputs/96/runtime 没有当前成功 invocation 残留；
4. 停止 historical batch wrapper；
5. 每个业务预测日只运行 python main.py --96 TARGET_DATE。

正式 LIVE 流程：DB sync -> LIVE_DYNAMIC Snapshot -> FeatureView -> seven model legs -> ledger -> 30-day learner -> SLSQP -> fuse -> final。成功 LIVE canonical Snapshot 长期保留，用于未来严格 replay。

## 9. 每日生产成功标准

每个正式日必须同时满足：

    process exit = 0
    delivery_status = NORMAL
    postflight = PASS
    fallback = false
    DA3 each 96
    RT4 each 96
    one consistent snapshot_id
    FeatureView target_truth_mask = true
    SGDFNet anchor rows = 96
    SGDFNet fallback = false
    RT916 stride = 24
    DA weight rows = 9
    RT weight rows = 12
    DA fuse = 96
    RT fuse = 96
    submission_ready = 96
    no NaN
    next_day_readiness = PASS

机械审计：

    python scripts/server/audit_96_artifacts.py \
      --output-root outputs/96 \
      --phase prediction \
      --start TARGET_DATE \
      --end TARGET_DATE \
      --resource-mode split_process

LIVE target actual 允许 partial；不要给正常 LIVE audit 强加 --require-target-actual。

## 10. 失败处理

DB sync 失败：不运行模型、不使用 stale DB，修 DB 后重跑同一天。

TARGET_FORECAST_NOT_READY：说明目标日预测型电网信息尚不满足 contract；不要用 D-1 forecast 冒充，等数据库源准备好后重跑。

单模型失败或少于96：当日不算完成，不继续下一天；检查 runs/<D>/logs/pipeline.log，修复后重跑 D。

learner readiness 失败：检查 outputs/96/ledger、bootstrap_manifest.json、prediction/actual complete days；不要通过缩短 required_days 绕过。

GPU OOM/CUDA：不要改变 CPU/GPU DAG 或同时跑多个 GPU 模型；先查 nvidia-smi、Torch/CUDA 和其他 GPU 进程。formal96 当前只支持一个 GPU worker。

## 11. Codex 服务器执行纪律

Codex 开始服务器任务时必须先读：

    AGENTS.md
    docs/README.md
    docs/RUNBOOK.md
    docs/SERVER_96_DEPLOYMENT_BACKFILL.md
    docs/DATA_CONTRACT_96.md
    docs/LEAKAGE_AUDIT_96.md
    docs/OUTPUT_CONVENTION.md
    docs/PROJECT_LAYOUT.md

禁止使用 chat memory 替代仓库事实。

Codex 可以：安装/核验环境、构建/部署 predictor release、配置服务器本地 secrets、跑 deployment doctor、恢复 ledger、做 full-source dry-run/apply、从8/17顺序 resume 到 latest_closed_day、做每日 artifact audit、最后切每日正式运行。

Codex 不得：改模型数学结构、改 DA3/RT4 模型池、改 learner 30/90/lag2、改 SLSQP、改 CPU2/GPU1、改 RT916 stride24、改 SGDFNet anchor、把 fixed14/15 重新变成 LIVE cutoff、用 target truth 补 FeatureView、并行写多个 historical target day、裸覆盖 production ledger、用 legacy ledger_96/runs_96 代替 outputs/96、遇到错误自动降级交付。

## 12. Codex 最终交付格式

服务器部署阶段完成时至少汇报：

    PHASE=
    STATUS=
    SERVER_ENV=
    RELEASE_DOCTOR=
    DB_SYNC=
    LEDGER_BOOTSTRAP=
    FULL_SOURCE_IMPORT=
    CATCHUP_START=
    CATCHUP_END=
    COMPLETED_DAYS=
    SKIPPED_DAYS=
    FAILED_DAYS=
    AVG_SECONDS_PER_DAY=
    ESTIMATED_REMAINING_HOURS=
    LAST_SUCCESSFUL_DAY=
    NEXT_DAY_READINESS=
    DAILY_PRODUCTION_READY=
    OPEN_ISSUES=

只有 FAILED_DAYS=0 且 DAILY_PRODUCTION_READY=true 才允许宣布服务器接续完成。
