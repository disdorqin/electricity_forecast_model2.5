---
status: active
date: 2026-09-21
owner: formal96 server standard operating procedure
audience: Codex / server operator
canonical_entry:
  single_day: python main.py --96 YYYY-MM-DD
  range: python main.py --96 --start START --end END --require-target-actual --skip-existing-final
---

# Formal96 服务器标准流程 SOP

## 0. 用途

本文件是服务器 Codex 的标准执行顺序。目标不是重新设计项目，而是把已经本地验收过的 formal96 生产链原样部署到服务器，恢复 production ledger，从 2026-08-17 接续到最近闭合日，最后进入每日正式预测。

执行顺序固定：

    1. 拉代码 / predictor release
    2. 建 Python 3.11 环境
    3. 安装项目依赖
    4. 特别核验 TimesFM 本地库 + 本地 checkpoint
    5. 核验 CUDA / DB / release
    6. 恢复 current production ledger
    7. old-server 240 天 full-source dry-run
    8. full-source apply
    9. DB sync，读取 latest_closed_day
   10. 一条 formal96 range 命令从 2026-08-17 跑到 latest_closed_day
   11. 全区间 audit
   12. 转入每日 python main.py --96 TARGET_DATE

任何一步失败都先修该步，不允许通过改模型、缩短 learner history、允许 fallback 等方式绕过。

## 0.1 当前已验证服务器基线（2026-09-21）

本项目已经在真实服务器完成一次从环境到历史接续的完整闭环，后续部署应优先复用这些事实，不要重新探索：

    Python = 3.11.14
    Torch  = 2.6.0+cu124
    GPU    = RTX 4090 48GB（本次验收机器；其他 CUDA GPU 允许，但必须 doctor PASS）
    TimesFM import = 仓库内 TimesFMBackend/src/timesfm
    TimesFM checkpoint = models/timesFM/model.safetensors（约 0.9GB，Git 外部资产）
    formal profile = split_process / CPU2 / GPU1
    learner = 30 complete days / lag2 / lookback90 / smape_reg-SLSQP

已完成：

    old-server 240-day full-source apply = PASS
    historical catch-up 2026-08-17..2026-09-19 = COMPLETE
    per-day artifact audit = 34/34 PASS

## 0.2 本次踩坑后冻结的“最快正确路径”

**不要一上来重装环境。** 新服务器第一步先盘点已有资产：

    which python / python3 / conda
    /root/miniconda3/bin/conda env list   # 若 shell PATH 未加载 conda，直接查常见安装路径
    nvidia-smi
    find <workspace> -maxdepth 3 -name model.safetensors
    find <workspace> -maxdepth 3 -path '*/outputs/96/ledger'

本次服务器一开始看起来“没有 python/conda”，实际只是登录 shell 没把 Miniconda 加 PATH，`epf96` 环境已经存在。以后必须**先查已有 env，再决定是否创建/安装**。

标准最短路径：

    A. 确认/创建 Python 3.11 env
    B. requirements 一次安装；不要 pip install timesfm
    C. 放好 TimesFM checkpoint
    D. secret 放 release 外部，source 到进程环境
    E. strict deployment doctor PASS
    F. 恢复/验证 outputs/96/ledger，确认 30-day learner readiness
    G. 只在确需补历史时 full-source + range
    H. 正常每日只执行 python main.py --96 TARGET_DATE

如果 A～F 已 PASS，**禁止为了“干净”重新建环境、重新下载模型、重新 bootstrap ledger**。生产部署目标是复用已验证状态，而不是每次从零开始。

---

# 1. 冷启动必须先读

Codex 开始前必须读取：

    AGENTS.md
    README.md
    docs/README.md
    docs/SERVER_96_STANDARD_SOP.md
    docs/SERVER_96_DEPLOYMENT_BACKFILL.md
    docs/RUNBOOK.md
    docs/DATA_CONTRACT_96.md
    docs/LEAKAGE_AUDIT_96.md
    docs/OUTPUT_CONVENTION.md
    docs/PROJECT_LAYOUT.md

禁止使用聊天记忆替代仓库事实。

---

# 2. 环境基线

## 2.1 Python

必须使用：

    Python 3.11.x

原因：

- deployment doctor 明确要求 3.11.x；
- TimesFMBackend/pyproject.toml 要求 >=3.11,<3.12；
- 当前本地正式验收基线是 Python 3.11.14。

先检查现有环境：

    which conda || true
    /root/miniconda3/bin/conda env list  # SeetaCloud 等镜像常见路径示例

若已有 `epf96` 且 Python/依赖 doctor 能通过，直接复用；只有不存在或确实损坏时才创建：

    conda create -n epf96 python=3.11 -y
    conda activate epf96

或用等价独立 venv。自动化脚本也可以直接使用环境内 Python 的绝对路径，避免非交互 shell 未加载 conda PATH。

不要直接污染系统 Python，也不要仅凭 `python3: command not found` 就判定服务器没有可用 Python 环境。

## 2.2 安装项目依赖

在项目根：

    python -m pip install --upgrade pip
    pip install -r requirements.txt

requirements.txt 已锁定 formal96 当前生产栈，包括：

    numpy 1.26.4
    pandas 2.2.2
    scipy 1.13.1
    scikit-learn 1.4.2
    lightgbm 4.6.0
    torch 2.6.0+cu124
    torchvision 0.21.0+cu124
    pyarrow 25.0.0
    pymysql 1.2.0
    huggingface_hub 0.36.2
    safetensors 0.7.0
    jax 0.4.30
    jaxlib 0.4.30
    einshape 1.0

正式 GPU 生产必须让 torch.cuda.is_available() = True。

---

# 3. TimesFM：服务器最容易配置错的地方

## 3.1 不要安装 pip timesfm

严禁：

    pip install timesfm

也不要做 editable install 指向其他旧项目目录。

原因：项目正式 TimesFM 不是依赖外部 pip 包，而是直接使用仓库自带：

    TimesFMBackend/src/timesfm/

TimesFMBackend/price_forecast_copy_分时段预测.py 会把：

    TimesFMBackend/src

强制插入 sys.path，并要求 import timesfm 命中这个本地实现。

外部 pip timesfm 历史上可能把 import 指到错误目录，或引入不兼容 JAX/Numpy，从而导致模型缺失/报错。

## 3.2 TimesFM 必须存在的本地代码

必须有：

    TimesFMBackend/
    TimesFMBackend/infer.py
    TimesFMBackend/src/timesfm/
    TimesFMBackend/src/timesfm/timesfm_2p5/

这部分代码应该来自 Git/release，本身不需要单独 pip install。

## 3.3 TimesFM Python 依赖

由根 requirements.txt 安装：

    huggingface_hub[cli]==0.36.2
    safetensors==0.7.0
    jax==0.4.30
    jaxlib==0.4.30
    einshape==1.0
    scikit-learn==1.4.2
    torch==2.6.0+cu124

TimesFMBackend/requirements.txt 只是同一套局部锁定的说明，不需要再重复安装两次。

## 3.4 TimesFM 本地 checkpoint

部署包必须包含：

    models/timesFM/model.safetensors
    models/timesFM/config.json

**GitHub 同步边界：** `config.json` 可以进入 Git；`model.safetensors` 当前约 925 MB，普通 GitHub 单文件限制不适合直接提交，因此主仓库不会携带该权重。服务器 clone/pull 完代码后，必须从已验证的本地生产资产或安全对象存储把这一份 checkpoint 传到部署根 `models/timesFM/model.safetensors`，再运行 deployment doctor。禁止为了省事让生产任务现场联网下载来掩盖缺失权重。

正式生产默认模型目录由代码解析为部署根自己的：

    <PREDICTOR_ROOT>/models/timesFM

generic PROJECT_ROOT 不得重定向 TimesFM。

只有明确需要外置 checkpoint 时才允许专用变量：

    TIMESFM_MODEL_DIR

正常部署不要设置它。

## 3.5 TimesFM 必做验证

环境安装后运行：

    python -c "from TimesFMBackend.price_forecast_copy_分时段预测 import _import_timesfm; m=_import_timesfm(); print(m.__file__)"

输出路径必须位于当前部署根：

    .../TimesFMBackend/src/timesfm/...

不能是 site-packages 里的其他 timesfm，也不能是其他 checkout。

再检查 checkpoint：

    python -c "from pathlib import Path; p=Path('models/timesFM'); print((p/'model.safetensors').is_file(), (p/'config.json').is_file())"

必须输出：

    True True

最终 deployment doctor 还会再次做 TimesFM model resolution 检查。

---

# 4. GPU / CUDA 核验

先：

    nvidia-smi

然后：

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

预期：

    torch ~= 2.6.0+cu124
    cuda_available = True

TimeMixer 和 RT916 正式生产依赖 GPU。

formal96 调度固定：

    CPU workers = 2
    GPU workers = 1

不要为了服务器更强就自行开多个 GPU model worker。

---

# 5. Release / DB / writable doctor

生产服务器优先部署 predictor release，而不是整个研究仓。

开发侧 release 构建：

    python scripts/server/build_predictor_release.py --apply --output-dir <NEW_PREDICTOR_DIR>

服务器 predictor 根至少必须包含：

    main.py
    requirements.txt
    cli/
    pipelines/
    runners/
    fusion/
    utils/
    optim/
    lightGBM/
    TimesFMBackend/
    TimeMixer/
    SGDFNet/
    RT916_SpikeFusionNet/
    models/LightGBM/best_model_日前电价.pkl
    models/timesFM/model.safetensors
    models/timesFM/config.json

数据库凭据使用服务器环境变量或服务器本地 `.env`，不提交 Git。**strict predictor release 内禁止放 `.env`**；正确做法是在 release 外保存 secret，运行前 `source`/注入到进程环境。本次真实部署已验证：把 `.env` 放进 release 会被 strict doctor 正确拒绝。

代码同步若遇到 GitHub HTTP/2/网络不稳定，不要卡数小时反复 fetch；可以切 HTTP/1.1、使用已验证 release 包或从可信开发机传输 release/source archive，但最终必须对 commit/release manifest 做一致性检查。

先运行：

    python scripts/env_check.py

再运行：

    python scripts/server/doctor_96_deployment.py \
      --root <PREDICTOR_ROOT> \
      --strict-release \
      --require-cuda \
      --check-db \
      --check-writable

要求：

    failures = 0

若此时还没恢复 ledger，可以先不做 target-date readiness gate。

**阶段门禁：** 环境阶段只有同时满足 Python 3.11、CUDA可用、TimesFM import 指向当前 release、本地 checkpoint 存在、DB check PASS、writable PASS，才允许进入 ledger/预测阶段。Doctor 已全绿时不要继续重装依赖。

---

# 6. Ledger：为什么一定要迁

formal96 不是“七模型预测完直接平均”。

正式流程：

    七模型 prediction
      -> prediction ledger
      -> actual ledger
      -> 30-day learner
      -> smape_reg / SLSQP
      -> period weights
      -> fuse
      -> final

学习器固定需要：

    required complete days = 30
    history lag = 2
    max lookback = 90

例如 target=T：

    learner 最多只能使用到 T-2
    在此前 90 个日历日中选择最近 30 个完整日

完整日要求包括：

    DA prediction: 3 models x 96
    RT prediction: 4 models x 96
    actual: 96

因此服务器如果只有代码和 DB、没有 production ledger，虽然模型可能能预测，但正式 learner 会因为没有 30 个历史完整日而 fail-fast。

这就是为什么 ledger 是生产状态，不是普通 cache。

canonical 路径：

    outputs/96/ledger/

不要使用旧：

    outputs/ledger_96
    outputs/runs_96

---

# 7. 恢复 current production ledger

先把当前开发/生产环境已经验收的：

    outputs/96/ledger/

恢复到服务器 predictor 的同一路径。

如果有成功 LIVE canonical Snapshot，也建议迁：

    outputs/96/runs/<DATE>/snapshot/
    outputs/96/runs/<DATE>/run_manifest.json

因为未来 replay 会优先复用真实 Snapshot。

---

# 8. 旧服务器 240 天历史并入

旧 source：

    2025-12-18 .. 2026-08-14
    240 complete days

本地已验证：

    DA3x96 240/240
    RT4x96 240/240
    actual96 240/240

GitHub production sync 会保留该 source 的 **parquet-only ledger seed**（不上传大 CSV/run/log）：

    outputs/archive/server_backtest_96/original_server_prediction_20251218_20260814/ledger

bootstrap 实际只读取 prediction/actual 的 parquet，因此这4个 parquet 足够。若服务器希望把 seed 放到仓库外，只需复制到例如：

    /srv/formal96_seed/original_server_20251218_20260814/ledger

## 8.1 dry-run

    python scripts/server/bootstrap_96_production_ledger.py \
      --source-ledger /srv/formal96_seed/original_server_20251218_20260814/ledger \
      --target-ledger outputs/96/ledger \
      --target-date 2026-09-20 \
      --days 30 \
      --history-scope full-source \
      --runtime-root outputs/96/runtime

本地当前状态对应的预期：

    source_days = 240
    imported candidate = 210
    overlap = 30
    current production wins
    readiness = PASS
    applied = false

服务器 current ledger 如果比本地更新，import/overlap 数量可以变化；规则不能变化。

## 8.2 apply

dry-run PASS 后先备份/hash current ledger，再执行：

    python scripts/server/bootstrap_96_production_ledger.py \
      --source-ledger /srv/formal96_seed/original_server_20251218_20260814/ledger \
      --target-ledger outputs/96/ledger \
      --target-date 2026-09-20 \
      --days 30 \
      --history-scope full-source \
      --runtime-root outputs/96/runtime \
      --apply

要求：

    status = COMPLETE
    applied = true
    final_readiness = PASS

并生成：

    outputs/96/ledger/bootstrap_manifest.json

禁止手工 concat / overwrite parquet。

---

# 9. 获取最近闭合日

先单独做一次 full DB sync：

    python main.py --pipeline sync_dataset \
      --resolution 15min \
      --sync-source db \
      --sync-mode full \
      --force-sync

读取：

    outputs/96/sync/sync_manifest.json

字段：

    latest_closed_day

定义：

    START = 2026-08-17
    END   = latest_closed_day

不要把当前尚未闭合的 LIVE target 放进 historical range。

---

# 10. 一条命令补历史

formal96 已原生支持 range，不需要再写循环脚本。

推荐：

    python main.py --96 \
      --start 2026-08-17 \
      --end <LATEST_CLOSED_DAY> \
      --require-target-actual \
      --skip-existing-final

等价短写：

    python main.py --96 2026-08-17 <LATEST_CLOSED_DAY> \
      --require-target-actual \
      --skip-existing-final

该命令会自动变成：

    ledger_full_range
    resolution = 15min
    output_profile = production
    resource_mode = split_process

range 在批次开始只做一次 formal DB full sync，然后按日期串行执行每一天的完整正式链：

    Snapshot route
      -> FeatureViewBuilder
      -> DA3 / RT4
      -> prediction / actual ledger
      -> 30-day learner
      -> SLSQP
      -> fuse
      -> final
      -> postflight

不是只跑模型 prediction。

## 10.1 三态 Snapshot 自动生效

对每一天自动：

    有 canonical LIVE Snapshot
      -> STORED_LIVE_SNAPSHOT_REPLAY

    没有 Snapshot 且是 closed historical day
      -> HISTORICAL_PROXY_V1
      -> D actual/RT only p1..p56

    当前 unclosed target
      -> LIVE_DYNAMIC

因此不要另写历史 cutoff 脚本。

## 10.2 resume

保留：

    --skip-existing-final

如果服务器中断，直接重新执行同一条 range 命令。

skip gate 会检查 final、96 slots、价格、run manifest、formal96 四阶段和 manifest errors；合法完成日才跳过。

正式 catch-up 不建议加入：

    --continue-on-error

某一天失败就停，先修该日再继续。

---

# 11. Range 完成后的全区间验收

历史闭合区间当前采用**逐日 audit**。原因是 `ledger_full_range` 生成 `range_summary.csv`，而 `audit_96_artifacts.py` 的多日 prediction 模式目前仍期待 `prediction_range_manifest.json`；这个 wrapper 接口不一致不能被误判为单日产物失败。

标准做法：对 START..END 每一天执行：

    python scripts/server/audit_96_artifacts.py \
      --output-root outputs/96 \
      --phase prediction \
      --start YYYY-MM-DD \
      --end YYYY-MM-DD \
      --resource-mode split_process \
      --require-target-actual

同时检查 range 目录中的 `range_summary.csv`。本次 `2026-08-17..2026-09-19` 已按此口径逐日审计 **34/34 PASS**。

每个日还应满足：

    exit = 0
    delivery_status = NORMAL
    postflight = PASS
    fallback = false
    DA3 each 96
    RT4 each 96
    SGDFNet anchor rows = 96
    SGDFNet fallback = false
    RT916 stride = 24
    DA weights = 9
    RT weights = 12
    DA fuse = 96
    RT fuse = 96
    final submission = 96
    no NaN

---

# 12. 时间预算

开发机 2026-08-17 实测：

    DB full sync       4m37s
    model/full chain  10m17s
    single-day total  14m56s

服务器 RTX 4090 48GB 上同一 8/17 实测：

    DB full sync       ~3m14s
    formal chain       ~6m39s
    single-day total   ~9m54s

range 的 DB sync 在批次开始只做一次，因此历史批跑后续日通常低于“单日完整 sync + full chain”的时间。本次 34 日区间已成功无人值守跑完。新服务器容量规划仍先保守按 10～15 min/day，上线第一天后再用真实 manifest 时间重算 ETA。

---

# 13. 补齐以后每日生产

历史 catch-up 结束后，不再跑 range。

每天业务预测只执行：

    python main.py --96 TARGET_DATE

正式 LIVE 每次都会重新 DB sync，再创建预测时点真实 Snapshot。

成功标准：

    NORMAL
    postflight PASS
    fallback=false
    next_day_readiness PASS

成功 LIVE canonical Snapshot 长期保留。

---

# 14. Codex 禁止事项

服务器 Codex 不得：

- pip install timesfm；
- 把 TimesFM import 指到其他 checkout/site-packages；
- 删除 models/timesFM/model.safetensors；
- 允许 TimesFM 正式运行时临时联网下载来掩盖缺失 checkpoint；
- 改模型数学结构或模型池；
- 改 learner 30/90/lag2；
- 改 SLSQP；
- 改 CPU2/GPU1；
- 改 SGDFNet D->T DA anchor；
- 改 RT916 stride24；
- 把固定14/15点重新变成 LIVE serving cutoff；
- 没有 ledger 就缩短 history requirement；
- 手工覆盖 production ledger parquet；
- 并行跑多个 historical target day；
- 遇到失败自动 degraded delivery；
- 使用 ledger_96/runs_96 作为 formal96 state。

---

# 15. Codex 阶段汇报模板

环境完成：

    PHASE=SERVER_ENV
    PYTHON=
    TORCH=
    CUDA=
    GPU=
    TIMESFM_IMPORT_PATH=
    TIMESFM_CHECKPOINT=
    RELEASE_DOCTOR=
    DB_CHECK=
    STATUS=

ledger 完成：

    PHASE=LEDGER_RESTORE
    CURRENT_LEDGER=
    SOURCE_240D=
    FULL_SOURCE_DRY_RUN=
    FULL_SOURCE_APPLY=
    FINAL_READINESS=
    STATUS=

range 完成：

    PHASE=HISTORICAL_CATCHUP
    START=2026-08-17
    END=
    COMPLETED_DAYS=
    SKIPPED_DAYS=
    FAILED_DAYS=
    AVG_SECONDS_PER_DAY=
    ARTIFACT_AUDIT=
    NEXT_DAY_READINESS=
    DAILY_PRODUCTION_READY=
    STATUS=

只有环境、ledger、range 三阶段均 PASS，才能宣布服务器正式转入每日生产。

对于已经完成本次 warm-start/catch-up 的现有服务器，后续不需要每天重复 ENV / bootstrap / range；每日标准动作只剩 `python main.py --96 TARGET_DATE` + 当日 postflight/artifact 检查。

---

# 16. 2026-09-20 部署复盘（历史经验）

本节记录本次冷启动中的可复用经验，不改变 formal96 的生产契约。

## 16.1 代码、模型、状态必须分开同步

- 只有在 GitHub 已经包含正确 release commit 时，直接 clone/pull 才适合作为代码同步方式；若远端仍是旧 commit，不能把“远端可访问”误判为“代码已同步”。
- `models/timesFM/model.safetensors`、LightGBM checkpoint、`.env`、formal96 ledger 和 old-server source 不由普通 Git checkout 提供，必须分别核验、传输和记录哈希。
- `.env` 只能放在服务器 release 外部或受控环境变量中，禁止进入 Git、release manifest 或 doctor 的 release 目录。

## 16.2 大文件与依赖安装

- TimesFM 大文件优先使用可断点续传的分片并发传输；每个分片完成后再合并，并与源 SHA256 比对。并发连接不宜无限增加，连接过多会触发 SSH reset，必须在 pip 安装前关闭上传连接。
- Python 3.11 环境先用可用的 conda/venv 建立，再按根 `requirements.txt` 安装；不要反复切换 uv、conda 和多个 pip mirror，避免重复下载 CUDA 大包。
- Torch/CUDA 大 wheel 若普通 pip resolver 长时间停滞，可直接下载项目锁定版本并用本地 wheel 安装，然后补齐同版本依赖；仍须最终执行 `pip check`/导入检查，不得借此改变锁定版本。
- `pip install timesfm` 永远不是 TimesFM 修复方案；必须验证 import 路径落在当前部署根的 `TimesFMBackend/src/timesfm/`。

## 16.3 Release、数据和门禁顺序

- full research checkout 会被 `doctor_96_deployment.py --strict-release` 判为非 release；服务器正式目录必须先由 `build_predictor_release.py --apply` 生成，再注入静态模型和可迁移 production state。
- 先核验 DB 凭据和 `sync_dataset` 条件，再判断 preflight；代码/模型齐全不等于 96 canonical input 已在服务器存在。
- ledger 恢复必须走 `bootstrap_96_production_ledger.py` 的 dry-run/apply，不能以 legacy `outputs/ledger_96` 或手工 concat 替代 `outputs/96/ledger`。
- 只有 Python/Torch/CUDA、TimesFM import/checkpoint、strict doctor、DB sync、ledger readiness 全部通过后，才允许执行 `python main.py --96 DATE`；禁止用 fallback 或缺失模型掩盖环境问题。

## 16.4 传输耗时记录

本次耗时异常的根因是单路大文件传输、重复取消依赖安装和过晚发现 release/DB 外部状态要求。后续冷启动应先做：远端 commit/ignored assets 清单 → release 构建 → `.env`/DB 检查 → ledger staging → 环境安装；每个大文件只传一次并保留可恢复路径。
