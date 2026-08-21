# 特征预计算 / Feature Store + Warm-start 增量训练 — 调研报告（EFM3 落地版）

> 项目背景：山东省电力现货价格预测（24点正式交付 + 96点辅助），7 异构模型（lightgbm / timesfm / timemixer / sgdfnet / rt916）+ Ledger 融合。痛点：每个模型各自从原始宽表 `data/shandong_pmos_96_full_v2.xlsx`（35列）读数据、各自构建特征，214 天滚动回测每天重训 → 特征工程重复 + 训练墙钟长。
> 关联文档：`docs/工业界时序预测训练加速与精度提升调研报告.md`（加速三件套）、`docs/EFM3_链路稳健性_训练加速_融合改进_实施计划.md`（Phase 2 #2/#3）。
> 调研日期：2026-08-15。方法：webfetch 实证抓取（Feast / Tecton-Databricks / Chronon / SageMaker / LightGBM / PyTorch / TimesFM 官方文档）+ 领域知识。凡网络实证抓取的引用带链接；经验性建议标注「推断/经验」，落地前须 A/B 实测。

---

## 0. TL;DR

1. **"一次性算好特征、回测按日期切片"是工业界的标准形态**，对应 feature store 的 **point-in-time join / time-travel** 语义（Feast、Tecton、Chronon、SageMaker 全部支持，本文实证了 Feast 与 Chronon 的官方文档）。本质是：**特征定义写一次，全量历史一次算完存 parquet，模型运行时只按 `asof` 时间戳切片**——不是"每天重算特征"，而是"每天读一行切片"。
2. **本项目不需要引入 Feast/Tecton 这类重型框架**（单机、单宽表、数据量 <1GB，跑在 CPU/GPU 本地）。正确做法是 **30 行内实现一个轻量 `FeatureStore` 服务**（特征函数注册表 + parquet/npy 缓存 + 数据版本指纹），保留其核心语义（asof 切片、离线物化、增量追加）即可。
3. **特征失效的唯一触发源是"源数据变化"或"特征定义代码变化"**，用**数据文件哈希 + 特征定义版本**做缓存键；用 `materialize_incremental` 语义只追加尾行，不整体重算。
4. **Warm-start 续训是滚动回测的最大加速项之一**：LightGBM 用 `init_model=上一日 booster` 续训（官方 `lightgbm.train` 支持）；PyTorch 用 `state_dict` 恢复（官方 checkpoint 教程）；TimesFM 官方推荐 **LoRA 微调 / 冻结主干**。续训只是"初始点=前一日权重"，训练窗口仍只到 target 日前一天（防泄漏铁律不变）。
5. **何时续训 vs 全量重训**：靠**分布漂移检测**（特征分布 PSI/KS、验证损失 drift、预测分布漂移）触发全量重训；平滑漂移走续训，regime 突变（节假日/规则变更/数据源口径变化）走全量。
6. **本项目的 feature store 插入点应在 `ledger_predict` 之上**（scheduler 之前一次性物化 → 各模型 adapter 只读切片），**不是塞进各模型内部**。DA/RT 用**两个物理命名空间**（resolution 决定 shift 常量），SGDFNet 分**两阶段物化**（DA 预测 → 作为 da_anchor 喂 RT），CPU/GPU 并行天然安全（同一进程 ThreadPool 只读共享 DataFrame）。

---

## 1. 特征预计算 / Feature Store 工业实践

### 1.1 时序回测中"一次性算好特征、按日期切片"的成熟做法

**核心概念：point-in-time join（时点正确连接）。**

Feast 官方文档（已实证）对 point-in-time join 的定义：特征值是"带时间戳的时序记录"，join 时**对每一行，扫描该行事件时间戳向前到 TTL 窗口内的特征值**，从而复现"过去某一时刻特征的真实状态"：

> "Feast is able to join features ... in a point-in-time correct way. This means Feast is able to reproduce the state of features at a specific point in the past. ... Feast will scan backward in time from the entity dataframe timestamp up to a maximum of the TTL time specified."
> — https://docs.feast.dev/getting-started/concepts/point-in-time-joins.md

Tecton（被 Databricks 收购，原帖已并入 Databricks blog，已实证）更直白地把这叫做 **time-travel**，并强调它与"训练-服务一致性"的关系：

> "They provide point-in-time correct views of the state of the world for each example used to train a model (a.k.a. 'time-travel')."
> — https://www.tecton.ai/blog/what-is-a-feature-store/

**对本项目含义**：滚动回测里，每一天 `D` 的每个样本的"事件时间" = 该时点的 `asof` 时间（24 点口径 = 日前截断时刻；96 点口径 = **14:00 / p56**）。特征值只允许取 `asof` 之前的行。把 214 天回测的所有 `(date, asof)` 切片列成 entity dataframe，**全历史特征一次算好、按 asof 切片 join**，天然防泄漏（skill §4.2 的 cutoff=14 铁律在这个架构里是"特征矩阵物化时统一施加"，而不是每个模型每次各施加一遍——后者恰恰是当前重复劳动的根源）。

**Chronon（Airbnb 开源）的表述**（已实证）——声明式定义 + 自动回填训练集 + 防泄漏：

> "Eliminate data leakage with guaranteed point-in-time correctness. Generate accurate training datasets using state-of-the-art aggregation algorithms."
> — https://chronon.ai/index.html

Chronon 的时间窗聚合写法（`Aggregation(input_column=..., operation=SUM/LAST_K, windows=[3天, 14天])`）本质上就是"滚动统计特征声明一次、批量回填全历史"——与我们想要的 `rolling(24)/rolling(168)` 一次算完是同一件事。

**SageMaker Feature Store**（已实证）离线层用 **parquet/S3**，在线层用 DynamoDB，正是"离线大表 + 在线低延迟 KV"的通用分层：
> "Online FS is Dynamo, offline parquet/S3." — featurestore.org 对照表（已实证）
> https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-getting-started.html

**Kaggle 时序竞赛的通行做法**（领域知识，竞赛社区共识）：
- **M5 forecasting**（2019，~5000 队）：绝大多数前排方案都是 **LightGBM/XGBoost + 大量 lag 特征 + 滚动统计（rolling mean/std over 7/28 天窗）+ 日历特征**，训练脚本**一次性对全历史生成特征表**（`df.groupby(series).shift()` / `rolling()` 整列计算），然后**滚动回测时按日期窗口切 DataFrame**——没有人在每次 fold 里重算 rolling。这是"离线预计算 + 切片"在最大规模时序竞赛中的自然形态。
- **GEFCom2014**：赢家阵营用"梯度提升 + 分位数回归平均(QRA)"，特征工程同样是一次性对训练期算好滞后/温度交互，不进回测循环重算。

**关键判据（推论，可直接落项目）**：**凡 `shift(N)`/`rolling(N)` 这类"纯历史算子"，其值在固定数据上对所有后续 target 日都是确定的**——不需要"每天重算"，只需要"每天切片"。只有两类东西需要每日处理：
1. **新增数据行**（新一天的实际值爬回来）→ **增量追加**，只算新尾行及其带动的滚动窗口（`materialize_incremental`）；
2. **"截至 cutoff 可见"的列**（96 点 RT 的 p56 遮蔽）→ 在**物化阶段**按 asof 语义把不可见行置 NaN，切片阶段无脑取用。

### 1.2 特征版本管理与缓存失效/重建

**Feast 的语义**（已实证 data ingestion 页）：offline store 存全量历史（供训练），online store 存最新值（供在线推理），两者之间靠 **`materialize_incremental`** 增量同步：

> "Ingesting from batch sources is only necessary to power real-time models. This is done through materialization. ... materialize_incremental fetches the latest values for all entities in the batch source and ingests these values into the online store."
> — https://docs.feast.dev/getting-started/concepts/data-ingestion.md

**对本项目的落地规则（推断 + 工程常识，建议采纳）**：

- **缓存键 = f(source_hash, feature_def_version, resolution)**。`source_hash` 用数据文件的 **mtime + 大小 + 内容哈希**（skill 里已有 B2"缓存无数据版本指纹"的 P2 修复项，正好接上）。`feature_def_version` 是特征函数注册表的 git 版本/哈希。**数据更新后哈希变 → 自动整体重建或增量追加**；特征代码改动 → 版本号变 → 重建。
- **增量 vs 全量**：爬虫每天只加 ~1 天数据 → 特征矩阵**增量追加尾行**（先算新行原值列，再滚动窗口只需重算受影响窗口的滑动均值），只在"数据回填/修正历史段"（skill §1 的 actual 修正事故场景）时**全量重建**。建议默认全量重建（数据 <1GB，重建也就几秒~几分钟），增量优化放到后期。
- **manifest 落地"失败要响亮"**（skill §4.4 原则 B）：每次物化写 `manifest.json`（数据指纹、特征版本、物化时间、行数、每列 NaN 数），feature store 切片时校验指纹一致，不一致即告警（复用现有 manifest + delivery_report 告警段）。

### 1.3 特征存储格式（parquet / npy / 内存）与读取性能

| 方案 | 优点 | 缺点 | 本项目结论 |
|---|---|---|---|
| **parquet** | 列存、高压缩、自带 schema、可只读部分列、工业标准（SageMaker/Spark/Feast 都默认） | 每次读要解压（pyarrow 快） | **主存储**，单文件覆盖全历史 |
| **npy/npz**（NumPy 二进制） | 加载即 mmap 快，无需解析 | 无列名/无时间索引语义、改 schema 易错 | 可作为"已定型特征矩阵"的内存缓存加速层 |
| **pickle/DataFrame 直接内存** | 零转换、与 pandas 无缝 | 不可跨进程/跨版本、易腐坏 | 会话内缓存，不落盘 |
| CSV/Excel | 人类可读 | 慢、大、无类型 | 只作原始源 |

**读取性能要点（工程常识）**：
- 数据量量级估算：96 点 × 4 年 ≈ 96×365×4 ≈ 14 万行 × 数十列 float32 ≈ **几十 MB**——**完全可以在内存中整体载入**，214 天回测每天切片是内存操作，不是 IO。
- 推荐路径：**parquet 落盘（权威、可复用、可审计）→ 首次物化后整体 `pd.read_parquet` 进内存缓存 → 回测期零磁盘 IO**。每模型进程/线程各自持有同一份只读 DataFrame（见 §2.3 线程安全）。
- pyarrow 引擎读 parquet 快于 openpyxl 读 xlsx 一个数量级以上；建议**把 `shandong_pmos_96_full_v2.xlsx` 首次导入即转 parquet** 作为 feature store 原始输入（顺带解决"每次 30MB xlsx 解析"的重复开销）。

### 1.4 工具/框架盘点

| 工具 | 类型 | 关键能力 | 是否适合本项目 |
|---|---|---|---|
| **Feast** | 开源（Linux Foundation）| point-in-time join、offline/online store、registry、materialize_incremental | 重量级：要建 registry 服务、配 offline store（BigQuery 等），单机场景杀鸡用牛刀 |
| **Tecton → Databricks FS** | 商业（被 Databricks 收购，2025-08）| 托管 transform + 存储 + serving + 监控 + registry 五件套（已实证定义） | 云托管，无本地部署场景 |
| **Chronon**（Airbnb）| 开源（Apache 2.0）| 声明式 DSL、时间窗聚合、自动 PIT backfill、防泄漏（已实证） | 概念最佳参考；但依赖 Spark/Flink 生态 |
| **Feathr**（LinkedIn/Microsoft）| 开源 | PIT join、跨平台 | 同上，偏 Spark |
| **SageMaker / Vertex AI Feature Store** | 云托管 | 离线 parquet + 在线 KV（已实证）| 云依赖 |
| **自研轻量 FeatureStore（本项目）** | 30~100 行 | 特征函数注册表 + parquet 缓存 + 指纹版本 + asof 切片 | **推荐** |

**结论（明确建议）**：本项目**不要装 Feast/Chronon**——它们为"多实体、流式事件、多数据源、在线低延迟 serving"设计，本项目是**单实体（山东市场）、单宽表、批处理、数据 <1GB**。直接自研 50 行以内的轻量服务，**只借语义、不借框架**，零新增依赖（不违反 requirements.txt 最小化）。Tecton/Feast 的文档价值在于给了我们正确的**词汇表和语义**（time-travel、asof、materialize_incremental、training-serving skew），照着实现即可。

---

## 2. 避免重复特征工程的具体模式

### 2.1 "特征函数 + 缓存层"架构设计

**工业标准形态（Feast/Tecton/Chronon 一致）：特征定义声明式注册 → 执行引擎批量算 → 缓存 → 消费方只读。**

```
特征定义（函数注册表，唯一事实源）
        │ 声明一次
        ▼
FeatureStore.build(resolution, source)
        │ 全历史一次性物化（含 asof 遮蔽、shift/rolling、日历、外生）
        ▼
parquet 缓存 + manifest（指纹/版本/NaN 统计）
        │ 只读
        ▼
模型 adapter 按 (model, task, target_date, asof) 切片取用
```

**谁触发、何时失效（规则表）**：

| 事件 | 触发 | 失效动作 |
|---|---|---|
| 爬虫写入新一天数据 | 指纹变化 | 增量追加 / 全量重建 |
| 历史段回填修正（actual 修正） | 指纹变化 | **全量重建**（必须，防止半截重算污染） |
| 特征定义代码改动（加 lag、改窗口） | 特征版本变化 | 全量重建新版本，旧版本保留可回滚 |
| 模型侧只改超参/结构 | 无变化 | **不重建**（特征与模型解耦，正是 feature store 的价值） |
| 96 点 p56 遮蔽规则变化 | cutoff 常量变化 | 全量重建（作为特征版本一部分） |

**模型 adapter 的改造方向（关键）**：当前每个模型 pipeline 内部（如 `lightGBM/train_da_fix.py` 的 `feature_engineering`、`SGDFNet/src/sgdfnet/data_contract.py` 的 `preprocess_dataframe`）各自调 `shift/rolling`。改造后 adapter 应改为：
```
X, cols = feature_store.slice(model="sgdfnet", task="realtime",
                              target_date=D, asof=p56_cutoff, resolution=96)
```
模型内部**不再出现任何 shift/rolling 调用**，只做"取切片 + 归一化/转换"。这同时消灭 skill §2 的 24/96 shift 混用隐患——**shift 常量只存在于特征注册表一处**。

### 2.2 rolling/lag 特征一次性 vs 每日重算的边界

**判定准则（给实现者）**：
- **一次性算**：`lag_N`、`rolling_mean/std/min/max`、同时段均值、差分、calendar（hour/weekday/month/节假日 flag）、外生 fcast 列、`da_anchor` 滞后——全部是"纯历史可确定性函数"，**全历史一次算完**。
- **每日要处理的只有两种**：
  1. **新尾行**（新一天爬回来的数据）→ 增量追加；
  2. **asof 遮蔽**（96 点 p56 之后的列置 NaN）→ 物化时统一做。
- **边界判断口诀**：特征值在 `t` 时刻是否**只依赖 ≤ asof(t) 的信息**？若是 → 离线物化；若否（依赖 target 日实际值/未来信息）→ **根本不该存在**（泄漏）。

**rolling 的增量技巧（可选优化，非必须）**：滚动均值可写成"前缀和/滑动窗口"，追加一天时只需 O(窗口) 更新而非 O(全量)。但对 <1GB 数据，**直接全量重建更简单可靠**，先不做增量复杂度，等真成为瓶颈再优化（这与实施计划 Phase 2 的"只缓存不改变特征值"零精度损失目标一致）。

### 2.3 多模型共享特征的线程安全/并发读取

**本项目实际情况（已实证代码）**：`runtime/resource_scheduler.py` 用 **`ThreadPoolExecutor`（同进程线程池）**，且注释明确写"GPU 模型共享主进程 CUDA 上下文，避免多进程各 init CUDA 崩溃"。也就是说：
- **所有模型跑在同一个进程的不同线程** → 共享内存 feature store 在"只读"语义下**天然线程安全**（无写竞争）。
- 特征矩阵**物化发生在 scheduler 之前**（单线程一次完成），之后全部线程只读切片 → **无锁安全**。
- 缓存写盘用 **tmp + rename 原子写**（skill Phase 1 P0 已有"CSV 原子写"项，feature store 沿用同一模式）。
- CPU 模型（lightgbm/sgdfnet/timesfm）与 GPU 模型（timemixer/rt916）并行：特征已物化进内存，**各自线程独立切片**，互不干扰。

**线程安全结论（明确）**：本项目不需要 multiprocessing 锁/共享内存协议。唯一要注意的是**不要在特征 DataFrame 上做 in-place 修改**（`df[col] = ...` 会污染共享对象），切片前 `.copy()` 或让归一化在模型内对副本做。

---

## 3. Warm-start 增量训练在时序滚动回测中的应用

### 3.1 相邻日权重续训的工业案例

- **本质**：滚动回测相邻两日 `D-1` 与 `D` 的训练数据只差 1 天，模型参数高度相似。以 `D-1` 权重为初始点、用少得多的 epoch 续训，等价于"把训练时间从 O(重训) 降为 O(微调)"。Lago et al. 2021 综述（本领域权威，已在另一份报告引用）的实证结论是**日更模型训练必须快（<30min）**，而 warm-start 正是达成该要求的工程手段之一。
- **LightGBM/XGBoost 的 continue training**：官方 `lightgbm.train(..., init_model=前一日 booster)`（见 §3.3）；XGBoost 是 `xgb.train(..., xgb_model=prev)`。GBM 的"续训"本质是**继续加树**，不是权重热启动——所以要**控制续训轮数 + 定期全量重置**（见 §3.2/§3.3 的树数增长问题）。
- **PyTorch**：官方 checkpoint 教程（已实证）明确支持"保存/恢复完整 checkpoint 以 resuming training"——`model_state_dict + optimizer_state_dict + epoch`，恢复后从断点继续。这正是相邻日续训的标准机制。
- **TimesFM**（本项目在用，官方 README 已实证）：Google 官方在 2026-04 加入 **HuggingFace Transformers + PEFT (LoRA) 微调示例**，且模型 2.5 版从 500M 减到 200M 参数（更便宜的快），README 强调 "freeze 主干 + 轻量头"。**回测场景建议冻结主干只训轻量头/LoRA adapter，或干脆零样本**（实施计划 Phase 2 #6 已列）。

### 3.2 何时续训 vs 何时全量重训（分布漂移检测）

**不是"永远续训"，而是"漂移小续训、漂移大重训"。判断信号（工程实践）**：

| 信号 | 检测方法 | 动作 |
|---|---|---|
| 验证损失 drift | 最近 5~10 日 val loss 相对基线均值显著上升（如 >2σ 或超阈值%） | 全量重训 |
| 特征分布漂移 | 特征分布 PSI（Population Stability Index）/ KS 检验（最近 30 日 vs 训练期）| 超阈值 → 全量重训 |
| 目标分布/regime 突变 | 节假日切换、规则变更、数据口径变化（skill §1 actual 修正）、季节拐点 | 全量重训 |
| 预测分布漂移 | 最近预测误差分布 vs 历史误差分布 | 告警 + 评估 |
| 平滑漂移（日常日更） | 无明显跳变 | **续训**（省时留预算收敛） |

**推荐本项目默认策略（结合 skill §4.4 设计原则）**：
1. **默认每日常态 = warm-start 续训**（相邻工作日数据高度相关，省 50%+ 墙钟）；
2. **触发全量重训的显式事件**：`(a)` 节假日/周末切换日（周一的训练集相对周五是"新增工作日分布"，建议周一全量，周二~周五续训）；`(b)` 数据回填/修正；`(c)` 漂移信号超阈值（先写一个轻量 PSI/val-loss 检测函数，跑 2 周看信号质量再启用自动触发）；`(d)` 每 N 天（如 7 或 14 天）强制全量一次兜底，防"树/权重长期漂移累积"。
3. **落地方案必须是双路径**：adapter 支持 `warm_start={True|False}`，调度层（ledger_predict / runner）按上述规则决定传什么——**先做开关，再补自动漂移检测**（分阶段，见 §4.4）。

### 3.3 Warm-start 的具体做法

**LightGBM（CPU-only，按 skill 勿开 GPU）——官方 API 已实证**：

`lightgbm.train` 的参数（https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html）：
> **init_model** (str/Path/Booster or None) – Filename of LightGBM model or Booster instance used for **continue training**.
> **keep_training_booster** – Whether the returned Booster will be used to keep training...

```python
import lightgbm as lgb

prev = lgb.Booster(model_file=f"ckpt/{target_prev}/model.txt")   # 前一日
# 续训：从 prev 继续加树；用 keep_training_booster=True 保留可续训的 booster 语义
bst = lgb.train(
    params,
    train_set,
    num_boost_round=200,          # 续训轮数远小于全量（全量如 1000+）
    valid_sets=[val_set],
    init_model=prev,              # ★ warm-start 入口
    keep_training_booster=True,   # 返回值可继续作为 init_model 再续训
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)
# 存盘：model.save_model(f"ckpt/{target}/model.txt")
```

**⚠️ 树数增长问题（重要坑）**：continue training 是**加树**，每日 +200 棵会无限膨胀 → 需要：`(a)` 续训轮数小（几十~几百）；`(b)` 定期（每 7~14 天或全量重训日）从零重建；`(c)` 也可以用 `learning_rate` 降低来压缩树数。建议**每期设 `max_iteration` 上限并配合早停**，在全量重训日清空重来。

**PyTorch（timemixer/sgdfnet/rt916，GPU 云）——官方教程已实证**：
https://pytorch.org/tutorials/beginner/saving_loading_models.html

```python
# 保存完整 checkpoint（官方推荐 resuming 用 dict）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),   # ★ 保留动量/Adam 状态
    'loss': loss,
    'lr_scheduler_state_dict': scheduler.state_dict(), # 本项目有 cosine/plateau
    'scaler_state_dict': scaler.state_dict(),           # ★ AMP GradScaler 状态（skill 说 AMP 已内置）
}, f"ckpt/{target}/model.tar")

# 恢复续训
checkpoint = torch.load(f"ckpt/{target_prev}/model.tar", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint.get('lr_scheduler_state_dict', {}))
scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
# ★★ 重建 DataLoader（新窗口），绝不续用旧迭代器
train_loader = build_loader(feature_store.slice(... target=D ...))
model.train()
# → 少量 epoch 续训即可
```

**要点（官方文档明确 + 工程常识）**：
- 恢复后**必须重建 DataLoader**（官方教程的 checkpoint 教程隐含：checkpoint 存 epoch 不存 DataLoader，因为数据窗口变了）；
- **LR 调度重启策略**：续训日从 `lr = base_lr` 重新 warmup（前一日 cosine 已衰减到接近 0，不重置会原地踏步）；
- AMP `scaler` 状态保存恢复（`torch.amp.GradScaler`），否则 BF16/FP16 缩放因子从零开始；
- 跨设备：保存 GPU 训练、恢复 CPU（本机 smoke）用 `torch.load(path, map_location='cpu')`（官方文档专门一节）。

**TimesFM（已实证官方 README）**：冻结主干，只训练 LoRA adapter 或轻量输出头；2.5 版有官方 finetuning 示例（HuggingFace Transformers + PEFT）。回测期不整体微调主干。

### 3.4 防泄漏边界（warm-start 的合规红线）

- **铁律不变（skill §4.2）**：续训只是"**初始点 = 前一日权重**"，**训练窗口仍只到 target 日前一天（cutoff=14 / p56 遮蔽）**。续训绝不等于"把 target 日数据塞进训练"。
- **具体检查点**：
  1. 前一日模型 `D-1` 训练时只见过 `≤ D-1` 的数据（这由滚动语义天然保证）；
  2. `D` 日续训的 **DataLoader/特征切片必须用 `asof ≤ cutoff(D)` 的行**，target 日 p56 之后列已遮蔽；
  3. **checkpoint 不携带"未来"状态**：optimizer 状态只反映 `≤ D-1` 的梯度，无泄漏；
  4. **SGDFNet 依赖 DA anchor 的特殊边界**：RT 模型用 `da_anchor`（日前价）做特征。DA 是"昨天发布的日前价格"，RT 训练时 `da_anchor` 必须取**该时点已可见的日前值**（见 §4.2 两阶段物化），不能把 target 日实际 RT 价当 anchor。
  5. 回归验证：沿用 `scripts/tests/check_preflight_health.py` 的防泄漏项（p56 截止），新增"warm-start 后训练窗终点 = target-1"断言。

---

## 4. 针对本项目的推荐架构

### 4.1 Feature store 插入点：在 ledger_predict 之上，而非模型内部

**现状**（已实证）：`ledger_predict.py` → `ResourceScheduler`（同进程线程池）→ 各模型 adapter（`runners/adapters/*_v1.py`）→ 每个模型内部各自 `read_excel + feature_engineering`。

**推荐插入位置**：**`ledger_predict` 的 task 构建之前**，加一个**物化步骤**：

```
ledger_predict.run_ledger_predict(D)
  │ 1. 物化（新增，一次）
  │    FeatureStore.ensure( resolution, source=data/shandong_pmos_96_full_v2.xlsx )
  │      → 特征矩阵 parquet（含 asof 遮蔽、shift/rolling/calendar）
  │      → manifest.json（指纹+版本+NaN 统计）
  │ 2. 各模型（scheduler 线程池）
  │    adapter 改为：feature_store.slice(model, task, D, asof)  ← 只读内存切片
  │      ↓（若 warm_start）
  │    runner: 加载 ckpt/target_prev → 续训 → 存 ckpt/target
```

**为什么放这里（论证）**：
- 在 scheduler 之前物化 → 一次算、全部模型共享，且**在并发启动前完成，规避线程写竞争**；
- 各模型 adapter 的 `feature_engineering` 删除/降级为"归一化+转换"，**口径只在一处**（消灭 24/96 shift 混淆，skill §2 红线）；
- `ledger_full_range`（214 天循环调 ledger_predict）天然复用：物化带指纹，回测跑第 2 遍时**零重算**。
- 与 `ledger_predict` 的**缓存层（per-model predictions CSV）**正交：那个缓存的是"预测结果"，feature store 缓存的是"特征矩阵"，两层叠加正好对应 skill Phase 1 的缓存加固与 Phase 2 的加速。

### 4.2 DA/RT 口径分离、SGDFNet 依赖、CPU/GPU 并行的处理

**（a）DA/RT 口径分离 —— 两个物理命名空间**：
- `feature_store/{resolution}/da/matrix.parquet` 与 `feature_store/{resolution}/rt/matrix.parquet`（`resolution ∈ {24, 96}`）。
- shift 常量**硬编码在特征注册表**：`lag_1day = resolution`（24→shift(24)，96→shift(96)），`lag_7day = 7*resolution`。**模型侧无任何 shift**，从源头杜绝 24/96 混用（skill §2）。
- 96 点 RT 矩阵物化时**统一施加 p56 asof 遮蔽**；24 点 DA 矩阵物化时按日前截断时刻遮蔽。

**（b）SGDFNet 的 DA-anchor 依赖 —— 两阶段物化**：
SGDFNet RT 预测依赖 `da_anchor`（日前价格列，已实证 `SGDFNet/src/sgdfnet/data_contract.py:232` 与 `_fill_da_anchor_fallback`）。处理顺序：
1. **先物化 DA 矩阵并跑完 DA 腿**（DA 矩阵里的 `da_anchor` = 源表 da 价列 / 或 DA 模型预测的日前价）；
2. **再物化 RT 矩阵**，其中 `da_anchor` 特征引用第 1 步的产物（这就是"先训 DA 再训 RT"在 feature store 层面的实现）；
3. RT 矩阵的 `da_anchor` 滞后特征（`da_lag_24/da_lag_168`，data_contract.py:403-404）在物化时对 RT asof 做遮蔽。
> 实现提示：若 da_anchor 直接用源表 DA 价列，则两个矩阵可并行物化；若用 DA 模型预测值，则 RT 矩阵物化需排在 DA 预测之后。**先按源表列实现（零行为变化、零精度损失），DA 预测值版本作为后续增强**（A/B 对比）。

**（c）CPU/GPU 并行 —— 零成本安全**：
- 特征矩阵在 scheduler 前物化进内存 → CPU 线程（lightgbm/sgdfnet/timesfm）与 GPU 线程（timemixer/rt916）**各自只读切片**，不竞争、不加锁；
- 特征 DataFrame **只读共享**，模型内归一化对 `.copy()` 做（§2.3）；
- parquet 写盘用 tmp+rename 原子写。

### 4.3 建议的代码骨架（落地即用）

```python
# feature_store.py  （新增，约 60 行核心，无新依赖）
import hashlib, json
from pathlib import Path
from datetime import datetime
import pandas as pd

FEATURE_ROOT = Path("outputs/feature_store")

def _feature_def_version(resolution: int) -> str:
    # 特征注册表代码哈希；改特征就变，模型改动不变
    return hashlib.md5(str(sorted(FEATURE_REGISTRY[resolution].keys())).encode()).hexdigest()[:8]

def _source_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}-{st.st_size}-{hashlib.md5(path.read_bytes()[:1<<20]).hexdigest()[:8]}"

class FeatureStore:
    def __init__(self, resolution: int, source: Path):
        self.res = resolution
        self.source = source
        self.version = f"res{resolution}_v{_feature_def_version(resolution)}_{_source_fingerprint(source)}"
        self.dir = FEATURE_ROOT / self.version
        self.da_path = self.dir / "da_matrix.parquet"
        self.rt_path = self.dir / "rt_matrix.parquet"

    def ensure(self) -> "FeatureStore":
        if not (self.da_path.exists() and self.rt_path.exists()):
            self._build()          # 一次性物化（含 p56 遮蔽、shift/rolling）
            self._write_manifest()
        self.da = pd.read_parquet(self.da_path)   # 回测期只读，进内存一次
        self.rt = pd.read_parquet(self.rt_path)
        return self

    def slice(self, model: str, task: str, target_date: str, asof=None) -> pd.DataFrame:
        df = self.rt if task == "realtime" else self.da
        sel = df[df["business_day"] <= target_date]      # 训练窗：<= target 前一天由外部切
        if asof is not None:                              # 96 点 RT：p56 遮蔽已物化
            sel = sel[sel["ds"] <= asof]
        return sel[FEATURE_REGISTRY[self.res][model]]     # 按模型列子集 + 副本
```

```python
# FEATURE_REGISTRY：唯一事实源（shift 常量只在这里）
FEATURE_REGISTRY = {
    24: {"lightgbm": ["lag_24h", "lag_168h", "price_rolling_mean_24h", "hour", "month",
                     "day_of_week", "is_holiday", ...],
         "sgdfnet":  [...], "timemixer": [...], "timesfm": [...], "rt916": [...]},
    96: {"lightgbm": ["lag_96", "lag_672", ...], ...},   # 96 点 shift(96)/672
}
```

**注意**：上面是概念骨架。实际落地要把**每个模型现有的 `feature_engineering` 里的列名/口径逐列对齐**（先列模型特征清单 → 合并成注册表 → 物化 → 用现有某天的输出做逐位 `assert_frame_equal` 验证零精度损失，即实施计划 §2.1 的验收方式）。

### 4.4 分阶段实施建议（可插入现有实施计划的 Phase 2）

| 阶段 | 内容 | 验收 |
|---|---|---|
| **S1 盘点+注册表**（0.5~1 天）| 列出现有 5 模型×2task 的特征清单与口径（shift 常量、p56、da_anchor）；建 `feature_store.py` 特征注册表（只声明，不缓存）| 注册表列名/口径与各模型现状逐项一致 |
| **S2 离线物化+零损失验证**（1~2 天）| 实现 `FeatureStore.ensure/build/slice`；单日回测对比：物化切片 vs 现状重算 → `assert_frame_equal` 逐位相等 | **特征逐位一致（黄金基线 diff 空）**；单日预测结果与现状逐字节一致 |
| **S3 全链路接入**（1~2 天）| `ledger_predict` 前加 `FeatureStore.ensure`；5 模型 adapter 全部改读切片；4 件套回归 + 黄金基线 diff | 全量回归 4 件套通过；submission_ready.csv 与黄金基线逐字节一致；214 天墙钟下降 |
| **S4 warm-start 续训**（2~3 天）| adapter 加 `warm_start` 开关 + ckpt 目录；LightGBM `init_model`、PyTorch checkpoint（含 scaler/scheduler）续训；调度层按 §3.2 规则决策（周一/节假日/漂移→全量）| 同精度下 epoch 减 ≥50%；防泄漏断言（训练窗终点=target-1）通过；漂移检测函数跑 2 周出信号质量报告 |
| **S5 版本管理+增量**（可选，后期）| manifest 指纹告警、增量追加、DA 预测值作 anchor 的 A/B | 指纹失效即告警；增量语义正确 |

**与现有实施计划的衔接**：S1~S3 对应 `EFM3_链路稳健性_训练加速_融合改进_实施计划.md` Phase 2 的 #2（特征预计算）；S4 对应 #3（warm-start）；S5 对应 Phase 1 的 B2（缓存指纹）。

---

## 5. 风险与纪律

- **零精度损失是硬约束**：S2 必须逐位 diff 通过才进 S3；任何"预计算值 ≠ 重算值"必须查清（大概率是 asof 遮蔽或 shift 常量问题，可能是现有 bug 也可能是新引入 bug——这正是统一口径的额外价值）。
- **防泄漏铁律不因优化而放松**：物化的 p56 遮蔽、训练窗终点=target-1、da_anchor 取可见日前值，三条全部进 `check_preflight_health.py` 断言。
- **失败要响亮**：物化失败/指纹不一致 → manifest + delivery_report 告警段，绝不停在最响的位置（能运行是底线，skill §4.4）。
- **A/B 再放大**：warm-start 与漂移检测先在 1~2 天小规模验证信号质量，再全量（skill §4.2 / 实施计划风险纪律）。
- **GPU 重活只在服务器**，本机 CPU-only 不做会误导的判断（skill §4）。

---

## 参考资料（本次 webfetch 实证核对）

1. Feast — *Point-in-time joins*：https://docs.feast.dev/getting-started/concepts/point-in-time-joins.md （asof 语义、TTL 向后扫描）
2. Feast — *Data ingestion*：https://docs.feast.dev/getting-started/concepts/data-ingestion.md （offline/online store、materialize_incremental、push source）
3. Tecton / Databricks — *What is a Feature Store?*：https://www.tecton.ai/blog/what-is-a-feature-store/ （5 组件：transformation/storage/serving/monitoring/registry；time-travel；training-serving skew；batch/streaming/on-demand transform）及 https://www.databricks.com/blog/ 上该文归档
4. Chronon (Airbnb) — https://chronon.ai/index.html （声明式特征、point-in-time correctness、时间窗聚合、防泄漏、生产案例 Airbnb/Stripe/Netflix/OpenAI）
5. SageMaker Feature Store — https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-getting-started.html ；featurestore.org 对照表（离线 parquet/S3 + 在线 Dynamo/Redis）https://www.featurestore.org/
6. LightGBM — `lightgbm.train` API（init_model=continue training、keep_training_booster）：https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html ；调参页（save_binary 数据集缓存、early stopping）：https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
7. PyTorch — *Saving and Loading Models*：https://pytorch.org/tutorials/beginner/saving_loading_models.html （state_dict、checkpoint dict、resume training、warmstart strict=False、map_location 跨设备）
8. TimesFM (Google Research) — https://github.com/google-research/timesfm （TimesFM 2.5、200M 参数、LoRA/PEFT 微调示例、冻结主干思路）
9. 已有项目调研（衔接）：`docs/工业界时序预测训练加速与精度提升调研报告.md`（Lago 2021 arXiv:2008.08004、Smith ICLR 2018、Hubicka/Marcjasz/Weron 校准窗口集成、asinh VST）、`docs/EFM3_链路稳健性_训练加速_融合改进_实施计划.md`
10. 领域经验（非本次实证抓取，按通用实践描述）：Kaggle M5 Accuracy 前排方案（LightGBM+lag/rolling 一次性特征表+滚动切窗）、GEFCom2014 赢家（GBM+QRA）、工业 MLOps 的 PSI/KS 漂移检测惯例。

> 注：凡给出具体数值/收益（如"省 50%+ 墙钟"、"S1~S5 工期"）为经验量级估计，落地前以本项目 1~2 天小规模 A/B 实测校准；设计取舍（自研轻量 store vs Feast）与来源判断在正文已标注"实证"或"推断"。
