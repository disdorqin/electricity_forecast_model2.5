# FeatureStore 特征预计算设计（S1 盘点 + S2 架构）

> 项目：EFM3 山东电力现货价预测。触发：用户确认启动特征预计算设计（"一次性算好特征、按日期切片"，消灭每个模型每天重复读 30MB xlsx + 重复算 shift/rolling）。
> 日期：2026-08-16。S1 特征盘点完成（explore 全代码核验，本文件即 S1+S2 设计稿）。
> 关联：`docs/archive/agent-research-2026-08/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`（概念与工业实践）。

> **Formal-96 policy (2026-09-19):** this document describes the retained
> FeatureStore shadow/experiment path. It is not the production input or ledger
> source. Formal `--96` uses DB sync → immutable D/T snapshot → FeatureViewBuilder
> on the single full model store and records `feature_store.mode=off`; the p56/
> fixed-cutoff references below remain historical compatibility notes and must
> not be used by the façade.

---

## 0. 核心收益（实测支撑）

- 刚实测：读 96 点 xlsx（30MB）要 **49.8s**（openpyxl 解析慢），特征计算本身 ~0s。
- 现状：LightGBM(`infer_da_fix.py:98`/`infer_fix.py:105`) / SGDFNet / TimeMixer / RT916 各自每次预测都 `read_excel` → **每模型每天重复 49.8s+**。
- 改造后：物化一次 ~50s，之后每次读 parquet ~0.1s → **单日省 3-7min，214 天回测省 10-25h**。
- 零精度损失硬约束（S2 逐位 diff 验证）。不影响链路（物化在 scheduler 前，各模型读切片代替读 xlsx）。

---

## 1. 特征注册表（唯一事实源，shift 常量只在此一处）

```python
# utils/feature_store.py  — FEATURE_REGISTRY
# 结构: {resolution: {namespace_da/rt: {model: {"cols": [...], "shift_consts": {...}}}}}
# 关键: shift 常量一律用 resolution 的倍数（N = slots_per_day），消灭 24 点遗留命名（48h/168h 实为 2N/7N）
FEATURE_REGISTRY = {
    96: {
        "da": {  # DA 命名空间（日前，target 日全掩，无 p56 遮蔽）
            "lightgbm": {
                "cols": ["hour", "month", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
                         "lag_price_target", "price_rolling_mean_24h",
                         "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
                         "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
                         "ramp_load", "ramp_solar", "prev_day_avg", "prev_day_max", "prev_day_min"],
                "shift_consts": {"lag_target": [96, 672], "rolling_mean": [96, 96]},
                "daily_stats": ["prev_day_avg/max/min"],  # groupby 业务日 → shift(1 天)
            },
            "timemixer": {"paradigm": "window", "seq_len": 384, "future_cols": [...]},
            # timesfm: 段窗口 + exog（无行式特征，物化价值低，仅原始列）
        },
        "rt": {  # RT 命名空间（p56 遮蔽在物化时统一施加）
            "sgdfnet": {
                "cols": [45 列生产特征集],
                "shift_consts": {"hist_lag24": [96], "delta_lag_24": [96], "delta_roll_mean_6": [576], ...},
                "special": "da_anchor=D+1日前价(已发布合法); delta_lag_1 需 resolution 化(原硬编码24=96点下6h隐患)",
            },
            "rt916": {"shift_consts": {"lag_48h": [192], "lag_168h": [672], ...}, "asof": 14, "recompute": True},
            "timemixer": {"paradigm": "window", "cutoff_hour_rt": 14, "baseline_lag1_cutoff": True},
        },
    },
    24: { ... },  # shift 常量 24/48/168 等
}
```

**注册表原则**：
1. **双命名空间** `da`/`rt`（物理上 `feature_store/{res}/da_matrix.parquet` + `rt_matrix.parquet`）。
2. **shift 常量 resolution 化**：`lag_N = N×resolution`（96 点 96/192/672；24 点 24/48/168）。命名统一用 `lag_{N}day`，不用 48h/168h 遗留名。
3. **daily 统计特征**（LightGBM morning/prev_day、RT916 prevday）都是"groupby 业务日 → shift(1天)"——注册表声明为 `daily_stats`，物化时特殊处理。
4. **两阶段依赖**：SGDFNet `da_anchor`、TimeMixer `da_values`、RT916 `da_pred` 都依赖 DA 产物 → 物化顺序 DA → RT。

---

## 2. 物化与切片（S2 核心）

```
FeatureStore.ensure(resolution, source)
  ├─ 版本 = f"res{res}_v{特征版本}_{源文件指纹}"（指纹= mtime+size+hash）
  ├─ 1. DA 矩阵物化（无 p56 遮蔽, target 日全掩 y）→ da_matrix.parquet
  ├─ 2. 跑 DA 腿（可选: DA 模型预测作 da_values/anchor）
  ├─ 3. RT 矩阵物化（p56 asof 遮蔽 + 重算 y 侧特征 + da_anchor 引用 DA）→ rt_matrix.parquet
  ├─ manifest.json（指纹/版本/NaN统计）
  └─ 读入内存（回测期零磁盘 IO）

FeatureStore.slice(model, task, target_date, asof=None) → DataFrame（只读副本）
  ├─ task=rt: df = rt_matrix[ds <= asof]  (p56 已物化遮蔽)
  └─ task=da: df = da_matrix[ds < target_date]  (日前全可见)
```

**关键设计**：
- **物化在 `ledger_predict` 之前**（scheduler 之前单线程一次完成），之后各模型线程只读切片 → 线程安全（与 resource_scheduler ThreadPool 兼容）。
- **cutoff 重算是强制阶段**：物化时对 y 侧特征统一施加 "asof 后重算"（LightGBM 整列遮蔽、RT916 `recompute_target_dependent_selected_features` 模式、SGDFNet visible 帧），不是可选。
- **原子写**：parquet 用 tmp+rename，manifest 记指纹。
- **增量**：爬虫新增尾行 → 指纹变 → 增量追加或全量重建（数据 <1GB，默认全量重建更简单可靠）。

---

## 3. 各模型接入改造

| 模型 | 现状 | 改造 |
|---|---|---|
| LightGBM | `load_and_process_data` 每次 read_excel + feature_engineering | 读切片（已物化特征），删 feature_engineering 的 shift/rolling |
| SGDFNet | `preprocess_dataframe` 内部全量特征 | 读切片；`da_anchor` 由 DA 矩阵提供；`delta_lag_1` 硬编码 24 → 注册表 resolution 化（修复 96 点 6h 隐患） |
| TimeMixer | `make_past_features` 窗口内 rolling | 全表 rolling 物化后切片（等值）；`da_values/baseline` 由 DA 产物提供 |
| RT916 | `process_features` + asof 后 recompute | 读切片；`asof=14` 遮蔽已在物化统一做 |
| TimesFM | 段窗口 + exog（无行式特征） | 复用原始列，物化价值最低，可暂不改 |

---

## 4. 分阶段实施（S1 已完成盘点）

| 阶段 | 内容 | 验收 |
|---|---|---|
| **S1** ✅ | 特征清单盘点 + 注册表设计 | 本文件 |
| **S2** | 实现 `utils/feature_store.py`（ensure/build/slice）+ 单日物化；对照现有某天输出 `assert_frame_equal` 逐位相等 | 特征逐位一致，黄金基线 diff 空 |
| **S3** | `ledger_predict` 前加 ensure；5 模型 adapter 改读切片；4 件套回归 + 黄金基线 | 全链路回归通过，submission_ready 逐字节一致，214 天墙钟下降 |
| S4 | warm-start 续训（LightGBM init_model / PyTorch checkpoint） | 同精度下 epoch 减 ≥50%，防泄漏断言通过 |

**建议先做 S2（零损失验证）**：选 1 个模型（SGDFNet 或 LightGBM DA）物化 → 逐位 diff → 确认可行再铺开。

---

## 5. 风险与纪律

- **零精度损失硬约束**：S2 必须逐位 diff 通过才进 S3。
- **防泄漏不放松**：p56 遮蔽、训练窗终点=target-1、da_anchor 取可见日前值，三条进 check_preflight_health 断言。
- **失败要响亮**：物化失败/指纹不一致 → manifest + delivery_report 告警段。
- **SGDFNet delta_lag_1 隐患**：物化时显式 resolution 化，顺带修复 96 点下原 6h 语义（当前 RT 生产用 SGDFNet，此项有实际影响，需 A/B 确认不改坏现有精度）。

---

## 6. 24 点价差实验迁移与优化计划 v2（2026-08-19）

> 状态：active（实验设计，尚未接入正式生产）
> 依据：`scripts/experiments/spread_direction_24/` 的 30 天 Safe Mixed / Masked Direct 实验、
> `range_manifest.json`、`evaluation_ledger.parquet` 和 15/15 日冻结融合结果。

### 6.1 目标与当前判断

价差任务的目标为 `实时电价 - 日前电价`，最终只评价方向。当前 Safe Mixed Lag 在相同 30 天窗口上的最佳融合方向准确率约 **62.78%**，高于严格 Masked Direct 的约 **61.11%**，因此下一阶段以 Safe Mixed Lag 为主线，Masked Direct 作为严格安全基线保留。

14:00 后不可见的 D-1 价差只能使用截止时刻以前可证明存在的历史值补全。当前 `p15-p24 → D-2 同时段（lag48）` 是合法基线，不允许恢复裸 `lag24`。

### 6.2 FeatureStore 复用方案

价差实验不再让每个模型重复读取原始宽表、重复生成滚动特征和重复构造补全价差。复用 96 点链路的结构，但不混用 24/96 的物理行数和缓存目录：

```text
raw canonical hourly source
  └─ common 24-point feature base（一次物化）
       ├─ spread/masked_direct view
       ├─ spread/safe_mixed_lag view
       ├─ spread/lag168 view
       └─ spread/adaptive_fill view
            └─ model read-only slices
```

每个缓存版本至少包含：

- `resolution=24/hourly`；
- `task=spread`；
- `input_scheme` 与填充策略版本；
- cutoff=14:00；
- 源数据 SHA256；
- 特征注册表版本；
- 信息边界和 NaN 审计；
- 允许的来源最大时间 `source_max_ds<=cutoff`。

模型只读取只读切片，不能自行重新计算 shift/rolling 或绕过来源标记。缓存签名还必须包含模型名、训练窗口、epoch、seed、device 和 schema；参数不一致时禁止复用。

### 6.3 加速实施顺序

| 阶段 | 工作 | 主要收益 | 验收 |
|---|---|---|---|
| A0 | 固化 Safe Mixed / Masked Direct 信息边界和合同测试 | 防泄漏 | target-day DA/RT/spread、actual 特征和 cutoff 后 RT 全部遮蔽 |
| A1 | 24 点价差 common FeatureStore | 消除每模型重复读源表和重复特征工程 | 与现有实验输入逐位 diff，源 SHA 和 manifest 一致 |
| A2 | 将不同填充策略实现为派生 view | 快速比较 lag48、lag168、滚动中位数、自适应混合 | 每种 view 单独版本、单独审计、不得串缓存 |
| A3 | 模型输出缓存和运行签名 | 重跑融合或改指标时不重复训练 | 命中缓存必须逐日、逐模型、逐段完整 |
| A4 | 模型级批次隔离和并行调度 | 避免 CPU/GPU 长跑互相干扰 | LightGBM/SGDFNet CPU 与 TimeMixer CUDA 批次可审计合并 |
| A5 | warm-start / 增量训练候选 | 进一步减少 TimeMixer 重复训练 | 先做精度等价 A/B，未通过前不进入正式模拟 |

FeatureStore 主要节省数据读取和特征构造时间；当前 30 天实验中耗时主体仍是 TimeMixer 训练，因此不能把 FeatureStore 的收益误报为整体训练时间线性下降。若要进一步压缩墙钟时间，必须单独验证模型权重复用或 warm-start 的训练语义。

### 6.4 单模型上限路线

固定 Safe Mixed 输入后，按以下顺序寻找单模型上限：

1. 比较 `lag48`、`lag168`、lag48/lag168 加权、历史同槽位滚动中位数和波动率自适应填充；
2. 加入填充来源、来源滞后日、可见标记、新鲜度和近期价差统计；
3. 对 `1-8/9-16/17-24` 分段报告方向、正向、负向、balanced、MAE；
4. 使用严格更早日期的结果做 walk-forward 选择；
5. 计算“事后每时段最佳模型”的 oracle 诊断上限。oracle 只用于判断理论余量，不得用于生产权重。

目标日 DA 电价、目标日 RT/价差和 cutoff 后实际电网特征仍不得进入输入。训练可以使用历史实际类电网特征，验证和推理必须使用对应时点可获得的预测类电网特征。

### 6.5 融合优化路线

当前融合已经超过单个最优模型，但 15 天开发窗偏短，分时段权重存在过拟合风险。后续按以下顺序推进：

1. 全局非负归一化连续价差权重作为稳定基线；
2. 三时段权重作为候选；
3. 使用收缩权重：`w_final = α*w_segment + (1-α)*w_global`；
4. 开发窗扩大到至少 30 天后再评估时段权重；
5. 最后探索只依赖历史信息的动态门控：近期模型表现、价差波动率、峰平谷、模型预测分歧和填充来源；
6. 最终仍先融合连续价差，再取 `sign`，不拆正负两套权重、不提前做类别投票。

融合是否接近理论上限，使用 `oracle - 当前融合` 的差距判断：差距小则优先稳定化，差距大才增加动态门控复杂度。每次必须同时对照最佳单模型、等权、全局权重和分时段权重。

### 6.6 月度验收门槛

- 30 天、三模型、24 点账本完整：`30×3×24=2160` 行；
- 每个 `(target_day, model, period)` 严格 8 行；
- 所有模型状态为 `ok`，无 silent fallback；
- 预测、实际价差无 NaN；
- 正向、负向和 balanced accuracy 同时报告；
- 融合测试窗必须严格晚于权重学习窗；
- 通过 cutoff、来源最大时间、目标日遮蔽和缓存签名审计；
- 产物只放 `outputs/experiments/`，未获批准前不得写入 `outputs/24/feature_store/spread`。

### 6.7 已完成的加速与首轮优化结果（2026-08-20）

- 已实现 `utils/feature_store.py::ensure_spread_base()`，版本为
  `spread_hourly_v1`：一次物化 24 点公共源表、业务日/时段键和价差列。
- `run_masked_spread_experiment.py` 默认使用共享实验缓存
  `outputs/experiments/spread_direction_24_shared_cache/`；as-of 视图按
  `(source_sha256, feature_store_version, target_day, input_scheme, cutoff)` 签名复用。
- 首次运行记录 `asof_cache_misses`，第二次运行记录 `asof_cache_hits`，且两次视图 SHA256 一致；缓存命中不会跳过合同审计信息。
- Windows 深路径下 sidecar manifest 采用短文件名，避免将 `.parquet.manifest.json.tmp` 写入时超过 MAX_PATH。
- 30 天 Safe Mixed 填充策略首轮结果：滚动同槽位中位数方向准确率 **60.42%**，lag48 **54.03%**，lag168/weekly **50.83%**，as-of lag **54.31%**。滚动中位数是当前最值得继续做成模型输入派生 view 的候选。
- 加入滚动中位数候选后，15/15 静态分段融合测试为 **61.67%**；严格早期历史的 prequential dynamic reliability 在 20 天测试窗为 **62.08%**。当前仍是 screening 证据，不得直接接正式生产。
- 回顾性 oracle 显示测试窗的分段-日选择上限约 **83.06%**、逐点选择上限约 **95.56%**；说明当前融合距离可利用的互补上限仍有较大空间，下一步优先做历史信息驱动的动态门控，而不是继续盲目增大单模型参数。

### 6.8 扩大权重学习窗后的验证结果（2026-08-20）

为验证“训练加速后可以扩大权重学习天数”的判断，补齐了 **2026-06-16～2026-08-14 共 60 天**的实验账本：严格 Masked Direct 的 LightGBM/SGDFNet/TimeMixer 与 Safe Mixed 的四个填充基线通过 `combine_ledgers.py` 合并，最终为 `60×7×24=10080` 行；truth、NaN、模型日完整性均通过聚合校验。

在同一 60 天候选池上冻结后半段测试窗，比较 30/30、40/20、45/15、50/10 的权重学习：

| 权重学习窗 | 学习分段权重测试 | 等权测试 | 备注 |
|---:|---:|---:|---|
| 30 天 / 30 天测试 | 56.53% | **61.11%** | 学习权重明显过拟合负向占比与分段波动 |
| 40 天 / 20 天测试 | 57.29% | **62.29%** | 当前扩窗方案中最佳，但测试窗只有20天 |
| 45 天 / 15 天测试 | 58.06% | 60.56% | 学习权重仍未稳定超过等权 |
| 50 天 / 10 天测试 | 57.92% | 58.33% | 测试窗过短，仅作敏感性参考 |

结论不是“权重学习无效”，而是当前网格步长 0.05 的分段权重在 30～50 天上仍会追逐阶段性符号分布；扩窗后模型选择更稳定，但不能直接把拟合出来的分段权重上线。当前应采用：

1. 以滚动同槽位中位数作为主基线，SGDFNet 作为互补模型，TimeMixer 作为候选而非强制入选；
2. 先保留等权/全局权重作为稳健锚点，再做 `w_final = α*w_segment + (1-α)*w_anchor` 的收缩，`α` 只能用更早开发窗选择；
3. 以严格 prequential 结果作为动态门控验收，而不是用同一测试窗标签反推权重；本次 15 天 warm-up、30 天滚动历史的 45 天测试中，pair-gate **61.94%**，滚动中位数单模型 **61.30%**；
4. 继续积累至少一个完整月度窗口后，再决定是否保留分段权重。当前结果仍只属于实验区，不进入 `outputs/24/feature_store/spread`。

### 6.11 TimeMixer单模型结构实验结果（2026-08-21）

在同一 FeatureStore、同一安全填充策略和同一严格 cutoff 下，使用全部历史数据进行12个月滚动训练，评估窗口固定为开发30天、确认15天、留出15天。实验入口为 `scripts/experiments/spread_direction_24/run_timemixer_structure_experiment.py`，结果只写实验区。

安全输入为：D-1 p1-p14价差，p15-p24使用同槽位滚动中位数，缺失回退D-2同槽位；目标日实际价差只作为评估标签。

| 结构 | 留出方向 | 留出 balanced | 留出 MAE | signed-spread sMAPE |
|---|---:|---:|---:|---:|
| 统一24→24 | **60.28%** | **55.95%** | **46.13** | 143.81% |
| 三段独立、无输入权重 | 56.39% | 51.48% | 46.15 | 147.41% |
| 三段独立、输入权重 | 59.44% | 54.48% | 46.63 | **142.87%** |
| 共享编码器、三头、等权损失 | 56.94% | 51.86% | 47.72 | 143.97% |
| 共享编码器、困难度损失权重 | 55.83% | 51.00% | 47.47 | 146.40% |

开发窗口按三段历史困难度确定 `9-16` 最难，固定损失权重为 `[0.80, 1.35, 0.85]`；该权重只用于确认/留出，不随目标日变化。结果表明：

1. TimeMixer内部，统一24→24明显优于三个分段结构；
2. 输入加权改善数值sMAPE，但没有改善balanced方向准确率；
3. 共享编码器三头未超过统一模型；
4. 困难时段加大损失出现过拟合，暂不继续做损失权重学习；
5. 当前留出窗口SGDFNet balanced约61.24%，所以统一结构暂时只作为跨模型迁移候选，不直接接生产或权重学习。

### 6.12 跨模型迁移验证结果（2026-08-21）

已将可迁移的安全输入策略（D-1 p1-p14可见价差、p15-p24同槽位滚动中位数，D-2同槽位回退）应用到 SGDFNet 和 LightGBM，仍使用12个月滚动训练及30/15/15切分。实验入口为 `scripts/experiments/spread_direction_24/run_rolling_fill_model_migration.py`。

| 模型 | 留出方向 | 留出 balanced | 留出 MAE | signed-spread sMAPE |
|---|---:|---:|---:|---:|
| SGDFNet | 60.00% | **61.24%** | 55.72 | 140.21% |
| LightGBM | 55.00% | 51.62% | 46.75 | 147.68% |

迁移结果没有超过已有 SGDFNet 强基线，也没有证明滚动填充策略能普遍提升其他模型；因此暂不启动融合权重学习。TimeMixer 的统一24→24是其模型结构内部的候选改进，不能未经重写直接等同迁移到 SGDFNet/LightGBM。后续若继续，应优先围绕 SGDFNet 的单模型上限和更长时间窗口稳定性验证，而不是直接扩大融合复杂度。

### 6.9 LEAR 与共享主干的60天验证（2026-08-20）

为验证“引入经典强基线”和“共享主干是否能跨时段迁移”两条路线，复用同一份严格因果 FeatureStore 基座和同一评价账本，运行 2026-06-16～2026-08-14 共60天：

| 模型 | 方向准确率 | balanced accuracy | MAE | signed-spread sMAPE |
|---|---:|---:|---:|---:|
| 滚动同槽位中位数 | 61.39% | **52.33%** | 75.20 | **133.88%** |
| `lear_shared_lasso` | 62.15% | 50.20% | **73.18** | 144.41% |
| `lear_segmented_lasso` | 60.21% | 50.09% | 74.02 | 145.35% |
| SGDFNet | 57.71% | **53.31%** | 87.61 | 141.26% |
| `shared_trunk_mlp` | 52.99% | 49.28% | 89.35 | 145.66% |

该窗口实际负价差为 `900/1440=62.5%`，因此 `lear_shared_lasso` 的 62.15% 总准确率主要来自负向偏置，不能称为方向能力提升。当前共享主干 MLP 不具备接入价值；LASSO/LEAR 适合作为工程基线和数值误差补充，但未超过滚动中位数的 balanced/sMAPE 组合。

价差 sMAPE 使用 signed-spread 专用公式：

```text
100% × mean(2 × |pred - true| / (|pred| + |true|))
```

不使用价格链路的 `max(value, 50)` 地板；两者为0时该项记0%，仅一方为0时记200%。

### 6.10 外部工程经验转化为本项目路线

公开的电价预测工程基准普遍把 LEAR、DNN/树模型和简单滞后基线放在同一滚动评测框架中，并同时提供 MAE、sMAPE/MASE 与 Diebold–Mariano、Giacomini–White 等差异检验；其核心经验不是“模型越复杂越好”，而是强简单基线、长时间滚动窗口、严格信息边界和可复现比较。可参考 [epftoolbox](https://github.com/jeslago/epftoolbox)、[LEAR 原论文](https://arxiv.org/abs/1509.01966) 和 [电价预测综述/基准](https://arxiv.org/abs/2008.08004)。

据此，当前优先级调整为：

1. **先修正目标与状态建模**：对滚动中位数/SGDFNet 的错误样本按时段、正负号、价差幅度、预测分歧和填充来源分层，确认性能瓶颈是否集中在 `p15-p24` 或 regime 切换；
2. **再做历史信息驱动的门控**：只使用目标日前可得的近期准确率、波动率、来源新鲜度和模型分歧，采用收缩到全局锚点的非负连续权重；
3. **最后再做模型复杂化**：shared trunk 必须采用 resolution-aware slot/segment head，并以 balanced、signed-spread sMAPE 和 prequential 结果共同验收；
4. **迁移到24/96电价链路时只迁移工程能力**：FeatureStore、缓存签名、账本审计、walk-forward 和显著性检验可复用，价差的小时特征、填充策略和模型结论不可直接复制到96点电价任务。
