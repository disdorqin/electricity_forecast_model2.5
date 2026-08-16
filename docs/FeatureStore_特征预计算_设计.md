# FeatureStore 特征预计算设计（S1 盘点 + S2 架构）

> 项目：EFM3 山东电力现货价预测。触发：用户确认启动特征预计算设计（"一次性算好特征、按日期切片"，消灭每个模型每天重复读 30MB xlsx + 重复算 shift/rolling）。
> 日期：2026-08-16。S1 特征盘点完成（explore 全代码核验，本文件即 S1+S2 设计稿）。
> 关联：`docs/archive/agent-research-2026-08/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`（概念与工业实践）。

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
