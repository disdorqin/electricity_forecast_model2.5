# 96 点全链路调试记录（2026-08-03）

> 场景：智川云 RTX 3090 上 2025-12 预热 → 2026-01-01 全链路，暴露 7 个 96 点兼容 bug。
> 全部已修复并在本地 2026-01-01 完整链路验证 `delivery_status=NORMAL`。
> 24 点（hourly）主线保持逐字节不变。

---

## Bug 1: keep_cols 漏 business_period → 预测账本整列 None

**症状**：`outputs/ledger_96/*/prediction/prediction_ledger.parquet` 的 `business_period` 整列 None。
`validate_ledger_window` preflight 的 `astype(int)` 崩溃（`int() argument must be ... not NoneType`）。

**根因**：`pipelines/ledger_predict.py::_predict_via_registry` 和两个 adapter
（`runners/adapters/lightgbm_v1.py`、`timesfm_v1.py`）的 `keep_cols` 白名单
**没含 `business_period`**。虽然 `standardize_business_columns` 96 点正确算出该列，
但 keep_cols 截断时把它丢了 → append 进账本时全 None。

**本地 07-16 账本正常**的原因：走 seed 脚本（`seed_96_ledger_cache.py`），绕过该路径。
**服务器异常**：走 `ledger_backfill → ledger_predict → _predict_via_registry`，命中 bug。

**修复**：3 处 keep_cols 加 `business_period`。commit `308fc07`。

---

## Bug 2: append 时从 ds 重建 business_period（兼容旧缓存 CSV）

**症状**：即使 keep_cols 修好，**已生成的旧预测 CSV（无 business_period 列）** 仍会写进账本为 None。
且 dedup key（`_ledger_key_cols`）只在有该列时才加 → key 退化为 hour_business → **96 点压成 24 点**。

**根因**：`append_predictions_to_ledger` 不重建缺失的 business_period。

**修复**：append 时若 `business_period` 缺失/全 None 且 `ds` 含非整分钟（15min 档），
从 `ds` 重建 business_period/hour_business/period。commit `345b931`。
另附 `scripts/rebuild_prediction_ledger_96.py`：扫描 runs_96 已有预测 CSV 重建整个账本，
**不重跑模型**（本地验证 DA 288 / RT 384，0 坏天）。

---

## Bug 3: delivery_quality preflight 硬编码 hour_business

**症状**：`validate_ledger_window → _check_ledger_against_grid` 报 `KeyError: 'hour_business'`
（96 点账本无该列的分组逻辑）或把 96 行误判为 24 行。

**根因**：`counts` 分组和 `n_expected` 硬编码 `hour_business`（24 点语义）。

**修复**：`_check_ledger_against_grid` 用 `slot_col`（96=business_period）分组和计期望行数。
commit `e47175a`。

---

## Bug 4: build_ledger_training_table merge key 用 hour_business → 4 倍笛卡尔爆炸

**症状**：`ledger_weight` 报 `coverage failed: expected_rows=8640, actual_rows=34560`（整整 4 倍）。

**根因**：`build_ledger_training_table`（`pipelines/prediction_ledger.py`）merge key
是 `["task", "business_day", "hour_business"]`。96 点账本同一 hour_business 有 4 个 15min 档，
pred 和 actual 各 4 行互相 merge → 16 行/时 → 每模型每天 96×4=384 行 → 30 天×3 模型×384=34560。

**修复**：merge key 优先 `(task, business_day, business_period)`；hourly 无该列自动回退。
commit `fbb1996`。验证 30 天 training 表 DA 8640 / RT 11520 精确匹配。

---

## Bug 5: GEF 权重学习 periods 硬编码 24 点 → weights.csv 空

**症状**：`ledger_weight` training 表正确（8640/11520 行）后，`weights.csv` 是 2 字节空文件，
`_validate_weights` 的 `pd.read_csv` 崩 `No columns to parse`。

**根因**：`DailyLedgerGEF(GEFConfig(window_days=30))` **没传 resolution** →
`GEFConfig.__post_init__` 不触发 → `periods` 保持默认 `("1_8","9_16","17_24")`（24 点）。
96 点数据 period 是 `"1_32","33_64","65_96"` → `fit` 里匹配不到任何行 → weights 空。

**修复**：
- `ledger_weight` 传 `resolution=res` 给 GEFConfig → periods/n_expected 自动变 96 点
- `daily_ledger_gef.fit` 排序用 business_period（96 点），回退 hour_business（hourly）

commit `702c5af`。验证 realtime weights 12 行（4 模型×3 段），各段权重和=1.0，**权重非等权**
（sgdfnet 1_32 段 0.88）→ 真 30 天学的动态权重。

---

## Bug 6: emergency_fallback 96 列契约

**症状**：postflight FAIL → 触发 fallback → `KeyError: ['hour_business'] not in index`。

**根因**：`FALLBACK_COLUMNS` 硬编码 hour_business（24 点）。

**修复**：96 点用 `FALLBACK_COLUMNS_96`（business_period）。commit `c82ecf5`。

---

## Bug 7: postflight 把 classifier 降级误判为 FAIL

**症状**：classifier 失败降级（ExtremPriceClf 模块缺失）→ `complete_with_warnings` →
postflight 期望 `complete` → 误判 FAIL → 触发 fallback。

**根因**：`validate_daily_submission` 的 stage 检查不允许 `complete_with_warnings`。

**修复**：classifier 允许 `complete_with_warnings`（计划 §11：分类器失败 ⇒ 官方输出回退未修正值）。
commit `c82ecf5`。

---

## 本地验证（决定性）

```bash
python main.py --pipeline ledger_full --date 2026-01-01 --resolution 15min \
  --data-path data/shandong_pmos_96_full_v2.xlsx \
  --ledger-root outputs/ledger_96 --runs-root outputs/runs_96
```

结果：`delivery_status=NORMAL`，exit 0，`submission_ready.csv` 96 行
（business_period 1-96，DA/RT 对齐），postflight PASS，权重为真 30 天动态权重。

## 经验教训

1. **96 点 vs 24 点的本质差异 = business_period(96) vs hour_business(24)**：
   所有按 hour 分组的逻辑（merge key、dedup key、期望行数、排序）都要检查是否漏了 96 语义。
2. **账本 append 是易错点**：keep_cols 白名单、dedup key、merge key 三处都要含 business_period。
3. **服务器 vs 本地差异**：本地 seed 脚本绕过 `_predict_via_registry`，掩盖了 keep_cols bug。
   真正的生产路径（backfill→predict→append）才会暴露。
4. **参数化要传到底**：GEFConfig 这类子组件若依赖 resolution，必须在调用处显式传入，
   否则默认 24 点逻辑静默产生空结果。
