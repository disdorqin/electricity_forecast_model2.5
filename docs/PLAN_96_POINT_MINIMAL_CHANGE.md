# 96 点预测兼容层 — 最小改动实施规划

> **性质**：实施规划文档。基于 2026-08-02 全仓库代码审计（管线硬编码清单 + 7 模型兼容性评估）。
> **原则**：不动 24 点生产主线；新增 Resolution 抽象层 + 独立 96 点 ledger；全部改动走兼容参数化。
> **前置**：96 点数据已就绪（`epf_market_data_96` 列齐全、实际值已修复验证）；目标价 = `epf_unit_data_96` 单机组出清价（da_cq_price/rt_cq_price）。

---

## 1. 目标与约束

| 项 | 决定 |
|---|---|
| 目标 | 96 点（15 分钟）预测：日前/实时 7 模型全兼容 |
| 目标价 | `da_cq_price`（日前）/ `rt_cq_price`（实时），单机组出清价 |
| 数据 | `epf_market_data_96`（13 组特征）+ `epf_unit_data_96`（价格），已同步本地 |
| 原则 | 最小改动、不碰 24 点主线、新增兼容层、运行出错风险最低 |
| 分片 | 沿用三段：`1_8`/`9_16`/`17_24` → 96 点下为 `1_32`/`33_64`/`65_96` |

---

## 2. 核心设计：Resolution 抽象层

新增 **`utils/resolution.py`**（单一事实来源），一个轻量 dataclass：

```python
@dataclass(frozen=True)
class Resolution:
    label: str                     # "hourly" | "15min"
    slots_per_day: int             # 24 | 96
    slots_per_period: int          # 8  | 32
    period_names: tuple[str, ...]  # ("1_8","9_16","17_24") | ("1_32","33_64","65_96")
    slot_column: str               # "hour_business" | "business_period"
    freq: str                      # "h" | "15min"
    minutes_per_slot: int          # 60 | 15

HOURLY  = Resolution("hourly", 24, 8,  ("1_8","9_16","17_24"),  "hour_business", "h", 60)
QUARTER = Resolution("15min", 96, 32, ("1_32","33_64","65_96"), "business_period", "15min", 15)
```

**关键设计决策**：`slot_column` 在 hourly 模式保持 `hour_business`（**外部契约零变化**），96 点用 `business_period`。这避免改动 24 点主线的列名。

---

## 3. 管线层改动（约 45 处硬编码 → 参数化）

### 3.1 已确认的硬编码分类

| 类别 | 数量 | 模式 | 处理 |
|---|---|---|---|
| A 行数/集合校验 | ~25 | `n != 24`、`set(range(1,25))`、`*24` | 换成 `resolution.slots_per_day` |
| B 循环 | ~6 | `for hour in range(1,25)` | 换成 `range(1, resolution.slots_per_day+1)` |
| C 周期边界 | ~8 | `VALID_PERIODS`、`infer_period` 1-8/9-16/17-24 | 从 `resolution.period_names` 计算 |
| D 槽算术 | ~12 | `hour 0→24`、`replace(hour=...)` | 换成 `resolution` 感知的槽换算 |
| E 模型内部 | 5 | 模型训练硬编码 24 | 单独模型改造（见 §4） |

### 3.2 具体文件改动（最小化，全部加参数默认值）

**新增文件**：
- `utils/resolution.py` — Resolution dataclass + HOURLY/QUARTER 常量

**修改文件**（每个只加一个 `resolution=HOURLY` 参数，默认值保证 24 点行为逐字节不变）：

| 文件 | 改动 |
|---|---|
| `utils/business_day.py` | 加 `resolution` 参数；`hour_business_from_timestamp` 增加 96 点分支（`business_period_from_timestamp`）；`infer_period` 从 resolution 取周期 |
| `fusion/contracts.py` | `VALID_PERIODS` 改从 resolution 计算；`infer_period` 复用 business_day |
| `cli/parser.py` | `--resolution` 已存在，扩展为传进 pipeline（hourly→HOURLY，15min→QUARTER） |
| `pipelines/ledger_full.py` | `_validate_final`/`_build_submission_ready` 加 resolution；96 点输出 `submission_ready_96.csv` |
| `pipelines/prediction_ledger.py` | `check_ledger_coverage` 的 n_expected 参数化；ledger 根目录 96 点用 `ledger_96/` |
| `pipelines/ledger_weight.py` | `_EXPECTED_HOURS`→resolution；expected_rows 的 `*24`→`*slots_per_day` |
| `pipelines/ledger_fuse.py` + `fusion/apply_daily_ledger_weights.py` | **最核心**：`for hour in range(1,25)`→`for slot in range(1,resolution.slots_per_day+1)`；period 从 resolution |
| `pipelines/delivery_quality.py` | 9 处 24 硬编码 → resolution |
| `pipelines/emergency_fallback.py` | 5 处循环 + period 边界 → resolution |
| `fusion/learners/daily_ledger_gef.py` | `GEFConfig.periods` 从 resolution 注入；`n_expected` 参数化 |
| `pipelines/ledger_classifier.py` | 无需改动（逐行，分辨率无关） |

> **最小改动策略**：每个函数只加 `resolution=HOURLY` 默认参数，24 点调用方完全不传 → 行为逐字节不变。96 点调用方传 `QUARTER`。

---

## 4. 模型层改动（7 模型兼容评估）

### 4.1 评估结论

| 模型 | 目标 | 难度 | 关键改动 | 分辨率无关部分 |
|---|---|---|---|---|
| **lightGBM** | DA | **中** | 小时掩码→96点段、lag 48/168→192/672、rolling(24)→rolling(96)、重训 6 个子模型 | 树模型、训练循环 |
| **SGDFNet** | RT | **中** | lag 24/168→96/672、rolling(12)→rolling(48)、段区间→96点、`target_hour` 逻辑 | HGB 回归、delta 方法、校准 |
| **TimesFM** | DA+RT | **高** | ~30 处 24 →96、`_build_segments`、freq "h"→"15min"、horizon 守卫移除 | 模型加载、外生变量、指标 |
| **TimeMixer** | DA+RT | **高** | seq_len 168→672、SEGMENTS→96点、rolling/diff 常数×4、时间特征 /96 | backbone、训练循环 |
| **RT916** | RT | **高** | 段长 8→32、seq_len 72→288、lag 常数×4、`editable_horizon` | TimesBlock、损失、训练 |

### 4.2 实施顺序建议

**阶段 1（先跑通 96 点 demo）**：
- TimesFM（零样本，无需重训，改造集中在 wrapper）+ LightGBM（每日重训，改造成本低）
- 这两个最快，能先跑出 96 点预测验证链路

**阶段 2（补全 7 模型）**：
- SGDFNet（中难度，改造集中在 data_contract.py）
- TimeMixer、RT916（高难度，段拆分重写）

### 4.3 关键原则

- 每个模型新增 `--resolution` 感知配置，**默认 hourly 走原路径**（逐字节不变）
- 96 点模式传 QUARTER → 模型重训于 96 点数据
- 滞后常数用 `resolution` 换算（`slots_per_day` 系数），不写死

---

## 5. 数据与目标价（已就绪）

| 项 | 状态 |
|---|---|
| 特征 | `epf_market_data_96` 13 组 fcast/actual，2022-01-01 起，每天精确 96 行 ✅ |
| 目标价 | `epf_unit_data_96.da_cq_price/rt_cq_price`，2022-01-01 起，0 缺失 ✅ |
| 实际值 | 已修复验证（7/8 列真实实际值，试验机组保持原样）✅ |
| 本地同步 | `data/remote_96/parquet/`（market + unit）✅ |

---

## 6. 交付输出（96 点）

```
outputs/runs_96/YYYY-MM-DD/final/submission_ready_96.csv  — 96 行，列：
  business_day, ds, business_period(1..96), hour_business(派生=ceil(p/4)),
  period, dayahead_price, realtime_price
```

- manifest 加 `resolution: "15min"` + `expected_rows: 96`
- postflight 复用（参数化行数/槽集合/p96 的 ds 检查）

---

## 7. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 24 点主线回归 | 每个函数默认 `resolution=HOURLY`；回归测试钉死（40/40、29/29、16/16、41/41） |
| 模型 96 点重训耗时长 | 阶段 1 先用 TimesFM 零样本跑通；模型逐一接入 |
| 96 点实际值质量 | 已验证 7/8 列真实（试验机组平台无实际值，保持原样） |
| 回滚 | 默认 HOURLY，revert 单文件即可；96 点用独立 ledger_96/ 不污染 24 点 ledger |

---

## 8. 工作量估计

| 阶段 | 内容 | 估计 |
|---|---|---|
| 1 | Resolution 抽象层 + 管线参数化 + 回归测试 | 3-5 天 |
| 2 | TimesFM + LightGBM 96 点改造 + demo 跑通 | 5-7 天 |
| 3 | SGDFNet + TimeMixer + RT916 接入 | 5-8 天 |
| 4 | 96 点完整链路 + 交付校验 + 文档 | 3-5 天 |

**总计约 3-4 周**（模型重训时间另计）。

---

_由 96 点预测规划任务生成（2026-08-02）。证据来源：`docs/PLAN_24_AND_96_POINT_FORECASTING_ARCHITECTURE.md`（策略 C）+ 全仓库硬编码审计。_
