# EFM3 三专项落地设计 — 报错接口 / 特征预计算+热启动 / 极端价混合修正

> 日期：2026-08-15（v2）。整合 6 份调研：
> `链路稳健性容错筛查报告`、`工业界时序预测训练加速调研报告`、`多模型融合策略调研报告`、
> `极端电价修正_前置vs后置_调研报告`、`特征预计算_FeatureStore_与WarmStart增量训练_调研报告`。
>
> 本设计回应三个专项反馈：①报错机制留接口、本地测试可发现；②特征预计算（充分理解项目：DA/RT 不混、
> SGDFNet 先 DA 后 RT、CPU/GPU 同用、零精度损失）+ warm-start；③极端价混合修正方案再设计。

---

## 专项一：报错/告警机制设计（留接口，本地测试可发现）

### 1.1 设计目标
- **现在（测试期）**：所有降级/回退/缺失**本地可发现**——测试脚本能断言、日志醒目。
- **未来（生产期）**：接口预留，可接前端/DB/邮件，但**本期不实现**。
- 核心：不把"静默"当默认，任何非 NORMAL 都要"有迹可循"。

### 1.2 接口设计：`DegradationEvent` 记录器

新增 `utils/degradation.py`，一个轻量事件总线：

```python
# utils/degradation.py（新增，~60行，无新依赖）
@dataclass
class DegradationEvent:
    ts: str            # 发生时间
    stage: str         # ledger_predict / ledger_weight / fuse / classifier / final / fallback
    kind: str          # model_missing / cache_bad / training_days_short / classifier_failed / null_output / degraded / ...
    severity: str      # HIGH / MED / LOW
    message: str
    detail: dict       # 可序列化

class DegradationHub:
    """进程内事件收集器（单例）。"""
    _events: list[DegradationEvent] = []
    @classmethod
    def emit(cls, **kw): ...          # 记录到内存列表
    @classmethod
    def snapshot(cls) -> list[dict]:  # 导出全部事件（供 manifest/report）
    @classmethod
    def reset(cls): ...
```

### 1.3 接入点（各阶段 emit 事件）

| 阶段 | 触发 | kind | severity |
|---|---|---|---|
| ledger_predict | 模型失败 / 队列异常 / 缓存 NaN | model_missing / cache_bad | HIGH |
| ledger_weight | 30 完整日不足（软门降级）| training_days_short | MED |
| ledger_fuse | 缺模型导致权重缺失 | fuse_partial | MED |
| ledger_classifier | 分类器失败降级 | classifier_failed | MED |
| final_outputs | 校验发现 null | null_output | HIGH |
| fallback | 触发应急降级 | fallback | HIGH |

### 1.4 本地可发现的三种形态（本期实现）

1. **manifest 顶层 `degradations` 字段**：`run_manifest.json` 加 `degradations: [...]`，非空即表示"本次非纯净交付"。
2. **delivery_report 告警段**：报告尾新增「⚠️ 降级与告警」段，列出所有事件（运维一眼可见）。
3. **测试断言接口**（关键——本地测试要能发现）：
   - `check_preflight_health.py` 扩展：检查 `degradations` 是否有不该出现的 HIGH 事件；
   - 新增 `scripts/tests/check_degradation_events.py`：构造"模型失败/缓存坏/训练日不足"场景 → 断言对应事件被记录。

### 1.5 生产期接口（预留，不实现）
- `DegradationHub` 预留 `emit_to = []` 注册表：未来可接 Webhook/DB/邮件，本期留空。
- manifest 的 `degradations` 字段即未来前端渲染的数据源，格式已定，无需改动。

---

## 专项二：特征预计算 Feature Store + Warm-start 续训

### 2.1 核心原则（用户立规）
- **充分理解项目**：DA/RT 不混、SGDFNet 先 DA 后 RT、CPU/GPU 同用。
- **零精度损失**：预计算只缓存不改变特征值（逐位 diff 验收）。
- **省时留收敛**：warm-start 省下的墙钟用于让模型充分收敛，精度只升不降。

### 2.2 架构：`FeatureStore`（新增，`utils/feature_store.py` ~60行）

```
ledger_predict.run(D)
  │ 1. FeatureStore.ensure(resolution, source)   # 一次性物化（含 asof 遮蔽/shift/rolling）
  │     ├─ feature_store/{res}/da_matrix.parquet   # DA 命名空间（shift(24)或shift(96)）
  │     ├─ feature_store/{res}/rt_matrix.parquet   # RT 命名空间（含 p56 asof 遮蔽）
  │     └─ manifest.json（指纹+版本+NaN统计）
  │ 2. ResourceScheduler（CPU/GPU 并行）
  │     └─ adapter 改为 feature_store.slice(model, task, D)  # 只读内存切片
```

### 2.3 关键设计决策（回应三个约束）

**（a）DA/RT 不混淆 —— 双命名空间 + 特征注册表唯一事实源**
- `feature_store/{res=24}/da|rt/` 与 `feature_store/{res=96}/da|rt/` 物理分离
- shift 常量**只在 FEATURE_REGISTRY 一处**：`lag_1d = resolution`（24→shift(24)，96→shift(96)）
- 模型侧**无任何 shift**，从源头杜绝 24/96 混用（skill §2 红线）

**（b）SGDFNet 先 DA 后 RT —— 两阶段物化**
- 阶段 1：物化 DA 矩阵 + 跑完 DA 腿（da_anchor = 源表 da 价列，零行为变化）
- 阶段 2：物化 RT 矩阵，`da_anchor` 引用阶段 1 产物；`da_lag_24/168` 按 RT asof 遮蔽
- 先用源表列实现（零精度损失），DA 预测值版本作为后续 A/B 增强

**（c）CPU/GPU 同用 —— 零成本安全**
- 特征矩阵在 scheduler 前物化进内存 → 各线程（CPU: lightgbm/sgdfnet/timesfm；GPU: timemixer/rt916）只读切片
- 线程池同进程共享内存 → 天然线程安全；归一化对 `.copy()` 做，不污染共享
- parquet 原子写（tmp+rename）

### 2.4 版本与失效
- 缓存键 = `res{resolution}_v{特征定义哈希}_{源文件指纹}`
- 数据更新（爬虫/回填）→ 指纹变 → 自动重建；特征代码改 → 版本变 → 重建
- 回测第 2 遍零重算（指纹命中）

### 2.5 Warm-start 续训（`runner` 层）

| 模型 | 续训方式 | 注意 |
|---|---|---|
| LightGBM | `lgb.train(init_model=前日booster)` | **树数无限增长**：续训轮数小 + 每7~14天全量重置 |
| TimeMixer/RT916/SGDFNet(PyTorch) | `state_dict` 恢复（含 optimizer/scheduler/scaler） | **重建 DataLoader**；LR 从 base 重新 warmup；周一/节假日全量重训 |
| TimesFM | 冻结主干，只训轻量头/LoRA | 回测期不整体微调 |

**防泄漏铁律**：续训只是"初始点=前一日权重"，训练窗仍只到 target-1 天（cutoff=14/p56）；SGDFNet 的 da_anchor 必须是已可见的日前值。

### 2.6 验收
1. 预计算前后某日特征矩阵 `assert_frame_equal` 全等 → **零精度损失** ✓
2. 4 件套回归 + 黄金基线 diff 空 ✓
3. warm-start 同精度下 epoch 减少 ≥50%（省时）；或同预算下精度提升 ✓
4. 214 天回测墙钟 3 天 → ~1 天 ✓

---

## 专项三：极端价混合修正方案（再设计）

### 3.1 根因（调研定论）
- **不是"前置/后置"的位置问题，是后置门控用了被均化污染的 `y_fused`**。
- `classifier_bridge.py:90-92` 的 `final_pred==1 ∧ y_fused≤100 → -80`：多数模型漏报时融合均值被拉高 >100 → 真负价被门槛挡掉。
- 数学：共享 h(p) 时前置软修正 ≡ 后置软修正；前置价值只在 per-model 差异强度。

### 3.2 推荐方案：软修正 + 置信度兜底（混合）

```
h(p) = 0                          p ≤ p_lo (0.25)
     = (p − p_lo)/(p_hi − p_lo)   p_lo < p < p_hi
     = 1                          p ≥ p_hi (0.80)

y_final = y_fused + h(p)·(−80 − y_fused)      # 单公式，去掉 y_fused≤100 门槛
```

- **去掉 `y_fused≤100`**（污染门控）
- **分级软修正**：p 中等时只拉偏一部分（误报自限）
- **高置信兜底**：p≥p_hi 直接触底 −80
- **可选一致性门**：`p≥θ ∧ min_m ŷ_m ≤ 30`（鲁棒算子，双源证据才放行）

### 3.3 融入链路（分类器放在哪）
- **分类器本身前置运行**（它是纯特征驱动，`ExtremPriceClf` 不依赖模型输出，天然可先行）
- 输出 `final_prob` → 修正公式在 `ledger_classifier` 阶段应用（等价于后置软修正，一行改动）
- **先做等价后置软修正**（零管线重构），A/B 后再决定是否上 per-model 差异化强度（V5）

### 3.4 验证实验（V0-V4 对照）
| 方案 | 公式 |
|---|---|
| V0 现状 | `p≥θ ∧ y_fused≤100 → -80` |
| V1 硬前置 | `p≥θ → 每模型置-80 再融合` |
| V2 软修正 | `y_fused + h(p)(-80-y_fused)`（无门槛）|
| V3 混合 | V2 + `p≥p_hi → -80` |
| V4 一致性 | V2 + `p≥θ ∧ min_m ŷ_m≤30 → -80` |

**指标（事件级，按事件数加权）**：
- M1 = P(y_fused>100 | y≤-50, p≥θ)：现状门槛挡住真事件比率 → 目标 0
- M2 = P(p≥θ | y≤-50)：分类器召回上限
- M3 = 事件加权 SMAPE 相对 V0 变化；M4 = 新增误报事件率
- Diebold-Mariano 检验（V0 为参照）

### 3.5 与链路集成
- `classifier_bridge.py:90-92` 换公式；`ledger_classifier` 透传 p_lo/p_hi/方案标识
- manifest 记录方案（v0/v2/v3）与 M1/M2 诊断量
- 96 点路径：分类器按小时广播 p 到 4 个 15 分钟刻度，逐刻度应用

---

## 实施顺序建议（与主计划衔接）

```
Phase 1（链路稳健，已计划）：
  ① 修 D1/D2/B1（96点null崩溃/粒度断言/坏缓存）→ ② 建 DegradationHub + manifest/report/测试三形态
Phase 2（加速）：
  ③ FeatureStore 物化（S1 盘点注册表 → S2 物化+逐位diff → S3 接入+回归）
  ④ Warm-start（开关 → 漂移检测 → 周一重训）
Phase 3（精度融合）：
  ⑤ 极端价软修正 V2（一行改动）→ ⑥ V3/V4 对照 → ⑦ 事件加权/融合守卫
Phase 4（交付验证）：全绿 + 对照通过 + 服务器 smoke → 全量回测
```

**每步纪律**：改一处测一处、A/B 对照、不批量盲跑、4 件套+黄金基线+健康检查全绿（skill §4.2/§4.4）。
