# 极端电价分类修正：前置(per-model) vs 后置(融合后)——调研报告

> 项目背景：山东省电力现货价格预测（24 点正式交付 + 96 点辅助），7 个异构模型（lightgbm/timesfm/timemixer/sgdfnet/rt916）+ BGEW 自适应加权融合 + 极端价分类器。
> 现状（后置）：`ledger_classifier` 对 RT 融合结果做修正——`final_pred==1 且 y_fused<=100 → 置为 -80`（`fusion/classifier_bridge.py:90-92`）。
> 调研日期：2026-08-15。证据等级标注：**[Web验证]** = 本次联网检索到原文；**[知识]** = 本领域成熟文献，未逐字核对。
> 关联文档：`docs/多模型融合策略调研报告.md` §2.3/§3（已建议"分类器前置"）、`docs/EFM3_链路稳健性_训练加速_融合改进_实施计划.md` 第3行 P2 项、`docs/PLAN_96_POINT_PARAMETERIZED_COMPATIBILITY_FINAL_REVIEW.md` §10/§11。

---

## 0. 结论（TL;DR）

1. **根因不是"前置/后置"的位置，而是后置门控用了被污染的量 `y_fused`**。融合的均化效应把少数模型报负的信号抹平，`y_fused` 被拉高越过 100 门槛 → 真负价被漏掉。用"多数模型漏报时 y_fused 必偏高"这个统计量做事件门控，逻辑上自相矛盾。
2. **数学上：当修正函数是"共享概率 p 的同一函数"时，前置软修正 ≡ 后置软修正**（二者都是 `ŷ_f + h(p)·(−80−ŷ_f)`）。前置要产生真价值，必须利用**每模型自身的信息**（如各自 ŷ_m），或至少去掉 `y_fused<=100` 这道污染门控。
3. **推荐混合方案**：前置 per-model **软**修正（按 p 分级拉向 −80）+ 融合 + 后置**置信度门控**兜底（`p≥p_hi` 才触底，**不再依赖 ŷ_f**）。核心改动是把 `classifier_bridge.py` 的硬门 `p≥θ ∧ ŷ_f≤100` 换成**只依赖 p 的分级软修正**。
4. **前置的风险是真实的**：分类器共享 → 7 个模型的修正决策完全相关，分类器误报会被融合放大（硬修正时 7 路全置 −80）。但软修正按 p 分级，误报只造成与 p 成比例的拉偏，自限。净效果必须用事件级实验测。
5. **需要实验验证**：V0(现状) vs V1(硬前置) vs V2(软前置) vs V3(混合) 四方案，事件级指标（漏报/误报/修正精度）+ 全局指标 + 分类器召回上限归因。详见 §5。

---

## 1. 现状与失败模式（代码级定位）

### 1.1 现有链路

- 融合：`ledger_fuse` → `fusion/apply_daily_ledger_weights.py:153` `y_fused = Σ w_m · ŷ_m`（BGEW 权重，逐 task/period 在线学习）。
- 分类器：`ExtremPriceClf/merge_model/core/extreme_price_radar/` 两阶段级联（p1 OOF 概率特征 → stage2 LightGBM/CatBoost/XGBoost），**纯特征驱动**（竞价空间预测值/净负荷/新能源渗透率 + 日前电价 lag24/48/168 + 实时电价 lag48/168，见 `features.py`），不依赖任何模型输出 → **天然可以在融合前运行**。输出 `final_prob`、`threshold`、`final_pred`（阈值由 P-R 曲线在 Precision≥0.70 下最大 Recall 寻优，`classifier.py:50-92`）。
- 修正：`classifier_bridge.py:90-92`：
  ```python
  merged["y_fused_corrected"] = merged["y_fused"]
  mask = (merged["final_pred"] == 1) & (merged["y_fused"] <= 100)
  merged.loc[mask, "y_fused_corrected"] = -80.0
  ```

### 1.2 失败模式

对真实负价事件小时（y≈−80），设：
- D = 报出负价的模型集合（ŷ_m ≤ δ≈0），W_D = Σ_{m∈D} w_m；
- 漏报模型输出均值 μ_miss（正常时段模型预测多在 +40~+250）。

融合期望：
```
E[ŷ_f] = W_D·E[ŷ|D] + (1−W_D)·μ_miss
```
当"多数模型漏报"（W_D=0.2~0.35）且漏报模型此时普遍报高位（μ_miss≈150，尖峰邻接时段常见），E[ŷ_f]≈0.3×(−60)+0.7×150 = 87，再叠加漏报群右尾（个别模型 180~250）→ **P(ŷ_f>100) 显著不为 0**。`y_fused<=100` 门槛挡住 → 真负价漏掉。

**统计本质**：门槛用 `ŷ_f` 当"这是不是负价事件"的检验统计量。但 `ŷ_f` 在模型分歧时是**高方差、有偏**的统计量——它在 y=−80 与 y=正常 两类下的分布高度重叠。用它做门控 = 用被污染的量判断事件，系统性牺牲事件召回换取低误报。

> 附带确认：LightGBM 腿内部已有自有负价钳制（`train_fix.py:729` preds<-80→-80；DA 阈值 0.7 才修正、RT 更激进），说明**模型间对负价事件的表达能力本就异构**——有的腿能输出接近 −80，有的完全不能。这正好是前置 per-model 方案可利用的信息。

---

## 2. 文献/工业界证据

### 2.1 极端价在 EPF 中是公认的独立难题（需要专门的机制）

- **[Web验证]** Weron (2014)《Electricity price forecasting: A review of the state-of-the-art with a look into the future》, IJF 30:1030-1081 (DOI 10.1016/j.ijforecast.2014.08.008)——EPF 领域综述圣经，明确把"价格尖峰/负价"列为标准模型最难处理的成分，处理手段分类为**预处理（spike filtration）、专用模型（regime-switching）、后处理**三大流派。
- **[Web验证]** Lago, Marcjasz, De Schutter, Weron (2021)《Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark》, Applied Energy 293:116983 (DOI 10.1016/j.apenergy.2021.116983)——近年最权威 EPF 综述；best practices 强调多样异构模型集成 + 稳健预处理；指出极端价必须专门处理，通用误差指标（SMAPE 等）会掩盖极端价技能。
- **[Web验证]** Gani et al. (2026)《Deep Time-Series Models Meet Volatility: ... Australian NEM》, arXiv:2602.01157——实测结论："**all models experience substantial degradation under extreme and negative prices**"，直接支持"融合前必须先修正/保护极端价信号"的必要性。

### 2.2 极端价检测 = 独立分类问题（两段式）

- **[Web验证]** Lu, Dong, Li (2005)《Electricity market price spike forecast with data mining techniques》, EPSR 73:97-104 (DOI 10.1016/j.epsr.2004.06.002)——最早把尖峰预测当**独立分类任务**的经典论文，用分类器判定尖峰事件再独立出价。本项目"分类器修正"正是此流派。
- **[Web验证]** Mount, Ning, Cai (2006)《Predicting price spikes in electricity markets using a regime-switching model with time-varying parameters》, Energy Economics 28:807-823 (DOI 10.1016/j.eneco.2005.09.008)——regime-switching（正常/尖峰两状态）的经典模型：**先判状态、状态内再出价**，结构上与"分类器前置+修正"同构。
- **[Web验证]** Alghumayjan, Yi, Xu (2026)《A Few-Shot LLM Framework for Extreme Day Classification in Electricity Markets》, arXiv:2602.16735——最新实践把"极端日分类"作为**独立于回归的前置模块**（德克萨斯市场，spike-day 二分类），与本研究方案方向一致。
- **[知识]** GEFCom2014 电价赛道（[Web验证] Hong et al. 2016, IJF 32:896-913, DOI 10.1016/j.ijforecast.2016.02.001）：参赛队普遍发现通用误差会把尖峰技能抹掉，需把尖峰日单独建模/校正——工业竞赛层面的共识。

### 2.3 变换/修正应在组合之前（"先变换后融合"学派）

- **[Web验证]** Uniejewski, Weron, Ziel (2019)《Variance Stabilizing Transformations for Electricity Spot Price Forecasting》, IEEE Trans. Power Systems 34:1979-1989 (DOI 10.1109/tpwrs.2017.2734563)——电价必须**先做方差稳定变换（asinh 族）再建模**，直接处理尖峰带来的异方差；变换发生在组合/建模之前。
- **[Web验证]** Janczura (2024)《Expectile regression averaging method for probabilistic forecasting of electricity prices》, arXiv:2402.07559——预期分位回归平均（模型平均 + 分位/预期分位组合），明确结论："**a variance stabilizing transformation should be applied prior to modelling**"——组合前做变换。
- **[Web验证]** Uniejewski (2026)《Variance Stabilizing Transformations for Electricity Price Forecasting in Periods of Increased Volatility》, EPSR 257:112992 (DOI 10.1016/j.epsr.2026.112992)——高波动时期 VST 收益最大（LEAR 最高 −14.6%）；"**rolling averaging across transformations** delivers the most robust improvements"——变换/修正作为**成员级处理再做平均**，与"per-model 修正再融合"完全同构。

### 2.4 极端市场概率预测：spike 过滤 + 后处理 + 组合

- **[Web验证]** Cornell, Dinh, Pourmousavi (2023)《A probabilistic forecast methodology for volatile electricity prices in the Australian NEM》, IJF (arXiv:2311.07289)——南澳高波动市场，方案 = **spike filtration（预处理）+ 多步后处理 + 分位回归做组合**；组合优于所有成员模型。注意它同时用了过滤(前)与后处理(后)，是"混合"的工业先例。
- **[Web验证]** Ziel & Steinert (2016)《Electricity Price Forecasting using Sale and Purchase Curves: The X-Model》, Energy Economics 59:435-454 (DOI 10.1016/j.eneco.2016.08.008)——供给侧(供需曲线)特征能预测尖峰与**连续 6 小时负价概率**：极值信号靠"专门特征+专门模型"抓，而不是靠通用回归均值。

### 2.5 组合理论（前置 bias 修正的正统性）

- **[知识]** Bates & Granger (1969) 组合开创、Granger & Ramanathan (1984) 回归组合（加截距吸收成员偏置）、Timmermann (2006)《Forecast Combinations》综述（Handbook of Economic Forecasting）——组合理论的正统做法是**组合前先做成员去偏**：若成员预测有系统偏置，先偏置校正再组合，否则组合继承偏置。极端价的"多数漏报拉高均值"正是"相关、不可分散的偏置"，线性组合（含 BGEW）无法消除，必须成员级事件修正。
- **[Web验证]** Lugosi & Mendelson (2021)《Robust multivariate mean estimation: The optimality of trimmed mean》, Annals of Statistics 49(1) (DOI 10.1214/20-aos1961)——鲁棒均值估计理论：**截断均值/中位数等鲁棒算子**对离群/少数极端信号比算术平均稳健得多。这给"用 min/低分位数做门控、而非用被均化污染的均值"提供了理论背书（§4 的"一致性兜底"用到的正是这个思想）。

### 2.6 直接相关（先例不足的诚实说明）

- 未检索到专门回答"极端价修正放融合前 vs 融合后孰优"的权威 paper——这是一个**工程组合层问题**，文献只在相邻维度给了证据（§2.3 先变换后组合 / §2.5 先去偏再组合 / §2.2 独立检测器）。本项目结论需用自己的 A/B 回测支撑，这也是 §5 实验的必要性。

---

## 3. 数学/统计论证

### 3.1 关键恒等式：共享 h 时前置 ≡ 后置

设共享修正函数 h(p)（对概率 p 的分级强度），对每个模型做软修正：
```
ŷ_m' = ŷ_m + (−80 − ŷ_m)·h(p) = (1−h(p))·ŷ_m − 80·h(p)
```
再融合：
```
ŷ_f' = Σ w_m ŷ_m' = (1−h(p))·Σ w_m ŷ_m − 80·h(p)·Σw_m = (1−h(p))·ŷ_f − 80·h(p)
     = ŷ_f + h(p)·(−80 − ŷ_f)
```
**结论：当所有模型用同一个 h(p) 时，"per-model 前置软修正再融合"与"融合后软修正"是同一个表达式。** 位置本身不产生差别。因此：

> 真正起作用的不是位置，而是三件事：
> (a) **形**——软/分级（h 连续）vs 硬（`p≥θ ∧ ŷ_f≤τ` 阶跃）；
> (b) **门控量**——只依赖 p（事件一致性）vs 依赖 ŷ_f（被污染）；
> (c) **差异度**——per-model h_m(p, ŷ_m)（只有这个才让"前置" ≠ 任何"后置 ŷ_f 的函数"）。

### 3.2 融合均化效应如何抹平极端价（后置缺陷的根）

平方误差分解：误差 e_m = y − ŷ_m，组合误差
```
(y − ŷ_f)² = (Σ w_m e_m)² = Σ_m w_m² e_m² + 2Σ_{m<n} w_m w_n e_m e_n
```
- 正常小时：e_m 弱相关、近似无偏 → 平均把方差摊薄（`(Σ w_m e_m)² ≤ Σ w_m e_m²`），这是融合赢单一模型的原因。
- 负价事件小时：报负模型的 e≈0，**漏报模型 e_m ≈ y−ŷ_m = −80−(+150) = −230（同号！）**。同号大误差 → 交叉项 `2Σ w_m w_n e_m e_n > 0` 巨大且为正 → **组合误差不仅不摊薄，反而被"大家错得一样离谱"主导**。平均化把少数正确信号（e≈0）稀释成多数错误信号的加权混合。

更根本地：平方损失模型预测的是条件均值 E[y|x]。对罕见极值事件，条件分布众数在正常区间，**E[y|x] 被拉回正常水平——每个平方损失模型都系统性低估极值幅度**（回归到均值）。这是**所有模型共有的、同号的、相关的偏置**，不属于"可分散方差"。组合理论（§2.5）说：平均只能缩可分散方差，缩不了相关偏置。**BGEW 权重再准也消除不了它**——极端价必须由事件级非线性修正处理，问题只在非线性放在哪里、用什么门控。

### 3.3 前置（per-model）为什么保住信号

前置后，即使只有少数模型被修正：
```
ŷ_f' = (1−W_D)·ŷ_f^miss + Σ_{m∈D} w_m·(−80)   （D 内硬修正时）
```
- W_D=0.3、漏报群 ŷ_f^miss≈+150 → ŷ_f' = 0.7×150 + 0.3×(−80) = 105−24 = 81 → **被拉下 100 门槛**；再叠加分级强度 h 或后置兜底即可触底。负价信号在融合前被保留，不再被平均抹掉。
- 若用**每模型自身 ŷ_m** 决定强度 h_m（模型自己报得低 → 修正更信任），则 ŷ_f' = ŷ_f + Σ_m w_m h_m(p,ŷ_m)(−80−ŷ_m)，**这不是任何 ŷ_f 的函数的等价物**——这才是"前置"独有的自由度：把"模型已倾向负价"这一额外证据纳入修正强度，即 3.1 的(c)。

### 3.4 前置风险：误判会被融合放大吗？——会，但可自限

分类器是**共享的**（同一份 p 喂给所有模型）→ 7 个模型的修正决策**完全相关**：

- **硬前置**（p≥θ 就置 −80）：分类器误报一次 → 7 路全置 −80 → 融合恰为 −80。误报损伤与现状硬后置（`p≥θ ∧ ŷ_f≤100`）同级，但**去掉了 ŷ_f 门槛的天然保护**，新增误报量 = 今天被 `ŷ_f>100` 挡住的那部分误报。这是前置方案真实的新增风险。
- **软前置**（分级 h(p)）：误报且 p 中等 → 只拉偏一部分（h=0.5 → 融合 ≈ 0.5·ŷ_f + 0.5·(−80)），**损伤与 p 成比例，自限**。分类器阈值寻优已约束 Precision≥0.70（`classifier.py:32`），高 p 处误报率本身受控。
- **结论**：前置把"漏报(假负)"风险转嫁给"误报(假正)"风险，转换比例由分类器校准质量决定。**分级软修正是两者间的可调杠杆**——这是推荐软而非硬的核心理由。

### 3.5 权重怎么处理

- **不建议**因前置修正改动 BGEW 在线更新机制（正常小时占 ~95%，权重被正常小时主导，修正对权重影响小）。
- **可选子实验**：权重学习是否用"修正后"的历史输出重放（保持训练/部署一致）。若不重放，训练时看到的是未修正输出、部署时用修正输出，存在轻微不一致；因事件稀少，影响有限，值得 A/B 量化（§5.3）。

---

## 4. 结论与推荐方案

### 4.1 方案裁决

| 方案 | 原理 | 缺陷 | 裁决 |
|---|---|---|---|
| **后置硬修正（现状）** | `p≥θ ∧ ŷ_f≤100 → −80` | 门控量被均化污染，真事件常被 `ŷ_f>100` 挡住 | ✗ 根因所在，必须改 |
| 前置硬修正 | `p≥θ → 每模型置 −80` 再融合 | 共享分类器误报被 7 路放大，无 ŷ_f 保护 | △ 可行但误报风险最高 |
| 前置软修正（共享 h） | 每模型 `ŷ_m+h(p)(−80−ŷ_m)` 再融合 | ≡ 后置软修正，无独立自由度 | ◯ 方向对，但未用尽前置优势 |
| **混合（推荐）** | **前置 per-model 软修正（分级 h，可选按 ŷ_m 加权强度）→ 融合 → 后置置信度兜底（p≥p_hi 触底，无 ŷ_f 门）** | 需实验调 (p_lo,p_hi)，误报自限 | ✓ **推荐** |

### 4.2 推荐落地公式（可直接改 `classifier_bridge.py:90-92`）

```
h(p) = 0                              p ≤ p_lo
     = (p − p_lo)/(p_hi − p_lo)       p_lo < p < p_hi
     = 1                              p ≥ p_hi
y_final = y_fused + h(p)·(−80 − y_fused)      # 单公式，去掉 y_fused<=100
```
- 初值建议：p_lo≈0.25、p_hi≈0.80（分类器当前阈值 ~0.55 附近），实验网格扫描。
- **概念上按"前置 per-model"部署**（对 `all_model_predictions_long.csv` 的每行 y_pred 做修正再融合，等价于上式，但为 (c) 自由度留口）；**实现上先做等价后置软修正**（一行改动、零管线重构），A/B 见分晓后再决定是否上 per-model 差异化强度。
- 可选兜底（实验评估）：`p ≥ p_hi` 时直接置 −80，保证高置信事件触底；用**一致性门控**替代 `ŷ_f≤100`——`p≥θ ∧ (min_m ŷ_m ≤ 30)`（鲁棒算子，§2.5 背书），双源证据一致才放行，天然免疫均化污染。

### 4.3 与现有决策衔接

- 本方案是 `docs/多模型融合策略调研报告.md` §3 第 3 条（"分类器前置"）与 `docs/EFM3_链路稳健性_训练加速_融合改进_实施计划.md` P2（"分类器前置：逐模型软修正(p→-80)再融合"）的**具体化与修正**：结论从"前置"细化成"软修正 + 无 ŷ_f 门 + 置信度兜底"，并明确前置的价值在 (c) 差异度而非位置本身。
- 96 点路径：分类器按小时广播 p 到 4 个 15 分钟刻度（`PLAN_96_POINT_PARAMETERIZED_COMPATIBILITY_FINAL_REVIEW.md` §11 已有广播设计），上式逐刻度应用即可，无需原生 96 点分类器。

---

## 5. 验证实验设计（可落地）

### 5.1 回测框架

复用现有 5 阶段链路做**离线重放**：历史窗口 ≥30 完整训练日账本（建议 2023-07 ~ 2026-08 或既有 214 天回测窗），逐日重放 `ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs`，只替换 classifier 修正逻辑，其余（权重/BGEW）冻结。运行前先跑 `scripts/tests/check_preflight_health.py`（11 项全绿，skill §4.2）。

### 5.2 方案对照（A/B）

- **V0 基线**：现状 `p≥θ_clf ∧ y_fused≤100 → −80`。
- **V1 硬前置**：`p≥θ_clf` 时每模型置 −80，再融合。
- **V2 软前置/软后置**：`y_fused + h(p)(−80−y_fused)`，无 ŷ_f 门槛。
- **V3 混合**：V2 + `p≥p_hi → −80` 兜底。
- **V4 一致性兜底**：V2 + 后置 `p≥θ_clf ∧ min_m ŷ_m≤30 → −80`。

### 5.3 指标与检验

**全局**：MAE / RMSE / capped-SMAPE（交付口径）+ Diebold–Mariano 检验（V0 为参照）。

**事件级（按事件数加权，skill §3 纪律）**：
- 负价事件召回/精确（y≤−50 定义事件，连续事件聚合）；
- 修正精度：被修正时 |ŷ_final−(−80)| 与落在 [−90,−70] 的比例；
- 误报代价：`p≥θ 但 y>阈值` 的修正对全局误差的增量。

**直接回答用户问题的诊断量**：
- **M1 = P(y_fused>100 | y≤−50, p≥θ)**：现状门槛挡住真事件的比率（目标 →0）。
- **M2 = P(p≥θ | y≤−50)**：分类器自身召回（本方案的天花板；若 M2 低，先救分类器再谈位置）。
- **M3 = 事件加权 SMAPE 相对 V0 变化**；**M4 = 新增误报事件率**。

**敏感性/子实验**：
- (p_lo, p_hi) 网格（如 (0.2,0.6)/(0.25,0.8)/(0.3,0.9)）；θ_clf 上下扫描（F2 最优 vs Precision 0.70）；
- **权重一致性**：BGEW 在"修正后历史输出"上重学 vs 不重学；
- 分季节/分时段（冬季午间负价高发段单独看）；96 点路径按小时广播跑同一对照。

**统计稳健性**：事件量少 → 事件配对检验 + bootstrap 置信区间；报告**事件级**而非小时级平均。

### 5.4 回归与交付保障

- 改动后跑 4 件套（`check_delivery_stability.py` 29 项 / `check_target_day_nan_regression.py` 16 项 / `check_sync_dataset.py` 41 项 / `check_adaptive_realtime_weight_days.py` 40 项）+ 黄金基线 `outputs/golden_baseline_24/` 逐字节 diff 确认 V0 之外无交付漂移。
- 修正后的官方输出是否进入 submission 维持现状（24 点不进入、96 点进入，`ledger_full.py:389`），本次只动修正逻辑不动输出契约。

### 5.5 前置（per-model 差异化强度，V5 可选）

仅当 V2/V3 显示"软修正方向正确但被漏报群拉偏"时上：`h_m = h(p)·(1−α·σ(ŷ_m−τ_m))`（模型自己报得高 → 修正强度打折），或为每个模型用自己的 per-model 概率（若未来有模型级分类器）。这一步才真正兑现"前置"的 (c) 自由度，需单独评估。

---

## 6. 参考文献

**本次联网核实（[Web验证]）**
1. Weron (2014) IJF 30:1030-1081, DOI 10.1016/j.ijforecast.2014.08.008
2. Lago, Marcjasz, De Schutter, Weron (2021) Applied Energy 293:116983, DOI 10.1016/j.apenergy.2021.116983
3. Nowotarski & Weron (2018) RSER 81:1548-1565, DOI 10.1016/j.rser.2017.05.234
4. Uniejewski, Weron, Ziel (2019) IEEE TPWRS 34:1979-1989, DOI 10.1109/tpwrs.2017.2734563
5. Uniejewski (2026) EPSR 257:112992, DOI 10.1016/j.epsr.2026.112992
6. Cornell, Dinh, Pourmousavi (2023) IJF (arXiv:2311.07289)
7. Mount, Ning, Cai (2006) Energy Economics 28:807-823, DOI 10.1016/j.eneco.2005.09.008
8. Lu, Dong, Li (2005) EPSR 73:97-104, DOI 10.1016/j.epsr.2004.06.002
9. Alghumayjan, Yi, Xu (2026) arXiv:2602.16735
10. Hong et al. (2016) IJF 32:896-913, DOI 10.1016/j.ijforecast.2016.02.001
11. Ziel & Steinert (2016) Energy Economics 59:435-454, DOI 10.1016/j.eneco.2016.08.008
12. Janczura (2024) arXiv:2402.07559
13. Gani et al. (2026) arXiv:2602.01157
14. Pan & Ezzat (2026) arXiv:2607.02623（ICML 2026 FM4S WS）
15. Lugosi & Mendelson (2021) Annals of Statistics 49(1), DOI 10.1214/20-aos1961

**领域成熟文献（[知识]，未逐字核对）**
16. Bates & Granger (1969) OR Quarterly 20:451-468
17. Granger & Ramanathan (1984) J. Forecasting 3:167-174
18. Timmermann (2006) "Forecast Combinations", Handbook of Economic Forecasting Vol.1

---

## 7. 落地清单

| 项 | 文件 | 改动 |
|---|---|---|
| 1 | `fusion/classifier_bridge.py:90-92` | 硬门 → 分级软修正 `y_fused+h(p)(−80−y_fused)`（去 ŷ_f 门槛）；兜底与一致性门为实验开关 |
| 2 | `pipelines/ledger_classifier.py` | 透传 p_lo/p_hi、是否兜底等参数；manifest 记录方案标识（v0/v2/v3）与 M1/M2 诊断量 |
| 3 | `fusion/apply_daily_ledger_weights.py:153` | 可选：前置 per-model 修正注入点（读每小时 p）——V5 才用 |
| 4 | `ExtremPriceClf` 推理产物 | 确认每时刻 `final_prob` 落盘（已有 `-80_prob.csv`，可复用） |
| 5 | 实验脚本 | `scripts/` 新增回测 A/B：V0/V1/V2/V3/V4 + §5.3 指标 + 4 件套回归 |
