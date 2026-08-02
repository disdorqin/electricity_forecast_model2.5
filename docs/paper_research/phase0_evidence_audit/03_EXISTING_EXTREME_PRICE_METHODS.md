# 仓库现有极端价格方法

## 状态定义

`I`=implemented；`C`=connected；`X`=executed successfully；`E`=empirically evaluated。四列分别判定，避免把“代码存在”误写成“有效”。

## 1. 全量机制矩阵

| 机制 | 位置 | 具体作用 | I | C | X | E | 证据结论 |
|---|---|---|:---:|:---:|:---:|:---:|---|
| RT916 SpikeResidualBranch | `models/RT916/*` | 原序列/差分/Z-score/soft mask，经多尺度 dilated Conv1d 输出 `spike_delta` | VERIFIED | VERIFIED RT | PARTIALLY VERIFIED | UNKNOWN | 有权重/内部 val，不足以证明 OOS 极端改善 |
| DynamicPeriodGate | RT916 model/core | `base_pred + gate*spike_delta`，gate 使用 base/spike/频谱上下文 | VERIFIED | VERIFIED | PARTIALLY VERIFIED | UNKNOWN | 条件融合但非 signed-tail router |
| TailWeightedHuberMSELoss | RT916 loss/core | 训练低/高分位数和剧烈差分加权 | VERIFIED | VERIFIED | PARTIALLY VERIFIED | UNKNOWN | 正负尾对称，阈值来自训练数据；无消融表 |
| RT916 三时段训练 | `models/RT916/core.py:196-225` | 01-08/09-16/17-24 独立 multi-output | VERIFIED | VERIFIED | VERIFIED for some runs | PARTIALLY VERIFIED | 内部 validation；非冻结 test |
| SGDFNet delta residual | `models/SGDFNet/models.py`, pipeline | 预测 `RT-DA`，输出 `DA+delta`，再做 segment bias | VERIFIED | VERIFIED RT | PARTIALLY VERIFIED | UNKNOWN | 通用残差思想已存在，但不是 post-hoc cross-backbone 模块 |
| SGDFNet tail sample weight | YAML/training | tail 样本放大 | VERIFIED | active weight=1.0 | N/A inactive | N/A | 功能存在但 active profile 不强调 tail |
| SGDFNet conditional tail metrics | `metrics.py:89-112` | 按 `abs(RT-DA)` 评估 tail | VERIFIED | VERIFIED | VERIFIED | PARTIALLY VERIFIED | 仅度量；evaluated-frame quantile 不可用于 test 调参 |
| TimeMixer regime calibration | `models/TimeMixer/repro_pipeline.py` | peak/valley/spike-day affine/bias 修正 | VERIFIED | production defaults `none` | UNKNOWN | UNKNOWN | 实现存在但默认未连入最终效果 |
| TimeMixer direct 24/3x8 output | TimeMixer pipeline | 避免逐点递归压平尖峰 | VERIFIED | VERIFIED | PARTIALLY VERIFIED | UNKNOWN | 最新 metrics NaN；Vault 声称改善未验证 |
| LightGBM solar negative classifier/regressor | LightGBM pipeline | solar 时段判断负价并下压 | VERIFIED | VERIFIED DA | PARTIALLY VERIFIED | UNKNOWN | exact cutoff 风险；artifact 不完整 |
| Daily negative-price classifier stage1 | `models/classifier/*` | 基于日级/时段特征筛选负价日 | VERIFIED | VERIFIED before bridge | VERIFIED in fixtures | PARTIALLY VERIFIED | stage1 dynamic threshold search 未调用，默认 0.55 |
| Cascade stage2 | `cascade_daily.py` | 灰区二次分类，后20%以 F2 选阈值 | VERIFIED | VERIFIED | VERIFIED in fixtures | PARTIALLY VERIFIED | 未见冻结 test 与阈值稳定性 |
| Hard correction | `fusion/classifier_bridge.py:78-95` | mask 后预测直接设为 -80 | VERIFIED | bridge connected | VERIFIED in fixtures | UNKNOWN | 当前 submission 读取 uncorrected，最终断链 |
| DailyLedgerGEF | `fusion/DailyLedgerGEF.py` | 三段历史误差指数加权，weight floor | VERIFIED | VERIFIED | VERIFIED fixtures | PARTIALLY VERIFIED | floor50 指标系统性忽略负价差异 |
| Time-of-day segmentation | 多模型/fusion | 09-16 单独建模/加权 | VERIFIED | VERIFIED | VERIFIED | PARTIALLY VERIFIED | 负尾集中支持；正高价不集中，不能统一解释 |
| Emergency historical median | `emergency_fallback.py` | 7d/30d/all 同小时中位数 | VERIFIED | VERIFIED on full failure | VERIFIED 2026-07-17 | NOT empirically evaluated | 交付 fallback，不是性能 no-harm |
| Prediction intervals/conformal | production search | 未发现 | NO | NO | NO | NO | active pipeline 无 uncertainty contract |
| Online drift update | rolling retrain/ledger | 每日滚动重训/历史权重 | PARTIALLY VERIFIED | PARTIALLY VERIFIED | PARTIALLY VERIFIED | UNKNOWN | 没有 drift detector + 预注册响应评估 |

## 2. 对最终输出的影响

- **VERIFIED**：融合预测 `y_fused` 进入 classifier bridge，corrected 文件确实会产生不同数值。
- **VERIFIED**：当前 `ledger_full.py:386-431` 构建提交时读取 uncorrected RT 文件，因此 classifier 与 hard `-80` 不影响当前代码最终 submission。
- **VERIFIED**：RT916/SGDFNet/TimeMixer/TimesFM 只有在模型成功、长表写入、融合权重非零并通过 fuse 校验时才影响 `y_fused`。
- **VERIFIED**：2026-07-17 正常链路失败后 emergency fallback 直接决定交付值；这时上述模型均不决定最终值。
- **PARTIALLY VERIFIED**：历史 fixture submission 使用 corrected 值，说明某个历史版本/脚本曾接通修正，但不能覆盖当前代码事实。

## 3. 与 tentative signed-tail correction hypothesis 的重叠

| 假设元素 | 仓库已有内容 | 差距 | 状态 |
|---|---|---|---|
| positive/negative signed routing | 负价 classifier；RT916 对称双尾 loss；TimeMixer peak calibration | 无统一正/负双路 router；正尖峰标签未定义 | PARTIALLY VERIFIED |
| post-hoc correction | hard -80、segment bias、regime calibrator、fusion | 模块不统一，且 current final 断链 | VERIFIED overlap |
| model-agnostic | fusion 接多个候选；Vault 有通用残差设想 | correction 未以一致 contract 接到 linear/MLP/LSTM/Transformer | UNKNOWN as contribution |
| uncertainty-aware | Vault 有 conformal/概率预测候选 | 生产无分位数、区间、校准误差或 epistemic signal | VERIFIED absent |
| no-harm fallback | weight floor、emergency delivery fallback | 无相对 base 的 validation constraint、reject option 或统计上界 | VERIFIED absent |
| sparse extreme expert | RT916 gated branch；Vault `残差模块.md:2` | 未以 OOF residual 库跨 backbone 训练/验证 | PARTIALLY VERIFIED |

**审计判断：VERIFIED**，tentative hypothesis 与既有仓库/笔记高度重叠于“残差、门控、负价分类、尾部损失、分段融合”。**UNKNOWN**：它是否具备文献新颖性或可发表贡献；本阶段禁止据此声明创新。

## 4. 明显方法风险

1. **VERIFIED**：负价修正 hard-code `-80` 和 `y_fused<=100`，选择期与跨市场有效性未知。
2. **VERIFIED**：floor50 fusion objective 会将所有负值裁到 50，模型在负尾的相对优劣无法影响该部分 SMAPE。
3. **VERIFIED**：stage1 threshold search 实现存在但调用被注释；文档若称动态阈值，与 active 实现不符。
4. **PARTIALLY VERIFIED**：RT916 对称 tail 权重可能将正/负机制混在一起；是否伤害常态和单侧尾部必须做 signed ablation。
5. **UNKNOWN**：分类标签、tail quantile、spike day、peak threshold 和 hard correction 是否曾查看最终测试集后确定。
6. **VERIFIED**：多套校准/融合连续叠加但没有统一 OOF residual 训练，容易产生二次拟合和不可归因增益。
7. **UNKNOWN**：现有权重、预测和日志是否可由锁定环境重复；TimesFM 本地模型缺失已在运行记录中出现。

## 5. 下一阶段只应验证的事实

- 不把 `>300` 或 `>500` 当论文阈值；仅在 training/validation 预注册正尖峰事件定义。
- 先修复最终值追踪并保存 `base/fused/gate/correction/final` 每层 OOS 值。
- 在统一 forecast origin/as-of contract 下重建 OOF residual，而不是复用含未来值的 exact 输出。
- 独立报告 positive tail、negative tail、sign flip、event recall、event magnitude、overall no-harm；不得只看 floor50 SMAPE。
- 对每个机制做 add-one/remove-one ablation，并跨至少 linear、tree、MLP、recurrent、Transformer/foundation backbone 验证。

以上是研究约束，不是最终方法推荐或新颖性结论。
