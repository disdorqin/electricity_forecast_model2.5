# 研究约束与开放问题

## 1. 任何未来方法必须尊重的事实

1. **VERIFIED**：目标是两个相关但不同的 24 点任务（DA 和 RT），业务日从 01:00 到次日 00:00；不能把普通 midnight-to-midnight 数据切片直接当等价 benchmark。
2. **VERIFIED**：负价和深负价集中在 09-16，但审计用高价切片不集中于该时段；signed tails 不能共享一个固定时段先验。
3. **VERIFIED**：现有仓库已包含 residual、classification、gating、tail loss、fusion、hard correction 和 fallback。未来论文必须以这些为内部 baseline/ablation，而不能将概念存在本身当创新。
4. **VERIFIED**：floor50 SMAPE 会抹去负价误差；论文结论必须基于未裁剪总体指标、两侧条件指标和事件指标共同成立。
5. **VERIFIED**：当前 classifier corrected 值没有进入 current-code final submission；实验前必须建立逐层 prediction lineage。
6. **VERIFIED**：default exact wrappers 存在 cutoff 风险；只有 `cutoff_safe` 或等价 as-of 重建的 base predictions 才可进入论文实验。
7. **UNKNOWN**：外生 forecast 的发布/修订时刻；在得到数据字典前，不能声称所有未来外生特征可用。
8. **UNKNOWN**：正尖峰定义、市场价格层级、时区、规则变更和冻结 final test；必须先由业务/数据证据确定。
9. **VERIFIED**：现有 Vault 几乎没有负电价/正电价尖峰原论文证据；Phase-0 不支持新颖性声明。
10. **PARTIALLY VERIFIED**：私有山东数据可做内部验证，但 CCF-A 目标还需要多个公开市场、跨 backbone、严格 OOS 和可复现实验。

## 2. 当前不受支持的假设

| 假设 | 状态 | 缺失证据 |
|---|---|---|
| 09-16 同时是正尖峰与负价的统一极端窗口 | VERIFIED | 该假设已被审计数据否定；高价审计切片主要不在该窗口 |
| hard `-80` 是稳定最优修正值 | UNKNOWN | 选择期、跨年/跨市场稳定性、validation-only 搜索 |
| signed routing 比对称 tail loss 更好 | UNKNOWN | 统一 OOS ablation |
| uncertainty 能可靠决定 correction/fallback | UNKNOWN | 校准的 uncertainty signal、coverage 和 selective risk |
| post-hoc 模块对所有 backbone model-agnostic | UNKNOWN | 统一 input/output contract 与 linear/MLP/LSTM/Transformer 结果 |
| no-harm 可由 weight floor 或 emergency fallback 保证 | VERIFIED | 该假设已被代码审计否定；二者分别约束权重/文件交付，不约束相对 base risk |
| 已有 fixture 证明改进 | VERIFIED | 该假设已被证据审计否定；fixture 是静态/烟雾证据且与 current code 不一致 |
| train/val/test 日期切分足以无泄漏 | VERIFIED | 该假设已被代码审计否定；exact wrapper 可在目标日内访问未来行 |

## 3. 可证伪研究问题（不等于最终 idea）

1. **RQ1**：在预注册正/负 tail 标签下，signed 双路修正相对单一对称 tail expert，是否同时降低 positive-tail MAE 与 negative-tail MAE，且总体 MAE 的恶化不超过 validation 冻结容限？
2. **RQ2**：用严格 rolling OOF residual 训练 correction，相对用 in-sample residual，跨日期和跨市场增益是否显著更稳定？
3. **RQ3**：修正触发器加入经 validation 校准的不确定性后，selective risk-coverage 曲线是否支配仅按点预测/时段触发？
4. **RQ4**：一个共享 post-hoc contract 在 linear、tree、MLP、LSTM、Transformer/foundation 至少五类 backbone 上是否取得方向一致的 tail 改善？
5. **RQ5**：负价 classifier 的 soft residual correction 是否比 hard `-80` 更少产生 sign-flip 和正常时段伤害？
6. **RQ6**：移除 `hour-of-day` 后，router 是否仍能识别负尾；若不能，其收益是否只是记忆 09-16 的时间先验？
7. **RQ7**：事件级 recall、峰/谷幅度误差、发生时刻误差是否会改变 floor50 SMAPE/总体 MAE 给出的模型排名？
8. **RQ8**：跨年度规则/供需漂移下，固定阈值与在线校准阈值的 tail risk 差异是否超过其正常区间成本？
9. **RQ9**：特征 as-of 重建后，exact 模式中观察到的 base/correction 增益有多少消失，从而量化历史 leakage inflation？
10. **RQ10**：对每侧 tail 分别使用 train-quantile、EVT threshold 和业务阈值，结论是否对合理定义稳健，而不在 test 上选择最佳阈值？

每个问题必须先固定 train/validation/test、允许的阈值搜索范围、主要指标与 no-harm 容限；最终 test 只运行一次。

## 4. 公共数据需要具备的能力

- **REQUIRED**：至少两个、理想三个不同市场的 DA/RT 或 intraday electricity price；保留负价与高价，不做 floor clipping。
- **REQUIRED**：原始时间戳、timezone、DST 重复/缺失小时规则、市场日和价格区域/节点标识。
- **REQUIRED**：可定义 forecast origin；外生量必须区分 forecast 与 realized，最好包含 issue time/vintage。
- **REQUIRED**：连续多年覆盖，以形成 train、validation、冻结 test 和 regime-shift period。
- **REQUIRED**：足够 tail event 数以按 signed side、hour、season 做置信区间；过少时必须报告不可辨识性。
- **REQUIRED**：许可允许代码、处理脚本和派生 split 公开；私有山东数据只作为额外 external/industrial validation。
- **DESIRABLE**：统一的 DA/RT price、load forecast、renewable forecast、weather forecast；若无 vintage，明确限制并只使用可安全重建特征。
- **DESIRABLE**：市场规则/price cap 变更记录，支持 concept drift 分层评价。

## 5. 实验前置门槛

1. **BLOCKER / UNKNOWN**：拿到正式数据字典和发布时刻表，生成机器可校验 feature-availability contract。
2. **BLOCKER / VERIFIED**：将 current final submission 明确改为 corrected 或 uncorrected，并记录唯一 source-of-truth；在生产代码修改获批前，本审计不改代码。
3. **BLOCKER / UNKNOWN**：冻结正尖峰、负价、深负价、事件合并和 09-16 分层定义；只能在 train/validation 决定。
4. **BLOCKER / UNKNOWN**：冻结完整日期 split，书面确认 final test 未参与现有阈值和模型选择；若已污染则另划未来期。
5. **BLOCKER / VERIFIED**：禁用论文实验中的 exact unsafe wrappers；为每个 base 保存 cutoff-safe OOF predictions。
6. **BLOCKER / UNKNOWN**：选择并验证 2-3 个公开市场，建立同一 task contract，而不是只追求列名一致。
7. **BLOCKER / PARTIALLY VERIFIED**：锁定环境、TimesFM 权重来源、随机种子、模型 artifact hashing 和失败策略。
8. **BLOCKER / VERIFIED**：实现统一 metrics suite，禁止以 floor50 SMAPE 单独选模型/权重。

## 6. 系统检索与 idea generation 的入口条件

达到上述 1-6 后，才能开展以 electricity price spike、negative price、signed tail、post-hoc correction、selective/no-harm forecasting、probabilistic/conformal、mixture-of-experts 和 cross-market transfer 为核心的系统检索。**UNKNOWN**：tentative hypothesis 是否新颖；在完成检索、去重和基线复现前，不应生成最终标题或宣称贡献。
