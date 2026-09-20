# 96→24 跨分辨率价差方向桥接

> status: active  
> 日期：2026-08-22  
> 责任领域：高频96点信息向24点价差方向目标的迁移与聚合  
> 验证依据：山东96点 R1 预测 + 山东24点 canonical 真值，所有最终方向指标只按24点 canonical 计算

## 核心动机

山东96点 R1 复现与跨年测试显示15分钟级价差方向具有很强可预测性，但96点价格目标与24点 canonical 价格目标不是简单四点平均的同一口径。因此不把96点聚合真值当作最终目标，而把96点预测视为高频辅助表征，学习到24点 canonical direction 的映射。

## 当前证据

### A. Paper-protocol / privileged 上限（不可部署）

- R1 paper protocol 96点方向：2023→2024 91.32%，2024→2025 90.28%，2025→2026-08-16 89.07%。该协议使用15分钟 lag1、目标日进行中的实际状态等信息，**不是 D-1 14:00 可部署96点预测**。
- 把这类高频预测 q50 简单四点 mean 后，用24点 canonical direction 验收：2026-01-01~08-14 为66.83%；17–24为70.74%。这只能证明“高质量高频状态对24点有桥接价值”，不能作为正式24点可上线成绩。
- 使用2026已发生月份训练 learned bridge 时，paper-protocol高频输入可在多个冻结窗口达到73%~78% raw direction；同样只作为 privileged/oracle bridge 证据，禁止计入正式70%门槛。
- 96点真实四点平均方向与24点 canonical direction 只有约64.70%一致，说明需要显式 domain/resolution bridge，不能把96点平均真值替代24点真值。

### B. D-1 14:00 Ahead（strict-v3，可部署口径）

- `run_r1_ahead.py` 只使用 D-1 14:00 前可见的价差状态、safe mixed lag、D日预测型电网特征；不使用 target-day actual、target-day DA 或 D-1 p57–p96 realized 信息。
- 已增加反事实信息边界审计：篡改目标日整天 DA/RT/spread 或 D-1 p57–p96 后，44个预测特征最大变化均为0；篡改允许的 D-1 p1–p56 后特征显著变化。2026-02/04/06/08 四个代表日全部 PASS。
- 已增加 source manifest 强门控：`aggregate_to_24.py` 与 `learned_aggregate_to_24.py` 默认拒绝 R1 paper-protocol/oracle 预测；只有明确标记 `D-1 14:00` 且全部 forbidden-source=false 的96预测才可进入24点实验。
- 已修正旧实现“任一特征 NaN 就删整行”的评估偏差；strict-v3 让 LightGBM 原生处理 NaN，并强制每个测试日96/96完整输出。2024/2025/2026测试分别35136/35040/21888行，完整性全PASS。
- strict-v3 96 q50方向：2023→2024 **56.21%**（balanced 54.89%），2024→2025 **55.76%**（53.48%），2025→2026-08-16 **55.79%**（53.69%）。
- strict-v3 2026 q50简单96→24、只以24点 canonical truth验收：mean **57.86%**、median **57.92%**、vote **57.84%**，balanced约50.3%~50.5%。
- strict-v3 OOF learned bridge（2024+2025训练、2026冻结测试）未超过简单聚合：最佳仍为 median 57.92%；logistic 53.97%、global LGB 50.94%、period LGB 50.81%。
- 2026年内四个严格前向 bridge 窗口也没有稳定的真实增益。最近窗口（bridge标签只用至2026-06-29，测试07-01~08-14）简单mean/median **61.94%**，但同期全负基线已 **61.67%**，balanced约 **50.5%**；learned period仅56.39%。因此该raw数字不能视作有效方向能力。
- 结论：**当前 strict 96→24 路线未达到正式候选水平；主要瓶颈是合法96 Ahead对正价差的识别能力不足，而不是聚合器本身。旧73%~78% bridge只保留为 privileged/oracle上限证据。**

## 可研究方向

1. learned aggregation：四个15分钟预测、quantile、confidence、regime probability → 24点 direction。
2. conditional bridge：只在96点预测内部冲突、接近零或特定时段触发学习器；其余保持mean baseline。
3. multi-resolution consistency：训练时同时监督96点局部目标与24点 canonical direction，约束高频表示服务于低频业务目标。
4. period-specific bridge：17–24可学习增强；9–16优先保守/专门建模；不允许一个全局桥接器破坏已强时段。
5. 未来论文可描述为 cross-resolution representation transfer / multi-resolution consistency learning，但必须以山东24点 canonical direction 为最终验收指标。
