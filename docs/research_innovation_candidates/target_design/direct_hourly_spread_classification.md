# 直接小时级价差方向/区间分类

> status: paused  
> 日期：2026-08-22  
> 责任领域：直接24点价差目标设计、方向分类、区间分类与多任务监督  
> 验证依据：最终只使用山东24点 canonical 的 RT-DA spread 方向作为主指标

## 文献触发

Das et al., IEEE Access 2022, *Forecasting Nodal Price Difference Between Day-Ahead and Real-Time Electricity Markets Using Long-Short Term Memory and Sequence-to-Sequence Networks*，直接研究小时级 DA/RT price difference。

论文的关键结果不能混读：
- 下一小时 value forecast 的整体 sign/direction accuracy：Seq2Seq 79.41%，XGBoost 75.95%，SVR 76.25%，BiLSTM 75.54%；2019跨期测试 Seq2Seq 73.28%。
- 大价差子集方向：Seq2Seq 对 |论文定义的高价差阈值| 样本可达约93%–95%。
- 42个价差区间的分类 accuracy：BiLSTM最高90.24%，Seq2Seq一层encoder为92.17%。这不是“24小时整体正负方向92%”，但证明把连续spread改成离散band监督可能比纯回归更容易学习稳定市场状态。

## 对山东24点的启发

1. 直接分类而非只回归后取sign：negative / near-zero / positive，或更细的强负/弱负/近零/弱正/强正。
2. 多任务：同时预测band、sign和spread数值，最终只用sign作为甲方主指标。
3. band边界不能照搬PJM美元阈值，应仅在山东训练集内按固定业务/分位数规则确定，并冻结到测试集。
4. 对极端spread可设辅助loss，提高少数但经济重要的强方向样本识别能力。
5. 文献使用48小时lag并主要做下一小时预测，不能把79%或90%+直接当作我们的D-1 14:00→D整日24点可达成绩；复现时必须另设严格24-step协议。

## 已完成复现（2026-08-22）

- `das_seq2seq_hourly.py` 保留论文核心设置：48h lag、encoder-decoder LSTM、Adam、batch64、dropout20%。山东2025年前9个月训练、10月验证、11~12月冻结测试1441小时：direction **79.78%**、balanced **79.35%**，与论文Seq2Seq 79.41%的量级高度一致，说明方法迁移成功。
- 把同一回归Seq2Seq机械扩成 `D-1 14:00 → 34步递推 → 取D日24点`，2026-01~08-14仅 **60.87%** direction、balanced 51.16%，明显负类塌缩。
- `das_seq2seq_direction.py` 按论文band/classification思想改为直接输出未来34个方向概率，仍只输入截止D-1 14:00的48小时spread；2026-01~08-14为 **51.68%** / balanced 51.49%，当前版本同样失败。
- 结论：Das Seq2Seq 的强项在短 horizon；不能靠单纯拉长decoder解决我们的24点提前预测。但“小时级直接价差Seq2Seq能复现到约80%”本身是有效经验，可作为未来短期/在线子模块或 privileged teacher。

## 当前处理

- 按用户决定，Seq2Seq 主线暂时搁置，不继续追加长跨度结构、5-band或多任务实验。
- 保留 `next-hour ≈79.78%` 的成功复现作为短 horizon 经验与未来潜在 Teacher 证据；`D-1 14:00→D日24点` 的60.87%/51.68%失败结果同样保留，不再投入当前冲70主线。
- 所有最终数字仍只以24点 canonical truth计算。
