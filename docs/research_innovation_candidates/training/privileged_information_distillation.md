# 训练期特权信息与价差方向知识蒸馏

> status: active
> 日期：2026-08-22
> 责任领域：24点山东价差方向预测的训练机制创新
> 验证依据：山东 `data/24/canonical/shandong_pmos_hourly.csv`、P6 Feature Cube、冻结/滚动回测 manifest、信息边界审计；参考 LUPI / generalized distillation 文献

## 研究问题

业务目标是：在正式预测时只使用当时可获得的信息，预测山东 24 点 `实时电价-日前电价` 的方向；阶段目标为**方向准确率冲击 70%**。

历史训练样本中存在大量在该样本最终结束后可获得、但真实推理时不可获得的信息，例如目标日实际负荷、实际风光、真实 forecast error、完整 D-1 晚间市场状态。传统做法会把这些列全部丢弃；本方向尝试把它们改造成**训练期特权信息（privileged information）**：Teacher 可以使用，正式 Student 不可以使用。

核心原则：

```text
历史训练样本
regular X:   正式预测时可获得的 P6 / forecast / D-1 14:00 context
privileged X*: 历史结束后才知道的 actual、realized error、完整中间状态
label y:     spread direction

Teacher(X, X*) -> soft probability / regime / quantile knowledge
Student(X)   -> 正式部署，只输入 regular X
```

这区别于把未来真实量直接塞进线上模型：**privileged feature 只能作为训练机制，不得成为 Student 推理输入。**

## 山东 Phase 1 特权信息集合

第一轮只使用项目已有山东数据，不引入其他数据集；最终全部指标也只在山东上计算。

Teacher 候选输入：

1. P6 全部合法输入；
2. 目标日 actual fundamentals：地方电厂、联络线、风电、光伏、核电、自备、试验机组、直调负荷、竞价空间、新能源；
3. realized forecast errors：上述 actual - forecast；
4. actual 侧物理关系：实际 residual load、renewable share、bidding-space ratio 等；
5. 目标日前电价可作为 teacher-only 市场状态信息进行消融，但不得默认进入 P6 Student；
6. D-1 p15-p24 的完整历史市场状态可作为 teacher-only 中间序列统计量进行后续实验。

禁止作为 Teacher feature 的直接答案：目标日实时电价、目标日真实 spread、目标日 direction 本身。它们只能作为监督标签。

## 第一轮冲刺实验

### P0：Regular baseline

P6 regular features -> LightGBM direction classifier。

### P1：Oracle Teacher ceiling

`P6 + privileged fundamentals/errors/state` -> Teacher classifier。测试时临时允许 Teacher 读取 holdout 的 privileged columns，**仅用于测信息上限，不可部署**。

判断线：

- Teacher < 70%：当前 privileged 信息仍不足，70%需要新的数据源/新状态建模；
- Teacher 70~75%：存在可利用空间，但蒸馏难度较高；
- Teacher >= 75%：继续强攻 Student；
- Teacher >= 80%：说明 70% Student 具有较强信息基础。

### P2：Generalized Distillation Student

Teacher 用 privileged 信息训练；Student 只用 P6 regular X。使用 Teacher soft probability 构造软监督，并与 hard label 混合：

`soft_target = (1-alpha) * y + alpha * p_teacher`

alpha 只在训练/验证段选择，测试段冻结。第一轮保持模型简单，优先 LightGBM regressor/classifier，避免把收益混入大型架构变化。

### P3：Privileged ablation

分别移除 actual fundamentals、realized errors、DA state、D-1 evening state，确认 Teacher 的主要增益来自哪里。若某类 privileged 信息对 Teacher 很强但 Student 无法吸收，则下一轮把该类信息改造成 auxiliary task，例如 forecast-error prediction / evening-regime reconstruction。

## 评价口径

主指标：

- Direction Accuracy（第一目标）
- Positive Accuracy
- Negative Accuracy
- Balanced Direction Accuracy
- 1–8 / 9–16 / 17–24 分段方向准确率

最终目标：山东测试集 Direction Accuracy >= 70%。

所有结果必须明确标注：

- `deployable_student`：正式预测无需 privileged feature；
- `oracle_teacher`：仅上限诊断，不可上线；
- 训练区间、验证区间、测试区间；
- 山东数据快照、seed、regular/privileged feature list；
- 测试段不得参与 alpha / threshold / feature selection。

## 可迁移到论文的创新表达

若实验成功，可形成的科研叙事不是“训练时作弊”，而是：

> 面向提前出清场景中训练期/推理期信息不对称问题，构建基于市场实现态特权信息的 Teacher–Student 学习框架；Teacher 利用历史实现态的物理量、预测误差和中间市场状态形成更丰富监督，Student 在不增加部署期信息需求的前提下学习其软概率、状态或分布知识，从而提升提前时点的价差方向预测。

后续可以继续发展为：多任务 privileged distillation、regime distillation、forecast-error auxiliary supervision、period-aware distillation。

## 相关方法依据

- Karlsson et al., AISTATS 2022, *Using Time-Series Privileged Information for Provably Efficient Learning of Prediction Models*：训练阶段利用预测时点与未来结果之间的中间时间序列，测试阶段只使用 baseline 信息。https://proceedings.mlr.press/v151/k-a-karlsson22a.html
- Tian, Wen, Fu, Expert Systems with Applications 2024, *Multi-step ahead prediction of carbon price movement using time-series privileged information*, DOI: 10.1016/j.eswa.2024.124825：将 LUPI / generalized distillation 用于多步价格方向预测，与本项目的“未来中间信息训练可见、推理不可见”问题高度相似。
- Martínez-García et al., Knowledge-Based Systems 2025, *Teacher privileged distillation: How to deal with imperfect teachers?*, DOI: 10.1016/j.knosys.2025.113338：强调 privileged teacher 并非天然有效，需要判断 privileged feature 是否真正提供额外信息，并处理 imperfect teacher。

## 当前状态

2026-08-22：进入 Phase 1。先测 Oracle Teacher 信息上限，再做第一版 soft-target Student；仍完全位于实验区，未触碰正式 DA/RT/Spread 生产链路。
