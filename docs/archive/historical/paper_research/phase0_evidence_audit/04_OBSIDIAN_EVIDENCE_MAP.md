# Obsidian 证据地图

Vault：`D:\science\load_forecast\disdorqin_load_forecast`。本轮只读扫描 778 个文件，其中 398 个 Markdown、6 个 PDF；未做广泛外部检索，未补写 Vault。

## 1. 总体诊断

- **VERIFIED**：`科研总目录/99_系统数据/07_Review_Matrix_综述矩阵/review_matrix_latest.md:1-46` 的主题是 electric **load** forecasting；矩阵中仅一项明确为 day-ahead electricity wholesale **price** forecasting（LT-Conformer，line 43）。
- **VERIFIED**：检索未找到英文 `negative electricity price`/`negative price` 专门论文条目；中文“负电价”命中主要是项目/会议笔记，不是原始论文。
- **VERIFIED**：唯一在本轮按 PDF 原文核验、且与在线漂移相关的附件是 Jagait et al. (2021)，但任务是四户住宅负荷，不是电价或极端价格。
- **VERIFIED**：索引 `第一轮主题文献索引.md:3` 明示“有 PDF 待精读、无 PDF 仅候选”。因此 metadata/abstract 笔记不得升级为 paper-supported claim。
- **VERIFIED**：矩阵将 *A review on extreme learning machine* 放入“尖峰/极端”主题，但 ELM 的 “extreme” 是算法名；该项与极端价格事件没有直接证据关系。

## 2. 去重论文矩阵

| ID | Title / metadata | Task, data, horizon | Method / claimed contribution | Positive spike / negative price | Model scope / protocol / limitations | Source and strength |
|---|---|---|---|---|---|---|
| P1 | *Load Forecasting Under Concept Drift: Online Ensemble Learning With Recurrent Neural Network and ARIMA*. Rashpinder Kaur Jagait, Mohammad Navid Fekri, Katarina Grolinger, Syed Mir, 2021, IEEE Access, DOI 10.1109/ACCESS.2021.3095420 | Residential load; four proprietary London Hydro homes; hourly, 3 years/home, 25,560 readings; exact forecast horizon not recovered | rolling ARIMA + Online Adaptive RNN; average/weighted/squared/model-switching ensemble | Not electricity price; error spikes around drift only | Model-specific ensemble; chronological data stream. Authors state they did not quantify drift or performance during drift; future work on metrics (PDF p.15) | **VERIFIED original PDF**. Note under `.../02_概念漂移与在线学习/Load Forecasting Under Concept Drift_...md`; PDF `科研总目录/99_系统数据/04_OpenAccess_PDFs_开放获取PDF/10.1109_access.2021.3095420.pdf`, pp.1,9,15 |
| P2 | *Learning to Extrapolate and Adjust: Two-Stage Meta-Learning for Concept Drift in Online Time Series Forecasting*. Weiqiu Chen et al., 2025, DOI 10.24963/ijcai.2025/542 | Abstract says multiple time series and three released electric-load datasets; horizons/markets unknown | LEAF: latent extrapolation for macro-drift + meta-learned surrogate loss adjustment for micro-drift; abstract calls it compatible with deep predictors | Not price; no signed tails in local evidence | Abstract claims model-agnostic across deep prediction models; evaluation/limitations not checked | **DOCUMENTED BUT NOT VERIFIED** metadata+abstract. `.../02_概念漂移与在线学习/Learning to Extrapolate and Adjust_...md:1-46`; no local PDF |
| P3 | *An Online Probability Density Load Forecasting Against Concept Drift Under Anomalous Events*. Chaojin Cao, Yaoyao He, 2024, IEEE TII, DOI 10.1109/TII.2023.3331076 | Load probability density under anomalous events; local note lacks full datasets/horizon | Note prompts inspection of quantile/CRPS/online adaptation; original method details not locally verified | Anomalous load, not electricity price; signed price tails absent | Model specificity and limitations unknown without PDF | **DOCUMENTED BUT NOT VERIFIED**. `.../02_概念漂移与在线学习/An Online Probability Density Load Forecasting Against Concept Drift Under Anomalous Events (...md:1-46`; supplementary `.pdf.md` is a duplicate record, not a verified attachment |
| P4 | *Online decoupling feature framework for optimal probabilistic load forecasting in concept drift environments*. Chaojin Cao, Yaoyao He, Xiaodong Yang, 2025, Applied Energy, DOI 10.1016/j.apenergy.2025.125952 | Probabilistic load forecasting; datasets/horizon unknown | Title/metadata suggest online decoupling; note contains only questions, no verified method | Not price | Protocol/limitations unknown | **DOCUMENTED BUT NOT VERIFIED**. `.../02_概念漂移与在线学习/Online decoupling feature framework...md:1-46`; no PDF |
| P5 | *Benchmarking Transformer Variants for Hour-Ahead PV Forecasting: PatchTST with Adaptive Conformal Inference*. Vishnu Suresh, 2025, Energies, DOI 10.3390/en18185000 | Hour-ahead PV, datasets unknown locally | Title says PatchTST + Adaptive Conformal Inference; note explicitly says method/experiments pending | Not electricity price | Conformal relevance only; limitations unknown | **DOCUMENTED BUT NOT VERIFIED**. `.../02_概念漂移与在线学习/Benchmarking Transformer Variants...md:1-45`; no PDF |
| P6 | *Leveraging Temporal Dependency in Probabilistic Electric Load Forecasting*. Yunyi Zhang, Yaoli Zhang, Ye Tian, 2024, SSRN, DOI 10.2139/ssrn.4944658 | Probabilistic load; horizon/data unknown | Title-level temporal dependence; note asks about Pinball/CRPS/PICP | Not price | Protocol/limitations unknown | **DOCUMENTED BUT NOT VERIFIED**. `.../04_概率预测与预测区间/Leveraging Temporal Dependency...md:1-43`; no PDF |
| P7 | *A Local-Temporal Convolutional Transformer for Day-Ahead Electricity Wholesale Price Forecasting*. authors absent in local note, 2025, DOI 10.3390/su17125533 | Day-ahead wholesale price; abstract says initial Australian market evaluation | LT-Conformer: Local-Temporal 1D Conv plus global temporal and cross-feature attention | Abstract cites volatility/rapid shifts, but does not mention signed positive spikes or negative prices | Model-specific architecture; split, exact horizon, datasets and author-stated limitations not checked | **DOCUMENTED BUT NOT VERIFIED** abstract. `.../06_其他候选文献/A Local-Temporal Convolutional Transformer...md:1-46`; no PDF |
| P8 | *A review on extreme learning machine*. 2021, DOI 10.1007/s11042-021-11007-7 | Review of ELM algorithm | Algorithm review | No evidence of price extremes | Irrelevant terminology collision | **VERIFIED as misclassification**, metadata at `review_matrix_latest.md:15`; original paper not audited |
| P9 | *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. 2021, AAAI, DOI 10.1609/aaai.v35i12.17325 | Generic long-sequence forecasting | ProbSparse attention/generative decoder (not re-extracted here) | No local price-tail evidence | Architecture baseline, not extreme-price method | **DOCUMENTED BUT NOT VERIFIED for this research claim**; local PDF exists but not directly relevant to Phase-0 extreme evidence |

No externally resolved bibliographic field was used. Venue names inferred from local note/DOI metadata are not treated as original-paper method evidence.

## 3. 原论文、笔记解释与未验证设想分层

### A. 原论文直接支持

- **VERIFIED** P1 PDF p.9：四户 proprietary residential hourly load、三年、每户 25,560 readings，目标是 energy consumption，含 calendar/weather features。
- **VERIFIED** P1 PDF p.15：作者明确未量化 concept drift，也未量化 drift period 的性能；这直接限制把其作为“异常事件可靠改善”证据。
- **VERIFIED** P1 PDF p.1/5-8：在线 adaptive RNN 与 rolling ARIMA 的多种 ensemble aggregation；这是 online ensemble 先例，不是 signed-price-tail correction 先例。

### B. Obsidian 笔记的解释

- **DOCUMENTED BUT NOT VERIFIED**：P2-P7 的“可借鉴 online update、Pinball/CRPS、尖峰区间校准、残差融合”等句子是笔记的阅读提示；各 note 自己标明待获取 PDF/待核对方法与实验。
- **DOCUMENTED BUT NOT VERIFIED**：`模型复现与测试.md:7-10` 记录 LightGBM 负价概率阈值与下压规则；它是工程笔记，需以当前代码为准。
- **DOCUMENTED BUT NOT VERIFIED**：`03_会议/01_当天汇报/spiketimesnet.md:7-38` 将 SpikeResidualBranch、DynamicPeriodGate、Tail-Weighted Loss 描述为创新；本审计只承认代码已实现，不承认新颖性。

### C. 用户自拟/未验证 ideas

| Idea | Path / lines | Evidence status | 与 tentative hypothesis 的关系 |
|---|---|---|---|
| 多基础模型滚动 OOS 残差库 + model identity + 通用残差模块 + 稀疏极端专家 | `科研总目录/01_大创/02_模型与方法/融合链路/残差模块.md:2` | UNVERIFIED IDEA | 高度重叠；尚无代码/实验引用 |
| 时段路由/MoE、多尺度卷积、共享 feature attention | `科研总目录/02_科研/02_科研创新点/spiketimesnet.md:9-27` | UNVERIFIED IDEA | routing/expert overlap；笔记同时写“有些累赘” |
| EquiFreqFormer 频域异构专家 + hierarchical gated residual correction | `科研总目录/02_科研/01_论文阅读/发表/EquiFreqFormer.md:3-20,49-54,143-181` | UNVERIFIED USER DESIGN | residual/gating/uncertainty proxy overlap；“首创”措辞无查新支持 |
| event-level metric 可能改变模型排名 | `科研总目录/02_科研/03_研究问题与假设/事件级评价指标会不会改变我们对最佳模型的判断？.md:12` | UNVERIFIED QUESTION | 应转成预注册可证伪问题 |
| normalization 可能压平尖峰 | `.../数据归一化是否在无意中压平了尖峰信号？.md:21` | UNVERIFIED QUESTION | 需要固定模型/seed 的 ablation |

### D. 重复与误分类

- **VERIFIED** P3 主记录和 `...supp1-3331076.pdf.md` 指向同一 DOI family，应去重；后者不是实际 PDF 证据。
- **VERIFIED** 主题索引、主题总览、review matrix 和具体 note 对同一论文的链接不应算多篇证据。
- **VERIFIED** P8 是 ELM 算法而非 extreme-event forecasting；“peak load”论文也是负荷峰值，不等于 price spike。
- **INFERRED** `发表/EquiFreqFormer.md` 的路径名可能造成“已发表”误读；文件内容呈自拟架构，未含论文 DOI/venue/实验，故本审计按 user-authored design 处理。

## 4. 文献覆盖缺口

1. **UNKNOWN**：电价正尖峰与负电价是否需要不同生成机制、标签和评价，Vault 无原论文证据。
2. **UNKNOWN**：已有 model-agnostic post-hoc correction、selective prediction/reject option、safe/no-harm calibration 的直接先例。
3. **UNKNOWN**：电价 spike forecasting 中 EVT、regime switching、Hawkes/jump、mixture-of-experts、two-part/hurdle models 的最近结果。
4. **UNKNOWN**：negative price 专用 probabilistic/quantile/conformal 评价如何处理 near-zero 与符号翻转。
5. **UNKNOWN**：跨市场公开 benchmark 的统一 forecast-origin 与 exogenous as-of contract。
6. **UNKNOWN**：P7 是否处理负价、事件 recall、严格 rolling split；需要原文，不可从摘要补全。
7. **UNKNOWN**：P2 LEAF 所称 model-agnostic 是否只适用于 deep models，以及其 OOS residual/更新时序是否适合 post-hoc electricity-price correction。

这些缺口是下一阶段系统文献搜索的输入，不是新颖性结论。
