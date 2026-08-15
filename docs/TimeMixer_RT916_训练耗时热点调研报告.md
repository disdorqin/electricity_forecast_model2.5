# TimeMixer 与 RT916_SpikeFusionNet 训练耗时热点与优化调研报告

> 项目背景：山东省电力现货价格预测（24点正式 + 96点辅助）。两个自研 GPU 模型（RTX3090）：
> - **TimeMixer**：MovingAvg(kernel=25) + PastDecomposableMixing（多尺度 MLP）+ 线性投影。入口 `TimeMixer/pipeline.py` / `repro_pipeline.py`。
> - **RT916_SpikeFusionNet**：TimesBlock（FFT 周期自适应 + 2D Inception 卷积）+ Spike 残差分支 + 动态门控。入口 `src/rt916_spikefusionnet/core.py` / `train.py`。
> 调研日期：2026-08-15。结论性质标注：**已实测**=代码直接可见/既有报告数字；**量级估计**=由论文基准+结构推导；**推断**=基于代码逻辑的合理外推。

---

## 0. 一句话画像（决定一切分析的前提）

两个都是**小参数、短序列、batch 不大**的模型（TimeMixer ~130k 参数、RT916 ~0.3-0.5M）。这种规模下 GPU 远未喂饱，**训练墙钟的主宰是「CPU 特征工程 + 每步 kernel launch/同步开销 + 重复训练份数」，而不是 GPU FLOPs**。

- TimeMixer 论文自有基准（A100, Appendix B Table 8）：seq 192→3072，s/iter 仅 0.007→0.016s、显存 1003→1411 MiB。纯 MLP 计算量**线性于序列长度且极小**。
- RT916 的 2D Inception 卷积作用在 `[B,64,T/p,p]` 小张量上，是 **latency/kernel-launch 主导**，GPU 利用率更低。

---

## 1. TimeMixer 耗时画像

### 1.1 架构（`TimeMixer/backbones.py`）
`past_proj(Linear)` → `make_scales`（avg_pool1d 逐级减半，3 级）→ 2×`PastDecomposableMixing`（每级 MovingAvg(kernel=25) 分解 seasonal/trend + 各 2×Linear(64,64) MLP + LayerNorm，跨尺度 interpolate）→ 池化 + future_mixer + 线性 head。全 MLP，无 FFT、无 conv。

### 1.2 关键超参与训练结构（`repro_pipeline.py:train_model` / `run_monthly_reproduction`）

| 项 | 值 | 出处 |
|---|---|---|
| epochs | **80**（pipeline.py 默认）；CLI 40；RunConfig 30 | `pipeline.py:59` |
| 早停 patience | 15 | `pipeline.py:60` |
| batch_size | **16** | `pipeline.py:61` |
| seq_len | **24点=96**、**96点=384**（4 天窗口） | `pipeline.py:66` `seq_len=4*res_n` |
| hidden_dim / blocks / scales | 64 / 2 / **3** | RunConfig |
| pred_len（OUTPUT_LEN 等价物） | 按段：24点每段 8、96点每段 32 | `_segments()` |
| 训练结构 | **segment_training=True → 3 段 × DA/RT = 6 个独立模型各自从头训** | `repro_pipeline.py:1834/1963` |
| 训练样本 | 12 个月滚动窗口（train ~288 天 + valid ~72 天） | `run_monthly_reproduction` |

### 1.3 加速底座检查
- ✅ AMP：autocast BF16（`repro_pipeline.py:1500-1504`），GradScaler 仅 FP16
- ✅ DataLoader：num_workers=4 + pin_memory + persistent_workers + prefetch_factor=2（L1376-1386），non_blocking 传输
- ✅ import 时打开 cudnn.benchmark + TF32（L28-33）
- ❌ `torch.compile`：未用
- ⚠️ **致命打架**：`set_seed()` → `utils/reproducibility.py:set_global_seed`，**L34 强制 `cudnn.benchmark=False`、L38 强制 `set_float32_matmul_precision("highest")`（关 TF32）**，运行前废掉上面两条开关。真正生效的只剩 autocast BF16。

### 1.4 耗时热点 Top3（24点，量级估计）
1. **6 个模型 × 最多 80 epoch 的重复训练**——墙钟主因子。每模型 ~18 step/epoch，6×80×18 ≈ 8600 step；GPU 步时 ~10-30ms，GPU 纯算合计 ~1.5-4 min/run。结构冗余是"同一网络 3 段各自从零训 + DA/RT 双份"。
2. **逐日 Python/pandas 特征工程重复构建**——`build_segment_arrays → make_sample → make_past_features/future_features/compute_blend_baseline` 对每天做 ~25 列 rolling/rank/reindex，DA、RT、每段各重算，每 run 6 次 × ~360 天。CPU 隐性大头。
3. **每 step 的 `.item()` 同步**（`train_model` L1537/1560）+ batch=16 的小启动开销。PyTorch AMP recipe 明令禁止每步 `.item()`。

### 1.5 耗时占比（24点，推断）
GPU 前向+反向 ~25-40%；CPU 特征工程+数组构建 ~30-45%；DataLoader/同步/overhead ~20-35%。96点 GPU 占比上升但 CPU 仍不低。

---

## 2. RT916_SpikeFusionNet 耗时画像

### 2.1 架构（`src/rt916_spikefusionnet/model.py` + `annual_model.py`）
`DataEmbedding` → **TimesBlock**（`FFT_for_Period` 用 `torch.fft.rfft` 找 top-k=2 周期 → 折叠成 2D `[B,64,T/p,p]` → 2×`InceptionBlockV1` 各 3 个 Conv2d(k=1,3,5)）→ norm → 线性头 + **SpikeResidualBranch**（3×dilated Conv1d）+ **CalendarRegimeGate**。训练用 `AnnualSpikeGatedTimesNet`（d_model=64、e_layers=1）。

### 2.2 关键超参（`core.py` CONFIG）
| 项 | 值 | 出处 |
|---|---|---|
| OUTPUT_LEN_LIST | **8**（24点）/ 32（96点） | CONFIG / `set_resolution` |
| INPUT_LEN_LIST | 8 | CONFIG |
| **seq_len** | **8×8+8 = 72**（24点）；32×8+32=**288**（96点） | `core.py:540` |
| D_MODEL / E_LAYERS / TOP_K / NUM_KERNELS | 64 / 1 / 2 / 3 | CONFIG |
| BATCH_SIZE / EPOCHS / PATIENCE | **64 / 12（96点8）/ 4** | CONFIG |
| LR / 优化器 | 3e-4 / AdamW + CosineAnnealing | `core.py:593-598` |
| 训练数据 | 12 个月窗口，`mod="all"` 训 3 个时段模型 | `_get_periods` |

### 2.3 加速底座检查
- ✅ AMP autocast BF16（`core.py:604-608`）
- ✅ DataLoader num_workers=4/pin/persistent/prefetch（L557-571）
- ✅ import 时 TF32 + benchmark（L37-46）
- ❌ torch.compile 未用
- ⚠️ **致命打架（比 TimeMixer 更狠）**：`core.py:set_seed`（L431-432）**无条件 `cudnn.deterministic=True` + `benchmark=False`**，每次 `train_single_period` 前调用；顶层 `pipeline.py:_apply_seed` 又走 `set_global_seed` → benchmark=False + TF32 关闭。RT916 是 conv 重型模型，被砍得最疼。

### 2.4 FFT 位置与开销
- 位置 1：`model.py:FFT_for_Period`（L7-42），每个 TimesBlock 前向一次 rfft。代码注释明确承认 **rfft 不支持 BF16，每次 `x.float()` 再转回**（L15-21）。
- 位置 2：`model.py:DynamicPeriodGate._period_features`（L189）在目标历史序列上**再来一次 rfft**。
- 开销：FFT 本身 O(T log T)、T=72 很小，绝对算力占比 ~5-15%（推断）；但**每次前向 2 处 × (BF16→FP32→BF16) 往返**，打断 autocast 连续区、无法进 tensor core、结构性阻止 CUDA graph 融合。费钱主要在破坏性而非绝对量。

### 2.5 耗时热点 Top3（24点，量级估计）
1. **TimesBlock 的 2D Inception 卷积小张量重复启动**——每 block 2×3=6 个 Conv2d 作用在 `[64,T/p,p]`（p≈8-24）矮胖小张量，加 k=2 循环内 `int(period_list[i].item())` 同步。latency 主导，GPU 利用率低。
2. **`cudnn.deterministic=True` + `benchmark=False` + TF32 被关** → conv 固定次优算法（估损 20-50%）。
3. **`ElectricityDataset.__init__` 逐窗 Python 循环**（`core.py:339-372`，TRAIN_STEPS=1 → 96点 ~3.4 万窗）+ 每 epoch `_validate` 全量 numpy 逆归一化（CPU）。一次性 CPU 大头 + 内存膨胀（96点 X ~1GB+）。

### 2.6 耗时占比（24点，推断）
GPU 前向+反向 ~35-50%；数据集构造+验证（CPU）~20-30%；同步/launch overhead ~15-25%；特征工程（pandas enrich）~10-15%。

---

## 3. 目标维度：24 点 vs 96 点对训练时间的影响

| 维度 | TimeMixer | RT916 |
|---|---|---|
| 24点 seq_len / 样本量 | 96 / ~290 天 | 72 / ~290 天÷3 |
| 96点 seq_len / 样本量 | 384（×4）/ ×4 | 288（×4）/ ×4 |
| 每步张量 | ×4 | ×4（2D 卷积变瘦长，利用率更差）|
| epochs | 不变（80）| 12→8（已补偿）|
| 96点相对 24点总训练墙钟（推断）| **×6~8** | **×3~5** |

结论：96点训练成本约为 24点 的 3-8 倍。若 96点 仅辅助口径，优化优先级必须 24点 在前。

---

## 4. 联网调研结论

### 4.1 TimeMixer 论文（ICLR 2024, arXiv:2405.14616）
- **scales 官方建议**：§4.2 "Analysis on number of scales" 明确——**长程 M=3，短程 M=1**。本项目每段预测 8/32 点（短程）却用 scales=3；多级混频 FLOPs=1+½+¼=1.75，降到 scales=1 省 **~43% PDM 计算**（head 输入也变小）。
- **blocks=2 是官方默认**；MovingAvg kernel=25 是官方默认；kernel 只影响精度不影响 FLOPs（AvgPool1d 为 O(T)）。
- **复杂度线性于 T**：Table 8（A100）seq 192→3072 s/iter 0.007→0.016、显存近平。减序列换速度收益有限，真正时间在训练份数/epoch/CPU 侧。

### 4.2 TimesBlock / FFT 复杂度
- TimesNet（ICLR 2023, arXiv:2210.02186）：rfft 找周期 O(T log T) + 2D inception 卷积 O(T)。官方周期是 batch 共享的 top-k；本项目 RT916 同样 batch 共享（`period_list` 形状 [k]，循环 k=2 次，正确）。
- 序列长 T 翻倍：FFT 微涨、2D 卷积行数翻倍（线性）；真正的伤是**周期变化后 reshape 形状变化**——cudnn benchmark 无法缓存、CUDA graph 无法捕获。
- **BF16 硬伤（已核实）**：`torch.fft.rfft` 不支持 bfloat16（PyTorch #117844/#139313），必须回退 FP32。

### 4.3 PyTorch 通用加速法（对照现状）
| 手段 | 收益依据 | 本项目现状 |
|---|---|---|
| AMP BF16 | RTX3090 BF16 Tensor ~71 TFLOPS vs FP32 35.58 → 理论 2×，实测 1.5-2× | ✅ 已接入（被 seed 副作用拖后腿）|
| torch.compile | 小 batch 下 eager dispatch CPU 开销可达 step 的 20-30%；compile 可省 15-40% | ❌ 未用；TimeMixer 是理想对象 |
| CUDA Graph (reduce-overhead) | 消除每步 launch；固定 shape 收益最大 | 与 RT916 动态 reshape 冲突 |
| 放大 batch + LR 缩放 | 减少更新次数、喂饱 GPU（Smith ICLR 2018）| batch 16/64 偏小可试 |
| 早停 / warm-start 续训 | 214 天回测相邻日高度相关，续训用极少 epoch（既有调研已量化）| 早停✅；warm-start 待接入 |

### 4.4 RTX3090 可达吞吐
- 算力：FP32 35.58 TFLOPS；TF32 Tensor 35.58（dense）；FP16/BF16 Tensor **~71 TFLOPS（dense，142 为稀疏营销数）**；带宽 936 GB/s；约为 A100 张量算力的 1/4-1/5。
- TimeMixer A100 0.007 s/iter（seq192）→ 本项目 24点（seq96, batch16）3090 单步约 **10-30ms**；96点（seq384）约 **30-100ms**（量级估计）。
- 24点单日全链路 ~10-25 min（6 模型 GPU 纯时 1.5-4 min + CPU/同步），与既有"单日 17 分钟、214 天 3 天"报告吻合。

---

## 5. 优化建议（带依据）

### TimeMixer
1. **修 `set_global_seed` 副作用**：cudnn.benchmark 恢复 True、float32_matmul_precision 改 `"high"`（TF32 开），或加 `OPTIM_*` 开关。零精度影响，估 10-30%。
2. **`torch.compile(model, mode="reduce-overhead")`**：全 MLP、固定 shape、无 graph break。估 15-40% 步时。
3. **去每步 `.item()` 同步**：loss 累计改 `(loss*len(yb)).sum()` 或每 epoch 同步一次。零成本。
4. **scales 3→1（或 2）**：论文短程建议 M=1，PDM 省 ~43% FLOPs。需验证集复验精度。
5. **特征预计算共享 + warm-start 续训**：既有 FeatureStore 方案落地，214 天省 10-25h（已量化）。
6. **segment_training 权衡**：6 份独立训练是最大结构冗余；可只训 DA 全段 + RT 续训，或 3 段共享主干。

### RT916
1. **删 `set_seed` 里 `cudnn.deterministic=True` / `benchmark=False` 硬编码**（加开关）——conv 重型模型最伤处，估 20-50%。
2. **FFT 局部 FP32 隔离**：`torch.autocast(enabled=False)` 包住 FFT 段，避免每前向 2 处 dtype 往返；或固定 top-k 周期后每 epoch 只算一次。
3. **`ElectricityDataset` 向量化**：`sliding_window_view` / `torch.Tensor.unfold` 替代逐窗 Python 循环。
4. **compile 只包卷积主干**（FFT/动态 reshape 段排除），或固定周期后整体 compile。
5. **batch 64→128/256 + LR 按 √2 缩放**：喂饱 2D 卷积。
6. 96点优先保 24点；EPOCHS 8 + 早停 4 已兜底。

---

## 6. RTX3090 上把训练时间减半——最优先 5 件事

| # | 动作 | 为什么最优先 | 估收益 | 风险 |
|---|---|---|---|---|
| 1 | **修复 seed/确定性开关破坏加速底座**（`set_global_seed` 恢复 benchmark+TF32；RT916 去 `deterministic=True`）| 3 行改动、零精度损失，解掉"开了 AMP 被二次关闭" | 20-40% | 极低 |
| 2 | **特征一次性预计算 + 共享 FeatureStore + 逐日切片** | CPU 最大隐性大头，既有调研量化省 10-25h | 省 30-70% 前置 CPU | 低 |
| 3 | **Warm-start 续训**（前一日权重初始化 + 5-20 epoch 替代 80/12 全量）| 214 天回测相邻日高度相关，砍 epoch = 直接减半墙钟 | 40-60% 总时 | 中（需防泄漏验证，已有方案）|
| 4 | **TimeMixer `torch.compile(mode="reduce-overhead")` + 每 epoch 聚合 loss** | 全 MLP 是 compile 甜点；去同步提整条流水线 | 20-40% GPU 步时 | 低 |
| 5 | **RT916 专项**：恢复非确定性 conv + FFT FP32 隔离 + 数据集向量化（+可选固定周期启 CUDA graph）| 三点叠加直接压 RT916 步时与构造耗时 | 30-50% | 中（需回归验证）|

> 1、3 叠加即可把 24点 单日 ~17 分钟压到 ~5-7 分钟（量级估计）；1-5 全做，214 天回测从 ~3 天降到 ~1-1.5 天可达。所有加速都必须在验证集复验 SMAPE（生产公式：值<50 先 clip 到 50），防"快而不准"。

---

## 7. 参考资料
1. Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*, ICLR 2024, arXiv:2405.14616（含 Appendix B 效率表、scales 分析）
2. Wu et al., *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*, ICLR 2023, arXiv:2210.02186
3. Smith et al., *Don't Decay the Learning Rate, Increase the Batch Size*, ICLR 2018, arXiv:1711.00489
4. PyTorch issue #117844 / #139313（rfft 不支持 bfloat16）
5. NVIDIA RTX 3090 规格（techpowerup/waredb）：FP32 35.58 / TF32 Tensor 35.58 / FP16·BF16 Tensor ~71 TFLOPS（dense）
6. Spheron torch.compile/CUDA Graph 实测（eager dispatch CPU 开销小 batch 达 20-30%）
7. 本项目既有报告：`docs/工业界时序预测训练加速与精度提升调研报告.md`、`docs/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`

> 注：凡给出具体百分比/时长的地方为量级估计（依据论文基准 + 结构推导），落地前请以 1-2 天小规模 A/B 实测校准。
