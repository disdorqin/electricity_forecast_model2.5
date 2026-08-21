# 深度调研：PyTorch 底层算子、CUDA 编程与自定义算子如何加速训练

> 日期：2026-08-15 ｜ 环境：GPU 云服务器 RTX 3090（torch 2.9.0 + CUDA 12.8，见 `docs/AUTOMATED_ITERATION_LOOP.md`）
> 目标：本项目两个自研 GPU 模型 **TimeMixer**、**RT916** 训练慢，评估"底层算子 / CUDA / Triton / C++ 扩展"这条路的真实收益与落地路径。
> 已对照代码：`TimeMixer/backbones.py`、`RT916_SpikeFusionNet/src/rt916_spikefusionnet/model.py`、`repro_pipeline.py`、`core.py`。
> 关联文档：`docs/工业界时序预测训练加速与精度提升调研报告.md`（高层的 AMP/特征预计算/warm-start 结论）、`docs/CPU_GPU_异构并行_调研报告.md`（CPU-GPU 并行，结论=模型内算子搬 CPU 收益≈0，正解=torch.compile/CUDA Graph）。本文聚焦"算子内部"这一层，与之互补，不重复。

---

## 0. 一句话结论（TL;DR）

1. **两个模型的真瓶颈不是"算力"，是"kernel 数量 + Python 调度 + CPU-GPU 同步"**。它们都是小型模型（hidden=64/128，batch=16~64，seq_len=168~288），每个 forward 有几十上百个微秒级 kernel。RTX 3090 对这种负载只能发挥个位数百分比的算力（memory/launch bound，而非 compute bound）。
2. **优先级（按收益/成本比）**：
   - **P0（零代码，半天）**：用 `torch.profiler` 实测确认瓶颈 → 确认 AMP(bfloat16)+TF32+cudnn.benchmark 已开（本项目已默认开，见 §1.5）。
   - **P1（1~2 天）**：`torch.compile(model, mode="reduce-overhead")` 自动算子融合 + CUDA Graph；并把 RT916 `TimesBlock` 里的 **`.item()` CPU-GPU 同步 + Python for 循环** 改掉（单点收益最大，§3.2a）。
   - **P2（2~4 天）**：Triton 写 2~3 个 fused kernel（§6）。**不需要写 C++/CUDA 扩展**，Triton 覆盖本项目 90% 需求。
   - P3（不推荐现在做）：手写 C++/CUDA。只有 Triton 表达不了、且收益能证明时才值得（§2.4）。
3. **预期收益**（按 NVIDIA/PyTorch 官方基准外推，需用本项目数据实测）：torch.compile 对 launch-bound 小模型常见 **1.3~2.5×**；CUDA Graph 再削掉 launch 开销；Triton 把 3 个热点各自再 **2~4×**。叠加训练日级双缓冲（已有报告），从"3 天→1 天"完全可行。

---

## 1. PyTorch 算子执行机制

### 1.1 张量运算如何 dispatch 到 CUDA kernel；kernel launch 开销

**机制**：eager 模式下每个张量算子 = 一次 CPU 端 dispatch（经过 `TensorIterator`/`op key` 分发）+ 一次 CUDA kernel 启动。GPU 上是异步队列，CPU 把 kernel 排进 stream 就返回继续（`torch.cuda` 异步执行语义 [1]）。

**launch 开销有多真实**：NVIDIA 官方用"每步 20 个微秒级 kernel"做了量化实验 [2]：

| 场景 | 每 kernel 有效耗时 |
|---|---|
| kernel 本体执行时间 | **2.9 μs** |
| 每次 launch + 同步 | **9.6 μs**（2.9 本体 + 6.7 开销） |
| 去掉每 kernel 同步、允许 launch 与执行重叠 | **3.8 μs**（还有 ~1μs 启动开销） |
| CUDA Graph 整图单次启动 | **3.4 μs** |

**对本项目的含义**：RT916 一个 forward 光 `TimesBlock` 就含 2×6 个 Conv2d + 若干 reshape/permute/stack/pad（§3.2）。这些 kernel 都是 1~10μs 级。**launch/调度开销占比常超过 50%**。所以"减少 kernel 数量"和"减少 CPU-GPU 同步"比"让某个 kernel 更快"更有效。

**反模式清单**（PyTorch 官方 tuning guide [3] 明列，对照本项目）：
- `tensor.item()`、`print(cuda_tensor)`、`nonzero()`、依赖 GPU 结果的 Python 分支 → 强制同步（RT916 中招，见 §3.2a）。
- 每 op 一 kernel 的 elementwise 链（`x - mean`、`std`、`z-score`、`sigmoid`…）→ 应融合（RT916 Spike 分支中招）。

### 1.2 torch.compile / Inductor 如何融合算子（减少 kernel launch）

**原理**：`torch.compile` = TorchDynamo 用字节码级追踪把一段 Python 函数编译成 FX 图 → Inductor 做算子融合 → 对 CUDA 生成 **Triton kernel**（并可选启用 CUDA Graph）[4][5]。

- **点运算融合**：连续的 elementwise（+、×、GELU、Dropout、LayerNorm 等）合成一个 kernel，数据只读/写一次全局内存（官方 tuning guide [3] 明确把"Fuse operations"列为通用优化第一条）。
- **reduction 融合**：`abs(x).sum()` 这类"变换+归约"会被融合成一个两阶段归约 kernel，中间 buffer 不落地（NVIDIA 2026 kernel fusion 官方文 [6] 给了 3GB→1GB 内存流量、3× 加速的例子）。
- **graph break**：遇到不支持的结构（如 Python `if tensor.sum() > 0`、`for` 遍历 GPU 标量）会截断成多张图，丢掉融合机会 [4][7]。**RT916 的 `TimesBlock` 就是这种结构**（§3.2a）。
- 实测：官方教程 4096×4096 点运算链 ~2.5× [7]；Inductor 对小模型的收益主要在"少 launch + 少内存往返"，与大模型（compute bound）不同。

### 1.3 CUDA Graph（图捕获重放）原理与适用场景

**原理**：把一段**形状固定**的 kernel 序列用 stream capture 捕获成 `cudaGraph_t` → `cudaGraphInstantiate` 预生成执行实例 → 之后每次只发一次 `cudaGraphLaunch`。CPU 启动开销从"每 kernel 一次"变成"整图一次" [2]。NVIDIA 官方文 [2]：图创建约 400μs 一次性，重放时每 kernel 有效成本 2.9μs→3.4μs（含启动）；叠加"内核体融合"（§2.3）是两种互补的融合层次 [6]。

**适用场景**（PyTorch 官方 tuning guide [3] 与 CUDA semantics [1]）：
- 训练 loop 内 **shape/内存地址不变**的重复计算 → 最理想。本项目滚动训练是"固定 batch/seq/epoch 多次迭代"，天然匹配。
- **约束**：捕获期间不能分配动态尺寸内存、不能有数据依赖的 CPU 分支；输入需要静态 buffer（PyTorch 侧用 `torch.cuda.graphs` 的 pool + 静态输入，或直接交给 `torch.compile(mode="reduce-overhead"/"max-autotune")` 自动启用）[1][3]。
- RTX 3090 是 Ampere，CUDA Graph 从 CUDA 10 起就支持，无硬件门槛。

### 1.4 torch.backends.cudnn.benchmark / TF32 / bf16 的实际收益

- **cudnn.benchmark=True**：cuDNN 对固定输入尺寸做卷积算法 autotune（跑几轮选最快的），之后一直复用 [3]。本项目已开（`repro_pipeline.py:32-33`、`core.py:43-44`）。注意：**决定不可复现**、且 conv 尺寸频繁变化时反伤，本项目固定 shape 是纯收益。
- **TF32**：在 Ampere+ 用 Tensor Core 跑 fp32 的 matmul/conv，**输入截到 10-bit 尾数，fp32 累加**。PyTorch 官方 [1]：A100 上大矩阵乘 **~7× 加速**，相对误差 ~2e-3（比全精度高约 2 个量级）。PyTorch 1.12+ 默认 **关闭** matmul TF32，需显式开。本项目已在 `core.py:40-42` 开启。⚠️ **但本项目矩阵太小（hidden 64~128），TF32 加速远达不到 7×，别抱高期待**；它的价值主要在把 fp32 的 Linear/Conv 从 CUDA 核换成 Tensor Core 核。
- **bf16 AMP**：本项目已默认（`perf_knobs.py` / `core.py:605-607` / `repro_pipeline.py:1501-1503`，bf16 + GradScaler 未用因 bf16 不需 scaler）。官方 tuning guide [3]：AMP + Tensor Core 对 Volta+ 整体可达 ~3×。本项目瓶颈不在 matmul，**AMP 已吃到，剩下的空间在融合与减 launch**。
- ⚠️ **bf16 陷阱（本项目已踩）**：`torch.fft` **不支持 bfloat16**，RT916 的 `FFT_for_Period` 必须 `x.float()` 再算再转回（`model.py:16-21`），多一次全张量 cast + 一次往返。见 §6 候选 3。

---

## 2. 自定义算子（C++/CUDA 扩展）与 Triton

### 2.1 torch.utils.cpp_extension 写 C++/CUDA 扩展的完整流程

按 PyTorch 官方 "Extending PyTorch" [8] 与自定义算子 landing page：

1. **写 C++ 算子 + CUDA kernel**：`forward/backward` 用 `at::Tensor`/`cudaMemsetAsync` 等，把 `__global__` 内核包进 `AT_CUDA_KERNEL_LAUNCH`。
2. **注册**：`TORCH_LIBRARY(mylib, m) { m.def("my_op(Tensor, int) -> Tensor"); }` + `TORCH_LIBRARY_IMPL`（或 `m.impl("my_op", my_op_cuda)`）。也可纯 Python 侧用 `torch.library.custom_op` 注册（新版更简单）[8]。
3. **构建**：`torch.utils.cpp_extension.load_inline(name=..., cpp_sources=..., cuda_sources=...)`（JIT 一次编译，开发期）或 `build_ext`/`setup.py`（发布期）[8]。
4. **接入 autograd**：写 `torch.autograd.Function` 包一层 `forward`/`backward`，`save_for_backward` 保存中间量，用 `torch.autograd.gradcheck` 验证梯度 [8]。
5. **验收纪律**：与 PyTorch 参考实现做数值对照 + 基准（Triton 教程 [9] 的标准做法是 `triton.testing.do_bench` 对比）。

**哪些场景值得写自定义算子**（[8] 的官方口径 + [6]）：
- 想**融合**多个算子成一个 kernel、省掉中间 tensor 的全局内存往返（memory-bound 场景收益最直接，NVIDIA 3× 例子 [6]）；
- 想在 **backward 少存 buffer**（`Function` + `setup_context`）省显存；
- 现有算子库表达不了的算法（自定义 scan、跨 stream、cuFFT 集成等）。

**不值得写的场景**：能用 `torch.compile` 自动融合的、能用 Triton 20 行写出来的 → 别写 C++/CUDA（编译环境 MSVC+CUDA Toolkit、维护成本、平台耦合全都要你来扛）。

### 2.2 Triton（OpenAI）—— 比手写 CUDA 简单得多

**本质**：Python 写的 CUDA DSL，`@triton.jit` 内核 + `tl.load/store/arange/program_id`，**编译器自动做 tile/block 调度、共享内存分配、寄存器重用**。官方定位就是"能写出接近峰值性能的自定义 DNN kernel 又不需要手写 CUDA" [9][10]。

- 学 20 行就能写出 vector add（官方教程 [9]）；本项目需要的滑动窗口均值、归一化、稀疏 mask 都是 Triton 的标准菜（官方有 fused softmax / layer-norm / attention 教程 [9]）。
- **与 torch.compile 的关系**：Inductor 的 CUDA 后端**就是生成 Triton kernel**；你手写的 `@triton.jit` kernel 在 torch.compile 图里会作为 **opaque 调用（graph break 边界）**，但可以直接用在训练循环里（绕开 compile，自己管 autograd 的 `Function` 包装）。Pytorch 2.5+ 也支持把自定义 triton kernel 注册进 Inductor（`@torch.library.custom_op` + triton 内核），让它参与编译。
- **注意**：Triton 数值上按 fp32 accumulate 但顺序可能与 cuBLAS/cuDNN 不同 → 与本项目 golden baseline 对比时允许小 epsilon 差异，但**必须逐位对齐生产验证逻辑**（本项目有"黄金基线逐字节一致"纪律，需为该报告新增"数值等价容差"验收标准，见 §5 验收）。
- **安装**：服务器 `pip install triton`（或随 torch 2.x 自带）；纯 Python 包，无编译环境要求。本机 CPU-only 的 `epf-2` 装不了也不该跑。

### 2.3 有没有现成的 fused kernel 库可用

| 库 | 覆盖 | 本项目价值 |
|---|---|---|
| **torch.compile / Inductor** | 自动点融合 + 归约融合 + CUDA Graph | ⭐ 最大，零代码 |
| **torch.optim Adam(..., fused=True)** | 优化器 step 融合（Adam 的 4 次 elementwise 合一 kernel） | ⭐ 一行代码、纯收益（本项目用 Adam） |
| flash-attn / flashattn2 | 融合注意力 | ✗ 本项目无自注意力 |
| apex | FusedAdam / FusedLayerNorm 等 | ⚠️ Windows/新 torch 兼容差，不建议，torch 内置 `fused=True` 已覆盖 |
| torchao / CUTLASS / cuDNN v8 API | 低层算子 | ✗ 过度设计 |

结论：**本项目没有 transformer/注意力结构，flash-attn 类用不上；最相关的现成融合是 Adam `fused=True` + torch.compile 自动点融合**。

### 2.4 C++/CUDA 什么时候才值得（回应用户"实在不行 C++ 也可以"）

判断清单（满足其一才值得写 C++/CUDA）：
1. Triton 表达不了：如需要与 cuFFT 在**同一个 kernel 内**结合（Triton 不提供 FFT 原语）、需要多 stream / 精确内存布局 / NCCL 集成；
2. 融合收益已用 `torch.compile`+Triton 证明过，但还有 10~20% 极致差距，且该 kernel 是热点；
3. 需要接入非 PyTorch 生态的 C/C++ 库。

**本项目现状不满足任一条** → 结论：**Triton + torch.compile 是甜点区，C++/CUDA 只做 P3 备胎**。

---

## 3. 具体到本项目两个模型

> 事实先行（已读源码，非臆断）：
> - **TimeMixer**（`TimeMixer/backbones.py`）：`past_proj`=Linear(past_dim→64)；`PastDecomposableMixing` ×2 blocks，每 block 3 个 scale，每个 scale 做 `MovingAvg(k=25)` 分解 + season/trend 两个 `Linear→GELU→Dropout→Linear` + LayerNorm，scale 间用 `F.interpolate` 拼接前一层 s/t；`make_scales` 连续 avg_pool1d；head 是 `Linear(64*(3+1)→64)→GELU→…→Linear→24`。训练配置：batch=16, seq=168, epochs=30。
> - **RT916**（`model.py`，生产入口 `core.py`）：`DataEmbedding`(num_variates→d_model=128) → `TimesBlock`×2，每块内 `FFT_for_Period`(rfft→abs→mean→topk) 选出 3 个周期，**Python for 循环逐周期**做 pad→reshape→permute→`InceptionBlockV1`(6 个 Conv2d + stack+mean)×2→reshape，再按周期权重 softmax 加权；另有 `SpikeResidualBranch`（3 个 dilated Conv1d + LayerNorm）与 `DynamicPeriodGate`（又一处 `torch.fft.rfft` + entropy/spike 统计）。训练：seq_len=288（=8×32+8，`core.py:540`），pred_len=8，d_model=128。

### 3.1 TimeMixer：MovingAvg（AvgPool1d）能否更快 + MLP 链

**MovingAvg 现状**：`nn.AvgPool1d(k=25, stride=1, pad=12)`（`backbones.py:8-24`）。这是**滑动窗口均值**，PyTorch 有专门 CUDA pool kernel，本身不差；但 stride=1 意味着窗口重叠计算（每输出 25 次加法）。理论更快方案：
- **前缀和（cumsum）**：`prefix = cumsum(x); trend[t] = (prefix[t+12]-prefix[t-13])/25`，把每个输出从 O(k) 降到 O(1)。**但** cumsum 引入大跨幅累加误差（fp32 下 ~1e-6 级），且要额外一个 scan kernel；对 168/288 长度收益有限。**不建议为了它写算子**。
- **正确姿势**：把"窗口均值 + seasonal=x-trend"合并进 Triton 一个 kernel（每个 program 一行，寄存器内滑动窗口累加）→ **1 个 kernel 取代 avg_pool(transpose 前) + transpose×2 + 截断 + 相减 ≈ 5 个 kernel**。这正是 §6 候选 1。
- **MLP 链（真正值得 fusion 的部分）**：season/trend 两个 `Linear→GELU→Dropout→Linear` + `x+y` + LayerNorm，都是小矩阵/点运算。`torch.compile` 能把其中 elementwise 部分（GELU→Dropout→add→LayerNorm）合成；Linear 走 cuBLAS(TF32)。**注意 torch.compile 对 `F.interpolate` + 跨 scale 依赖也能融合**，但 `PastDecomposableMixing.forward` 的 list 结构需要 dynamo 支持（PyTorch 2.x 已支持 list 传递，实测可跑）。

### 3.2 RT916：FFT 序列操作能否融合/优化 + "小而多 kernel"确认

**(a) `.item()` CPU-GPU 同步 + Python 循环 —— 本模型最大单点问题**（`model.py:99-101`）：

```python
for i in range(period_list.shape[0]):        # 每 forward 固定 3 次
    period = int(period_list[i].item())      # ← GPU→CPU 同步，卡死流水线
    ...
```

每次 `.item()` 都是一次**完整 cudaDeviceSynchronize**，GPU 上已排队的几十个 kernel 全部排空再重新启动。这是官方 tuning guide [3] 明令避免的"python control flow dependent on CUDA tensors"。**改法**：因为 `seq_len` 固定、输入 shape 固定，`period_list` 在训练中几乎不变 → 可以用 `torch.gather` / 索引直接向量化处理 3 个周期（去掉 Python loop），或至少把 `.item()` 换成**一次性**把 3 个 period 同步下来（一个 `.cpu().tolist()` 而非 3 次 `.item()`，且不在 kernel 密集区做）。

**(b) FFT 操作**：
- `FFT_for_Period`（`model.py:7-42`）：`rfft`（cuFFT，已高度优化，不值得自写 FFT）→ `abs().mean(0).mean(-1)` 是归约 → `topk`。fft 结果只用来选 top-3 周期，**不参与反向**（`freq_amp` 无 requires_grad 路径）→ 可以在 `torch.no_grad()` 下算，省掉自动求图。
- bf16 下 cast fp32（`model.py:16-21`）：全张量 `float()`+转回 = 2 次全量转换 + 显存往返。由于 `rfft` 不接受 bf16，**要么整段在 fp32 buffer 上做**（预分配，不逐 batch cast），要么接受 cast。这属于"小钱"，真正的大头是 (a) 的同步。
- **融合可行性**：fft 本身难进 Triton（无 FFT 原语），**不值得自写**；但"fft 输出后的统计 → topk → 加权"这一段可以在 Triton 里做（§6 候选 3）。

**(c) 多尺度 Inception 卷积**：`InceptionBlockV1`（`model.py:55-75`）6 个不同 kernel 的 Conv2d + `stack.mean`。每次 `TimesBlock` 跑 2 个 inception × e_layers=2 = 24 次 Conv2d，加上每周期 pad/reshape/permute/contiguous（**`contiguous()` 是真实拷贝！`model.py:116`**）。这些对 RTX 3090 都是 1~5μs 的 kernel。**cudnn.benchmark 已开，conv 本身算法已较优；大头是"每周期都要 permute+contiguous 重排内存"**。

**(d) Spike 分支 + 门控统计（`model.py:149-211`）**：`diff/mean/std/z/diff_z/soft_spike_mask`、`dominant_ratio/entropy/spike_strength` 是 **15+ 个 elementwise+归约 kernel**，全部 memory-bound、可融成 1~2 个 kernel（§6 候选 2）。

**(e) 结论：确实是"小而多 kernel"模型**。per-forward kernel 数粗估：embedding(1) + 2×TimesBlock×[fft(1-2) + 3 周期×(pad/cat + reshape + contiguous + 12 conv + stack/mean) + softmax 加权] + spike(3 conv + ~10 elementwise) + gate(fft + ~10 elementwise) + head ≈ **80~120 次 launch**，处理的数据总量却只有 ~1MB 级。这正是 CUDA Graph / torch.compile / 手工融合能发力的形态。

---

## 4. 落地路线（分阶段，每步可独立验收）

| 阶段 | 动作 | 预期收益 | 成本/风险 |
|---|---|---|---|
| **P0** | `torch.profiler`（`torch.profiler.profile` + `table(sort_by="cuda_time_total")`）在两个模型训练 loop 各跑 1 次，统计 kernel 数、`CUDA time` 占比、`sync` 次数。验证"launch-bound"假设（预期 GPU util <30%） | — | 低；先量后改 |
| **P1a** | `optim.Adam(..., fused=True)`（TimeMixer `repro_pipeline.py:1405`、RT916 `core.py` 的 optimizer 构造处） | 优化器 step 从 ~4 kernel → 1 | 极低，一行 |
| **P1b** | `torch.compile(model, mode="reduce-overhead")`（自动融合 + CUDA Graph），**保留 eager 开关**，黄金基线对照 | 1.3~2.5× | 中；graph break 需排查，RT916 的 loop 要先修 |
| **P1c** | 修 RT916 `TimesBlock`：`.item()`/Python loop → 向量化 gather（或一次性同步）。可顺带 `no_grad()` 包 FFT 统计 | 每 forward 省 3 次全同步 + 十几次 launch，**最大单点** | 中；需数值等价对照 |
| **P2** | Triton 写 §6 的 3 个 fused kernel | 热点 2~4× | 中；需要服务器有 `triton` |
| P3 | 若 P2 后仍有可量化的热点且 Triton 表达不了 → 再评估 C++/CUDA | — | 高，暂不做 |

**验收纪律（衔接 skill §3/§6）**：所有加速改造都走 `scripts/tests/` 四件套 + 黄金基线逐字节 diff；GPU 模型改造在**服务器**验证（本机 CPU-only 不跑会误导）；数值等价验收允许 **fp32 下 allclose(atol=1e-5, rtol=1e-4)** 而**非逐位**，但最终 `submission_ready.csv` 仍须通过既有回归。

---

## 5. 技术选型对比（Triton vs C++/CUDA vs torch.compile）

| 维度 | torch.compile | Triton | 手写 C++/CUDA |
|---|---|---|---|
| 改代码量 | ~1 行 | 每 kernel 20~60 行 | 每 kernel 100~300 行 + 构建 |
| 编译器/环境要求 | 随 torch | `pip install triton` | MSVC + CUDA Toolkit + nvcc 链路 |
| 融合能力 | 自动（点/归约）+ CUDA Graph | 手工、可精确控制 | 完全可控 |
| 可移植性 | torch 跨设备 | 跨架构（NVIDIA/AMD 后端） | 绑定 NVIDIA |
| 维护成本 | 无 | 低 | 高 |
| 本项目适用 | **P1 主力** | **P2 热点** | P3 备胎 |

---

## 6. 本项目最值得写自定义算子/Triton kernel 的 3 个点

### 候选 1：TimeMixer 滑动均值分解 + 逐元素修正融合（`MovingAvg` → fused kernel）
- **对象**：`backbones.py:8-24` `MovingAvg`（AvgPool1d k=25, stride=1）+ `seasonal = x - trend`。
- **做法**：Triton 一个 kernel：每个 program 处理一行（batch×channel），寄存器内维护 25 宽滑动窗口累加，一次算 `trend` 并就地 `seasonal = x - trend` 输出。**1 个 kernel 取代 transpose+avg_pool+截断+transpose+sub ≈ 5 个 kernel + 2 次显存往返**。
- **可行性**：高（官方 vector-add/layer-norm 教程同型 [9]）。数值：滑动窗口累加顺序与 AvgPool 不同 → 需 allclose 验收（§4）。
- **收益**：每 `PastDecomposableMixing` block ×3 scales ×2 blocks 省 ~6 次 launch；顺带与后续 `x+y`、LayerNorm 融合可再进一步（若 torch.compile 未覆盖）。

### 候选 2：RT916 Spike 分支统计量融合（z-score / soft mask / 门控特征）
- **对象**：`model.py:149-211` `SpikeResidualBranch` 前段 + `DynamicPeriodGate._period_features`。
- **做法**：Triton 一个 kernel 完成 `diff → mean/std → z → diff_z → soft_spike_mask → [B,4,L] 特征组装`（全是逐点+归约）；门控统计（`dominant_ratio / entropy / spike_strength`）的**非 FFT 部分**（abs→topk→ratio/entropy）也并入或单独一个 kernel。
- **可行性**：高（reduction + pointwise 是 Triton 标准场景 [9]）。
- **收益**：~15+ elementwise kernel → 1；省掉 `torch.stack` 的转置拷贝。

### 候选 3：RT916 周期分支的 pad/reshape/contiguous 重排融合（并消除 `.item()` 同步）
- **对象**：`model.py:96-125` `TimesBlock` 的周期循环。
- **两步走**：
  1. **先做结构化改造（P1c，不写 kernel）**：去掉 Python loop + `.item()`（见 §3.2a），用 `torch.gather`/一次性同步取 period，`permute+contiguous` 合并为一次 `reshape`（或直接构造 contiguous 视图，避免 `contiguous()` 双拷贝）。
  2. **再用 Triton**：写一个"三周期折叠 + 1D 多尺度卷积 + 周期权重加权"的 fused kernel：每个 program 负责一个 (batch, channel, period) 分片，寄存器内做 k=1/3/5/7/9/11 的 1D 卷积并求平均（替代 6 个 Conv2d + stack+mean），输出直接按 softmax 权重组合。一次内存读入（x）、一次写回（block 输出），**替代 ~30 次 launch**。
- **可行性**：中偏高。多尺度 1D 卷积在 Triton 里完全可写（官方有 conv 相关教程、matmul 教程同型 [9]），但周期动态变化时 grid/block 要按 `tl.constexpr` 静态化——**正因如此，第 1 步（把 period 变成每轮训练固定值）是前提**。若周期在训练中确实波动，则退化为"只做 1D inception 融合、周期循环保留在 Python 但不再同步"。
- **收益**：RT916 最热点，预计 forward 时间减半量级。

---

## 7. 参考资料

[1] PyTorch, *CUDA semantics*（TF32 开关/默认值、异步执行、streams、CUDA caching allocator、CUDA Graph 相关 allocator 选项）— https://docs.pytorch.org/docs/2.13/notes/cuda.html
[2] NVIDIA, *Getting Started with CUDA Graphs*（2.9/3.8/3.4μs 量化、capture→instantiate→replay）— https://developer.nvidia.com/blog/cuda-graphs/
[3] PyTorch, *Performance Tuning Guide*（fuse ops / torch.compile / CUDA Graphs / cudnn autotuner / AMP≈3× / zero_grad(set_to_none=True) / 避免 CPU-GPU 同步清单）— https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
[4] PyTorch, *Introduction to torch.compile*（Dynamo→Inductor→Triton、graph break、2.5× 点运算实测）— https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
[5] PyTorch, *torch.compiler 文档* — https://docs.pytorch.org/docs/2.13/torch.compiler.html
[6] NVIDIA, *Kernel Fusion in NVIDIA CUDA*（sum(abs(x)) 2 kernel→1，3GB→1GB 内存流量 3×；融合 vs CUDA Graph 是两层互补）— https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/
[7] PyTorch, *torch.compile programming model*（graph breaks 排查）— https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html
[8] PyTorch, *Extending PyTorch*（自定义 op / autograd.Function / save_for_backward / gradcheck / cpp_extension）— https://docs.pytorch.org/docs/2.13/notes/extending.html
[9] Triton, *Getting Started / Tutorials*（vector add、fused softmax、layer-norm、matmul、benchmark 方法论）— https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
[10] Triton, *Welcome to Triton*（Python DSL、aims 于"能写接近峰值性能的 kernel"）— https://triton-lang.org/main/index.html
[11] 项目内部：`docs/CPU_GPU_异构并行_调研报告.md`、`docs/工业界时序预测训练加速与精度提升调研报告.md`、`.opencode/skills/efm3-lessons/SKILL.md` §4.3-4.4
