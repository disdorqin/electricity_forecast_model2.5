# 深度调研：深度学习训练中 CPU 与 GPU 的混合并行

> 调研日期：2026-08-15 | 调研方式：PyTorch 官方文档 + DeepSpeed/ONNX/工业文献 + 本项目代码核验
> 主题：heterogeneous pipelining / CPU-GPU offloading / 设备间传输并行
> 关联项目：EFM3 电力价格预测（7 模型混合 CPU/GPU，scheduler 并发）
> ⚠️ 本文是调研报告，非实现承诺。所有设计结论标注了「已验证/推断待实验」。
> 红线提醒：本报告涉及对 timemixer/rt916 训练代码的改造设想，**不涉及爬虫/数据/交付文件**，不触发数据真实性红线；若后续落地实现，须按 `efm3-lessons` §3/§6 跑回归四件套 + 黄金基线 diff。

---

## 0. 一句话结论（TL;DR）

- **idea 原理上完全成立**，但它是**教科书级"软件流水线/预取"**，而非新技术：PyTorch 的 `DataLoader(num_workers)+pin_memory+non_blocking`、ONNX Runtime 的 Execution Provider 回退、TensorFlow 的 `tf.device` placement、DeepSpeed ZeRO-Offload、2025 年的 SiPipe（LLM 推理 CPU-GPU pipeline）都做了同样的事。
- **对本项目，把"模型内部 CPU 能做的算子搬去 CPU"收益很小甚至为负**（单 batch 几百 KB、传输 ~10µs、kernel launch ~5-10µs，同步开销可能吃掉收益）；GPU 是瓶颈的主因是**小 batch 的 kernel launch 开销 + 单 GPU 串行**，正解是 `torch.compile`/CUDA Graph/加大 batch，而不是 CPU offload。
- **唯一有实在收益、且与本项目现状（特征预计算 3-7min/天）契合的切法**：**「训练日级双缓冲流水线」**——CPU 用 pandas/numpy 预计算 D+1 天的特征矩阵，同时 GPU 训练 D 天；用 CUDA stream + 事件把 H2D 传输与训练重叠。预期加速 **1.05–1.2×**（受限于特征工程占比）。
- **论文价值：低-中**。可写但必须先做基线测量（每算子 CPU/GPU 实测时间表 + GPU 利用率曲线），否则就是"DataLoader 换个说法"。见 §3.4。

---

## 1. PyTorch 底层算子/执行机制（重点）

### 1.1 CPU→GPU 张量传输：`.cuda()/.to()/pin_memory` 机制与开销

**机制**（依据 PyTorch CUDA semantics 文档）：
- GPU 操作默认**异步**：调用算子只是**入队**到设备，不阻塞 CPU。CPU 可以继续执行其他指令，两个设备"看起来"并行。
- 但 **CPU↔GPU 拷贝默认是"同步屏障"**：PyTorch 在检测到跨设备拷贝时会自动插入同步，让语义上等价于同步执行。这是绝大多数"传输开销"的真实来源——不是带宽，是**同步**。
- `.to(device)` / `.cuda()` / `.copy_()` 支持显式 `non_blocking=True`，**跳过自动同步**，把拷贝也变成异步入队。要配合 `pin_memory` 才真正受益。
- `pin_memory=True`：把主机内存页**锁定**（`cudaHostAlloc`/`cudaHostRegister`），使 DMA 可以绕过 CPU 页表直接拷进显存；非锁定内存要先 CPU 侧拷贝到临时 buffer，多一次拷贝且会阻塞。PyTorch 2.13 起还提供 `pinned_use_cuda_host_register`、`pinned_reserve_segment_size_mb` 等调优。
- **开销数量级**（PCIe 3.0 x16 ~12 GB/s，PCIe 4.0 ~24 GB/s；RTX3090 云服务器多为 PCIe 4.0）：
  - 本项目单 batch：`[B=16, L=168, V≈13]` fp32 ≈ **140 KB** → H2D 约 **10–15 µs**；96 点 `L≈672` 约 560 KB → 约 25–50 µs。
  - 作为对比：单次 CUDA kernel launch 固定开销 **3–10 µs**；GPU 同步点 5–20 µs。
  - **结论：几百 KB 的传输本身微不足道，瓶颈永远是"同步点"和"小算子 launch 密集度"，不是传输带宽。**

### 1.2 CUDA streams 与异步传输重叠

- 每个设备有默认 stream，stream 内严格串行；**不同 stream 之间可并发**（只要资源允许）。
- 正确用法（文档原文示例）：非默认 stream 使用张量前必须 `s.wait_stream(default_stream)` 建立依赖，用完调 `A.record_stream(s)` 防止缓存分配器提前回收内存。**自动同步在非默认 stream 下不再生效，责任在用户。**
- backward 语义：**每个反向算子跑在对应前向算子所在的 stream 上**；若前向把独立分支放到不同 stream，反向也会自动并行。`loss.backward()` 与后续使用梯度的代码必须 `current_stream().wait_stream(s)` 同步，否则竞争。
- **这就是"GPU 算 + 传输并跑"的官方机制**：H2D 拷贝放入一个 side stream，与默认 stream 上的 kernel 并发；再用事件在"数据就绪"时让默认 stream 等待。
- 本项目 TimeMixer（`repro_pipeline.py:1375-1546`）与 RT916（`core.py:558-616`）**已经用了** `num_workers=4 + pin_memory=True + non_blocking=True + persistent_workers`——即 DataLoader 已把"CPU 侧取数/组 batch + 异步 H2D"与"GPU 训练"重叠了。**这部分红利已兑现。**

### 1.3 torch.compile / Inductor 的 CPU/GPU 代码生成

- 工作流：Dynamo（Python 级追踪）→ FX 图 → 图优化 passes → Inductor 降低为后端代码。
- 后端：**CUDA 走 Triton**（生成 Triton kernel，再编译成 CUDA）；**CPU 走 cpp wrapper + oneDNN/ATen**（Inductor CPU backend 有官方 debug/profile 教程）。
- **重要边界：torch.compile 是"逐设备"编译的**——它把一个算子子图全丢给一个设备后端编译，**不会把一张计算图拆成 CPU 部分 + GPU 部分分别跑并自动编排**。图上只要有一个跨设备 tensor 边界，Dynamo 就会在边界处停下（guard 按设备分片）。
- torch.compile 对**小模型小 batch 的真实价值**：算子融合 + 减少 kernel 个数 + 可叠加 CUDA Graph 捕获 → **直接压低 kernel launch 密集开销**。这恰恰是本项目 GPU 模型"GPU 是瓶颈"的正解之一（见 §3.3）。
- CUDA Graph（`torch.cuda.graphs`）：把一串 kernel 捕获成一张图重放，launch 开销从 N×5µs 降到 ~20µs 一次。RT916/TimeMixer 若配合固定形状 batch，可用。

### 1.4 PyTorch 官方有没有 heterogeneous execution（CPU 算子+GPU 算子混排一张图）？

**结论：训练侧没有"自动算子级 CPU/GPU 混排执行器"；只有两个半官方机制：**
1. **`torch.distributed.pipelining`**（alpha，PyTorch 2.x 官方 pipeline parallelism 库）：把模型切成 `PipelineStage`，每个 stage 指定 `device`。**stage 粒度**上支持异构设备（含 CPU stage），内部用 microbatch + schedule（GPipe 风格）流水并行。但它是为"大模型切多 GPU"设计，CPU stage 实践中不常见，且 stage 间通信（recv/send）走进程组。
2. **`torch.compile` 的逐设备编译**（§1.3）：CPU/GPU 各编译各的子图，跨界需用户自己 `.to()`。
3. **框架层对比**：TensorFlow 1.x 静态图原生支持 `tf.device('/CPU:0')`/`/GPU:0` 任意算子放置 + 软放置自动回退（§2.4）；PyTorch 的"device"是**数据属性**不是**图属性**，算子跟着张量走，没有全局 placement 规划器。**这是两框架哲学差异，也是本 idea 在 PyTorch 里"没有现成开关"的根因。**

---

## 2. 现有 CPU-GPU 混合/异构训练方案（工业界）

### 2.1 CPU offloading：DeepSpeed ZeRO-Offload / ZeRO-Infinity（最主流）

- 原理：**把优化器状态 + 优化器计算**从 GPU 卸到 CPU 内存，GPU 只做前反向。缓解 GPU 显存/算力压力。
- 关键点：CPU 侧用 DeepSpeed 自研 **`DeepSpeedCPUAdam`（比 PyTorch CPU Adam 快 5–7×）**；配 `contiguous_gradients`（梯度连续化）+ `overlap_comm`（梯度拷回 CPU 与反向重叠）避免 offload 变瓶颈。
- 规模：单卡 V100 训 10B GPT-2；ZeRO-Infinity 扩展 NVMe offload 到 1.9TB 参数（Microsoft Research blog）。
- **与本 idea 的关系**：ZeRO-Offload 是"**状态/辅助计算**放 CPU、**主算子**留 GPU"的权威先例——证明"CPU 干杂活、GPU 干主活"在工业界是成立且被量产的范式。但注意它卸的是**优化器**，不是**数据预处理**。

### 2.2 Pipeline parallelism 的 CPU-GPU 变体

- 鼻祖：**GPipe**（Huang et al., arXiv:1811.06965）microbatch 切分 + 流水；**PipeDream**（Narayanan et al., arXiv:2006.09503）异步 one-forward-one-backward。二者把模型沿层切成**多个 stage 放多张卡**。
- 异构变体：**PipePar**（Neurocomputing 2023，异构 GPU 上的模型划分与放置）；**HetPipe**（whimpy 异构 GPU 集群，DNN 训练整合 pipelined model parallelism + data parallelism）；BytePS（OSDI'20，**利用闲置 CPU 与带宽做梯度聚合**，CPU-GPU 协同）。
- **PyTorch 官方实现** `torch.distributed.pipelining` 支持 stage 级 `device`（见 §1.4）。**本 idea 在 stage 粒度上的形态**=把"特征工程"做成一个 CPU stage + "张量前反向"做成 GPU stage，用 pipeline schedule 串起来——但这需要把特征工程包成 `nn.Module` 且数据依赖切干净，对一个 24/96 点小模型是过度设计。

### 2.3 数据加载 CPU 预处理 + GPU 计算的标准 pipeline

- **PyTorch DataLoader**：`num_workers=N`（多进程取数+预处理）+ `prefetch_factor`（预取队列深度）+ `pin_memory=True` + 主循环 `tensor.to(device, non_blocking=True)`。这是工业界默认的"CPU 预处理与 GPU 计算重叠"方案，**本项目两个 GPU 模型均已配置**。
- NVIDIA DALI：GPU 侧/CPU 侧数据管线库，面向图像/视频流；本项目（表格时序）无必要。

### 2.4 "部分算子固定跑 CPU、部分跑 GPU"的现成框架

| 框架 | 机制 | 粒度 | 备注 |
|---|---|---|---|
| **ONNX Runtime** | Execution Provider 按**优先级**分片图：CUDA EP 能跑的节点归 GPU，其余自动回退 CPU EP | **算子/node 级** | 最接近"一图内 CPU+GPU 混排"的工业实现 |
| **TensorFlow** | `tf.device` 手动放置 + `set_soft_device_placement(True)` 自动回退 | **算子级（静态图）** | TF 原生异构 placement 哲学 |
| **ExecuTorch**（PyTorch 官方，推理） | 设备委派（CPU/GPU/NPU），只做推理 | 算子级 | 不覆盖训练 |
| **CoDL**（MobiSys'22） | 移动端 DL **推理**算子级 CPU-GPU 协同执行 + 能耗调度 | 算子级 | 最高 **4.93× 加速 / 62.3% 节能** |
| **SiPipe**（arXiv:2506.22033, 2025） | LLM **推理** pipeline：把 sampling 卸到 CPU、CPU 输入准备与 GPU forward 用**双缓冲 CUDA Graph（TSEM）**重叠 | stage/阶段级 | **与本 idea 结构最像的先例**，2.1× 吞吐 |
| DeepSpeed ZeRO-Offload | 优化器状态+计算卸 CPU | 状态级 | §2.1 |

**SiPipe 的 TSEM 设计值得抄**：预捕获两份 CUDA Graph 各绑一个 buffer，迭代交替使用——CPU 写 buffer A 时 GPU 读 buffer B，规避写后读（WAR）竞争，无需运行时改图。这正是"CPU 预处理与 GPU 计算重叠且保证正确性"的模板。**注意 SiPipe 是推理不是训练**，训练侧无同款公开先例（反向的 stream 语义使训练侧更复杂，见 §1.2）。

### 2.5 "CPU-GPU heterogeneous time series forecasting" 先例检索

- **直接先例几乎没有**：检索 `CPU-GPU hybrid time series forecasting / heterogeneous time series deep learning` 返回的多是通用 CPU-GPU 平台（用于 TS 预测）、CUDA 并行化、TensorFlow 多核调度等——**没有人**把"TS 模型内部特征工程拆 CPU、张量前反向拆 GPU 并重叠"当成独立研究点发表。
- 隐含先例：所有 PyTorch TS 模型的 `DataLoader` 流水、ONNX Runtime 的 CPU 回退、TF 的 placement、以及**梯度提升树生态**（LightGBM CPU/GPU 双后端）本质上都覆盖了该场景的一部分。
- **这意味着**：写论文时"我们首次系统化研究小规模多变量 TS 模型上的 CPU-GPU 异构流水"在检索层面站得住，但审稿人大概率认为"这是工程细节的组合，机制不新"——需要靠**扎实的测量方法论**（每算子实测 CPU/GPU 耗时 + 放置决策 + 端到端加速 + 精度不变性证明）支撑，见 §3.4。

---

## 3. 对本项目（EFM3 电力价格预测，7 模型混合）的可行性分析

### 3.1 现状还原（已对照代码，非臆断）

- **scheduler**（`runtime/resource_scheduler.py`）：`CPU_MODELS={lightgbm,sgdfnet,timesfm}`、`GPU_MODELS={timemixer,rt916}`。CPU 队列 2 线程并发，GPU 队列 1 线程串行（防 CUDA OOM），两队列由外层 2 线程 ThreadPool 并发。**跨模型并发已做**。
- **TimeMixer**（`TimeMixer/repro_pipeline.py`）：一次性 pandas/numpy 特征工程 → numpy 数组 → `ElectricityDailyDataset`；`DataLoader(num_workers=4, pin_memory, persistent_workers)`；train loop `xb.to(device, non_blocking=True)`；loop 内还有分段校准/风险加权（在 GPU tensor 上）。
- **RT916**（`RT916_SpikeFusionNet/src/rt916_spikefusionnet/`）：`core.py` 一次性 pandas 特征工程（`enrich_period_local_features`、interpolate、MinMaxScaler）→ numpy → DataLoader（同样 4 worker + pin_memory）。模型内部 `DynamicPeriodGate._period_features`（`model.py:184-211`）做 FFT/幅谱/熵/尖峰强度等**小而多的统计算子**，`SpikeResidualBranch`（z-score + sigmoid mask + 小卷积）。
- **瓶颈事实**（skill §4.12）：CPU 模型单配置 <0.1–10s；GPU 模型（TimeMixer/RT916，RTX3090 云）才是墙钟瓶颈；且 GPU 队列内两个模型**串行**。

### 3.2 idea 的三种切法，逐一判断

**切法 A：把 GPU 模型"forward 内部"的 CPU-able 算子（FFT/熵/尖峰统计/归一化）搬到 CPU 跑**
- 这些算子在 `model.py` 里已是**向量化 torch 算子**（不是 Python 循环），batch B=16、L=168，每个算子 GPU 上执行 <50µs。搬去 CPU = 每步多一次 D2H + 一次 H2D + 至少 1 个同步点（~20–50µs），**净收益 ≈ 0 甚至为负**。且它们参与反向（gate 的 period_features 只前向，但 spike_branch 参与），搬 CPU 后要么 `detach()`（改变语义）要么 CPU 侧再搭 autograd（成本高）。
- **结论：放弃。** 除非先用 profiler 实测某算子占比 >10% 且 batch 极小，否则别动。

**切法 B：训练日级双缓冲流水线 —— CPU 特征工程(D+1) 与 GPU 训练(D) 重叠**
- 现状：每训练日串行 `[特征工程 3-7min] → [GPU 训练]`。特征工程是纯 pandas/numpy（CPU），训练纯 GPU。**把"特征工程"放进一个后台线程/进程 + 双 buffer（buffer[0] 给 GPU 训练用，buffer[1] 给 CPU 预计算下一批），天然重叠。**
- 这正是 DataLoader 在"天"粒度上的放大版；复杂度低（无反向 stream 问题，因为重叠发生在**训练任务之间**而非 step 内），风险主要是内存（96 点特征矩阵 ~19MB，buffer×2 可忽略）。
- 项目已有 FeatureStore 预计算设计（skill §4.6/记忆），**该切法与 FeatureStore 是同一目标的两种实现**：FeatureStore=离线一次性预计算全量；切法 B=在线流水重叠。**若 FeatureStore 落地，切法 B 无独立价值**（都被覆盖）；若不做 FeatureStore，切法 B 是最便宜的替代。
- **结论：可行，收益中（详见 3.3），工程改动小。**

**切法 C：scheduler 级让两个 GPU 模型在单卡上用 stream 并发跑**
- 现因防 OOM 串行。用 stream 并发两个模型的 kernel 可能部分重叠（小模型留有空闲 SM），但**两者都小、都 launch 密集**，重叠收益有限且 OOM/显存碎片风险高。
- **结论：低优先，可做单次实验探底（`nvidia-smi` + 时间线），不宜当主方案。**

### 3.3 传输开销 vs 节省的权衡（量化）

- 传输：单 batch ~140–560 KB，H2D/D2H 各 **~10–50µs**（PCIe 4.0）。可忽略。
- 真正的开销项：**每步同步点 5–20µs × 每步次数** + **小算子 kernel launch 3–10µs × 算子数**。TimeMixer/RT916 每 step 数百个算子时 launch 开销可达 step 时间的 **10–30%**。
- **因此"GPU 是瓶颈"的主成分不是"CPU 没在帮忙"，而是 (a) kernel launch 密集、(b) GPU 队列串行、(c) 特征工程与训练串行。**
- 对应正解优先级：**① torch.compile / CUDA Graph（压 launch，预计 1.2–2× step 提速，零精度风险用 compile 前先量化）→ ② 切法 B 双缓冲（1.05–1.2× 天级）→ ③ 加大 batch（等价多微批）→ ④ 切法 A（放弃）。** 其中 ① 与 ② 正交可叠加，③ 与 ① 冲突（graph 需固定形状）。

### 3.4 创新点评估（写成小论文的价值）

- **诚实的判断**：本 idea 的机制 = 软件流水/预取 + 双缓冲 + stream 重叠，全部有工业先例（§2.4 表）。单独写"我们把特征工程搬 CPU"大概率被毙。
- **可辩护的论文角度**（任选一，且必须有数据支撑）：
  1. **测量驱动的算子级异构放置**：对 TS 小模型逐算子实测 CPU/GPU 耗时矩阵，用贪心/0-1 规划决定每算子放哪（仿 CoDL 的 profiling-based placement，但场景=训练+表格 TS）。卖点是"决策方法"而非"放 CPU"这个动作本身。
  2. **"GPU underutilization 是短视距 TS 小模型的系统性痛点"**：用本项目 24/96 点真实负荷（⚠️ 论文红线：**山东/山西数据不进论文**，用宁夏/甘肃/陕西/青海 + Lago/NEM/GEFCom/UniElecPrice 公开集）系统量化 launch 开销占比、验证 CUDA Graph + 日级双缓冲组合能拿回多少。卖点是**实证 + 系统化评测**，不是机制。
  3. **训练日级 pipelining 的调度算法**：把"特征工程(t_c) + 训练(t_g)"的天级调度形式化为流水线负载均衡问题（多市场/多任务并发时），提出抢占/优先级策略。卖点是调度理论。
- **不建议**：把"CPU-GPU 数据传输实现并行"当独立贡献写——那是 CUDA 基本功，审稿人（尤其 HPC 方向）会直接拒。

---

## 4. 可落地设计草案（若采纳切法 B + CUDA Graph）

> 范围：仅改 TimeMixer/RT916 的训练入口，不碰数据/交付/爬虫。**先做 1 天小样本 A/B 验证，不动生产链路。**

### 4.1 目标与验收

- 目标：单 GPU 训练墙钟时间 ↓（预期 1.05–1.2×），`submission_ready.csv` 与黄金基线**逐字节一致**（训练环节不产出交付文件，但精度/形状必须不变）。
- 验收：① 训练加速可复现（≥5 天均值）；② 模型精度 diff ≈ 0（同 seed、同超参、同随机序列）；③ 回归四件套全绿（若触碰共享代码）。

### 4.2 架构

```
训练日序列 D, D+1, D+2 ...
┌─────────────────────────────────────────────────────────┐
│  CPU 线程池（1 worker，FeatureBuilder）                    │
│  ├─ D+1: pandas 特征工程 (enrich/scale/interpolate)        │
│  ├─ D+2: ...  输出 → 双 buffer（numpy，已 pin_memory）      │
└───────────────┬─────────────────────────────────────────┘
                │ 就绪事件
┌───────────────▼─────────────────────────────────────────┐
│  GPU 训练主循环（每训练日）                                │
│  DataLoader(已建好 numpy) → non_blocking .to()            │
│  torch.compile(model) 或 CUDA Graph(固定形状)             │
│  前向+反向(step 内) → 下一日循环体（不等 CPU）             │
└─────────────────────────────────────────────────────────┘
```

要点：**重叠发生在"任务"粒度**，训练 loop 内不引入第二个 stream（避免 backward stream 语义复杂度），只有"特征矩阵就绪 → 训练"这一处依赖，用 `concurrent.futures` 的 Future + 双 buffer 即可，甚至不需要 CUDA stream（拷贝已在 DataLoader 的 `non_blocking` 路径上）。

### 4.3 实现细节

1. **双 buffer**：`buf[2] = [np.empty(shape), np.empty(shape)]`，`buf[1 - k]` 由后台线程写，训练读 `buf[k]`；`k ^= 1` 交替。内存各 ~19MB（96 点）/4MB（24 点），无压力。
2. **pin_memory 在源码生成处**：后台线程产出 numpy 后直接 `torch.from_numpy(...).pin_memory()`，交给 DataLoader/训练端 `.to(device, non_blocking=True)`，一步到位避免二次拷贝。
3. **就绪同步**：`threading.Event`（或直接复用 Future）；训练日开始时 `future.result()`，除非特征工程 > 训练，否则等 0ms。
4. **随机一致性**：特征工程与模型无关，纯数据变换，双缓冲不改变任何随机数序列（训练端 seed 逻辑不动）；**必须断言**：两路径产出的 numpy 数组 `np.testing.assert_array_equal`（bitwise），防浮点顺序差异。
5. **可选叠加 CUDA Graph**：若固定 `batch_size` 与 `seq_len`，可对 `train_step` 用 `torch.cuda.graph` 捕获（需先 warmup + 固定形状）；**先量化再决定**，graph 与动态 loss 权重/风险分段（TimeMixer 有分段训练）冲突时放弃 graph 只留 compile。

### 4.4 伪代码骨架

```python
from concurrent.futures import ThreadPoolExecutor

def build_features(day, out_buf):
    df = load_raw(day)
    arr = fe_pipeline(df)            # 纯 pandas/numpy，CPU
    out_buf[:, :] = arr              # 双 buffer 原地写

with ThreadPoolExecutor(max_workers=1) as pool:
    bufs = [np.empty_like(ref), np.empty_like(ref)]
    pending = pool.submit(build_features, days[0], bufs[0])
    for i, day in enumerate(days):
        pending.result()             # 等 D 的特征就绪（通常已就绪）
        trainer.fit(bufs[i % 2])     # GPU 训练 D：DataLoader+non_blocking+.to(device)
        if i + 1 < len(days):
            pending = pool.submit(build_features, days[i + 1], bufs[(i + 1) % 2])
```

### 4.5 预期加速比与风险

| 项 | 预期 |
|---|---|
| 加速比 | 理论 `≈ 1/(1 - t_feat/(t_feat+t_gpu))`。t_feat≈3-7min、t_gpu≈30-60min/天 → **1.05–1.2×**；若某市场特征工程占比更高则更大 |
| 加速比上限 | 受限于 `max(t_feat, t_gpu)`；特征工程 ≤ 训练时最多 2×（实际远到不了） |
| 内存 | +~40MB（双 buffer + pin） |
| 风险 1 | 浮点顺序/并行差异 → 用 `assert_array_equal` 卡死，零容忍 |
| 风险 2 | 后台线程与 DataLoader worker 争 CPU 核 → 特征工程线程 `nice`/绑定 1 核，避免拖慢 CPU 模型（scheduler 同时在跑 lightgbm 等） |
| 风险 3 | 服务器单 GPU 已有 TIMESFM_DEVICE=cpu 约定 → 本改动只在 GPU 训练进程生效，互不干扰 |
| 风险 4 | torch.compile 首次编译预热 ~1-2min → 只对长训练值得开 |

### 4.6 验证步骤（与项目纪律衔接）

1. 服务器上先跑 **1 个市场 3 天**小样本 A/B（现状 vs 双缓冲），用 `torch.profiler` 或 `nvidia-smi` 时间线确认 GPU 空闲段被吃掉。
2. 对比同 seed 的 loss/val 曲线与预测表数值 diff（bitwise）。
3. 共享代码若被触碰 → `check_delivery_stability` 等四件套 + 黄金基线 diff。
4. 结果回写本 skill + 共享记忆（`memory_put`, category=`domain:efm3`）。

---

## 5. 参考资料

- PyTorch CUDA semantics 文档（异步执行/streams/non_blocking/内存管理）：https://docs.pytorch.org/docs/stable/notes/cuda.html
- torch.compile 教程（Dynamo→FX→Inductor，CPU/CUDA 后端）：https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Inductor CPU backend 调试与性能：https://pytorch-cn.com/tutorials/intermediate/inductor_debug_cpu.html
- torch.distributed.pipelining 文档（PipelineStage device 参数，alpha）：https://docs.pytorch.org/docs/stable/distributed.pipelining.html
- DeepSpeed ZeRO-Offload 教程（CPUAdam 5–7×、overlap_comm、contiguous_gradients）：https://www.deepspeed.ai/tutorials/zero-offload/
- DeepSpeed ZeRO-Infinity（NVMe offload，1.9TB 参数）：Microsoft Research blog
- GPipe：Huang et al., arXiv:1811.06965；PipeDream：Narayanan et al., arXiv:2006.09503
- PipePar：Neurocomputing 2023（异构 GPU pipeline 划分/放置）
- HetPipe：DNN 训练于 whimpy 异构 GPU 集群（浙大学报 FITEE 综述引用）
- BytePS：OSDI'20（利用 CPU/带宽做梯度聚合）
- ONNX Runtime Execution Providers（EP 优先级/回退）：https://onnxruntime.ai/docs/execution-providers/
- TensorFlow GPU 指南（tf.device 手动放置 + soft device placement）：https://www.tensorflow.org/guide/gpu
- CoDL：MobiSys 2022，CPU-GPU co-execution（4.93×/62.3%）
- SiPipe：arXiv:2506.22033，LLM 推理 CPU-GPU pipeline（CPU sampling + TSEM 双缓冲 CUDA Graph，2.1×）
- 异构 CPU+GPU SGD：UC Merced 报告（batch 分区聚合）；A Novel Multi-CPU/GPU Collaborative Framework for SGD：ICPP'21
- DistDGLv2：混合 CPU/GPU GNN 训练（亿级异构图）
- 本项目代码：`runtime/resource_scheduler.py`、`TimeMixer/repro_pipeline.py`、`RT916_SpikeFusionNet/src/rt916_spikefusionnet/{core,model}.py`
