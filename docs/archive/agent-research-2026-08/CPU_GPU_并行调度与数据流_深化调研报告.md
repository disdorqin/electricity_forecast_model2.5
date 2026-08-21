# 深度学习训练中 CPU-GPU 并行调度与数据流设计 — 深化调研报告（v2）

> 调研日期：2026-08-15 | 调研方式：webfetch 实证（PyTorch 2.13 CUDA semantics / torch.utils.data / torch.distributed.pipelining / DeepSpeed ZeRO-Offload / NVIDIA Overlap Data Transfers blog）+ 本项目代码核验 + 领域知识
> 在 `docs/CPU_GPU_异构并行_调研报告.md`（v1）基础上深化。v1 结论已被 §0 复盘并修正/细化，不再重复其推导。
> ⚠️ 本报告为调研设计文档，非实现承诺。所有具体数值（t_feat/t_gpu、加速比、传输时延）标注「实测/公式/假设」；落地前须先实测。
> 红线提醒：本文不涉及爬虫/数据/交付文件改动，不触发数据真实性红线；若后续落地实现，须按 `efm3-lessons` §3/§6 跑回归四件套 + 黄金基线 diff，且特征路径必须 `assert_array_equal` 逐位一致。

---

## 0. TL;DR（v2 相对 v1 的三点新结论）

1. **Stream 级重叠红利本项目已吃满，日级双缓冲不应再引第二个 CUDA stream。** 代码核验：`optim/perf_knobs.py`（统一底座）与 TimeMixer `repro_pipeline.py:1375-1386/1504-1512`、RT916 `core.py:557-571/608-616` 已全部配置 `num_workers=4 + pin_memory + persistent_workers + prefetch_factor=2 + non_blocking`。日级重叠发生在**任务粒度**，纯 CPU 侧即可，引入 side stream 反而背上 backward stream 语义 + `record_stream` + 缓存分配器的复杂度，收益≈0。见 §1.4/§1.6。
2. **加速比对 t_gpu 高度敏感：warm-start/AMP 把 GPU 训练压得越短，日级双缓冲越值钱。** 公式 `S = (t_feat+t_gpu)/max(t_feat,t_gpu)`：t_feat≈3-7min 固定，t_gpu=60min→1.05-1.12×；t_gpu=15min→1.2-1.47×；t_gpu=8min→1.38-1.88×。反过来意味着 **FeatureStore 若把 t_feat 从分钟级压到秒级，双缓冲独立收益塌缩** → 双缓冲是 FeatureStore 落地前的廉价替代 + 落地后的兜底。见 §3.6/§3.7。
3. **最可落地方案 = 线程 + 双缓冲 + 故障回退（`DayPrefetcher`，<100 行），DataLoader 一行不动。** 伪代码见 §3.5。生产单日运行里，同样的机制适用于**日内 DA→RT 两腿流水**（RT 特征依赖 DA 预测的 da_anchor，`repro_pipeline.py` 的 DA 腿先于 RT 腿串行），这是生产环境唯一有重叠空间的切面。见 §3.2。

---

## 1. CUDA Streams 与异步（重点）

### 1.1 执行模型：异步入队、默认 stream、自动同步屏障

依据 PyTorch 2.13 CUDA semantics（已实证 https://docs.pytorch.org/docs/2.13/notes/cuda.html ）：

- **GPU 操作默认异步**：调用算子只是入队到设备，CPU 不等待。每设备一个默认 stream；**stream 内严格串行，不同 stream 间可并发**（资源允许时）。
- **跨设备拷贝默认是同步屏障**：`.to()/.copy_()` 在 CPU↔GPU 间插入自动同步，使语义等价同步。`non_blocking=True` 显式跳过该同步，把拷贝也变入队。→ 这就是 v1「传输开销真凶=同步而非带宽」的文档出处。
- 调试期可设 `CUDA_LAUNCH_BLOCKING=1` 强制同步（误差堆栈定位用，勿用于性能测量）。
- 计时必须 `torch.cuda.synchronize()` 或 CUDA Event，否则异步导致测量失真。

### 1.2 多 stream 并发：计算 stream + 数据搬移 stream

依据 NVIDIA 官方博客 *How to Overlap Data Transfers in CUDA C/C++*（Mark Harris，2012，已实证 https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/ ）与 CUDA C++ Programming Guide：

**H2D/D2H 拷贝与 kernel 重叠的三条硬性前提**：
1. 设备支持 *concurrent copy and execution*（compute capability ≥1.1 起全部支持，现代卡无虞）；
2. **被重叠的拷贝与 kernel 必须分属两个不同的非默认 stream**（默认 stream 是同步 stream，会与全设备其他 stream 互斥，无法重叠）；
3. **主机内存必须 pinned**（非 pinned 的 `cudaMemcpyAsync` 会退化为同步拷贝）。

**两种入队模式的工程结论**：
- 模式 A：逐 chunk `[H2D → kernel → D2H]` 全放同一 stream；模式 B：先全部 H2D → 再全部 kernel → 再全部 D2H。
- 老卡（单 copy engine，C1060）只有模式 B 能重叠；双 copy engine（C2050 起，H2D/D2H 各一个引擎）模式 A 也行；**K20 起 Hyper-Q 两种等价**。RTX 3090 属后者，无需纠结入队顺序，但入队顺序仍影响 copy engine 与 SM 的调度水位。
- PyTorch 对应物：`torch.cuda.Stream()` + `with torch.cuda.stream(s):`；stream 上的 `.to(device, non_blocking=True)` 即入队式 H2D。

### 1.3 H2D/D2H 与 kernel 重叠的 PyTorch 具体实现

- **`pin_memory=True`**：主机内存页锁定（`cudaHostAlloc`/`cudaHostRegister`），DMA 绕过 CPU 页表直取，且使 `non_blocking` 拷贝真正可用；非 pinned 内存的 non_blocking 拷贝会被 CUDA 降级或先做一次页对齐。PyTorch 2.13 起提供 `pinned_use_cuda_host_register`、`pinned_num_register_threads`、`pinned_reserve_segment_size_mb`、`pinned_max_cached_size_mb`、`pinned_max_round_threshold_mb` 等调优（缓存分配器章节，同文档）。
- **标准重叠模式**（下一批 H2D 与当前 kernel 并行）：
  ```python
  s = torch.cuda.Stream()
  with torch.cuda.stream(s):
      x_next = x_next_cpu.to(device, non_blocking=True)   # H2D 入队到 s
  torch.cuda.current_stream().wait_stream(s)              # 计算 stream 等 s 就绪
  ```
- **关键现实**：当 batch 很小时（本项目几百 KB），DataLoader 的 `pin_memory + non_blocking` 已让"拷贝入队早于下一步 kernel"，硬件 copy engine 与 SM 自动并行——**手写 side stream 只在"大批量多 chunk 显式切分"时才必要**。本项目无此场景。

### 1.4 CUDA 事件同步 / record_stream（以及为什么日级双缓冲不需要它）

同 PyTorch CUDA semantics 2.13 文档，**backward 的 stream 语义**（本项目设计决策的直接依据）：

- **每个反向算子跑在对应前向算子所在的 stream**；前向若把独立分支放多个 stream，反向自动复用该并行。
- **`loss.backward()` 与后续用梯度的代码若不在同一 stream 上下文，必须 `current_stream().wait_stream(s)`**；PyTorch 1.9 起默认 stream 不再自动同步 backward（BC note），漏同步=竞态。
- 非默认 stream 上使用张量前必须 `s.wait_stream(default_stream)` 建立依赖；用毕 `tensor.record_stream(s)` 防缓存分配器提前回收块。
- **推论（本项目关键决策）**：这些语义全部是"**step 内**把计算拆到多 stream"的成本。而日级双缓冲的重叠发生在**任务之间**（特征工程与训练是两个独立任务，无反向交叉），纯 CPU 侧即可完成，**不需要也不会收益于第二个 CUDA stream**。若强行在训练 step 内加 side stream，等于把 step 内复杂化的风险全背回来。

### 1.5 DataLoader num_workers / persistent_workers / prefetch 如何与 GPU 计算重叠

依据 PyTorch 2.13 `torch.utils.data` 文档（已实证 https://docs.pytorch.org/docs/2.13/data.html ）：

- `num_workers=N`：**dataset 访问、transform、collate_fn 全部跑在 N 个 worker 进程**，主进程只生成索引、收 batch → CPU 侧取数/预处理与 GPU 训练天然重叠（worker 进程隔离了 GIL）。
- `prefetch_factor=2`：每个 worker 预取 2 个 batch，**总预取队列深度 = prefetch_factor × num_workers**（`num_workers>0` 时默认 2）。
- `persistent_workers=True`：迭代结束后不杀 worker，多 epoch 复用（免每 epoch 的 fork/spawn 重启）。本项目多 epoch 训练直接受益。
- `pin_memory=True`：collate 输出直接 pinned → 后续 `.to(device, non_blocking=True)` 走 DMA 快路径。默认 pin 逻辑只识别 Tensor/映射/可迭代；自定义 batch 类型需自实现 `pin_memory()`。
- **随机性契约**：worker seed = `base_seed + worker_id`（主进程用 RNG 生成 base_seed 并消耗状态），多 worker shuffle 可复现。
- **平台差异**：Unix `fork`（<3.14）克隆地址空间、零序列化；Windows/macOS `spawn` 重导入主脚本（需 `if __name__=='__main__'` 保护 + 顶层可 pickle 的 collate_fn/dataset）。服务器（Linux）与 Windows 本机行为不同，实验须在服务器验证。
- **内存注意**：每个 worker 持有 dataset 副本，进程内存 ≈ dataset 大小 × workers；本项目特征矩阵 19MB 量级，无压力。

### 1.6 本项目现状核对（代码实证，覆盖 v1 的 §1.2 收尾）

- `optim/perf_knobs.py`：`make_optimized_loader`（`num_workers=min(4,cpu-1)`、`pin_memory=1&&cuda`、`persistent`、`prefetch_factor=2`）+ `to_device(non_blocking=1)`，TF32/AMP(BF16)/cudnn.benchmark 已内置。**这是统一底座，两个 GPU 模型已接入。**
- TimeMixer `repro_pipeline.py:1375-1386`（loader）与 `1504-1512`（`.to(non_blocking)`）；RT916 `core.py:557-571`（loader）与 `608-616`（`.to(non_blocking)`）——同款配置，**手工重写了一遍而非调用底座，但行为一致**。
- **结论：批级（batch 内）重叠已全部兑现。剩余唯一可回收的重叠 = 「天级特征工程」×「GPU 训练」，这是本报告的落点。**

---

## 2. 并行调度框架

### 2.1 DeepSpeed ZeRO-Offload：CPU offload 机制（详解）

依据 DeepSpeed 官方教程（已实证 https://www.deepspeed.ai/tutorials/zero-offload/ ）：

- **配置**：`zero_optimization: { stage: 2, offload_optimizer: { device: "cpu" }, contiguous_gradients: true, overlap_comm: true }`。
- **机制**：GPU 保留 fp16/bf16 参数与梯度；**fp32 优化器状态（master weights + momentum + variance）搬到主机内存**，优化器 step 在 CPU 上计算。每步循环：反向 → 梯度连续化（`contiguous_gradients`）→ **D2H 异步拷贝梯度与后续反向重叠（`overlap_comm`）** → CPU 用自研 `DeepSpeedCPUAdam`（AVX/AVX2 多线程，**比 PyTorch CPU Adam 快 5-7×**）算出新 master weights → **H2D 回传与下一步前向重叠**。调优：`--bind_cores_to_rank` 绑核 + `OMP_NUM_THREADS`。
- **规模**：单张 V100-32GB 训 10B 参数 GPT-2（上限 ~13B）。
- **对本项目的启示（明确为负样本）**：ZeRO-Offload 解决的是"**显存/算力不够→搬状态**"，把优化器计算放 CPU 是为了省 GPU 计算与显存；本项目模型 ~1M 参数量级、显存无压力，搬优化器只新增每步 PCIe 往返与 CPU 延迟，**不适用**。但它作为"CPU 干杂活、GPU 干主活 + 双向传输与主计算重叠"的工业先例依然成立——`overlap_comm` 与我们关心的 H2D 重叠是同构的。

### 2.2 Pipeline parallelism：GPipe / PipeDream / PyTorch pipelining

- **GPipe**（Huang et al., arXiv:1811.06965）：模型沿层切 K 个 stage 放 K 设备，microbatch 灌水/排水式调度，流水气泡 ≈ `(K-1)/(K-1+m)`。
- **PipeDream**（Narayanan et al., arXiv:2006.09503）：异步 one-forward-one-backward（1F1B），气泡更小、但激活/权重副本内存开销更大。
- **PyTorch 官方 `torch.distributed.pipelining`**（2.13，alpha，已实证 https://docs.pytorch.org/docs/2.13/distributed.pipelining.html ）：
  - 前端 `pipeline(module, mb_args, split_spec)` 用 **torch.export 追踪**切分模型为 `PipelineStage`；`PipelineStage(submodule, stage_index, num_stages, device)` 的 **`device` 允许 CPU 或 GPU**；调度器有 `ScheduleGPipe / Schedule1F1B / Interleaved1F1B / LoopedBFS / InterleavedZeroBubble / ScheduleZBVZeroBubble / ScheduleDualPipeV`。
  - 硬性前提：**静态形状**、`torchrun` 多进程启动（每 rank 一个进程组）、形状不符抛 `PipeliningShapeError`、梯度按 microbatch 数缩放。
  - 使用 `pipeline()` 自动切分要求模型可被 torch.export 全图追踪；`PipeSplitPoint` 可手动标切点。
- **对本项目的判断**：理论上「CPU 特征 stage + GPU 模型 stage」可用 2-stage pipeline 表达（CPU stage 无参数、只前向），但代价是：torchrun 起 2 进程、特征工程包成可追踪的 `nn.Module`、静态形状、进程组 P2P——对 24/96 点小模型是**彻头彻尾的过度设计**。**弃**。§3.5 的手写软件流水（线程+双缓冲）就是它 30 行的等价物，且无 alpha 风险。

### 2.3 双缓冲 / 预取模式：CPU 预处理下一个、GPU 算当前

- **CUDA 经典双缓冲**（NVIDIA overlap blog §1.2）：两个 buffer 轮换，拷贝 chunk i 时算 chunk i-1，规避写后读（WAR）竞争——本 idea 的最底层模板。
- **SiPipe**（arXiv:2506.22033, 2025）：LLM 推理把 sampling 卸 CPU，CPU 输入准备与 GPU forward 用**双缓冲 CUDA Graph（TSEM）**重叠，2.1× 吞吐——结构上与本 idea 最像，但它是**推理**（训练侧的 backward stream 语义使其不能直接套）。
- **软件流水 / prefetch 泛化**：DataLoader 的 `prefetch_factor`（§1.5）、`concurrent.futures` 预取、工业批处理管线——同一模式在"任务/批次"粒度的再泛化。
- **训练侧 day-level 双缓冲缺少公开先例的原因**（诚实说明）：step 内双缓冲受 backward stream 语义约束（§1.4）而复杂；**任务粒度（day）双缓冲无反向问题**，复杂度与 prefetch 同级——这正是本项目可安全采用的原因。

### 2.4 torch.distributed 在本项目（单机单 GPU）是否有用

- **结论：目前没有价值。** torch.distributed 的价值在多进程/多卡/多机通信与 collective（NCCL）；单机单卡上唯一能表达 CPU/GPU 混跑的官方路径是 `torch.distributed.pipelining` 的 stage 级 device（§2.2），已判过度设计。
- 若未来要用**独立进程**做特征 worker（规避 GIL 的终极手段），用 `multiprocessing` + `shared_memory` 比 dist 进程组轻一个数量级；dist 的 send/recv 语义可作参考但无必要引入。
- **唯一潜在用途（非本项目）**：多 GPU 或多机扩展时（如把 7 个模型分到多卡），DDP/FSDP 才进入视野——不在当前规划内。

---

## 3. 本项目落地设计

### 3.1 目标重构：把「每训练日 特征工程→训练 串行」变成软件流水

现状（代码实证）：
- 每 GPU 模型每训练日内部串行：`build_arrays(df, days_window)`（pandas/numpy，**≈3-7min/日**，`repro_pipeline.py:470-582` / `filter_available_days:585-622`）→ scaler fit（`:1349-1351`）→ DataLoader → 多 epoch 训练（t_gpu，GPU）。
- 批级重叠已兑现（§1.6）；**日级 t_feat 与 t_gpu 的串行是剩余最大可回收项**。
- 跨模型重叠已存在（`runtime/resource_scheduler.py` CPU 队列 2 线程 + GPU 队列 1 线程并发）——**注意特征 worker 会与 scheduler 的 CPU 模型（lightgbm 等）抢 CPU 核**，这是 §3.6 风险 1 的根源。

### 3.2 日级双缓冲：线程 / 进程 / 预计算表 三选一

| 方案 | 机制 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| **线程**（`ThreadPoolExecutor(1)` + 预分配双 buffer） | numpy/pandas 的 C 实现释放 GIL，与主线程 GPU 训练真并行 | 零 IPC、零拷贝、共享只读 DataFrame、代码最简、无 spawn 问题 | Python 解释器片段有 GIL 摩擦；与主线程/DataLoader worker 抢少量核 | **首选（§3.5）** |
| **进程**（`ProcessPoolExecutor(1)` + `multiprocessing.shared_memory`） | 独立进程，彻底规避 GIL | 完全并行，隔离特征 worker 崩溃 | 序列化/共享内存管理复杂；Windows spawn 重导入；需确保子进程不碰 CUDA | 线程实测重叠率<80% 再升级 |
| **预计算表**（FeatureStore 离线物化，`docs/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`） | 全历史一次性物化，日级只切片 | 最优、运行时特征成本≈0、顺带统一口径 | 需要特征注册表对齐 + 逐位 diff 验收（S1-S3 工期） | **长期正解（§3.7）** |

**线程方案"真并行"的依据**：特征工程的耗时主体是 pandas/numpy 向量化算子，其 C/C++ 执行释放 GIL；主线程的 torch 前反向同样释放 GIL → 两段"重 C 工作"真并行，GIL 只在 Python 解释器层片段互斥。约束：特征 worker 不得持有 CUDA tensor、不得做 torch 写操作（纯 numpy/pandas 即安全）。

### 3.3 批级流水 vs 日级预计算：哪个更适合本项目

- **批级（DataLoader 预取）**：重叠"取数+H2D"与"训练"，µs 级；**本项目已全部配置（§1.6），无剩余可榨**（最多实测调 `num_workers`/`prefetch_factor` 找边际，但收益预期 <1%）。
- **日级（预计算）**：重叠"特征工程（分钟级）"与"训练（分钟级）"，**未兑现，是主战场**。
- **结论**：两者正交。日级双缓冲包在 DataLoader **外层**，DataLoader 一行不动——这是本设计侵入性最小的原因。

### 3.4 传输开销 vs 重叠收益的量化

（PCIe 3.0 x16 实测 ~12 GB/s；PCIe 4.0 x16 理论 24 GB/s、实测 15-20 GB/s；RTX3090 云机多为 PCIe 4.0）

| 数据对象 | 大小 | H2D 耗时（估） | 结论 |
|---|---|---|---|
| 单 batch `[16,168,13]` fp32（24 点） | ~137 KB | 10-20 µs（PCIe4）/ 25-50 µs（PCIe3） | 可忽略 |
| 96 点 batch `[16,672,13]` | ~559 KB | 30-80 µs | 可忽略 |
| 单日特征矩阵（96 点） | ~19 MB | **~1-1.5 ms** | 日级切换一次 ~1ms，相对分钟级任务可忽略 |
| CUDA kernel launch | — | 3-10 µs/次 | **step 内数百算子时才是真瓶颈** |
| 同步点 | — | 5-20 µs/次 | `non_blocking` 已消除大部分；剩余来自 `.item()/numpy()` 类隐式同步 |

**量化结论（强化 v1）**：几百 KB 传输不是瓶颈；**日级 19MB 整表切换也不是（~1ms）**。可回收的只有两处：(a) 特征工程 3-7min 的墙钟（本报告方案）；(b) 小算子 kernel launch 密集（`torch.compile`/CUDA Graph，承 v1，正交）。

### 3.5 落地伪代码（推荐：线程 + 双缓冲 + 故障回退）

```python
# utils/train_pipeline.py — 日级双缓冲包装器（纯 CPU 特征 worker + 主线程 GPU 训练）
from concurrent.futures import ThreadPoolExecutor

class DayPrefetcher:
    """把每日的特征工程/数组构建移进后台线程，双 buffer 轮换。

    build_fn(day) -> (past, future, y, baseline) numpy 四元组：
        - 必须无状态、可重入、纯 numpy/pandas、不碰 torch CUDA；
        - 产出与现状路径 np.testing.assert_array_equal 逐位一致（验收红线）。
    """
    def __init__(self, build_fn, max_workers=1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._build = build_fn
        self._fut = None

    def start(self, day):
        self._fut = self._pool.submit(self._build, day)   # CPU：特征工程 + scaler + 数组

    def get(self):
        try:
            return self._fut.result()                     # 已就绪→0 等待；否则阻塞（等价串行语义）
        except Exception as e:
            raise RuntimeError(f"day prefetch failed: {e}") from e

    def shutdown(self):
        self._pool.shutdown(wait=False)


def run_daily_backtest(build_fn, trainer_fn, days):
    """trainer_fn(past, future, y, baseline, day) -> GPU 训练 + 返回该日预测。
    保持现有 DataLoader(non_blocking/pin_memory) 与训练循环不动。"""
    pf = DayPrefetcher(build_fn)
    pf.start(days[0])                                     # 首日冷启动，无重叠
    for i, day in enumerate(days):
        data = pf.get()                                   # 等 day 特征就绪（通常已就绪）
        trainer_fn(*data, day)                            # GPU 训练（分钟级）与【下一天特征工程】并行
        if i + 1 < len(days):
            pf.start(days[i + 1])                         # 立刻提交 D+1，重叠开始
    pf.shutdown()
```

**实现要点**：
1. **双 buffer 落点**：`build_fn` 内部把产出写进预分配的 `np.empty` buffer（或直接返回 ndarray，由 `ThreadPoolExecutor` 持有引用）→ 主线程读取时与 DataLoader 的 `pin_memory` 链路衔接（numpy → `torch.from_numpy(...).pin_memory()` → `.to(device, non_blocking=True)`），避免二次拷贝。
2. **就绪同步**：`future.result()` 天然阻塞；只有"特征工程 > 训练"（罕见）才会真正等——而这正是双缓冲该有的行为（等价串行）。
3. **故障语义（能运行是底线 + 失败要响亮，skill §4.4）**：`get()` 抛错 → 捕获 → 降级为同步 `build_fn(day)`（阻塞重算）+ 向 manifest `degradations` 写入 `{kind: "prefetch_fallback", day, model}` + delivery_report 告警段。
4. **GIL 摩擦兜底**：若实测重叠率 <80%（`torch.profiler` 或 `time.perf_counter` 拆解验证），升级为 `ProcessPoolExecutor(1)` + `multiprocessing.shared_memory.SharedMemory`（特征 worker 写共享 buffer，主进程零拷贝读；子进程严禁 import CUDA）。
5. **可选进阶（实测有收益才做）**：用 `torch.cuda.Stream` 把"下一日 19MB 矩阵 H2D"与"当日最后几个 step"重叠——收益 ~1ms/天，通常不值得（§3.4）。

**生产单日运行的同一机制（日内 DA→RT 流水）**：`repro_pipeline.py` 每训练日先 DA 腿再 RT 腿串行（`main()` 内 `da_pred_df` → `rt_pred_df`，`da_anchor` 依赖 DA 产出）。可把 `build_arrays(RT)` 提交给 `DayPrefetcher`，在 `train_DA` 期间并行构建 RT 特征 → **单日运行也能重叠**，这是生产环境唯一有重叠空间的切面。

### 3.6 预期加速比与风险

**加速比公式**（稳态，扣除首日冷启动）：`S = (t_feat + t_gpu) / max(t_feat, t_gpu)`，`t_feat≈3-7min`（skill 特征预计算记录，假设待实测）。

| t_gpu（假设） | S 区间 | 214 天回测总省时（t_feat=5min） |
|---|---|---|
| 60 min/日（现状） | 1.05-1.12× | ~17.75 h |
| 30 min/日 | 1.10-1.23× | ~17.75 h |
| 15 min/日（+warm-start） | 1.20-1.47× | ~17.75 h |
| 8 min/日（+warm-start+AMP） | 1.38-1.88× | ~17.75 h |

> 总省时 ≈ `min(t_feat,t_gpu) × (n_days-1)`，对 t_gpu≤t_feat 的区间与 t_gpu 无关（被 t_feat 封顶）；对 t_gpu>t_feat 区间则随 t_gpu 下降而上升。**关键洞察：warm-start/AMP 把 t_gpu 压得越小，双缓冲占比收益越大**——所以它应该排在 warm-start 之后做，做在 FeatureStore 之前（§3.7）。

**风险与缓解**：
| 风险 | 说明 | 缓解 |
|---|---|---|
| R1 CPU 争用 | 特征 worker 与 GPU 模型 DataLoader 4 worker + scheduler CPU 模型（lightgbm）抢核 | Linux 绑核 `os.sched_setaffinity`/nice；GPU DataLoader 降 `OPTIM_NUM_WORKERS=2` 做 A/B；实测 nvidia-smi + CPU 占用 |
| R2 GIL 摩擦 | Python 解释器层片段互斥 | 先实测重叠率；<80% 升进程方案（§3.5 要点 4） |
| R3 浮点/顺序一致性 | 特征 worker 与现状路径产出不同 | `np.testing.assert_array_equal` 逐位卡死；worker 禁止任何随机源；A/B 单日 diff |
| R4 特征 worker 崩溃/慢 | 影响当天输出 | `get()` 异常→同步回退 + manifest 告警（§3.5 要点 3） |
| R5 内存 | +2×19MB pinned | 可忽略；2.13 可配 `pinned_max_cached_size_mb` 防 pin pool 膨胀 |
| R6 平台差异 | Windows spawn 与本机验证 vs 服务器 Linux fork | 线程方案无 spawn 问题；最终性能结论只在服务器出（skill §4） |
| R7 torch.compile 预热 | 首次编译 1-2min | 仅对长训练开；compile 与双缓冲正交，分开关 |

### 3.7 与 FeatureStore 的衔接

- **FeatureStore（离线物化 + asof 切片）把 t_feat 从"每天 3-7min 重算"变成"内存切片（秒级）"** → 日级双缓冲的独立收益随之塌缩到 1-2%（只重叠"切片+scaler+数组构建+H2D"）。
- **正确顺序**：`S1-S3 FeatureStore 落地`（解耦）→ `warm-start`（压 t_gpu）→ 双缓冲降级为兜底/IO 抖动保护；**若 FeatureStore 未落地**，双缓冲是当前最便宜的替代。
- **两者接口天然兼容**：`build_fn(day)` 的实现换为 `feature_store.slice(model, task, day, asof) + scaler` 即可，`trainer_fn` 一行不改。双缓冲 wrapper ≈ 30-50 行，不值得单独立项，建议作为 FeatureStore 落地的前置切面（v1 §3.2"若 FeatureStore 落地则切法 B 无独立价值"在此修正为"**大幅降值但保留作兜底**"）。

### 3.8 与 torch.compile / CUDA Graph 的协同（承 v1，简述）

- 双缓冲解决"**CPU 空闲**"（天级），compile/CUDA Graph 解决"**GPU 小算子 launch 密集**"（step 级，1.2-2× 预期）——**正交可叠加**。
- CUDA Graph 需固定形状：本项目每日常重训、batch 数随窗口变化 → 先量化再决定；`torch.compile` 对动态形状更宽容，零精度风险前先跑量化 A/B（v1 §3.3 优先级 ① ② ③ 保持）。

---

## 4. 结论：本项目最可落地的 CPU-GPU 并行方案（三步走）

1. **先测**：对单 GPU 模型单训练日做时间拆解（`time.perf_counter` 分段包 `build_arrays` / `trainer`；或用 `torch.profiler`），拿到实测 `t_feat` 与 `t_gpu`——本报告所有加速比数字都是公式+假设，**落地前必须实测校准**（skill §4.2"模型重活只在服务器跑"）。
2. **落地**：§3.5 的 `DayPrefetcher`（线程方案）+ 现有 DataLoader 不动 + `assert_array_equal` 验收 + 故障回退进 manifest。预期 1.1-1.5×（依 t_gpu 实测），改动 <100 行，不触碰共享交付代码；**生产单日运行优先做"日内 DA→RT 两腿流水"（同一 wrapper）**。
3. **后续**：FeatureStore S1-S3（更本质的解耦）→ warm-start（压 t_gpu）→ 双缓冲降级兜底；GPU 内层再评估 `torch.compile`。**不建议**：模型内算子搬 CPU（v1 §3.2 切法 A）、手写 side stream 做 step 内重叠（§1.4）、`torch.distributed.pipelining` 两 stage 异构（§2.2）、DeepSpeed ZeRO-Offload（§2.1）。

---

## 5. 参考资料（本次实证 + 领域知识）

1. PyTorch 2.13 — *CUDA semantics*（异步执行 / streams / non_blocking / wait_stream / record_stream / backward stream 语义 / 缓存分配器与 pinned 调优 / CUDA Graph）：https://docs.pytorch.org/docs/2.13/notes/cuda.html
2. PyTorch 2.13 — *torch.utils.data*（DataLoader 参数 / prefetch_factor / persistent_workers / pin_memory / worker seed / fork vs spawn）：https://docs.pytorch.org/docs/2.13/data.html
3. PyTorch 2.13 — *Pipeline Parallelism*（torch.distributed.pipelining，alpha：pipeline/PipelineStage/GPipe/1F1B/Interleaved1F1B/LoopedBFS/ZeroBubble/DualPipeV、torch.export 追踪、静态形状、torchrun）：https://docs.pytorch.org/docs/2.13/distributed.pipelining.html
4. DeepSpeed — *ZeRO-Offload Tutorial*（offload_optimizer.cpu / DeepSpeedCPUAdam 5-7× / contiguous_gradients / overlap_comm / bind_cores_to_rank / 10B on single V100）：https://www.deepspeed.ai/tutorials/zero-offload/
5. NVIDIA — *How to Overlap Data Transfers in CUDA C/C++*（Mark Harris，2012；重叠三前提、copy engine、Hyper-Q、双缓冲两模式）：https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
6. NVIDIA — *How to Optimize Data Transfers in CUDA C/C++*（pinned memory 与传输优化前篇）：https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
7. GPipe：Huang et al., arXiv:1811.06965；PipeDream：Narayanan et al., arXiv:2006.09503
8. SiPipe（LLM 推理 CPU-GPU pipeline，TSEM 双缓冲 CUDA Graph，2.1×）：arXiv:2506.22033
9. v1 报告：`docs/CPU_GPU_异构并行_调研报告.md`（机制盘点、切法 A/B/C、论文价值评估）
10. FeatureStore 与 warm-start 设计：`docs/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`；`docs/EFM3_三专项落地设计_报错接口_特征预计算_极端价修正.md`
11. 本项目代码：`optim/perf_knobs.py`、`TimeMixer/repro_pipeline.py`、`RT916_SpikeFusionNet/src/rt916_spikefusionnet/core.py`、`runtime/resource_scheduler.py`

> 注：文中所有具体数值（t_feat≈3-7min、t_gpu 假设、传输时延、加速比）标注了来源性质；t_feat 来自项目 skill 特征预计算记录（历史实测），t_gpu 为假设区间，传输时延为 PCIe 带宽工程估算——落地前以本项目服务器实测为准。
