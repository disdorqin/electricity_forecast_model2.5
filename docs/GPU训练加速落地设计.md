# GPU 训练加速落地设计（TimeMixer / RT916）— 整合 5 份调研

> 日期：2026-08-15。整合：
> `TimeMixer_RT916_训练耗时热点调研报告`、`PyTorch底层算子_CUDA自定义算子_训练加速调研报告`、
> `矩阵计算底层优化_调研报告`、`CPU_GPU_并行调度与数据流_深化调研报告`、`CPU_GPU_异构并行_调研报告`。
>
> **核心结论（全部调研一致）**：两个 GPU 模型慢的**不是算力**（RTX3090 利用率个位数），
> 而是 **kernel launch 数量 + Python 调度 + CPU-GPU 同步点**。模型太小（hidden 64/128, batch 16-64），
> 属 launch-bound/memory-bound，不是 compute-bound。

---

## 一、真实瓶颈定位（已代码级确认）

### 1.1 隐藏雷：`set_global_seed` 二次关闭优化（最高优先，零风险）

`utils/reproducibility.py:34-38`：
```python
torch.backends.cudnn.benchmark = False                    # 强制关 benchmark
torch.set_float32_matmul_precision("highest")             # 强制关 TF32
```
而 `optim/perf_knobs.py` / `core.py:44` / `repro_pipeline.py:33` 想开 benchmark/TF32。
→ **每次训练前 set_global_seed 把它们二次关掉**。且 `core.py:432` 训练时也设 False。

**修复**：set_global_seed 尊重 perf_knobs 的环境开关（`OPTIM_CUDNN_BENCHMARK`），不无条件关；
TF32 用新的 `fp32_precision` API 且可开关。**预期收益 20-40%**（卷积自动调优 + TF32）。

### 1.2 RT916：`.item()` 同步 + Python loop（最大单点收益）

`model.py:100`：
```python
period = int(period_list[i].item())   # 每个 forward 同步 CPU-GPU
for i in range(period_list.shape[0]):  # Python loop
    ... conv 等 ...
```
每 forward 多次全同步（官方 tuning guide 明令禁止），且 period 循环无法被 torch.compile 融合。

**修复方向**：批量化 period 处理（把不同 period 的 reshape+conv 用统一 padding 到 max_period 向量化）；
或至少把 `.item()` 用 `detach().cpu()` 批量取，减少同步次数。

### 1.3 TimeMixer：6 模型 × 80 epoch 结构冗余 + 逐日特征重算

- 6 个段模型（3 段 × DA/RT）× 80 epoch 都从零训，相邻段/相邻日高度相关 → **warm-start 续训**可省大量。
- 每步 `.item()` 同步（同 RT916 问题）。
- 多尺度 `season_mlps/trend_mlps` 是 3 个独立小 Linear → 可合并成一发大 Linear（6 launch → 2）。

### 1.4 CPU 特征工程 vs GPU 计算

- 两模型已用 `num_workers=4 + pin_memory + non_blocking + prefetch=2`（DataLoader 级重叠已吃满）。
- 逐日 pandas 特征重算 × 6 模型 = CPU 侧主要耗时 → **特征预计算 FeatureStore** 解决。

---

## 二、加速方案（按性价比排序）

| 优先级 | 措施 | 类型 | 预期收益 | 风险 |
|---|---|---|---|---|
| **P0** | 修 `set_global_seed` 二次关闭 benchmark/TF32 | 配置修复 | 20-40% | 零（保持可复现开关）|
| **P0** | `optim.Adam(fused=True)` + `torch.compile(mode="reduce-overhead")` 包 backbone | 一行改动 | 1.3-2.5× | 低（RT916 需 fullgraph=False）|
| **P1** | RT916 `.item()` → 批量取/向量化 period 循环 | 代码重构 | 最大单点 | 中（需回归验证）|
| **P1** | TimeMixer 3 scale MLP 合并成一发 Linear | 架构微调 | 30-50% launch 减少 | 低 |
| **P1** | 特征预计算 FeatureStore（GPU 模型的 CPU 特征前置）| 数据层 | 每期省 30-70% | 零精度损失（逐位 diff）|
| **P1** | warm-start 续训（相邻日权重）| 训练策略 | 3天→1天 主支柱 | 低（防泄漏铁律）|
| **P2** | RT916 FFT 隔离（FP32 往返减少）| 代码 | 中 | 中 |
| **P2** | Triton fused kernel（TimeMixer MovingAvg / RT916 Spike 统计量）| 自定义算子 | 5-20% | 中（需装 triton）|
| **P2** | 固定 batch 形状 + warmup + CUDA Graph | 工程 | 配合 compile | 低 |
| **P3** | CUTLASS 定制 GEMM | 底层 | **不推荐**（小矩阵无优势）| 高 |

---

## 三、CPU-GPU 并行（用户 idea 深化）

### 3.1 结论：日级双缓冲，但顺序要对

- **批级重叠已吃满**（DataLoader 已配 num_workers+pin_memory）→ 不加 side stream（收益≈0 且复杂）。
- **真正可做**：日级双缓冲 `DayPrefetcher`（线程 + 双缓冲 + 故障回退，<100 行）——CPU 提前算 target+1 特征，GPU 训 target。
- **加速比敏感于 t_gpu**：t_gpu=60min→1.05×，t_gpu=8min→1.38-1.88×。
- **正确顺序**：FeatureStore（把 t_feat 压到秒级）→ warm-start（压 t_gpu）→ 双缓冲降级兜底。
  FeatureStore 已把特征压到秒级后，双缓冲收益塌缩——**先做前两者，双缓冲作兜底**。

### 3.2 生产单日：日内 DA→RT 两腿流水
- RT 特征依赖 da_anchor → **先跑 DA 腿（含特征物化），再跑 RT 腿**，两腿间 CPU 特征可与 GPU 训练重叠。

---

## 四、C++/CUDA 自定义算子：需要吗？

**结论：不需要 C++/CUDA，Triton 足够，且多数情况 torch.compile 就够。**

- 两模型是 launch-bound 小模型，**Triton fused kernel**（比 C++ 扩展易写）可做 3 个点：
  1. TimeMixer MovingAvg（AvgPool1d+相减 ~5 kernel → 1）
  2. RT916 Spike 分支统计量（diff/mean/std/z/soft_mask 15+ elementwise → 1）
  3. RT916 周期分支（修 .item() 后，三周期折叠+多尺度 conv 融合）
- **但 P0 的 torch.compile 已自动做大部分融合**，Triton 是 P2 补强。
- CUTLASS 对本项目小矩阵**无优势**（memory-bound，AI≈21 < 3090 ops:byte），不推荐。

---

## 五、实施顺序（建议）

```
阶段 A（本机可验证，零风险）：
  ① 修 set_global_seed 二次关闭 → 跑 GPU smoke 对比训练时间
  ② optim.Adam(fused=True) + torch.compile 包 backbone → 服务器 A/B
阶段 B（服务器）：
  ③ RT916 .item() 向量化 → 回归验证
  ④ TimeMixer MLP 合并 → 回归
  ⑤ FeatureStore 接入 GPU 模型（CPU 特征前置）
  ⑥ warm-start 续训
阶段 C（可选优化）：
  ⑦ DayPrefetcher 双缓冲（兜底）
  ⑧ Triton fused kernel（P2）
```

**验收**：服务器 RTX3090 上 214 天回测墙钟 3 天 → ≤1 天；单日 NORMAL 不变；指标不劣化（只升不降）。
每步先 1-2 天小窗 A/B，再全量。全程守 efm3-lessons skill（数据真实性/防泄漏/交付纪律）。

---

## 六、风险与纪律

- **可复现性**：修 set_global_seed 必须保留 deterministic 开关，默认非 determinism 才开 benchmark/TF32。
- **torch.compile 兼容**：RT916 有 `.item()` + Python loop → graph break，用 `fullgraph=False` 或先修 .item()。
- **防泄漏**：warm-start 续训训练窗仍只到 target-1 天；FeatureStore 遵守 cutoff=14/p56。
- **每步验证**：改一处测一处、A/B 对照、4 件套+黄金基线+健康检查全绿。
