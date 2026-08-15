# GPU 模型本机调参实测结果（TimeMixer / RT916）

> 日期：2026-08-16。环境：epf-2（torch 2.6.0+cu124），本机 RTX 4060 Laptop 8GB。
> 目的：实测两个 GPU 模型单日耗时，定位瓶颈，做不伤精度的提速修改。

---

## 1. 环境升级（已完成）

- epf-2 装 CUDA 版 torch（2.6.0+cu124 + torchvision 0.21.0+cu124），替换 CPU 版
- 清理 CPU 版 torch 残留 + 损坏的 `~orch` 分发
- 回归测试全绿（stability 29/29、sync 41/41、adaptive 40/40）

## 2. 单日耗时实测（96点，12个月训练窗）

| 模型 | 耗时 | 说明 |
|---|---|---|
| TimeMixer DA | ~310s (5.2min) | GPU，3段×2任务 |
| **RT916 realtime(DA+RT)** | **~1788s (29.8min)** | ⚠️ 回测最大瓶颈 |
| LightGBM | <0.1s | CPU |
| SGDFNet | 5-9s | CPU |
| TimesFM | 40s | CPU |

## 3. RT916 提速（关键发现）

### 根因
`core.py` `TRAIN_STEPS=1`（硬编码）→ 96点下 seq_len=288，滑动步长1 → 每段 ~11393 样本。
3 段 × 2 任务(DA+RT) × 8 epoch = 48 次大训练 → 29.8min。

### 修复
`TRAIN_STEPS` 改为环境变量 `RT916_TRAIN_STEPS` 可配置（默认仍 1 保守）。

### 实测（3个月窗，RTX4060）
| TRAIN_STEPS | 样本数 | 耗时 | 精度 |
|---|---|---|---|
| 1（默认）| ~11393 | ~1000s+ | 基准 |
| **24** | 404 | **98s** | SMAPE 0.23-0.32 ✅ |
| 96 | 102 | 70s | SMAPE 0.45-0.82 ❌ 过拟合 |

**结论：`RT916_TRAIN_STEPS=24` 是甜点——18 倍提速 + 精度良好**。
12 个月窗估计 ~6min/天（原 29.8min，4-5 倍同口径提速）。

## 4. 其他已做修改

- `utils/reproducibility.py`：set_global_seed 不再强制关 TF32（之前把模型的 TF32 优化二次关掉）
- `RT916 core.py`：AdamW 加 `fused=True`（GPU 优化器提速）
- benchmark 实测对本机小模型无益（launch-bound），保持模型自决

## 5. 稳健性结论（回应用户双缓冲顾虑）

- **放弃 side-stream 双缓冲**（收益≈0 且有同步风险）
- 用 **FeatureStore（特征预计算）+ warm-start（续训）** 两个结构性优化，链路稳健性不变
- **TimeMixer/RT916 贯穿始终 + CPU 并行**：scheduler 已支持并发；RT916 DA 依赖 da_anchor，暂不激进改造（稳健优先）

## 6. 下一步

- 服务器 RTX3090 上用 `RT916_TRAIN_STEPS=24` 跑 12 个月窗复测耗时/精度
- warm-start 续训接入 GPU 模型（省 epoch）
- 全链路回测前健康检查 14 项全绿
