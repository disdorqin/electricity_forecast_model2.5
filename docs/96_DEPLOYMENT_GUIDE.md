# 96 点服务器部署与运行指南

> 适用：在 GPU 云服务器（智川云 / AutoDL 等）上部署并运行 96 点（15min）全链路。
> 目标：12 月预热 → 2026-01 起完整回测，`ledger_full_range` 全自动串联。
> 本文档为**完整可复制步骤**，含所有已知坑的规避方法。

---

## 0. 前置：确认代码最新

```bash
cd ~
git clone https://github.com/disdorqin/electricity_forecast_model2.5.git
# 若克隆慢：git clone https://ghfast.top/https://github.com/disdorqin/electricity_forecast_model2.5.git
# 备用代理：gh-proxy.com / mirror.ghproxy.com / gh.zwy.one（前缀拼接）

cd electricity_forecast_model2.5
git pull origin main        # 确保拉到全部 96 点修复
# 关键修复 commit: 308fc07 keep_cols / 345b931 append重建 / fbb1996 training merge
#                 702c5af GEF resolution / c82ecf5 postflight+fallback
git log --oneline -3        # 确认最新含以上修复
```

---

## 1. 环境准备

### 1.0 镜像版本（智川云 / AutoDL 选择）

**选择**：`PyTorch 2.x + CUDA 12.x + Python 3.10/3.11` 镜像（不要选基础 Ubuntu / TensorFlow / 纯 Miniconda）。

**版本约束**（代码实测依据）：

| 组件 | 要求 | 说明 |
|---|---|---|
| Python | **≥ 3.10** | `requirements.txt` 注明；3.11 最稳 |
| PyTorch | **≥ 2.0** | RT916/TimeMixer/TimesFM PyTorch 后端共用 |
| CUDA | **≥ 11.8**（推荐 12.x） | RT916 训练用 **BF16**，需 Ampere 架构(30系)+CUDA 11.8+ 硬件支持 |
| GPU 卡 | **RTX 3090 24GB 起** | 20 系(2080Ti)不支持硬件 BF16，不要选 |
| huggingface_hub | ≥ 0.23 | TimesFM 权重下载 |

**已验证版本**：RTX 3090 + PyTorch 2.x + CUDA 12.x 实测跑通（智川云）。本地开发机为 torch 2.13.0。

> 选 `PyTorch 2.5.x + CUDA 12.x` 或 `2.4.x + cu12x` 均可，优先官方 PyTorch 镜像。

### 1.1 pip 清华源

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 1.2 安装依赖

```bash
pip install -r requirements.txt
# 预测链路依赖 requirements 已全覆盖。
# 爬虫专属依赖(Crypto/PIL/gmssl/websocket)不需要——服务器只跑预测链路。
```

### 1.3 TimesFM 权重（必做）

```bash
# ① 配 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# ② 装 aria2 多线程下载器（若无）
which aria2c || (apt-get update && apt-get install -y aria2)

# ③ 下载 hfd 脚本 + 多线程下载权重（约 8GB，10-20 分钟）
wget https://ghfast.top/https://raw.githubusercontent.com/huggingface/hfd/main/hfd.sh
chmod +x hfd.sh
cd ~/electricity_forecast_model2.5
HF_ENDPOINT=https://hf-mirror.com ./hfd.sh google/timesfm-2.5-200m-pytorch \
  --local_dir models/timesFM --tool aria2c -x 16

# ④ 校验完整（必须出现 "safetensors OK"）
python -c "from safetensors import safe_open; f=safe_open('models/timesFM/model.safetensors', framework='pt'); print('safetensors OK')"
```

> 权重放 tmux 后台下，防断开。下载中断会报 `incomplete metadata`，删掉重下即可。

---

## 2. 上传数据（服务器不跑爬虫）

从本地 `scp` 上传（服务器无 Windows 爬虫依赖）：
```bash
# 在你本地电脑执行
scp -P <端口> data/shandong_pmos_96_full_v2.xlsx root@<区域>.autodl.com:~/electricity_forecast_model2.5/data/
scp -r -P <端口> data/remote_96 root@<区域>.autodl.com:~/electricity_forecast_model2.5/data/
```
- `data/shandong_pmos_96_full_v2.xlsx`：96 点合并宽表（30MB，长列名）
- `data/remote_96/parquet/`：96 点原始镜像（17MB）

---

## 3. 冒烟测试（先跑 1 天，确认全链路通）

```bash
cd ~/electricity_forecast_model2.5
export PROJECT_ROOT=$(pwd)
export TIMESFM_DEVICE=cpu          # 关键：TimesFM 强制 CPU，避免和 GPU 模型抢 CUDA
echo 'export TIMESFM_DEVICE=cpu' >> ~/.bashrc

nohup python main.py --pipeline ledger_predict --date 2025-12-01 --resolution 15min \
  --data-path data/shandong_pmos_96_full_v2.xlsx \
  --ledger-root outputs/ledger_96 --runs-root outputs/runs_96 \
  > outputs/smoke.log 2>&1 &
tail -f outputs/smoke.log
```

**通过标准**（约 17 分钟）：
```
dayahead long table: 288 rows
realtime long table: 384 rows
Prediction ledger [dayahead]: 0 → 288 rows
Actual ledger [dayahead]: 0 → 96 rows
```
出现 `288/384` = 96 点正确（不是 72/96 的 24 点错误）。

---

## 4. 预热 + 全链路自动串联

确认冒烟通过后，跑完整串联（预热 12 月 → 自动接全链路）：

```bash
cd ~/electricity_forecast_model2.5
export PROJECT_ROOT=$(pwd)

tmux new -s run -d "bash scripts/auto_preheat_backtest.sh"
tmux ls                          # 应看到 run 会话
```

脚本逻辑（`scripts/auto_preheat_backtest.sh`）：
1. `ledger_backfill 2025-12-01~12-31`（预热，~8h）
2. 校验账本 ≥ 30 天
3. `ledger_full_range 2026-01-01~08-02`（全链路，~3 天，自动接续）

---

## 5. 进度查看

```bash
tail -f outputs/auto_preheat.log      # 预热阶段
tail -f outputs/auto_backtest.log     # 全链路阶段
tmux ls                               # 会话存活
ls outputs/runs_96/ | grep "^2026" | wc -l    # 已跑天数
```

---

## 6. 已知坑（务必规避）

### 6.1 TimesFM 抢 CUDA → 必须设 TIMESFM_DEVICE=cpu
TimesFM 是 CPU 模型，但代码默认 `cuda:0`。和 rt916/timemixer 并发时抢 CUDA init →
`CUDA error: initialization error` 刷屏。**每次新终端跑前 `export TIMESFM_DEVICE=cpu`**。

### 6.2 预热必须 96 点（15min），否则账本污染
若跑了 hourly 的 12 月数据，账本会 24 点（DA 72 / RT 96 行），preflight 全崩。
**冒烟阶段就确认 `288/384` 行**，不对立即停。

### 6.3 账本 business_period 必须完整
预测 CSV 无 business_period → 账本压成 24 点。当前代码已修（keep_cols + append 重建），
**拉最新代码后旧缓存可能缺列，用重建脚本补救**：
```bash
python scripts/rebuild_prediction_ledger_96.py --runs-root outputs/runs_96 --ledger-root outputs/ledger_96
```

### 6.4 GPU 单卡，勿开第二进程
`max_gpu_workers=1` 固定，GPU 任务串行。再开一个月进程会抢 GPU 崩溃。
**纯 CPU 分析脚本（analyze_96_ledger）可并行跑**。

### 6.5 git push 代理问题
本地若配了 `http.proxy=127.0.0.1:7890` 且代理失效，push 会失败。绕过：
```bash
git -c http.proxy= -c https.proxy= push origin main
```

---

## 7. 完整链路产物（跑完后应看到）

```
outputs/runs_96/2026-01-01/
├── dayahead/prediction/   各模型预测 + long表(288行)
├── dayahead/weight/       weights.csv（真30天动态权重）
├── dayahead/fuse/         fused_predictions.csv
├── realtime/prediction/
├── realtime/weight/
├── realtime/fuse/
├── realtime/final/        未修正 + 修正后
└── final/submission_ready.csv   96行, 7列契约
```

---

## 8. 耗时与成本参考（RTX 3090）

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 冒烟 1 天 | ~17 min | 全链路含 RT916 训练 |
| 预热 12 月 | ~8 h | 31 天 × 17min |
| 全链路 01-01~08-02 | ~3 天 | 214 天 × 17-20min |
| 智川 3090 单价 | 0.99 元/时 | 关机不按 GPU 计费 |

---

## 9. 故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| CUDA initialization error 刷屏 | TimesFM 抢 GPU | `export TIMESFM_DEVICE=cpu` |
| 长表 72/96 行 | 跑了 hourly | 用 `--resolution 15min` 重跑 |
| preflight KeyError hour_business | 账本 24 点 | 重建账本脚本 |
| coverage 4 倍爆炸 | 旧代码 | `git pull` 到 fbb1996+ |
| weights.csv 空 | GEF 没传 resolution | `git pull` 到 702c5af+ |
| 服务器被抢/关机 | 平台回收 | 重开卡，按本文档重来（数据盘若保留可跳过上传） |
