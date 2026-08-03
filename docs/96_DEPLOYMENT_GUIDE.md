# 96 点服务器部署与运行指南

> 适用：在 GPU 云服务器（智川云 / AutoDL 等）上部署并运行 96 点（15min）全链路。
> 目标：12 月预热 → 2026-01 起完整回测，`ledger_full_range` 全自动串联。
> 本文档为**完整可复制步骤**，含所有已知坑的规避方法。

---

## 0.1 智川云服务器连接信息（当前在用）

**主机信息**（智川云，容器实例）：

| 项 | 值 |
|---|---|
| 地址 | `sc01-ssh.gpuhome.cc` |
| 端口 | `30486` |
| 用户名 | `root` |
| 密码 | `vm5fdqav` |
| 连接命令 | `ssh root@sc01-ssh.gpuhome.cc -p 30486` |

**VSCode SSH 连接**（推荐）：
1. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host...` → `Add New SSH Host`
2. 粘贴 `ssh root@sc01-ssh.gpuhome.cc -p 30486`，选默认 config 文件
3. Connect → 输入密码 `vm5fdqav`
4. 成功后左下角显示 `SSH: sc01-ssh.gpuhome.cc`

**SSH config 快捷别名**（写入 `~/.ssh/config` 后可 `ssh zhichuan` 一键连）：
```text
Host zhichuan
  HostName sc01-ssh.gpuhome.cc
  User root
  Port 30486
```

**注意**：
- 密码是明文记录，若该卡共享/退租后需改密码（服务器 `passwd`）。
- 智川容器为按量计费，关机释放 GPU 但保留数据盘；重新开卡后数据盘是否保留需在控制台确认，若不保留需重新上传数据/权重。

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

### 1.0 镜像版本（智川云选择）

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

# ② 多线程下载权重（约 8GB，10-20 分钟）——不依赖 hfd.sh/ghfast 代理
cd ~/electricity_forecast_model2.5
mkdir -p models/timesFM
hf download google/timesfm-2.5-200m-pytorch --local-dir models/timesFM --max-workers 8

# 若提示 hf: command not found，先升级再重跑上面命令：
pip install -U "huggingface_hub[cli]"

# ③ 校验完整（必须出现 "safetensors OK"）
python -c "from safetensors import safe_open; f=safe_open('models/timesFM/model.safetensors', framework='pt'); print('safetensors OK')"
```

> 权重放 tmux 后台下，防断开：`tmux new -s dl -d "hf download google/timesfm-2.5-200m-pytorch --local-dir models/timesFM --max-workers 8"`。
> 下载中断/损坏会报 `incomplete metadata` 或校验失败，删 `models/timesFM/` 重下即可。
> 注：ghfast 等代理拉 hfd.sh 实测返回 404，故不采用 hfd.sh 方案。

---

## 2. 上传数据（服务器不跑爬虫）

从本地 `scp` 上传（服务器无 Windows 爬虫依赖），**目标为此台智川云**（`sc01-ssh.gpuhome.cc:30486`）：
```bash
# 在你本地电脑（Windows PowerShell / git bash）执行，密码 vm5fdqav
# 注意：需先在本地 cd 到本项目根目录，或在 scp 前用本地绝对路径
cd <本地项目根目录>   # 例如 D:\作业\...\electricity_forecast_model2.5
scp -P 30486 data/shandong_pmos_96_full_v2.xlsx root@sc01-ssh.gpuhome.cc:~/electricity_forecast_model2.5/data/
scp -r -P 30486 data/remote_96 root@sc01-ssh.gpuhome.cc:~/electricity_forecast_model2.5/data/
```
- `data/shandong_pmos_96_full_v2.xlsx`：96 点合并宽表（30MB，长列名）
- `data/remote_96/parquet/`：96 点原始镜像（17MB）

**上传后验证**（在服务器 VSCode 终端）：
```bash
ls -la ~/electricity_forecast_model2.5/data/shandong_pmos_96_full_v2.xlsx
ls ~/electricity_forecast_model2.5/data/remote_96/parquet/
```
两个都存在即上传成功。

> 若服务器数据盘保留了上次的 data/，可跳过上传，先 `ls data/shandong_pmos_96_full_v2.xlsx` 确认存在。

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
