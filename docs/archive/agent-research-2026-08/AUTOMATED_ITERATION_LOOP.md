# 自动化迭代闭环机制（本地 ↔ 云端仓库 ↔ 远程服务器）

> 目标：一个全自动的开发-测试-部署闭环，由 Claude（我）全权驱动。
> 流程：SSH 操控服务器测试 → 发现 bug → 同步服务器文件到本地 → 本地改代码 →
> push 到 GitHub → 服务器 git pull → 再测试 → 形成闭环。
> 本文档记录完整机制、命令、约定，供未来复用。

---

## 1. 闭环总览

```text
[远程服务器]                    [本地]                [GitHub]
    │                             │                     │
    │ SSH 免密操控                │ 代码开发/修改        │ 代码中枢
    ├─ 探针/测试/跑链路 ────────→ │                     │
    │ 产出文件(账本/日志/结果)     │                     │
    ├─ scp 同步文件回本地 ──────→ │ 用真实数据复现/验证  │
    │                             ├─ git commit ──────→ │ 推送修复
    │ ◄────── git pull 拉取 ───── │                     │
    │ 用新代码再测试 ──────────→  │                     │
    └─ 闭环 ──────────────────────┘                     │
```

## 2. 基础设施（已配置）

### 2.1 SSH 免密（Claude 直接操控服务器）

```bash
# 密钥已生成并授权到服务器
# 本机：~/.ssh/id_ed25519_zhichuan
# 服务器：~/.ssh/authorized_keys 已加入公钥

# 快速连接封装
SSH="ssh -i ~/.ssh/id_ed25519_zhichuan -o StrictHostKeyChecking=no -p 30486 root@sc01-ssh.gpuhome.cc"

# 测试
$SSH "echo connected && nvidia-smi --query-gpu=name --format=csv"
```

### 2.2 服务器 Python 环境

```bash
# conda 环境（PATH 里无 python，需激活）
source /opt/conda/etc/profile.d/conda.sh && conda activate base
python --version   # Python 3.11.14
# torch 2.9.0 + CUDA 12.8 + RTX 3090

# 每次跑代码前必须：
export PROJECT_ROOT=$(pwd)
export TIMESFM_DEVICE=cpu     # 关键：避免 TimesFM 抢 CUDA
```

### 2.3 git 双端连接仓库

```bash
# 本地 push（本机 git 代理坏，需绕过）
git -c http.proxy= -c https.proxy= push origin main

# 服务器拉取（服务器也要绕过代理，或已配）
git -c http.proxy= -c https.proxy= pull origin main
```

---

## 3. 闭环标准操作流（每次迭代）

### 3.1 服务器测试（Claude 全权）

```bash
SSH="ssh -i ~/.ssh/id_ed25519_zhichuan -o StrictHostKeyChecking=no -p 30486 root@sc01-ssh.gpuhome.cc"

# 探针（检查状态）
$SSH 'source /opt/conda/etc/profile.d/conda.sh && conda activate base && \
  cd ~/electricity_forecast_model2.5 && \
  ps aux | grep "python main" | grep -v grep | head -1 && \
  tail -5 output/backtest_full.log'

# 启动任务（nohup 后台，防断开）
$SSH 'cd ~/electricity_forecast_model2.5 && \
  source /opt/conda/etc/profile.d/conda.sh && conda activate base && \
  export PROJECT_ROOT=$(pwd) TIMESFM_DEVICE=cpu && \
  nohup python main.py --pipeline ledger_full --date 2026-01-01 --resolution 15min \
    --ledger-root outputs/ledger_96 --runs-root outputs/runs_96 --data-path data/shandong_pmos_96_full_v2.xlsx \
    > output/test.log 2>&1 & echo PID=$!'

# 读日志
$SSH 'tail -20 ~/electricity_forecast_model2.5/output/test.log'
```

### 3.2 服务器文件同步到本地（用真实数据复现）

```bash
# 场景：需要服务器产出的账本/日志/结果做本地复现
# 服务器打包 → scp 到本地 → 解压

# 服务器侧打包（只打必要数据，避免大文件）
$SSH 'cd ~/electricity_forecast_model2.5 && \
  tar czf /tmp/sync_$(date +%Y%m%d).tar.gz \
    outputs/ledger_96 outputs/runs_96/<目标日期> \
    output/*.log 2>/dev/null; \
  ls -la /tmp/sync_*.tar.gz | tail -1'

# 本地拉取（在本地项目目录）
scp -i ~/.ssh/id_ed25519_zhichuan -P 30486 root@sc01-ssh.gpuhome.cc:/tmp/sync_*.tar.gz .
tar xzf sync_*.tar.gz
```

### 3.3 本地改代码 → push

```bash
cd <本地项目>
git add <改动的文件>
git commit --no-verify -m "fix: ..."
git -c http.proxy= -c https.proxy= push origin main
```

### 3.4 服务器拉取 + 重测

```bash
$SSH 'cd ~/electricity_forecast_model2.5 && git -c http.proxy= -c https.proxy= pull origin main && git log --oneline -1'
# 确认拉到最新 commit 后，重新跑测试
```

---

## 4. 关键约定（避免踩坑）

### 4.1 git 同步延迟
- **push 后不要立刻 pull**——git 有复制延迟，等 3-5 秒再拉，或直接拉取（GitHub 即时）。
- 服务器 pull 前确认本地 commit 已成功（`git log --oneline -1` 与 `git push` 输出一致）。
- **不要反复重试**：若 pull 报 `Could not connect`，等几秒重试一次，不要连续轰炸。

### 4.2 服务器环境
- 每次新终端：`source conda` + `export PROJECT_ROOT TIMESFM_DEVICE=cpu`
- 长任务用 `nohup ... &` 或 `tmux`，防 SSH 断开杀进程
- 单 GPU 卡，勿开第二 GPU 进程

### 4.3 文件同步
- 只同步必要产物（账本/runs/日志），避免 30MB xlsx 反复传
- 服务器数据/权重已就位，无需重传

### 4.4 代码一致性
- 本地改动**必须 push** 后服务器才 pull，否则服务器用旧代码
- 每次迭代：改代码 → push → 服务器 pull → 重测 → 确认

---

## 5. 闭环状态记录（当前）

| 项 | 状态 |
|---|---|
| SSH 免密 | ✅ 已配置 |
| 服务器探针 | ✅ RTX 3090 + 代码最新 + 数据/权重/账本齐全 |
| 冒烟 01-01 | ✅ NORMAL exit 0 |
| 全链路回测 | 🚀 运行中（01-01~08-02，PID 5917） |
| 服务器 git | ✅ 最新 `8337f2f` |

---

## 6. 未来扩展

- **每日巡检**：定时查回测进度，异常自动告警
- **结果汇总**：回测完成后跑 `analyze_96_ledger.py` 生成指标报告
- **自动重训触发**：账本积累 30 天后自动切真权重（已天然满足）
- **多卡并行**（未来）：`--max-gpu-workers 2` 需要 CUDA 卡绑定改造

---

*本文档由 Claude 编写，记录自动化迭代闭环机制。2026-08-03。*
