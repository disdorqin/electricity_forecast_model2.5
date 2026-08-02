#!/usr/bin/env bash
# auto_preheat_backtest.sh
# 阶段1: 2025-12 预热（只写账本，已跑过的日子 cache hit 秒过）
# 阶段2: 预热账本>=30天后，自动接 2026-01-01 起全链路回测（weight+fuse+classifier+final）
# 用法: bash scripts/auto_preheat_backtest.sh   （建议放 tmux 里跑）
set -u
cd ~/electricity_forecast_model2.5 || { echo "cd 失败"; exit 1; }
export PROJECT_ROOT="$(pwd)"

echo "===== [$(date)] 阶段 1/2: 2025-12 预热开始 ====="
python main.py --pipeline ledger_backfill \
  --start 2025-12-01 --end 2025-12-31 \
  --resolution 15min --ledger-root outputs/ledger_96 --runs-root outputs/runs_96 \
  --data-path data/shandong_pmos_96_full_v2.xlsx \
  > outputs/auto_preheat.log 2>&1
echo "===== [$(date)] 预热结束（日志: outputs/auto_preheat.log）====="

echo "===== [$(date)] 校验预热账本天数 ====="
python -c "
import pandas as pd
df = pd.read_parquet('outputs/ledger_96/realtime/prediction/prediction_ledger.parquet')
days = df['target_day'].nunique()
print(f'账本覆盖天数: {days}')
assert days >= 30, f'预热账本不足30天({days})，中止全链路'
" || { echo "账本不足30天，停止"; exit 1; }

echo "===== [$(date)] 阶段 2/2: 全链路回测 2026-01-01 ~ 2026-08-02 开始 ====="
python main.py --pipeline ledger_full_range \
  --start 2026-01-01 --end 2026-08-02 \
  --resolution 15min --ledger-root outputs/ledger_96 --runs-root outputs/runs_96 \
  --data-path data/shandong_pmos_96_full_v2.xlsx \
  --skip-existing-final --continue-on-error \
  > outputs/auto_backtest.log 2>&1
echo "===== [$(date)] 全链路回测结束（日志: outputs/auto_backtest.log）====="

echo "===== [$(date)] 全部完成 ====="
