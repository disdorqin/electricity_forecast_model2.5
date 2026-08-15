# EFM3 电力预测项目 — Agent 强制约束

> 本文件对本 git 仓库**所有** agent 会话强制生效（仅本文件夹，不影响其他项目）。
> 作用：让 agent 在任何任务开始时自动加载经验教训 skill，防止重复踩坑。

## 强制规则（启动时自动执行）

1. **会话启动时**：调用 `skill` 工具加载 `efm3-lessons`（`.opencode/skills/efm3-lessons/SKILL.md`）。
   - 若 skill 工具不可用，直接读取该文件全文。
2. **关键阶段**：在执行以下动作**之前**，必须重新过一遍 `efm3-lessons` 的对应小节：
   - 改动任何数据文件 / 爬虫代码 → 读「§1 数据真实性红线」
   - 改动模型 / 特征 / 管道 → 读「§2 24 vs 96 口径」「§3 交付纪律」
   - 运行训练 / 回测 / 交付 → 读「§3 交付纪律」「§4 环境约定」
3. **写代码/改文件前**，对照「§6 执行 checklist」逐项自检。
4. **踩坑/修复后**：把新教训写回 `efm3-lessons` skill，并写共享记忆（`memory_put`，category=`domain:efm3` 或 `mech`）。

## 本机环境速查（详见 skill §4）

- Python: `conda epf-2`（CPU only），路径 `D:/computer_download/environment/conda/epf-2/python.exe`
- 主链路入口：`python main.py ...`
- 96点同步：`python main.py --pipeline sync_dataset --resolution 15min --sync-source db`
- 交付五阶段：`ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs`
- 回归：`scripts/check_delivery_stability.py` 等 4 件套
