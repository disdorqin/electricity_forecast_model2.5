# EFM3 电力预测项目 — Agent 强制约束

> 本文件对本 git 仓库**所有** agent 会话强制生效（仅本文件夹，不影响其他项目）。
> 作用：让 agent 在任何任务开始时自动加载经验教训 skill，防止重复踩坑。

## 强制规则（启动时自动执行）

1. **会话启动时**：调用 `skill` 工具加载 `efm3-lessons`（`.agents/skills/efm3-lessons/SKILL.md`）。
   - 若 skill 工具不可用，直接读取该文件全文。
2. **关键阶段**：在执行以下动作**之前**，必须重新过一遍 `efm3-lessons` 的对应小节：
   - 改动任何数据文件 / 爬虫代码 → 读「§1 数据真实性红线」
   - 改动模型 / 特征 / 管道 → 读「§2 24 vs 96 口径」「§3 交付纪律」
   - 运行训练 / 回测 / 交付 → 读「§3 交付纪律」「§4 环境约定」
3. **写代码/改文件前**，对照「§6 执行 checklist」逐项自检。
4. **踩坑/修复后**：把新教训写回 `efm3-lessons` skill，并写共享记忆（`memory_put`，category=`domain:efm3` 或 `mech`）。

## 模型、分辨率与质量门控约束

5. **生产模型池唯一来源**：生产代码必须从
   `fusion/model_pool.py` 读取 `DAYAHEAD_MODELS` / `REALTIME_MODELS`，禁止在
   pipeline、校验器和脚本中重复声明模型列表。
   - DA：`lightgbm, timesfm, timemixer`
   - RT：`timesfm, sgdfnet, timemixer, rt916`
   - LightGBM 的 RT 入口保留用于历史追溯，但必须拒绝生产调用；TimesFM RT 是当前正式候选。
6. **融合质量门控**：`ledger_fuse` 必须记录每个 `(task, period)` 的权重门控结果。
   权重低于 `--weight-prune-threshold` 的模型不得进入该段最终融合；至少保留一个最高权重模型，并在
   `model_quality_gate.csv`、`fused_debug.csv` 和 manifest 中可审计。关闭门控只能显式传 `--weight-prune-threshold 0`。
7. **分辨率约束**：通用生产代码不得新增裸写的 `24/96/8/32` 行数逻辑；统一从
   `utils/resolution.py` 获取槽位、时段和时间映射。24点专用测试必须明确标注 hourly-only。
8. **信息可得性约束**：新增特征必须注明来源、角色（forecast/actual/target/lag）、可得时间和允许任务；target-day `actual_*`、cutoff之后RT真值不得直接进入特征。

## 文档与产物约束

9. **文档唯一入口**：当前生效文档以 `docs/README.md` 和 `docs/DOCUMENT_ARCHITECTURE.md` 为准。代码行为变化必须同步核对根目录 `README.md` 和对应生效文档；若两者冲突，以代码和测试为准，并在同一任务中修正文档。普通内容必须更新已有负责文档，不能随意新建。
10. **研究材料归档**：调研报告、Agent过程记录、旧方案、历史验收和调参草稿统一放在 `docs/archive/`，不得继续堆在 `docs/` 当前生效区。归档不删除，必须保留可追溯路径。
11. **新增文档必须先请求批准**：智能体发现确实需要新增文档时，只能在回复中提出申请，说明名称、所属领域、不能合并的原因、维护内容和验证方式；在用户明确同意前，不得创建该文件。获批后，新文档首部注明 `status: active|historical|archived`、日期、责任领域和验证依据，并登记到 `docs/README.md`。
12. **运行产物隔离**：预测账本、runs、实验输出、日志和模型权重不得作为源码修改提交；需要共享时提交 manifest、摘要和哈希，不提交大体量生成物。

12b. **链路输出隔离**：原链路保留 24 点 `outputs/ledger/` + `outputs/runs/`、96 点
     `outputs/ledger_96/` + `outputs/runs_96/` 作为 legacy 兼容区。新链路必须显式传
     `--output-profile feature_store`，使用 `outputs/24/feature_store/` 或
     `outputs/96/feature_store/` 下的 `ledger`、`runs`、`cache`，不得与原链路混写。
     旧 `outputs/feature_store_chain/` 已归档，只读不再作为写入目标。

12c. **数据域隔离**：24 点 canonical 输入位于 `data/24/canonical/`；96 点权威实际
     唯一位于 `data/96/authoritative/pmos_96_全量.csv`，仅作实际值校验，不得冒充模型宽表。
     96 点价格/预测模型输入位于 `data/96/model_input/`，远端镜像位于 `data/96/remote/`。
     新增或迁移数据必须先运行 `scripts/tests/check_96_vs_24_actual.py`。

12d. **历史96点预测实验数据污染标记（2026-08-17）**：
     `data/96/model_input/shandong_pmos_96_model_input.xlsx` 及其历史副本
     `data/96/quarantine/legacy_root/shandong_pmos_96_full_v2.xlsx` 的电网特征存在
     `actual_*` 与 `fcast_*` 大面积重复，标记为 `historical-invalid-features`。
     由该输入生成的历史预测账本
     `outputs/ledger_96/{dayahead,realtime}/prediction/prediction_ledger.parquet`
     以及对应 `actual_ledger.parquet` **仅允许用于权重学习器/融合器相对实验**，不得用于
     宣称真实数据精度、重新训练生产模型或作为生产数据源。
     已核验该历史账本的 `y_true` 与旧宽表的 `日前电价`/`实时电价`逐点一致，因此可作为
     本次融合结构实验的价格标签；但该结论不解除特征污染标记。
     实验必须直接读取预测账本和价格实际账本，不得重新从上述污染宽表构造特征；实验产物
     统一放在 `outputs/experiments/`，待新鲜有效预测集建立后重新验证，并将旧实验结果归档清理。

## 依赖与复现约束

13. **依赖版本**：以根目录 `requirements.txt` 为唯一项目依赖入口；当前验证基线为 Python 3.11.14、Torch 2.6.0+cu124、NumPy 1.26.4、JAX/JAXLIB 0.4.30。TimesFM 必须使用项目内置 `TimesFMBackend/src/timesfm`，禁止直接 `pip install timesfm`。
14. **复现记录**：训练、回测或交付前记录 Python、Torch/CUDA、JAX、数据快照、git commit、seed、resolution、cutoff 和权重门控阈值。

## 本机环境速查（详见 skill §4）

- Python: `conda epf-2`，路径 `D:/computer_download/environment/conda/epf-2/python.exe`；当前环境已验证 CUDA Torch，CPU-only仅用于轻量模型和回归
- 主链路入口：`python main.py ...`
- 96点同步：`python main.py --pipeline sync_dataset --resolution 15min --sync-source db`
- 交付五阶段：`ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs`
- 回归：`scripts/check_delivery_stability.py` 等 4 件套
