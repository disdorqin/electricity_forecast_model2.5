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
4a. **先做影响面、再改代码**：生产链路中的 parser/runner/scheduler/model adapter/ledger/report 等存在强耦合。修改前必须检索调用方、被调用方、测试和生效文档，明确“直接改动 + 受牵连部分”；不能只修眼前报错而忽略相邻契约。
4b. **最小化修改 + 同步回归**：优先在最窄责任层修复，不顺手重构、不扩大算法范围。每个生产行为变化必须同时补能覆盖真实调用路径的最小回归，并同步更新对应 active 文档/skill；轻量单测 PASS 不能替代真实入口 smoke。
4c. **第一性原理 + 最小熵增**：设计新能力时先问“为满足业务正确性、训练/预测一致性和可审计性，最少需要新增什么状态与抽象”。禁止为了理论完备引入模型实际不消费的元数据层、重复数据表示或多套并行契约；能用一个共享入口解决的问题，不在五个模型内各自实现。优先复用现有列名、runner、adapter 和 parquet 形态，只在确有收益或安全需要时增加新字段/新文件/新层级。

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

## 🔴 实验防泄露最高优先级规则（2026-08-23 起，任何实验先审后跑）

> 这部分优先级高于“先跑出结果”。**任何实验如果没有先完成信息边界设计与代码级泄露审计，禁止训练、禁止回测、禁止汇报成绩。** 一旦事后发现泄露，相关全部结果立即作废并显式标记 invalid，不得继续引用。

8a. **先写 forecast-origin，再写模型。** 每个实验在代码/manifest 中必须显式写出：预测目标日 `D`、预测时点、该时点可见信息、不可见信息、标签最晚可得日、selection/validation/final-holdout。缺任何一项都不得开跑。

8b. **24点 spread 任务的固定业务协议**：目标为 `D` 日24点 `spread = RT - DA`，forecast origin 固定为 **D-1 14:00**。在这一时点：
- `D` 日 RT / spread / target-day `actual_*` 全部不可见，只能作为 label；
- `D-1` 当日实时/价差只允许使用 **p1-p14 / h1-h14 已发生部分**；D-1 14:00 后的任何 realized RT/spread/actual 禁止进入特征；
- `D` 日 forecast-type fundamentals 若业务上在 origin 已发布，可作为特征；是否允许 target-day DA 必须按该实验自己的正式 contract 明确声明，当前 strict direct-spread 实验默认 **禁止 target-day DA 作为特征**；
- 完整历史 realized spread / RT / actual 只能来自 **D-2 及更早**。

8c. **训练标签边界铁律：预测 D 时，完整监督训练标签最晚只能到 D-2。** 在 D-1 14:00，D-1 15:00~24:00 的完整标签尚未产生，因此任何 `train_days = ... : idx`、`day < target_day`、`shift(1 day)` 等写法都必须人工核验是否误收 D-1 全日标签。24点 strict-spread 实验必须统一通过一个 `strict_train_days()`/等价 helper 获取训练日，且 `training_last_day <= D-2` 必须写入 ledger 并断言。

8d. **D-1 可见部分只能作为部分特征，绝不能偷换成“D-1完整训练日”。** 可以把 D-1 p1-p14 的原始轨迹、统计量、状态特征作为输入，但不能因为这些点可见就把 D-1 整天24个 label 放入 fit/threshold/calibration/feature-selection/similar-day label pool。

8e. **所有二级学习也受同一边界约束。** 以下环节不得读取目标日或当时不可见的真值：threshold calibration、stacking、gating/router、ensemble weight、feature selection、similar-day label统计、regime prior、online adaptation、model selection、early stopping、超参选择。它们必须是 rolling/prequential，只能使用在该预测时点已经完整落地的历史 OOS 结果。

8f. **Similar-Day / KNN / analog 特别规则**：目标 D 的候选历史日必须 `<= D-2`；距离计算只能使用预测时点已知的 descriptor；候选日的 spread label 可以使用，因为它们已是 D-2 及更早。必须输出 causal audit，记录每个 D 的 `latest_candidate_day` 并断言 `<= D-2`。

8g. **最终留出集一次性打开。** selection / validation 用于idea筛选后，fresh final holdout 在模型/规则完全冻结前禁止读取其 label、禁止据其调 threshold/weight/feature/model。最终测试失败后若继续迭代，该块即失去“final”资格，必须换更新的全新时间块。

8h. **跨月份是强制验收，不接受单一15天漂亮数字。** 任何声称“提升”“接近70%”“达到70%”的 direct-spread 方案，至少同时报告多个月份 direction accuracy、positive recall、negative recall、balanced accuracy、all-negative baseline，以及每月/总体相对基线增益。若仅靠类别比例、单月、单时段得到高 raw accuracy，而 balanced 或跨月明显崩溃，不得称为有效突破。

8i. **实验启动双门禁**：
1. 先跑 `scripts/tests/check_preflight_health.py`，必须全绿；
2. 再做实验专属 leakage audit。至少检查 feature source/cutoff、`training_last_day`、final holdout 未触碰、source manifest。对于关键新特征优先增加 counterfactual audit：篡改禁止信息后，特征/预测应保持不变；篡改允许信息后应能产生变化。

8j. **结果汇报必须带 leakage status。** 每个关键结果必须明确标记 `STRICT/PASS`、`ORACLE/PRIVILEGED`、`INVALID-LEAKAGE` 或 `LEGACY-UNVERIFIED`。没有通过 strict audit 的数字不得与正式成绩并列，更不得作为“当前最优”。

8k. **2026-08-23 事故永久记录**：P6+Similar-Day 曾出现约69.44%的漂亮验证结果，后查出滚动训练误用了 D-1 完整24点标签；该结果已作废。根因是 `train_days` 截止写成 target-day 前一天而未考虑 forecast origin 位于 D-1 14:00。以后任何时间序列实验，先问“这个 label 在预测那一刻真的已经完整发生了吗？”，再看日期大小；**日历上的过去不等于业务时点上的可见。**

## 文档与产物约束

9. **文档唯一入口**：当前生效文档以 `docs/README.md` 和 `docs/DOCUMENT_ARCHITECTURE.md` 为准。代码行为变化必须同步核对根目录 `README.md` 和对应生效文档；若两者冲突，以代码和测试为准，并在同一任务中修正文档。普通内容必须更新已有负责文档，不能随意新建。
10. **研究材料归档**：调研报告、Agent过程记录、旧方案、历史验收和调参草稿统一放在 `docs/archive/`，不得继续堆在 `docs/` 当前生效区。归档不删除，必须保留可追溯路径。
11. **新增文档必须先请求批准**：智能体发现确实需要新增文档时，只能在回复中提出申请，说明名称、所属领域、不能合并的原因、维护内容和验证方式；在用户明确同意前，不得创建该文件。获批后，新文档首部注明 `status: active|historical|archived`、日期、责任领域和验证依据，并登记到 `docs/README.md`。
12. **运行产物隔离**：预测账本、runs、实验输出、日志和模型权重不得作为源码修改提交；需要共享时提交 manifest、摘要和哈希，不提交大体量生成物。

12a. **项目根目录禁止 runtime/测试污染（2026-09-19）**：项目根是源码与配置入口，不是临时运行目录。Agent、pytest、smoke、probe 和模型调试不得在根目录新建 `.g1_*`、`.g1_pytest_temp*`、`.tmp_pytest/`、`output_*`、`*_diag.log`、临时 parquet/csv/json 或模型 scratch。pytest 默认使用操作系统临时目录/`tmp_path`，**禁止**使用 `--basetemp=.g1_*`、`--basetemp=.tmp_pytest` 或任何 repo-root basetemp；Windows 临时目录权限异常时，只允许改用仓库外系统 temp 路径，不得通过全仓 ACL 修改绕过。需要持久化的 Agent/测试诊断统一写 `outputs/diagnostics/agent/<date>/<task>/`，一次性 probe 写 `outputs/diagnostics/probes/<name>/<timestamp>/`。`.gitignore` 只是兜底，不能把错误 writer 因为被 ignore 就视为合规。

12aa. **测试 harness 路径规则**：测试应优先使用 `tempfile.TemporaryDirectory()` / pytest `tmp_path`；测试构造的 ledger/runs/cache/runtime 必须位于该临时根或 `outputs/experiments/04_pipeline_audits/<named-test>/` 的显式隔离目录，禁止把 G.1/G.2/lifecycle/interrupt 等测试 scratch 写进正式 `outputs/96/{ledger,runs,cache,runtime}`。需要检查正式 production state 时只读，除非测试明确属于用户批准的 production acceptance。任何需要重定向 stdout/stderr 的本地验收，默认不落盘；必须留证据时写 `outputs/diagnostics/agent/...`。crawler source-mode runtime 固定为 `outputs/crawl/runtime_96/`；frozen EXE runtime 固定为 `<exe目录>/output_96/`，项目根 `output_96/` 禁止重新生成。

12b. **链路输出隔离**：原链路保留 24 点 `outputs/ledger/` + `outputs/runs/`、96 点
     `outputs/ledger_96/` + `outputs/runs_96/` 作为 legacy 兼容区。当前生产默认 profile 为
     `production`，统一写入 `outputs/24/{ledger,runs,cache,runtime,sync}` 或
     `outputs/96/{ledger,runs,cache,runtime,sync}`；`feature_store` 仅保留历史候选/影子链兼容，
     不再作为新生产默认。实验必须继续放 `outputs/experiments/`，不得污染生产 ledger。

12c. **数据域隔离**：24 点 canonical 输入位于 `data/24/canonical/`；96 点数据库忠实镜像/权威同步
     位于 `data/96/remote/` 与 `data/96/authoritative/pmos_96_全量.csv`。96 生产模型层长期只维护
     `data/96/model_input/shandong_pmos_96_model_input_full.parquet` 一份；闭合历史由逻辑筛选获得，
     as-of masked 输入仅作单次任务临时文件并在结束后删除。新增或迁移数据必须先运行
     `scripts/tests/check_96_vs_24_actual.py`。

12e. **formal 96 ledger 是可迁移生产状态（2026-09-19；Dynamic-v1 生产验收 2026-09-20）**：`outputs/96/ledger/` 是 `ledger_weight` 的长期历史状态。当前 formal96 的 serving 可见性唯一由 **DB sync → immutable D/T snapshot → FeatureViewBuilder** 决定；融合学习器仍只允许使用到 **T-2 的完整真值**。完整 `--96 T` 在 cold-start/readiness 前会先从 authoritative actual source 幂等结算 T-2，随后 selector 从 T-2 向前取最近30个完整日。`ledger_predict` 继续 append 当日 DA3/RT4 prediction，并在历史/结算场景有真值时更新 actual。新服务器可迁移已有 ledger 避免重新暖机，但禁止裸复制 legacy/FeatureStore：warm-start 必须经 `bootstrap_96_production_ledger.py` 审计、staging/readiness、原子 promote；迁移与 learner 同义，从 T-2 向前最多90日选择最近30个 DA3/RT4+actual96 完整日，T-1 prediction 仅在整池96槽且 cutoff 合法时可选携带，保留源 `data_cutoff/source_file/run_id/model_version` 和 `bootstrap_manifest.json`。T-1 prediction 可以提前存在，但即使 T-1 actual 物理上存在，formal selector 也必须主动跳过，避免历史回放把未来真值带入权重。formal `--96` façade 必须 fail-closed 拒绝已知 legacy roots（`outputs/ledger_96`、`outputs/runs_96`、旧 classifier/feature_store cache）；自定义甲方部署根允许，但不得落入这些兼容目录。正式生产放行必须至少有一次真实标准入口 `python main.py --96 T` 达到 `NORMAL/exit0/postflight PASS/fallback=false`，并机械确认 DA3/RT4 各96点、同 snapshot/protocol、SGDFNet anchor 与 artifact audit PASS；live target actual 可 partial，历史结算需显式 `--require-target-actual`。甲方发布必须走白名单 `build_predictor_release.py` + 独立 ledger bootstrap + `doctor_96_deployment.py --strict-release`；strict doctor 必须证明 TimesFM 从部署根自身 `models/timesFM` 解析，禁止 generic `PROJECT_ROOT` 把模型重定向回开发机。

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
- 交付口径：24 legacy/advanced compatibility 保留五阶段；formal 96 production 为
  `ledger_predict → ledger_weight → ledger_fuse → final_outputs`，classifier 仅 legacy/shadow/replay。
- 回归：`scripts/check_delivery_stability.py` 等 4 件套
