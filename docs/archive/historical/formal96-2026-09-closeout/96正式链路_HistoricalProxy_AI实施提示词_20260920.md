请用 DevSpace 打开当前 electricity_forecast_model2.5 项目，严格按：

docs/96正式链路_HistoricalProxy与全量Ledger补全_设计执行任务书_20260920.md

执行。

本轮只做两件事：

1. 扩展现有 bootstrap，使原服务器 2025-12-18～2026-08-14 的完整历史可用 full-source 模式安全并入 production ledger；当前 production wins，先只做 dry-run / temp-root 验证，不要实际覆盖 production。
2. 实现三态 Snapshot 路由：
   - 历史日且已有合法 canonical LIVE Snapshot → 直接复用该 Snapshot；
   - 历史日但没有 Snapshot → HISTORICAL_PROXY_V1（14:00 / p56，只在 Snapshot 层代理）；
   - 正式当前预测 → 原 LIVE Dynamic Snapshot。
   三态之后全部复用同一个 FeatureViewBuilder、七模型、learner、fusion。

额外要求：
- 每个正式 LIVE 成功日永久保留至少一份 canonical Snapshot；成功 run manifest 必须绑定 snapshot_id/path；
- 多 attempt 时不能按“最新文件夹”猜，只能按成功 provenance 取；
- TimesFM/TimeMixer/SGDFNet/RT916 不重新启用 model-local 14/15 点 serving cutoff；
- 不改模型数学逻辑、模型池、30/90/lag2、SLSQP、CPU2/GPU1、SGDFNet anchor、RT916 stride24、24点链路；
- 不运行 2026-08-17 真模型，不补 8/17～9/19，不租服务器、不批量回测；
- 不 bulk restore/delete/stage。

先做事实复核，再按文档 TODO 实现和测试。完成后只返回：

PHASE=
STATUS=READY_FOR_REVIEW 或 FAIL
FILES_CHANGED=
BEHAVIOR_CHANGED=
TESTS=
LEDGER_FULL_SOURCE_DRY_RUN=
THREE_WAY_ROUTE=
SNAPSHOT_RETENTION=
LIVE_REGRESSION=
24_REGRESSION=
OPEN_ISSUES=

不要自行宣布生产完成。