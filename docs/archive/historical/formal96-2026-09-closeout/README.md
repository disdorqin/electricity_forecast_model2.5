# formal96 2026-09 收口归档

> status: archived
> archive_date: 2026-09-20
> 目的：保存 formal96 从 fixed-cutoff 改造、Dynamic-v1、长期维护、clean deployment、Historical Proxy 到最终三态 Snapshot 收口的过程证据。

本目录内文档全部为历史过程材料，不再作为当前生产规则。当前行为以根 README 与 docs 下 active authority 文档为准。

归档文件：

- `24点硬编码与模型入口审计_20260816.md`：早期 24/96 参数化与硬编码审计。
- `96正式链路_改造计划.md`：96 正式链路改造任务书，已被当前实现取代。
- `96正式链路_收尾修补计划_20260919.md`：G.1/G.2 收尾过程。
- `96正式链路_问题账本.md`：96 改造过程问题/证据总账。
- `96正式链路_长期生产维护_PhaseM1_路径收敛与零新增污染_20260919.md`：路径与 runtime 治理过程。
- `96正式链路_长期生产维护_PhaseM2_资产归档清理与甲方部署_20260919.md`：clean deployment 与资产治理过程。
- `96正式链路_HistoricalProxy_AI实施提示词_20260920.md`：Historical Proxy 实施提示词。
- `96正式链路_HistoricalProxy与全量Ledger补全_设计执行任务书_20260920.md`：三态 Snapshot 与 full-source ledger 实施设计。

这些文档中的最终有效结论已经提炼进入：

- 根 `README.md`：当前用户可见状态；
- `docs/PROJECT_LAYOUT.md`：三态 Snapshot / FeatureView / runtime 架构；
- `docs/RUNBOOK.md`：正式命令、验收、迁移、回放；
- `docs/DATA_CONTRACT_96.md`：三态 Snapshot 数据语义；
- `docs/LEAKAGE_AUDIT_96.md`：历史回放与 proxy 防泄漏；
- `docs/OUTPUT_CONVENTION.md`：Snapshot、ledger、manifest 持久化规则；
- `docs/PROJECT_GOVERNANCE.md`：canonical LIVE Snapshot 长期保留规则；
- `docs/SERVER_96_DEPLOYMENT_BACKFILL.md`：服务器部署、8/17 起历史接续与每日生产。
