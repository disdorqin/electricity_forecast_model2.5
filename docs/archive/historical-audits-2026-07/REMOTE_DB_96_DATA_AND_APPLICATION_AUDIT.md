# 远程数据库 96 点数据 · 应用数据血缘 · 降级交付审计报告

- **日期**：2026-07-26 ｜ **性质**：只读审计 + 设计（未改任何生产代码/数据库/爬虫/输出）
- **判据原则**：远程库实况为准；「本地快照仅 2 天」不构成远程只有 2 天的证据（业主明示，本报告已按此重构假设）。
- **后续更新（2026-08-01）**：文中作为执行入口的 `sync_data_96.py` 已删除，改由 `main.py --pipeline sync_dataset --resolution 15min`（`sync_data_96_core.py`）同步至 `data/remote_96/parquet/`。
- **标记**：`LIVE VERIFIED`（本轮或 07-26 对远程库成功查询）/ `LOCAL DATA VERIFIED` / `CODE VERIFIED`（文件:行号）/ `OWNER APPROVED` / `UNRESOLVED` / `BLOCKED`。

---

## 1. 连接执行情况（§2）

- 本分析环境（云沙箱）：DB 主机 TCP 直连**再次复测超时**；设备侧隔离 VM 无法启动且按设计无网络 → **本环境无法充当“办公机”**。`BLOCKED`
- 既有成功连接事实（非本轮）：2026-07-26 曾以项目 `.env` + pymysql 直连成功（MySQL 5.7.40，只读 SELECT 探针）`LIVE VERIFIED(07-26)`；2026-07-16 官方 `sync_dataset --sync-source db` 在办公机成功拉取 39,816 行 `LOCAL DATA VERIFIED(manifest)`。
- **本轮交付一键执行包**（办公机项目根目录运行一条命令即可完成任务书 §2-§12 的全部实测）：
  ```powershell
  powershell -ExecutionPolicy Bypass -File output\run_local_audit.ps1
  ```
  该 runner 依次执行：`output/_db_audit_96_live.py`（v4 整合版只读深审，见 §2 本报告）→ 官方同步四模式 → `sync_data_96.py`；stdout/stderr/exit code 全部落盘 `output\audit_logs\`，同步矩阵写 `output\db_audit_96_sync_matrix.json`。凭据不出现于任何输出（脚本对 `SHOW GRANTS` 做了脱敏截断）。
- 连接元信息记录点（runner 自动采集）：服务器版本、可见 schema、账号只读相关权限、SSL 状态、审计起止时间、查询失败/权限受限清单。

## 2. 全库扫描与两张核心表深审（§3–§7）——工具已备，数字待回填

`output/_db_audit_96_live.py`（v4，整合取代 v1/v2/v3 查询面）逐项覆盖任务书要求：全 schema/表/视图/列/注释/索引清单（中英关键词全集）；`epf_market_data_96` 深审（schema 全列、MIN/MAX 日期与时间、总行数、完整/不完整 96 点日、缺失日清单、最长连续段、行数分布、**逐列空值率(总体+按年)与最早/最晚非空日期**、fcast 未来日填充、actual 回补、p1=D 00:15 / p96=D+1 00:00 语义违例计数、重复键）；`epf_unit_data_96` 深审（**无过滤 GROUP BY unit_id** 判定单机组是库内事实还是查询假象、逐机组起止/完整日/缺日、8 列空值率、各价格列最新完整日、create/update 时延分位数 p50/p90/p95/max、近 30 日「p56 前是否已入库当日 RT」逐日计数）；价格候选全集（不限两张表，逐列注释/实体级别/实体数/跨度/完整 96 日/与 market_96 连接键）；特征-目标历史对齐矩阵（含「价格全特征缺」「特征全价格缺」双向清单与两档特征契约）；跨分辨率实验（特征 5 法 × 最近 15 个双侧完整日；价格 6 法含电量加权 × 逐实体 × DA/RT，输出 MAE/RMSE/最大差/相关/容差精确率/偏差/对比与剔除小时数）。产出 12 个机读工件（文件名见任务书 §14，均已实现）。

**「market_96 是否自 2022-01-01 连续至最新爬取日」**：脚本直接输出布尔答案字段 `answer_full_history_since_2022` + 缺失日清单 → `PENDING（等待办公机执行）`；本轮血缘新证据**支持业主记忆**（见 §3-L3）。

## 3. 应用数据血缘审计（§11）——本轮已完成代码侧全链

```text
PMOS 端点 → 爬虫请求 → 响应字段 → 解析 → DB 表列 → upsert 键 → 历史 → 同步命令 → 本地文件
→ 数据集 → 模型输入 → 预测工件 → 账本 → 融合 → 分类器 → 官方输出 →（未来）前端/库交付
```

| # | 血缘结论 | 证据 |
|---|---|---|
| L1 | 96 点价格链：`DaJyjgfbPlantQuery24.do`（日前）/`YxJyjgfbPlantQuery24.do`（实时）按 `unitid` 请求 → `cqPrice/power/energy/kt` → `epf_unit_data_96.da_/rt_*`，upsert 键 (market_date, period_no, unit_id)，`data_time=market_date+15p 分钟`（区间末） | `CODE VERIFIED`（crawl.py:229-237、run_crawler.py:258-306、backfill:153-208） |
| L2 | 96 点市场特征链：`DaJyxxPlDa.do`（appkey=112）→ 8 个字段 `systemload/dfdcload/excload/fdload/gfload/sytsjz/selfunit/syjzzj` → **仅映射为 8 个 `actual_*` 列**（direct_load/local_plant/tie_line/wind/solar/nuclear/self_owned/test_unit） | `CODE VERIFIED`（三份写入脚本 MARKET_FIELD_MAP 完全一致：run_crawler.py:83-92、auto_fill_96.py:86-95、backfill:376-381） |
| L3 | **历史回填同时写市场表**：`backfill_unit_data_96.crawl_single_day` 每日在写 unit 数据的同时调用 `_upsert_market_overview` 写 market_96（:341-360）→ 若 2022→2026 回填曾整段执行，market_96 的 8 个 actual 列应具备与 unit 表同量级的历史——**支持业主“market_96 有全历史”的记忆**，待 live 审计定量 | `CODE VERIFIED` + `PENDING` |
| L4 | **重大发现：存在仓库外的第二写入方**。`epf_market_data_96` 的 26 个业务列中，本仓三份写入脚本只写 8 个 actual 列；但本地快照中 `fcast_*` 13 列及 `actual_bidding_space/new_energy/pos_reserve/neg_reserve/unit_maintenance` 均有值 → 这些列由**本仓之外的系统**写入（旁证：本仓迁移只有 unit 表 DDL，`init_database_tables` 只建 unit 表——market_96 表本身即为外部创建）。该外部写入方的排程/历史深度/是否仍在运行，决定 fcast 特征的预测时可得性与训练窗口，**必须经 live 审计的逐列「最早非空日期 + create/update 分布」来定** | `CODE VERIFIED`（写入面枚举）+ `LOCAL DATA VERIFIED`（快照含外部列值）+ `UNRESOLVED`（写入方身份，疑为兄弟项目 ../epf） |
| L5 | 小时表 `epf_market_data` 在本仓**只有读取**（sync/freshness/概率探针），写入方为外部历史项目 | `CODE VERIFIED`（全仓 grep 无该表 INSERT） |
| L6 | 数据止于 07-18 的原因排查顺序：① Cookie 过期（README §17.1 列为最常见故障，错误特征 HTTP 911/CSRF）→ ② 定时任务停摆（办公机休眠，§17.2）→ ③ 远端改版。`auto_fill_96 --dry-run` 一条命令即可判别（列缺失日+试连 PMOS）；本轮不修 | `CODE VERIFIED` + `UNRESOLVED` |
| L7 | 单机组：config.json 单 `unit_id` 驱动全部机组爬取 → 若库内确为 1 机组（live 无过滤复核），则属**爬虫配置限制**而非库结构限制 | `CODE VERIFIED` + `PENDING` |
| L8 | **本地快照仅 2 天的解释（按可能性排序）**：(a) 曾执行**限定日期/仅 market**的 `sync_data_96.py --type market [--start-date …]`——`sync_market_96` 单独运行**不写 manifest**，与「`outputs/data_sync_96/` 目录整体不存在」完全吻合；(b) `sync_all_96` 从未成功走完（否则必有 manifest）；(c) 远程 market_96 当时确实只有 2 天——与 L3/L4 证据张力最大，列为最弱假设。**远程有全历史而本地从未物化，与 (a)/(b) 完全自洽** | `CODE VERIFIED`（manifest 逻辑 sync_data_96.py:295-321）+ `LOCAL DATA VERIFIED`（目录缺失）+ `PENDING`（live 定案） |
| L9 | 下游链（数据集→模型→账本→融合→分类器→输出）已在前两轮完整核定：宽表列映射、7 腿输入、账本键、BGEW、−80 规则（置为 −80.0）、submission 构建源 | `CODE VERIFIED`（见 FINAL_REVIEW/ASSESSMENT 两文档） |

## 4. 同步模式矩阵（§12）——runner 自动生成

四模式 + `sync_data_96.py` 全部纳入 `run_local_audit.ps1`；逐命令记录 exit code/起止/日志路径，JSON 汇总。语义预判（`CODE VERIFIED`）：db/http=真远端刷新；local=本地文件提升非刷新；auto=db→http→local 回退。历史先例：db 模式 07-16 成功（39,816 行）。云端不可执行不作为生产路径不可用的证据。运行日盘中 RT 可得性由日志中「最新非空 RT 时间戳」+ live 审计 `unit96_dday_rows_updated_before_14` 联合判定。

## 5. 降级交付结构化设计（§13）——设计完成，未实施

**现有机制盘点**（`CODE VERIFIED`）：`run_manifest.json`（五阶段状态/errors/warnings/postflight/delivery_status/fallback 块）、`delivery_report.json/md`、`range_manifest`、exit code 0/2/1、`fallback_report.*`、`--strict-classifier` 开关、分类器现为非阻断旁路（`ledger_classifier.py:107-113`）、`run_id` 存在于账本行（`{model}_v2_{date}`）但**无全局 trace id**、无数据库运行状态表、无前端状态字段——前端交付层尚未存在。

**96 点分类器失败的批准行为**（`OWNER APPROVED`）：官方 RT 回退 `realtime_before_classifier`；保留有效融合 DA 与未修正 RT；`delivery_status=DEGRADED_DELIVERED`、exit code=2（复用现约定）；不向前端暴露堆栈；失败产物不进训练账本/缓存（现 fallback 已有同类保证，分类器输出本就不进账本 `CODE VERIFIED`）；不伪称修正成功。

**结构化字段设计**（落点：`run_manifest.json` 顶层新增 `delivery_record` 块，同构导出为未来入库行）：

```json
{
  "run_id": "ledger_full_15min_2026-xx-xx_<seed>",  // 建议升级为全局 run_id，并作 trace_id
  "target_day": "...", "resolution": "15min",
  "delivery_status": "DEGRADED_DELIVERED", "delivery_exit_code": 2,
  "classifier_status": "FAILED",
  "classifier_applied_to_official_output": false,
  "official_rt_source": "realtime_before_classifier",
  "degradation_reason_code": "CLASSIFIER_FAILURE",
  "degradation_summary": "96点负电价分类器失败，官方实时结果采用未修正融合值",
  "error_type": "<异常类名>", "error_stage": "ledger_classifier",
  "error_log_path": "outputs/runs_96/<d>/logs/classifier_error.log",
  "manifest_path": "...", "report_path": "...",
  "started_at": "...", "completed_at": "..."
}
```

**三层日志**：① 前端安全层——仅 `delivery_status + degradation_reason_code + degradation_summary`（非技术文案，如“预测已完成，实时修正降级”），无堆栈；② 运维层——manifest/delivery_report 现有结构 + 上述块（阶段状态、回退源、错误类别、修正是否生效、工件路径）；③ 开发层——完整堆栈+分类器配置+输入校验摘要+run_id/trace_id，仅写后端日志文件（新增 `logs/` 于当日 run 目录）。**存放建议**：短期 manifest-only（零 schema 变更，与现架构一致）；接入前端时增设**独立 run-status 表**（一行一 run，字段=上述块），预测结果表不混入状态字段——即“结果与状态分离、双写 manifest+状态表”。分类器失败时 `postflight` 仍按 96 行契约校验未修正官方文件（可通过），状态机不误升 FAILED。`ledger_full` 现有 `_finalize_delivery` 挂钩点即可承载该分支（实施属 R4/R5，不在本轮）。

## 6. GO/STOP 判定（§16）

**判定：`STOP`**（数据闸门未过；不得开始 R1）。

| 未过闸门 | 证据 | 归因 | 下一步 | 需业主？ |
|---|---|---|---|---|
| live 全库扫描未从有 DB 权限环境执行 | 本环境双通道复测 BLOCKED | 环境 | 办公机跑 runner（§1） | 执行即可 |
| 官方 DA/RT 目标未定 | 候选唯一但语义未确认；聚合实验未跑 | 业务定义+数据 | runner 出实验数据 → 业主评审 | ✅ |
| market_96 历史/外部写入方未知 | L3/L4 | 数据库+爬虫 | live 逐列最早非空+时延分布 | 评审结果 |
| 爬虫新鲜度断供（止 07-18） | LIVE VERIFIED(07-26) | 爬虫 | `auto_fill_96 --dry-run` 判因→修复（另立项） | ✅（修复批准） |
| 本地无法复现 96 点数据 | unit 本地缺失、market 仅 2 天 | 同步 | runner 内 `sync_data_96.py` | 执行即可 |
| 预测时可得性未知 | 时延审计未跑 | 数据 | live 审计 §5 节 | 评审结果 |

**解除路径（预计一次办公机会话完成）**：跑 runner（约 10–30 分钟）→ 回传 `output/db_audit_96_*.{csv,json}` + `audit_logs/*.log` → 回填 FINAL_REVIEW 闸门表 → 业主对目标列/实体级别/事件时长流程签字 → 复评 GO。

---

*本轮产物：本报告、`output/_db_audit_96_live.py`（v4 整合审计）、`output/run_local_audit.ps1`（一键 runner）；FINAL_REVIEW 已同步加注。未修改任何生产代码与数据库对象。*
