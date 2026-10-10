# AUX-V1 版本记录

## 2026-09-22 — `2026-09-22-disclosure-aux-v1`

- 新增独立辅助信息披露入口，不修改主 `crawl_96_local.py`、`crawl.py` 和 `epf_pmos_96_full`。
- 共享 `AuthConfig`、`AuthenticationStateMachine`、`RuntimeLock` 和 `PmosCrawler` transport。
- 实现 raw-first source registry、unit/constraint/event/curve/stat/contract parser 和 AUX DB writer。
- 7 张 `epf_pmos_aux_*` 表使用应用层 SHA256 `record_key` 幂等 upsert。
- `getFhChar`/`getZdLlx` 保持 raw-only；`jzdlzb` 保持 `ratio_status=UNDEFINED`。
- 本地 EXE：`crawl_disclosure_aux_v1.exe`，39,531,280 bytes，SHA256 `3D4385595AD508ECFFAECA6380641826C6CE21EDB98AD82173263CD33D69D565`。
- 本地验证：AUX 合约测试 10/10、主爬虫回归（含 AUX）80/80、`--help` PASS、`--db-check --no-db-upload` PASS、`py_compile` PASS、`git diff --check` PASS。
- 公司真机状态：`NOT_TESTED`。

## 2026-09-22 独立上线前验收

架构隔离、本地测试、EXE/DDL 对应均通过；但在公司正常写库前需要 AUX-V1-r1 小修：补 `output_aux/aux_crawler.log`、校正 `qctc_pm_trade_inside` URL contract、让 `SourceSpec.page_url` 真正参与 browser fallback、兼容 nested parallel-array curve。真实 source 参数/结构仍以公司首轮 `--capture-only` raw 为准。详见 `AUX-V1_上线前独立验收与r1修补设计.md`。

## 2026-09-22 — `2026-09-22-disclosure-aux-v1-r1`

- `SourceSpec` 增加 frontend path、resolved gateway URL、HAR evidence、参数合同和分页边界。
- 已按 HAR 恢复已确认 GET/query 接口；禁止有/无 `/qctc` 双试和 GET/POST 猜测。
- 新增 AUX-only request helper，真实使用 `page_url` 进行 browser fallback；未知/未验证来源保持 raw-only 或 disabled。
- `runLine` 多 series × point、`degLine` nested wrapper、Tsjz/Sbdjx/Dwby wrapper 解析已收紧。
- 日志固定 `output_aux/aux_crawler.log`，报告固定 `output_aux/aux_report.json`；旧 `report.json` 不删除且停止写入。
- BUILD_VERSION：`2026-09-22-disclosure-aux-v1-r1`。
- r1 EXE：`crawl_disclosure_aux_v1.exe`，39,539,362 bytes，SHA256 `9AF3B5F459A1BCE6F8EFBC1D2656818381DC5FEFF4E50CBB5AFA5582DBF51DE1`；`--help`、`--db-check --no-db-upload` PASS。
- HAR 已确认 15 个 GET 合同（含 3 unit、3 constraint、3 daRqxxpl event、2 curve、3 raw-only curve、1 contract；其中 raw-only 不结构化）。
- 本地 r1：18/18 AUX tests、主爬虫回归 88/88、`py_compile`/`git diff --check` PASS；公司真机状态：`NOT_TESTED`；首次公司运行必须 capture-only。

## 2026-09-22 — `2026-09-22-disclosure-aux-v1-r2`

- `[AUX-V1-r2]` `unit_type`/`unit_gengroup` 改为 raw-only 且 `target_table=None`，HTTP 200 字典也不写 `epf_pmos_aux_unit_master`。
- `[AUX-V1-r2]` `unit_constraint`、`unit_component`、`unit_constraint_jjcq` 默认 disabled；缺少真实 `unitid` 时 HTTP 前记录 `AUX_DEPENDENCY_MISSING unitid`，不发送空参数，不做自动 fan-out。
- `[AUX-V1-r2]` `all`/group 只选 enabled source；显式 source name 才能调试 disabled source。
- `[AUX-V1-r2]` `unit_master` 按前端 `start=0,length=100` 和 `recordsTotal` 做 bounded offset 分页；max_pages=200、max_rows=100000、重复页停止，未抓齐为 `PARTIAL/pagination_pending`。
- 本地 r2：AUX 测试 23/23 PASS，主爬虫回归（含 AUX）93/93 PASS，`py_compile`/`git diff --check` PASS。
- r2 EXE：`crawl_disclosure_aux_v1.exe`，38,454,017 bytes，SHA256 `7740E4D0484CAF9EA0705FF903448D89EDF6C50AA192CC6CDAE2ADD53D92AB4A`；`--help`、`--db-check --no-db-upload` PASS。
- 公司真机状态：`NOT_TESTED`；首次公司运行仍只允许 `--db-check --no-db-upload`、`--auth-only`、`--date YYYY-MM-DD --capture-only`。

## 2026-09-23 — AUX-V1-r2 配置缺失热修

- 根因：EXE 启动时假定 `config_disclosure_aux.json` 已随文件夹复制；缺失时在 report/log 初始化前抛裸 `FileNotFoundError`。
- `[AUX-V1-r2]` 缺失时自动在 EXE 同目录生成无敏感信息的配置，优先引用同目录认证/DB配置、再尝试上级目录；`db_upload=false` fail-safe，避免只拷 EXE 后意外写库。
- BUILD_VERSION：`2026-09-23-disclosure-aux-v1-r2`；缺失配置回归测试覆盖生成路径与默认禁写库行为。
- 热修 EXE：`crawl_disclosure_aux_v1.exe`，38,455,601 bytes，SHA256 `97B1B13AE7A78A213FB88944DF32C13C56AAF1A9592D7DAB0A4C1D8D389C815F`；无配置临时目录 smoke、`--help`、`--db-check --no-db-upload` PASS。
- 本地 r2 热修回归：AUX 24/24、主爬虫回归（含 AUX）94/94、`py_compile`、`git diff --check` PASS；公司状态仍 `NOT_TESTED`。

## 2026-09-23 — AUX-V1-r2a 响应诊断热修

- 审核公司 `aux_report.json` 与 `aux_crawler.log`：认证/QCTC context PASS；9/9 默认 source 均 FAILED_SOURCE、0 structured rows。Python 请求连接超时后浏览器 fallback 收到 503 Whitelabel / 504 nginx HTML。
- 根因现状属于请求链路/PMOS upstream 返回失败；HAR 对 inside path 是前端推导（不是 direct inside network capture），在拿到 raw response 前不猜改路径。没有成功抓取任何业务数据，capture-only 未写 DB。
- `[AUX-V1-r2]` AUX-only 修复：非 JSON HTTP 响应保留最多64KiB raw body、content-type/status 和短错误摘要；不再把 HTTP 503/504 报成 `JSONDecodeError`。未改主 `crawl.py`、认证/浏览器/state_machine 或主 EXE。
- BUILD_VERSION：`2026-09-23-disclosure-aux-v1-r2a`。AUX 测试 25/25，主爬虫回归（含 AUX）95/95，`py_compile` / `git diff --check` PASS。公司真实服务是否恢复仍 `NOT_TESTED`。
- r2a EXE：`crawl_disclosure_aux_v1.exe`，38,453,755 bytes，SHA256 `904D16695837E1AD163A6EADDF00FC88AE1F0B94E71C0F22FC30EC074A2993C7`；`--help`、`--db-check --no-db-upload` PASS。
- 当前公司报错发生在配置加载阶段，尚未启动认证/请求/DB写入；公司状态仍为 `NOT_TESTED`。

## 2026-09-23 — `2026-09-23-disclosure-aux-v1-r3` HAR 路由收敛

- 仅修改 AUX `disclosure_aux.py`、AUX tests/docs；未修改主 `crawl.py`、`crawl_96_local.py`、认证/browser/state_machine、`epf_pmos_96_full` 或主 EXE。
- 根据 `dist/crawler/output_96` 的实际 HAR 200 证据，新增 `unit_info` outside 路由并将其纳入默认 `all`；确认旧 ZCQ `unit_month_limit` 合约接口为 `POST /zcq/...` 且参数位于 query string，并保留 `jzdlzb` 的未定义语义。
- 无 direct network 200 证据的 inside source 从默认 `all` 移除；依赖 `unitid/faids` 的旧接口保持显式 disabled，不自动 fan-out 或猜测参数。
- r3 EXE：`crawl_disclosure_aux_v1.exe`，38,453,370 bytes，SHA256 `9D2707B514120D14C4D3F704F4518EB9278B1385B186FF5EA27E4DA2012F53E3`；EXE `--help`、`--db-check --no-db-upload` PASS。
- 本地 AUX tests、py_compile、`--help`、`--db-check --no-db-upload` 通过后再生成 EXE；公司真机状态仍为 `NOT_TESTED`。

## 2026-09-23 — `2026-09-23-disclosure-aux-v1-r4` 认证拒绝自动新浏览器恢复

- 根据 `output_aux/aux_report.json` 与日志确认 r3 已检测并复用既有 Chrome/QCTC 上下文，但 `unit_info` 的真实浏览器请求返回 HTTP 401 `authentication_rejected`；旧的 96 爬虫 recovery 逻辑也只会在 CDP 控制丢失时 force-new，AUX 过去没有 401 后的新浏览器恢复。
- `[AUX-V1-r4]` 仅 AUX 入口增加有界恢复：发生真实 401/403 后调用共享 `AuthenticationStateMachine`，禁用 CDP reuse、生成隔离 profile 并自动开新浏览器，然后重试当前 source collection 一次；再次 401/403 立即失败。无循环重试，无接口数据降级。
- 直接使用共享底座，没有修改 `crawl.py`、`crawl_96_local.py`、`run_crawler.py`、auth/browser/state_machine、96 DB 表/EXE。
- 最小回归覆盖 force-new 配置、恢复成功、重试仍失败时不循环。
- 公司端仍需使用新 EXE 做真实验证；本地测试不代表平台授权已恢复，状态记 `NOT_TESTED`。
- 本地回归：AUX 32/32；AUX + crawler runtime + shared auth + V10 P0 合计 102/102；`py_compile`、`git diff --check`、EXE `--help` 和隔离目录 `--db-check --no-db-upload` PASS。
- EXE：`crawl_disclosure_aux_v1.exe`，38,453,574 bytes，SHA256 `0D72452B227330FD08DFFDB64BF80BEF833B196F392A5AB390A872B2849EAC42`；`--help` 与隔离目录 `--db-check --no-db-upload` PASS；真实平台状态 `NOT_TESTED`。

## 2026-09-23 — `2026-09-23-disclosure-aux-v1-r5` 月度接口 CSRF 契约修补

- 使用新放入 `output_aux/pmos.sd.sgcc.com.cn17.har` 的实测链路：`GET /zcq/jysbys/ydfdcsxyhcx.do?appkey=81` 后，页面查询触发 `POST ...?method=getarcdetailNxdcFd&dmonth=...&smonth=...`，HTTP 200；请求带 `X-CSRF-TOKEN`、`X-Requested-With` 和该页面 Referer。
- `[AUX-V1-r5]` 只修改 AUX adapter：请求 `unit_month_limit` 前借现有浏览器读取 appkey=81 页面 CSRF meta，并把 token 仅暂存在 crawler 内存中；Python 和浏览器 POST 均带 CSRF header，浏览器 fallback 仍使用 HAR 确认的 POST/query 参数。缺 token 或页面不可访问时 fail-closed，不发裸 POST。
- 未修改共享 `crawl.py`、`crawl_96_local.py`、`run_crawler.py`、auth/browser/state_machine、96 表和主 EXE。
- 最小回归：AUX 34/34，AUX + crawler runtime + shared auth + V10 P0 合计 104/104；CSRF 页面获取与 Python/browser 两条传输路径带头验证、缺 token 不发请求均 PASS。`py_compile`、`git diff --check`、EXE `--help` 与隔离目录 `--db-check --no-db-upload` PASS。
- r5 EXE：`crawl_disclosure_aux_v1.exe`，38,455,120 bytes，SHA256 `4EAB5FD64076750FEA58708832A9C2A2041EBEBBDDA38D7049C3094495CA3091`。
- 公司真机仍待使用 r5 EXE 对 `unit_month_limit` 单日 capture-only 验证。

## 2026-09-23 — `2026-09-23-disclosure-aux-v1-r6` 统一 AUX 表

- 将 AUX DB 目标从旧七表收敛为唯一 `epf_pmos_aux_records`；raw 行写 `raw_json`，结构化 parser 字段原样写 `record_json`，公共来源/日期/状态/run provenance 独立成列。未确认字段（包括 `jzdlzb`）不改变语义。
- 修复 DDL 初始化器先按分号拆 SQL、导致注释里的分号切断 CREATE TABLE 并出现“函数返回成功但未建表”的问题；新增语句解析合同测试。
- 项目配置数据库原先没有 AUX 表；本次只创建 `epf_pmos_aux_records`，复核 23 列、0 行；未写入模拟数据，未修改主96表。
- 本地 AUX 合约测试 35/35 PASS；EXE `--help`、`--db-check --no-db-upload` 和 `--db-check` PASS；py_compile、git diff --check PASS。
- BUILD_VERSION：`2026-09-23-disclosure-aux-v1-r6`。
- EXE：`crawl_disclosure_aux_v1.exe`，38,456,555 bytes，SHA256 `EB5039538F385B3B45E72CAA0C8E2F487FE3929A5578D2042B379873039FD756`。
- 公司电脑使用 r6 EXE 的 live data ingest 尚未验证，状态 `NOT_TESTED`；数据库表已由当前项目 DB 配置创建，等待第一条真实 structured row 入库复核。

## 2026-09-23 — `2026-09-23-disclosure-aux-v1-r7` appkey=81 HTML/911 修补

- `[AUX-V1-r7]` 仅在 AUX adapter 修补 `unit_month_limit` 的 CSRF 页面获取：用 HTML `Accept`、抑制 `X-Requested-With`，优先走现有认证 session 的 GET；非200/异常时进行一次同标头 browser fetch。仍无 HTTP 200/CSRF 时 fail-closed，不发月度 POST。
- 月度数据 POST endpoint、query 参数、动态 CSRF header 与浏览器回退语义保持不变；未改共享 `crawl.py`、96 专用程序、auth/browser/state_machine 或主 EXE。
- 回归：AUX tests 36/36 PASS，覆盖 HTML header、HTTP 911 browser fallback、有效 CSRF POST、missing CSRF 不发请求。
- BUILD_VERSION：`2026-09-23-disclosure-aux-v1-r7`；公司 live 验证状态：`NOT_TESTED`。
- r7 EXE：`crawl_disclosure_aux_v1.exe`，38,456,617 bytes，SHA256 `FBEFA9F1E5C9D62E86FC214C08EAB5B011E94FC2BE5DC69AF47C680ED7769882`；EXE `--help` 与 `--db-check --no-db-upload` PASS。

## 2026-09-23 — `2026-09-23-disclosure-aux-v1-r8` appkey=81 文档导航修补

- `pmos.sd.sgcc.com.cn18.har` 中 appkey=81 GET 为 `200 text/html;charset=UTF-8`、无重定向、约 10KB，包含 CSRF meta；紧随的月度 POST 为 `200 application/json` 且含 1 条 data。未把 token 或响应业务值写入日志/文档。
- 失败 run 中 r7 从 `/home` 复用同源上下文执行 fetch，只取得 155 字节无 CSRF 页面。AUX-only 改为在已认证 PMOS 标签页真实导航 appkey=81，再由 CDP 读取页面 DOM 与导航状态；POST contract 不变，未拿到 HTTP 200/CSRF 仍不发 POST。
- 最小回归：AUX tests 36/36 PASS，覆盖 CDP 页面导航/DOM 提取、CSRF 缺失 fail-closed、POST 查询参数和浏览器回退；py_compile、git diff --check PASS。公司实网待用 r8 复验。
- BUILD_VERSION：`2026-09-23-disclosure-aux-v1-r8`。
- r8 EXE：`crawl_disclosure_aux_v1.exe`，38,458,055 bytes，SHA256 `C645CCF126740FFC7BBBCD3BD5D28B937E7CF068C2A095107127DDDDF4495ED8`；EXE `--help`、`--db-check --no-db-upload` PASS。

## 2026-09-23 — `2026-09-23-disclosure-aux-v1-r9` MySQL captured_at 时区兼容

- 真机 traceback：MySQL 1292 拒绝 `2026-09-23T23:20:54.034803+08:00` 写入 `DATETIME captured_at`；失败发生在 raw upsert 阶段。
- `[AUX-V1-r9]` 仅在 AUX DB writer 的 SQL 边界将 ISO datetime 解析为 Python naive `datetime`，保持输入本地墙钟时间并去掉 DATETIME 不支持的 offset；结构化 JSON 与本地 raw 保留来源字符串。表结构、主96数据库和采集请求不变。
- 最小回归覆盖带 `+08:00` 的完整 `upsert_source_result()` raw+structured 路径及 offset-bearing JSON 保留。
- 基线 r8 SHA256：`C645CCF126740FFC7BBBCD3BD5D28B937E7CF068C2A095107127DDDDF4495ED8`。
- 本地验收：AUX + crawler runtime + shared auth + V10 P0 106/106 PASS；`py_compile`、`git diff --check`、EXE `--help` PASS；隔离临时目录 `--db-check --no-db-upload` PASS（report `db_schema=PASS`，未连接/写入真实数据库）。
- r9 EXE：`crawl_disclosure_aux_v1.exe`，38,455,618 bytes，SHA256 `4CCBAAF5F5B647A2C83C4F57691D1BB9EA4107DA31E6D2FD151649862CCEADBB`。
- 公司真机：2026-09-24 使用 r9 完成单日 `unit_info` 写库 smoke，run_id `20260924T021452Z-81a83fe4`，report `PASS`，DB 2 行（raw=1、structured=1）。历史多年回补与 `unit_month_limit` 批量运行仍 `NOT_TESTED`。

## 2026-09-24 — `2026-09-24-disclosure-aux-v1-r10` 一键调度已登记来源

- 只新增 AUX 专属 `--source all-designed`；旧 `--source all`、group 与精确 source 选择规则保持不变，不修改主96爬虫、共享 auth/browser/state_machine 或 DB schema。
- `all-designed` 遍历 AUX source registry；daily/snapshot/15min 来源按日期调度，月度来源在同一个 lookback run 中每自然月只请求一次。
- 每个来源先检查 readiness。`UNVERIFIED`、缺少 unitid、或参数 builder 未覆盖参数契约时报告 `SKIPPED_NOT_READY`，不发请求、不伪造 raw 响应、不写入业务表；raw-only 来源只保存平台 raw，不声称有结构化字段。
- report 增加 all-designed 按业务日期的逐来源状态与当月 source 调度摘要。存在 failed/partial/skipped 时总状态 `PARTIAL`，exit=2；不能把部分可用误报全字段完成。
- 最小测试：AUX tests 39/39 PASS；AUX + crawler runtime + shared auth + V10 P0 109/109 PASS；`py_compile`、EXE `--help`、隔离候选目录 `--db-check --no-db-upload`、`git diff --check` PASS。
- r9 基线 SHA256：`4CCBAAF5F5B647A2C83C4F57691D1BB9EA4107DA31E6D2FD151649862CCEADBB`。
- r10 初版 EXE（61,487,282 bytes，SHA256 `F0A159B6DF79576B6708A6DA96807C6BA812348545BEA886DF96C9B5EFB58156`）因使用 `epf-2` 的 OpenSSL 3.6.x 打包，在公司 Windows 证书存储加载时触发 `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`，已标记作废，不要继续部署。
- `[AUX-V1-r10]` 仅重打包，不改业务源码：改用 `dist/build_artifacts/venv_build`（Python 3.11.9 / OpenSSL 3.0.13）构建，保留 PyMySQL TLS 校验，不通过关闭 SSL 绕过证书错误。冻结 SSL probe 的 Windows cert store 加载 PASS；本机 PyMySQL connect+close（无 SQL）PASS；`--help` 与隔离目录 `--db-check --no-db-upload` PASS。
- r10 修正版 EXE：`crawl_disclosure_aux_v1.exe`，38,461,453 bytes，SHA256 `AC133398AD70AFCD4566E46B9DD8E68CF43F12B51A50C812D7E93A5A88C08F74`。公司端修正版 live crawl 尚待复验，状态 `NOT_TESTED`。
- 公司 `all-designed` 实网验收：`NOT_TESTED`。首次建议使用当天 `--lookback 0`；检查报告中成功、空、失败和 `SKIPPED_NOT_READY` 来源后，再决定历史回补。

## 2026-09-24 — `2026-09-24-disclosure-aux-v1-r10-diag1` 阶段诊断日志

- 基线：r10 修正版 SHA256 `AC133398AD70AFCD4566E46B9DD8E68CF43F12B51A50C812D7E93A5A88C08F74`（38,461,453 bytes；OpenSSL 3.0.13 build）。
- 根据公司最新报告中 auth state 长时间 `unknown`、未进入 source 阶段的证据，仅增加 AUX 启动、锁、auth heartbeat、QCTC、DB schema/connect/upsert、日期/source、transport/parser/raw 阶段日志与阶段化 report 状态；不改变认证流程、接口请求、数据语义或 DB schema。
- HTTP error 仅记最多300字符脱敏摘要；成功响应不记正文；DB 凭据、Cookie、Token、CSRF、密码不进入日志。长阶段每20秒 heartbeat；report 保留阶段 RUNNING/PASS/FAIL。
- 测试：AUX + crawler runtime + shared auth + V10 P0 合计 112/112 PASS；`py_compile`、EXE `--help`、候选目录 `--db-check --no-db-upload` PASS。候选目录校验未连接/写入公司数据库。
- 新 EXE：`crawl_disclosure_aux_v1.exe`，38,465,116 bytes，SHA256 `3D41D76C83BFA11BA2653A92706F33D4EFE4B09CABB4B94AE87AE81C1BC55871`；使用 `venv_build` Python 3.11.9 / OpenSSL 3.0.13 构建。之前诊断候选 SHA 未发布，以本 SHA 为准。
- 公司新 build 诊断/真实采集状态：`NOT_TESTED`；旧日志中的 401/503 与 auth unknown 尚未因增加日志而改变，待公司使用此诊断包复测。

## 2026-09-24 — `2026-09-24-disclosure-aux-v1-r10-diag2` 对齐96点 QCTC transport

- 根因核对：96点生产 `_qctc_get()` 始终通过已认证浏览器同源 `fetch`；AUX 旧 `_aux_request()` 对现代 `/qctc/` 先走 Python `requests`。门户 Cookie 不携带 QCTC `sessionStorage` 上下文，因此 AUX 在 QCTC source 上收到 HTTP 401，即使 portal auth PASS。
- 最小修补：仅修改 AUX `disclosure_aux.py`，现代 `/qctc/` API 改为 inherited `_browser_req()` browser-primary；旧 `/zcq` legacy source 保留原 CSRF/Python-first 语义。未修改共享 `crawl.py`、`crawl_96_local.py`、auth/browser/state_machine、96 EXE 或 `epf_pmos_96_full`。
- 回归：AUX tests 43/43 PASS；AUX + shared crawler/V10 P0 tests 93/93 PASS；`py_compile` 与 `git diff --check` PASS。
- AUX EXE 已重打包：`crawl_disclosure_aux_v1.exe`，38,469,537 bytes，SHA256 `6DF9EBF475AC08B59B9004865306730C892EBF01DF406DF2968FA49B55715B3C`；`--help` PASS，隔离临时目录复制 DDL 后 `--db-check --no-db-upload` PASS。
- 公司 live 尚未用此 diag2 EXE 复验；部署后应优先单日 `unit_info` smoke，确认 report 的 transport=`browser_primary`，再检查 AUX 表。
- 公司数据库核验：`epf_pmos_96_full` 在 2026-09-24 有 96 行，`source_captured_at=2026-09-24 20:24:54`，`update_time=2026-09-24 20:25:03`；证明最新96点 EXE 已成功同步，但不等同 AUX 已成功。
- 当前 AUX r10-diag1 公司失败 run 的 `unit_info` 两次均为 HTTP 401；该证据与上述 transport 差异一致。

## 2026-09-24 — `2026-09-24-disclosure-aux-v1-r10-route1` HAR legacy 多页面路由

- 用户确认同一个 AUX 进程需要自主访问不同 PMOS 页面。HAR16/17/18 显示旧 ZCQ 来源并非同一个页面上下文：net contract 使用 appkey=18，generation contract 使用 appkey=93，generation hourly 使用 appkey=94，unit monthly limit 使用 appkey=81。
- `[AUX-V1-r10-route1]` 仅修改 AUX `disclosure_aux.py` 和 AUX 合约测试：为每个已确认 legacy source 保存实际 `page_url`；`net_contract_day` 改为 HAR 确认的 `POST /zcq/JyjgZcqXxpl.do?method=getTableDate` + query 参数 `startTime/endTime/type/sort`；浏览器 fallback/请求前在同一个认证 CDP target 中导航到对应 appkey 页面。appkey=81 仍由 CSRF 文档导航负责，避免二次导航丢 token。
- 未修改共享 `crawl.py`、`crawl_96_local.py`、主认证/browser/state_machine、96 表或主 EXE；未猜测没有 HAR HTTP 200 证据的 QCTC/维护接口。失败的 503/504 仍保留 raw 和真实失败状态，不跨来源补值。
- 本地 AUX 合约测试 44/44 PASS；`py_compile`、`git diff --check` PASS。公司真机必须使用新 EXE 先跑单日 `--source all-designed --lookback 0`，检查四个 appkey route 的实际 HTTP 状态后再做历史回补；公司状态 `NOT_TESTED`。
- r10-route1 EXE：`crawl_disclosure_aux_v1.exe`，38,467,433 bytes，SHA256 `8498A48B333AD046EC950F687C8B2F875B9358B19696D211E6C3DB48AA545DE0`；`--help` PASS。构建继续使用 `dist/build_artifacts/venv_build`（OpenSSL 3.0.13）。

## 2026-09-24 — `2026-09-24-disclosure-aux-v1-r10-route2` 全量 sweep 故障隔离

- 公司错误尾部 `AuxAuthRejected: HTTP 401` 后又出现 `AUX transport failure on 3 consecutive sources`，说明首次请求触发了一次认证恢复，但恢复后的连续页面/路由 transport exception 又被全量调度器升级为致命异常，掩盖了实际 source 状态。
- `[AUX-V1-r10-route2]` 仅在 AUX `collect()` 对批量 `all`/`all-designed` 增加 source-level failure isolation：连续三次 transport failure 时记录 source 并清零 bounded streak，继续后续日期/来源；精确 source/group 仍保留 circuit-breaker。401/403 认证拒绝和一次 force-new recovery 语义不变，仍 fail-closed，不做数据降级。
- AUX 测试 45/45 PASS；`py_compile`、`git diff --check` PASS。当前公司 run 的 401 根因仍需结合同一 run 的 `aux_crawler.log` 中首次 `source auth_rejected` 前后几行判断，不能仅凭尾部 RuntimeError 宣称平台恢复。
- r10-route2 EXE：`crawl_disclosure_aux_v1.exe`，38,469,330 bytes，SHA256 `C447D63E9BA22F5F1551863565CFE087E0ADD75710C0FCB976B9C1075E341A74`；`--help` PASS。公司状态仍为 `NOT_TESTED`。

## 2026-09-25 — `2026-09-25-disclosure-aux-v1-r11` 信息披露接口路径纠正（HAR20）

- **根因**：r10 之前登记为 `HAR_FRONTEND_JS` 的一批 QCTC 辅助信息接口（机组检修／输变电检修／备用／运行线等）路径是**从前端 JS 字符串推断**出来的，实测全部 HTTP 503 Whitelabel。HAR17/18/19 中共 18810 条 URL 里 `qctc_pm_trade_inside` **真实请求 0 次**。
- **HAR20 实测真实契约**（用户按 8 个信息披露 tab 逐一点击后抓包）：
  `GET /qctc/qctc_pm_trade_outside/informationDisclosure/<模块>/<方法>?pdate=YYYY-MM-DD[&versions=]`
  —— 是 **outside**（不是 inside），且带 `informationDisclosure` 段。每个请求携带
  **`X-Web-Path: /qctc-trade/informationDisclosure/<页面>`**；无 CSRF、无 Bearer、无 body。
- `[AUX-V1-r11]` 仅新增与最小修补，未改共享 `crawl.py`／`crawl_96_local.py`／认证／browser／state_machine／96 表：
  - 新增 11 个 `group=disclosure` source（HAR20 逐条 HTTP 200 证据）：`dcst_forecast_load`、
    `dcst_forecast_tieline`、`dcst_forecast_unit_overhaul`、`dcst_tmp_table_cols`、`dcst_tmp_load`、
    `dcst_tmp_update_time`、`dcst_tmp_block`、`dcst_tmp_spare`、`dcst_tmp_unit_overhaul`、
    `dcst_tmp_open_stop`、`dcst_tmp_trans_overhaul`。其中 `dcst_forecast_load`／`dcst_tmp_load`
    默认关闭（96 主链路已覆盖负荷），`dcst_tmp_table_cols`／`dcst_tmp_update_time` 为 raw-only。
  - `_aux_request` 在路径含 `/informationdisclosure/` 时注入 `Accept` + `X-Web-Path`（缺失会被网关拒绝）。
  - `_payload_rows` 增补 `TableData`；`_payload_container_state` 增补 `TableData`/`blockTreeData`/
    `chartsMap`/`tieLineTableCols`/`sparexx`/`updateTime`，避免把成功响应误判为空/PARTIAL。
  - 新增统一解析器 `parse_disclosure`（兼容 data 为 list／tableData／TableData／blockTreeData）。
  - 方法名是前端字符串拼接产物（如 `RealityTmpDataGetUnitOverhaulData`、
    `getTieLineRealityTmpDataGetPowerTransmissionAndTransformationOverhaulDataData`），**按原文照抄**。
- 回归：AUX 45→**51 tests PASS**；AUX + crawler runtime + auto_crawler + V10 P0 合计 **121/121 PASS**；
  `py_compile` PASS；隔离目录 `--db-check --no-db-upload` PASS（`db_schema` 校验，未连接真实库）；
  EXE `--help` PASS 且 `--source` 已列出全部 11 个 `dcst_*`。
- r11 EXE：`crawl_disclosure_aux_v1.exe`，38,472,398 bytes，SHA256
  `3A04B5360D3FC552D2745C9A37B3AD9E8BE8561A5739A43E79DC1329CCD5A4D4`；使用
  `dist/build_artifacts/venv_build`（Python 3.11.9 / OpenSSL 3.0.13）构建。r10-route2 EXE 已备份到
  `dist/crawler/archive/exe_versions_20260925/`。
- 公司真机状态：`NOT_TESTED`。建议首跑 `--source all-designed --lookback 0`，逐条核对
  `aux_report.json` 中 `dcst_*` 的 HTTP 状态；`ForecastData/getUnitOverhaulData` 在 HAR20 中为
  **504（nginx 网关超时，非权限）**，需按可重试处理。

### r11 公司真机结果（2026-09-25 21:26~21:32，首次真机 PASS）

run `20260925T133028Z-abfa77c9`，`--source all-designed --lookback 0`，date=2026-09-25：
33 sources → complete=9 / empty_valid=1 / partial=2 / failed=13 / skipped=8。**DB 写入成功**
（公司跑之前已把 `db_upload` 改为 true）：

| source | HTTP | rows | DB 写入 |
|---|---|---|---|
| `dcst_tmp_spare`（备用 96 点） | 200 | 170 | 171 行 |
| `dcst_tmp_block`（阻塞断面树） | 200 | 85 | 86 行 |
| `dcst_forecast_load` / `dcst_forecast_tieline` / `dcst_tmp_load` | 200 | 96 各 | ✓ |
| `dcst_tmp_unit_overhaul` / `dcst_tmp_open_stop` | 200 | 1 各 | ✓ |
| `dcst_tmp_table_cols` | 200 | 0（raw-only） | ✓ |
| `dcst_tmp_trans_overhaul` | 200 | 0（EMPTY_VALID，当天本就无数据） | ✓ |
| `dcst_forecast_unit_overhaul` / `dcst_tmp_update_time` | **504** | 0 | 网关超时 |

- 单点首跑（21:26 两次）FAIL 均为 **HTTP 401「登录信息失效」**（复用陈旧 CDP 会话），
  第三次（21:28）PASS。属认证会话过期，不是 r11 代码缺陷；force-new recovery 生效后正常。

## 2026-09-25 — `2026-09-25-disclosure-aux-v1-r11b` 真机结果收尾

依据 r11 首次真机报告做四处收尾，仍然只改 AUX 专属 `disclosure_aux.py`：

1. **14 个 `qctc_pm_trade_inside` 作废来源降级为 `UNVERIFIED`** —— `all-designed` 直接
   报告 `SKIPPED_NOT_READY`，**不再发送注定 503 的请求**（原本每轮浪费 14 次请求并制造噪声）。
   路径保留作历史，文档已注明作废原因。
2. **unitid 兜底**：`collect()` 在未显式传 `--unitid` 时回落到 `self.unit_id`
   （即 `config.json` 的 `unit_id` = `7B2B5622…`），解锁 unit_constraint /
   unit_component / unit_constraint_jjcq 三个依赖来源。
3. **504 重试**：browser_primary 遇到 504 退避重试最多 2 次（504 是 nginx 网关超时、非权限
   拒绝，HAR20 与真机都显示它是间歇性的）；重试耗尽仍如实报 FAILED，不伪造数据。
4. **修复 `unit_month_limit` 导航回归**：新增 AUX 专属 `_navigate_for_document()`，轮询
   `location.href` 直到 **path + query 命中目标**才读 DOM；共享的 `_navigate_for_fetch`
   只校验同源，会读到上一个页面的 560 KB DOM，导致 CSRF meta 丢失（r8 曾成功、r10 起失败）。

- 回归：AUX 55 + crawler runtime + auto_crawler + V10 P0 合计 **125/125 PASS**；
  `py_compile` PASS；隔离目录 `--db-check --no-db-upload` PASS。
  两个既有测试（r8 文档导航、all-designed 跳过策略）已按新行为同步更新。
- r11b EXE：`crawl_disclosure_aux_v1.exe`，38,473,075 bytes，SHA256
  `4E76A28E268BA202F36BEF42AB382965E5F0F4EED31F56B960982F8743C00821`；
  构建仍用 `dist/build_artifacts/venv_build`（Python 3.11.9 / OpenSSL 3.0.13）。
  r11 EXE 已备份到 `dist/crawler/archive/exe_versions_20260925/`。
- 公司真机状态：`NOT_TESTED`（r11b 尚未上真机）。建议复跑
  `--source all-designed --lookback 0`，重点看 `unit_month_limit` 是否恢复、
  以及 504 重试后是否转成功。
- 尚未纳入：HAR19 中长期模块（火电/新能源成交电量电价 `dlxxxqcx/dlxxxqYhCx.do`、`jygl/mymarket.do`
  统计、`jysbys/ydhdjzcx.do` 月度合约）。障碍是 legacy `/zcq` CSRF 取法需泛化，且中长期系统使用
  独立 `dyid/unitid`（真机值仅保留在本地运行证据，不进入治理文档）。`RealityData`（正式实际版）方法名
  仍无 HAR 证据，需 actual10426 页面的同题抓包。

## 2026-09-25/26 — `-r11c` / `-r11d`（登录恢复定稿，标记 STABLE）

### r11c：force_new 恢复不换 profile（对齐 96）

- 症状：复用会话后端过期 → force_new 重登，走完 CFCA 证书后**卡在 CERTIFICATE→PIN**，约 6 分钟后
  浏览器自行 `exit code=0`，CDP 断连，auth 600s 超时（2026-09-25 22:38 公司真机）。
- 根因：AUX force_new 分支曾新建临时 profile `pmos_auto_profile_aux_recovery_*`；96 的 force_new
  **只 `replace(browser_reuse=False)`、profile 不变**。临时 profile 下 UKey 客户端未初始化 →
  原生 PIN 弹窗不出现 → `WindowsPinHandler` 找不到窗口 → 状态机挂死。
- 修复：`crawl_disclosure_aux.py:_auth()` 改回 96 写法——只禁复用，**profile 目录不变**。
  仅改 AUX 专属文件；共享 `auto_crawler/` 与 96 源码**零改动**。

### r11c-v2：补全打包依赖（漏 requests/websocket-client/pymysql）

- 症状：公司机运行 r11c 首版立即 `ModuleNotFoundError: No module named 'requests'`。
- 根因：隔离 build venv 只装了 numpy/pandas/openpyxl/Pillow/pyinstaller，漏 AUX 运行时依赖。
- 修复：补装 `requests websocket-client pymysql pycryptodome` 后重打。EXE 43,009,160 B，
  SHA256 `85A3B4B0EC5D9A2CC6BBF105CB15C7C7EF66886CF39156086C6E1CD4C190A4A7`（本机 `ModuleNotFoundError=False`）。

### r11d：CDP 导航竞态容忍重试（**历史稳定回滚点；当时发布版**）

- 症状：`auth` 约 30s 失败 `Inspected target navigated or closed`——打开 `?service=...#/dashboard`
  重定向瞬间 evaluate 撞上；共享状态机因文本含 `"cdp"` 判**致命、不重试**。
- 修复：**只改 AUX 入口** `_auth()`，加「导航竞态容忍重试」：第 1 次按 force_new 启动；失败后 sleep 6s，
  第 2/3 次以 `browser_reuse=True` 挂回**同一浏览器**重试。新增 `_AUTH_MAX_ATTEMPTS=3`/`_AUTH_RETRY_DELAY_SEC=6.0`。
- **EXE（当前发布）**：`crawl_disclosure_aux_v1.exe`，**43,009,396 bytes**，SHA256
  `FAB2665D3218A2D102BDE5D4DD311AE1FE0CE64B3A02F2D4F2453709F29C5878`，
  BUILD_VERSION `2026-09-25-disclosure-aux-v1-r11d`。
- 备份（`dist/crawler/archive/exe_versions_20260925/`）：`..._r11c_v2.exe`（85A3B4B0…）、
  `..._r11c_broken_deps.exe`（漏依赖残废版，42,101,119 B）、`..._r11b.exe`。

### r11d 公司真机验证：**PASS**（2026-09-26 00:14~00:20）→ 标记 STABLE

run `20260925T161421Z-0f215f3b`（date=2026-09-24，source=all-designed），status=PARTIAL。

- **认证 PASS**：reuse 9222 成功（41s）；采集首源 401 → `AUX_BROWSER_RECOVERY_START` → force_new
  Edge（端口 9224，profile `pmos_auto_profile_msedge`）成功（54s）→ `collect_retry PASS`。
  **未再出现 CDP 导航竞态、未卡登录。**
- **落库**：db_schema_init/db_connect PASS；**15 源 upsert 全 PASS，合计 595 行**
  （spare 193 / forecast_load·tieline·tmp_load·block 各 97 / unit_info 2 / 机组检修 2 / 必开必停 2 / 其余各 1）。
- **采集**：33 源 → COMPLETE 11 / EMPTY_VALID 1 / PARTIAL 2 / FAILED_SOURCE 1 / SKIPPED 18；
  96 点级：dcst_forecast_load/tieline/block/tmp_load 各 96、spare 192。
- **残留**：`net_contract_day`/`generation_contract_limit` = HTTP 0 未取到数据；`unit_month_limit` =
  `LEGACY_ZCQ_PAGE_UNAVAILABLE url=/zcq/jysbys/ydfdcsxyhcx.do http_status=0`（legacy 页面不可用，
  **不是** CSRF）；18 源按设计跳过（14 UNVERIFIED + 4 缺 unitid）。

> **状态：r11d = AUX-V1 当前稳定版（STABLE）。** 公司真机跑通并成功写库。
> **除甲方新增明确需求外，尽量不再改动**；后续任何修改都必须重新真机验证并更新本节。

### 历史回补命令（2022 → 今）

```bat
cd /d "D:\爬虫电网\辅助信息披露爬虫"
crawl_disclosure_aux_v1.exe --date 2026-09-24 --lookback 1727 --source all-designed
```

1727 天覆盖 2022-01-01 ~ 2026-09-24。**必须用 `--source all-designed`**（monthly 源按月去重）；
**不要用 `--source all`** 做多年 daily lookback（monthly 源会每天重复）。建议按年拆批
（`--date 2022-12-31 --lookback 364` 等）；长跑期间需与 96 主爬虫错开（共享 `.crawler.lock`）。

## 2026-10-08 — `AUX-V1-r12` 探索模式 `--explore`（新增功能，**已打包 / 真机 NOT_TESTED**）

### 动机

公司电脑没有 Python 等任何运行环境，此前每次调查接口都要人工 F12 导出 HAR 再搬回来，
做好爬虫后还要搬到公司机试跑、把日志搬回来复盘。r12 把「调查接口」这一步做成
**一条命令 + 一个 zip**：登录后由人正常点网页，程序后台被动录制，自动生成中文《发现清单》。

### 范围与红线（加法，不碰已验证主链）

- 新增 `scripts/crawler/collect/crawl_disclosure_aux_explore.py`（独立旁路模块），
  在 `scripts/crawler/apps/crawl_aux.py` 加 `--explore` 分发（`--doctor` 分支之后）。
- **只读录制**：不解析入库、不写数据库、不调用任何采集接口、不改 AUX/96 业务代码；
  共享认证与采集实现**只 import 复用，不修改**（`_auth`、`_resolve`、`load_aux_config`、
  `configure_aux_logging`、`SOURCE_REGISTRY`、`RunReport`、`RuntimeLock`）。
- 业务 `BUILD_VERSION` **保持工作区现值 `2026-09-29-disclosure-aux-v1-r11m` 不动**
  （避免与其它窗口在飞的 r11e–r11m 冲突）；探索模块自带
  `EXPLORE_BUILD = 2026-10-08-disclosure-aux-v1-r12-explore`，两者在产物与日志里同时可见。
- 凭证脱敏后才落盘（Cookie / Authorization / CSRF / ticket / token / 口令；口令类按整词、
  认证类按子串，避免误伤 `mapping` 等业务字段）；zip 内排除日志与配置文件。

### 工作方式

CDP 浏览器级 WebSocket + `Target.setAutoAttach(flatten)` 逐个 session `Network.enable`，
被动收 `requestWillBeSent`/`responseReceived`/`loadingFinished`/`loadingFailed`，
按 (方法, host+path, 排序参数名集合) 聚合接口契约（同一接口点 30 天合成 1 条 + 参数值样例），
从响应体抽点分字段路径并按 6+1 组目标关键词（供需/系统预测/检修/火电合约占比/煤价/钢铁网/价格类）打标，
对照 `SOURCE_REGISTRY` 标「已登记 / 未登记（候选新来源）」。控制台输入**非空行**才结束录制。

### 本地验证（真机前）

- 新增 `scripts/tests/test_crawl_aux_explore.py`：12 test 全 PASS（静态过滤、凭证脱敏、
  参数值不敏感聚合、session finalize 与清单内容、body 失败非致命、zip 排除日志/配置、
  apps 分发不落到采集实现、模块缺失返回 2、EOF≠回车、GBK 打印安全）。
- AUX 既有套件：r12 前 3 failed / 52 passed → r12 后 **3 failed / 64 passed**（同一批既有失败，未新增）。
- 共享 crawler 回归（runtime / auto_crawler / V10 P0）**70 passed**。
- 无头 Chrome 真链路 probe（`outputs/diagnostics/probes/explore_cdp_smoke/20261008/`）：
  本地站点 + 真实 CDP 录制 9 条业务请求 → 2 个接口契约，跳过 6 个静态资源，body 取回错误 0，
  清单正确列出 `参数名 d`、`参数值样例 d=2026-10-01`、响应字段 `data.tableData[].contractRatio`
  与命中「火电合约占比」。

### probe 期间发现并已在生产代码修掉的坑

- Chrome ≥111 对 DevTools WebSocket 的 Origin 校验返回 403：`_open_socket()` 用
  `suppress_origin=True`（L1-REUSE 挂回的不是我们自己启动的浏览器，不能假设它带
  `--remote-allow-origins=*`）。
- CDP auto-attach 会带上扩展 / service-worker / favicon 噪声流量：新增
  `_ALLOWED_TARGET_TYPES`（page/iframe/webview/other）与 `_is_business_url`
  （排除 `chrome://`、`devtools://`、`blob:`、`file://` 等）。
- stdin **EOF 不等于回车**：原实现会被 EOF 立即结束录制，已改为只接受非空行，
  并把 `EXPLORE_NO_CONSOLE` 记为 WARN 继续录制。
- 接口参数样例不能被 `_safe_diag_text` 处理：它的凭证正则以 `[^\s,;]+` 收尾，会把整串
  `a=1&b=2` 吃掉（对错误日志正确、对参数值样例致命），故本模块自带 `_flat()` 只做压平与截断。

### 打包前发现的基线偏差（重要，先看这段再回滚）

打包 r12 前核对发布目录里的 EXE，发现**上一节写的「当前发布 = r11d」已经不成立**：

- `dist/crawler/辅助信息披露爬虫/crawl_disclosure_aux_v1.exe` 实际是
  **2026-09-29 21:36 构建、43,092,292 B、SHA256 `3210e3ee…`，自报版本 `2026-09-29-disclosure-aux-v1-r11m`**。
- 归档区 `exe_versions_20260926/` 里已有 `r11e / r11f / r11g / r11h / r11i / r11k / r11l` 七个包，
  r11d 也已归档（`fab2665d…`，与上一节记录一致）。
- 结论：r11e–r11m 由其它窗口连续构建并部署，但**没有写进本记录与发布台账**（相关过程只出现在
  `.workbuddy/memory/2026-09-29~10-01.md`）。因此**不能把 r11d 当作当前回滚点**；
  现网真实基线是 r11m。r12 相对它的源码增量只有 `apps/crawl_aux.py` 与
  `crawl_disclosure_aux_explore.py` 两个文件（已按 mtime 核对，r11m 之后无其它 crawler 源码改动），
  即 **r12 = r11m 采集链 + 探索模式**，不含来路不明的第三方改动。
- 待补：r11e–r11m 的真机状态需由负责该线的窗口确认并补记本节；本节的 r12 打包不替它们背书。

### EXE 构建与本地 smoke（2026-10-08）

- 规格：`dist/build_artifacts/aux_v1_r12_spec/crawl_disclosure_aux_v1.spec`
  （入口 `scripts/crawler/apps/crawl_aux.py`；`hiddenimports` 在 r11c/r11g 基础上
  显式加 `scripts.crawler.collect.crawl_disclosure_aux_explore`，`websocket` 仍在列）。
- 构建：`dist/build_artifacts/venv_build`（OpenSSL 3.0.13，符合打包红线）。
- 产物：**38,589,245 B**，SHA256 `11bf6b230d9d2b71814c2547a5938938fbc37c61f27111af394723f388f9d32a`，
  自报 `2026-09-29-disclosure-aux-v1-r11m` + 探索能力 `2026-10-08-disclosure-aux-v1-r12-explore`。
- 体积比 r11m 小包约 4.5 MB，**不是缺依赖**：与 `build/aux_r11i` 的 `Analysis-00.toc` 逐项对比，
  `websocket / requests / pymysql / scripts.crawler.*` 条目数完全一致，差异全部来自构建 venv 里
  `pytz` 已被 `tzdata + 标准库 _zoneinfo` 取代（numpy 2.4.6 / pandas 3.0.5）。
  AUX 采集链本身不 import pandas，pandas 是被认证/滑块链拖进来的，故该漂移不改变 AUX 业务行为；
  **但这条 venv 漂移是既成事实，后续任何重打包都会带上，别再按体积判断包好坏。**
- 冻结包 smoke（在隔离的 staged 目录跑，不碰发布目录配置）：
  `--help` 正常打印采集 CLI 且首错非 `ModuleNotFoundError`；`--explore --help` 正常进入探索分支并列出
  探索参数；`--doctor` 分支顺序未被 `--explore` 分发破坏。

### 状态

- EXE：**已打包并已放入发布目录**（覆盖前已把 r11m 归档为
  `dist/crawler/archive/exe_versions_20261008/crawl_disclosure_aux_v1_r11m.exe`，
  新包同时另存一份 `crawl_disclosure_aux_v1_r12_explore.exe`）。
- 公司真机：`NOT_TESTED`。探索模式必须在公司机（唯一能登录的地方）跑一次才算通过；
  真机通过前，r11m 仍是**唯一有部署历史**的 AUX 包，出问题直接回滚那个文件即可。
- 采集链：零改动，`--source all-designed` 等既有命令与写库行为不受影响（冻结包 `--help` 与
  `--doctor` 已验证；但采集真机状态继承自上面「待补」的 r11e–r11m，不由本次 r12 重新背书）。

## 2026-10-08 — `2026-10-08-disclosure-aux-v1-r13` 检修计划 + 火电合约占比（先爬通，**已打包 / 真机部分 PASS：检修链路通，合约链路两缺陷待修**）

### 动机

业务要求「检修计划、火电合约占比改程序先爬一下看看」。r12 真机探索
（`outputs/diagnostics/agent/2026-10-08/aux_explore_r12/接口对照分析.md`）已拿到契约：

- 检修类真实接口在 informationDisclosure **`ForecastData/*` 模块**——原 registry 里
  5 条 `RealityTmpData/*` 路径是虚构的（真机证实 RealityTmpData 只有
  getLoadData/getTableCols/getUpdateTime），这正是 AUX 一周白爬的根因之一；
- 本主体合约成交曲线在 zcq `dlxxxqYhCx.do` 的 `get24/96CjTableData`
  （页面 appkey=21/15，HTML 带 `_csrf` meta，r12 bodies/0152/0154/0156）；
- 全省口径的中长期合约成交行 `net_contract_day`（appkey=18）此前从未默认启用。

### 变更（只动 AUX，不动 96 链路，**不加 DDL**）

1. 重指 5 条虚构死路径：`dcst_tmp_block→ForecastData/getBlockData`、
   `dcst_tmp_spare→getSpareData`、`dcst_tmp_open_stop→getOpenAndStopUnitData`、
   `dcst_tmp_trans_overhaul→getPowerTransmissionAndTransformationOverhaulData`（23 行主变/线路检修真机证据）、
   `dcst_tmp_unit_overhaul→getUnitOverhaulData`（与 dcst_forecast_unit_overhaul 同端点，改禁用别名）。
   page 全部切到 `forecast10424`，参数契约 `pdate+versions`，evidence 升 `EXPLORE_NETWORK`。
2. 新增 `zcq_contract_curve24` / `zcq_contract_curve96`（POST query
   `method=get24|96CjTableData&dyid&userProp=1&sDate/eDate=整月&jylx=ALL` + DataTables
   draw/start/length，offset 分页 length=50；行 `{pdate:"20261001",point,cjdl,cjjj}`，
   recordsFiltered 真机证据 744）。unitid 依赖型（`--unitid`/config，不 fan-out）。
3. `net_contract_day` 启用默认采集。
4. CSRF 泛化：`_unit_month_limit_csrf_headers` → `_zcq_csrf_headers(page_url)` 按页缓存；
   `_ZCQ_CSRF_SOURCES = {unit_month_limit, net_contract_day, zcq_contract_curve24/96}`。
5. `parse_contract`：quantity 加 `cjdl`、price 加 `cjjj`；曲线源 `use_row_date=True`
   （行内 pdate 进 record_key，否则同月同点行碰撞）+ 采集 dyid 注入为 unit_id；
   旧 contract 源行为逐字不变。
6. 顺带修 r11f 遗留：`_browser_lost_count` 未在 `__init__` 初始化——首个 transport
   失败在 except 分支里直接 AttributeError（真实运行会崩，非仅测试问题）。

### 本地测试

- `test_disclosure_aux.py` **63 passed**（新增 R13 类 8 用例）；`test_crawl_aux_explore.py` 12 passed。
- 同步 3 处存量漂移（均为其它窗口 r11m/r11f 未提交改动 vs r11b 测试的脱节，非本版本功能）：
  inside 计数 14→12（r11m 检修源迁址）、force_new 不再注入 `_aux_recovery_` 临时 profile
  （r11f 设计：临时 profile 吞 CFCA/UKey 弹窗）、reality_tmp 断言源 spare→load（原源已重指）。

### 数据库

- 无 DDL：全部进 `epf_pmos_aux_records`（record_json/raw_json），爬通看到真实数据后再决定
  是否在 96 表加消费列。

### EXE 构建与发布（2026-10-08）

- 规格：`dist/build_artifacts/aux_v1_r13_spec/crawl_disclosure_aux_v1.spec`（与 r12 spec
  仅注释/workpath 差异）；构建 venv：`dist/build_artifacts/venv_build`。
- 产物：**38,590,564 B**，SHA256
  `1b2522fdbf63ab617bb146095d2173d5dcad3ff03bb54a6e601d1dd2db348d7a`，
  自报 `2026-10-08-disclosure-aux-v1-r13`。
- 冻结包 smoke（staged 目录）：`--help` 正常且源清单含 `zcq_contract_curve24/96`；
  `--explore --help` 正常；无 ModuleNotFoundError。
- 发布：已覆盖发布目录 EXE；归档 `dist/crawler/archive/exe_versions_20261008/crawl_disclosure_aux_v1_r13.exe`。
- 回滚点：r12 = `crawl_disclosure_aux_v1_r12_explore.exe`（同归档区）；
  最后一个有真机 PASS 的仍是 r11d。

### 状态

- 公司真机：**部分 PASS（2026-10-08 20:13，run `20261008T121031Z-b169b713`，`--source all`，两日）**。
  - 检修链路 PASS：`dcst_tmp_trans_overhaul` 23 行（r11l 前为 0 行）、`dcst_tmp_spare` 192、
    `dcst_tmp_block` 96、`dcst_tmp_open_stop` 1、`dcst_forecast_unit_overhaul` 1、
    `dcst_forecast_tieline` 96；DB `epf_pmos_aux_records` 有写入。
  - **缺陷 1（待批后修）**：`parse_disclosure` 的 `record_key` 粒度不足——设备检修 23 行
    在库里塌成 2 行、备用 192 行塌成 97 行（正/负备用碰撞）。键里只取了
    `pdate/type/mold/dataType` 的第一个非空值，`subjectName`/`valuetime1`/`valuetime2` 未进键。
  - **缺陷 2（待批后修）**：`net_contract_day`、`unit_month_limit` 仍
    `LEGACY_ZCQ_CSRF_MISSING`（该错自 r11i 起持续存在，非 r13 回归）。日志显示 document
    读到的 body_len 为 555,235 / 561,814 / 618，而 r12 录制里 appkey=18/21/15 的页面只有
    15,057 / 11,002 / 11,208 B 且都带 `<meta name="_csrf">`，即导航没有落在目标 appkey 页上。
  - 曲线源 `zcq_contract_curve24/96` 报 `AUX_DEPENDENCY_MISSING unitid`：需显式
    `--unitid <dyid>`（真机实值不进入版本治理文档）。
- 原始计划命令（已由 `--source all` 覆盖）：先
  `crawl_disclosure_aux_v1.exe --source dcst_tmp_trans_overhaul --capture-only` 单源看
  检修原始数据，再 `--source zcq_contract_curve24` 看 744 行合约曲线，最后全量。

## 2026-10-08 — `2026-10-08-disclosure-aux-v1-r14` 三项收口修补（当前发布）

### 目标与变更

r14 不新增数据库表，也不扩大 96 主爬虫影响面，只针对 r13 真机暴露的三项缺陷做最窄修补：

1. `zcq_contract_curve24/96` 不再把 CLI `--unitid` 作为硬前置；优先使用已有配置/采集上下文可确定的机组 ID，避免“页面已有真实 dyid 但批次仍 SKIPPED”的假缺依赖。
2. `parse_disclosure` 的 `record_key` 粒度扩展到 `mold / subjectId / subjectName / valuetime1 / valuetime2` 等稳定业务维度，修复 r13 中检修 23 行塌成 2 行、备用 192 行塌成 97 行的问题。
3. legacy ZCQ 文档导航后显式验证目标 DOM 的 `_csrf` meta，并记录导航 URL/body_len/meta 命中诊断；拿不到目标页 CSRF 继续 fail-closed，不发送猜测 POST。

### 当前源码与构建

- 业务核心：`scripts/crawler/collect/disclosure_aux.py`
- 应用入口：`scripts/crawler/apps/crawl_aux.py`
- tracked canonical spec：`scripts/crawler/build/crawl_disclosure_aux_v1.spec`
- 发布 EXE：`dist/crawler/辅助信息披露爬虫/crawl_disclosure_aux_v1.exe`
- 文件大小：38,590,704 bytes
- SHA256：`7FAD847631A0DC7FA9A9D83C67631150EA307D7BFBBBF24F66F0793C6E518388`
- 自报版本：`2026-10-08-disclosure-aux-v1-r14`

### 状态

- 本地发布身份与版本台账一致。
- **公司真机：NOT_TESTED。** r14 修的是 r13 真机缺陷，因此在公司机重新跑到对应检修/备用/legacy ZCQ/curve source 之前，不得把 r14 标成稳定版。
- 最近一个完整真机稳定回滚点仍为 r11d；r13 仅为 `PARTIAL_LIVE_PASS`。

### Git 治理补充（2026-10-10）

此前 Git 只提交了 `collect/disclosure_aux.py` 的 r14 核心，而 `apps/crawl_aux.py`、`crawl_disclosure_aux.py`、explore、AUX DB writer、resilience/runtime_lock 等实际运行依赖仍停留在 working tree。此次 crawler-only 收口必须把**当前运行依赖整体**纳入同一个源码状态；archive EXE/历史文档仍只留本地，不进入本次提交。
