# PMOS 辅助信息披露爬虫 AUX-V1

与96点主爬虫业务隔离，复用认证/浏览器底座。当前生产写库表为 `epf_pmos_aux_records`，不修改 `epf_pmos_96_full`。

> **当前发布（2026-10-10 核验）**：`crawl_disclosure_aux_v1.exe` 自报 `2026-10-08-disclosure-aux-v1-r14`，38,590,704 bytes，SHA256 `7FAD847631A0DC7FA9A9D83C67631150EA307D7BFBBBF24F66F0793C6E518388`。r14 尚未公司真机复验，状态 **NOT_TESTED**；r13 为 **PARTIAL_LIVE_PASS**；最后一个完整真机稳定回滚点仍为 r11d。详细 revision 以 `AUX-V1_版本记录.md` 为准。

## 使用

```powershell
python scripts/crawler/collect/crawl_disclosure_aux.py --help
python scripts/crawler/collect/crawl_disclosure_aux.py --db-check --no-db-upload
python scripts/crawler/collect/crawl_disclosure_aux.py --date YYYY-MM-DD --capture-only
```

正式写库命令应在部署配置确认 `db_upload=true` 后运行：

```powershell
.\crawl_disclosure_aux_v1.exe --date YYYY-MM-DD --lookback 0 --source all
```

一次调度全部已登记来源（r10 新入口）：

```powershell
.\crawl_disclosure_aux_v1.exe --date YYYY-MM-DD --lookback 0 --source all-designed
```

`all-designed` 与旧 `all` 有意区分：旧 `all` 仍只跑默认启用的 `unit_info`、`unit_month_limit`；
`all-designed` 会在同一轮遍历 source registry，日来源逐日调度、月来源每月一次。参数未确认、
证据标为 `UNVERIFIED`、缺少 `unitid` 或必要参数的来源会在 report 标为
`SKIPPED_NOT_READY`，不会发送猜测请求；有些已登记 source 是 raw-only，只能保证原始响应入库，
不会产生 structured 字段。存在此类跳过/失败时本轮为 `PARTIAL`，不能当作全字段成功。

不添加 `--capture-only`、`--dry-run` 或 `--no-db-upload` 才会写数据库。`--db-check` 在可用 DB 配置且允许写入时会创建 AUX 表；带 `--no-db-upload` 只校验 DDL。

## 探索模式 `--explore`（AUX-V1-r12，免 F12）

> 当前 r14 EXE **继续包含** r12 引入的探索模式；探索入口与业务采集入口均由 `scripts/crawler/apps/crawl_aux.py` 分发。当前发布身份见本文顶部。探索能力曾在 r12 独立构建中完成本地 smoke，但 r14 整包仍需按当前版本重新做公司真机验证，不能沿用旧 SHA 当成当前发布身份。

用于回答「这个业务字段到底在哪个接口、要哪些参数」。它**只录制、不采集、不入库**，
替代「F12 手动导出 HAR 再拷回来」的老流程。

```bat
cd /d "D:\爬虫电网\辅助信息披露爬虫"
crawl_disclosure_aux_v1.exe --explore
```

操作流程（公司电脑只需要这一条命令）：

1. 程序照常走 AUX 认证（UKey/Cookie 复用现有链路），浏览器打开并登录；
2. 把浏览器切到前台，**像平时查数据一样逐个点开你要调查的页面**，每个页面等数据加载出来再点下一个；
3. 期间可以自由开新标签页、来回切换，全部都在录制范围内，不用按 F12、不用导出任何东西；
4. 点完回到命令行窗口按【回车】结束（结束前不要关浏览器）；到 `--explore-max-sec`（默认 1800 秒）也会自动结束并出清单，不会白跑。

产物落在 `output_aux/explore/<run_id>/`，同时在旁边生成一个可直接拷回的
`explore_<run_id>.zip`：

- `发现清单.md`——中文清单。按「目标字段命中概览 → 接口明细」组织，先给出供需关系 / 系统预测 / 检修计划 / 火电合约占比 / 煤价 / 钢铁网 / 价格类这 6+1 组关键词的命中情况，再逐个接口列出：调用次数、状态码、**参数名、参数值样例**、是否已在 AUX 登记表（`已登记` / `未登记（候选新来源）`）、价值线索、响应字段路径和一行样例数据；
- `requests.jsonl`——按接口契约（方法 + 路径 + 参数名集合）聚合的请求索引，同一接口点 30 天会合成 1 条契约；
- `bodies/`——每个接口一份原始响应，供开发机复现字段；
- `explore_meta.json`——录制元数据（端口、结束原因、请求计数、静态资源跳过数、body 取回失败数）。

可选参数：`--explore-no-bodies` 只记接口与字段名不存响应体（产物最小，先把清单拷回时可用）；
`--explore-out <绝对路径>` 换产物目录；`--config` 指定 AUX 配置。

脱敏红线：Cookie / `Authorization` / CSRF / ticket / token / 口令类键值一律脱敏后才落盘，
参数值样例保留可读性；zip 内**不含** `aux_crawler.log` 与 `config_disclosure_aux.json`，
所以整包拷回不带走会话凭证和数据库密码。

探索模式与采集链路共用 `.crawler.lock`，不能和 96 主爬虫或 AUX 正常采集并发。
细节实现见 `scripts/crawler/collect/crawl_disclosure_aux_explore.py`。可复现打包规格以 `scripts/crawler/build/crawl_disclosure_aux_v1.spec` 为 canonical；`dist/build_artifacts/` 仅是本机构建工作区。

## 部署配置

EXE 目录的 `config_disclosure_aux.json` 指向认证配置、数据库配置和输出目录。公司电脑如只复制 AUX 文件夹，需复制/配置有效的 `config.json` 与 `db_config.json`；不要把真实账号密码提交到仓库。缺少 AUX 配置时程序会生成安全默认配置 `db_upload=false`。

## 运行产物和隔离

- raw：`output_aux/raw/YYYY-MM-DD/`
- 日志：`output_aux/aux_crawler.log`
- 累计报告：`output_aux/aux_report.json`
- 与主爬虫共享 `../output_96/.crawler.lock`，两套程序不能并发。
- 事件、检修和月度合约不广播为96点；`jzdlzb` 未确认业务定义前不解释为火电合约占比。

`r10-diag1` 在认证、QCTC context、数据库 schema/connect/upsert、日期批次、单个 source 请求/解析/raw 保存等阶段增加开始/结束/耗时/失败日志。超过20秒未结束的阶段每20秒输出 heartbeat；report 中对应 stage 会保留 `RUNNING`，方便识别卡在哪一步。HTTP 错误只记录截断且脱敏的响应摘要；成功响应正文不写普通日志。若程序被强制结束，查看 `aux_report.json` 最新 run 的 `stages` 和 `aux_crawler.log` 最后一个 `AUX phase` / `AUX source` 即可定位，不要只看是否有浏览器页面。

`r10-diag2` 对齐已验证可运行的96点传输路径：现代 `/qctc/` 来源（如 `unit_info`）必须通过已认证浏览器同源 `fetch` 发送，不能先用 Python `requests`；门户 Cookie 不包含 QCTC sessionStorage 上下文，Python 直连会得到 HTTP 401。旧 `/zcq` 来源仍保留各自的 CSRF/Python-first 契约。此修补只改 AUX adapter，不修改共享 `crawl.py`、`crawl_96_local.py` 或96点表。

`r10-route1` 按 HAR16/17/18 收敛旧 ZCQ 的分路：同一个 AUX 进程可以在已认证 CDP 标签页中按来源切换页面上下文，再发送对应请求，不需要人工逐个打开页面。`net_contract_day` 使用 `POST /zcq/JyjgZcqXxpl.do?method=getTableDate`（appkey=18）；`generation_contract_limit` 使用 appkey=93；`generation_hourly_net` 使用 appkey=94；`unit_month_limit` 使用 appkey=81，并先完成 HTML/CSRF 文档导航。共享浏览器底座未改，AUX 只在发送 legacy 浏览器请求前做精确同源导航；Python 直连成功时仍保留原有优先路径。

`r10-route2` 修复批量 sweep 的故障隔离：某个页面连续三次 transport exception 不再直接掩盖 `all`/`all-designed` 批次，程序会记录具体 source、保留 raw/失败状态并继续后续日期/来源；窄范围 source 仍保留三次 circuit-breaker。HTTP 401/403 仍按认证拒绝执行一次 force-new recovery，重试仍被拒绝时不会伪造数据。

这解决的是“同一爬虫访问不同 PMOS 页面上下文”的程序问题，不等于平台对所有来源都已授权。没有 HAR 真实 HTTP 200 证据的来源仍保持 `UNVERIFIED/SKIPPED_NOT_READY`，不能为了让批次显示成功而猜测接口。批量运行应先使用 `--source all-designed --lookback 0`，审阅 `aux_report.json` 中每个 source 的 HTTP 状态、raw 和 structured 行数；出现 503/504 仍应按平台上游或权限问题处理，而不是用别的来源补值。

## Source 选择

`--source all` / group 仅包含 `enabled_by_default=true` 的来源；当前默认 source 为 `unit_info`、`unit_month_limit`，以及 r11 新增的 `dcst_forecast_tieline`、`dcst_forecast_unit_overhaul`、`dcst_tmp_block`、`dcst_tmp_spare`、`dcst_tmp_unit_overhaul`、`dcst_tmp_open_stop`、`dcst_tmp_trans_overhaul`、`dcst_tmp_table_cols`、`dcst_tmp_update_time`（`dcst_forecast_load`/`dcst_tmp_load` 与 96 重复，默认关闭）。其它来源可逐项显式调试，也可由 `--source all-designed` 统一调度。批量模式不解除安全契约：依赖 unitid 的 source 缺少真实 ID 时不发 HTTP；UNVERIFIED、参数合同不完整的来源明确跳过；raw-only source 不会伪造结构化记录。

## 已确认的运行契约

- r2：`unit_type` / `unit_gengroup` 只保存 raw，不生成 structured unit 记录；依赖 unitid 的约束 source 不会发送空 unitid；`all` 与 group 不会隐式启用 disabled source；`unit_master` 分页有 max_pages=200、max_rows=100000 上限，未抓齐为 `PARTIAL/pagination_pending`。
- r3：默认 source 收敛到 HAR 有真实 200 证据的 `unit_info`、`unit_month_limit`；`jzdlzb` 原值保留，比例业务定义为 `UNDEFINED`。
- r4：source 请求 401/403 后只进行一次 force-new 浏览器认证和一次重试，不循环。
- r5：`unit_month_limit` 按 HAR 页面读取 CSRF meta，POST 带动态 CSRF header；无法取得 token 时 fail-closed，不发送无 token 请求。
- r6：数据库收敛为单一统一表，raw 与 structured payload 分开保留，公共 provenance 列可直接筛选。
- r7：`unit_month_limit` 的 appkey=81 CSRF 页面按 HTML document 请求，移除继承的 AJAX 标头；直连非200时以相同 HTML 标头进行一次浏览器 fetch，仍失败则 fail-closed。
- r8：HAR18 确认成功路径是 HTTP 200 的完整 HTML 文档导航（含 CSRF meta），而 r7 same-origin fetch 得到 155 字节无 token 响应；AUX 现在在已认证标签页执行真实文档导航并读取 DOM，失败仍 fail-closed。
- r10-diag2：现代 QCTC API 参照96点 `_qctc_get` 改为浏览器同源 transport primary；不能把 Python 401 当成最终结果。

## 数据库

单表 `epf_pmos_aux_records` 同时保存 raw 与结构化记录：`record_kind=raw` 的平台原始响应放入 `raw_json`；`record_kind=structured` 的完整解析字段放入 `record_json`。公共列记录来源、业务日期、状态、请求键、raw_hash、run_id 和 captured_at。字段说明及查询示例见 `辅助信息披露数据库字段说明.md`，DDL 见 `辅助信息披露数据库表设计.sql` 和源码 `scripts/crawler/sync_db/sql/003_create_epf_pmos_aux.sql`。

`captured_at` 的 MySQL 列为不带时区的 `DATETIME`；写库时将 ISO-8601 时间解析为 MySQL 可接受的 datetime 值，并去掉列无法保存的时区标记。结构化 `record_json` 与本地 raw 仍保留来源时间字符串。

部署排错、r9 首次实网写库证据和历史回补方式见 `辅助信息披露爬虫_AUX-V1最终架构与实施设计.md` 第 14 节。尤其注意公司 EXE 同目录配置必须 `db_upload=true`、DDL 必须与 EXE 同目录；旧 `--source all` 不应用于多年 daily lookback。r10 `all-designed` 会对 monthly source 按月去重，但多年全来源回补仍应先做单日 smoke、审阅逐来源 report 后再运行。

### r10 Windows TLS 打包要求

r10 初版若以 `epf-2`（OpenSSL 3.6.x）打包，PyMySQL 初始化 Windows 系统证书时可能报 `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`。当前发布 EXE 已用 `dist/build_artifacts/venv_build`（OpenSSL 3.0.13）重打包；不要关闭 TLS 校验或用部署机环境变量绕过。请使用本目录当前 EXE，并核对版本记录中的 SHA256。公司真实采集仍需重新验证。

原 AUX 七表仅作为旧版本设计记录；本次不删除已有表。当前项目配置所查数据库未发现旧 AUX 表。

## 2026-09-24 r10-route2 公司新日志复盘

最近一次 `--source all-designed --lookback 0` 的 run_id 为
`20260924T143513Z-70af553f`，build=`2026-09-24-disclosure-aux-v1-r10-route2`，
最终 `PARTIAL`，不是完整成功：22 个 source 中 `complete=1`、`failed=12`、
`partial=1`、`skipped_not_ready=8`。

- 认证本身成功：新浏览器 `debug_port=9224`，QCTC ticket HTTP 200，约 7.9 秒后
  `bearer_present=true`。因此这次尾部的 transport 错误不能再笼统归因于“没有登录”。
- `unit_info` 已真正成功：HTTP 200、structured=1，数据库写入 raw=1 + structured=1。
- `unit_master`、`unit_type`、`unit_gengroup`、`special_unit_tag`、
  `transmission_maintenance`、`reserve_security`、`run_line`、`debug_line` 等
  inferred QCTC inside 来源仍返回 HTTP 503 Whitelabel；这说明平台 upstream/接口证据仍不成立，
  不是解析器成功后丢数据，raw 已保存，不能用其他字段补值。
- `net_contract_day` 在切换 appkey=18 时出现 `Inspected target navigated or closed`；
  这是 CDP 页面切换竞态，已记录为失败并继续批次，不应伪称接口无数据。
- `generation_contract_limit` 返回 HTTP 0 空响应，保留为 `PARTIAL`；需要后续单独复验
  appkey=93 页面与 POST，不可当作完整结果。
- `unit_month_limit` 页面本身 HTTP 200、body 约 560 KB 且浏览器 token 存在，但页面中未找到
  当前正则能识别的 `_csrf`/`_csrf_header` meta，因此严格报
  `LEGACY_ZCQ_CSRF_MISSING`，没有发送无 CSRF 的月度 POST。这是页面模板/CSRF 注入形式待核对，
  不是数据库写入失败。

当前结论：r10-route2 已解决“连续 3 次 transport exception 直接掩盖后续 source”的调度问题，
但没有把 503、HTTP 0 或缺 CSRF 伪装成成功。后续如要继续修月度接口，需要提供该次
appkey=81 HTML 的安全字段名/页面 HAR（不提供 token 值）；如要启用 inside 来源，需要对应
真实 network HTTP 200 HAR，而不是仅有前端 JS 路径。

## 2026-09-25 r11：信息披露接口路径纠正（HAR20 实测）

**旧登记的 inside 路径全部作废。** 此前 `special_unit_tag`／`transmission_maintenance`／
`reserve_security`／`run_line`／`debug_line` 等来源标注为 `HAR_FRONTEND_JS`，路径是从 QCTC
前端 JS bundle 里读字符串推断的；HAR17/18/19 合计 18810 条 URL 中 `qctc_pm_trade_inside`
**真实请求 0 次**，实测一律 HTTP 503 Whitelabel。这些条目保留但默认关闭，不再作为目标。

HAR20（用户在「市场披露预测信息查询」页逐个点开 8 个 tab 后抓包）给出真实契约：

```
GET https://pmos.sd.sgcc.com.cn:18080/qctc/qctc_pm_trade_outside/informationDisclosure/<模块>/<方法>?pdate=YYYY-MM-DD
请求头：Accept: application/json, text/plain, */*
        X-Web-Path: /qctc-trade/informationDisclosure/<页面>     ← 必须，缺失会被网关拒绝
        无 CSRF、无 Bearer、无 body
```

`[AUX-V1-r11]` 新增 11 个 `group=disclosure` 来源（每个都有 HAR20 HTTP 200 证据）：

| source | 模块/方法 | 页面 | 说明 |
|---|---|---|---|
| `dcst_forecast_tieline` | `ForecastData/getTieLineData` | forecast10424 | 联络线预测（带 `versions=`） |
| `dcst_forecast_unit_overhaul` | `ForecastData/getUnitOverhaulData` | forecast10424 | 机组检修（预测侧，HAR20 为 504，可重试） |
| `dcst_forecast_load` | `ForecastData/getLoadData` | forecast10424 | 负荷预测（96 已覆盖，默认关闭） |
| `dcst_tmp_spare` | `RealityTmpData/getSpareData` | actualTemporary10425 | **备用**（正/负备用 96 点） |
| `dcst_tmp_unit_overhaul` | `RealityTmpData/RealityTmpDataGetUnitOverhaulData` | actualTemporary10425 | **机组检修** |
| `dcst_tmp_trans_overhaul` | `RealityTmpData/getTieLineRealityTmpDataGetPowerTransmissionAndTransformationOverhaulDataData` | actualTemporary10425 | **输变电检修** |
| `dcst_tmp_open_stop` | `RealityTmpData/RealityTmpDataGetOpenAndStopUnitData` | actualTemporary10425 | **必开必停机组** |
| `dcst_tmp_block` | `RealityTmpData/getBlockData` | actualTemporary10425 | **阻塞**（断面树） |
| `dcst_tmp_table_cols` | `RealityTmpData/getTableCols` | actualTemporary10425 | 表头定义（raw-only） |
| `dcst_tmp_update_time` | `RealityTmpData/getUpdateTime` | actualTemporary10425 | 更新时间（raw-only） |
| `dcst_tmp_load` | `RealityTmpData/getLoadData` | actualTemporary10425 | 实际负荷（96 已覆盖，默认关闭） |

方法名是前端字符串拼接产物，**必须按原文照抄**，不要“规范化”。
`X-Web-Path` 由 AUX 在请求时自动注入；`pdate` 用纯日期；`ForecastData` 系列额外带空值 `versions=`。

**尚未纳入**：中长期模块（火电/新能源成交电量电价 `dlxxxqcx/dlxxxqYhCx.do`、
`jygl/mymarket.do` 统计、`jysbys/ydhdjzcx.do` 月度合约）——需要泛化 legacy `/zcq` 的 CSRF 取法，
且该模块使用自成一体的 `dyid/unitid`。`RealityData`（正式实际版）的方法名仍无证据，
需要 `actual10426` 页面同题抓包后才能登记。

### r11 真机结果（2026-09-25）

`--source all-designed --lookback 0` 首次真机跑通，**并成功写入 `epf_pmos_aux_records`**：
备用 170 行、阻塞断面 85 行、负荷/联络线各 96 行、检修与必开必停各 1 行；
输变电检修当天为空（EMPTY_VALID，与 HAR20 一致）。仅 `dcst_forecast_unit_overhaul`、
`dcst_tmp_update_time` 两个接口返回 504（nginx 网关超时）。

单点首跑若遇到 `HTTP 401 登录信息失效`，是复用了陈旧的浏览器会话，重跑一次即可
（force-new 认证恢复会接管），不是程序缺陷。

### r11b 收尾（2026-09-25）

- 14 个作废的 `qctc_pm_trade_inside` 来源降级为 `UNVERIFIED`：`all-designed` 不再为它们发请求，
  每轮省掉 14 次注定 503 的调用。
- `--unitid` 未传时自动回落到配置的 `unit_id`，解锁三个机组约束类来源。
- 504 退避重试最多 2 次；重试耗尽仍如实报 FAILED。
- 修 `unit_month_limit`：新增 AUX 专属的严格导航等待（轮询 `location.href` 直到命中目标页），
  解决 r8 成功、r10 起失败的 CSRF 丢失回归。

## 2026-09-25/26 r11c / r11c-v2 / r11d —— 登录恢复定稿（标记 STABLE）

- **r11c**：`force_new` 恢复改回 96 写法——只 `replace(browser_reuse=False)`、**不换 profile**。
  此前错误地新建临时 profile `pmos_auto_profile_aux_recovery_*`，导致 UKey 原生 PIN 弹窗不出现、
  状态机卡死、浏览器自行退出、CDP 断连。修复后复用正常 `pmos_auto_profile`。
- **r11c-v2**：补全打包依赖。首版 r11c 的隔离 venv 漏装 `requests`/`websocket-client`/`pymysql`，
  公司机启动即 `ModuleNotFoundError: No module named 'requests'`。
- **r11d（历史稳定回滚点；当时发布）**：`_auth()` 加「CDP 导航竞态容忍重试」。打开 `?service=...#/dashboard` 会触发顶层
  重定向，重定向瞬间 evaluate DOM 会撞 `Inspected target navigated or closed`，共享状态机因文本含
  `"cdp"` 判**致命、不重试**。修复：第 1 次按 force_new 启动；失败后 sleep 6s 等导航完成，第 2/3 次以
  `browser_reuse=True` 挂回**同一浏览器**重试。**共享模块与 96 源码零改动。**
- **公司真机验证 PASS（2026-09-26 00:14~00:20，run `20260925T161421Z-0f215f3b`）**：auth PASS
  （reuse 9222 → 首源 401 → force_new Edge 9224 成功）；**15 源 upsert 全 PASS，写库 595 行**；
  33 源 → complete 11 / EMPTY_VALID 1 / PARTIAL 2 / FAILED_SOURCE 1 / SKIPPED 18。
  残留：`net_contract_day`/`generation_contract_limit` HTTP 0 未取到；`unit_month_limit` legacy
  页面不可用；18 源按设计跳过（14 UNVERIFIED + 4 缺 unitid）。
- **当前 EXE**：`crawl_disclosure_aux_v1.exe`，**43,009,396 bytes**，SHA256
  `FAB2665D3218A2D102BDE5D4DD311AE1FE0CE64B3A02F2D4F2453709F29C5878`，
  BUILD_VERSION `2026-09-25-disclosure-aux-v1-r11d`。

> **状态：STABLE。** 公司真机跑通并成功写库。除甲方新增明确需求外，**尽量不再改动**；
> 后续任何修改都必须重新真机验证。

### 打包红线（AUX，务必遵守）

用隔离 venv 打包 AUX 前，必须装全运行时依赖：

```
pip install numpy pandas openpyxl Pillow pyinstaller requests websocket-client pymysql
```

`gmssl`/`pycryptodome` 仅 96 的 `auth/auth.py` 使用，AUX 走 `auto_crawler` 不 import，可不装。
漏装后三个会让 EXE **启动即 `ModuleNotFoundError`**。验证法：本机跑一次 EXE，若首错**不是**
`ModuleNotFoundError`，即依赖已打齐（通常会停在 config/浏览器错误，属预期的下一阶段）。

### 历史回补命令（2022 → 今）

```bat
cd /d "D:\爬虫电网\辅助信息披露爬虫"
crawl_disclosure_aux_v1.exe --date 2026-09-24 --lookback 1727 --source all-designed
```

1727 天 = 2022-01-01 ~ 2026-09-24。**必须用 `--source all-designed`**（monthly 源按月去重）；
**不要用 `--source all`** 做多年 daily lookback（`unit_month_limit` 会每天重复请求）。建议按年拆批
（如 `--date 2022-12-31 --lookback 364`）；长跑期间与 96 主爬虫错开（共享 `../output_96/.crawler.lock`）。
