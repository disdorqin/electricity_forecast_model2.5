# PMOS 辅助信息披露爬虫 AUX-V1：最终架构与实施设计

状态：implemented; r10 all-designed batch mode local contract PASS; company live validation pending
日期：2026-09-24
目标：新增一套独立辅助信息披露爬虫，复用主爬虫认证能力，但与主96点业务、输出和数据库写入完全隔离。

## 1. 最终架构决定

采用“两套业务爬虫 + 一套共享认证底座”。

共享：
- scripts/crawler/auth/auto_crawler/*
- AuthenticationStateMachine
- AuthConfig
- 浏览器/CDP/Chrome→Edge/UKey/CFCA 能力
- 现有 PmosCrawler 的通用浏览器 fetch / HTTP transport 能力

隔离：
- 程序入口
- 辅助业务接口和 parser
- raw/output/report
- 数据库 writer
- 数据库表
- 版本线

主爬虫继续：
- scripts/crawler/collect/crawl_96_local.py
- scripts/crawler/collect/crawl.py
- scripts/crawler/sync_db/run_crawler.py
- epf_pmos_96_full
- crawl_96_auto_v10.exe

辅助爬虫新增：
- scripts/crawler/collect/disclosure_aux.py
- scripts/crawler/collect/crawl_disclosure_aux.py
- scripts/crawler/sync_db/disclosure_aux.py
- scripts/crawler/sync_db/sql/003_create_epf_pmos_aux.sql
- scripts/tests/test_disclosure_aux.py
- dist/crawler/辅助信息披露爬虫/crawl_disclosure_aux_v1.exe

原则：AUX-V1 不修改主96点接口映射、主表 schema、主入口和主 DB writer。

## 2. 认证和浏览器复用边界

辅助入口不得 import crawl_96_local.py 的私有 _ensure_browser_state_machine。

正确做法：

1. 从辅助配置读取 auth_config_path，默认 ../config.json。
2. AuthConfig.from_file(auth_config_path)。
3. AuthenticationStateMachine(auth_cfg, reporter=aux_reporter).run()。
4. 只在内存中使用 result.cookie 和 result.debug_port。
5. AUX-V1 默认不写回主 config.json，避免辅助任务改变主程序配置。

辅助业务 crawler 建议：

class PmosDisclosureAuxCrawler(PmosCrawler)

目的仅为复用当前已经验证过的：
- requests session/cookie
- _browser_req()
- ensure_qctc_context()
- 浏览器 fetch / 同源导航能力

禁止把 AUX endpoint 继续添加进 scripts/crawler/collect/crawl.py。所有辅助接口必须留在 disclosure_aux.py。

这是 AUX-V1 的最小侵入方案。未来只有在主/辅两边都稳定后，才考虑提取公共 transport class；AUX-V1 不做此重构。

## 3. 并发规则

主爬虫与辅助爬虫不得同时运行，因为二者可能共用：
- PMOS登录态
- CDP端口
- UKey
- 浏览器 profile

辅助程序默认复用主爬虫同一 OS lock：

../output_96/.crawler.lock

通过 scripts/crawler/runtime_lock.py 的 RuntimeLock 获取。

这样：
- 主程序运行时，辅助程序直接退出并提示“主/辅助爬虫已有实例运行”；
- 辅助程序运行时，主程序也会被同一锁挡住；
- 不需要修改主程序。

辅助自身 output_aux 不用另建第二个互斥锁。

## 4. 辅助运行目录

部署目录：

dist/crawler/辅助信息披露爬虫/

建议最终内容：

- crawl_disclosure_aux_v1.exe
- config_disclosure_aux.json
- config_disclosure_aux.example.json
- README.md
- 辅助信息披露数据库表设计.sql
- 辅助信息披露数据库字段说明.md
- AUX-V1_版本记录.md
- output_aux/ 运行时自动创建

配置不保存主账号/DB密码，默认通过相对路径复用父目录已有配置：

auth_config_path = ../config.json
db_config_path = ../db_config.json
shared_lock_path = ../output_96/.crawler.lock

因此公司电脑部署时，把整个“辅助信息披露爬虫”文件夹放到主 crawler 目录旁边即可。

## 5. AUX-V1 CLI

必须支持：

--date YYYY-MM-DD
--lookback N
--source unit|constraint|event|curve|stat|contract|all
--capture-only
--dry-run
--auth-only
--db-check
--no-db-upload

默认：
- source=all
- lookback=1
- raw 永远保存
- DB 打开时只写 epf_pmos_aux_* 表

## 6. Source Registry 设计

不要在 main() 里堆大量 if/else。建立 SOURCE_REGISTRY，每个 source 至少包含：

- name
- group
- method
- api_url / path
- page_url
- resolution
- parser
- target_table
- enabled_by_default

所有请求先进入统一 capture_source()：
1. 发真实请求；
2. 保存 raw；
3. 计算 raw_hash；
4. 标记 COMPLETE / EMPTY_VALID / PARTIAL / FAILED_SOURCE；
5. parser 仅在响应结构符合契约时执行；
6. structured row 计算 record_key；
7. 可选写 DB。

## 7. AUX-V1 数据来源与目标表

### 7.1 unit

接口：
- /qctc_pm_trade_inside/DaUnitParamQuery/getDataList
- /qctc_pm_trade_inside/DaUnitParamQuery/getTypeList
- /qctc_pm_trade_inside/DaUnitParamQuery/getGengroupList

目标：
epf_pmos_aux_records

火电筛选必须依赖平台 fuel_type/type/jzlx/正式查询参数，禁止名称模糊判断。

### 7.2 constraint

接口：
- DaJysbPlant/getUnitConstraint
- DaJysbPlantCommon/initComponent
- JJCQDaJysbPlantCommon/getUnitConstraint

目标：
epf_pmos_aux_records

energyDeclarationModule/startDailyQuote 等“申报/合约”部分写 contract，不混进 constraint。

### 7.3 event

接口：
- getTsjzTableAndText
- getSbdjxTableAndText
- getDwbyTableAndText
- unitMaintenancePlanUpdate/*
- unitMaintenancePlanOutQuery/*
- UnitMaintenanceSchedule/getTableData

目标：
epf_pmos_aux_records

event_type 至少：
- maintenance_plan
- commissioning_plan
- special_unit_tag
- transmission_maintenance
- reserve_security

### 7.4 curve

接口：
- getRunLine
- getDegLine
- getMaxMinZdSc（字段未确认时先 raw-only，确认后再写结构化）
- 其它明确的时间点曲线

目标：
epf_pmos_aux_records

禁止把日级事件广播成96点。

getFhChar 与主96点 ForecastData 同义部分只做 raw 审计，不重复写第二份生产事实；getZdLlx 语义未确认前 raw-only。

### 7.5 stat

接口：
- /qctc_pm_trade_inside/trade/marketDetailsQuery/zcqEnergyPriceQuery/getTableData

目标：
epf_pmos_aux_records

火电必须使用平台口径参数：
ztType=1
tjInfo=1
tjType=t
rqType=1/2/3

### 7.6 contract

接口：
- /qctc_pm_zcq_base/JyjgByDayQuery/getTableDate
- /jysbys/fdczxsbedcx.do?method=getTableRows
- /jysbys/fdczxsbedcx.do?method=getvlddetail
- /fdaxsjhyxc.do?method=getTableDate
- 月度机组合约上限
- energyDeclarationModule/startDailyQuote 相关申报

目标：
epf_pmos_aux_records

jzdlzb 只保留原字段名与 ratio_status=UNDEFINED；甲方确认分子分母前禁止重命名为“火电合约占比”。

## 8. Raw-first 原则

每一次 HTTP/浏览器响应都必须先形成 raw record。

本地：
output_aux/raw/YYYY-MM-DD/<source_name>_<timestamp>_<hash8>.json

数据库：
epf_pmos_aux_records

raw 至少记录：
- run_id
- source_group
- source_page
- source_api
- request params（不得包含 token/cookie/password）
- business_date
- resolution
- http_status
- business_code
- source_status
- source_row_count
- captured_at
- raw_hash
- raw_json
- schema_version

`captured_at` 在 MySQL 中使用无时区 `DATETIME`；AUX writer 写 SQL 时将 aware ISO-8601 timestamp 转为 naive datetime（保留本地墙钟值），offset-bearing 原字符串仍保留在结构化 JSON / 本地 raw。

结构化 parser 失败时：
- raw 仍保留；
- source_status=PARTIAL 或 FAILED_SOURCE；
- 不把猜测字段写进结构化表。

## 9. 状态语义

COMPLETE：
响应成功，结构符合契约，关键字段可解析。

EMPTY_VALID：
HTTP/业务成功，平台明确返回合法空结果。

PARTIAL：
响应成功但字段/部分记录不足，raw 保留，能确认的字段可写结构化表。

FAILED_SOURCE：
HTTP失败、业务code失败、JSON结构错误、parser异常。

401/403：
视为认证问题。记录后终止本轮 AUX，禁止继续几十个 source 重复打失败请求。

单个业务 source 的 404/5xx/字段变化：
记录 FAILED_SOURCE，其他 source 可继续。

## 10. 幂等策略

所有结构化表由应用计算 record_key = SHA256(稳定业务身份字段)。

重复运行：
INSERT ... ON DUPLICATE KEY UPDATE

不以 raw_hash 作为结构化唯一身份，因为同一业务对象的值可能后续修订。

raw 表：
request_key + raw_hash 唯一。
相同请求、相同原始响应不重复保存；内容变化会形成新的 raw snapshot。

## 11. 数据库表

AUX-V1 使用 1 张统一记录表：

1. `epf_pmos_aux_records`

raw 与 structured 数据共用这张表：raw 原文放 `raw_json`，每条解析记录的业务字段放 `record_json`；来源字段名和原值都保留。公共审计列支持按业务日期、source、类型、状态和 run_id 查询。新增来源字段无需逐列 ALTER。

主表 epf_pmos_96_full 禁止 ALTER。

## 12. 源码版本标记

辅助爬虫独立版本线：

AUX-V1
AUX-V2
...

不要和主爬虫 V10 混用。

所有新增代码的具体位置必须标：

# [AUX-V1] ...

例如：
# [AUX-V1] shared auth entry
# [AUX-V1] raw-first source capture
# [AUX-V1] maintenance parser
# [AUX-V1] aux DB upsert

遵守“删除警惕、新增权衡”：
- AUX-V1 原则上不删除主爬虫任何代码；
- 不为了抽象漂亮去重构 crawl.py；
- 新文件优先；
- 若确实修改共享 auth，只能是已经独立证明的通用 bug，并必须跑主爬虫全部回归。

## 13. 验收

必须做到：

- 主爬虫源码业务逻辑不变；
- epf_pmos_96_full schema/写入逻辑不变；
- AUX 使用现有 AuthenticationStateMachine；
- AUX 和主程序不能并发；
- raw 永远先保存；
- 1张 `epf_pmos_aux_records` 统一表可重复建表；
- 相同记录重复运行不重复插入；
- 合法空与失败可区分；
- 事件不广播成96点；
- 未确认语义字段 raw-only；
- 所有 AUX 新增代码有 [AUX-V1] 具体注释；
- 主 crawler 全部旧测试继续通过；
- AUX fixture/parser/db tests 通过；
- 公司 live smoke 前状态只能写 NOT_TESTED。

## 14. 公司首次写库验收与故障复盘（2026-09-23～24）

本节是 AUX 首次公司实网部署的已验证运维事实，后续排错先按这里核对，不要重复猜测 HAR 或重写认证链。公司配置、账号密码不入文档。

### 14.1 首次 live smoke 证据与结论

- 当前候选为 `2026-09-23-disclosure-aux-v1-r9`。公司单日任务 `2026-09-23 --source unit_info` 的 run_id 为 `20260924T021452Z-81a83fe4`，report 总体 `PASS`，source `COMPLETE`、rows=1。
- 日志先记录 Python HTTP transport connect timeout，随后 AUX browser fallback 对同一请求成功返回 HTTP 200；此处 WARNING 表示直连失败，不等于整个数据源失败。最终以 browser response、source stage 和 report 结果判定。
- MySQL `epf_pmos_aux_records` 中该 run 有 2 行：`record_kind=raw` 一行、`record_kind=structured` 一行；`captured_at` 正常。当前已证明的是一个目标日期、`unit_info` 一行的真实写库；**不是** 2022 至今历史回补，也不是全机组 fan-out 或 `unit_month_limit` 验收。

### 14.2 本轮故障与根因

1. **`JSONDecodeError: line 7 column 16`**：运行目录实际被加载的 `config_disclosure_aux.json` 不是合法 JSON。常见原因是把 JSON 布尔值写成 Python 的 `True`；JSON 必须写小写 `true`。此外 Windows 隐藏扩展名会让 `config_disclosure_aux.json.json` 被误认为目标文件。应查看 EXE 同目录的确切文件名和内容；命令行 `args.config=null` 表示走 EXE 默认 config，不代表配置内容正确。
2. **数据库表零行但 raw 已落盘**：当时有效配置明确 `db_upload=false`。CLI `no_db_upload=false` 只说明未从命令行关闭上传，不能覆盖配置中的 false；实际启用条件还要求 `db_upload=true`，且不是 `--capture-only` / `--dry-run` / `--no-db-upload`。项目开发目录 config 与公司 EXE 目录 config 不是同一个配置来源，不能用开发目录副本推断公司运行行为。若要审计，report 应记录脱敏的 `db_upload_effective`、`db_config_path` 和写入计数；r9 仍未把这些有效配置/写入计数纳入 report，必须同时查 MySQL。
3. **冻结 EXE 找不到 DDL**：`FileNotFoundError: D:\\爬虫电网\\辅助信息披露数据库表设计.sql` 是打包目录缺少 AUX 专用 DDL。冻结模式从 EXE 同目录读取 `辅助信息披露数据库表设计.sql`；将该文件复制到 EXE 旁边，再运行 `--db-check`。该 DDL 只 `CREATE TABLE IF NOT EXISTS epf_pmos_aux_records`，不修改 `epf_pmos_96_full`。不要只凭旧 report 或本地 `--db-check --no-db-upload` 推断公司数据库已建表。
4. **`runtime lock is already held` / exit=2**：`output_96/.crawler.lock` 是操作系统文件锁。锁文件存在本身不表示被占用，也不是应删除的 stale sentinel；真正报 already-held 说明另一个进程持有该锁。`tasklist | findstr /i crawl` 无输出、`findstr` exit=1 只说明没有匹配该进程名，不证明没有持锁进程（可能是 Python、不同 EXE 名或主爬虫）。先用命令行/进程详情查持有者并让进程正常退出；不要删除锁文件。
5. **MySQL 1292 `Incorrect datetime value ... +08:00`**：平台 `captured_at` 是带时区 ISO-8601 字符串，而数据库列是无时区 `DATETIME`。AUX-V1-r9 只在 DB writer SQL 边界将其解析为 naive Python datetime，保留原本地墙钟值；结构化 JSON 和本地 raw 仍保存原始时间文本。无需改表结构或改 MySQL 模式。已用完整 raw+structured upsert contract test 覆盖。
6. **HTTP 15 秒 connect timeout 后成功**：日志中的 `AUX transport failed; one browser fallback` 后若 browser fetch 对相同源/日期返回 HTTP 200，且 report source 为 `COMPLETE`，则本条抓取成功；不能只看到第一条 WARNING 就判失败。反之必须以最终 report status、raw、DB 行共同判断。

### 14.3 从 2022 回补的正确入口与范围

- 截至 2026-09-24，日期区间 `2022-01-01` 至 `2026-09-24` 共 1728 个日历日；`--date 2026-09-24 --lookback 1727` 是 inclusive 区间。
- 当前 `--source all` 仅含默认启用的 `unit_info`（daily）与 `unit_month_limit`（monthly）。不要对 1728 个日期使用 `--source all`：月度接口会按每个日日期重复查询同一个 `dmonth/smonth`，会产生多余请求和按不同 request/date provenance 保存的重复历史行。
- daily 历史只跑 `unit_info`；monthly 合约每个自然月只跑一次 `unit_month_limit`，从 2022-01 到 2026-09 共 57 次。两种频率分开调度。
- 简单 daily 命令（当前需在部署配置确认 `db_upload=true` 后使用）：

  ```cmd
  crawl_disclosure_aux_v1.exe --date 2026-09-24 --lookback 1727 --source unit_info
  ```

- monthly source 需每月一次。PowerShell 可在 AUX EXE 目录执行：

  ```powershell
  for ($d = [datetime]'2022-01-01'; $d -le [datetime]'2026-09-01'; $d = $d.AddMonths(1)) {
      $monthDate = $d.ToString('yyyy-MM-dd')
      & .\crawl_disclosure_aux_v1.exe --date $monthDate --lookback 0 --source unit_month_limit
      if ($LASTEXITCODE -ne 0) { Write-Warning "unit_month_limit failed: $monthDate exit=$LASTEXITCODE" }
  }
  ```

- `unit_info` 实网 smoke 当前仅观察到每日 1 条结构化记录、同一机组实体；这是 endpoint 实际回包证据，不足以宣称全机组历史。需要全机组范围时，先以真实 `unit_master` 清单核验 unitid 数量和分页完整性，再另行批准 bounded fan-out；不能把 lookback 天数误当机组数量。
- 长区间不要只看控制台首行或退出码。每一批都核对最新 `aux_report.json`：run_id、build、日期范围、source status/rows/error；再按 run_id 查询 `epf_pmos_aux_records` 的 raw/structured 数量和 captured_at。中断批次以落盘 raw/report/DB 实际状态判断，不凭“程序跑了很久”判断成功。

### 14.4 r10 一键调度全部已登记来源

- 用户需要一个入口启动整套辅助来源，不应把原 `--source all` 改成隐式启用所有 source：r2 已建立并测试的 `all/group` allowlist 契约必须保持。
- 新增独立 `--source all-designed`。它遍历 source registry；`--source all`、group 与具体 source 的既有行为不变。
- `all-designed` 在每个业务日期调度日/快照/15min source；`resolution=monthly` 的 source 在 lookback 中每个自然月仅调度一次。运行日期按新到旧排列，因此月度请求使用该月份窗口内最新日期，builder 再按 source contract 转为月份参数。
- 每个 source 先核验 readiness：`UNVERIFIED`、缺少真实 `unitid`、参数 builder 未覆盖 `param_contract` 时生成 `SKIPPED_NOT_READY` 报告项，不发送 HTTP、不伪造 raw 响应、不写入业务表。其它 source 仍继续。
- raw-only source 只保存 raw 记录，不产生 structured row。故一键模式的目标是统一调度全部已登记 source，而非承诺所有 source 都能返回结构化数据；任何 skipped/failed/partial 均使本轮 status=`PARTIAL`、exit=2。
- **尚未获得 live evidence 的字段**：maintenance_plan/maintenance_init、unit_count_stat 保持 `UNVERIFIED` 并跳过；`fh_char_raw` 因 `type` 参数没有可验证 builder 而跳过；unitid-dependent source 在未传真实 ID 时跳过，不执行自动 fan-out。公司端 all-designed smoke 必须审查 report，不能将 `PARTIAL` 宣称全字段成功。
- 单日入口：`crawl_disclosure_aux_v1.exe --date YYYY-MM-DD --lookback 0 --source all-designed`。多年历史运行可带 `--lookback N`；月度 source 按月去重，仍需先单日 smoke 并确认公司 EXE/config/DDL/DB。r10 尚未通过公司全来源 smoke。

### 14.5 r10-diag1 阶段诊断日志

2026-09-24 公司报告出现一次 `auth.state=unknown` 后长时间无新日志，最后被用户中断；当时无法从静默区间区分认证轮询等待还是采集阻塞。`r10-diag1` 只增加 AUX 诊断，不改变认证选择、接口、重试、数据解析或 DB 语义：

- 启动记录 build/run_id、参数、配置/输出路径、DB 上传有效开关和 DB 配置是否存在；不记录 DB host/user/password。
- lock acquire、auth、QCTC context、DDL/schema、DB connect、date collect、DB upsert 记录阶段起止、耗时和异常类型；长阶段每20秒写一次 heartbeat。report 的 stage 会先落 `RUNNING`，正常结束改 `PASS`，异常改 `FAIL`；外部强杀时保留最后运行阶段。
- source 记录 selection、日期、resolution、evidence、API path、参数名（不记录参数值）、HTTP 状态、Content-Type、响应字节数、payload/解析行数、raw hash/path、耗时和错误摘要。仅 HTTP error 可记录最多300字符的脱敏正文预览；成功响应正文不进入普通日志。
- DB 记录 schema/connect/upsert 分阶段耗时及 raw/structured 行数；不打印连接凭据。OpenSSL/Windows cert store 错误可由 `db_connect` 阶段直接定位。
- 日志字段不得包含 Cookie、Authorization、Token、CSRF、密码或完整响应业务数据。若新日志发现上述敏感值，停止传播并按凭据泄露处理。
- 诊断能定位“卡在认证、QCTC、DB、具体日期/source/传输/解析/写库”的哪一层，但不会自动修复平台 HTTP 401/503。每个 run 仍以 report 最终状态和数据库核验为准。
