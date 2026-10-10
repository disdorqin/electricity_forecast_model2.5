# scripts/crawler — 当前 PMOS 96 点链路

> [!CAUTION]
> **冻结区：未经项目负责人明确指示，任何 Agent / Codex / 自动化任务都不得修改 `scripts/crawler/` 下的源码、配置模板、构建脚本或同步逻辑。**
> 默认只允许读取、核验和运行现有 crawler 链路；如需修改，必须先获得负责人针对 crawler 的明确授权。

> **当前 96 发布状态（2026-10-10 核验）：** `dist/crawler/crawl_96_auto_v10.exe` 自报 `2026-09-28-runtime-resilience-v10-r11`，44,363,878 bytes，SHA256 `F71A272F703BED360DFFCB06007B77A25874A789ECD8F738D6489C9B8F957BB9`。V10 仍只做运行时稳定性/防御，不改变字段映射、数据口径、模型或 24 点链路。r11 在 r1-r4 基础上已接入应用层 `apps/crawl_96.py` + `resilience/`，包含死页面拒绝复用、孤儿浏览器/profile 治理、日志轮转等；真实部署目录有 2026-09-30 单日 PASS 证据。tracked canonical spec 为 `scripts/crawler/build/crawl_96_auto_v10.spec`。

## 24 点部署复盘平台爬虫（当前有效的独立入口）

`platform_review.py` 与 `platform_review_update.py` 是本项目用于抓取
`http://47.114.107.96/prediction-review` 的 **24 点电价预测复盘平台** 的专用爬虫，
不是 PMOS/国网 96 点爬虫，也不读取或写入 `epf_pmos_96_full` 数据库。

> **[2026-09-27 迁移]** 这两个文件已从 `scripts/crawler/` 根层迁入**应用层**
> `scripts/crawler/apps/tools/`，与原 PMOS 爬虫入口分置，避免与 96/AUX 链路混淆。
> 迁移时已同步修正 `platform_review_update.py` 的仓库根推导
> （`parents[2]` → `parents[4]`，因层级加深两级）与 import 路径。

今后应从以下入口运行：

```text
scripts/crawler/apps/tools/platform_review.py
scripts/crawler/apps/tools/platform_review_update.py
```

更新命令：

```text
python scripts/crawler/apps/tools/platform_review_update.py --start YYYY-MM-DD --end YYYY-MM-DD
```

该入口会登录平台、读取启用的交付模型字典、一次性导出复盘 Excel，再拆出详细数据和统计报告，
按 `time` 合并到 `outputs/platform_review/`；新抓取区间覆盖旧行时以新数据为准，区间外旧数据保留。
当前平台交付模型包括 1.0 模型和 2.0 模型，2.0 模型代码为 `epf_2.0_fusion_model`。

实际值边界：平台导出的 `日前电价`、`实时电价` 必须与
`data/24/canonical/shandong_pmos_hourly.csv` 对照；已验证现有重叠数据逐点一致，
因此平台爬虫的主要用途是补齐 2.0 模型预测值。实际值不一致时必须停止并报告，不能静默覆盖 canonical 数据。
该平台数据只用于 24 点复盘、指标计算和对照，不得接入 formal96 生产链路或替代 PMOS 数据源。

## 唯一生产入口

公司电脑只运行当前发布：

```text
dist/crawler/crawl_96_auto_v10.exe
```

V10 启动即持有 `<exe目录>/output_96/.crawler.lock`。并发启动只记录
`RUN_ALREADY_ACTIVE`、打印“已有爬虫实例运行”并返回非零，不创建 RunReport、浏览器或业务
raw/CSV/DB。已登录页面 Cookie 连续失效约 8--12 秒时，状态机只在当前 CDP 会话导航
`login_url` 恢复一次；第二次失效显式失败。只有 DevTools/CDP/WebSocket/target 真正丢失
才会记录 `BROWSER_CONTROL_LOST` 并允许新浏览器恢复。

它按以下顺序执行：

1. 启动浏览器认证状态机并读取最新 Cookie；
2. 默认通过新版 QCTC 接口获取日前预测、实时实际和机组级日前/实时结果（旧接口模式才使用
   `DaJyxxPlDa` / `DaJyxxPlYx` 命名）；
3. 保存原始响应和审计结果；
4. 保存本地运行副本：打包 EXE 固定写 `<exe目录>/output_96/pmos_96_全量.csv`；源码调试固定写 `<项目根>/outputs/crawl/runtime_96/pmos_96_全量.csv`。缺失字段保留为空并记录覆盖情况；
5. 仅将同一份数据幂等写入 `epf_pmos_96_full`。

日前预测和实时实际不会互相回填。来源不完整时保留已获取值、空值和覆盖报告，
不伪造缺失数据。

### QCTC 认证说明

QCTC 是两层认证结构：

1. `auth` 阶段负责旧 PMOS 门户的账号密码/UKey 登录和 Cookie 获取；
   启动地址使用带 `service` 交易回跳参数的统一认证入口，不能只直接打开
   `/#/dashboard`，否则部分环境会在 `#/login`/`#/outNet` 停留；
2. `collect` 阶段需要同一个浏览器标签页处在 `:18080` 的同源页面，才能用
   `fetch`（`credentials:'include'`）调用 QCTC 接口；SPA 建立自己
   `Authorization: Bearer` 时，`_browser_fetch` 的 JS 会自动带上它。

进入 QCTC 前，程序先在已登录的门户标签页导航到：
`/psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin`。
随后轮询 CDP `/json`，兼容原标签页、新标签页和新窗口，识别
`pmos.sd.sgcc.com.cn:18080/qctc/` 或 `/qctc-trade/` target 后再进入
`forecast10424`；直达超时才保留现有门户 DOM 菜单兜底，不直接把业务页当作 SSO 入口。

> ## ⚠ 2026-09-16 修正：Bearer 是观测项，不是放行条件
>
> 此前把「拿不到 Bearer」当成硬失败（`ensure_qctc_context()` 直接抛错），
> 导致 09-15、09-16 两轮补数**一次数据请求都没发出去**就中止。但这条因果链
> 从未被验证：
>
> - 2026-09-13 的日志里 QCTC 接口**成功返回过 `status=200` 并写入 96 行**
>   （当时是否携带 Bearer 没有记录）；
> - 全量日志中 `bearer_present: True` 出现 **0 次**；
> - 502 来自旧 `:18080/trade/*.do` 接口，`status=0` 来自页面被弹回门户后的
>   跨源 fetch —— 两者各有独立成因。
>
> 现行做法：`ensure_qctc_context()` 超时只记 `QCTC_CONTEXT_SOFT_MISSING`
> （event=`WARN`、stage=`PARTIAL`）并返回 `False`，采集继续；`fetch_csrf_token()` 在 QCTC 模式下
> 恒返回 `True`（新接口不使用 `_csrf`）。真实可用性由数据接口返回码判定。
>
> 同时 `_browser_fetch()` 增加两条行为：
> 1. 标签页已在接口同源时**不再导航**，直接 fetch（避免把可用文档换成会被
>    弹回门户的 SPA 路由）；
> 2. fetch 返回 `status=0` 或 `502` 时，换同源兜底页
>    （`:18080/zcq/main/index.do` → `:18080/favicon.ico`）重试一次。
>    这是**待验证的兜底假设**，不改变成功路径。

若旧门户自动跳回 `/dashboard`，程序会保留浏览器窗口等待
`qctc_auth_wait_sec`（默认 90 秒），并先尝试点击门户中的 QCTC/信息披露/现货菜单，
让门户自己生成 SSO 跳转；找不到菜单时才提示人工点击。若等待期间 CDP 端口中断，
主循环会记录 `BROWSER_RECOVERY_START`，放弃故障浏览器并启动新浏览器重新认证，
而不是继续使用已失效的 9222 端口。QCTC 模式同时跳过旧 `/trade` 附加接口，
避免 502 噪声干扰核心诊断。

### UKey PIN

`pin_handler=windows` 只有在 `resolved_pin` 非空时才启用；PIN 来源优先级为
环境变量 `PMOS_UKEY_PIN` → `config.json:ukey_pin`。两者都空时
`build_pin_handler()` 会降级为 `ManualPinHandler`，并打印
`pin.handler_fallback=manual reason=pin_not_configured`
（2026-09-16 现场就是这个原因导致每次登录都要人工输 PIN）。
`WindowsPinHandler` 只向标题匹配的 UKey 弹窗的可见输入框写 PIN，写入后回读校验，
找不到确定按钮时退回 Enter，弹窗被拒最多重试 3 次。

审阅部署 EXE 时优先查看 `<exe目录>/output_96/crawler.log` 和 `report.json` 的 `events`；源码调试对应位置为 `outputs/crawl/runtime_96/`：
`AUTH`、`COOKIE_FETCH`、`QCTC_CONTEXT_*`、`COLLECT`、`LOCAL_SAVE`、`DB_SYNC`
分别对应认证、Cookie、QCTC 上下文、采集、本地保存和数据库同步。
其中 `QCTC_PORTAL_MENU_CLICK` 表示程序已自动点到门户菜单（附 SSO 跳转），
`QCTC_PORTAL_MENU_NOT_FOUND` 表示没找到可点菜单、转为等待人工
（门户是 iframe 框架页，顶层 `document` 搜不到菜单，属已知局限）；
这两个码是 2026-09-15 12:29 才加进 `crawl.py:_try_portal_qctc_entry` 的。
`QCTC_CONTEXT_SOFT_MISSING` 是 2026-09-16 新增的软告警码。

## 当前源码运行时

下列为当前源码中的职责文件；标记为“否”的辅助工具不属于主 EXE 运行时依赖：

| 文件 | 用途 |
|---|---|
| `collect/crawl_96_local.py` | 96 点任务编排、认证、审计和输出 |
| `collect/crawl.py` | PMOS HTTP 数据接口核心 |
| `collect/crawl_da_only.py` | 日前最终版手工补数辅助程序，不属于主 EXE 链路 |
| `sync_db/run_crawler.py` | `epf_pmos_96_full` 建表和上传支持 |
| `auth/browser_session.py` | 浏览器 CDP Cookie 获取 |
| `auth/auth_runtime.py` / `auth/auth.py` | 认证模式兼容和账号密码兜底 |
| `auth/cfca_runtime.py` | CFCA/UKey 运行时辅助 |
| `collect/crawl_96_local.spec` | 主 EXE 的 PyInstaller 构建配置 |
| `config.example.json` / `db_config.example.json` | 当前配置模板 |

### 当前生产依赖关系

```text
crawl_96_auto_v10.exe
└─ crawl_96_local.py                 主编排
   ├─ scripts/crawler/auth/            自动登录模块
   │  ├─ auth_runtime.py               认证路由
   │  ├─ browser_session.py             CDP Cookie
   │  ├─ auth.py                         账号密码国密登录
   │  ├─ cfca_runtime.py                UKey/CFCA辅助
   │  └─ auto_crawler/                  浏览器认证状态机
   │     ├─ config.py                   认证配置模型与默认值
   │     ├─ state_machine.py            登录/滑块/UKey/CDP状态流转
   │     ├─ browser.py                  Chrome/Edge CDP、页面和Cookie操作
   │     ├─ page.py                     PMOS页面状态识别
   │     └─ handlers.py                 滑块、账号、PIN等交互处理器
   ├─ scripts/crawler/collect/         接口、字段和96点采集
   │  ├─ crawl.py                       PMOS接口请求和响应解析
   │  └─ crawl_96_local.py              主编排
   └─ scripts/crawler/sync_db/         仅创建并同步 canonical 表 epf_pmos_96_full
      └─ run_crawler.py
```

| 文件/目录 | 当前是否运行时依赖 | 功能说明 |
|---|---:|---|
| `collect/crawl_96_local.py` | 是 | 任务入口；确定日期、认证、四类接口采集、覆盖审计、本地落盘和上传队列 |
| `runtime_lock.py` | 是 | `output_96/.crawler.lock` OS 级单实例锁；异常退出由操作系统释放 |
| `auth/auto_crawler/config.py` | 是 | 浏览器认证配置数据结构、配置校验和默认参数 |
| `auth/auto_crawler/state_machine.py` | 是 | 浏览器登录状态机；负责打开浏览器、检测页面状态、等待登录、读取 Cookie |
| `auth/auto_crawler/browser.py` | 是 | Chrome DevTools Protocol 连接、标签页、Cookie 和浏览器进程管理 |
| `auth/auto_crawler/page.py` | 是 | 识别 PMOS 登录页、滑块页、交易主页等页面状态 |
| `auth/auto_crawler/handlers.py` | 是 | 账号密码、滑块模板、人工滑块、UKey/PIN 等交互处理 |
| `collect/crawl.py` | 是 | PMOS HTTP 请求、日期切换、CSRF、日前/实时/价格接口和字段解析 |
| `sync_db/run_crawler.py` | 是 | 创建并同步唯一目标 `epf_pmos_96_full` |
| `auth/auth_runtime.py` | 是 | 根据 `auth_mode` 选择浏览器、静态 Cookie 或账号密码兼容路径 |
| `auth/browser_session.py` | 是 | 浏览器认证失败时的 CDP Cookie 获取/校验兜底 |
| `auth/auth.py` | 条件依赖 | 仅 `auth_mode=account` 时使用的国密账号密码登录实现 |
| `auth/cfca_runtime.py` | 条件依赖 | 配置启用 CFCA/UKey 时探测本机服务和辅助 PIN 窗口操作 |
| `dist/crawler/config.json` | 是 | 公司电脑的 PMOS、浏览器、认证和爬取配置；Cookie 只保存在本机 |
| `dist/crawler/db_config.json` | 是（开启 DB 上传时） | 远程 MySQL 连接配置；不应提交或复制到不可信位置 |
| `dist/crawler/运行爬虫.cmd` / `自动运行96点.cmd` | 启动辅助 | 分别提供交互运行和定时增量运行，不包含爬取逻辑 |
| `dist/har_captures/crawler_main_2026_09/pmos.sd.sgcc.com.cn6.har` | 否 | 接口诊断参考，不参与生产运行（2026-10-08 起 HAR 统一归位 `dist/har_captures/`） |

`dist/crawler/crawl_96_auto_v10.exe` 是单文件打包程序，因此公司电脑不需要安装
Python，也不需要携带 `scripts/` 源码。公司电脑真正需要维护的外部文件只有
`config.json`、`db_config.json` 和浏览器/网络环境；部署 EXE 的 `output_96/` 是程序运行后生成的数据、raw 包和失败上传队列。该路径固定相对 EXE 目录，不从 `config.json` 读取；源码调试则使用项目内 `outputs/crawl/runtime_96/`，避免污染仓库根目录。

## 数据库交付表

`epf_pmos_96_full` 与 `data/96/authoritative/pmos_96_全量.csv` 保持同构。v7 在原 canonical 字段上新增 QCTC 信息披露扩展，但仍不增加模型派生特征：

```text
ForecastData: 原8个预测字段 + 全网负荷预测
RealityData: 原9个正式实际字段 + 全网负荷实际
RealityTmpData: 10个“临时实际”独立字段
ForecastBoundaryData: 6个“边界*预测”独立字段
日前/实时/备用: 保持原字段
```

RealityTmpData 和 ForecastBoundaryData 只作为独立数据资产进入 authoritative；是否进入正式模型由 snapshot/as-of 与模型特征契约决定，不能直接覆盖正式 actual 或 ForecastData。

数据库还保留 `id`、`unit_id`、`source_captured_at`、`create_time`、`update_time` 五个技术/审计元数据列。唯一键为 `market_date + 时段 + unit_id`。

数据库初始化/迁移由当前运行时使用
`sync_db/sql/002_create_epf_pmos_96_full.sql`。如果发现旧英文宽表，
当前程序会先将其改名为 `epf_pmos_96_full_legacy_时间戳`，不会把旧数据冒充新表。

## 日前/实时价格字段链路（排查重点）

日前、实时价格一直属于链路字段，不能从市场预测源表推断：

```text
日前价格：QCTC `trade/DaJyjgfbPlantQuery/getDetail96`
  -> crawl.py 解析为 da.cqPrice
  -> epf_pmos_96_full.日前出清价格
  -> 模型输入兼容字段：日前电价

实时价格：QCTC `YxJyjgfbPlantQuery/getDetail96`
  （正式结果为空时查询 `YxJyjgfbPlantQueryTmp/getDetail96`）
  -> crawl.py 解析为 rt.cqPrice
  -> epf_pmos_96_full.实时出清价格
  -> 模型输入兼容字段：实时电价
```

`epf_pmos_96_full` 是当前唯一同步目标，市场特征和机组价格在本地 raw 中分别采集后
合并写入该表。历史源表和旧英文宽表不属于当前同步链路。

### HAR 文件审计结论

全部 PMOS 抓包 HAR 自 2026-10-08 起统一归位于 `dist/har_captures/`（清单、接口语义与
目标字段覆盖审计见 `dist/har_captures/README.md`）。
`dist/har_captures/crawler_main_2026_09/pmos.sd.sgcc.com.cn6.har` 不是生产输入，只是一次浏览器抓包参考，
抓包日期为 `2026-09-14`，且只涉及一个光伏机组。该文件不能作为完整价格数据集：

- 市场预测/实际接口的响应中没有 `cqPrice`；这是正常的，因为它们只提供市场特征。
- 日前机组明细 `getDetail96` 虽然包含 96 个 `cqPrice` 键，但本次抓包只有 40 个
  非空值，且是单个机组，不是全省 96 点价格。
- 实时正式明细、实时临时明细均为空；省级实时价格网格的 96 个价格也全部为空。
- 文件中的原始字段叫 `cqPrice`，不会直接出现中文字段 `日前电价`、`实时电价`。

因此，之前把这个 HAR 描述成“已经验证完整日前/实时价格链路”是不准确的；它只能
证明接口字段形状存在，不能证明价格数据已发布或已完整采集。当前程序运行时会对
日前和实时分别统计覆盖率；未完整时仍保存已获取数据，并在对应 runtime 的 `report.json`
中标记 PARTIAL，不跨来源补值。

## 公司电脑部署

`dist/crawler/` 根目录当前唯一 96 入口为 `crawl_96_auto_v10.exe`（V10-r11）。旧 V9/v3 顶层路径当前不存在；需要回滚时只能按版本台账中的 SHA256 从 `archive/exe_versions_*` 选择候选并复制到发布路径。配置文件和总 README 与当前 EXE 同目录；
日前补数程序位于 `dist/crawler/tools/日前补数/`，旧程序和诊断材料位于
`dist/crawler/archive/`，不得从归档目录作为生产入口运行。

建议首次运行：

```text
crawl_96_auto_v10.exe --auth-only
crawl_96_auto_v10.exe --date YYYY-MM-DD --force
```

日常增量运行：

```text
crawl_96_auto_v10.exe --lookback 1
```

部署说明见 `dist/crawler/README.md`；本文件只维护源码架构和运行时依赖说明。

## AUX 辅助信息披露链路（独立入口）

当前 AUX 发布为 r14（`2026-10-08-disclosure-aux-v1-r14`，SHA256 `7FAD8476…8388`），r14 公司真机仍为 NOT_TESTED；r13 是 PARTIAL_LIVE_PASS，最后完整真机稳定回滚点是 r11d。详细状态以 `dist/crawler/辅助信息披露爬虫/AUX-V1_版本记录.md` 为准。

AUX 与 96 点链路共享认证/浏览器底座，但业务、配置和写库表完全独立
（表 `epf_pmos_aux_records`，不触碰 `epf_pmos_96_full`）。源码入口：

```text
scripts/crawler/apps/crawl_aux.py                      应用层入口（EXE 入口）
scripts/crawler/collect/crawl_disclosure_aux.py        采集实现
scripts/crawler/collect/crawl_disclosure_aux_explore.py  探索模式（只录制）
```

`--explore` **[AUX-V1-r12]** 是录制用的旁路：登录成功后由人工在正常页面点击，
程序通过 CDP 被动记录业务请求、参数值样例与响应字段路径，按
（方法 + 路径 + 参数名集合）聚合成接口契约，输出一份中文《发现清单.md》并连同
原始响应打包为单个 zip。凭证（Cookie / Authorization / CSRF / token / 口令）在落盘前
脱敏，zip 内不含日志与配置文件。它**不解析、不入库、不调用任何采集接口**，
用于替代「F12 手工导出 HAR 再搬运」的证据采集方式。完整部署与命令见
`dist/crawler/辅助信息披露爬虫/README.md`。

## 历史程序

除上文明确恢复的 24 点部署复盘平台入口外，旧 PMOS 自动爬取、旧补数、旧认证自检和一次性迁移工具均已移至
`scripts/crawler/archive/`；历史数据库回填程序位于 `scripts/sync/archive/legacy/`。
归档内容只用于追溯，不是当前入口。
