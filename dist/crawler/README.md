# PMOS 96 点爬虫部署包

> **版本规则（2026-10-10 核验）：** 修改、打包或部署爬虫前先读 `dist/crawler/爬虫版本台账与发布治理.md`。版本号代表一次完整功能迭代；同一目标下的小修只增加 rN 构建修订，不机械升级 V11。当前仍为 V10，正式文件名保持 `crawl_96_auto_v10.exe`；回滚必须从 archive 中按 SHA256 选择，不能再依赖已不存在的顶层 V9/v3 路径。

> **Dynamic-v1 predictor note（2026-09-20）：** Dynamic-v1 设计/审查/决策文档已移入 `archive/docs/20260922_历史设计与提示词/`，仅作历史追溯；96点预测生产规范已由根 `README.md` 与 `docs/RUNBOOK.md`、`docs/DATA_CONTRACT_96.md`、`docs/LEAKAGE_AUDIT_96.md`、`docs/PROJECT_LAYOUT.md`、`docs/OUTPUT_CONVENTION.md` 接管。不要依据旧稿中的 fixed-p60 / 待做 / NEED_USER 状态修改当前 formal96。

> **辅助信息披露爬虫（AUX-V1）：** 当前发布已到 r14（SHA256 `7FAD8476…8388`，38,590,704 B），但 r14 公司真机状态仍为 **NOT_TESTED**；r13 为 **PARTIAL_LIVE_PASS**，最后一个完整真机稳定回滚点仍是 r11d。96 与 AUX 继续保持业务/DB 隔离；当前状态见 `辅助信息披露爬虫/README.md` 与 `辅助信息披露爬虫/AUX-V1_版本记录.md`。

> **经验文档长期维护：** 开始排查或修改本目录任何爬虫前，先读 `爬虫常见问题与经验教训.md`，再读对应 V10/AUX active 架构文档。新问题解决后，只把已核实根因、证据范围、最小修复和回归结果追加到经验文档；待验证推测必须标注，不覆盖历史记录、不把一次 smoke 扩大解释为全量验收。

## 当前 V10 发布状态（2026-10-10，V10-r11）

当前部署入口为 `crawl_96_auto_v10.exe`，自报 `BUILD_VERSION=2026-09-28-runtime-resilience-v10-r11`，大小 44,363,878 bytes，SHA256 `F71A272F703BED360DFFCB06007B77A25874A789ECD8F738D6489C9B8F957BB9`。tracked canonical spec 为 `scripts/crawler/build/crawl_96_auto_v10.spec`；`dist/build_artifacts/` 只作为本机构建工作区。

r11 沿用 r1-r4 的稳定性 P0 能力，并在应用层 `apps/crawl_96.py` + `resilience/` 上继续收口：Guardian/preflight、死 favicon/404/error/blank 页面拒绝复用、无主 profile/孤儿浏览器治理、日志按日轮转和认证失败有界恢复。业务字段映射、raw-first、本地总表与 `epf_pmos_96_full` 写库口径不因这些防御改动而改变。

部署目录已有 r11 单日真实 `PASS` 记录（2026-09-30 run `20260930T043227Z-a0fcffc5`，complete=1、upload_failures=0）；`--db-verify` PASS 只算数据库核验，不替代完整采集验收。

### r1-r4 历史基线（保留作设计追溯）

V10 早期 P0 修补包括：

- `output_96/.crawler.lock` 使用操作系统文件锁，整轮运行单实例；第二实例打印
  `已有爬虫实例运行` 并以非零退出，不创建 RunReport、不启动浏览器、不写 raw/CSV/DB；
- 已登录 dashboard 的 Cookie 失效时，在同一 CDP 浏览器导航回 `login_url`，最多恢复一次；
  不因登录超时、滑块、UKey、Cookie、QCTC 上下文或 401/403 切换第二浏览器；
- 只有 DevTools/CDP、WebSocket、target 或页面控制能力真正丢失时才允许 browser fallback，
  报告事件为 `BROWSER_CONTROL_LOST`；
- QCTC 上下文缺失是 `QCTC_CONTEXT_SOFT_MISSING`（PARTIAL）观测项，继续真实业务请求；
  业务 200/code=0 才成功，401/403 仍硬拒绝。
- Windows 浏览器 discovery 固定检查 C 盘标准 Chrome/Edge 路径；未配置 `browser_path` 时
  候选顺序为 Chrome → Edge，Chrome 仅在 bootstrap 阶段失败时才回退 Edge。
- 复用既有 CDP 前强制执行 Runtime.evaluate 健康门禁；`chrome-error://`、`edge://`、
  `about:blank`、非 PMOS runtime 或 evaluate/WebSocket 失败均记录
  `BROWSER_REUSE_UNHEALTHY`，改用独立 profile 进入同一 bootstrap/Edge fallback 链。
- bootstrap 进一步检查 bounded DOM/runtime render probe；PMOS URL 持续空白时记录
  `BROWSER_BOOTSTRAP_RENDER_STALLED`，再按原候选链尝试 Edge，不把短暂 7--13 秒空白误判为失败。
- Edge discovery 在既有路径基础上增加 Windows App Paths（32/64 registry view）和运行中
  `msedge.exe` 的只读 PowerShell executable path 兜底。

V10-r4 SHA256：`5B72CB925C81DE0226FBCE9E8011D1742D84DDF031E27F32745DFB3BAF0EE401`。
V10-r3 基线 SHA256：`CA8B22595CE496AC9A713F10E29B1F17AE8537A81FB1B006EE8630C9CE3836C8`。
V10-r2 基线 SHA256：`0A9C97DD07520C2766A6CD502AE8528AFDA9A6914612F8B849B5601E928C499D`。
V10-r1 基线 SHA256：`38E8BB262A4554F41361536DFA185306980AE5CCEA5C43F3941B4229981F3F8B`。

## 一、正式程序

正式生产只运行：

```text
crawl_96_auto_v10.exe
```

它包含完整链路：

```text
自动登录 PMOS
  -> 获取 Cookie / Bearer token
  -> 采集预测、实际、电价和机组数据
  -> 96 点覆盖审计
  -> 写入本地 raw 和总表
  -> 同步 MySQL epf_pmos_96_full
```

当前价格口径固定为：

| 数据 | 接口 | 目标字段 |
|---|---|---|
| 日前出清 | `DaJyjgfbPlantQuery/getDetail96`（二次出清/最终版） | `日前出清价格` |
| 实时出清 | `YxJyjgfbPlantQuery/getDetail96`（正式版） | `实时出清价格` |
| 实时兜底 | `YxJyjgfbPlantQueryTmp/getDetail96` | 仅正式版无数据时使用 |

接口原始字段 `cqPrice` 映射为对应价格字段。日前、实时数据严格分开，不互相填充。

v7 同时把 QCTC 信息披露的多套语义分列写入 `epf_pmos_96_full`：

| 来源 | 数据库字段 |
|---|---|
| ForecastData | 原8个预测字段 + `全网负荷预测` |
| RealityData | 原9个正式实际字段 + `全网负荷实际` |
| RealityTmpData | `直调负荷临时实际` 等10个独立临时实际字段 |
| ForecastBoundaryData | `边界全网负荷预测`、`边界直调负荷预测`、`边界外电预测`、`边界风电预测`、`边界光伏预测`、`边界核电预测` |

正式实际、临时实际、ForecastData、Boundary 永远分列保存，不互相覆盖。历史旧运行不会自动补出临时/Boundary 值；这些列从 v7 后续采集开始积累。

## 二、正式目录

```text
dist/crawler/
├─ crawl_96_auto_v10.exe      唯一生产入口（当前 r11）
├─ config.json                PMOS、浏览器和认证配置
├─ config.example.json        配置模板
├─ db_config.json             MySQL 配置（不要外发）
├─ db_config.example.json     MySQL 配置模板
├─ README.md                  本说明文件
├─ 辅助信息披露爬虫/          独立 AUX-V1 EXE/config/output/DDL（不属于 96 点主入口）
│  ├─ crawl_disclosure_aux_v1.exe
│  ├─ README.md
│  └─ 辅助信息披露爬虫_AUX-V1最终架构与实施设计.md
│
├─ tools/日前补数/             辅助程序，不是主生产入口
│  ├─ crawl_da_only.exe       只采集日前最终版
│  ├─ config_da.example.json
│  └─ 运行日前补数.cmd
│
└─ archive/                   历史程序、抓包、备份和运行产物
```

正式目录不放 JS 探测脚本、HAR 抓包、备用 EXE 和历史 README。历史设计稿与旧实施提示词在
`archive/docs/20260922_历史设计与提示词/` 中保留用于追溯；HAR 和运行产物仍按原有
`archive/`、`info/`、`output_96/` 目录管理。

## 文档分层

当前维护只读取以下 active 文档：

- `README.md`：部署入口、当前 V10 运行行为和目录边界；
- `爬虫常见问题与经验教训.md`：跨 V10/AUX 的常见配置、浏览器、采集、数据库及回补问题；每次确认新根因后长期追加并标注证据范围；
- `爬虫版本台账与发布治理.md`：版本、构建修订和发布规则；
- `爬虫源码基线与增量修改规则.md`：源码唯一基线和最小增量原则；
- `爬虫源码版本注释规范.md`：源码版本注释格式；
- `V9稳定性边界审计与V10最小恢复设计.md`：当前 V10 稳定性设计；
- `辅助信息披露爬虫/README.md`、`辅助信息披露爬虫/辅助信息披露爬虫_AUX-V1最终架构与实施设计.md`：当前 AUX-V1 入口、实施与运维规则；同目录旧 `*_规划与数据库设计.md` 仅保留讨论历史。

Dynamic-v1 设计演化稿和 V10 历史实施提示词不再放在顶层，归档索引见
`archive/docs/20260922_历史设计与提示词/README.md`。

## 三、配置

首次部署：

1. 复制 `config.example.json` 为 `config.json`；
2. 填写 PMOS 账号、密码、机组 ID 和浏览器参数；
3. 如需上传数据库，复制 `db_config.example.json` 为 `db_config.json` 并填写连接信息；
4. `config.json` 和 `db_config.json` 必须与 `crawl_96_auto_v10.exe` 同目录。

配置文件含账号、密码、Cookie 或数据库凭据，不提交 Git，不复制到不可信位置。

## 四、主程序运行

```powershell
.\crawl_96_auto_v10.exe --auth-only
.\crawl_96_auto_v10.exe --lookback 1
.\crawl_96_auto_v10.exe --date YYYY-MM-DD --force
.\crawl_96_auto_v10.exe --dry-run
.\crawl_96_auto_v10.exe --db-check
.\crawl_96_auto_v10.exe --db-verify YYYY-MM-DD
```

程序运行后会自动生成：

```text
output_96/
├─ crawler.log
├─ report.json
├─ pmos_96_全量.csv
├─ raw/YYYY-MM-DD.json
└─ upload_queue/
```

日前和实时数据不完整时，程序仍保存已获取数据；缺失点和字段记录在
`output_96/report.json`，不会跨来源补值。

### UKey 口令（PIN）必须配置，否则每次都要人工输入

登录走到证书环节时，UKey 会弹出「验证UKey用户口令」窗口。程序可以自动填写，
**前提是配置里有 PIN**；否则会打印
`pin.handler_fallback=manual reason=pin_not_configured env=PMOS_UKEY_PIN`，
并停在 `auth.waiting_for_human action=ukey_pin`，直到有人手动输完
（2026-09-16 实测等了 26 秒）。

二选一即可（改完立即生效，**不需要重新打包**）：

```json
// dist/crawler/config.json —— 在现有内容里补 ukey_pin 一项即可
{ "pin_handler": "windows", "ukey_pin": "你的6位PIN" }
```

```powershell
setx PMOS_UKEY_PIN "你的6位PIN"     # 环境变量优先于 config.json
```

成功标志：日志出现 `pin.submitted window=... mode=click`，且**不再**出现
`auth.waiting_for_human action=ukey_pin`。

如果配了 PIN 仍要人工输入，查这几条诊断日志：

| 日志 | 处理 |
|---|---|
| `pin.window_no_edit` | 没找到可见输入框，把日志里的 `children` 发回核对窗口标题 |
| `pin.confirm_button_missing fallback=enter` | 没有确定按钮，已自动改用 Enter，可忽略 |
| `pin.settext_failed` | 输入框拒绝写入，通常是权限问题 → **以管理员身份运行 exe** |
| `pin.window_still_present attempt=N` | PIN 被拒会重新弹窗，最多自动重试 3 次 |

> 历史旧配置里若还存在 `ukey_auto_submit` / `ukey_pin_env`，源码不会读取它们；当前模板已移除这些误导键，PIN 使用 `pin_env` / `ukey_pin`。

### QCTC 页面被弹回门户时如何判断

旧门户 Cookie 与新版 QCTC 是两层认证，但**「没有 Bearer token」不等于接口不可用**：
2026-09-13 的日志里，QCTC 数据接口在 `:18080` 的浏览器上下文里成功返回过
`status=200` 并写入 96 行数据；而 09-15、09-16 两轮补数因为把「拿不到 Bearer」
当成了硬失败，**一次数据请求都没发出去**就中止了。

现行行为（2026-09-16 起）：

- `QCTC_CONTEXT_READY`：SPA 自建上下文就绪，直接采集；
- `QCTC_CONTEXT_SOFT_MISSING`：没有 Bearer，但**继续调用接口**，由接口返回码
  判定真实可用性——这是观测项，不是失败；
- `浏览器 fetch 复用同源页面`：标签页已经在 `:18080` 同源时不再导航，直接请求；
- `改用同源兜底页重试`：仅在 fetch 返回 `status=0`（页面被弹回门户导致跨源）
  或 `502` 时，换 `:18080` 的其他同源页重试一次。

程序仍会先尝试点击门户里的 QCTC/信息披露/现货菜单；找不到菜单就等待人工。
如果 CDP 端口中断，程序会放弃故障浏览器、启动新浏览器重新认证后再重试，不会继续
使用已经失效的 9222。QCTC 模式还会跳过旧 `/trade` 附加接口，避免把无关的 502
当成核心采集失败。
详细过程请审阅 `output_96/crawler.log` 和累计更新的 `output_96/report.json`
（建议看报告顺序：`status` → `stages.qctc_context` → `dates` → `events`）。

## 五、日前辅助补数程序

`tools/日前补数/crawl_da_only.exe` 只请求日前最终版接口，不负责自动登录、实时数据或数据库同步。

它用于手工补数和接口核验。需要把从浏览器复制的 `curl.txt`、`cookie.txt` 或 `token.txt`
放在 `tools/日前补数/` 中，再运行：

```powershell
.\tools\日前补数\crawl_da_only.exe --start 2022-01-01 --end YYYY-MM-DD
```

输出位于 `tools/日前补数/output_da/`，不能直接替代主程序数据库同步。

## 六、源码三类职责

源码位于项目 `scripts/crawler/`：

```text
auth/       自动登录：账号密码、滑块、Cookie、UKey/CFCA
collect/    爬虫：PMOS 接口、字段映射、96 点审计
sync_db/    数据库：仅建表、写入 epf_pmos_96_full
runtime_lock.py  单实例 OS 文件锁（不参与业务数据映射）
```

主程序内部实际依赖：

```text
collect/crawl_96_local.py      主流程
collect/crawl.py               接口和字段解析
auth/auth_runtime.py           认证路由
auth/auth.py                   账号密码登录
auth/browser_session.py        浏览器 CDP/Cookie
auth/cfca_runtime.py           UKey/CFCA
auth/auto_crawler/*            登录状态机、滑块和浏览器控制
sync_db/run_crawler.py         数据库同步
sync_db/sql/*                  数据库建表 SQL
```

## 七、数据库交付表

正式表：

```text
epf_pmos_96_full
```

业务字段与：

```text
data/96/authoritative/pmos_96_全量.csv
```

保持一致，日前和实时价格字段分别为：

```text
日前出清价格
实时出清价格
```

验证字段和日期范围：

```sql
SHOW FULL COLUMNS FROM epf_pmos_96_full;

SELECT
    COUNT(*) AS total_rows,
    COUNT(`日前出清价格`) AS da_price_rows,
    COUNT(`实时出清价格`) AS rt_price_rows,
    MIN(market_date) AS min_date,
    MAX(market_date) AS max_date
FROM epf_pmos_96_full;
```

## 八、故障排查

- 认证失败：查看 `output_96/crawler.log`；
- PMOS 无法访问：确认国网内网和浏览器网络；
- 日前/实时价格不完整：查看 `output_96/raw/YYYY-MM-DD.json` 的审计结果；
- 数据库失败：查看 `output_96/upload_queue/`，不要手动用旧表补写；
- SSL 报错：使用当前 EXE 的 `--ssl-check`，不要用旧版 EXE；
- 旧程序：只允许在 `archive/` 中追溯，禁止重新作为生产入口。
