# 爬虫源码全局关系图（CODEGRAPH）

> 最后核实：2026-10-10｜范围：`scripts/crawler/**`（排除 archive/pycache 后 44 个 .py）
> 用途：改动任何一个文件前，**先查这里**确认它被谁依赖、会不会影响 96 / AUX。
> 状态标记：🟢 活跃（当前链路在用）｜🔴 遗留（重构后已失效）｜🆕 新增

---

## 一、目录总览

### 1.1 根层（`scripts/crawler/`）

| 文件 | 职责 | 状态 | 被谁依赖 |
|---|---|---|---|
| `__init__.py` | 包声明 | 🟢 | 全部 |
| `observability.py` | 运行报告（`report.json` 原子写） | 🟢 **共享** | 96 入口:85、AUX 入口:35 |
| `runtime_lock.py` | 进程锁（`.crawler.lock`） | 🟢 **共享** | 96 入口:86、AUX 入口:36 |
| `log_rotation.py` | 96/AUX 日志轮转与 retention | 🟢 **共享** | 96/AUX |

> 24 点复盘平台入口已迁入 `apps/tools/`，根层不再保留 `platform_review*.py` 当前入口。

### 1.2 `auth/` —— 认证

| 文件 | 职责 | 状态 |
|---|---|---|
| `auto_crawler/state_machine.py` | **认证状态机**（登录→滑块→CFCA→PIN→Cookie） | 🟢 主链路 |
| `auto_crawler/browser.py` | 浏览器启动 / CDP / profile | 🟢 主链路 |
| `auto_crawler/page.py` | 页面状态判定（PageSnapshot） | 🟢 主链路 |
| `auto_crawler/config.py` | `AuthConfig` | 🟢 主链路 |
| `auto_crawler/handlers.py` | 滑块 / PIN 处理 | 🟢 主链路 |
| `auto_crawler/main.py` + `__main__.py` | auth 独立入口 | 🟢 |
| `auto_crawler/frozen_entry.py` | frozen 环境入口适配 | 🟢 |
| `auth_runtime.py` | 96 入口调用的认证运行时 | 🟢 |
| `cfca_runtime.py` | CFCA / UKey 原生弹窗协助 | 🟢 |
| `browser_session.py` | 浏览器会话工具 | 🟢 |
| `auth.py` | **旧手工 Cookie 认证**（PmosAuth） | 🔴 遗留（非当前链路） |

### 1.3 `collect/` —— 采集 + 两个入口

| 文件 | 职责 | 状态 |
|---|---|---|
| `crawl.py` | PMOS 接口核心（`PmosCrawler`） | 🟢 |
| **`crawl_96_local.py`** | ★ **96 爬虫入口**（1400+ 行，含业务逻辑） | 🟢 主入口 |
| **`crawl_disclosure_aux.py`** | ★ **AUX 爬虫入口** | 🟢 主入口 |
| `disclosure_aux.py` | AUX 采集核心（源注册表 / 请求 / 解析） | 🟢 |
| `crawl_da_only.py` | 日前最终版手工补数工具（standalone，非主 EXE） | 🟢 辅助工具 |

### 1.4 `sync_db/` —— 落库

| 文件 | 职责 | 状态 |
|---|---|---|
| `run_crawler.py` | 96 落库（建表 + upsert `epf_pmos_96_full`） | 🟢 |
| `disclosure_aux.py` | AUX 落库（upsert `epf_pmos_aux_records`） | 🟢 |
| `sql/` | DDL | 🟢 |

### 1.5 `resilience/` —— 🆕 防御机制（2026-09-27 新建）

| 文件 | 职责 |
|---|---|
| `codes.py` | 根因码 + 处置动作（词汇表） |
| `diagnose.py` | ★核心：异常/响应/页面/环境 → 根因码 |
| `browser_env.py` | profile 锁检测与安全清理、残留进程探测 |
| `health.py` | 启动前健康检查（含后端会话探活） |
| `backoff.py` | 指数退避 + 全抖动 + 重试预算 |
| `circuit.py` | 熔断器 |
| `checkpoint.py` | 自动断点续跑 + 失败清单（本地 JSON） |
| `guardian.py` | 统一编排入口（决策中枢） |
| `doctor.py` | **环境自检**（`--doctor`）：版本 / 配置 / profile 锁与占用 / CDP 端口 / 输出目录 / DB 配置 逐项体检 + 结论建议 |

### 1.6 `apps/` —— 🆕 应用层（**96 与 AUX 均已切为主入口**）

| 文件 | 职责 | 状态 |
|---|---|---|
| **`crawl_96.py`** | **96 主入口**：修路径 → Guardian 防御预检 → 转发 `collect.crawl_96_local.main` → 失败根因诊断 | 🟢 **主入口** |
| **`crawl_aux.py`** | **AUX 主入口**：同上，转发 `collect.crawl_disclosure_aux.main` | 🟢 **主入口** |
| `tools/platform_review.py` | 演示站库（从根层迁入） | 🟢 |
| `tools/platform_review_update.py` | 演示站 CLI（迁入，**`parents[2]`→`parents[4]`、import 路径已同步改**） | 🟢 |

> 两个主入口都在**启动浏览器之前**做环境治理（清理无主 `Singleton*` 锁、CDP 健康检查、
> 复用优先判定），把「残留进程 / profile 锁 → 浏览器连不上」这一大类故障在进入实现模块前化解。
> 逃生开关：`PMOS_DISABLE_RESILIENCE=1` 可完全退回原始行为。
> ⚠ 防御 import **必须写在文件顶层**（PyInstaller 用 AST 收集；函数内延迟 import 会漏打）。

---

## 二、依赖关系图

### 2.1 96 主链路（🟢 生产）

```
apps/crawl_96.py  ★主入口（2026-09-27 起）
   ├── resilience/guardian.py         防御预检：清无主锁 / 健康检查 / 复用优先
   └── collect/crawl_96_local.py  ★实现（被转发，逻辑未改）
         ├── collect/crawl.py            PmosCrawler / parse_number / period_no_from_time
         ├── auth/auto_crawler/config.py      AuthConfig
         ├── auth/auto_crawler/state_machine.py  AuthenticationStateMachine
   ├── auth/auth_runtime.py             ensure_authenticated_config
   ├── sync_db/run_crawler.py           init_database_tables / upsert_full_dataset_table
   ├── observability.py                 RunReport / cookie_summary
   └── runtime_lock.py                  RuntimeLock
```

### 2.2 AUX 链路（🟢 生产）

```
apps/crawl_aux.py  ★主入口（2026-09-27 起）
   ├── resilience/guardian.py           防御预检：清无主锁 / 健康检查 / 复用优先
   └── collect/crawl_disclosure_aux.py  ★实现（被转发，逻辑未改）
         ├── collect/disclosure_aux.py   采集核心 + 源注册表 + 三层防护
         ├── auth/auto_crawler/{browser,config,state_machine}.py
         ├── sync_db/disclosure_aux.py   upsert_source_result
         ├── observability.py
         └── runtime_lock.py
```

### 2.3 共享基座（**改这里会同时影响 96 和 AUX**）

```
observability.py   ─┐
runtime_lock.py    ─┼── 被两个入口同时 import → 签名与位置冻结
auth/auto_crawler/ ─┘
```

### 2.4 独立支路（不交叉）

```
auth/auto_crawler/__main__.py → main.py        （auth 单独调试入口）
apps/tools/platform_review_update.py → platform_review.py   （演示站，与 PMOS 无关）
```

---

## 三、数据流

```
[1] 认证   auth/auto_crawler  浏览器登录 → Cookie + CDP 调试端口
                                     ↓
[2] 采集   collect/*          通过该 CDP 端口在浏览器内发业务请求（同源，绕过跨域）
                                     ↓
[3] 落库   sync_db/*          upsert（唯一键幂等）
                                     ↓
[4] 报告   observability.py   report.json（阶段/日期摘要）
```

⚠ **关键耦合**：采集依赖认证交出的 **CDP 端口**；浏览器一死，采集全部失败
→ 这正是 `resilience` 要治理的核心（根因码 `CDP_PORT_DEAD`）。

---

## 四、🔴 历史遗留 / 已失效（**不要碰、不要照抄**）

以下引用路径在目录重构后**已不存在**，相关脚本目前不可用：

| 失效引用 | 实际位置 |
|---|---|
| `scripts.crawler.crawl` | 应为 `scripts.crawler.collect.crawl` |
| `scripts.crawler.run_crawler` | 应为 `scripts.crawler.sync_db.run_crawler` |
| `scripts.crawler.browser_session` | 应为 `scripts.crawler.auth.browser_session` |

上述旧包路径只用于识别历史代码/文档中的失效引用；当前 `collect/crawl_da_only.py` 已是 standalone 日前补数工具，不依赖这些旧 import。`auth/auth.py` 仍仅为条件兼容路径，96/AUX 主链默认走 `auto_crawler`。

---

## 五、打包关系（哪个 spec 打哪个 exe）

| exe | spec 的 Analysis entry | 产物 |
|---|---|---|
| `crawl_96_auto_v10.exe` | `scripts/crawler/build/crawl_96_auto_v10.spec` → `apps/crawl_96.py` | 44,363,878 B（V10-r11，SHA256 `F71A272F…57BB9`） |
| `crawl_disclosure_aux_v1.exe` | `scripts/crawler/build/crawl_disclosure_aux_v1.spec` → `apps/crawl_aux.py` | 38,590,704 B（AUX-r14，SHA256 `7FAD8476…8388`） |

> 两个 tracked canonical spec 均放在 `scripts/crawler/build/`；`dist/build_artifacts/` 只作为本机构建工作区。两个 EXE 的 entry 都是 `apps/` 应用层入口。
> 96 spec 使用 `collect_submodules('scripts.crawler')`；AUX spec 显式收集 explore 与 `scripts.crawler.resilience`，避免 frozen 包静默漏模块。

---

## 六、改动安全速查

| 我要改… | 影响面 | 风险 |
|---|---|---|
| `observability.py` / `runtime_lock.py` | 96 + AUX | 🔴 高（冻结） |
| `auth/auto_crawler/*` | 96 + AUX | 🔴 高 |
| `collect/crawl.py` | 96（+AUX 部分） | 🟠 中 |
| `collect/disclosure_aux.py` | 仅 AUX | 🟠 中 |
| `collect/crawl_96_local.py` | 仅 96 | 🟠 中 |
| `sync_db/*` | 落库 | 🟠 中 |
| `resilience/*` | 96 + AUX preflight/doctor | 🟠 中（已是生产入口依赖） |
| `apps/crawl_96.py` / `apps/crawl_aux.py` | 对应生产 EXE 主入口 | 🔴 高（打包入口） |
| `platform_review*.py` | 仅演示站 | 🟢 低 |

### ⚠ 移动文件必查三项（曾踩坑）
1. `Path(__file__).resolve().parents[N]` —— **层级变了 N 就要改**
   （`platform_review_update.py` 原 `parents[2]`，迁入 `apps/tools/` 后必须改 `parents[4]`）
2. spec 的 `Analysis` entry 路径
3. 扁平 import fallback（`try: scripts.crawler.X / except: X`）
