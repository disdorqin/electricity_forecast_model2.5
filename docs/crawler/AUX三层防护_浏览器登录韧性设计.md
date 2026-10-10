# AUX 爬虫「三层防护」浏览器登录韧性设计

> 版本：r11f｜日期：2026-09-27｜适用范围：AUX 辅助信息披露爬虫（96 主爬虫后续同步）
> **状态：代码已实施并完成本机冒烟（BUILD `2026-09-27-disclosure-aux-v1-r11f`，
> SHA256 `76704a75…9a84ed15`，43,016,691 B）；公司机真机验证待执行。**
> **执行顺序（以程序为准）**：候选顺序由 `browser_executable_candidates()` 决定
> （显式 `browser_path` 优先，否则 Chrome→Edge），调度器只做去重与 profile 跟随。
> 实测 `L1-REUSE(已有进程) → L2-BROWSER1(chrome, 原 profile) → L2-BROWSER2(msedge, <原profile>_msedge)`；
> 采集期 401 恢复（`force_new=True`）跳过 L1，直接从 chrome 重开。
> 定位：解决长跑（1500+ 天）场景下**认证环节单点脆弱**导致的整轮白跑问题。
> 专利素材：本文第三节「检测判据」与第四节「降级决策矩阵」为可申请专利的核心创新点。

---

## 一、问题背景（来自 2026-09-27 真机复盘）

一次 1500 天回补长跑中，认证环节发生三类失效，任一出现即导致**整轮白跑**：

| 失效模式 | 现场表现 | 后果 |
|---|---|---|
| **A. 浏览器进程残留/Profile 占用** | `bootstrap_launcher_exited code=0` → 20s 超时「未打开可控 PMOS 页面」 | 新浏览器永远起不来 |
| **B. Cookie/会话失效** | 页面显示「网页已失效」，人工 F5 可恢复 | 程序不会刷新，卡到 600s 超时 |
| **C. 浏览器中途死亡** | CDP 端口 `WinError 10061 主动拒绝`，每个源 2s 超时刷屏 | **不触发重认证**，一路白跑到手动停止 |

根本矛盾：**登录是整条链路的核心前置**——登录拿不到，后面 33 个源 × N 天的采集全部归零。

---

## 二、三层防护总体架构

```
                    ┌─────────────────────────────────┐
                    │   认证请求（启动时 / 采集期 401 │
                    │   / 浏览器死亡 / 页面失效）     │
                    └───────────────┬─────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ L1 复用层：优先复用已有浏览器进程                  │
        │   ├─ B1 Cookie 有效  → LOGGED_IN → ✅ 完成         │
        │   ├─ B2 Cookie 失效  → 刷新重导航（等价人工 F5）   │
        │   │                    → 仍失败 → 降级 L1' 重开    │
        │   └─ B3 网络不可达   → 降级 L2                     │
        └───────────────┬───────────────────────────────────┘
                        ▼  L1 全部失效
        ┌───────────────────────────────────────────────────┐
        │ L2 切换层：按候选列表逐个浏览器完整重试            │
        │   Chrome(1st) → Edge(2nd) → （可配置顺序）        │
        │   每个浏览器独立走完 登录→滑块→CFCA→UKey→Cookie  │
        │   判据：打不开网站 / 登录超时 / 凭据被拒 → 换下一个│
        └───────────────┬───────────────────────────────────┘
                        ▼  L2 全部失效
        ┌───────────────────────────────────────────────────┐
        │ L3 兜底层：诊断与有序退出                          │
        │   插件未安装(UKey 阶段) → 已由 LNA 参数解决        │
        │   三层用尽 → 输出结构化诊断，自主退出（不死循环）   │
        └───────────────────────────────────────────────────┘
```

**关键约束**：每一层必须**完整执行完毕**（含其内部所有重试）才允许降级到下一层；
三层全部用尽才退出。禁止"一次失败就跳层"。

---

## 三、检测判据（核心技术点 · 专利素材）

### 3.1 「Cookie/会话失效」检测

| # | 判据 | 检测位置 | 说明 |
|---|---|---|---|
| D1 | 页面文本命中失效关键词 | `PmosPage.snapshot()` DOM `innerText` | **新增**。对应人工看到的「网页已失效」 |
| D2 | `check_login(cookie)`=False 但页面显示已登录 | `state_machine` stale session | 已有，本次强化为可降级 |
| D3 | 业务接口返回 HTTP 401/403 | `disclosure_aux.capture_source` | 已有，`AuxAuthRejected` |

**D1 关键词表**（PMOS 门户实际文案）：
```
已失效 / 会话已过期 / 会话超时 / 登录已失效 / 登录信息失效
登录状态已失效 / 请重新登录 / 重新登录 / 身份验证已过期 / 用户未登录
```

### 3.2 「浏览器连不上网 / 打不开网站」检测

| # | 判据 | 检测位置 | 说明 |
|---|---|---|---|
| N1 | URL 为 `chrome-error://` / `about:blank` / 非 PMOS | `snapshot()` URL 判定 | **新增**，早于 LOADING 分支 |
| N2 | 页面文本命中网络错误 | `snapshot()` DOM `innerText` | **新增** |
| N3 | CDP 端口连接失败 `WinError 10061 / Max retries` | 请求层 `_aux_request` | **新增为可降级异常**（原为静默 continue） |
| N4 | 页面 `502 Bad Gateway` | `snapshot()` | 已有 `GATEWAY_ERROR` + history.back |

**N2 关键词表**（Chrome/Edge 错误页实际文案）：
```
无法访问此网站 / 网页无法访问 / 找不到该网页
ERR_INTERNET_DISCONNECTED / ERR_NAME_NOT_RESOLVED
ERR_CONNECTION_TIMED_OUT / ERR_CONNECTION_REFUSED
ERR_NETWORK_CHANGED / 没有互联网连接 / 网络连接中断
```

### 3.3 「浏览器进程死亡」检测（修复长跑白跑的关键）

判据：连续 N 次（默认 3）请求出现 `ConnectionError` 且目标是 **CDP 调试端口**（127.0.0.1:922x），
且伴随 `WinError 10061`（目标计算机积极拒绝）→ 判定**浏览器已死亡**，而非网络抖动。

区别对待：
- 目标是 **PMOS 业务域名**的 ConnectionError → 网络问题，按可重试处理；
- 目标是 **127.0.0.1:922x CDP 端口** → 浏览器死亡，触发三层重认证。

### 3.4 「Profile 被占用」检测（失效模式 A）

判据：`launch_browser` 启动器进程退出码 `code=0` **且** 分配的 debug 端口从未建立监听
→ 判定为 **Profile 单实例锁冲突**（Chrome/Edge 对同一 `--user-data-dir` 只允许一个实例）。

此时**不换 Profile**（换临时 Profile 会导致 CFCA/UKey 原生弹窗不出现，历史教训），
而是输出确定性诊断："请先关闭残留浏览器进程"，并按 L2 切换候选浏览器。

---

## 四、降级决策矩阵（核心技术点 · 专利素材）

| 当前层 | 观测到的判据 | 动作 | 目标层 |
|---|---|---|---|
| L1 | D2（Cookie 失效） | 重导航 login_url（等价 F5），限 N=2 次 | 留 L1 |
| L1 | D2 刷新后仍失效 | 重开新浏览器（同浏览器放弃） | L1' → L2 |
| L1 | D1 且刷新无效 | 重开新浏览器 | L1' → L2 |
| L1 | N1/N2（网络不可达） | 放弃复用，启动新浏览器 | L2 |
| L1 | N3（浏览器死亡） | 放弃复用，启动新浏览器 | L2 |
| L2 某浏览器 | bootstrap 超时（打不开网站） | 切换下一个候选浏览器 | L2 下一个 |
| L2 某浏览器 | 登录超时 / 滑块反复失败 / CFCA 卡死 | **切换下一个候选浏览器**（本次新增） | L2 下一个 |
| L2 某浏览器 | 登录成功 | ✅ 完成 | — |
| L2 全部 | 均失败 | 输出诊断，自主退出 | L3 |
| 采集期 | D3（401/403） | 触发完整三层重认证 | 回到 L1 |
| 采集期 | N3（浏览器死亡） | 触发完整三层重认证 | 回到 L1 |
| L3 | 三层用尽 | 结构化诊断 + 退出（不死循环） | END |

---

## 五、实施清单（r11f）

| # | 文件 | 改动 | 对 96 影响 |
|---|---|---|---|
| 1 | `auth/auto_crawler/page.py` | `PageSnapshot` 增加 `session_expired` / `network_unreachable` 字段（默认值，向后兼容）；`snapshot()` 增加 D1/N1/N2 检测 | 96 不读新字段 → **零影响** |
| 2 | `auth/auto_crawler/state_machine.py` | ① `session_expired` → 重导航刷新；② 新增可选**浏览器轮换**（`extra["auth_browser_rotation"]`，仅 AUX 启用） | 96 无此 extra 键 → **行为不变** |
| 3 | `auth/auto_crawler/browser.py` | `launcher_exited code=0` + 端口未监听 → 确定性诊断 `PROFILE_LOCKED` | 仅增加日志 → **零影响** |
| 4 | `collect/crawl_disclosure_aux.py` | `_auth` 升级为三层调度器：L1 复用 → L1' 重开 → L2 逐浏览器 → L3 退出 | 仅 AUX |
| 5 | `collect/disclosure_aux.py` | 连续 3 次 CDP 端口 ConnectionError → 抛 `AuxBrowserLost`（不再静默 continue） | 仅 AUX |
| 6 | `collect/disclosure_aux.py` | `BUILD_VERSION` → r11f | — |

**红线**：共享模块（`page.py` / `state_machine.py` / `browser.py`）只做**向后兼容的加法**，
96 主爬虫行为零变化；所有新策略由 AUX 通过 `AuthConfig.extra` 开关启用。

---

## 六、验证计划

1. **本机冒烟**：`--db-check` 确认 `build=...-r11f`、无 `ModuleNotFoundError`；
2. **公司机真机**：单日 `--date 2026-09-24 --lookback 0 --source all-designed`，
   观察阶段日志出现 `L1_REUSE` / `L2_ROTATE` / `SESSION_EXPIRED_REFRESH` 等新事件；
3. **长跑验证**：续跑 1500 天回补，确认中途浏览器死亡后能自动恢复而非白跑。
