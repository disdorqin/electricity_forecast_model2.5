# 给实施 AI / Codex 的提示词：最小化修复 PMOS UKey + QCTC SSO

你正在维护项目：

```text
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5
```

先阅读：

```text
scripts/crawler/QCTC_SSO_UKEY_MINIMAL_DESIGN.md
scripts/crawler/README.md
```

并检查：

```text
dist/crawler/output_96/crawler.log
dist/crawler/output_96/report.json
dist/crawler/output_96/pmos.sd.sgcc.com.cn11.har
```

## 总原则

这是一次**最小修改**任务。现有自动登录程序已经基本稳定：账号密码、滑块、CFCA/UKey 证书选择、Cookie 获取都能工作。禁止为了修 QCTC 而重构认证系统。

先检查当前 working tree 和未提交修改，绝不能覆盖/回退用户已有变更。只做必要的增量修改。

---

## 已确认事实

1. 门户认证成功，日志已有 `auth_cookie PASS`，不是 Cookie 获取失败。
2. UKey 人工输入的 PIN 就是程序需要的 PIN；当前自动失败是因为 `resolved_pin` 为空。
3. 当前 `WindowsPinHandler` 已具备 Win32 自动填 PIN、点击/Enter、重试等能力。不要重写它，除非正确提供 PIN 后日志仍证明处理器失败。
4. 真人成功链路：

```text
账号密码 → 滑块 → UKey → 门户 dashboard
→ 现货市场
→ 现货交易系统-新架构
→ QCTC SSO
→ :18080/qctc-trade
→ 数据页面
```

5. 门户 HAR 已确认菜单：

```text
menuId = 1595696533569933312
name = 现货交易系统-新架构
menuPath = /psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin
```

6. 最新 `pmos.sd.sgcc.com.cn11.har` 还确认人工点击附近存在：

```text
POST /px-common-authcenter/sso/token
POST /px-common-gateway/px-gateway/GatewayConfig/pageClick
```

`pageClick` 请求体带上述 menuId/menuPath。其业务体可能返回 401，因此不要把 pageClick 成功作为 SSO 前置条件。主目标是复现浏览器进入 `/psso?...SSOLogin` 的真实导航。

7. 当前错误是代码在门户 Cookie 成功后直接导航：

```text
:18080/qctc-trade/informationDisclosure/forecast10424
```

此时没有经过 QCTC SSO，日志出现：

```text
bearer_present=False
→ 被弹回 /#/dashboard
```

8. 进入 QCTC 后不需要模拟每一个左侧业务菜单。现有接口抓取设计继续保留：

```text
ForecastData/getLoadData
RealityData/getLoadData
DaJyjgfbPlantQuery/getDetail96
YxJyjgfbPlantQuery/getDetail96
```

---

# 任务 A：UKey，优先只改配置

检查当前：

```text
scripts/crawler/auth/auto_crawler/config.py
scripts/crawler/auth/auto_crawler/handlers.py
```

如果当前代码与设计文档一致，则不要修改 UKey 算法。

确保部署配置支持：

```json
"pin_handler": "windows",
"pin_env": "PMOS_UKEY_PIN",
"ukey_pin": "<由部署方提供，禁止写入示例/日志>",
"ukey_window_title": "验证UKey用户口令",
"pin_submit_mode": "click"
```

PIN 来源优先级保持：

```text
PMOS_UKEY_PIN 环境变量 > config.json:ukey_pin
```

注意：`ukey_pin_env` 或 `pin_env` 是“环境变量名”，不是 PIN 值。

不要在源码、README 示例、Git diff、日志里写真实 PIN。

如果只是因为部署配置没有 `ukey_pin`，不要为此修改 EXE 内逻辑；配置好后现有 `WindowsPinHandler` 应直接生效。

验收日志必须出现：

```text
pin.submitted
```

并继续出现：

```text
auth_cookie PASS
```

只有配置 PIN 后仍出现 `pin.window_no_edit`、`pin.settext_failed` 等明确 Win32 失败证据时，才允许小改 `handlers.py`。

---

# 任务 B：QCTC SSO，只修改 collect 层

主修改文件：

```text
scripts/crawler/collect/crawl.py
```

原则上不要修改：

```text
scripts/crawler/auth/auto_crawler/state_machine.py
scripts/crawler/auth/auto_crawler/page.py
scripts/crawler/auth/auto_crawler/browser.py
scripts/crawler/sync_db/*
```

不要改变当前门户登录入口/滑块/CFCA，只在 `auth_cookie PASS` 之后建立第二层 QCTC SSO。

## B1. 增加明确的 QCTC SSO 常量

使用：

```text
https://pmos.sd.sgcc.com.cn/psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin
```

例如：

```python
QCTC_PORTAL_SSO_URL = (
    "https://pmos.sd.sgcc.com.cn/psso?service="
    "https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin"
)
```

不要把这个逻辑塞进 `AuthConfig.service_url`；这是 collect 阶段的第二层认证，不是门户账号登录入口。

## B2. 修改 `ensure_qctc_context()`，顺序必须调整

**不要再以“直接打开 forecast10424”作为首次建立 QCTC 上下文的第一动作。**

目标流程：

```text
1. 先扫描同一 debug_port 是否已有 QCTC page/target
2. 已有且 context 可用 → 直接复用
3. 没有 → 从已登录 portal target Page.navigate(QCTC_PORTAL_SSO_URL)
4. 短时间轮询 DevTools /json，允许：
   - 当前 tab 原地跳转
   - 新 tab/window 被创建
5. 找 URL 满足以下特征的 target：
   host = pmos.sd.sgcc.com.cn:18080
   path 含 /qctc 或 /qctc-trade
6. 一旦找到，关闭旧 CDP client，连接新 target 的 webSocketDebuggerUrl
7. 等待 QCTC 页面初始化
8. 检查 sessionStorage/localStorage 的 token/access_token/accessToken/id_token
9. bearer_present=True → QCTC_CONTEXT_READY
10. 如果 QCTC 已就绪但不是 forecast10424，再导航到 forecast10424
11. SSO 自动路径超时后，才使用现有 `_try_portal_qctc_entry()` DOM 点击 fallback
12. 最后保留人工等待兜底
```

注意：真人点击可能新开 tab，因此**必须支持 target 切换**。不要一直持有 dashboard 的 websocket。

尽量复用现有：

```text
_get_pmos_page()
_CdpClient
_browser_page_state()
```

可以增加一个很小的 helper，例如：

```text
_wait_for_qctc_page(debug_port, timeout_sec)
```

或者小幅扩展 `_get_pmos_page(prefer_qctc=True)`，使其能识别：

```text
:18080/qctc/...
:18080/qctc-trade/...
```

不要引入 Selenium/Playwright，不要新增大型依赖。

## B3. 现有 DOM 菜单点击不要删除

现有：

```text
_try_portal_qctc_entry()
```

保留作为 fallback。不要删除已有人工等待路径。这样新 SSO 失败时可以快速回退，不影响旧行为。

## B4. Bearer 与接口成功关系

当前系统已有同源 Cookie fetch 兜底。不要把“没有 Bearer”改成门户登录硬失败。

优先成功指标：

```text
qctc_route=True
bearer_present=True
```

最终数据可用性仍以 QCTC 接口 HTTP/业务 code 判断。

---

# 任务 C：日志，仅增加安全诊断

新增事件建议：

```text
QCTC_SSO_NAVIGATE
QCTC_TARGET_FOUND
QCTC_STORAGE_STATE
QCTC_CONTEXT_READY
QCTC_SSO_FALLBACK_DOM
```

日志只允许输出：

```text
URL/path
是否存在 token
storage key 名
是否新 tab
耗时
HTTP status/business code
```

严禁输出：

```text
UKey PIN
Admin-Token
X-Ticket
Bearer token
sso/token 返回值
完整 Cookie
```

---

# 任务 D：禁止触碰的数据逻辑

不要修改以下内容，除非测试证明和本任务直接有关：

```text
QCTC_FORECAST_MAP
QCTC_ACTUAL_MAP
日前最终版 DaJyjgfbPlantQuery
实时 YxJyjgfbPlantQuery
96点 period 映射
_complete_96
validate_forecast_actual_separation
raw/CSV/DB 同步
COALESCE upsert
数据库 schema
```

当前运行还没有进入真正逐日采集，所以不能用“没有数据”作为理由去重写字段映射。

---

# 任务 E：版本与测试

实现完成后：

1. 更新 `scripts/crawler/collect/crawl_96_local.py` 的 `BUILD_VERSION`，建议：

```text
2026-09-16-qctc-sso-minimal-v1
```

2. 做 Python 语法/导入检查。
3. 如果仓库有相关测试，运行相关测试；不要为了通过无关测试大改代码。
4. 检查 git diff，确保修改面尽量小。
5. 先在真机跑单日或 `--lookback 1`，不要先全量。

真机验收顺序：

```text
pin.submitted
→ auth_cookie PASS
→ QCTC_SSO_NAVIGATE
→ QCTC_TARGET_FOUND
→ QCTC_CONTEXT_READY（优先）
→ QCTC 接口开始返回
→ 至少一个 date 进入 report.dates
→ raw/CSV 正常
→ DB 逻辑保持原样
```

如果 `QCTC_TARGET_FOUND` 失败，保留日志并停止继续猜数据字段。

---

# 任务 F：重新打包 EXE

使用现有：

```text
scripts/crawler/collect/crawl_96_local.spec
```

必须遵守 spec 文件顶部的构建要求：使用项目指定、PMOS TLS 已验证兼容的 build venv/OpenSSL 环境。

不要改变 spec 架构，除非新增代码无法被现有 `collect_submodules('scripts.crawler')` 收入（正常情况下不需要改）。

生成：

```text
crawl_96_auto_v3.exe
```

部署时：

- 先备份旧 EXE；
- 不删除旧版本；
- 不覆盖真实 `config.json` 中其它账号/DB设置；
- 只补 UKey PIN 配置；
- 不把真实 PIN 写入 `config.example.json`。

构建后核对 EXE 修改时间和 `BUILD_VERSION` 日志，确保真机跑的是新 EXE。

---

# 完成后的汇报格式

请向用户报告：

1. 实际修改了哪些文件；
2. 每个文件为什么改；
3. 哪些登录代码明确没有动；
4. UKey 是“代码修复”还是“配置修复”；
5. QCTC SSO 主路径与 fallback；
6. 本地测试结果；
7. EXE 输出位置、大小、修改时间；
8. 真机下一次需要重点观察的日志事件；
9. 如果仍失败，下一步需要用户提供哪一小段日志/HAR，而不是继续盲改。

最重要：**保持最小 diff，优先保护当前已经能正常工作的自动登录。**
