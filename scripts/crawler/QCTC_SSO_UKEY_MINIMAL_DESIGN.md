# PMOS 爬虫最小修改设计：UKey 自动 PIN + QCTC 新架构 SSO

> 适用项目：`electricity_forecast_model2.5/scripts/crawler`
>
> 设计日期：2026-09-16
>
> 核心原则：**现有账号密码、滑块、CFCA/UKey 证书、Cookie 登录流程已经可用，禁止大改；只修 UKey PIN 配置与门户→QCTC 的第二层 SSO。**

## 1. 目标与非目标

### 1.1 本次目标

1. UKey 弹窗出现后自动输入用户提供的 PIN，并自动确认，不再人工输入。
2. 门户登录成功后，自动完成真人操作中的：
   `现货市场 → 现货交易系统-新架构 → QCTC SSO → qctc-trade`。
3. QCTC 上下文建立后，继续沿用现有接口直接抓取预测、实际、日前、实时数据。
4. 重新打包 `crawl_96_auto_v3.exe`，保持甲方机器无 Python 可运行。

### 1.2 明确非目标

本次**不修改**：

- 账号密码提交逻辑；
- 滑块识别与拖动逻辑；
- CFCA 证书选择逻辑；
- `AuthenticationStateMachine` 的状态定义与主循环；
- Cookie 成功判定；
- 预测/实际/日前/实时字段映射；
- 96 点补齐、PARTIAL、预测/实际防污染逻辑；
- CSV/raw/DB 写入逻辑；
- MySQL schema；
- 已验证的接口 URL 与数据口径。

不要为了 QCTC SSO 去重写整个浏览器认证模块。

---

## 2. 已确认事实

### 2.1 门户认证已经成功

最新运行已经出现：

- `auth_cookie = PASS`
- `Admin-Token`
- `X-Ticket`
- Cookie 成功写回配置

因此当前问题不是“Cookie 没拿到”。

### 2.2 UKey 自动化失败原因已确认

当前 `AuthConfig` 的 PIN 来源：

```text
环境变量 PMOS_UKEY_PIN
        ↓ 没有时
config.json: ukey_pin
```

当前 `WindowsPinHandler` 已经具备：

- 精确标题匹配 + UKey/口令关键字兜底；
- 找原生 Edit 控件；
- `WM_SETTEXT` 写入；
- 回读检查；
- `WM_CHAR` 兜底；
- 点击“确定/确认/提交/OK”；
- 找不到按钮时 Enter 兜底；
- 最多重试。

因此 **UKey 不需要重写算法**。

之前日志：

```text
pin.handler_fallback=manual reason=pin_not_configured env=PMOS_UKEY_PIN
```

说明只是实际运行配置未提供 `resolved_pin`。

### 2.3 QCTC 是第二层认证，不等同于门户 Cookie

真人操作：

```text
账号密码 → 滑块 → UKey → 门户 dashboard
→ 现货市场
→ 现货交易系统-新架构
→ QCTC SSO
→ :18080/qctc-trade
→ 数据页面
```

现有失败流程：

```text
门户 Cookie 成功
→ 直接打开 /qctc-trade/informationDisclosure/forecast10424
→ bearer_present=False
→ 被弹回 /#/dashboard
```

所以失败点是“门户→QCTC 新架构”的 SSO 桥。

---

## 3. HAR 证据

### 3.1 旧 HAR：门户菜单树

门户菜单响应明确存在：

```text
名称：现货交易系统-新架构
menuId：1595696533569933312
menuPath：/psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin
```

### 3.2 最新 HAR

文件：

```text
dist/crawler/output_96/pmos.sd.sgcc.com.cn11.har
```

人工点击“现货交易系统-新架构”附近，HAR 记录：

```text
POST /px-common-gateway/px-gateway/GatewayConfig/pageClick
```

请求体包含：

```json
{
  "menuId": "1595696533569933312",
  "menuCode": "",
  "menuPath": "/psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin",
  "menuType": "menu"
}
```

同一时段门户多次调用：

```text
POST /px-common-authcenter/sso/token
```

并返回成功 token 数据。

HAR 中 `pageClick` HTTP 层为 200，但业务体曾返回 `status=401`，因此该接口更像点击审计/网关记录，**不应作为 QCTC SSO 成功前置条件**。真正必须复现的是浏览器通过 `/psso?...SSOLogin` 进入 QCTC 的导航行为，并观察最终 QCTC 上下文。

---

## 4. 最小修改方案

## 4.1 UKey：配置修复优先，代码不动

### 实际部署配置

实际运行的公司机配置（日志中是 `D:\爬虫电网\config.json`）必须满足以下二选一：

方案 A，推荐：

```text
Windows 环境变量：PMOS_UKEY_PIN=<实际 PIN>
```

方案 B，现场验证方便：

```json
{
  "pin_handler": "windows",
  "pin_env": "PMOS_UKEY_PIN",
  "ukey_pin": "<实际 PIN>",
  "ukey_window_title": "验证UKey用户口令",
  "pin_submit_mode": "click"
}
```

注意：

- `ukey_pin_env`/`pin_env` 表示“环境变量名称”，不是 PIN 值；
- 不要把实际 PIN 写进 `config.example.json`；
- 不要把实际 PIN 打印进日志；
- 不要为了这个问题修改账号登录、滑块和 CFCA 状态机。

### 验收

原日志：

```text
pin.handler_fallback=manual
```

应消失，改为出现：

```text
pin.submitted ... mode=click|enter
```

随后仍能出现：

```text
auth_cookie PASS
```

如果配置 PIN 后才出现 `pin.window_no_edit` / `pin.settext_failed`，才允许进一步修改 `handlers.py`。

---

## 4.2 QCTC：只改 `collect/crawl.py` 的上下文建立部分

### 设计原则

保留当前 `_try_portal_qctc_entry()` 作为最后兜底，但主路径不要再依赖“DOM 猜菜单”。

增加一个明确的 QCTC SSO 地址：

```text
https://pmos.sd.sgcc.com.cn/psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin
```

建议新增常量：

```python
QCTC_PORTAL_SSO_URL = (
    "https://pmos.sd.sgcc.com.cn/psso?service="
    "https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin"
)
```

不要把它混进 `auth/auto_crawler/config.py` 的门户登录 `service_url`。门户登录已经正常工作，QCTC SSO 应属于 collect 阶段。

### `ensure_qctc_context()` 修改后的主流程

当前流程：

```text
直接 Page.navigate(forecast10424)
→ 无 Bearer
→ 回 dashboard
→ 猜 DOM 菜单
```

改为：

```text
A. 检查是否已经存在可用 QCTC target
   ├─ 有：继续检查 token/context
   └─ 无：进入 B

B. 当前门户已登录时，用当前浏览器直接导航到 QCTC_PORTAL_SSO_URL
   → 记录 QCTC_SSO_NAVIGATE

C. 在同一个 debug_port 下轮询 DevTools targets
   → 允许同 tab 跳转
   → 也允许新 tab/window
   → 优先寻找 URL host=pmos.sd.sgcc.com.cn:18080
     且 path 包含 /qctc 或 /qctc-trade

D. 找到 QCTC target 后，重新绑定 CDP 到该 target
   → 记录 QCTC_TARGET_FOUND

E. 等待 QCTC SPA / storage 初始化
   → 检查 sessionStorage/localStorage 的 token/access_token/accessToken/id_token
   → bearer_present=True 时记录 QCTC_CONTEXT_READY

F. 如果已经进入 QCTC origin，但当前不是 forecast 页面
   → 再导航 forecast10424
   → 然后继续原接口采集

G. 只有 SSO 导航失败/超时后，才调用现有 `_try_portal_qctc_entry()`

H. 最后才保留人工等待兜底
```

### 为什么必须支持新 tab

真人点击“现货交易系统-新架构”可能新开页面。不能一直持有登录阶段那个 dashboard 的 websocket target。

代码必须按 debug port 重新扫描 target，并切换到实际 QCTC target。

### 成功条件

优先成功条件：

```text
origin = https://pmos.sd.sgcc.com.cn:18080
qctc_route = true
bearer_present = true
```

保留现有 Cookie/same-origin fetch 兜底时，不要把“Bearer 缺失”本身等同于账号登录失败；最终仍以 QCTC 接口真实 HTTP/业务返回判断数据是否可取。

---

## 5. 建议新增的日志/Report 事件

只增加诊断，不记录密钥值：

```text
QCTC_SSO_NAVIGATE
  sso_path=/psso
  target_service=:18080/qctc/admin/sdsso/SSOLogin

QCTC_TARGET_FOUND
  target_url=<截断后的URL>
  same_tab=true|false

QCTC_STORAGE_STATE
  bearer_present=true|false
  session_storage_keys=[仅键名]
  local_storage_keys=[仅键名]

QCTC_CONTEXT_READY

QCTC_SSO_FALLBACK_DOM
```

严禁打印：

- UKey PIN；
- Admin-Token/X-Ticket 值；
- Bearer Token 值；
- `/sso/token` 返回的 token 值。

---

## 6. 允许修改 / 禁止修改文件

### 允许修改（优先级顺序）

1. `scripts/crawler/collect/crawl.py`
   - 仅 QCTC SSO/target/context 建立相关函数。
2. `scripts/crawler/collect/crawl_96_local.py`
   - 只允许更新 `BUILD_VERSION`；除非确有必要，不改主编排。
3. `scripts/crawler/config.example.json`
4. `dist/crawler/config.example.json`
   - 仅补齐 PIN 配置说明，不写真实 PIN。
5. 实际部署 `config.json`
   - 配置实际 PIN；不提交真实凭据。

### 原则上禁止修改

- `auth/auto_crawler/state_machine.py`
- `auth/auto_crawler/page.py`
- 滑块相关代码
- 账号密码相关代码
- `sync_db/*`
- 字段映射
- DB schema
- 96点完整性与数据口径

### `handlers.py`

默认**不改**。只有在实际 PIN 已正确提供后，日志仍证明 Win32 UKey 输入失败，才做针对性小修。

---

## 7. 实施顺序

### 阶段 1：UKey 配置验证

只配置 PIN，不修改 Python。

验收：

```text
pin.submitted
→ auth_cookie PASS
```

### 阶段 2：QCTC SSO 最小代码修改

只改 `collect/crawl.py`。

建议先跑：

```text
单日 / lookback 1
```

不要直接全量回补。

### 阶段 3：接口验证

必须看到：

```text
QCTC_CONTEXT_READY
```

然后才看：

```text
ForecastData/getLoadData
RealityData/getLoadData
DaJyjgfbPlantQuery/getDetail96
YxJyjgfbPlantQuery/getDetail96
```

确认至少一个业务日真正进入 `dates`，而不是 `collect=START, dates={}`。

### 阶段 4：重新打包 EXE

使用现有：

```text
scripts/crawler/collect/crawl_96_local.spec
```

注意 spec 已明确：必须使用项目指定的 `venv_build` / 兼容 OpenSSL 构建环境，不要用其它 Python 环境随意重打。

构建前更新版本号，例如：

```text
2026-09-16-qctc-sso-minimal-v1
```

构建后把新 `crawl_96_auto_v3.exe` 放到测试部署目录，**保留旧 EXE 备份，不覆盖到无法回滚**。

---

## 8. 回归测试清单

必须逐项通过：

- [ ] 自动启动浏览器/CDP 仍正常；
- [ ] 账号自动填充仍正常；
- [ ] 滑块仍能自动完成；
- [ ] CFCA 仍能识别；
- [ ] UKey PIN 自动输入；
- [ ] 门户 `auth_cookie PASS`；
- [ ] 自动进入 `/psso?...qctc/admin/sdsso/SSOLogin`；
- [ ] 能发现同 tab 或新 tab 的 QCTC target；
- [ ] QCTC 页面不再因为少走 SSO 而立即弹回 dashboard；
- [ ] `QCTC_CONTEXT_READY`；
- [ ] Forecast 接口真实返回；
- [ ] Actual 接口真实返回；
- [ ] 日前最终版接口真实返回；
- [ ] 实时接口真实返回；
- [ ] raw 正常写入；
- [ ] CSV 正常追加；
- [ ] DB 逻辑没有因本次修改发生变化；
- [ ] 日志中没有任何 PIN/Token/Cookie 明文。

---

## 9. 回滚策略

本次必须保持易回滚：

1. 代码修改集中在 `collect/crawl.py`；
2. 原 `_try_portal_qctc_entry()` 不删除，只降级为 fallback；
3. 原数据采集 API 不改；
4. 原登录状态机不改；
5. 新 EXE 与旧 EXE 同时保留，文件名或 archive 中标记版本；
6. 若 QCTC SSO 新路径异常，可以立即恢复旧 `crawl.py` / 旧 EXE，不影响已验证的门户登录。

---

## 10. 最终架构

```text
现有自动登录（保持不动）
账号 → 滑块 → CFCA → UKey自动PIN → Portal Cookie PASS
                                      |
                                      v
                     新增最小 QCTC SSO Bridge
 Portal /#/dashboard
       |
       v
 /psso?service=:18080/qctc/admin/sdsso/SSOLogin
       |
       v
 扫描/切换到 QCTC CDP target
       |
       v
 qctc-trade + Bearer/Cookie Context
       |
       v
 现有 API 直接采集（保持不动）
 Forecast / Actual / DA Final / Realtime
       |
       v
 raw → CSV → canonical DB
```

这个方案的核心是：**只模拟真人必须完成的“进入现货交易系统-新架构”这一跳；进入 QCTC 后继续使用已经验证过的接口，不模拟每个业务菜单。**
