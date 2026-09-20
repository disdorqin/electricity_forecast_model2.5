# PMOS QCTC 二次认证最小修复设计（Ticket 版）

## 1. 文档目的

本文只解决当前已经被真机日志证实的两个问题：

1. 已经获得 QCTC Bearer 的情况下，程序错误地再次导航业务页面，破坏了一个原本可用的 QCTC 浏览器上下文；
2. 全新浏览器登录后，程序把门户菜单 URL `/psso?service=...` 当成真正 SSO 地址直接访问，但人工点击菜单实际上会先取得 SSO ticket，再打开 `SSOLogin?ticket=...`。

本次仍然坚持“最小修改”。账号密码、滑块、CFCA、UKey Win32 输入、门户 Cookie、字段映射、四类 QCTC API、96 点审计、CSV、数据库同步都不重构。

主修改范围：

```text
scripts/crawler/collect/crawl.py
scripts/crawler/collect/crawl_96_local.py   # 仅 BUILD_VERSION
```

UKey 本次主要是部署配置修复，不要求重写 `handlers.py`。

---

## 2. 真机日志已经证明的事实

### 2.1 17:01 的运行已经成功建立 QCTC 二次认证

真机日志曾出现：

```text
QCTC target 已发现
https://pmos.sd.sgcc.com.cn:18080/qctc-trade/dayAheadTransaction/declare/fillDataMaintenance31237
```

紧接着：

```text
qctc_route=True
bearer_present=True
sessionStorage_keys=['token', 'userInfo', 'tokenTime', 'roles']
```

同时 Cookie 摘要中还出现过：

```text
QCTC_SSO_BROWSER
XHXJG_TOKEN
X-Token
```

因此下面这些问题已经可以排除：

- 门户 Cookie 完全没拿到；
- QCTC 永远拿不到 Bearer；
- 必须依次点击第二页的每个业务菜单才能取得数据；
- 当前四类 API 路径根本不存在。

只要人工完成“现货市场 → 现货交易系统-新架构”，浏览器确实可以进入 QCTC，并获得 `sessionStorage.token`。

### 2.2 当前程序在成功后又主动破坏了上下文

17:01 已经有：

```text
origin=https://pmos.sd.sgcc.com.cn:18080
qctc_route=True
bearer_present=True
```

但 `ensure_qctc_context()` 随后因为当前 path 不是 `forecast10424`，又执行：

```text
Page.navigate(qctc_forecast_page)
```

随后页面变成：

```text
https://pmos.sd.sgcc.com.cn:18080/
qctc_route=False
bearer_present=True
```

这说明“为了准备 Forecast API 必须先切到 Forecast 页面”的假设是错误的。

`_browser_fetch()` 本来就支持：

```text
当前浏览器页面与 API 同源
→ 不再导航
→ 直接在当前页面执行 fetch
→ 从 sessionStorage.token 自动补 Authorization: Bearer
```

因此一旦 QCTC target 已经处于 `:18080` 且 Bearer 存在，就应该立即视为 READY，不再切页面。

---

## 3. 为什么上一版 `/psso?service=...` 直达失败

17:04 的新浏览器运行证明：

```text
Portal Cookie PASS
→ Page.navigate(/psso?service=...SSOLogin)
→ 页面短暂停留 /psso
→ 回到 /#/dashboard
→ bearer_present=False
```

所以：

```text
直接访问门户 /psso?service=...
```

并不等于人工点击菜单。

这不是推测，而是已经由真机日志验证。

---

## 4. 门户前端真实人工点击逻辑

历史 HAR 中的门户菜单树明确给出：

```text
菜单：现货交易系统-新架构
menuId=1595696533569933312
menuPath=/psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin
```

进一步查看门户 `app.*.js` 后，真正逻辑已经还原。

门户处理 `/psso?service=...` 时，并不是请求 `/psso` 本身，而是：

```text
1. 从菜单 URL 中解析 service
2. 获取当前门户的 SSO ticket
3. 构造：service + ?ticket=<ticket>
4. window.open(最终 URL)
```

对应 QCTC 实际目标：

```text
https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin?ticket=<SSO ticket>
```

HAR 还明确记录门户调用：

```text
POST https://pmos.sd.sgcc.com.cn/px-common-authcenter/sso/token
body={}
```

返回结构：

```json
{
  "status": 0,
  "message": "Success",
  "data": "<SSO ticket>"
}
```

`data` 是敏感短期票据，禁止写日志、report、raw 或配置文件。

`GatewayConfig/pageClick` 只是门户点击统计记录，且抓包里业务体甚至可能返回 401；它不是完成 QCTC SSO 的必要条件，不需要复现。

---

## 5. 正确的自动 QCTC SSO 流程

目标流程：

```text
门户登录完成
↓
auth_cookie PASS
↓
扫描 CDP target
↓
如果已经存在“:18080 + Bearer”上下文
    → 直接 QCTC_CONTEXT_READY
    → 不导航任何业务页面
否则
    ↓
在 portal target 中取得 SSO ticket
    ↓
打开：
https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin?ticket=...
    ↓
轮询 CDP /json
    ↓
发现 QCTC target
    ↓
切换 websocket target
    ↓
等待 sessionStorage.token
    ↓
QCTC_CONTEXT_READY
    ↓
直接在当前同源页面 fetch 四类 API
```

---

## 6. 修复 A：已有 Bearer 时绝对不要再导航 Forecast 页面

### 6.1 READY 条件调整

当前 READY 条件过强，大致要求：

```text
qctc_route=True
origin=:18080
ready=complete
bearer_present=True
```

应调整核心判定为：

```text
origin == https://pmos.sd.sgcc.com.cn:18080
AND
bearer_present == True
AND
ready in {interactive, complete}  # complete 最佳
```

`qctc_route` 可以作为诊断字段，但不要再作为唯一放行条件。

原因：`sessionStorage` 是 origin 级浏览上下文数据；对后续 `fetch` 而言，真正必要的是同源 + token，不是当前 Vue route 必须等于某个业务 path。

### 6.2 删除成功后的强制业务页导航

当前类似逻辑：

```python
if bearer_ready:
    if current_path != forecast_path:
        Page.navigate(forecast_page)
        ...
```

应改为：

```python
if qctc_origin and bearer_present:
    report QCTC_CONTEXT_READY
    return True
```

不要为了 ForecastData 再切页面。

后续 `_browser_fetch()` 已经会：

```text
选择 QCTC 页面
同源则直接 fetch
从 sessionStorage.token 添加 Authorization
```

因此这一处删除导航是最小、也是最重要的修复。

---

## 7. 修复 B：复现门户真实 ticket SSO，而不是导航 `/psso`

### 7.1 不再把下面 URL 当作最终导航 URL

错误做法：

```text
https://pmos.sd.sgcc.com.cn/psso?service=https://.../qctc/admin/sdsso/SSOLogin
```

该 URL 只是门户菜单的“描述型 URL”。

真正的新系统入口必须带 ticket：

```text
https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin?ticket=<ticket>
```

### 7.2 推荐在 portal 浏览器上下文内完成 ticket 获取

新增一个很小的 helper，例如：

```text
_open_qctc_via_portal_ticket(cdp)
```

推荐全部在浏览器 JS 中完成，避免 ticket 回到 Python 层。

建议 CDP `Runtime.evaluate` 使用：

```text
awaitPromise=True
returnByValue=True
userGesture=True
```

JS 逻辑：

```javascript
(async () => {
  const blank = window.open('about:blank', '_blank');

  const response = await fetch('/px-common-authcenter/sso/token', {
    method: 'POST',
    credentials: 'include',
    cache: 'no-store',
    headers: {
      'Accept': 'application/json, text/plain, */*',
      'Content-Type': 'application/json;charset=UTF-8',
      'ClientTag': 'OUTNET_BROWSE'
    },
    body: '{}'
  });

  const payload = await response.json();
  if (!response.ok || payload.status !== 0 || !payload.data) {
    if (blank) blank.close();
    return {
      ok: false,
      http_status: response.status,
      business_status: payload && payload.status,
      message: String(payload && payload.message || '').slice(0, 100)
    };
  }

  const target =
    'https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin' +
    '?ticket=' + encodeURIComponent(payload.data);

  if (blank) {
    blank.location.href = target;
  } else {
    window.location.href = target;
  }

  return {
    ok: true,
    opened_new_window: !!blank,
    http_status: response.status,
    business_status: payload.status
  };
})()
```

注意：返回 Python 的结果中绝对不能包含 `payload.data`。

这里先同步打开 `about:blank`，再异步请求 ticket，是为了降低异步 `window.open` 被浏览器 popup blocker 拦截的概率；`userGesture=True` 再进一步提高成功率。

若无法新开窗口，就原标签页导航 `SSOLogin?ticket=...`，这仍然比访问 `/psso?service=...` 正确，因为真正的 ticket 已经携带。

### 7.3 继续保留 target 扫描

调用上述 helper 后继续轮询：

```text
http://127.0.0.1:<debug_port>/json
```

识别：

```text
pmos.sd.sgcc.com.cn:18080
```

优先 URL path 包含：

```text
/qctc/
/qctc-trade/
```

找到后切换 `_CdpClient`。

不要假设一定同 tab，也不要假设一定新 tab。

---

## 8. Bearer 的判定与 target 选择

### 8.1 Bearer 判定

继续读取以下存储键：

```text
token
access_token
accessToken
id_token
```

只记录：

```text
bearer_present=True/False
session_storage_keys=[...]
```

绝对不记录 token 内容。

### 8.2 如果已有成功 QCTC 页面

像 17:01 的：

```text
/qctc-trade/dayAheadTransaction/declare/fillDataMaintenance31237
```

已经完全够用。

不要切到：

```text
informationDisclosure/forecast10424
```

因为 `_browser_fetch()` 只需要同源文档，不需要页面 UI 与 API 一一对应。

---

## 9. 后续 API 如何工作

保持现有接口完全不变：

```text
informationDisclosure/ForecastData/getLoadData
informationDisclosure/RealityData/getLoadData
informationDisclosure/RealityTmpData/getLoadData
trade/DaJyjgfbPlantQuery/getDetail96
YxJyjgfbPlantQuery/getDetail96
YxJyjgfbPlantQueryTmp/getDetail96
```

`_browser_fetch()` 已经会在浏览器中：

```text
credentials='include'
+
读取 sessionStorage.token
+
Authorization: Bearer <token>
```

这部分不要重写。

真机已经证明无 Bearer 时 API 会明确返回：

```text
HTTP 401
code=2
msg=登录信息失效！
Full authentication is required to access this resource
```

因此当前首要目标不是改字段，而是让 API 发出时 Bearer 保持有效。

---

## 10. 修复 C：没有 Bearer 时不要继续刷几十天 401

上一轮在 QCTC 上下文失败以后继续跑了多个日期，每天四类接口重复返回 401，浪费调试时间。

建议调试/生产都增加快速失败：

```text
SSO 自动尝试失败
+ DOM fallback 失败
+ 最终 bearer_present=False
→ qctc_context FAIL
→ 本轮停止进入 dates 循环
```

或者至少在第一个核心 QCTC API 返回明确认证型 401：

```text
code=2
Full authentication is required
```

时立即终止后续日期，记录：

```text
QCTC_AUTH_REJECTED
```

不要再继续 31 天 × 多接口的重复失败。

---

## 11. UKey 配置设计

当前新版自动处理器读取：

```text
PMOS_UKEY_PIN 环境变量
>
config.json:ukey_pin
```

推荐公司机真实配置同时保留新旧字段以兼容：

```json
{
  "pin_handler": "windows",
  "pin_env": "PMOS_UKEY_PIN",
  "ukey_pin_env": "PMOS_UKEY_PIN",
  "ukey_pin": "<现场6位PIN>",
  "ukey_auto_submit": true,
  "ukey_window_title": "验证UKey用户口令",
  "pin_submit_mode": "click"
}
```

真实 PIN 不写入：

```text
config.example.json
README
Git
日志
report.json
```

如果使用环境变量，`ukey_pin` 可以留空。

只有已经正确提供 PIN 后仍出现：

```text
pin.window_no_edit
pin.settext_failed
pin.window_multiple_edits
```

才考虑修改 `handlers.py`。

---

## 12. 当前配置中的旧字段说明

`browser_service_url=https://pmos.sd.sgcc.com.cn/#/dashboard` 当前仍可保留，因为第一层门户登录已经稳定成功。

不要再试图让 `browser_service_url` 同时承担 QCTC 二次认证职责。

认证明确分成：

```text
第一层：Portal 登录
目标：dashboard + Portal Cookie

第二层：QCTC Ticket SSO
目标：sso/token → SSOLogin?ticket=... → sessionStorage.token
```

`trade_entry` 建议统一为最终版：

```text
DaJyjgfbPlantQuery.do?appkey=187
```

不要再使用旧 `DaJyjgfbPlantFirQuery` 作为当前最终版入口配置。

---

## 13. 建议新增日志

只新增安全诊断：

```text
QCTC_EXISTING_CONTEXT_READY
QCTC_TICKET_REQUEST_START
QCTC_TICKET_REQUEST_OK
QCTC_TICKET_REQUEST_FAIL
QCTC_SSO_WINDOW_OPENED
QCTC_TARGET_FOUND
QCTC_STORAGE_STATE
QCTC_CONTEXT_READY
QCTC_AUTH_REJECTED
```

允许记录：

```text
HTTP status
business status
是否打开新窗口
target URL/path（不能含 ticket）
bearer_present
storage key 名
elapsed
```

禁止记录：

```text
SSO ticket
Bearer token
Admin-Token
X-Ticket
UKey PIN
完整 Cookie
```

注意：任何包含 `?ticket=` 的 URL 在写日志前必须剥离 query 或直接只记录 path。

---

## 14. 最小修改边界

本轮原则上只修改：

```text
scripts/crawler/collect/crawl.py
scripts/crawler/collect/crawl_96_local.py  # BUILD_VERSION
```

原则上不要修改：

```text
scripts/crawler/auth/auto_crawler/state_machine.py
scripts/crawler/auth/auto_crawler/page.py
scripts/crawler/auth/auto_crawler/browser.py
scripts/crawler/auth/auto_crawler/handlers.py
scripts/crawler/sync_db/*
```

禁止为了本任务修改：

```text
QCTC_FORECAST_MAP
QCTC_ACTUAL_MAP
DaJyjgfbPlantQuery 字段解析
YxJyjgfbPlantQuery 字段解析
96点 period 映射
CSV 结构
DB schema
DB upsert
```

---

## 15. 本地测试要求

至少增加/补充以下测试：

1. 已有 QCTC target + bearer 时，`ensure_qctc_context()` 直接返回 True，且不调用 `Page.navigate(forecast10424)`；
2. Portal ticket helper 成功时，只返回 `ok/status`，Python 侧看不到 ticket 内容；
3. ticket 请求失败时不会打开 QCTC target；
4. 新 target 出现时能切换 websocket；
5. `qctc_route=False` 但 `origin=:18080 + bearer=True` 时仍可判定上下文可用；
6. Bearer 缺失/认证 401 时能快速失败，不继续遍历大量日期；
7. 原有 28 个 crawler/auth 测试全部继续通过。

---

## 16. 新 EXE 真机验收流程

第一次不要跑历史全量。

建议：

```cmd
crawl_96_auto_v3.exe --date 2026-09-15 --force --no-db-upload
```

期望日志：

```text
pin.submitted
↓
auth_cookie PASS
↓
（若已有QCTC上下文）
QCTC_EXISTING_CONTEXT_READY

或

QCTC_TICKET_REQUEST_OK
↓
QCTC_SSO_WINDOW_OPENED
↓
QCTC_TARGET_FOUND
↓
QCTC_STORAGE_STATE bearer_present=True
↓
QCTC_CONTEXT_READY
↓
浏览器 fetch 复用同源页面
↓
ForecastData HTTP 200
RealityData HTTP 200
DaJyjgfbPlantQuery HTTP 200
YxJyjgfbPlantQuery HTTP 200/业务有效返回
```

最关键验收点：

```text
QCTC_CONTEXT_READY 之后不应该再出现“为了进入 forecast10424 而主动导航”的日志。
```

如果 API 仍然 401，需要记录：

```text
QCTC target path
bearer_present
storage key names
HTTP status
```

但不要记录 ticket/token 值。

---

## 17. 回滚策略

重新打包前备份当前 EXE：

```text
crawl_96_auto_v3_before_ticket_sso.exe
```

如果新 ticket SSO 有问题，可以直接换回旧 EXE，不影响真实配置和历史输出。

不要覆盖：

```text
config.json
db_config.json
output_96
```

---

## 18. 最终结论

当前真正需要修的不是整个登录系统，也不是业务字段，而是两个小点：

```text
A. 已有 :18080 + Bearer 时立即 READY，绝不再强制跳业务页面；
B. 没有 QCTC 上下文时，严格复现门户真实逻辑：
   /sso/token → SSOLogin?ticket=... → 新/当前 target → sessionStorage.token。
```

这两点都已经有真机日志和门户前端 JS 作为直接证据，下一版不应该再使用“直接导航 `/psso?service=...`”作为主方案。
