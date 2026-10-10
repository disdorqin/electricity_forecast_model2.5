# 历史方案：PMOS SSO 二次认证与 UKey（凝练版）

> 凝练日期：2026-09-27｜原始文档已归档至 `scripts/crawler/archive/design_20260916/`
> 原始文件（4 份，共约 38 KB）：
> `QCTC_SSO_TICKET_FIX_DESIGN.md`、`QCTC_SSO_UKEY_MINIMAL_DESIGN.md`（设计稿）
> `QCTC_SSO_TICKET_FIX_PROMPT.md`、`QCTC_SSO_UKEY_AI_PROMPT.md`（**给 AI 的任务提示词，任务已完成，仅作留档**）
>
> 本文只保留**当前仍有参考价值**的机制结论，并明确标注**已被后续推翻**的部分。

---

## 一、仍然有效的结论

### 1.1 门户 → QCTC 的二次跳转：要复现「导航行为」，不要直连

真人从门户菜单进入 QCTC 时，实际发生的是：

```
菜单 menuPath = /psso?service=https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin
   → 浏览器取得 SSO ticket
   → 打开 SSOLogin?ticket=...
   → QCTC (Vue SPA) 自建 sessionStorage.token
```

**关键洞察（HAR 实测，至今有效）**：
- `POST /px-common-authcenter/sso/token` 在 HAR 里 HTTP 层返回 200，但业务体曾返回 `status=401`；
  它更像**点击审计 / 网关记录**，**不应被当作 SSO 成功的前置条件**。
- 真正必须复现的是：**浏览器通过 `/psso?service=...SSOLogin` 进入 QCTC 的导航行为**，
  并以最终 QCTC 上下文（以及业务接口真实返回码）为准。

> 对应到方法论：放行判据必须是业务接口的真实返回码，而不是某个中间接口的 200。

### 1.2 UKey PIN 的配置方式

```jsonc
// config.json
{ "ukey_pin": "",            // 允许为空
  "pin_env": "PMOS_UKEY_PIN" // 真键名（不是 ukey_pin_env / ukey_auto_submit，那两个在源码里零引用）
}
```

- PIN 实际来源是 `resolved_pin = os.environ.get(pin_env, ukey_pin)`；
- 因此 **config 里 `ukey_pin` 为空、靠环境变量是正常设计，不是故障**；
- 若未配置，日志会出现 `pin.handler_fallback=manual reason=pin_not_configured env=PMOS_UKEY_PIN`；
- 日志 `pin.submitted mode=click` **只有 `WindowsPinHandler` 会打** → 出现即证明 PIN 自动输入成功。

### 1.3 临时 profile 是死路（后续被反复验证）

UKey 原生弹窗依赖 UKey 客户端在该 profile 下完成初始化。
换用 `*_fallback_<ts>` / `*_recovery_*` 等**临时 profile** 时，弹窗不会出现 → 卡死在 CERTIFICATE。
（2026-09-25、2026-09-27 两次复盘均印证。当前 AUX 三层轮换已强制 `browser_fallback=False` 以规避。）

---

## 二、已被推翻 / 已过时的结论

| 原结论 | 现状 |
|---|---|
| 「没有 Bearer 就一定失败」 | ❌ **已推翻**。日志中 `bearer_present: True` 出现 0 次，而有过 `fetch 成功 + 96 行写库`。502 实际来自旧 `:18080/trade/*.do`；`status=0` 来自页面被弹回门户后的跨源 fetch |
| Bearer 作为放行前置条件 | ❌ 已降级为**观测项**（`QCTC_CONTEXT_SOFT_MISSING` / stage=WARN），采集继续，由接口返回码判定 |
| 「已有 Bearer 时禁止再导航业务页面」这一修复的**紧迫性** | ⚠ 前提不成立（Bearer 本就不是必需），该修复不再是关键路径 |

---

## 三、对当前的指导

1. **不要再用 Bearer 做前置门禁**（它只是观测项）；判定一律看业务接口返回码。
2. 认证状态判定的**优先级**：业务接口返回码 > 页面文本 › Cookie/URL 表象
   （Cookie 存在但后端 session 已过期是最经典的坑，已由 `resilience.probe_session_from_response` 覆盖）。
3. 需要重新导航进入 QCTC 时，走 `/psso?service=...SSOLogin` 的**导航语义**，
   不要把 `sso/token` 当成功判据。
4. PIN 问题先查 `PMOS_UKEY_PIN` 环境变量与 `pin.submitted` 日志，再看 config。
