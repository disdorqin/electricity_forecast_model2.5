from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .browser import CdpSession


class PageState(str, Enum):
    LOADING = "loading"
    LOGIN_READY = "login_ready"
    SLIDER = "slider"
    CERTIFICATE = "certificate"
    AUTHENTICATING = "authenticating"
    GATEWAY_ERROR = "gateway_error"
    LOGGED_IN = "logged_in"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PageSnapshot:
    state: PageState
    url: str
    detail: str = ""
    login_form: bool = False
    slider_visible: bool = False
    certificate_visible: bool = False
    # [AUX-V1-r11f] 三层防护检测位。均为向后兼容的默认字段：96 主爬虫不读取，
    # 行为零变化；AUX 侧据此做「会话失效刷新」与「网络不可达降级」判定。
    session_expired: bool = False
    network_unreachable: bool = False


# [AUX-V1-r11f] 判据 D1：门户实际文案（人工 F5 能恢复的「网页已失效」场景）。
SESSION_EXPIRED_MARKERS = (
    "已失效", "会话已过期", "会话超时", "登录已失效", "登录信息失效",
    "登录状态已失效", "请重新登录", "重新登录", "身份验证已过期", "用户未登录",
)

# [AUX-V1-r11f] 判据 N2：Chrome/Edge 原生错误页文案。
NETWORK_UNREACHABLE_MARKERS = (
    "无法访问此网站", "网页无法访问", "找不到该网页", "没有互联网连接",
    "网络连接中断", "err_internet_disconnected", "err_name_not_resolved",
    "err_connection_timed_out", "err_connection_refused", "err_network_changed",
    "err_address_unreachable", "err_proxy_connection_failed",
)

# [AUX-V1-r11f] 判据 N1：浏览器自身错误页 / 空白页的 URL 前缀。
_UNREACHABLE_URL_PREFIXES = ("chrome-error://", "about:blank", "edge-error://")


def _text_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    lowered = (text or "").lower()
    return tuple(marker for marker in markers if marker.lower() in lowered)


def _url_unreachable(url: str) -> bool:
    lowered = (url or "").lower()
    return any(lowered.startswith(prefix) for prefix in _UNREACHABLE_URL_PREFIXES)


class PmosPage:
    def __init__(self, session: CdpSession):
        self.session = session

    def snapshot(self) -> PageSnapshot:
        data = self.session.evaluate("""(() => {
          const text = (document.body?.innerText || '').replace(/\\s+/g, ' ');
          const visible = el => {
            if (!el || !(el.offsetWidth || el.offsetHeight || el.getClientRects().length)) return false;
            const style = getComputedStyle(el), rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity || 1) > 0.01
              && rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0
              && rect.top < innerHeight && rect.left < innerWidth;
          };
          const inputs = [...document.querySelectorAll('input')];
          const hasPassword = inputs.some(x => visible(x) && (x.type === 'password' || /密码/.test(x.placeholder || '')));
          const norm = value => (value || '').replace(/\\s+/g, '');
          // PMOS 登录页会预置隐藏的滑块 DOM；只接受当前视口中提示文字本身的可见节点。
          const slider = [...document.querySelectorAll('*')].some(x => {
            const rect = x.getBoundingClientRect();
            return visible(x) && norm(x.innerText) === '向右滑动完成验证'
              && rect.width < 600 && rect.height < 160;
          });
          // Element UI 将真实 radio input 设为透明，必须从可见标签识别证书类型。
          const cfca = [...document.querySelectorAll('*')].some(x => visible(x)
            && norm(x.innerText) === 'CFCA');
          return {url: location.href, ready: document.readyState, hasPassword, slider, cfca,
            text: text.slice(0, 500)};
        })()""") or {}
        url = str(data.get("url") or "")
        body_text = str(data.get("text") or "")
        # [AUX-V1-r11f] 判据 N1/N2：网络不可达必须早于 PMOS 域名判定。浏览器原生
        # 错误页的 URL 是 chrome-error://，永远不含 pmos 域名；若按原顺序判定会被
        # 归类成「等待 PMOS 导航」，从而空等到 login_timeout 才失败。
        unreachable_url = _url_unreachable(url)
        unreachable_text = _text_markers(body_text, NETWORK_UNREACHABLE_MARKERS)
        if unreachable_url or unreachable_text:
            return PageSnapshot(
                PageState.LOADING, url,
                "network_unreachable url_marker=%s text_markers=%s"
                % (unreachable_url, list(unreachable_text)),
                network_unreachable=True,
            )
        if "pmos.sd.sgcc.com.cn" not in url.lower():
            return PageSnapshot(PageState.LOADING, url, "waiting_for_pmos_navigation")
        # [AUX-V1-r11f] 判据 D1：门户「网页已失效」文案（人工按 F5 即可恢复的场景）。
        # 这里只打标记、绝不改变 state 取值，保证 96 主爬虫的状态机分支完全不变；
        # AUX 侧读取该标记后触发等价人工 F5 的刷新恢复。
        expired_markers = _text_markers(body_text, SESSION_EXPIRED_MARKERS)
        session_expired = bool(expired_markers)
        detail = "login_form=%s slider_visible=%s certificate_visible=%s" % (
            bool(data.get("hasPassword")), bool(data.get("slider")), bool(data.get("cfca")),
        )
        if session_expired:
            detail = "%s session_expired_markers=%s" % (detail, list(expired_markers))
        # 认证服务会把最初的 service 参数带回旧版 :18080 交易入口。该入口在
        # 部分公司网络中返回 nginx 502；这不是认证失败，回退一页即可回到门户。
        if "502 bad gateway" in body_text.lower():
            return PageSnapshot(PageState.GATEWAY_ERROR, url, "gateway_502",
                                session_expired=session_expired)
        # 新版门户的 dashboard 即已登录主页，不需要再跳转旧 :18080 交易入口。
        if "#/dashboard" in url.lower() and not data.get("hasPassword"):
            return PageSnapshot(PageState.LOGGED_IN, url, "portal_dashboard",
                                session_expired=session_expired)
        # 已有 DevTools 浏览器可能停留在旧版 ZCQ 首页。它虽然不是新的
        # /#/dashboard URL，但页面已经显示企业名称和业务菜单，继续等待会
        # 让状态机无意义地阻塞到 600 秒；此时同样可以直接读取 Cookie，
        # 后续 QCTC 数据请求会自行导航到正确的信息披露页面。
        old_logged_in = (
            "/zcq/main/index.do" in url.lower()
            and not data.get("hasPassword")
            and any(marker in body_text for marker in ("返回首页", "退出", "常用菜单"))
        )
        if old_logged_in:
            return PageSnapshot(PageState.LOGGED_IN, url, "legacy_zcq_logged_in",
                                session_expired=session_expired)
        if ":18080/trade" in url.lower() and "%2ftrade" not in url.lower():
            return PageSnapshot(PageState.LOGGED_IN, url, "trade_url",
                                certificate_visible=bool(data.get("cfca")),
                                session_expired=session_expired)
        if data.get("cfca"):
            return PageSnapshot(PageState.CERTIFICATE, url, detail, bool(data.get("hasPassword")),
                                bool(data.get("slider")), True, session_expired=session_expired)
        if data.get("hasPassword"):
            # 登录表单和隐藏滑块模板可能同时存在；认证状态机先处理表单。
            return PageSnapshot(PageState.LOGIN_READY, url, detail, True, bool(data.get("slider")),
                                session_expired=session_expired)
        if data.get("slider"):
            return PageSnapshot(PageState.SLIDER, url, detail, slider_visible=True,
                                session_expired=session_expired)
        if data.get("ready") != "complete":
            return PageSnapshot(PageState.LOADING, url, detail, session_expired=session_expired)
        return PageSnapshot(PageState.UNKNOWN, url, detail + " text=" + body_text[:120],
                            session_expired=session_expired)

    def recover_from_gateway_error(self) -> bool:
        """仅在已确认的 nginx 502 页执行一次浏览器后退，不重放登录或证书请求。"""
        result = self.session.evaluate("""(() => {
          if (!/502\\s+Bad\\s+Gateway/i.test(document.body?.innerText || '')) return false;
          if (history.length <= 1) return false;
          history.back();
          return true;
        })()""")
        return bool(result)

    def submit_login(self, username: str, password: str) -> dict:
        if not username or not password:
            raise RuntimeError("未从环境变量读取到 PMOS 账号和密码")
        expression = """(() => {
          const username = %s, password = %s;
          const visible = el => !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
          const setValue = (el, value) => {
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
            setter ? setter.call(el, value) : (el.value = value);
            for (const name of ['input','change','blur']) el.dispatchEvent(new Event(name, {bubbles:true}));
          };
          const inputs = [...document.querySelectorAll('input')].filter(visible);
          const pass = inputs.find(x => x.type === 'password' || /密码/.test(x.placeholder || ''));
          const user = inputs.find(x => x !== pass && /账号|用户|user|account/i.test(`${x.placeholder} ${x.name}`))
            || inputs.find(x => x !== pass && x.type !== 'password');
          if (!user || !pass) return {ok:false, reason:'inputs_not_ready'};
          setValue(user, username); setValue(pass, password);
          const norm = s => (s || '').replace(/\\s+/g, '');
          const buttons = [...document.querySelectorAll('button,[role="button"],input[type="submit"]')].filter(visible);
          const button = buttons.find(x => norm(x.innerText || x.value) === '登录');
          if (!button || button.disabled) return {ok:false, reason:'button_not_ready'};
          button.focus();
          // 使用浏览器原生 click() 触发 Vue/Element 的真实 click handler。
          // 仅 dispatchEvent 在部分 PMOS 页面上会返回“已点击”，但不会提交登录请求。
          button.click();
          return {ok:true, reason:'submitted', native_click:true,
            user:user.value === username, pass:pass.value.length > 0};
        })()""" % (json.dumps(username), json.dumps(password))
        return self.session.evaluate(expression) or {}

    def select_cfca_and_verify(self) -> dict:
        result = self.session.evaluate("""(() => {
          const visible = el => {
            if (!el || !(el.offsetWidth || el.offsetHeight || el.getClientRects().length)) return false;
            const style = getComputedStyle(el), rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity || 1) > 0.01
              && rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0
              && rect.top < innerHeight && rect.left < innerWidth;
          };
          const norm = s => (s || '').replace(/\\s+/g, '');
          const cfcaText = [...document.querySelectorAll('*')].find(x => visible(x) && norm(x.innerText) === 'CFCA');
          if (!cfcaText) return {ok:false, reason:'cfca_label_missing'};
          const radio = cfcaText.closest('label, .el-radio, [role="radio"]') || cfcaText.parentElement;
          if (!radio) return {ok:false, reason:'cfca_control_missing'};
          radio.click();
          const input = radio.querySelector('input[type="radio"]');
          if (input) {
            input.dispatchEvent(new Event('input', {bubbles:true}));
            input.dispatchEvent(new Event('change', {bubbles:true}));
          }
          const dialog = cfcaText.closest('.el-dialog, [role="dialog"]') || document;
          const buttons = [...dialog.querySelectorAll('button, [role="button"], .el-button')].filter(visible);
          const verify = buttons.find(x => norm(x.innerText || x.value) === '验证');
          if (!verify || verify.disabled) return {ok:false, reason:'verify_missing'};
          verify.dispatchEvent(new MouseEvent('click', {bubbles:true, cancelable:true, view:window}));
          return {ok:true, reason:'cfca_verified'};
        })()""") or {}
        return dict(result)
