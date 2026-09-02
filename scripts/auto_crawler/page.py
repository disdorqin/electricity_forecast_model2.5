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
        if "pmos.sd.sgcc.com.cn" not in url.lower():
            return PageSnapshot(PageState.LOADING, url, "waiting_for_pmos_navigation")
        detail = "login_form=%s slider_visible=%s certificate_visible=%s" % (
            bool(data.get("hasPassword")), bool(data.get("slider")), bool(data.get("cfca")),
        )
        # 认证服务会把最初的 service 参数带回旧版 :18080 交易入口。该入口在
        # 部分公司网络中返回 nginx 502；这不是认证失败，回退一页即可回到门户。
        if "502 bad gateway" in str(data.get("text") or "").lower():
            return PageSnapshot(PageState.GATEWAY_ERROR, url, "gateway_502")
        if ":18080/trade" in url.lower() and "%2ftrade" not in url.lower():
            return PageSnapshot(PageState.LOGGED_IN, url, "trade_url", certificate_visible=bool(data.get("cfca")))
        if data.get("cfca"):
            return PageSnapshot(PageState.CERTIFICATE, url, detail, bool(data.get("hasPassword")),
                                bool(data.get("slider")), True)
        if data.get("hasPassword"):
            # 登录表单和隐藏滑块模板可能同时存在；认证状态机先处理表单。
            return PageSnapshot(PageState.LOGIN_READY, url, detail, True, bool(data.get("slider")))
        if data.get("slider"):
            return PageSnapshot(PageState.SLIDER, url, detail, slider_visible=True)
        if data.get("ready") != "complete":
            return PageSnapshot(PageState.LOADING, url, detail)
        return PageSnapshot(PageState.UNKNOWN, url, detail + " text=" + str(data.get("text") or "")[:120])

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
          for (const kind of ['pointerdown','mousedown','pointerup','mouseup','click']) {
            button.dispatchEvent(new MouseEvent(kind, {bubbles:true, cancelable:true, view:window}));
          }
          return {ok:true, reason:'submitted', user:user.value === username, pass:pass.value.length > 0};
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
