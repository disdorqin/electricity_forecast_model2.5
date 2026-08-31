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
    LOGGED_IN = "logged_in"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PageSnapshot:
    state: PageState
    url: str
    detail: str = ""


class PmosPage:
    def __init__(self, session: CdpSession):
        self.session = session

    def snapshot(self) -> PageSnapshot:
        data = self.session.evaluate("""(() => {
          const text = (document.body?.innerText || '').replace(/\\s+/g, ' ');
          const visible = el => !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
          const inputs = [...document.querySelectorAll('input')];
          const hasPassword = inputs.some(x => visible(x) && (x.type === 'password' || /密码/.test(x.placeholder || '')));
          const slider = [...document.querySelectorAll('*')].some(x => visible(x) && /向右滑动完成验证/.test(x.textContent || ''));
          const cfca = !!document.querySelector('input[type="radio"][value="CFCA"]');
          return {url: location.href, ready: document.readyState, hasPassword, slider, cfca,
            text: text.slice(0, 500)};
        })()""") or {}
        url = str(data.get("url") or "")
        if ":18080/trade" in url.lower() and "%2ftrade" not in url.lower():
            return PageSnapshot(PageState.LOGGED_IN, url, "trade_url")
        if data.get("slider"):
            return PageSnapshot(PageState.SLIDER, url)
        if data.get("cfca"):
            return PageSnapshot(PageState.CERTIFICATE, url)
        if data.get("hasPassword"):
            return PageSnapshot(PageState.LOGIN_READY, url)
        if data.get("ready") != "complete":
            return PageSnapshot(PageState.LOADING, url)
        return PageSnapshot(PageState.UNKNOWN, url, str(data.get("text") or "")[:120])

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

    def select_cfca_and_verify(self) -> bool:
        result = self.session.evaluate("""(() => {
          const radio = document.querySelector('input[type="radio"][value="CFCA"]');
          if (!radio) return {ok:false, reason:'cfca_missing'};
          const label = radio.closest('label') || radio.parentElement;
          (label || radio).click();
          radio.dispatchEvent(new Event('change', {bubbles:true}));
          const norm = s => (s || '').replace(/\\s+/g, '');
          const buttons = [...document.querySelectorAll('button')].filter(x => x.offsetParent !== null);
          const verify = buttons.find(x => norm(x.innerText) === '验证');
          if (!verify || verify.disabled) return {ok:false, reason:'verify_missing'};
          verify.click();
          return {ok:true, reason:'cfca_verified'};
        })()""") or {}
        return bool(result.get("ok"))

    def probe_authenticated(self, trade_base: str, paths: tuple[str, ...]) -> bool:
        """在浏览器会话内检查两个真实交易入口，规避打包 Python 的 TLS 差异。"""
        urls = [trade_base.rstrip("/") + "/" + path.lstrip("/") for path in paths]
        expression = """(async () => {
          const urls = %s;
          const bad = /captcha|loginform|top[.]location[.]href|window[.]location[.]href|请输入账号/i;
          const results = [];
          for (const url of urls) {
            try {
              const response = await fetch(url, {credentials:'include', cache:'no-store'});
              const text = (await response.text()).slice(0, 1600);
              results.push({ok:response.status === 200 && !bad.test(text), status:response.status});
            } catch (error) {
              results.push({ok:false, status:0});
            }
          }
          return {ok:results.length > 0 && results.every(x => x.ok), results};
        })()""" % json.dumps(urls)
        result = self.session.evaluate(expression, await_promise=True, timeout=30) or {}
        return bool(result.get("ok"))
