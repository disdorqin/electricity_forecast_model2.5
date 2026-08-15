#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PMOS 端到端自动爬虫 v2（带详细日志 + 按HAR正确顺序）
====================================================

登录流程（严格按浏览器HAR还原）：
  1. seed_session          访问登录页/交易页，取得cookie
  2. captcha/get           获取滑块验证码图片
  3. secureKey/get         获取SM2公钥（密码加密用）
  4. captcha/check         提交滑块校验（不带SKIP）
  5. captcha/get           刷新验证码（第二次）
  6. getSecureKey          获取信封SM2公钥
  7. encryption/login      加密登录（带SKIP）
  8. captcha/get           刷新验证码（第三次）
  9. getSecureKey          再次获取信封公钥
  10. encryption/verify    验证登录结果（带SKIP）

日志输出：
  - output/auto_crawler_v2_YYYYMMDD.log          主日志
  - output/auto_crawler_v2_detail_YYYYMMDD.log   完整HTTP请求/响应
  - output/captcha_images/                       每次登录的验证码图片
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import secrets
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_FROZEN = getattr(sys, "frozen", False)
if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).resolve().parents[2]
    for _p in (str(BASE_DIR), str(BASE_DIR / "scripts" / "crawler")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

import requests
import numpy as np
from PIL import Image

try:
    from gmssl import sm2 as _gm_sm2
    from gmssl import sm3 as _gm_sm3
    from gmssl.sm4 import CryptSM4, SM4_ENCRYPT, SM4_DECRYPT
except ImportError as e:
    print(f"[FATAL] gmssl 未安装: {e}")
    print("请运行: pip install gmssl")
    sys.exit(1)

try:
    from Crypto.Cipher import AES as _AES
except ImportError as e:
    print(f"[FATAL] pycryptodome 未安装: {e}")
    print("请运行: pip install pycryptodome")
    sys.exit(1)

try:
    import pymysql
except ImportError:
    pymysql = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

CRAWLER_DIR = BASE_DIR if _FROZEN else BASE_DIR / "scripts" / "crawler"
OUTPUT_DIR = BASE_DIR / "outputs" / "crawl"
CAPTCHA_DIR = OUTPUT_DIR / "captcha_images"
CONFIG_PATH = CRAWLER_DIR / "config.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CAPTCHA_DIR.mkdir(parents=True, exist_ok=True)

today_str = date.today().strftime("%Y%m%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pmos_v2")

_fh = logging.FileHandler(
    str(OUTPUT_DIR / f"auto_crawler_v2_{today_str}.log"),
    encoding="utf-8", mode="a",
)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(_fh)

_detail_fh = logging.FileHandler(
    str(OUTPUT_DIR / f"auto_crawler_v2_detail_{today_str}.log"),
    encoding="utf-8", mode="a",
)
_detail_fh.setLevel(logging.DEBUG)
_detail_fh.setFormatter(logging.Formatter("%(message)s"))
_detail_logger = logging.getLogger("pmos_v2_detail")
_detail_logger.addHandler(_detail_fh)

AUTH_HOST = "https://pmos.sd.sgcc.com.cn"
TRADE_BASE = "https://pmos.sd.sgcc.com.cn:18080/trade"
CAPTCHA_IMG_W = 310
SKIP_HEADER = {"Intercept-Headers": "SKIP"}
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/json;charset=UTF-8",
}


def _log_http(direction: str, step: str, method: str = "", url: str = "",
              status: int = 0, headers: dict = None, body: Any = None):
    sep = "=" * 70
    lines = [f"\n{sep}", f"[{direction}] {step}", f"  {method} {url}" if url else ""]
    if status:
        lines.append(f"  Status: {status}")
    if headers:
        for k, v in (headers or {}).items():
            val = str(v)
            if k.lower() == "cookie" and len(val) > 80:
                val = val[:80] + "..."
            lines.append(f"  {k}: {val}")
    if body is not None:
        body_str = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)
        if len(body_str) > 1500:
            body_str = body_str[:1500] + "...[truncated]"
        lines.append(f"  Body: {body_str}")
    _detail_logger.debug("\n".join(lines))


def _save_img(name: str, data: bytes):
    if not data:
        return
    try:
        p = CAPTCHA_DIR / name
        with open(p, "wb") as f:
            f.write(data)
    except Exception:
        pass


def b64_to_bytes(s: str) -> bytes:
    if not s:
        return b""
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def sm2_encrypt(pub_key_hex: str, plain_text: str) -> str:
    data = plain_text.encode("utf-8")
    for _ in range(8):
        try:
            if pub_key_hex.startswith("04"):
                stripped = pub_key_hex[2:]
                pk = stripped if len(stripped) == 128 else pub_key_hex
            else:
                pk = pub_key_hex
            crypt = _gm_sm2.CryptSM2(public_key=pk, private_key="00" * 32, mode=1)
            ct = crypt.encrypt(data)
            if ct is None:
                continue
            return "04" + ct.hex()
        except Exception:
            continue
    raise RuntimeError("SM2加密连续失败8次")


def sm3_hex(s: str) -> str:
    data_ints = [b for b in s.encode("utf-8")]
    return _gm_sm3.sm3_hash(data_ints)


def sm4_cbc_encrypt(key_hex: str, iv_hex: str, plain_text: str) -> str:
    body = plain_text + sm3_hex(plain_text)
    data = body.encode("utf-8")
    crypt = CryptSM4()
    crypt.set_key(bytes.fromhex(key_hex), SM4_ENCRYPT)
    ct = crypt.crypt_cbc(bytes.fromhex(iv_hex), data)
    return ct.hex()


def sm4_cbc_decrypt(key_hex: str, iv_hex: str, cipher_hex: str) -> str:
    data = bytes.fromhex(cipher_hex)
    crypt = CryptSM4()
    crypt.set_key(bytes.fromhex(key_hex), SM4_DECRYPT)
    pt = crypt.crypt_cbc(bytes.fromhex(iv_hex), data)
    text = pt.decode("utf-8", errors="replace")
    if len(text) > 64:
        head, tail = text[:-64], text[-64:]
        if sm3_hex(head) == tail:
            return head
    return text


def aes_ecb_encrypt(key_str: str, plain_str: str) -> str:
    key = key_str.encode("utf-8")[:16].ljust(16, b"\0")
    data = plain_str.encode("utf-8")
    pad = 16 - (len(data) % 16)
    data = data + bytes([pad]) * pad
    cipher = _AES.new(key, _AES.MODE_ECB)
    return base64.b64encode(cipher.encrypt(data)).decode("ascii")


def detect_gap_x(original_bytes: bytes, jigsaw_bytes: bytes) -> float:
    orig = np.asarray(Image.open(BytesIO(original_bytes)).convert("RGB"), dtype=np.float64)
    jig_img = Image.open(BytesIO(jigsaw_bytes)).convert("RGBA")
    jig_arr = np.asarray(jig_img, dtype=np.float64)
    jh, jw = jig_arr.shape[:2]
    mask = jig_arr[:, :, 3] > 50
    jig_rgb = jig_arr[:, :, :3]
    oh, ow = orig.shape[:2]

    scores = []
    for x in range(0, ow - jw + 1):
        region = orig[0:jh, x:x + jw, :]
        if region.shape[1] != jw:
            continue
        diff = np.abs(region - jig_rgb) * mask[:, :, None]
        scores.append(diff.sum())

    scores_arr = np.array(scores, dtype=np.float64)
    best_x = int(np.argmin(scores_arr))

    x0 = float(best_x)
    if 1 <= best_x < len(scores_arr) - 1:
        y_m1, y_0, y_p1 = scores_arr[best_x - 1], scores_arr[best_x], scores_arr[best_x + 1]
        denom = 2.0 * (y_m1 + y_p1 - 2.0 * y_0)
        if abs(denom) > 1e-12:
            x_sub = x0 + (y_m1 - y_p1) / denom
        else:
            x_sub = x0
        x_sub = max(x0 - 1.0, min(x0 + 1.0, x_sub))
    else:
        x_sub = x0

    browser_x = max(0.0, x_sub - 1.7)
    logger.debug("[slider] best_x=%d subpixel=%.4f browser_x=%.4f", best_x, x_sub, browser_x)
    return browser_x


class PMOSAutoCrawler:

    def __init__(self, username: str, password: str, unit_id: str,
                 auth_host: str = AUTH_HOST, trade_base: str = TRADE_BASE):
        self.username = username
        self.password = password
        self.unit_id = unit_id
        self.auth_host = auth_host.rstrip("/")
        self.trade_base = trade_base.rstrip("/")

        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.session.verify = False

        self.x_ticket = f"{secrets.token_hex(32)}.{secrets.token_hex(20)}"
        self.jsessionid = secrets.token_hex(16)

        self.session.cookies.set("X-Ticket", self.x_ticket)
        self.session.cookies.set("JSESSIONID", self.jsessionid)
        self.session.cookies.set("ClientTag", "OUTNET_BROWSE")
        self.session.cookies.set("X-Token", "undefined")
        self.session.cookies.set("CurrentRoute", "/dashboard")
        self.session.cookies.set("Gray-Tag", username.encode("utf-8").hex())

        self.session.headers.update({
            "X-Ticket": "undefined",
            "X-Token": "null",
            "ClientTag": "OUTNET_BROWSE",
            "CurrentRoute": "/outNet",
            "Origin": self.auth_host,
            "Referer": self.auth_host + "/",
        })

    def _req(self, method: str, path: str, payload: dict = None,
             with_skip: bool = False, step: str = "", raw: bool = False):
        url = self.auth_host + path
        headers = {}
        if with_skip:
            headers.update(SKIP_HEADER)
        headers["X-Ticket"] = "undefined"
        headers["X-Token"] = "null"

        _log_http("REQ", step, method, url, headers=headers, body=payload)

        if method.upper() == "POST":
            resp = self.session.post(url, json=payload, headers=headers, timeout=30)
        else:
            resp = self.session.get(url, headers=headers, timeout=30)

        _log_http("RESP", step, status=resp.status_code, headers=dict(resp.headers),
                  body=resp.text[:2000] if not raw else f"<{len(resp.content)} bytes>")

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} @ {path}: {resp.text[:200]}")

        if raw:
            return resp
        return resp.json()

    def step1_seed_session(self):
        step = "1-seed_session"
        logger.info("[%s] 访问登录页/交易页，建立会话cookie...", step)
        urls = [
            (self.auth_host + "/", "登录页"),
            (self.trade_base + "/DaJyjgfbPlantQuery.do?appkey=187", "交易页"),
        ]
        for url, desc in urls:
            try:
                resp = self.session.get(url, timeout=15, verify=False, allow_redirects=True)
                logger.info("[%s] %s -> %d (final: %s)", step, desc, resp.status_code, resp.url[:80])
            except Exception as e:
                logger.warning("[%s] %s 访问失败: %s", step, desc, e)
        logger.info("[%s] 当前cookie: %s", step, self._cookie_summary())

    def step2_captcha_get(self, attempt_label: str = "first") -> dict:
        step = f"2-captcha_get_{attempt_label}"
        logger.info("[%s] 获取滑块验证码...", step)
        resp = self._req("POST", "/px-common-authcenter/auth/v2/captcha/get",
                         {"captchaType": "blockPuzzle"}, step=step)

        if resp.get("status") != 0 or not resp.get("data", {}).get("repData"):
            raise RuntimeError(f"获取验证码失败: {resp}")

        rd = resp["data"]["repData"]
        original = b64_to_bytes(rd.get("originalImageBase64", ""))
        jigsaw = b64_to_bytes(rd.get("jigsawImageBase64", ""))

        ts = datetime.now().strftime("%H%M%S")
        _save_img(f"captcha_{attempt_label}_{ts}_original.png", original)
        _save_img(f"captcha_{attempt_label}_{ts}_jigsaw.png", jigsaw)

        logger.info("[%s] secretKey=%s... token=%s...", step,
                    (rd.get("secretKey") or "")[:8], (rd.get("token") or "")[:12])
        logger.info("[%s] original=%d bytes, jigsaw=%d bytes", step, len(original), len(jigsaw))

        return {
            "secret_key": rd.get("secretKey"),
            "token": rd.get("token"),
            "original": original,
            "jigsaw": jigsaw,
        }

    def step3_secureKey_get(self) -> tuple:
        step = "3-secureKey_get"
        logger.info("[%s] 获取密码SM2公钥...", step)
        resp = self._req("POST", "/px-common-authcenter/auth/v2/secureKey/get", {}, step=step)
        if resp.get("status") != 0:
            raise RuntimeError(f"secureKey/get 失败: {resp}")
        d = resp["data"]
        logger.info("[%s] pubKey=%s... secureCode=%s", step, d["pubKey"][:20], d["secureCode"])
        return d["pubKey"], d["secureCode"]

    def step4_captcha_check(self, cap: dict, candidates: list) -> tuple:
        step = "4-captcha_check"
        logger.info("[%s] 尝试 %d 个候选位置...", step, len(candidates))
        for idx, (x, point_json, captcha_verification) in enumerate(candidates):
            logger.info("[%s] 候选 #%d: x=%d", step, idx, x)
            payload = {
                "captchaType": "blockPuzzle",
                "pointJson": point_json,
                "token": cap["token"],
            }
            resp = self._req("POST", "/px-common-authcenter/auth/v2/captcha/check",
                             payload, step=step)
            ok = bool(resp.get("data", {}).get("repData", {}).get("result"))
            if ok:
                logger.info("[%s] 候选 #%d x=%d 校验通过!", step, idx, x)
                return captcha_verification, x
            logger.info("[%s] 候选 #%d x=%d 校验失败", step, idx, x)
        raise RuntimeError("所有候选位置均校验失败")

    def step6_getSecureKey(self, attempt_label: str = "first") -> tuple:
        step = f"6-getSecureKey_{attempt_label}"
        logger.info("[%s] 获取信封SM2公钥...", step)
        resp = self._req("POST", "/px-common-authcenter/auth/v2/getSecureKey", {}, step=step)
        if resp.get("status") != 0:
            raise RuntimeError(f"getSecureKey 失败: {resp}")
        d = resp["data"]
        logger.info("[%s] secureKey=%s... secureCode=%s", step, d["secureKey"][:20], d["secureCode"])
        return d["secureKey"], d["secureCode"]

    def step7_encryption_login(self, captcha_verification: str,
                                pub_key: str, secure_code: str,
                                env_pub: str, env_secure: str) -> dict:
        step = "7-encryption_login"
        logger.info("[%s] 构造加密登录请求...", step)

        form = {
            "cookieTicketKey": secrets.token_hex(32),
            "loginName": self.username.strip(),
            "username": self.username.strip(),
            "authKey": sm2_encrypt(pub_key, self.password),
            "secureCode": secure_code,
            "dnInfo": "",
            "isCfcaLogin": False,
            "loginFrom": 1,
            "twoFactorType": False,
            "clientTag": "OUTNET_BROWSE",
            "randomCode": "ERWEFX",
            "captchaVerification": captcha_verification,
            "origin": "PHBSD",
        }
        logger.info("[%s] 表单字段: %s", step, ", ".join(form.keys()))

        a = secrets.token_hex(16)
        d = secrets.token_hex(16)
        sm2_env = sm2_encrypt(env_pub, a + "," + d)
        plain = json.dumps(form, separators=(",", ":"), ensure_ascii=False)
        sm4_env = sm4_cbc_encrypt(a, d, plain)

        body = {"authKey": {"sm4": sm4_env, "sm2": sm2_env, "secureCode": env_secure}}

        logger.info("[%s] 发送加密登录请求...", step)
        resp_text = self._req("POST", "/px-common-authcenter/auth/v2/encryption/login",
                              body, with_skip=True, step=step, raw=True).text

        logger.info("[%s] 原始响应长度: %d", step, len(resp_text))

        cipher = resp_text.strip().strip('"')
        try:
            parsed = json.loads(resp_text)
            if isinstance(parsed, dict):
                for k in ("data", "cipher", "encryptData"):
                    if isinstance(parsed.get(k), str):
                        cipher = parsed[k]
                        break
        except Exception:
            pass

        if not cipher:
            raise RuntimeError(f"encryption/login 响应异常: {resp_text[:200]}")

        try:
            decrypted = sm4_cbc_decrypt(a, d, cipher)
            login_result = json.loads(decrypted)
        except Exception as e:
            logger.error("[%s] 解密失败: %s; cipher前60=%s", step, e, cipher[:60])
            raise RuntimeError(f"登录解密失败: {e}")

        logger.info("[%s] 登录结果 status=%s message=%s", step,
                    login_result.get("status"), login_result.get("message", "")[:100])

        if str(login_result.get("status")) != "0":
            raise RuntimeError(f"登录失败 status={login_result.get('status')} msg={login_result.get('message')}")

        return login_result

    def step10_encryption_verify(self, login_result: dict,
                                  env_pub: str, env_secure: str):
        step = "10-encryption_verify"
        logger.info("[%s] 验证登录结果...", step)

        a = secrets.token_hex(16)
        d = secrets.token_hex(16)
        sm2_env = sm2_encrypt(env_pub, a + "," + d)
        plain = json.dumps(login_result, separators=(",", ":"), ensure_ascii=False)
        sm4_env = sm4_cbc_encrypt(a, d, plain)
        body = {"authKey": {"sm4": sm4_env, "sm2": sm2_env, "secureCode": env_secure}}

        resp = self._req("POST", "/px-common-authcenter/auth/v2/encryption/verify",
                         body, with_skip=True, step=step)
        logger.info("[%s] verify 响应: %s", step, json.dumps(resp, ensure_ascii=False)[:300])

    def _cookie_summary(self) -> str:
        parts = [f"{c.name}={c.value[:12]}..." for c in self.session.cookies]
        return ", ".join(parts)

    def cookie_string(self) -> str:
        return "; ".join(f"{c.name}={c.value}" for c in self.session.cookies)

    def login(self, max_retry: int = 3) -> str:
        last_err = None
        for attempt in range(1, max_retry + 1):
            try:
                logger.info("=" * 60)
                logger.info("登录尝试 #%d / %d", attempt, max_retry)
                logger.info("=" * 60)

                self._reset_session()
                self.step1_seed_session()

                cap = self.step2_captcha_get("first")
                pub_key, secure_code = self.step3_secureKey_get()

                candidates = self._solve_captcha(cap)
                captcha_verification, x = self.step4_captcha_check(cap, candidates)

                logger.info("[5-captcha_refresh] 刷新验证码（浏览器行为复刻）...")
                try:
                    self.step2_captcha_get("second")
                except Exception as e:
                    logger.warning("[5-captcha_refresh] 失败(可忽略): %s", e)

                env_pub, env_secure = self.step6_getSecureKey("first")

                login_result = self.step7_encryption_login(
                    captcha_verification, pub_key, secure_code, env_pub, env_secure)

                logger.info("[8-captcha_refresh] 再次刷新验证码...")
                try:
                    self.step2_captcha_get("third")
                except Exception as e:
                    logger.warning("[8-captcha_refresh] 失败(可忽略): %s", e)

                logger.info("[9-getSecureKey] 再次获取信封公钥...")
                try:
                    env_pub2, env_secure2 = self.step6_getSecureKey("second")
                except Exception as e:
                    logger.warning("[9-getSecureKey] 失败, 用第一次的: %s", e)
                    env_pub2, env_secure2 = env_pub, env_secure

                try:
                    self.step10_encryption_verify(login_result, env_pub2, env_secure2)
                except Exception as e:
                    logger.warning("[10-verify] 异常(可继续): %s", e)

                cookie = self.cookie_string()
                logger.info("登录成功! cookie长度=%d", len(cookie))
                logger.info("cookie字段: %s", self._cookie_summary())

                self._save_cookie_cache(cookie)
                return cookie

            except Exception as e:
                last_err = e
                logger.error("登录尝试 #%d 失败: %s", attempt, e)
                logger.debug(traceback.format_exc())
                time.sleep(2)

        raise RuntimeError(f"登录连续失败 {max_retry} 次: {last_err}")

    def _reset_session(self):
        self.session.cookies.clear()
        self.x_ticket = f"{secrets.token_hex(32)}.{secrets.token_hex(20)}"
        self.jsessionid = secrets.token_hex(16)
        self.session.cookies.set("X-Ticket", self.x_ticket)
        self.session.cookies.set("JSESSIONID", self.jsessionid)
        self.session.cookies.set("ClientTag", "OUTNET_BROWSE")
        self.session.cookies.set("X-Token", "undefined")
        self.session.cookies.set("CurrentRoute", "/dashboard")
        self.session.cookies.set("Gray-Tag", self.username.encode("utf-8").hex())

    def _solve_captcha(self, cap: dict) -> list:
        step = "captcha_solve"
        logger.info("[%s] 检测滑块缺口...", step)
        if not cap["original"] or not cap["jigsaw"]:
            raise RuntimeError("验证码图片为空")

        browser_x = detect_gap_x(cap["original"], cap["jigsaw"])
        secret = cap["secret_key"]
        token = cap["token"]

        candidates = []
        for offset in [0, -1, 1, -2, 2, -3, 3]:
            x = round(browser_x + offset)
            if 0 <= x <= 310:
                point = json.dumps({"x": float(x), "y": 5}, separators=(",", ":"))
                point_json = aes_ecb_encrypt(secret, point)
                captcha_verification = aes_ecb_encrypt(secret, token + "---" + point)
                candidates.append((x, point_json, captcha_verification))

        logger.info("[%s] browser_x=%.2f, %d个候选", step, browser_x, len(candidates))
        return candidates

    def _save_cookie_cache(self, cookie: str):
        cache_path = OUTPUT_DIR / ".cookie_cache.json"
        try:
            cache_path.write_text(json.dumps({
                "cookie": cookie,
                "saved_at": time.time(),
                "username": self.username,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Cookie已缓存到 %s", cache_path)
        except Exception as e:
            logger.warning("Cookie缓存写入失败: %s", e)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error("配置文件不存在: %s", CONFIG_PATH)
        logger.error("请创建 config.json:")
        example = {
            "username": "你的PMOS账号",
            "password": "你的PMOS密码",
            "unit_id": "你的机组ID",
            "base_url": "https://pmos.sd.sgcc.com.cn:18080/trade",
            "auth_host": "https://pmos.sd.sgcc.com.cn",
        }
        logger.error(json.dumps(example, ensure_ascii=False, indent=2))
        sys.exit(1)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = ["username", "password", "unit_id"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        logger.error("config.json 缺少: %s", missing)
        sys.exit(1)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="PMOS自动爬虫v2（详细日志版）")
    parser.add_argument("--date", help="爬取指定日期 (YYYY-MM-DD)")
    parser.add_argument("--test-login", action="store_true", help="只测试登录，不爬取数据")
    parser.add_argument("--no-db", action="store_true", help="不写入数据库")
    args = parser.parse_args()

    print("=" * 60)
    print("  PMOS 自动爬虫 v2 (详细日志版)")
    print("=" * 60)

    cfg = load_config()
    crawler = PMOSAutoCrawler(
        username=cfg["username"],
        password=cfg["password"],
        unit_id=cfg["unit_id"],
        auth_host=cfg.get("auth_host", AUTH_HOST),
        trade_base=cfg.get("base_url", TRADE_BASE),
    )

    logger.info("账号: %s", cfg["username"])
    logger.info("机组: %s", cfg["unit_id"])
    logger.info("日志: %s", OUTPUT_DIR / f"auto_crawler_v2_{today_str}.log")
    logger.info("详情: %s", OUTPUT_DIR / f"auto_crawler_v2_detail_{today_str}.log")
    logger.info("验证码: %s", CAPTCHA_DIR)

    try:
        cookie = crawler.login()
        logger.info("登录成功!")

        if args.test_login:
            logger.info("测试登录模式，跳过数据爬取")
            return

        logger.info("接下来可以使用此cookie爬取数据")
        logger.info("Cookie: %s", cookie[:80] + "...")

    except Exception as e:
        logger.error("登录失败: %s", e)
        logger.error(traceback.format_exc())
        logger.info("")
        logger.info("请将以下日志文件发送给开发者排查:")
        logger.info("  主日志: %s", OUTPUT_DIR / f"auto_crawler_v2_{today_str}.log")
        logger.info("  详情日志: %s", OUTPUT_DIR / f"auto_crawler_v2_detail_{today_str}.log")
        logger.info("  验证码: %s", CAPTCHA_DIR)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        logger.exception("程序异常: %s", e)
        sys.exit(1)
    finally:
        if _FROZEN:
            try:
                input("\n按回车键退出...")
            except EOFError:
                pass
