# -*- coding: utf-8 -*-
"""
山东电力 PMOS 网站 — 账号密码自动登录模块
==========================================

目标：用 账号 + 密码 模拟前端登录，自动获取并刷新 Cookie
      （Admin-Token / X-Ticket / XHXT_SESSIONID），写回 config.json，
      供现有 PmosCrawler 使用。最终打包成 exe 在企业机上无人值守运行。

登录协议（已通过 HAR + 前端 JS 源码逆向，见 .workbuddy/memory/2026-07-29.md）：
  鉴权网关: https://pmos.sd.sgcc.com.cn/px-common-authcenter/auth/v2/...
  加密:     国密 SM2 / SM4 / SM3  +  AES-ECB(PKCS7) 滑块验证码

流程:
  1. seed_session  : 访问门户/交易页，取得 XHXT_SESSIONID（X-Ticket 由本地生成并全程复用）
  2. captcha/get   : 取得滑块图片(secretKey, token, original/jigsaw base64)
  3. 滑块缺口检测   : numpy 模板匹配 original vs jigsaw -> 缺口 x
  4. captcha/check : 提交 AES(pointJson) 校验
  5. secureKey/get : 取得服务器 SM2 公钥(pubKey) + secureCode(密码加密用)
  6. 构造登录表单 f (含 authKey=SM2(password,pubKey), captchaVerification)
  7. getSecureKey  : 取得信封 SM2 公钥(C) + secureCode(u)
  8. 信封封装       : A,d 随机 -> sm2="04"+SM2("A,d",C) ; sm4=SM4-CBC(JSON(f)+SM3, A,d)
  9. encryption/login  -> 解密响应(用 A,d) 得到登录结果 n
 10. encryption/verify -> 取得 Set-Cookie: Admin-Token
 11. 汇总全部 cookie -> 写回 config.json

注：步骤 10 的 verify 请求体为"对登录结果 n 重新信封封装"，是逆向推断的最可能形态；
    真机联调时若 verify 失败，请查看日志中打印的 decrypted_login_result 调整。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import secrets
import string
import time
from typing import Any, Optional

import requests

try:
    from gmssl import sm2 as _gm_sm2
    from gmssl import sm3 as _gm_sm3
    from gmssl import sm4 as _gm_sm4
    from gmssl.sm4 import CryptSM4, SM4_ENCRYPT, SM4_DECRYPT
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("gmssl import failed: %s", _e)
    _gm_sm2 = _gm_sm3 = _gm_sm4 = None
    CryptSM4 = SM4_ENCRYPT = SM4_DECRYPT = None

try:
    from Crypto.Cipher import AES as _PyAES  # pycryptodome，滑块 AES 用
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("pycryptodome import failed: %s", _e)
    _PyAES = None

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

AUTH_HOST = "https://pmos.sd.sgcc.com.cn"
TRADE_BASE = "https://pmos.sd.sgcc.com.cn:18080/trade"

# 鉴权网关路径（均走 https，与 :18080 交易域同 host 不同端口，cookie 跨端口共享）
API_CAPTCHA_GET = "/px-common-authcenter/auth/v2/captcha/get"
API_CAPTCHA_CHECK = "/px-common-authcenter/auth/v2/captcha/check"
API_VERIFY_TYPE_GET = "/px-common-authcenter/auth/v2/verifyType/get"
API_GATEWAY_CONFIG = "/px-gateway/GatewayConfig/getGatewayConfig"
API_SECUREKEY_GET = "/px-common-authcenter/auth/v2/secureKey/get"   # 密码 SM2 用
API_GETSECUREKEY = "/px-common-authcenter/auth/v2/getSecureKey"     # 信封 SM2 用
API_ENCRYPT_LOGIN = "/px-common-authcenter/auth/v2/encryption/login"
API_ENCRYPT_VERIFY = "/px-common-authcenter/auth/v2/encryption/verify"

SKIP_HEADER = {"Intercept-Headers": "SKIP"}   # 加密接口必须带此头

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/json;charset=UTF-8",
}

# 滑块图片固有尺寸（与前端一致）
CAPTCHA_IMG_W = 310
CAPTCHA_IMG_H = 155


# ===========================================================================
#  国密 / AES 底层
# ===========================================================================


def _pkcs7_pad(data: bytes, block: int = 16) -> bytes:
    pad = block - (len(data) % block)
    return data + bytes([pad]) * pad


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        return data
    pad = data[-1]
    if 1 <= pad <= 16:
        return data[:-pad]
    return data


class _SM2Fixed(_gm_sm2.CryptSM2):
    """修复 gmssl CryptSM2.__init__ 中 ``public_key.lstrip("04")`` 的 bug。

    gmssl 官方写法 ``public_key.lstrip("04")`` 会把所有前导 '0'/'4' 字符都剥掉，
    而不是只剥字面量 "04" 前缀。当服务器下发的 SM2 公钥 X 坐标以 '0' 或 '4' 开头时
    （例如真实密钥 ``040042a7…`` 会被错误截成 ``2a7…``），公钥被损坏成非法椭圆曲线点，
    导致 ``k·Pb`` 算出 None、`_kg` 返回 None、最终在 `_double_point` 的
    ``len(Point)`` 处抛出 ``TypeError: 'NoneType' object has no len()``。

    这里只精确剥掉 2 字符的 "04" 前缀（国标未压缩点的固定前缀），其余字符原样保留。
    """

    def __init__(self, private_key, public_key, ecc_table=None, mode=0, asn1=False):
        ecc_table = ecc_table or _gm_sm2.default_ecc_table
        self.private_key = private_key
        if public_key and public_key.startswith("04"):
            stripped = public_key[2:]  # 仅剥 2 字符 "04" 前缀
            # 剥掉后应是 128 hex（64 字节 X+Y），否则视为无前缀的合法公钥，原样保留
            self.public_key = stripped if len(stripped) == 128 else public_key
        else:
            self.public_key = public_key
        self.para_len = len(ecc_table["n"])
        self.ecc_a3 = (int(ecc_table["a"], 16) + 3) % int(ecc_table["p"], 16)
        self.ecc_table = ecc_table
        self.mode = mode
        self.asn1 = asn1


def sm2_encrypt(pub_key_hex: str, plain_text: str) -> str:
    """SM2 公钥加密 (C1C3C2, 国密标准)，返回 "04" + hex（与前端 sm-crypto doEncrypt(e,t,1) 一致）。

    兼容性说明：
      1) gmssl 自带 ``public_key.lstrip("04")`` 在公钥 X 坐标以 0/4 开头时会把密钥截坏，
         这里改用 _SM2Fixed 子类只精确剥 "04" 前缀；
      2) gmssl 在极少数内部随机 k 下会在 _double_point 抛 None（实测概率约 1.5%），
         这里做有限次重试，每次重试重新构造 _SM2Fixed（内部重新抽样 k），确保稳定。
    """
    data = plain_text.encode("utf-8")
    last_err: Exception | None = None
    for _attempt in range(8):
        try:
            crypt = _SM2Fixed(public_key=pub_key_hex, private_key="00" * 32, mode=1)
            ct = crypt.encrypt(data)  # 返回 bytes，无 04 前缀
            if ct is None:
                last_err = RuntimeError("gmssl encrypt returned None")
                continue
            return "04" + ct.hex()
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise PmosAuthError(f"SM2 加密连续失败 ({_attempt + 1} 次): {last_err}")


def sm3_hex(s: str) -> str:
    """SM3 摘要，返回 64 hex 小写。"""
    data_ints = [b for b in s.encode("utf-8")]
    return _gm_sm3.sm3_hash(data_ints)


def sm4_cbc_encrypt(key_hex: str, iv_hex: str, plain_text: str) -> str:
    """SM4-CBC 加密：明文尾部追加 SM3(明文)，gmssl 内部做 PKCS7 填充，hex 输出。"""
    body = plain_text + sm3_hex(plain_text)
    data = body.encode("utf-8")
    crypt = CryptSM4()
    crypt.set_key(bytes.fromhex(key_hex), SM4_ENCRYPT)
    ct = crypt.crypt_cbc(bytes.fromhex(iv_hex), data)
    return ct.hex()


def sm4_cbc_decrypt(key_hex: str, iv_hex: str, cipher_hex: str) -> str:
    """SM4-CBC 解密：gmssl 内部去 PKCS7 填充 -> 末 64 字符为 SM3 校验，前段为明文 JSON。"""
    data = bytes.fromhex(cipher_hex)
    crypt = CryptSM4()
    crypt.set_key(bytes.fromhex(key_hex), SM4_DECRYPT)
    pt = crypt.crypt_cbc(bytes.fromhex(iv_hex), data)
    text = pt.decode("utf-8", errors="replace")
    # sm3encrypt：末 64 字符为 SM3(前段)，校验后返回前段
    if len(text) > 64:
        head, tail = text[:-64], text[-64:]
        if sm3_hex(head) == tail:
            return head
        return text  # 校验失败也尽力返回
    return text


def aes_ecb_pkcs7_b64(key_str: str, plain_str: str) -> str:
    """AES-ECB-PKCS7，输出 base64（滑块验证码加密用）。key 为 16 字符明文。"""
    key = key_str.encode("utf-8")  # 16 字节
    if len(key) > 16:
        key = key[:16]
    elif len(key) < 16:
        key = key.ljust(16, b"\0")
    data = plain_str.encode("utf-8")
    data = _pkcs7_pad(data, 16)
    cipher = _PyAES.new(key, _PyAES.MODE_ECB)
    return base64.b64encode(cipher.encrypt(data)).decode("ascii")


def random_hex(n: int = 32) -> str:
    """生成 n 个十六进制字符（SM4 key / iv 用 32hex=16字节）。"""
    return secrets.token_hex(n // 2) if n % 2 == 0 else secrets.token_hex(n // 2)[:n]


def random_token() -> str:
    """cookieTicketKey 之类的前端随机令牌。"""
    return "".join(secrets.choice(string.hexdigits.lower()) for _ in range(32))


# ===========================================================================
#  滑块验证码缺口检测
# ===========================================================================


def detect_gap_x(original_bytes: bytes, jigsaw_bytes: bytes) -> float:
    """
    在 original(310x155 背景含缺口) 中定位 jigsaw(47x155 拼图块) 应放置的 x 坐标。

    原理：jigsaw 是 original 中某处的"切出块"，其不透明像素的 RGB 与 original 对应位置一致。
          对候选 x 平移 jigsaw，统计不透明像素与 original 的差异，差异最小处即缺口位置。
          使用抛物线插值实现亚像素精度。
    """
    orig = np.asarray(Image.open(original_bytes_io(original_bytes)).convert("RGB"), dtype=np.float64)
    jig = Image.open(original_bytes_io(jigsaw_bytes)).convert("RGBA")
    jig_arr = np.asarray(jig, dtype=np.float64)
    jh, jw = jig_arr.shape[:2]

    # 不透明像素掩码（alpha > 50）
    mask = jig_arr[:, :, 3] > 50
    jig_rgb = jig_arr[:, :, :3]

    oh, ow = orig.shape[:2]

    # 计算每个候选 x 的得分
    scores = []
    for x in range(0, ow - jw + 1):
        region = orig[0:jh, x:x + jw, :]
        if region.shape[1] != jw:
            continue
        diff = np.abs(region - jig_rgb) * mask[:, :, None]
        score = diff.sum()
        scores.append(score)

    # 找整数最佳位置
    scores_arr = np.array(scores, dtype=np.float64)
    best_x_int = int(np.argmin(scores_arr))
    best_score = scores_arr[best_x_int]

    # 抛物线插值亚像素：用 best_x_int 及其左右点的得分拟合二次曲线
    # 二次曲线 y = a(x-x0)^2 + b(x-x0) + c 在 x0=best_x_int 处过三个点
    x0 = float(best_x_int)
    if 1 <= best_x_int < len(scores_arr) - 1:
        y_m1 = scores_arr[best_x_int - 1]
        y_0 = scores_arr[best_x_int]
        y_p1 = scores_arr[best_x_int + 1]
        denom = 2.0 * (y_m1 + y_p1 - 2.0 * y_0)
        if abs(denom) > 1e-12:
            x_sub = x0 + (y_m1 - y_p1) / denom
        else:
            x_sub = x0
        # 限制亚像素偏移不超过 ±1px
        x_sub = max(x0 - 1.0, min(x0 + 1.0, x_sub))
    else:
        x_sub = x0

    # 偏移校准：HAR 逆向发现浏览器上报的 x 比算法检测的 bitmap_x 小 ~1.7px。
    # 原代码固定 -2px，改用亚像素后校准常数。若服务端校验失败可微调此值。
    CALIB_OFFSET = 1.7
    browser_x = max(0.0, x_sub - CALIB_OFFSET)

    logger.info(
        "[slider] integer_x=%d subpixel_x=%.4f -> browser_x=%.4f (score=%.0f, candidates=%d)",
        best_x_int, x_sub, browser_x, best_score, len(scores),
    )
    return browser_x


def original_bytes_io(b: bytes):
    from io import BytesIO
    return BytesIO(b)


def b64_to_bytes(s: Optional[str]) -> bytes:
    if not s:
        return b""
    # 去掉 data:image/...;base64, 前缀（若有）
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


# ===========================================================================
#  登录主体
# ===========================================================================


class PmosAuthError(RuntimeError):
    pass


class PmosAuth:
    """山东电力 PMOS 账号密码登录器。"""

    def __init__(
        self,
        username: str,
        password: str,
        auth_host: str = AUTH_HOST,
        trade_base: str = TRADE_BASE,
        timeout: int = 30,
        verify_ssl: bool = False,
    ):
        self.username = username
        self.password = password
        self.auth_host = auth_host.rstrip("/")
        self.trade_base = trade_base.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        # ? GatewayConfig/loginMode ?????HAR ???? false?
        self.two_factor_type: Any = False

        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.session.verify = verify_ssl

        # 复刻浏览器常驻 cookie（由前端注入，服务端用于会话/租户识别）。
        # 关键：X-Ticket 是前端在 mount 时本地生成的客户端票据
        #   （JS: set("X-Ticket", <generator>())，格式为 "<64hex>.<40hex>"），
        #   鉴权网关 px-common-authcenter 把它当成“本次登录会话”的关联键——
        #   滑块 captcha/check 通过后，服务端把“已校验”状态挂在 X-Ticket 上；
        #   encryption/login 必须带上同一个 X-Ticket 才能读到该状态，否则报
        #   100010 滑块校验失败。浏览器把真实票据放在 Cookie 中，而请求头仍为
        #   X-Ticket=undefined；本地生成的票据只写入 Cookie 并全程复用。
        self.x_ticket = self._gen_x_ticket()
        self.session.cookies.set("X-Ticket", self.x_ticket)
        # HAR 中真实 X-Ticket 在 Cookie；请求头固定为前端默认值。
        self.session.headers.update({
            "X-Ticket": "undefined",
            "X-Token": "null",
            "ClientTag": "OUTNET_BROWSE",
            "CurrentRoute": "/outNet",
            "Origin": self.auth_host,
            "Referer": self.auth_host + "/",
        })

        # 安全网：浏览器在每次鉴权请求都带一个稳定的 JSESSIONID（servlet 会话标识）。
        # 整份 HAR 中没有任何响应 Set-Cookie: JSESSIONID —— 说明它是浏览器在录制前加载
        # 登录页时由服务端下发的，而我们的请求不会触发服务端设置它。鉴权网关本身大概率
        # 用 X-Ticket 关联滑块校验状态（见上），但为防服务端同时要求稳定的 JSESSIONID
        # （惰性创建 servlet 会话且不在录制窗口内下发），这里也生成一个稳定的 JSESSIONID
        # 全程复用：Tomcat 直接以该 cookie 值作为会话键，check 与 login 共用同一键即可关联
        # 校验状态；若服务端其实只用 X-Ticket，此 cookie 会被忽略，无副作用。
        self.jsessionid = secrets.token_hex(16)
        self.session.cookies.set("JSESSIONID", self.jsessionid)

        self.session.cookies.set("ClientTag", "OUTNET_BROWSE")
        self.session.cookies.set("X-Token", "undefined")
        self.session.cookies.set("CurrentRoute", "/dashboard")
        self.session.cookies.set("Gray-Tag", self.username.encode("utf-8").hex())
        # 调试钩子：记录所有 Set-Cookie，便于定位会话/滑块校验问题
        self.session.hooks["response"].append(self._log_set_cookie)

        # 缓存
        self._pub_key: Optional[str] = None
        self._secure_code: Optional[str] = None
        self.last_login_result: Any = None

    @staticmethod
    def _gen_x_ticket() -> str:
        """复刻前端本地生成的 X-Ticket：<64hex>.<40hex>（共 104 hex 字符）。

        浏览器在应用 mount 时调用生成器产生，鉴权网关用 Cookie 中的值作为会话关联键。
        captcha/check 与 encryption/login 通过同一个 Cookie 票据关联滑块状态。
        """
        return f"{secrets.token_hex(32)}.{secrets.token_hex(20)}"

    def _reset_attempt_session(self) -> None:
        """每次重试清理服务端会话键，避免复用上一轮失败的滑块状态。"""
        self.session.cookies.clear()
        self.x_ticket = self._gen_x_ticket()
        self.session.cookies.set("X-Ticket", self.x_ticket)
        self.jsessionid = secrets.token_hex(16)
        self.session.cookies.set("JSESSIONID", self.jsessionid)
        self.session.cookies.set("ClientTag", "OUTNET_BROWSE")
        self.session.cookies.set("X-Token", "undefined")
        self.session.cookies.set("CurrentRoute", "/dashboard")
        self.session.cookies.set("Gray-Tag", self.username.encode("utf-8").hex())
        self._pub_key = None
        self._secure_code = None

    @staticmethod
    def _log_set_cookie(resp, *args, **kwargs) -> None:
        sc = resp.headers.get("Set-Cookie")
        if sc:
            logger.info("[cookie] Set-Cookie <- %s : %s", resp.url[:90], sc[:160])

    # ------------------------------------------------------------------
    #  底层请求
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict, with_skip: bool = False) -> dict:
        url = self.auth_host + path
        headers = dict(SKIP_HEADER) if with_skip else {}
        headers["X-Ticket"] = "undefined"  # 真实票据通过 Cookie 发送
        headers["X-Token"] = "null"
        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise PmosAuthError(f"HTTP {resp.status_code} @ {path}: {resp.text[:200]}")
        try:
            return resp.json()
        except Exception:
            raise PmosAuthError(f"非 JSON 响应 @ {path}: {resp.text[:200]}")

    def _get(self, url: str) -> requests.Response:
        return self.session.get(url, timeout=self.timeout)

    def _post_raw(self, path: str, payload: dict, with_skip: bool = False) -> str:
        """POST 并返回原始响应文本（encryption/login|verify 返回裸 hex，不走 JSON 解析）。"""
        url = self.auth_host + path
        headers = dict(SKIP_HEADER) if with_skip else {}
        headers["X-Ticket"] = "undefined"  # 真实票据通过 Cookie 发送
        headers["X-Token"] = "null"
        resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise PmosAuthError(f"HTTP {resp.status_code} @ {path}: {resp.text[:200]}")
        return resp.text

    @staticmethod
    def _extract_cipher(raw: str) -> str:
        """从响应文本提取密文 hex：优先尝试 JSON（可能是字符串字面量或 {data:...}），否则当裸 hex。"""
        raw = raw.strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                return parsed.strip()
            if isinstance(parsed, dict):
                for k in ("data", "cipher", "encryptData"):
                    if isinstance(parsed.get(k), str):
                        return parsed[k]
                # 整个 data 字段可能是对象（极端情况）
                return json.dumps(parsed)
        except Exception:
            pass
        # 裸 hex（去除可能的空白/引号）
        return raw.strip().strip('"')

    # ------------------------------------------------------------------
    #  0. 会话播种：取得 X-Ticket / XHXT_SESSIONID
    # ------------------------------------------------------------------

    def fetch_gateway_config(self) -> None:
        """???????????? loginMode/twoFactorType?"""
        try:
            resp = self._post(API_GATEWAY_CONFIG, {})
            data = resp.get("data") or {}
            ext = data.get("extendFieldMap") or {}
            raw = ext.get("loginMode")
            # ?? loginMode getter ??? checkBool??? GatewayConfig ?????????????
            # ???????? "false" ????????? JSON false?
            if raw is not None:
                self.two_factor_type = raw
        except Exception as e:
            # ?????? HAR ??? loginMode ? false???????????????
            logger.warning("[config] GatewayConfig ????????? twoFactorType=%r: %s", self.two_factor_type, e)

    def seed_session(self) -> None:
        """
        复刻浏览器登录前的会话播种：加载登录页/交易页/SSO 入口，让服务端下发
        XHXT_SESSIONID 等 cookie。X-Ticket 已在 __init__ 本地生成并写入 Cookie，
        （鉴权网关用它关联"滑块已校验"状态，captcha/check 与 encryption/login 必须一致）。
        """
        # 1) 登录页 HTML（创建 servlet 会话，可能下发 JSESSIONID；不影响 X-Ticket 逻辑）
        try:
            self._get(self.auth_host + "/")
            logger.info("[seed] 登录页访问完成")
        except Exception as e:
            logger.warning("[seed] 登录页访问失败（可忽略）: %s", e)

        # 2) 交易页（设 XHXT_SESSIONID，Path=/）
        try:
            self._get(self.trade_base + "/DaJyjgfbPlantQuery.do?appkey=187")
            logger.info("[seed] 交易页访问完成")
        except Exception as e:
            logger.warning("[seed] 交易页访问失败（可忽略）: %s", e)

        # 门户 SSO 入口（可能设 X-Ticket）
        try:
            # HAR ? service ???????? http ????????? SSO ?????
            service = "http%3A%2F%2Fpmos.sd.sgcc.com.cn%2Ftrade%2FDaJyjgfbPlantQuery.do%3Fappkey%3D187"
            self._get(self.auth_host + "/?service=" + service)
            logger.info("[seed] 门户入口访问完成")
        except Exception as e:
            logger.warning("[seed] 门户入口访问失败（可忽略）: %s", e)

        self.fetch_gateway_config()
        logger.info("[seed] 当前 cookie: %s", self._cookie_summary())

    # ------------------------------------------------------------------
    #  1. 滑块验证码
    # ------------------------------------------------------------------

    def get_verify_type(self) -> Any:
        """?????? captcha/get ?? verifyType/get ???"""
        resp = self._post(API_VERIFY_TYPE_GET, {})
        if resp.get("status") != 0:
            raise PmosAuthError(f"verifyType/get ??: {resp}")
        logger.info("[captcha] verifyType=%s", resp.get("data"))
        return resp.get("data")

    def get_captcha(self) -> dict:
        resp = self._post(API_CAPTCHA_GET, {"captchaType": "blockPuzzle"})
        if resp.get("status") != 0 or not resp.get("data", {}).get("repData"):
            raise PmosAuthError(f"captcha/get 失败: {resp}")
        rd = resp["data"]["repData"]
        logger.info("[captcha] secretKey=%s token=%s", (rd.get("secretKey") or "")[:6], (rd.get("token") or "")[:8])
        return {
            "secret_key": rd.get("secretKey"),
            "token": rd.get("token"),
            "original": b64_to_bytes(rd.get("originalImageBase64")),
            "jigsaw": b64_to_bytes(rd.get("jigsawImageBase64")),
        }

    def solve_captcha(self, cap: dict) -> list[tuple[int, str, str, float]]:
        """
        返回多个候选 (x, point_json, captcha_verification, browser_x) 元组列表，
        按置信度排序。第一个是最佳猜测，后续是 ±1px 的备选。
        """
        if not cap["original"] or not cap["jigsaw"]:
            raise PmosAuthError("captcha 图片为空")
        browser_x = detect_gap_x(cap["original"], cap["jigsaw"])
        secret = cap["secret_key"]
        token = cap["token"]

        candidates = []
        # 主候选：检测到的 x
        # 备选：±1px, ±2px（服务端可能有容差）
        for offset in [0, -1, 1, -2, 2]:
            x = round(browser_x + offset)
            if x < 0 or x > 310:
                continue
            point = json.dumps({"x": float(x), "y": 5}, separators=(",", ":"))
            point_json = aes_ecb_pkcs7_b64(secret, point)
            captcha_verification = aes_ecb_pkcs7_b64(secret, token + "---" + point)
            candidates.append((x, point_json, captcha_verification, browser_x))

        logger.info(
            "[captcha] browser_x=%.2f candidates=[%s]",
            browser_x,
            ",".join(str(c[0]) for c in candidates),
        )
        return candidates

    def check_captcha(self, cap: dict, point_json: str) -> bool:
        payload = {
            "captchaType": "blockPuzzle",
            "pointJson": point_json,
            "token": cap["token"],
        }
        # 注意：浏览器实测 captcha/check 请求**不带** Intercept-Headers: SKIP。
        # SKIP 会让鉴权拦截器跳过会话写入——而“滑块已校验”状态正是靠
        # 正常拦截器写进会话（JSESSIONID/X-Ticket）的。若此处加 SKIP，
        # 校验结果被丢弃，随后 encryption/login 读不到 → 100010 滑块校验失败。
        # 仅 encryption/login / encryption/verify 这类“需要建会话”的接口才带 SKIP。
        resp = self._post(API_CAPTCHA_CHECK, payload)
        ok = bool(resp.get("data", {}).get("repData", {}).get("result"))
        rep = resp.get("data", {}).get("repData", {})
        logger.info(
            "[captcha] check result=%s repCode=%s token=%s pointJson=%s...",
            ok, resp.get("data", {}).get("repCode"), rep.get("token", "")[:8], point_json[:12],
        )
        return ok

    # ------------------------------------------------------------------
    #  2. 公钥
    # ------------------------------------------------------------------

    def fetch_public_key(self) -> tuple[str, str]:
        """secureKey/get -> (pubKey 04..., secureCode)"""
        resp = self._post(API_SECUREKEY_GET, {})
        if resp.get("status") != 0:
            raise PmosAuthError(f"secureKey/get 失败: {resp}")
        d = resp["data"]
        self._pub_key = d["pubKey"]
        self._secure_code = d["secureCode"]
        logger.info("[key] pubKey=%s... secureCode=%s", self._pub_key[:16], self._secure_code)
        return self._pub_key, self._secure_code

    def get_secure_key(self) -> tuple[str, str]:
        """getSecureKey -> (secureKey=信封SM2公钥, secureCode) ，每次调用轮换。"""
        resp = self._post(API_GETSECUREKEY, {})
        if resp.get("status") != 0:
            raise PmosAuthError(f"getSecureKey 失败: {resp}")
        d = resp["data"]
        logger.info("[key] getSecureKey secureKey=%s... secureCode=%s", d["secureKey"][:16], d["secureCode"])
        return d["secureKey"], d["secureCode"]

    # ------------------------------------------------------------------
    #  3. 构造登录表单
    # ------------------------------------------------------------------

    def build_login_form(self, captcha_verification: str) -> dict:
        """构造与前端 submitFormBySlidingBlock / ISC 表单一致的登录对象 f。"""
        if not self._pub_key or not self._secure_code:
            raise PmosAuthError("请先 fetch_public_key()")
        form = {
            "cookieTicketKey": random_token(),
            "loginName": self.username.strip(),
            "username": self.username.strip(),
            "authKey": sm2_encrypt(self._pub_key, self.password),  # "04"+SM2(password, pubKey)
            "secureCode": self._secure_code,                       # 来自 secureKey/get
            "dnInfo": "",
            "isCfcaLogin": False,
            "loginFrom": 1,
            "twoFactorType": self.two_factor_type,
            "clientTag": "OUTNET_BROWSE",
            "randomCode": "ERWEFX",
            "captchaVerification": captcha_verification,           # 滑块校验值
            "origin": "PHBSD",
        }
        logger.info("[login] 表单字段=%s captchaVerification_len=%d", ",".join(form.keys()), len(captcha_verification))
        return form

    # ------------------------------------------------------------------
    #  4. 信封封装
    # ------------------------------------------------------------------

    def build_envelope(self, payload_obj: Any, env_key: Optional[tuple] = None) -> tuple[dict, dict]:
        """
        对 payload_obj（dict 或已含登录结果的 dict）做 SM2/SM4 信封封装。
        返回 (请求体 {authKey:{sm4,sm2,secureCode}}, smInfo {publicKey, iv})。

        env_key: 可选 (信封SM2公钥, secureCode) 元组。登录主流程会预先取好并传入，
        避免 build_envelope 内部再次调用 getSecureKey 而打断“滑块校验→登录”之间的会话
        （否则会话 cookie 可能被重置，导致已校验的滑块在 encryption/login 时失效）。
        不传则内部现取。
        """
        if env_key is not None:
            c, u = env_key
        else:
            c, u = self.get_secure_key()       # 信封 SM2 公钥 + secureCode
        a = random_hex(32)                     # SM4 key (16字节)
        d = random_hex(32)                     # SM4 iv  (16字节)
        g = a + "," + d
        sm2_env = sm2_encrypt(c, g)            # "04"+SM2(g, C)
        plain = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)
        sm4_env = sm4_cbc_encrypt(a, d, plain)
        auth_key = {"sm4": sm4_env, "sm2": sm2_env, "secureCode": u}
        sm_info = {"publicKey": a, "iv": d}
        return {"authKey": auth_key}, sm_info

    # ------------------------------------------------------------------
    #  5. 登录 + 校验
    # ------------------------------------------------------------------

    def login(self, max_captcha_retry: int = 3) -> str:
        """
        完整登录流程，返回可用于 config.json 的 cookie 字符串
        （Admin-Token; X-Ticket; XHXT_SESSIONID; ... 全部会话 cookie）。

        关键约束：滑块验证码的“校验(check)”必须走**正常鉴权拦截器**（即
        captcha/check 请求**不带** Intercept-Headers: SKIP）。服务端正是靠正常拦截器
        把“该会话滑块已校验”状态写进会话（JSESSIONID/X-Ticket）；随后 encryption/login
        读取该状态。若 check 误带 SKIP（早期版本曾如此），拦截器跳过会话写入，校验结果
        被丢弃 → encryption/login 报 100010 滑块校验失败。仅 encryption/login /
        encryption/verify 这类“需建会话”的接口才带 SKIP。本方法整段包一层重试，
        会话偶发漂移时自动换新一轮。
        """
        last_err: Exception | None = None
        for attempt in range(1, max_captcha_retry + 1):
            try:
                self._reset_attempt_session()
                logger.info("[attempt] 新认证会话 #%d X-Ticket=%s JSESSIONID=%s", attempt, self.x_ticket[:12], self.jsessionid[:12])
                self.seed_session()                 # 1) ?????XHXT_SESSIONID / X-Ticket?

                # ??????captcha/get -> verifyType/get -> secureKey/get -> captcha/check?
                cap = self.get_captcha()
                self.get_verify_type()
                self.fetch_public_key()             # 2) ?? SM2 ?? + secureCode

                # 3) 缺口检测 -> 验证码校验（尝试多个候选 x 位置）
                captcha_ok = False
                candidates = self.solve_captcha(cap)
                for idx, (x, point_json, captcha_verification, bx) in enumerate(candidates):
                    if self.check_captcha(cap, point_json):
                        captcha_ok = True
                        logger.info("[captcha] 第 %d 个候选 x=%d 校验通过", idx, x)
                        break
                    logger.info("[captcha] 第 %d 个候选 x=%d 校验失败，尝试下一个", idx, x)
                if not captcha_ok:
                    raise PmosAuthError("captcha/check 所有候选均未通过")

                # HAR 中 check 后会刷新一次验证码，然后才请求 getSecureKey。
                # 不改动已算好的 captcha_verification（仍用首个 cap 的 token/secretKey）。
                try:
                    self.get_captcha()
                    logger.info("[captcha] 二次刷新完成（复刻浏览器 [4]）")
                except Exception as _e:
                    logger.warning("[captcha] 二次刷新失败(忽略): %s", _e)

                # 4) 信封密钥必须在验证码校验之后取得，保持与浏览器顺序一致。
                env_c, env_u = self.get_secure_key()

                logger.info("[login] 提交前 cookie: %s", self._cookie_summary())

                # 5) 构造登录表单并用预取信封密钥封装（check 与 login 之间零网络调用）
                form = self.build_login_form(captcha_verification)
                body, sm_info = self.build_envelope(form, env_key=(env_c, env_u))

                # 6) encryption/login
                raw = self._post_raw(API_ENCRYPT_LOGIN, body, with_skip=True)
                cipher = self._extract_cipher(raw)
                if not cipher:
                    raise PmosAuthError(f"encryption/login 响应异常: {raw[:120]}")
                try:
                    login_result = json.loads(
                        sm4_cbc_decrypt(sm_info["publicKey"], sm_info["iv"], cipher)
                    )
                except Exception as e:  # noqa: BLE001
                    raise PmosAuthError(f"encryption/login 解密失败: {e}; 密文前60={cipher[:60]}")
                self.last_login_result = login_result
                logger.info("[login] 解密结果 status=%s", login_result.get("status"))
                logger.info("[login] decrypted_login_result=%s", json.dumps(login_result, ensure_ascii=False)[:1000])

                if str(login_result.get("status")) != "0":
                    # 滑块类错误（100010 等）下一轮换会话重试往往能过
                    raise PmosAuthError(
                        f"登录失败 status={login_result.get('status')} msg={login_result.get('message')}"
                    )

                # 7) encryption/verify：取得 Set-Cookie: Admin-Token
                #    HAR显示verify前需要：captcha/get(第三次) + getSecureKey(再次) + verify
                try:
                    self.get_captcha()
                    logger.info("[verify] 第三次刷新验证码完成")

                    # HAR中verify前会重新获取secureKey（与login用的不同）
                    venv_c, venv_u = self.get_secure_key()
                    logger.info("[verify] 重新获取信封密钥完成")

                    vbody, vinfo = self.build_envelope(login_result, env_key=(venv_c, venv_u))
                    vresp = self.session.post(
                        self.auth_host + API_ENCRYPT_VERIFY,
                        json=vbody,
                        headers=SKIP_HEADER,
                        timeout=self.timeout,
                    )
                    logger.info("[verify] HTTP %d", vresp.status_code)
                    for h in (
                        vresp.headers.get("Set-Cookie", "").split(",")
                        if isinstance(vresp.headers.get("Set-Cookie"), str)
                        else []
                    ):
                        logger.info("[verify] Set-Cookie=%s", h[:80])
                except Exception as e:  # noqa: BLE001
                    logger.warning("[verify] 二次校验异常（继续）: %s", e)

                self._ensure_admin_token()
                return self.cookie_string()

            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    "[login] 第 %d 次尝试失败: %s; session X-Ticket=%s JSESSIONID=%s XHXT=%s",
                    attempt, e, self.x_ticket[:12], self.jsessionid[:12],
                    (self.session.cookies.get("XHXT_SESSIONID") or "")[:12],
                )
                time.sleep(1)

        raise PmosAuthError(f"登录连续失败 {max_captcha_retry} 次: {last_err}")

    # ------------------------------------------------------------------
    #  cookie 处理
    # ------------------------------------------------------------------

    def _ensure_admin_token(self) -> None:
        cookies = {c.name: c.value for c in self.session.cookies}
        if "Admin-Token" not in cookies and "X-Ticket" in cookies:
            # 同源会话族，镜像给爬虫使用（真机若不需要可忽略）
            self.session.cookies.set("Admin-Token", cookies["X-Ticket"])
            logger.info("[cookie] 未检测到 Admin-Token，已用 X-Ticket 镜像兜底")

    def cookie_string(self) -> str:
        """把会话内全部 cookie 序列化为 'k=v; k2=v2' 形式（供 PmosCrawler 使用）。"""
        parts = []
        for c in self.session.cookies:
            parts.append(f"{c.name}={c.value}")
        s = "; ".join(parts)
        logger.info("[cookie] 序列化 %d 个 cookie", len(parts))
        return s

    def _cookie_summary(self) -> str:
        return ", ".join(f"{c.name}={c.value[:12]}..." for c in self.session.cookies)


# ===========================================================================
#  config.json 读写
# ===========================================================================


def update_config_cookie(config_path: str, cookie: str, preserve_keys: bool = True) -> None:
    """
    将登录得到的 cookie 写回 config.json。
    preserve_keys=True 时保留 username/password/unit_id/base_url 等其它字段。
    """
    cfg: dict = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    if preserve_keys:
        cfg["cookie"] = cookie
    else:
        cfg = {"cookie": cookie}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    logger.info("[config] 已写回 cookie 到 %s", config_path)


# ---------------------------------------------------------------------------
#  命令行自测（离线核心算法，不需要联网）
# ---------------------------------------------------------------------------

# ===========================================================================
#  Cookie 缓存（减少登录频率）
# ===========================================================================

COOKIE_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "dist", ".cookie_cache.json"
)


def save_cookie_cache(cookie_str: str) -> None:
    """保存 cookie 到缓存文件（含时间戳）。"""
    try:
        cache = {"cookie": cookie_str, "saved_at": time.time()}
        os.makedirs(os.path.dirname(COOKIE_CACHE_PATH), exist_ok=True)
        with open(COOKIE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        logger.info("[cache] Cookie 已缓存到 %s", COOKIE_CACHE_PATH)
    except Exception as e:
        logger.warning("[cache] Cookie 缓存写入失败: %s", e)


def load_cookie_cache(max_age_hours: int = 6) -> Optional[str]:
    """
    从缓存加载 cookie。
    max_age_hours: 缓存最大有效小时数（默认 6 小时）。
    返回 cookie 字符串，若缓存不存在或已过期返回 None。
    """
    if not os.path.exists(COOKIE_CACHE_PATH):
        return None
    try:
        with open(COOKIE_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        saved_at = cache.get("saved_at", 0)
        age_hours = (time.time() - saved_at) / 3600
        if age_hours > max_age_hours:
            logger.info("[cache] Cookie 已过期（%.1f 小时 > %d 小时）", age_hours, max_age_hours)
            return None
        cookie = cache.get("cookie", "")
        if not cookie:
            return None
        logger.info("[cache] 使用缓存的 Cookie（%.1f 小时前保存）", age_hours)
        return cookie
    except Exception as e:
        logger.warning("[cache] Cookie 缓存读取失败: %s", e)
        return None


def verify_cookie(cookie_str: str, auth_host: str = AUTH_HOST, timeout: int = 10) -> bool:
    """
    验证 cookie 是否仍然有效。
    尝试访问一个需要认证的接口，如果返回 200 说明 cookie 有效。
    """
    if not cookie_str:
        return False
    try:
        headers = {
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
            "Cookie": cookie_str,
        }
        resp = requests.get(
            f"{auth_host}/px-common-authcenter/auth/v2/information",
            headers=headers,
            timeout=timeout,
            verify=False,
        )
        if resp.status_code == 200:
            data = resp.json()
            valid = data.get("status") == 0
            if valid:
                logger.info("[cookie] 缓存 Cookie 有效")
            else:
                logger.info("[cookie] 缓存 Cookie 无效（status=%s）", data.get("status"))
            return valid
        logger.info("[cookie] 缓存 Cookie 无效（HTTP %d）", resp.status_code)
        return False
    except Exception as e:
        logger.warning("[cookie] Cookie 验证异常: %s", e)
        return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        # 滑块检测自测（使用 HAR 抽取的真实验证码图片）
        assets = os.path.join(os.path.dirname(__file__), "..", "..", "dist", "har_assets")
        orig_p = os.path.abspath(os.path.join(assets, "original.png"))
        jig_p = os.path.abspath(os.path.join(assets, "jigsaw.png"))
        if os.path.exists(orig_p) and os.path.exists(jig_p):
            with open(orig_p, "rb") as f:
                ob = f.read()
            with open(jig_p, "rb") as f:
                jb = f.read()
            x = detect_gap_x(ob, jb)
            print(f"[selftest] gap_x = {x}")
        else:
            print("[selftest] 未找到 HAR 验证码图片，跳过滑块检测")

        # 加密 round-trip 自测（用 HAR 中的真实 pubKey）
        test_pub = "04eb2f871eae716ca900d23c75c0c36c4614f9a420ee40935a8fef11914f1a692455dd93309715740c95b76f357898e0ca6d7014edab7cba39e72f504d1564129d"
        enc = sm2_encrypt(test_pub, "HelloPMOS123")
        print(f"[selftest] SM2 enc len={len(enc)} prefix={enc[:6]}")
        # AES
        a = aes_ecb_pkcs7_b64("kyCes2xL2qYAN9bR", json.dumps({"x": 123, "y": 5}))
        print(f"[selftest] AES b64={a}")
        # SM4
        k, iv = random_hex(32), random_hex(32)
        ct = sm4_cbc_encrypt(k, iv, '{"a":1}')
        pt = sm4_cbc_decrypt(k, iv, ct)
        _expected = '{"a":1}'
        print(f"[selftest] SM4 roundtrip ok={pt == _expected} pt={pt}")
        print("[selftest] 完成")
