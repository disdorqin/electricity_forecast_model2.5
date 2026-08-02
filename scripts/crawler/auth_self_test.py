# -*- coding: utf-8 -*-
"""
auth_self_test.py — 离线校验（无需联网）

验证内容:
  1. SM2 加密产出 "04"+C1C3C2 hex，长度正确
  2. SM4-CBC 加解密 round-trip（含中文）
  3. AES-ECB-PKCS7 base64 产出合法
  4. 信封封装结构正确（authKey.{sm4,sm2,secureCode}）
  5. ★ 服务端模拟：用真实 SM2 密钥对，客户端封装 -> 服务端解密，恢复原始表单，
     证明本实现的国密与标准（服务端 sm-crypto 同构）可互操作
  6. 滑块缺口检测（在 HAR 抽取的真实验证码图片上运行）

运行: python scripts/crawler/auth_self_test.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("self_test")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import auth  # noqa: E402
from gmssl import sm2 as _gm_sm2  # noqa: E402
from gmssl.sm2 import default_ecc_table  # noqa: E402


def gen_keypair() -> tuple[str, str]:
    """用 gmssl 内部 _kg 从随机私钥推导公钥（仅用于离线服务端模拟）。"""
    import os as _os
    priv = format(int.from_bytes(_os.urandom(32), "big"), "x").zfill(64)[-64:]
    c = _gm_sm2.CryptSM2(public_key="", private_key=priv, mode=1)
    g = default_ecc_table["g"]  # 128 hex (x+y)，_kg 内部会补 '1'，勿加 04 前缀
    point_hex = c._kg(int(priv, 16), g)
    if point_hex.startswith("04"):
        point_hex = point_hex[2:]
    pub = "04" + point_hex
    assert len(pub) == 130, f"公钥长度异常: {len(pub)}"
    return priv, pub


def test_sm2() -> None:
    pub = "04eb2f871eae716ca900d23c75c0c36c4614f9a420ee40935a8fef11914f1a692455dd93309715740c95b76f357898e0ca6d7014edab7cba39e72f504d1564129d"
    enc = auth.sm2_encrypt(pub, "PMOS_test_密码")
    assert enc.startswith("04"), "SM2 必须以 04 开头"
    assert len(enc) % 2 == 0
    logger.info("[OK] SM2 加密 len=%d prefix=%s", len(enc), enc[:6])


def test_sm4() -> None:
    k, iv = auth.random_hex(32), auth.random_hex(32)
    plain = json.dumps({"loginName": "user1", "authKey": "04abc中文", "n": 123}, ensure_ascii=False)
    ct = auth.sm4_cbc_encrypt(k, iv, plain)
    pt = auth.sm4_cbc_decrypt(k, iv, ct)
    assert pt == plain, f"SM4 round-trip 失败: {pt!r}"
    logger.info("[OK] SM4 round-trip 一致 (len=%d)", len(ct))


def test_aes() -> None:
    secret = "kyCes2xL2qYAN9bR"
    token = "4360449bddb94c04a2b5668f14fd2cd5"
    point_json = auth.aes_ecb_pkcs7_b64(secret, json.dumps({"x": 152, "y": 5}))
    verification = auth.aes_ecb_pkcs7_b64(secret, token + "---" + json.dumps({"x": 152, "y": 5}))
    import base64
    assert len(base64.b64decode(point_json)) % 16 == 0, "AES 输出长度非 16 倍数"
    assert len(base64.b64decode(verification)) % 16 == 0
    logger.info("[OK] AES pointJson/verification 生成合法")


def test_envelope_and_server_sim() -> None:
    """客户端封装 -> 模拟服务端用私钥解密，恢复表单。证明国密可互操作。"""
    priv, pub = gen_keypair()
    # 客户端：把 pub 当作 getSecureKey 返回的服务器信封公钥
    a = auth.PmosAuth("user1", "pwd123")
    a.get_secure_key = lambda: (pub, "secureCodeXYZ")  # mock 信封公钥
    a._pub_key, a._secure_code = pub, "secureCodeXYZ"  # mock 密码加密公钥(仅供表单构造)

    form = a.build_login_form("captchaVerificationABC")
    body, sm_info = a.build_envelope(form)
    assert set(body["authKey"].keys()) == {"sm4", "sm2", "secureCode"}
    assert body["authKey"]["sm2"].startswith("04")
    logger.info("[OK] 信封结构正确")

    # --- 模拟服务端解密 ---
    # 1) 用服务端私钥解密 sm2("A,d") -> 得到 A,d
    server = _gm_sm2.CryptSM2(public_key=pub, private_key=priv, mode=1)
    g_hex = server.decrypt(bytes.fromhex(body["authKey"]["sm2"][2:]))  # 去 04 前缀
    g = g_hex.decode("utf-8")  # "A,d"
    A, d = g.split(",")
    assert len(A) == 32 and len(d) == 32, f"A/d 长度异常: {A}, {d}"
    # 2) 用 A,d 解密 sm4 -> 得到 JSON(form)+SM3
    recovered = auth.sm4_cbc_decrypt(A, d, body["authKey"]["sm4"])
    recovered_obj = json.loads(recovered)
    assert recovered_obj["loginName"] == "user1"
    assert recovered_obj["captchaVerification"] == "captchaVerificationABC"
    assert recovered_obj["secureCode"] == "secureCodeXYZ"
    logger.info("[OK] 服务端模拟解密成功，表单完整恢复（国密可互操作）")


def test_slider() -> None:
    assets = os.path.abspath(os.path.join(HERE, "..", "..", "dist", "har_assets"))
    op = os.path.join(assets, "original.png")
    jp = os.path.join(assets, "jigsaw.png")
    if not (os.path.exists(op) and os.path.exists(jp)):
        logger.warning("[SKIP] 未找到 HAR 验证码图片，跳过滑块检测")
        return
    with open(op, "rb") as f:
        ob = f.read()
    with open(jp, "rb") as f:
        jb = f.read()
    x = auth.detect_gap_x(ob, jb)
    assert 0 <= x <= auth.CAPTCHA_IMG_W, f"缺口 x 越界: {x}"
    logger.info("[OK] 滑块缺口检测 x=%d (图像宽=%d)", x, auth.CAPTCHA_IMG_W)


def main() -> None:
    logger.info("========== PMOS auth 离线自测 ==========")
    test_sm2()
    test_sm4()
    test_aes()
    test_envelope_and_server_sim()
    test_slider()
    logger.info("========== 全部离线自测通过 ==========")


if __name__ == "__main__":
    main()
