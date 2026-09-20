#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只爬「日前出清（二次出清 / 最终版）」96 点价格的极简爬虫。

不登录、不点浏览器、不连数据库 —— 只用一个从浏览器复制出来的 Cookie
（或整条 cURL）去打一个接口：

    /qctc/qctc_pm_trade_outside/trade/DaJyjgfbPlantQuery/getDetail96

口径说明：日前出清有两版。本项目要的是**二次出清（最终版）**，页面
/qctc-trade/dayAheadTransaction/declare/towFdcResultQuery10454。另一版
首次版（DaJyjgfbPlantFirQuery）近期只发布部分时段，实测 2026-09-01 仅 28/96 点。

不管当天是不是满 96 点，都原样落盘，并在覆盖统计里标出每天拿到几个点。

------------------------------------------------------------
三种凭据文件（放 exe 同目录，任选其一）：

  1. config.json     —— 最全面，见 config_da.example.json
  2. curl.txt        —— 浏览器 Copy as cURL (bash) 原样粘贴（推荐，最省事）
  3. cookie.txt + token.txt —— 纯文本各一行

cookie.txt / token.txt 允许写成下面任意一种，程序会自动清洗：
        Admin-Token=xxx; X-Ticket=yyy
        "Admin-Token=xxx; X-Ticket=yyy"
        {"cookie":"Admin-Token=xxx; X-Ticket=yyy"}
        Bearer xxx

------------------------------------------------------------
用法：
    crawl_da_only.exe                        # 用 config.json 里的日期
    crawl_da_only.exe --start 2022-01-01 --end 2026-09-14
    crawl_da_only.exe --date 2026-09-14
    crawl_da_only.exe --config my.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(sys.executable).parent.resolve() if getattr(sys, "frozen", False) \
    else Path(__file__).resolve().parents[3]
# 源码运行时配置仍统一放在 scripts/crawler；冻结运行时则和 exe 同目录。
RUNTIME_DIR = BASE_DIR if getattr(sys, "frozen", False) else BASE_DIR / "scripts" / "crawler"

# ---------------------------------------------------------------- 默认配置
DEFAULTS = {
    "host": "https://pmos.sd.sgcc.com.cn:18080",
    "unit_id": "7B2B5622A6FA5E9BE0531001C10A211E",
    # 日前口径 = 二次出清（最终版）。首次版接口 DaJyjgfbPlantFirQuery 近期只发布
    # 部分时段（实测 2026-09-01 仅 28/96 点），不能作为建模/结算口径。
    "api_path": "/qctc/qctc_pm_trade_outside/trade/DaJyjgfbPlantQuery/getDetail96",
    "page_path": "/qctc-trade/dayAheadTransaction/declare/towFdcResultQuery10454",
    "pdate_mode": "plain",          # iso = 2026-09-14T00:00:00.000Z ; plain = 2026-09-14
    "cookie": "",
    "token": "",
    "curl_file": "curl.txt",
    "cookie_file": "cookie.txt",
    "token_file": "token.txt",
    "start_date": "2022-01-01",
    "end_date": "",                 # 空 = 今天
    "delay_sec": 0.4,
    "timeout_sec": 30,
    "retries": 3,
    "insecure": False,
    "output_dir": "output_da",
    "keep_partial": True,           # 不满 96 点也落盘
    "stop_on_auth_error": True,
}

FIELDS = ("cqPrice", "kt", "power", "energy", "bq")


def load_config(explicit: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if explicit:
        cands = [Path(explicit)]
    else:
        # 优先自己的 config_da.json；没有再退回同目录的 config.json（与全量爬虫共用凭据）
        cands = [RUNTIME_DIR / "config_da.json", RUNTIME_DIR / "config.json"]
    path = next((p for p in cands if p.is_file()), cands[0])
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k.startswith("_") or v is None:
                        continue
                    cfg[k] = v
                print(f"[配置] 已读取 {path.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[配置] ⚠ {path.name} 解析失败（{e}），改用内置默认值")
    return cfg


# ---------------------------------------------------------------- 凭据清洗
def clean_cookie(raw: str) -> str:
    """把各种乱七八糟的写法洗干净成 'a=1; b=2'。"""
    s = (raw or "").strip()
    if not s:
        return ""
    if s[0] in "{[":
        try:
            j = json.loads(s)
        except Exception:  # noqa: BLE001
            j = None
        if isinstance(j, dict):
            for k in ("cookie", "Cookie", "COOKIE", "cookies", "value"):
                v = j.get(k)
                if isinstance(v, str) and v.strip():
                    s = v.strip()
                    break
            else:
                s = next((v.strip() for v in j.values()
                          if isinstance(v, str) and v.strip()), "")
        else:
            m = re.search(r'"(?:cookie|cookies|value)"\s*:\s*"([^"]+)"', s, re.I)
            if m:
                s = m.group(1)
    # 外层引号
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    # 换行/制表 → 空格，避免 "Invalid header value"
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s.lower().startswith("cookie:"):
        s = s.split(":", 1)[1].strip()
    return s


def clean_token(raw: str) -> str:
    """token 只留值，去掉 Bearer 前缀、引号、换行。"""
    s = (raw or "").strip()
    if not s:
        return ""
    if s[0] in "{[":
        try:
            j = json.loads(s)
        except Exception:  # noqa: BLE001
            j = None
        if isinstance(j, dict):
            for k in ("token", "Token", "access_token", "accessToken",
                      "authorization", "Authorization", "value"):
                v = j.get(k)
                if isinstance(v, str) and v.strip():
                    s = v.strip()
                    break
        else:
            m = re.search(r'"(?:token|access_?token|authorization|value)"\s*:\s*"([^"]+)"',
                          s, re.I)
            if m:
                s = m.group(1)
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    s = re.sub(r"[\r\n\t]+", "", s).strip()
    if s.lower().startswith("bearer "):
        s = s[7:].strip()
    return s


# ---------------------------------------------------------------- 输入解析
def _read(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=enc).strip()
        except UnicodeDecodeError:
            continue
        except Exception:  # noqa: BLE001
            return ""
    return ""


def parse_curl(text: str) -> dict:
    """从 Chrome 'Copy as cURL' 文本里抠出 url / headers。"""
    out: dict = {"headers": {}}
    m = re.search(r"curl\s+(['\"])(.*?)\1", text, re.S)
    if m:
        out["url"] = m.group(2).strip()
    for _q, value in re.findall(r"-H\s+(['\"])(.*?)\1", text, re.S):
        if ":" not in value:
            continue
        k, v = value.split(":", 1)
        out["headers"][k.strip().lower()] = v.strip()
    if "url" not in out:
        m = re.search(r"(https?://[^\s'\"]+)", text)
        if m:
            out["url"] = m.group(1)
    return out


def resolve_source(args, cfg: dict) -> dict:
    """优先级：命令行 > config.json > curl.txt > cookie.txt / token.txt"""
    src = {
        "host": cfg["host"], "unitid": args.unitid or cfg["unit_id"],
        "cookie": clean_cookie(args.cookie or cfg.get("cookie", "")),
        "token": clean_token(args.token or cfg.get("token", "")),
        "headers": {},
        "api_path": cfg["api_path"], "page_path": cfg["page_path"],
        "pdate_mode": cfg["pdate_mode"],
    }

    curl_file = Path(args.curl) if args.curl else (RUNTIME_DIR / str(cfg["curl_file"]))
    if curl_file.is_file():
        c = parse_curl(_read(curl_file))
        u = urllib.parse.urlsplit(c.get("url", ""))
        if u.netloc:
            src["host"] = f"{u.scheme}://{u.netloc}"
        if u.path:
            src["api_path"] = u.path
        q = urllib.parse.parse_qs(u.query)
        if q.get("unitid"):
            src["unitid"] = q["unitid"][0]
        if c["headers"].get("cookie"):
            src["cookie"] = clean_cookie(c["headers"]["cookie"])
        if c["headers"].get("authorization"):
            src["token"] = clean_token(c["headers"]["authorization"])
        if c["headers"].get("x-web-path"):
            src["page_path"] = c["headers"]["x-web-path"]
        src["headers"] = {k: v for k, v in c["headers"].items()
                          if k in ("accept", "accept-language", "origin", "referer")}
        print(f"[输入] 已读取 {curl_file.name}")

    if not src["cookie"]:
        f = RUNTIME_DIR / str(cfg["cookie_file"])
        if f.is_file():
            src["cookie"] = clean_cookie(_read(f))
            print(f"[输入] 已读取 {f.name}")
    if not src["token"]:
        f = RUNTIME_DIR / str(cfg["token_file"])
        if f.is_file():
            src["token"] = clean_token(_read(f))
            print(f"[输入] 已读取 {f.name}")

    if not src["cookie"]:
        print(f"[输入] ⚠ 没读到 cookie，看了看：{BASE_DIR}")
    return src


# ---------------------------------------------------------------- HTTP
def make_opener(insecure: bool):
    if insecure:
        ctx = ssl._create_unverified_context()  # noqa: S323
    else:
        ctx = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def get_json(opener, src, cfg, url):
    headers = {
        "Accept": "application/json, text/plain, */*",
        "X-Web-Path": src["page_path"],
        "Referer": src["host"] + src["page_path"],
    }
    headers.update(src["headers"])
    if src["token"]:
        headers["Authorization"] = "Bearer " + src["token"]
    if src["cookie"]:
        headers["Cookie"] = src["cookie"]
    req = urllib.request.Request(url, headers=headers, method="GET")
    with opener.open(req, timeout=cfg["timeout_sec"]) as resp:
        raw = resp.read().decode("utf-8", "replace")
    return json.loads(raw)


def pull_day(opener, src, cfg, date_str):
    """返回 (period_no -> row, error)"""
    pdate = date_str + "T00:00:00.000Z" if str(cfg["pdate_mode"]).lower() == "iso" else date_str
    url = (f"{src['host']}{src['api_path']}?pdate={urllib.parse.quote(pdate)}"
           f"&unitid={urllib.parse.quote(src['unitid'])}")
    last = ""
    for attempt in range(int(cfg["retries"])):
        try:
            j = get_json(opener, src, cfg, url)
            if j.get("code") != 0:
                raise RuntimeError(f"code={j.get('code')} msg={str(j.get('msg'))[:120]}")
            rows = j.get("data")
            if isinstance(rows, dict):
                rows = rows.get("data")
            if not isinstance(rows, list):
                raise RuntimeError(f"返回结构异常: {str(j)[:200]}")
            out = {}
            for r in rows:
                p = str(r.get("periodid") or r.get("periodId") or "").strip()
                if not p:
                    continue
                hh, mm = p.split(":")[:2]
                out[(int(hh) * 60 + int(mm)) // 15] = r
            return out, ""
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (401, 403):
                return {}, f"认证失效(HTTP {e.code})：Cookie/token 过期，请在浏览器重新复制"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < int(cfg["retries"]) - 1:
            time.sleep(1.5 * (attempt + 1))
    return {}, last


# ---------------------------------------------------------------- 主流程
def is_num(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if s in ("", "-", "null", "None"):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def daterange(start: str, end: str):
    d = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += dt.timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser(description="只爬日前出清（二次出清/最终版）96 点价格")
    ap.add_argument("--config", help="配置文件路径（默认 ./config.json）")
    ap.add_argument("--start", help="开始业务日 YYYY-MM-DD")
    ap.add_argument("--end", help="结束业务日 YYYY-MM-DD（默认今天）")
    ap.add_argument("--date", help="只爬一天")
    ap.add_argument("--cookie", help="Cookie 字符串")
    ap.add_argument("--token", help="Authorization token（可省）")
    ap.add_argument("--unitid", help="机组ID")
    ap.add_argument("--curl", help="cURL 文本文件路径（默认 ./curl.txt）")
    ap.add_argument("--delay", type=float, help="请求间隔秒")
    ap.add_argument("--insecure", action="store_true", help="跳过 HTTPS 证书校验")
    ap.add_argument("--out", help="输出目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.delay is not None:
        cfg["delay_sec"] = args.delay
    if args.insecure:
        cfg["insecure"] = True
    if args.out:
        cfg["output_dir"] = args.out

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    src = resolve_source(args, cfg)
    if not src["cookie"] and not src["token"]:
        print("\n❌ 没有任何凭据。把下面任一文件放到 exe 同目录：")
        print(f"   {BASE_DIR}")
        print("   a) curl.txt   —— 浏览器 F12 → Network → getDetail96 → 右键 Copy as cURL (bash)")
        print("   b) cookie.txt —— 纯文本一行：Admin-Token=xxx; X-Ticket=yyy")
        print("   c) config.json 里填 cookie / token")
        return 2

    if args.date:
        dates = [args.date]
    else:
        end = args.end or cfg.get("end_date") or dt.date.today().isoformat()
        start = args.start or cfg.get("start_date") or end
        dates = list(daterange(start, end))

    print(f"[配置] 主机   : {src['host']}")
    print(f"[配置] 接口   : {src['api_path']}")
    print(f"[配置] 机组   : {src['unitid']}")
    print(f"[配置] Cookie : {'有 (' + str(len(src['cookie'])) + ' 字符)' if src['cookie'] else '无'}")
    print(f"[配置] Token  : {'有 (' + str(len(src['token'])) + ' 字符)' if src['token'] else '无'}")
    print(f"[配置] 日期   : {dates[0]} → {dates[-1]}  共 {len(dates)} 天")
    print(f"[配置] 输出   : {out_dir}\n")

    opener = make_opener(bool(cfg["insecure"]))
    rows_out: list[list] = []
    cover: list[list] = []
    t0 = time.time()
    ok96 = 0

    for i, d in enumerate(dates):
        got, err = pull_day(opener, src, cfg, d)
        n = 0
        for pno in sorted(got):
            r = got[pno]
            if not is_num(r.get("cqPrice")):
                continue
            n += 1
            rows_out.append([d, r.get("periodid", ""), pno, r.get("cqPrice", "")] +
                            [r.get(f, "") for f in FIELDS[1:]])
        if n == 96:
            ok96 += 1
        cover.append([d, len(got), n, err.replace(",", ";")])
        flag = "" if n == 96 else "  ⚠"
        if (i + 1) % 10 == 0 or n != 96 or err:
            print(f"  [{i+1}/{len(dates)}] {d}  返回 {len(got)} 行 / 价格有效 {n}/96{flag}"
                  + (f"  {err[:90]}" if err else ""))
        if err and "认证失效" in err and cfg["stop_on_auth_error"]:
            print("\n⛔ 凭据失效，已停止。请在浏览器重新复制 Cookie/token 后重跑。")
            break
        if i < len(dates) - 1:
            time.sleep(float(cfg["delay_sec"]))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    detail = out_dir / f"da_最终版_{dates[0]}_{dates[-1]}_{stamp}.csv"
    with open(detail, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["market_date", "periodid", "period_no", "cqPrice", "kt", "power", "energy", "bq"])
        w.writerows(rows_out)

    coverfile = out_dir / f"da_覆盖统计_{dates[0]}_{dates[-1]}_{stamp}.csv"
    with open(coverfile, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["market_date", "接口返回行数", "价格有效点数", "错误"])
        w.writerows(cover)

    print("\n" + "=" * 58)
    print(f"完成：{len(rows_out)} 行明细，用时 {time.time() - t0:.1f}s")
    print(f"满 96 点的天数：{ok96} / {len(cover)}")
    print(f"明细   : {detail}")
    print(f"覆盖统计: {coverfile}")
    if ok96 < len(cover):
        print("⚠ 有日期不满 96 点 —— 把「覆盖统计」发我，我看是不是要换接口。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(130)
