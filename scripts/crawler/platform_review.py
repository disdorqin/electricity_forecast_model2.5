#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【AI电力交易平台】电价预测复盘数据抓取 — 共享库

⚠️ 重要区分
   本模块面向的是「AI电力交易平台」演示站 http://47.114.107.96/ (账号 user / user123),
   **这是我们自建的平台展示站点, 与国网山东省电力交易平台 PMOS
   (pmos.sd.sgcc.com.cn) 是两个完全不同的系统**。
   项目内其他爬虫 (scripts/crawler/crawl.py / run_crawler.py / auto_fill_96.py 等)
   都是国网 PMOS 方向, 靠 Cookie + 数据库同步;
   本模块只与 47.114.107.96 打交道, 靠账号密码登录 + Bearer Token, 不依赖数据库。

平台接口要点(前端 axios baseURL=/api/v1, 代码里 URL 又带 /api/v1, 叠加成双前缀):
  登录      POST /api/v1/api/v1/auth/login        {username, password} -> {token, ...}
  模型字典  GET  /api/v1/api/v1/models/dict
  复盘导出  GET  /api/v1/api/v1/prediction-review/export?startDate&endDate&metric&models&modelLabels
                 一次性返回整段日期范围, 含「详细数据」+「统计报告」两个 sheet
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import requests

BASE = "http://47.114.107.96"
LOGIN_URL = BASE + "/api/v1/api/v1/auth/login"
MODELS_URL = BASE + "/api/v1/api/v1/models/dict"
EXPORT_URL = BASE + "/api/v1/api/v1/prediction-review/export"

# 平台默认账号(演示环境)
DEFAULT_USER = "user"
DEFAULT_PASS = "user123"


def login(username: str, password: str, timeout: int = 30) -> str:
    """登录平台, 返回 Bearer Token。"""
    resp = requests.post(
        LOGIN_URL, json={"username": username, "password": password}, timeout=timeout
    )
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"登录接口返回非 JSON: {resp.status_code} {resp.text[:200]}")
    if data.get("code") != 200:
        raise RuntimeError(f"登录失败: code={data.get('code')} msg={data.get('msg')}")
    token = (data.get("data") or {}).get("token")
    if not token:
        raise RuntimeError("登录成功但未返回 token")
    return token


def get_active_models(token: str, timeout: int = 30) -> list[dict]:
    """拉取模型字典, 返回交付且启用的模型。"""
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(MODELS_URL, headers=headers, timeout=timeout)
    data = resp.json()
    if data.get("code") != 200:
        raise RuntimeError(f"查询模型字典失败: {data.get('msg')}")
    models = [
        m
        for m in data.get("data", [])
        if m.get("isActive") == 1 and m.get("isDeliveryModel") == 1
    ]
    if not models:
        raise RuntimeError("没有找到启用的交付模型")
    return models


def export_review(
    token: str,
    start: str,
    end: str,
    models: list[dict],
    metric: str = "comprehensive_accuracy",
    timeout: int = 300,
) -> bytes:
    """调用「电价预测复盘」导出接口, 返回 xlsx 原始字节。"""
    headers = {"Authorization": f"Bearer {token}"}
    params = {"startDate": start, "endDate": end, "metric": metric}
    for m in models:
        params.setdefault("models", []).append(m["modelCode"])
        params.setdefault("modelLabels", []).append(m["displayName"] or m["deliveryModelName"])

    resp = requests.get(EXPORT_URL, headers=headers, params=params, timeout=timeout)
    ctype = resp.headers.get("Content-Type", "")

    if "json" in ctype.lower():
        # 接口可能返回 JSON 错误
        try:
            err = resp.json()
            raise RuntimeError(f"导出失败: code={err.get('code')} msg={err.get('msg')}")
        except ValueError:
            pass
    if not resp.content[:4].startswith(b"PK"):  # xlsx 是 zip (PK)
        raise RuntimeError(f"导出返回非 xlsx 内容: {resp.text[:300]}")
    resp.raise_for_status()
    return resp.content


def xlsx_to_csvs(xlsx_path: Path, out_prefix: Path) -> list[Path]:
    """把 xlsx 每个 sheet 转成一个 csv, 命名 <out_prefix>_<sheet名>.csv。"""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    generated: list[Path] = []
    try:
        for ws in wb.worksheets:
            out = out_prefix.parent / f"{out_prefix.name}_{ws.title}.csv"
            with open(out, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                for row in ws.iter_rows(values_only=True):
                    writer.writerow(["" if v is None else v for v in row])
            generated.append(out)
    finally:
        wb.close()
    return generated


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
