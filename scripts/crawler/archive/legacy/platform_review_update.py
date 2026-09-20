#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【AI电力交易平台】电价预测复盘数据集 — 命令行更新工具

⚠️ 重要区分: 本工具面向「AI电力交易平台」演示站 http://47.114.107.96/ (user / user123),
  这是自建平台站点, **与国网山东电力交易平台 PMOS (pmos.sd.sgcc.com.cn) 完全无关**。
  它不走 PMOS Cookie / 数据库, 而是账号密码登录 + 一次性导出接口。

数据集存放(稳定路径, 更新即覆盖):
  outputs/platform_review/
    ├─ 电价预测复盘.xlsx            原始导出(详细数据 + 统计报告两个 sheet)
    ├─ 电价预测复盘_详细数据.csv     逐小时: 实时电价/日前电价 + 各模型预测价
    └─ 电价预测复盘_统计报告.csv     全量 + 分月的综合准确率统计

用法:
  # 更新到最新(自动: 从数据集最早日期 或 2026-01-01 ~ 今天)
  python scripts/crawler/archive/legacy/platform_review_update.py

  # 指定抓取区间(明细按时间合并去重, 不影响区间外的旧数据)
  python scripts/crawler/archive/legacy/platform_review_update.py --start 2026-01-01 --end 2026-08-06

  # 只指定结束日期(从数据集最早日开始)
  python scripts/crawler/archive/legacy/platform_review_update.py --end 2026-08-06

  # 换账号 / 换输出目录
  python scripts/crawler/archive/legacy/platform_review_update.py --user user --password user123 --out outputs/platform_review
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

# 把项目根加进 sys.path, 保证从任意 cwd 运行都能 import scripts 包
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crawler.archive.legacy.platform_review import (  # noqa: E402
    DEFAULT_PASS,
    DEFAULT_USER,
    export_review,
    get_active_models,
    login,
    xlsx_to_csvs,
)

# 数据集文件(稳定命名)
XLSX_NAME = "电价预测复盘.xlsx"
DETAIL_NAME = "电价预测复盘_详细数据.csv"
STATS_NAME = "电价预测复盘_统计报告.csv"
# 平台最早有数据的日期(数据集为空时默认起始)
FALLBACK_START = "2026-01-01"


def _parse_cover(prefix: Path, detail_csv: Path) -> tuple[str, str]:
    """从现有明细 csv 读数据集覆盖范围 (min_date, max_date), 文件不存在则返回 (None, None)。"""
    if not detail_csv.exists():
        return None, None
    import csv

    dates = set()
    with open(detail_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        col = "日期" if "日期" in (reader.fieldnames or []) else "time"
        for row in reader:
            v = (row.get("日期") or row.get("time") or "").strip()
            if v:
                dates.add(v[:10])
    if not dates:
        return None, None
    return min(dates), max(dates)


def _merge_detail(existing_csv: Path, new_csv: Path, out_csv: Path) -> None:
    """按 time 合并明细: 区间内新数据覆盖旧数据, 区间外旧数据保留。"""
    import pandas as pd

    new = pd.read_csv(new_csv, dtype={"time": str})
    if existing_csv.exists():
        old = pd.read_csv(existing_csv, dtype={"time": str})
        df = pd.concat([old, new], ignore_index=True)
        df = df.drop_duplicates(subset="time", keep="last")
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = new
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="AI电力交易平台·电价预测复盘数据集更新")
    ap.add_argument("--start", default=None, help="抓取起始日期 YYYY-MM-DD(默认: 数据集最早日)")
    ap.add_argument("--end", default=None, help="抓取结束日期 YYYY-MM-DD(默认: 今天)")
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--password", default=DEFAULT_PASS)
    ap.add_argument(
        "--metric",
        default="comprehensive_accuracy",
        help="统计指标: accuracy/comprehensive_accuracy/spread_direction_accuracy/mae/profit_per_mwh",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "platform_review"),
        help="输出目录(默认 outputs/platform_review)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / XLSX_NAME
    detail_path = out_dir / DETAIL_NAME
    stats_path = out_dir / STATS_NAME

    # ---------- 日期解析 ----------
    old_min, old_max = _parse_cover(out_dir, detail_path)
    start = args.start or old_min or FALLBACK_START
    end = args.end or date.today().isoformat()
    print(f"[日期] 抓取区间: {start} ~ {end}  (现有数据集覆盖: {old_min} ~ {old_max})")

    # ---------- 登录 & 导出 ----------
    print(f"[1/4] 登录平台 http://47.114.107.96 ...")
    token = login(args.user, args.password)
    print("      登录成功")

    models = get_active_models(token)
    model_codes = ", ".join(m["modelCode"] for m in models)
    print(f"[2/4] 模型: {model_codes}")

    t0 = time.time()
    content = export_review(token, start, end, models, metric=args.metric)
    xlsx_path.write_bytes(content)
    print(f"[3/4] 导出完成 {len(content)/1024:.1f} KB, 耗时 {time.time()-t0:.1f}s -> {xlsx_path}")

    # ---------- 转 csv 并合并 ----------
    with tempfile.TemporaryDirectory(prefix="platform_review_") as tmp:
        tmp_prefix = Path(tmp) / "export"
        csvs = xlsx_to_csvs(xlsx_path, tmp_prefix)
        new_detail = next(c for c in csvs if c.name.endswith("_详细数据.csv"))
        new_stats = next(c for c in csvs if c.name.endswith("_统计报告.csv"))

        _merge_detail(detail_path, new_detail, detail_path)

        # 统计报告: 只有本次区间能覆盖现有全部数据时才覆盖(否则保留旧的, 避免聚合口径变小)
        covered = (old_min is None) or (start <= old_min and end >= old_max)
        if covered:
            import shutil

            shutil.copy2(new_stats, stats_path)  # 跨盘符(临时目录可能在 C:, 输出在 D:)
            print(f"      统计报告已更新 -> {stats_path}")
        else:
            print(f"      本次区间未覆盖现有全部数据({old_min}~{old_max}), 统计报告保留旧版")

    # ---------- 汇总 ----------
    import csv

    with open(detail_path, encoding="utf-8-sig") as f:
        n_new = sum(1 for _ in f) - 1
    new_min, new_max = _parse_cover(out_dir, detail_path)
    print(f"[4/4] 完成! 数据集: {new_min} ~ {new_max}, 共 {n_new} 小时记录")
    print(f"      {xlsx_path}")
    print(f"      {detail_path}")
    print(f"      {stats_path}")
    print("\n提示: 数据集已放行 git 跟踪(outputs/platform_review/), 可直接提交推送。")


if __name__ == "__main__":
    main()
