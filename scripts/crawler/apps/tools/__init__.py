"""应用层工具集：与 PMOS 主链路无关的独立小工具。

当前成员：
    ``platform_review.py``         演示站（47.114.107.96）复盘数据抓取库
    ``platform_review_update.py``  上者的命令行入口

⚠ 该演示站是本项目自建的展示平台，**与国网山东电力交易平台 PMOS 完全不同**，
不使用 PMOS Cookie、不写 `epf_pmos_*` 表、不参与 96/AUX 采集链路。

[2026-09-27] 由 `scripts/crawler/` 根层迁入。迁移时已同步修正：
`platform_review_update.py` 的 `ROOT` 由 `parents[2]` 改为 `parents[4]`
（层级加深两级），import 路径改为 `scripts.crawler.apps.tools.platform_review`。
"""
