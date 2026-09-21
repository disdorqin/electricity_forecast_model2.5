# 爬虫历史程序归档

本目录只保存历史程序，不是当前运行入口。当前 96 点 PMOS 爬虫唯一运行程序为：

```text
dist/crawler/crawl_96_auto_v6.exe
```

当前程序依赖的运行时模块仍保留在 `scripts/crawler/` 根目录，包括
`crawl_96_local.py`、`crawl.py`、`run_crawler.py`、`browser_session.py`、
`auth_runtime.py` 和 `cfca_runtime.py`。这些文件不能移动，否则当前 exe 的
重新构建和脚本模式会失效。

## 归档内容

- `legacy/auto_crawler_v2.py`、`legacy/auto_fill_96.py`、`legacy/run_full.py`：旧的
  自动爬取/补数入口；不再用于日常运行。
- `legacy/setup_windows_task.bat`、`legacy/auth_self_test.py`：旧定时任务和旧认证
  自检入口。
- `legacy/platform_review.py`、`legacy/platform_review_update.py`：独立演示平台复盘
  程序，与 PMOS 96 点生产链路无关。
- `migrations/migrate_authoritative_96_to_full.py`：一次性历史迁移工具，不属于日常
  爬虫入口；如需重新初始化交付表仍可从归档路径手工运行。

归档是保留，不是删除。归档程序不会被当前 exe 自动导入，也不会出现在公司电脑
部署目录的当前入口中。
