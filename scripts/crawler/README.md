"""scripts/crawler — 国网PMOS爬虫模块

功能:
  自动爬取国网电力市场数据，写入 MySQL 数据库并更新本地文件。

目录结构:
  __init__.py              模块标记
  crawl.py                 爬虫核心 (认证、爬取)
  run_crawler.py           主入口 (爬取 → DB → 文件)
  config.example.json      配置模板 (复制为 config.json 使用)
  setup_windows_task.bat   Windows 定时任务安装脚本

使用:
  1. 复制 config.example.json → config.json，填入 Cookie 和机组ID
  2. 确保 .env 已配置数据库连接信息
  3. 首次运行建表: python scripts/crawler/run_crawler.py --init-db
  4. 日常运行:     python scripts/crawler/run_crawler.py
  5. 定时任务:     右键管理员运行 setup_windows_task.bat

数据库:
  epf_unit_data_96    — 机组级96点电价/出力数据 (新表)
  epf_market_data_96  — 全省级96点市场特征数据 (已有表)
