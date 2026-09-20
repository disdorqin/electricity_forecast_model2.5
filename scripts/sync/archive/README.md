# 96 点历史同步程序归档

`legacy/` 中的 `backfill_actual_96.py` 和 `backfill_unit_data_96.py` 是历史回填
程序，保留用于审计和追溯，不是当前日常同步入口。当前日常链路使用：

```text
dist/crawler/crawl_96_auto_v3.exe
main.py --pipeline sync_dataset --resolution 15min --sync-source db
```

归档程序不能作为生产数据源，也不能绕过当前的实际值/预测值来源审计。
