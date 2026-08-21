# scripts/crawler — 爬虫模块

本目录包含两套互相独立的爬虫:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ① 国网PMOS爬虫 (`pmos.sd.sgcc.com.cn`) — 项目历史/主链路

自动爬取国网电力市场数据，写入 MySQL 数据库并更新本地文件。

| 文件 | 说明 |
|---|---|
| `crawl.py` | 爬虫核心 (认证、爬取) |
| `run_crawler.py` | 主入口 (爬取 → DB → 文件) |
| `config.example.json` | 配置模板 (复制为 config.json 使用) |
| `setup_windows_task.bat` | Windows 定时任务安装脚本 |

使用:
1. 复制 `config.example.json` → `config.json`，填入 Cookie 和机组ID
2. 确保 `.env` 已配置数据库连接信息
3. 首次运行建表: `python scripts/crawler/run_crawler.py --init-db`
4. 日常运行: `python scripts/crawler/run_crawler.py`
5. 定时任务: 右键管理员运行 `setup_windows_task.bat`

数据库:
- `epf_unit_data_96`    — 机组级96点电价/出力数据 (新表)
- `epf_market_data_96`  — 全省级96点市场特征数据 (已有表)

### 96 点本地全量爬虫（无 Python 电脑）

公司电脑使用 `dist/crawler/crawl_96_local.exe`。新版程序从 HAR5 验证过的接口分别
读取日前预测（`DaJyxxPlDa`）和实时实际（`DaJyxxPlYx`），两套来源不会互相回填。
核心预测和实际必须各自完整 96 点且通过同值污染审计，否则只保存原始包、不写入总表。

输出目录（位于 exe 同目录）:
```
output_96/
├─ pmos_96_全量.csv       # 通过审计的 96 点总表
└─ raw/YYYY-MM-DD.json    # 当日所有接口原始响应、来源和审计结果
```

首次全量爬取:
```text
crawl_96_local.exe --start 2022-01-01
```

替换历史旧表或污染日期时使用 `--force`；先用 `--date YYYY-MM-DD --force` 做单日
冒烟检查，再运行全量。机组日前/实时明细、备用、检修、断面、价格曲线等暂不确定
语义的字段保存在 `raw`，不会被错误广播到 96 个时段。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ② AI电力交易平台复盘更新 (`http://47.114.107.96`, 账号 user/user123)

> ⚠️ **注意: 这是「AI电力交易平台」自建演示站, 与上面的国网 PMOS 是完全不同的系统。**
> 它不走 PMOS 的 Cookie / 数据库同步, 而是账号密码登录 + 一次性导出接口。
> 该平台是**独立数据源**, 抓的是平台展示的「电价预测复盘」结果 (日前/实时电价 + 1.0/2.0 模型预测)。

| 文件 | 说明 |
|---|---|
| `platform_review.py` | 共享库 (登录 / 模型字典 / 复盘导出 / xlsx→csv) |
| `platform_review_update.py` | 命令行更新工具 (主入口) |

数据集 (稳定路径, 已放行 git 跟踪):
```
outputs/platform_review/
├─ 电价预测复盘.xlsx            原始导出 (详细数据 + 统计报告 两个 sheet)
├─ 电价预测复盘_详细数据.csv     逐小时: 实时/日前电价 + 各模型预测价
└─ 电价预测复盘_统计报告.csv     全量 + 分月综合准确率统计
```

用法:
```bash
# 更新到最新 (自动: 从数据集最早日期 ~ 今天)
python scripts/crawler/platform_review_update.py

# 指定抓取区间 (明细按 time 合并去重, 不影响区间外旧数据)
python scripts/crawler/platform_review_update.py --start 2026-01-01 --end 2026-08-06

# 只指定结束日期
python scripts/crawler/platform_review_update.py --end 2026-08-06
```
