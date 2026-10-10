# 应用层（apps）

> 本目录存放**各具体爬虫的应用层入口**。业务采集/解析仍留在 `../collect/`，但入口已经不再只是薄转发：当前 96/AUX 都在这里接入 resilience preflight、doctor、失败诊断和有界认证恢复。实现模块路径保持不变，避免打包与导入被改坏。

## 现有入口

| 入口 | 转发到 | 说明 |
|---|---|---|
| `crawl_96.py` | `collect/crawl_96_local.py:main` | 96 点市场数据 |
| `crawl_aux.py` | `collect/crawl_disclosure_aux.py:main` | 辅助信息披露 |
| `tools/platform_review_update.py` | `tools/platform_review.py` | 演示站复盘数据（**与 PMOS 无关**） |

## 运行方式

```bash
# 源码模式（推荐走 apps，才能包含当前 resilience/doctor 行为）
python scripts/crawler/apps/crawl_96.py --date 2026-09-24
python scripts/crawler/apps/crawl_aux.py --date 2024-08-17 --lookback 960 --source all-designed

# 生产模式仍用打包好的 exe（入口不变）
dist/crawler/crawl_96_auto_v10.exe
dist/crawler/辅助信息披露爬虫/crawl_disclosure_aux_v1.exe
```

## 新增一个爬虫怎么做

1. 在 `../collect/` 写实现模块（含 `main(argv=None) -> int`）；
2. 在本目录新建 `crawl_xxx.py`，复用现有应用层结构；是否接入 Guardian/doctor 由业务需要决定，但不要把业务 parser/DB writer 塞进入口；
3. 如需打包，在 `../build/` 新建 tracked canonical spec，`Analysis` 应指向 **apps 入口**，而不是绕过应用层直指实现模块；
4. 在 `../../../docs/crawler/CODEGRAPH_爬虫源码全局关系图.md` 补一条依赖记录。

## 为什么保留 apps/collect 分层

`collect/crawl_96_local.py` 内部依赖：

```python
BASE_DIR = Path(__file__).resolve().parents[3]      # 推导仓库根
sys.path.insert(0, BASE_DIR / "scripts" / "crawler")  # 支持扁平 import
try:    from scripts.crawler.collect.crawl import ...
except: from crawl import ...                        # 打包 / 源码 双分支
```

薄封装**不改变实现模块的 `__file__`**，因此上面三项全部保持原样。
`apps/` 与 `collect/` 同级 → 薄封装自己的 `parents[3]` 也指向同一仓库根。
