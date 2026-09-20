请基于当前 working tree 做最小修复，先阅读：

- `scripts/crawler/QCTC_SSO_TICKET_FIX_DESIGN.md`
- 最新 `dist/crawler/output_96/crawler.log`
- `dist/crawler/output_96/report.json`

只重点改 `scripts/crawler/collect/crawl.py`，`crawl_96_local.py` 只更新 BUILD_VERSION。

核心要求：

1. 已有 `:18080` QCTC 页面且 `bearer_present=True` 时，立即 `QCTC_CONTEXT_READY`，禁止再导航 `forecast10424`。
2. 没有 QCTC 上下文时，不再直接导航 `/psso?service=...`；严格复现门户：`POST /px-common-authcenter/sso/token` → 取 ticket → 打开 `https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin?ticket=...` → 扫描/切换 CDP target → 等 `sessionStorage.token`。
3. ticket/token/PIN/Cookie 值绝不写日志、report、raw；日志只写是否存在、状态码、path。
4. 保留现有 DOM fallback，但降级为 fallback。
5. Bearer 缺失或明确认证 401 时快速失败，不要继续刷几十天 401。
6. 不改账号密码、滑块、CFCA、UKey Win32 算法、字段映射、四类 QCTC API、CSV、DB。
7. 增加针对“已有 Bearer 不导航”“ticket SSO”“新 target 切换”“认证失败快速退出”的测试；原有测试必须继续通过。
8. 重新打 `crawl_96_auto_v3.exe`，备份旧 EXE，并报告修改文件、测试、BUILD_VERSION、新 EXE 路径/大小/时间。

保持最小 diff，不要重构整个认证系统。