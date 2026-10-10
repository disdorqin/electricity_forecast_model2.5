# Crawler build specs

本目录保存可进入 Git 的 **canonical PyInstaller spec**。`dist/build_artifacts/` 只作为本机构建工作区、venv 和中间产物，不再承担源码版本真相。

当前规格：

- `crawl_96_auto_v10.spec` → `scripts/crawler/apps/crawl_96.py` → `crawl_96_auto_v10.exe`
- `crawl_disclosure_aux_v1.spec` → `scripts/crawler/apps/crawl_aux.py` → `crawl_disclosure_aux_v1.exe`

Windows 打包必须使用 `dist/build_artifacts/venv_build`（Python 3.11 / OpenSSL 3.0.13 口径），禁止用 OpenSSL 3.6.x 环境替代。打包后仍需核对 EXE 自报 `BUILD_VERSION`、文件大小和 SHA256，并把结果写入 `dist/crawler/爬虫版本台账与发布治理.md`；本地 smoke 不能替代公司真机结果。

实际凭据不写入 spec。`dist/crawler/config.json`、`db_config.json`、AUX 真实配置和浏览器 profile 仍属于部署状态，不应由构建规格携带。
