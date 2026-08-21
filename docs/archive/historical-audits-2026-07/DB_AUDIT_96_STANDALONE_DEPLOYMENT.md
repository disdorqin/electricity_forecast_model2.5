# db_audit_96 自足版部署与构建手册

- **目的**：在“近乎裸机”的远端 Windows（无 Python/Conda/pip/Git/管理员权限/一般外网，仅可达远程 MySQL）上，一键完成 96 点数据只读深审。
- **本轮性质**：仅新增独立审计工具与文档；未修改生产代码/爬虫/既有 EXE/数据库/凭据；未开始 96 点实施。

## 1. 交付物清单

| 文件 | 位置 | 作用 |
|---|---|---|
| `output/db_audit_96_standalone.py` | 仓库 | 独立审计程序源码（标准库 + PyMySQL，零项目依赖，零 pandas/numpy/dotenv） |
| `output/db_audit_96.spec` | 仓库 | PyInstaller onefile 规格（UPX 关闭以降杀软误报；不打包任何数据文件） |
| `output/build_db_audit_96.ps1` | 仓库 | 开发机构建脚本（定位 Python→装构建依赖→源码自检→onefile 构建→冻结自检→可选 `-OneDir` 兜底包） |
| `output/package_db_audit_96.ps1` | 仓库 | 生成 `dist/db_audit_96_deployment.zip`（EXE+CMD+README.txt；**不含** .env/config.json） |
| `output/run_db_audit_96.cmd` | 仓库+ZIP | 双击启动器：定位 EXE→建 audit_output→运行→打印退出码与结果路径→pause |
| `docs/DB_AUDIT_96_STANDALONE_DEPLOYMENT.md` | 仓库 | 本手册 |

## 2. 构建（开发机，允许联网装构建依赖；远端零安装）

```powershell
# 项目根目录
powershell -ExecutionPolicy Bypass -File output\build_db_audit_96.ps1           # 主包 dist\db_audit_96.exe
powershell -ExecutionPolicy Bypass -File output\build_db_audit_96.ps1 -OneDir   # 追加兜底 dist\db_audit_96_portable\
powershell -ExecutionPolicy Bypass -File output\package_db_audit_96.ps1         # 打包 dist\db_audit_96_deployment.zip
```

等效手工命令：`python -m pip install pyinstaller pymysql` → `python -m PyInstaller --clean --noconfirm output\db_audit_96.spec`。
构建脚本内置两道冒烟：源码 `--selftest` 与冻结 EXE `--selftest`（均离线、不连库、不需 .env）。
**注意**：本分析会话所在的 Linux 沙箱无法交叉编译 Windows EXE 且包源被策略拦截，故 EXE 须由开发机构建；源码级 12 项自检已在本会话通过（见 §7）。

## 3. 部署（远端裸机）

1. 将 ZIP 内 `db_audit_96.exe`、`run_db_audit_96.cmd`（README.txt 随附）复制到既有爬虫目录（`.env`、`config.json` 旁），如 `D:\爬虫电网\`。
2. 双击 `run_db_audit_96.cmd`（或 CMD/PowerShell 运行 `db_audit_96.exe`）。
3. 结束后回传整个 `audit_output\` 文件夹。
远端只需要这四个文件：`db_audit_96.exe` + `run_db_audit_96.cmd` + 既有 `.env` + 既有 `config.json`。无任何运行时下载。

## 4. 配置发现（精确行为）

- 基准目录：冻结态取 `Path(sys.executable).parent`；源码态取脚本目录（**不依赖当前工作目录**）。
- 搜索顺序：① CLI `--env-file/--config-file` 显式路径 → ② EXE 同目录 → ③ 当前工作目录。识别 `.env`、`config.json`、`config`（兼容 Windows 隐藏扩展名）。
- 键位（与既有爬虫/项目实测一致，`utils/database_operate.py` 与 backfill 的键集）：主用 `DB_HOST / DB_PORT / DB_USER / DB_PWD / DB / DB_CONNECT_TIMEOUT`；同义兼容 `DB_USERNAME / DB_PASSWORD / DB_NAME / DB_DATABASE / MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE`（顶层或 `db`/`database` 子对象）。
- 优先级：**CLI 显式文件 > .env > config.json**。`config.json` 的 `cookie` 只登记入脱敏黑名单，**绝不读取使用、绝不输出**。
- 缺配置时打印可操作指引（含两行示例命令）并以退出码 3 结束，同时写 `AUDIT_FAILED.txt`。

## 5. 只读与安全保障

- SQL 守卫：剥离注释/空白后首词必须 ∈ {SELECT, SHOW, DESCRIBE, DESC, EXPLAIN, WITH}；语句中出现 INSERT/UPDATE/DELETE/REPLACE/CREATE/ALTER/DROP/TRUNCATE/RENAME/GRANT/REVOKE/CALL/LOAD/LOCK/UNLOCK/SET GLOBAL/SET SESSION/COMMIT 即拒绝（`create_time/update_time` 列名已豁免）；拒绝多语句。
- 会话：`autocommit=False` + `START TRANSACTION READ ONLY`（MySQL 5.7 支持）+ 结束 `ROLLBACK`；连接 `connect_timeout` 取 .env（默认 10s）、`read_timeout=180s`。
- 查询纪律：全程聚合/information_schema/有界 LIMIT；分位数用 COUNT+OFFSET 单行取值；不拉全表明细；不触碰任何认证类表。
- 脱敏：密码与 Cookie 进入全局擦除表，**所有**输出通道（控制台/日志/JSON/CSV/异常栈/失败标记文件）统一 `[REDACTED]`；host/user 掩码（`123.***.***.45` / `ep***er`）；`SHOW GRANTS` 仅保留权限动词，剥离账号/主机；凭据不进源码/spec/构建脚本/ZIP/文档。

## 6. 输出与退出码

`audit_output/`（可 `--output-dir` 改）：12 个审计工件（schema_inventory / table_profiles / column_profiles / target_candidates / feature_dictionary / history_alignment / field_availability / price_aggregation / feature_aggregation / anomalous_dates 各 CSV + live_summary.json + query_failures.csv）+ `db_audit_96.log` + 成功 `AUDIT_COMPLETE.txt` / 失败 `AUDIT_FAILED.txt`。CSV 一律 UTF-8-BOM（中文 Excel 直开）；日志 UTF-8。审计范围与既有 `_db_audit_96_live.py`（v4）完全一致（全库扫描、两表深审含逐列按年空值率与最早非空、单机组无过滤判定、p56 前当日 RT 入库计数、时延分位数、六法价格聚合、`answer_full_history_since_2022` 布尔答案等）。

退出码：**0** 完整成功 ｜ **2** 部分完成（存在被权限/超时拒绝的查询，明细在 query_failures.csv）｜ **3** 连接或配置失败 ｜ **4** 输出文件系统失败 ｜ **5** 内部异常。CMD 启动器逐码给出人话解释并 pause，窗口不会一闪而过。

## 7. 验证状态（诚实申报）

| 验证项 | 状态 |
|---|---|
| 源码模式 12 项自检（.env 解析/引号/优先级/脱敏/掩码/SQL 白名单 6 例/黑名单 8 例/CSV-BOM/CSV 脱敏/JSON/缺配置检测） | ✅ 本会话通过（ALL PASS，exit 0） |
| 缺配置启动路径（诊断输出 + AUDIT_FAILED.txt + exit 3） | ✅ 本会话通过 |
| onefile/onedir 构建、冻结 EXE 自检 | ⏳ 待开发机执行 `build_db_audit_96.ps1`（脚本内置） |
| 干净 Windows（无 Python/Conda/源代码、仅 EXE+CMD+测试 .env/config.json）冒烟 | ⏳ 待开发机/干净账户执行（清单见下） |
| 真机连库产出审计工件 | ⏳ 待部署执行；**在此之前不宣称 live 审计成功** |

干净机冒烟清单：能启动；找到同目录配置；创建 audit_output；不请求安装任何依赖；连接结果清晰显示；日志已脱敏；退出码符合文档。异常场景设计已覆盖：无 .env（exit 3）、错误凭据（exit 3 + 脱敏错误）、数据库网络不可达（connect_timeout 后 exit 3）、输出目录不可写（exit 4）、只读账号权限不足（exit 2 + query_failures.csv）、中文路径/含空格路径（Path API + UTF-8，chcp 65001）、任意工作目录（基于 exe 目录定位）、杀软拦截 onefile（onedir 兜底 + UPX 关闭）。

## 8. 已知限制

onefile 首启需解包到临时目录（数秒；受限环境用 onedir 兜底）；EXE 体积约 10–15 MB（Python 运行时 + PyMySQL）；未做代码签名（企业杀软可能提示，加白名单即可）；分位数为精确 OFFSET 法，超大表上稍慢但有界；`answer_full_history_since_2022` 判定阈值为“起点 ≤2022-01-05 且零缺日”，边界情况以缺失日清单人工复核。

## 9. 范围申明

本轮未修改：生产模型代码、爬虫及其既有 EXE、数据库 schema 与数据、凭据文件、14:00 截止、SGDFNet 锚；未训练模型；未开始 96 点兼容实施。审计程序结构性只读（§5），运行不写远程库任何记录。
