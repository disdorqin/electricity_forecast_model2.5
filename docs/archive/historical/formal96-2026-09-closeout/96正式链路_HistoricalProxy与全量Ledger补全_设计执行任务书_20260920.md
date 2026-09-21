---
status: IMPLEMENTED_REVIEWED_READY_FOR_LOCAL_0817_TEST
date: 2026-09-20
scope: formal96-history-ledger-import-and-three-way-snapshot-routing
production_entry: python main.py --96 TARGET_DATE
live_protocol: formal96_dynamic_snapshot_v1
historical_proxy_protocol: formal96_historical_proxy_v1
implementation_scope:
  - full server ledger import
  - three-way snapshot routing
out_of_scope:
  - 2026-08-17..2026-09-19 backfill execution
  - model algorithm changes
  - learner/fusion changes
---

# 96点正式链路：全量 Ledger 并入 + 三态 Snapshot 路由设计

## 2026-09-20 独立审核结论

实现已完成并经 ChatGPT 独立代码/真实资产审核。AI 初版存在两个基础生命周期偏差，已在审核中最小修复：

1. 三态路由原先按 UTC 墙钟日期判断历史/当前；现改为 formal `--96`（除 `--finish`）先强制 DB sync，再使用 sync manifest 的 `latest_closed_day` 作为历史闭合事实。历史 Proxy 因此不会依赖墙钟日期；同时历史 target 也不会跳过 authoritative DB refresh。
2. formal96 `--force` 原先会删除整个 `runs/T/`，包括永久 LIVE Snapshot；现 formal96 force 只清理可重建 run 产物并保留 `runs/T/snapshot/`。24点/legacy force 的原有全清理行为不变。

审核证据：真实 full-source source `2025-12-18..2026-08-14` 240/240 日 audit PASS；dry-run 识别新增210日、重叠30日、current production wins、readiness PASS，且 `applied=false`。真实 2026-09-20 canonical LIVE Snapshot resolver PASS。针对性 route/retention/façade 测试 23/23 PASS；Dynamic smoke、Snapshot smoke、Preflight 15/15、24 target-day regression 16/16、相关 server/lifecycle pytest 35 passed、py_compile、git diff --check 均 PASS。

**当前允许的下一步仅为：用户本地终端手工运行 2026-08-17 做真实七模型验证。production full-source ledger 仍未 apply，8/17～9/19 仍未批量补跑。**

---

本阶段只做两件事：

1. 将旧服务器 2025-12-18～2026-08-14 的完整 prediction/actual 历史安全并入当前 production ledger。
2. 将 python main.py --96 T 的 Snapshot 路由升级为三态：已有真实 Snapshot 回放 / 无 Snapshot 历史代理 / 正式 LIVE。

除此之外尽量不改。

核心原则：

    路由改
    Snapshot source 改
    provenance/cache 跟着改

    FeatureViewBuilder 不分叉
    五模型数学逻辑不改
    training 不改
    learner 不改
    SLSQP/fuse/final 不改
    24点不改

---

# 1. 当前事实

## 1.1 旧服务器历史

Source：

    outputs/archive/server_backtest_96/
    original_server_prediction_20251218_20260814/
    ledger/

已核实：

    2025-12-18 ～ 2026-08-14
    240 天

    DA prediction:
      3 models x 96
      240/240 完整

    RT prediction:
      4 models x 96
      240/240 完整

    DA actual:
      240/240 x 96

    RT actual:
      240/240 x 96

这批数据必须作为长期 production-compatible historical state 保留，不重跑，不改写 provenance。

## 1.2 当前 production ledger

当前 prediction target days：

    2026-07-16 ～ 2026-08-16
    2026-09-20

因此 full-source merge 真正新增：

    2025-12-18 ～ 2026-07-15
    共 210 天

重叠：

    2026-07-16 ～ 2026-08-14

重叠区只审计，current production wins。

## 1.3 Snapshot 实际大小

当前真实 2026-09-20 Snapshot：

    values.parquet    约 31～37 KB
    manifest          约 2.7 KB

一份成功 Snapshot 约 40 KB。

因此每天永久保留一个 canonical LIVE Snapshot，年存储量约十几 MB，完全可接受。

冻结策略：

    每个真实 LIVE target day
    至少永久保留 1 份成功 canonical Snapshot

失败 attempt：
    可继续保留诊断 TTL
    不作为 historical replay source

---

# 2. 三态路由

用户入口保持不变：

    python main.py --96 T

不新增用户必须记忆的 --backtest / --historical 参数。

正式 ModeResolver：

    main.py --96 T
        |
        v
    ModeResolver
        |
        +-- A. T 有合法 canonical LIVE Snapshot
        |       |
        |       v
        |   STORED_LIVE_SNAPSHOT_REPLAY
        |
        +-- B. T 没有 Snapshot，且 T 已是 closed historical day
        |       |
        |       v
        |   HISTORICAL_PROXY_V1
        |
        +-- C. T 是当前正式预测目标
                |
                v
            LIVE_DYNAMIC
            -> DB sync
            -> 新 Dynamic Snapshot

之后三条路径全部汇合：

    Snapshot
      -> FeatureViewBuilder
      -> 原七模型腿
      -> prediction
      -> ledger_weight
      -> ledger_fuse
      -> final

---

# 3. Route A：历史回放且已有真实 Snapshot

这是最高优先级历史回放方式。

场景：

    某天曾经真正 LIVE 运行过
    之后用户再次执行 python main.py --96 那一天

例如：

    2026-09-20 已有真实 Dynamic Snapshot
    以后 2026-09-20 已经成为历史日
    再回放 2026-09-20

必须直接使用当时保存的 canonical LIVE Snapshot。

禁止：

    用今天数据库重新构造那一天 Snapshot
    用 Historical Proxy 覆盖真实 Snapshot
    取 snapshot 目录中“最新一个 attempt”冒充 canonical

## 3.1 canonical Snapshot 如何选

只能读取成功 run manifest / Stage1 provenance 明确绑定的：

    snapshot_id
    snapshot path
    serving protocol
    attempt id

必须验证：

    protocol = formal96_dynamic_snapshot_v1
    snapshot target = T
    snapshot decision day = T-1
    values.parquet 存在
    snapshot_manifest.json 存在
    hash / snapshot_id 一致
    当时 Stage1 DA3/RT4 provenance 可验证

验证通过：

    route = STORED_LIVE_SNAPSHOT_REPLAY

验证失败：

    不静默挑别的 attempt
    按明确错误 / 后续 historical policy 处理
    具体 fail-closed 由实现 AI 按现有 provenance contract 最小接入

## 3.2 为什么每天必须保存 Snapshot

真实 LIVE Snapshot 是未来严格历史复现最重要的资产。

因此以后每次正式 LIVE Stage1 成功：

    必须保留 canonical Snapshot
    不删除 values.parquet
    不删除 snapshot_manifest.json

FeatureView 仍可按现有逻辑成功后清理，因为 FeatureView 可由 Snapshot + base model store 重建。

Snapshot 不大，长期保留。

---

# 4. Route B：历史回放但没有真实 Snapshot

这才使用 HISTORICAL_PROXY_V1。

适用：

    历史 target T 已 closed
    但当时从未保存真实 LIVE Snapshot

历史缺失：

    RealityTmp vintage
    provisional RT vintage
    严格 ForecastData publication vintage

因此无法严格重建当时信息，只能构造 operational proxy。

## 4.1 proxy 固定边界

冻结：

    proxy_cutoff = 14:00
    visible_until_period = p56

注意：

    14:00 只属于 HISTORICAL_PROXY_V1
    LIVE_DYNAMIC 永远没有固定 14:00 cutoff
    STORED_LIVE_SNAPSHOT_REPLAY 更不使用 proxy cutoff

选择 p56：

    与旧服务器历史 RT cutoff 一致
    比当前下午最新数据更保守
    不依赖今天 crawler 的更新时间
    改动最小
    可审计

## 4.2 Historical Proxy Snapshot

定义：

    T = target day
    D = T-1

D 日：

    DA:
      p1..p96 保留

    final actual:
      p1..p56 保留
      p57..p96 mask

    RealityTmp:
      全 NaN
      因历史 vintage 不存在

    final RT:
      p1..p56 保留
      p57..p96 mask

T 日：

    ForecastData:
      使用历史 T 自己的 latest-state forecast

    actual truth:
      全 mask

    RT truth:
      全 mask

    DA truth:
      按当前 task contract

FeatureViewBuilder 不改：

    actual tail:
      ForecastData
      -> latest closed same-period
      -> recent closed-history median

    RT tail:
      same-day DA
      -> latest closed same-period
      -> median

Historical Proxy provenance：

    run_mode = HISTORICAL_PROXY_V1
    snapshot_kind = historical_proxy
    proxy_cutoff_period = 56
    actual_prefix_source = historical_final
    rt_prefix_source = historical_final
    historical_vintage = UNVERIFIED_LEGACY_VINTAGE
    strict_historical_vintage_proven = false

---

# 5. Route C：正式 LIVE

当前 Dynamic-v1 完全不改。

正式生产：

    DB sync
      -> 当时数据库有什么
      -> Snapshot 冻结什么
      -> FeatureViewBuilder
      -> 原七模型

不引入固定 cutoff。

必须持久保存本次成功 canonical Snapshot，供未来 Route A 回放。

---

# 6. 三态优先级

路由优先级必须明确：

    1. VALID_STORED_LIVE_SNAPSHOT
       -> STORED_LIVE_SNAPSHOT_REPLAY

    2. NO_STORED_SNAPSHOT + HISTORICAL_CLOSED
       -> HISTORICAL_PROXY_V1

    3. LIVE_TARGET
       -> LIVE_DYNAMIC

但正式当前预测不能因为目录里残留错误旧 snapshot 就被误判。

ModeResolver 必须结合：

    target day
    snapshot protocol
    success provenance
    target closed/live state
    run manifest

不能只判断“文件夹存在”。

--finish：

    不重新 ModeResolve
    永远复用当前 Stage1 snapshot/provenance

---

# 7. FeatureView 与五模型

本阶段设计判断：

    主要改路由
    其他模型层不动

这是正确的，但还必须同步 provenance/cache/assertion。

## 7.1 FeatureViewBuilder

三态都使用同一个 FeatureViewBuilder。

禁止创建：

    HistoricalFeatureViewBuilder
    ReplayFeatureViewBuilder

FeatureView 继续是唯一 serving visibility policy。

## 7.2 LightGBM

不新增 cutoff。

继续读取 FeatureView。

## 7.3 TimesFM

formal96 继续 exact。

不恢复 cutoff_safe。

## 7.4 TimeMixer

三态 serving 都保持：

    dynamic_serving = true

内部 fixed cutoff 只保留 training / legacy compatibility。

## 7.5 SGDFNet

三态 serving 都保持：

    dynamic_serving = true

禁止历史模式重新启用 model-local：

    RT after cutoff -> DA
    actual after cutoff -> forecast

这些已经由 Snapshot + FeatureView 完成。

anchor 不变：

    target T anchor = D 日 DA 96点

## 7.6 RT916

三态 serving 都保持 dynamic path。

asof_hour 不重新成为 serving safety source。

stride=24 不改。

## 7.7 总结

正式 manifest 必须继续表达：

    serving_visibility_source = FeatureViewBuilder

模型只消费已经安全路由后的 FeatureView。

---

# 8. Cache / provenance 必须跟路由一起改

虽然模型不改，但 cache identity 必须区分三态。

至少包含：

    target_day
    task
    model
    resolution
    resource_mode
    run_mode
    snapshot_id
    snapshot_kind
    serving_protocol
    proxy_policy_version
    proxy_cutoff_period

规则：

    STORED_LIVE cache
      不能给 HISTORICAL_PROXY

    HISTORICAL_PROXY cache
      不能给 LIVE

    LIVE cache
      不能给 proxy

    proxy policy 改变
      必须 cache miss

已有真实 LIVE Snapshot / prediction 不得被 proxy overwrite。

---

# 9. 全量旧服务器 Ledger 并入

本阶段第二项代码任务。

不要新建迁移器。

扩展：

    scripts/server/bootstrap_96_production_ledger.py

保留默认 deployment warm-start 行为。

增加显式 full-source 模式，例如：

    --history-scope full-source

具体 CLI 名称 AI 可最小调整。

## 9.1 full-source 行为

Source：

    2025-12-18 ～ 2026-08-14
    240 天

Current：

    2026-07-16 ～ 2026-08-16
    2026-09-20

新增：

    2025-12-18 ～ 2026-07-15
    210 天

Overlap：

    2026-07-16 ～ 2026-08-14

规则：

    source audit 全240天
    current production second
    current wins
    source 只补 current 不存在的 key
    overlap 只审计
    不修改旧 source prediction provenance
    不把旧服务器 history 改写成 Dynamic protocol

继续复用：

    staging
    readiness
    atomic promote
    manifest
    SHA256
    dry-run default
    explicit apply
    idempotency

manifest 至少增加：

    source_range
    source_days
    imported_days
    skipped_overlap_days
    conflicts
    final_range
    final_readiness

---

# 10. 当前阶段明确不做

本 AI 实施阶段不要执行：

    2026-08-17 ～ 2026-09-19 补跑
    任何批量 historical run
    服务器租赁/部署
    8/17 真模型本地测试

这些由用户在本阶段代码审核通过后手工进行。

因此文档和 AI 输出不要把“批量补跑成功”当本轮 completion gate。

---

# 11. AI 实施 TODO

只做以下两组。

## A. 全量 Ledger merge

A1. 复核 source/current
A2. 扩展 bootstrap full-source mode
A3. temp root dry-run
A4. 验证 current wins / idempotent / atomic
A5. 不 apply production，除非用户后续明确让执行

## B. 三态 Snapshot route

B1. ModeResolver
B2. canonical stored LIVE Snapshot resolver
B3. Snapshot retention contract
B4. HISTORICAL_PROXY_V1 p56
B5. LIVE_DYNAMIC 零回归
B6. FeatureView 共用
B7. cache/provenance 隔离
B8. 五模型 dynamic_serving contract 回归
B9. tests

本阶段 AI 完成后只返回：

    STATUS=READY_FOR_REVIEW

不要自行跑 8/17 真模型全链，也不要批量补历史。

---

# 12. 必须测试

## Ledger

- source 240天完整审计
- full-source dry-run 识别新增210天
- overlap current wins
- rerun idempotent
- promote failure 不污染 target
- 默认 learner_minimum 行为不回归

## Route

- valid stored LIVE snapshot -> Route A
- historical closed no snapshot -> Route B
- current live target -> Route C
- failed/incomplete snapshot 不冒充 Route A
- finish 不重判

## Snapshot retention

- LIVE success 后 canonical snapshot 永久存在
- run manifest 明确绑定 snapshot_id/path
- 多 attempt 时按成功 provenance 选，不按目录时间猜
- failed attempt 不作为 replay source

## Historical Proxy

- D actual p1..p56 retained
- p57..p96 masked
- tmp all NaN
- D RT p1..p56 retained
- tail masked
- D DA 96
- T truth invisible
- deterministic snapshot_id

## Models

- FeatureViewBuilder 是三态共同 visibility source
- TimesFM exact
- TimeMixer dynamic_serving=true
- SGDFNet dynamic_serving=true
- RT916 dynamic serving path
- SGDFNet anchor contract
- RT916 stride24

## Regression

- 当前 2026-09-20 LIVE Dynamic contract 不变
- deployment tests 不变
- 24点不变
- py_compile
- git diff --check

---

# 13. 禁止事项

禁止：

- 改模型数学结构
- 改模型池
- 改 learner 30/90/lag2
- 改 SLSQP
- 改 CPU2/GPU1
- 改 SGDFNet anchor
- 改 RT916 stride
- 重新让模型内部 fixed 14/15 成为 serving cutoff
- 用今天下午 Snapshot boundary 回放旧历史
- 删除成功 LIVE Snapshot
- 用 proxy 覆盖真实 LIVE 日
- 新建第二套 FeatureView
- 直接覆盖 production ledger
- 执行 8/17～9/19 补跑
- 改24点
- bulk restore/delete/stage

---

# 14. 本阶段完成标准

AI 只有在以下全部通过后才能：

    STATUS=READY_FOR_REVIEW

要求：

1. full-source merge 功能完成并 dry-run 测试通过；
2. 三态 route 完成；
3. canonical LIVE Snapshot retention / replay 完成；
4. Historical Proxy p56 完成；
5. cache/provenance 三态隔离；
6. FeatureView 与五模型无数学逻辑变化；
7. LIVE Dynamic-v1 regression PASS；
8. 24点 regression PASS；
9. docs 只更新实现事实，不宣称 8/17 已跑；
10. git diff --check PASS。

之后流程由用户/ChatGPT负责：

    AI 改完
      -> ChatGPT 独立审核
      -> 用户本地手工跑 2026-08-17
      -> ChatGPT 看结果
      -> 再决定服务器历史补跑

---

# 15. 本轮实现执行记录（2026-09-20）

- `bootstrap_96_production_ledger.py` 增加显式 `--history-scope full-source`。归档 2025-12-18..2026-08-14 先做全 240 日 canonical DA3/RT4 + actual96 审计，再在 OS 临时 staging 与当前 production 合并；冲突 key 由 current production 保留，默认不 promote。
- Dynamic-v1 路由增加 `STORED_LIVE_SNAPSHOT_REPLAY`、`HISTORICAL_PROXY_V1`、`LIVE_DYNAMIC` 三态。Proxy 使用 `formal96_historical_proxy_v1`、D/RT p56 前缀和 target truth mask，三态共用 FeatureViewBuilder。
- 成功快照 manifest 增加 values SHA256、route/run_mode/snapshot_kind；run manifest/Stage1 provenance 绑定 exact snapshot path，不按目录新旧猜测。--finish 继续只复用已验证 Stage1 provenance。
- 受控证据：full-source real archive dry-run `AUDIT_PASS`，source_days=240、imported_day_count=210、skipped_overlap_day_count=30、final_readiness=PASS；three-way route contract PASS；Dynamic smoke PASS；既有 lifecycle/as-of 22/22 PASS；未执行 2026-08-17 真模型或批量历史回测。
