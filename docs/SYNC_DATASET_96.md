# SYNC_DATASET_96 — 96-Point Database Synchronization

> Native 96-point (15-minute) local mirror synchronization for
> `electricity_forecast_model2.5`. This feature downloads required_core 96-point
> tables **read-only** from the remote MySQL database to the local dev machine.
> It does **not** modify the crawler, schemas, or any remote rows.

---

## 1. CLI Design

Resolution is selected with `--resolution`. The default remains **hourly**, so
all pre-existing commands are unchanged.

```powershell
# --- Hourly (legacy, unchanged) ---
python main.py --pipeline sync_dataset --sync-source db --force-sync
python main.py --pipeline sync_dataset --resolution hourly --sync-source db --force-sync

# --- 96-point (new) ---
python main.py --pipeline sync_dataset --resolution 15min --sync-source db --sync-mode full --force-sync
python main.py --pipeline sync_dataset --resolution 15min --sync-source db --sync-mode incremental
```

| Argument | Values | Default | Notes |
|---|---|---|---|
| `--resolution` | `hourly`, `15min` | `hourly` | Routes to legacy (`sync_data`) or 96-point (`sync_data_96_core`). |
| `--sync-source` | `auto`, `db`, `http`, `local` | `auto` | For 15min: `http` is **NOT supported** (no 96-point HTTP endpoint). Use `db`/`auto`/`local`. |
| `--sync-mode` | `full`, `incremental` | `full` | 15min only. |
| `--sync-overlap-days` | int | `7` | Overlap window for incremental re-pull (captures late backfills). |
| `--include-extended` | flag | off | Also downloads optional extended 96-point tables. |
| `--force-sync` | flag | off | Ignored for 15min (15min always refreshes the local mirror). |

Backward compatibility: omitting `--resolution` keeps hourly behavior for `db`,
`http`, `local`, `auto`. The hourly regression test (`scripts/check_sync_dataset.py`)
passes **41/41** after this change.

---

## 2. Source Tables & Classification

Determined from `dist/audit_output/*` (live audit) and the 2026-07-28 sync.

| Table | Class | In first dataset? | Notes |
|---|---|---|---|
| `epf_market_data_96` | `required_core` | ✅ Yes | Market grid features (13 fcast + 13 actual). No price. |
| `epf_unit_data_96` | `required_core` | ✅ Yes | Unit-level 96-point DA/RT clearing prices (the targets). |
| `epf_market_congestion_96` | `optional_extended` | ⚠️ With `--include-extended` | 阻塞 data (~806k rows). |
| `epf_market_tie_line_96` | `optional_extended` | ⚠️ With `--include-extended` | 联络线 data (~1.34M rows). |
| `daily_weather_*` / `hourly_weather_*` | `metadata_only` | ❌ Not 96-point core | Reserved for the ECMWF/weather design (separate task). |

The first production-compatible local dataset contains **both required_core
tables**; optional tables are downloaded to separate raw files only when
`--include-extended` is set.

---

## 3. Output Layout

```
data/remote_96/
├── raw/                      # transparency / portability
│   ├── epf_market_data_96.csv.gz
│   └── epf_unit_data_96.csv.gz
├── parquet/                  # model work (preferred)
│   ├── epf_market_data_96.parquet
│   └── epf_unit_data_96.parquet
├── metadata/
│   ├── schema_inventory.csv
│   ├── column_dictionary.csv
│   └── source_table_manifest.json
└── (mirror root; not committed — gitignored)

outputs/data_sync_96/
├── sync_manifest.json        # full manifest (task §9 fields)
└── sync_report.md            # human-readable summary
```

- Raw remote tables remain distinct from processed model data.
- Hourly files are **never** overwritten.
- Market and unit tables are never combined during raw sync.
- Original remote column names/types are preserved (varchar kept as `object`).
- Writes are **atomic**: temp file + rename; a failed sync does not destroy the
  prior valid local mirror.
- `data/` and `outputs/` are already in `.gitignore` → nothing synced is committed.

---

## 4. Synchronization Modes

### Full (`--sync-mode full`)
Downloads the complete history for all requested tables. Used by a new team
member or a clean machine. Bounded/streamed `SELECT` (`pymysql` `SSCursor`,
`arraysize=2000`) — no `ORDER BY` over the whole table, no full-table lock.

### Incremental (`--sync-mode incremental`)
1. Reads the latest local `market_date`.
2. Queries the remote from `latest_local - overlap_days` (default 7) to capture
   late backfills of recent days.
3. Concatenates with the existing local mirror and **deduplicates by the true
   key** (`(market_date, period_no)` for market; `(market_date, period_no,
   unit_id)` for unit).
4. Atomically replaces local files + updates the manifest.

> Actual values may be backfilled later on the remote — the overlap window is why
> we re-pull recent days instead of assuming the local latest-day is final.

---

## 5. Manifest (`outputs/data_sync_96/sync_manifest.json`)

Records (per task §9): `resolution, source, sync_mode, started_at, completed_at,
status, database_server_version, tables_requested, tables_succeeded,
tables_failed, local_paths, rows_per_table, min/max_market_date_per_table,
distinct_days_per_table, complete_96_days_per_table, incomplete_days_per_table,
duplicate_key_count, latest_complete_day,
latest_non_null_date_per_critical_column, remote_row_count, local_row_count,
row_count_match, schema_fingerprint, data_fingerprint_or_checksums, overlap_days,
warnings, errors, classification, records[]`.

**No credentials, hosts, cookies, or passwords** are written.

---

## 6. Data-Integrity Validation (task §10)

After sync, each table is validated. Measured result from the 2026-07-28 full
sync (both tables → all checks PASS):

| Check | `epf_market_data_96` | `epf_unit_data_96` |
|---|---|---|
| start_date ≤ 2022-01-01 | ✅ 2022-01-01 | ✅ 2022-01-01 |
| max_date ≥ audit baseline | ✅ 2026-07-29 ≥ 2026-07-27 | ✅ 2026-07-18 ≥ 2026-07-18 |
| every complete day = 96 periods | ✅ 1,671 days | ✅ 1,660 days |
| no duplicate keys | ✅ 0 | ✅ 0 |
| `period_no` within 1..96 | ✅ | ✅ |
| p1/p96 interval-end semantics | ✅ | ✅ |
| remote/local row-count match | ✅ 160,416 / 160,416 | ✅ 159,360 / 159,360 |

The live DB may be **newer** than the audit baseline (market now reaches
2026-07-29) — that does not fail validation. A local mirror with **fewer** rows
than the remote query fails (unless explicitly explained).

---

## 7. Read-Only Guarantee

The 96-point path issues **only bounded/streamed `SELECT`** statements
(`COUNT(*)` for reconciliation, `SELECT *` with optional date-range `WHERE`).
No `INSERT/UPDATE/DELETE/REPLACE/CREATE/ALTER/DROP/TRUNCATE`, no crawler state
change, no schema modification. A dedicated test (`test_read_only_sql_only`)
asserts no mutating SQL is ever issued.

---

## 8. Tests

`scripts/check_sync_dataset_96.py` — **33/33 pass**. Covers:

- CLI: default hourly, explicit hourly, 15min parse, invalid resolution fails,
  full vs incremental mode, overlap-days parse.
- Config: existing config loads, missing config fails safely, secrets not logged.
- DB sync (mocks): full downloads all rows, incremental uses overlap, dedup works,
  multiple units preserved, atomic rollback on write error, partial-table failure
  preserves prior valid file, read-only SQL only.
- Integrity: 96 rows/day, period_no 1..96, p1/p96 mapping, no dup keys, manifest
  row-count match, remote/local match.
- Hourly regression: `scripts/check_sync_dataset.py` **41/41** unchanged.

Run:
```powershell
python scripts/check_sync_dataset_96.py
python scripts/check_sync_dataset.py
```

---

## 9. First-Run for a New Team Member

1. Clone the repo.
2. Provide the database configuration (`.env` with `DB_HOST/DB_PORT/DB_USER/
   DB_PWD/DB/DB_CONNECT_TIMEOUT`) — the same config the crawler uses.
3. `python main.py --pipeline sync_dataset --resolution 15min --sync-source db --sync-mode full --force-sync`
4. Local mirror appears under `data/remote_96/`; manifest under
   `outputs/data_sync_96/`.

No manual crawler-machine access or export copying is required.
