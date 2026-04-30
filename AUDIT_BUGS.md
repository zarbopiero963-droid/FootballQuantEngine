# CTO-Level Bug Audit — FootballQuantEngine
**Date:** 2026-04-29
**Branch:** `claude/cto-level-audit-vJGLE`
**Methodology:** 5 parallel specialized agents — security, math, performance, architecture, test coverage
**Total findings:** 22 (2 CRITICAL · 5 HIGH · 9 MEDIUM · 6 LOW)

---

## CRITICAL

### BUG-001 — Markowitz PSD guarantee swallowed silently
**File:** `engine/markowitz_math.py` lines 120–130
**Agent:** Mathematical Correctness

```python
# CURRENT — silently ignores failure
try:
    shift = abs(min_eigenvalue) + 1e-6
    matrix[i][i] += shift
except Exception:
    pass   # ← non-PSD matrix used unchanged; optimizer produces negative variance

# FIX
shift = abs(min_eigenvalue) + 1e-6
for i in range(n):
    matrix[i][i] += shift
```

**Impact:** Portfolio optimization runs on a non-positive-definite covariance matrix → variance can go negative → Sharpe gradient blows up → wrong bet sizing on every allocation cycle.

---

### BUG-002 — SQL injection via dynamic `ALTER TABLE` in migration runner
**File:** `database/db_manager.py` line 97
**Agent:** Security

```python
# CURRENT — table, col, col_type injected via f-string
conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

# FIX — allowlist with explicit raises (assert is stripped by python -O)
ALLOWED_TABLES = {"fixtures","snapshots","odds_history","predictions","clv_bets"}
ALLOWED_TYPES  = {"INTEGER","TEXT","REAL","BLOB"}
if table not in ALLOWED_TABLES:
    raise ValueError(f"Disallowed table: {table!r}")
if col_type not in ALLOWED_TYPES:
    raise ValueError(f"Disallowed column type: {col_type!r}")
if not col.replace("_","").isalnum():
    raise ValueError(f"Invalid column name: {col!r}")
conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
```

**Impact:** Any future caller that passes external data into `_MIGRATIONS` can drop tables or extract data. Pattern must not spread to other migration helpers.

---

## HIGH

### BUG-004 — Race condition: `LiveUpdater` metrics read/written without lock
**File:** `data/live_updater.py` lines 40–43, 100–114
**Agent:** Security + Architecture

`polls_total`, `polls_errors`, `last_poll_time`, `last_live_count` are written by the background polling thread and read by `health()` on the main/UI thread with no synchronization. Python's GIL does not protect compound operations (`+= 1` is not atomic at the bytecode level when interrupted between LOAD and STORE).

```python
# FIX — add self._lock = threading.Lock() in __init__, then:
def _poll_once(self) -> None:
    with self._lock:
        self.polls_total += 1
    try:
        ...
        with self._lock:
            self.last_live_count = len(live_fixtures)
            self.last_poll_time  = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        with self._lock:
            self.polls_errors += 1

def health(self) -> dict:
    with self._lock:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
```

---

### BUG-005 — Race condition: `OddsStream` `StreamHealth` updated without lock
**File:** `live/odds_stream.py` lines 289–298, 372–437
**Agent:** Security + Architecture

`StreamHealth` fields (`total_polls`, `consecutive_failures`, `total_errors`, `last_error_msg`, `total_changes`) are mutated in `_poll_source()` (background thread) and read via `health()` (main thread). The existing lock at line 357 only protects the copy operation, not the individual field updates in the polling path.

**Fix:** Acquire `self._lock` around every field mutation inside `_poll_source()`.

---

### BUG-006 — Sharpe gradient division-by-zero when covariance is non-PSD
**File:** `engine/markowitz_math.py` line 57
**Agent:** Mathematical Correctness

If BUG-001 is triggered (PSD fix silently skipped), portfolio variance ≤ 0. The Sharpe gradient computation divides by `sqrt(variance)` with no guard → `ZeroDivisionError` or `nan` propagates through the entire optimizer run.

**Fix:** Assert `portfolio_variance > 0` before the gradient step, or propagate the PSD fix from BUG-001 so the matrix is always valid by the time gradients are computed.

---

### BUG-007 — JSON loaded from user input and cache without schema validation
**Files:** `ui/manual_context_window.py`, `config/settings_manager.py`, `cache/cache_manager.py`
**Agent:** Security

```python
parsed = json.loads(content)      # no depth/size/type check
data   = json.load(f)             # settings.json
entry  = json.load(fh)            # cache files
```

A crafted deeply-nested JSON can exhaust the Python call stack (recursion limit ~1000). A large array JSON can exhaust memory. Neither `json.loads` nor `json.load` limit nesting depth.

**Fix:** After loading, validate against a schema (`jsonschema.validate`) or enforce `isinstance` checks on top-level keys + `len()` guards before processing.

---

### BUG-008 — Missing error context manager in `prediction_repository` and `odds_repository`
**Files:** `database/prediction_repository.py` lines 20–49, `database/odds_repository.py` lines 18–43
**Agent:** Security

```python
# CURRENT — connection not closed on exception
conn = connect()
cursor = conn.cursor()
cursor.execute(...)
conn.commit()
conn.close()   # never reached if commit raises

# FIX
from database.db_manager import get_db
with get_db() as conn:
    conn.execute(...)   # auto-commit + auto-rollback
```

**Impact:** On `SQLITE_BUSY` or disk-full, the connection leaks and the write is silently lost.

---

## MEDIUM

### BUG-003 — Missing input validation on `get_fixtures_by_status` (not SQL injection)
**File:** `database/fixtures_repository.py` lines 184, 205, 211
**Agent:** Security
**Note:** The query is already correctly parameterized (`?` placeholders, bound parameters) — this is not a SQL injection. The concern is missing allowlist validation: an unbounded or unknown status list can cause excessive IN clauses and silently accepts arbitrary strings.

```python
# CURRENT — parameterized but no allowlist
placeholders = ",".join("?" * len(statuses))
conn.execute(f"SELECT * FROM fixtures WHERE status IN ({placeholders})", statuses)

# FIX — validate against known status codes first
VALID = {"NS","FT","1H","HT","2H","ET","BT","P","INT","LIVE"}
if len(statuses) > 20:
    raise ValueError(f"Too many status values (max 20), got {len(statuses)}")
invalid = set(statuses) - VALID
if invalid:
    raise ValueError(f"Unknown status codes: {invalid}")
```

**Impact:** Without an allowlist, callers can pass thousands of values (DoS) or future code paths may introduce injection risk if the parameterization pattern is incorrectly copied.

---

### BUG-009 — Blocking DB/API calls on the UI main thread
**Files:** Multiple UI windows (`ui/dashboard_window.py`, `ui/fixture_selector.py`, others)
**Agent:** Performance

Long-running operations (fixture queries, API calls, model runs) are called directly from Qt event handlers without `QThread` or `concurrent.futures`. Result: UI freezes for 1–10 s on slow networks or large datasets.

**Fix:** Move all I/O to worker threads; emit results via Qt signals back to the main thread.

---

### BUG-010 — N+1 query pattern in CSV importer
**File:** `data/csv_importer.py`
**Agent:** Performance

For each row in the CSV a separate `INSERT OR IGNORE` is issued in a loop outside a transaction. A 500-row import issues 500 individual transactions.

**Fix:** Wrap the entire import loop in a single `with conn:` block and use `executemany()`.

---

### BUG-011 — DataFrame row-by-row mutation in backtest metrics
**File:** `training/backtest_metrics.py`
**Agent:** Performance

Iterates rows with `.iterrows()` and builds results with repeated `df.append()` (deprecated) or list concatenation. For 10 k rows this is O(n²) due to copy-on-append.

**Fix:** Build a list of dicts, then `pd.DataFrame(rows)` once at the end; replace `.iterrows()` with vectorised `.assign()` / `.apply()`.

---

### BUG-012 — Missing index on `downloaded_seasons` lookup column
**File:** `database/db_manager.py` (schema creation)
**Agent:** Performance

The `downloaded_seasons` table has no index on `(league_id, season)` which is the predicate used in every existence check before a data fetch. On a populated DB this becomes a full table scan.

**Fix:**
```sql
CREATE INDEX IF NOT EXISTS idx_dl_seasons_league_season
    ON downloaded_seasons(league_id, season);
```

---

### BUG-013 — Unlimited cache growth in `CacheManager`
**File:** `cache/cache_manager.py`
**Agent:** Performance

Cache entries are written to disk with TTL but the manager never evicts expired entries proactively. On a long-running session the cache directory grows without bound.

**Fix:** Add a periodic eviction sweep (e.g. on every 100th write or via a background timer) that deletes entries older than their TTL.

---

### BUG-014 — Unbounded callback list in `LiveUpdater`
**File:** `data/live_updater.py`
**Agent:** Performance

`self._callbacks` is a plain list with no deduplication or max-size cap. If a UI component registers a callback on every refresh (e.g. inside a slot connected to a signal), the list grows and every poll fires N duplicate callbacks.

**Fix:** Use a `set` of weak references, or enforce `if cb not in self._callbacks` on registration.

---

### BUG-015 — Circular import resolved via lazy import inside function
**File:** (identified by architecture agent across engine/ modules)
**Agent:** Architecture

At least one module uses a function-level `import` to break a circular dependency rather than restructuring the dependency graph. This hides the coupling, makes static analysis incomplete, and can cause `ImportError` at unexpected call sites.

**Fix:** Extract the shared dependency into a third module (e.g. `engine/types.py` or `engine/contracts.py`) and import from there at module level.

---

### BUG-016 — Poisson score matrix not renormalized after truncation
**File:** `quant/models/poisson_engine.py` lines 140–153
**Agent:** Mathematical Correctness

The score matrix is truncated at `max_goals` without renormalizing. The ~0.1 % probability mass above the truncation threshold is silently lost. Downstream modules that sum the matrix directly will undercount.

**Fix:** Either document that callers must renormalize, or add:
```python
total = sum(matrix.values())
if total > 0:
    matrix = {k: v / total for k, v in matrix.items()}
```

---

## LOW

### BUG-017 — API response fields accessed without `None` guard after `.get()`
**File:** `quant/providers/api_football_client.py` lines 141–165
**Agent:** Security

Chained `.get()` calls return `None` when an optional field is absent; the result is then passed to `int()` / `float()` / string operations without checking. A malformed API response triggers `TypeError` rather than a graceful fallback.

**Fix:** Add `or 0` / `or ""` defaults, or use `pydantic` models for response parsing.

---

### BUG-018 — SQLite database has no encryption at rest
**File:** `database/db_manager.py`
**Agent:** Security

The SQLite file is stored in plain text in `~/.footballquantengine/`. On a shared machine or if the filesystem is compromised, all prediction history, bet records, and API credentials cached in the DB are readable.

**Recommendation:** Evaluate `sqlcipher` for sensitive deployments. At minimum, restrict file permissions to the running user (`chmod 600`).

---

### BUG-019 — `simulate_joint_prob` passes Cholesky factor directly where matrix expected
**File:** `tests/property/test_copula_properties.py` line 95 (call site)
**Agent:** Test Coverage

`simulate_joint_prob(probs, corr, ...)` documents the second argument as the Cholesky lower-triangular factor `L`, but the test passes the raw correlation matrix. The function decomposes it internally — this is correct only because the test always uses `uniform_corr_matrix` which is already PSD. A non-PSD matrix passed by a future caller would fail inside `simulate_joint_prob` with a non-obvious error.

**Fix:** Rename the parameter to `corr_matrix` and decompose internally with explicit documentation, or decompose externally and document that callers must pass `L`.

---

### BUG-020 — `derandomize=True` CI profile not documented
**File:** `.github/workflows/quant-contracts.yml` / `conftest.py`
**Agent:** Test Coverage

The CI Hypothesis profile sets `derandomize=True`, generating the same deterministic example sequence on every run. This is intentional for reproducibility but means CI never explores new corners of the input space unless `max_examples` is increased.

**Recommendation:** Document this choice in `conftest.py` and consider running a weekly nightly job with `derandomize=False` and higher `max_examples=10000`.

---

### BUG-021 — No smoke tests for critical engine modules
**Agent:** Test Coverage

The following high-value engines have zero unit or integration tests:
- `engine/sentiment_engine.py` (NLP scoring → Elo adjustment)
- `engine/smart_money_tracker.py` (Betfair spike detection)
- `engine/latency_arb.py` (async HFT pipeline)
- `engine/weather_engine.py` (OpenWeatherMap → λ multiplier)
- `engine/lineup_sniper.py` (absent-player impact)
- `engine/bayesian_runner.py` (orchestration layer)

**Fix:** Add at minimum one contract test per engine: valid input → output has expected type and is in a sensible range.

---

### BUG-022 — Telegram token potentially logged in stack traces on debug builds
**File:** `notifications/telegram_notifier.py`
**Agent:** Security

`_safe_url()` masks the token in explicit log calls, but if the `requests` library raises an exception with the URL in the message (e.g. `ConnectionError: ... https://api.telegram.org/bot<TOKEN>/sendMessage`), Python's default traceback will print the full URL including the token.

**Fix:** Catch `requests.RequestException` before it propagates and re-raise with the URL already masked via `_safe_url()`.

---

## Priority Matrix

| ID | Severity | Effort | Risk | Fix First? |
|----|----------|--------|------|------------|
| BUG-001 | CRITICAL | 5 min | Wrong bet sizing | ✅ YES |
| BUG-002 | CRITICAL | 15 min | Data integrity | ✅ YES |
| BUG-003 | MEDIUM | 15 min | Missing allowlist | Before prod |
| BUG-004 | HIGH | 30 min | Data races → torn reads | ✅ YES |
| BUG-005 | HIGH | 30 min | Data races → torn reads | ✅ YES |
| BUG-006 | HIGH | 5 min | Optimizer crash | ✅ YES |
| BUG-007 | HIGH | 1 h | DoS / memory | Before prod |
| BUG-008 | HIGH | 30 min | Silent data loss | Before prod |
| BUG-009 | MEDIUM | 2 d | UX freeze | Sprint |
| BUG-010 | MEDIUM | 1 h | Import slowness | Sprint |
| BUG-011 | MEDIUM | 2 h | Backtest perf | Sprint |
| BUG-012 | MEDIUM | 10 min | Query slowness | Sprint |
| BUG-013 | MEDIUM | 2 h | Disk growth | Sprint |
| BUG-014 | MEDIUM | 30 min | Memory leak | Sprint |
| BUG-015 | MEDIUM | 1 d | Maintainability | Next quarter |
| BUG-016 | MEDIUM | 30 min | Probability leak | Sprint |
| BUG-017 | LOW | 2 h | Graceful errors | Next quarter |
| BUG-018 | LOW | 1 d | Data confidentiality | Next quarter |
| BUG-019 | LOW | 15 min | Test correctness | Sprint |
| BUG-020 | LOW | 30 min | CI coverage | Next quarter |
| BUG-021 | LOW | 3 d | Missing coverage | Next quarter |
| BUG-022 | LOW | 30 min | Token exposure | Before prod |

---

*Audit conducted: 2026-04-29 | Auditors: 5 parallel specialized agents (security, math, performance, architecture, test-coverage) | Methodology: static code analysis, full-repo traversal*
