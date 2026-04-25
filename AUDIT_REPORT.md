# CTO-Level Audit Report — FootballQuantEngine
**Date:** 2026-04-25  
**Branch:** `claude/cto-level-audit-vJGLE`  
**Scope:** Full codebase (~34 k LOC, 200+ Python modules)  
**Methodology:** 4 parallel specialist agents — Security · Architecture · Test/CI · Quant Correctness

---

## Executive Summary

| Domain | Grade | Top Risk |
|--------|-------|----------|
| Security | C+ | Plaintext credential storage; ZIP path traversal (fixed) |
| Architecture | B- | Duplicate modules (`analysis/` vs `analytics/`); God classes |
| Test / CI | D+ | 68 % codebase untested; API secrets exposed to all branches |
| Quant Correctness | B | Temporal leakage in backtest SQL (fixed); naive Kelly ignores correlation |

**5 bugs fixed in this branch** (see §Applied Fixes).  
**1 claimed bug invalidated** by independent verification (Poisson defense normalisation — see §Quant).

---

## Applied Fixes

| # | File | Issue | Severity |
|---|------|-------|----------|
| 1 | `training/backtest_engine.py:28-39` | Temporal leakage — `MAX(timestamp)` included post-kickoff odds; rewritten to `timestamp < f.match_date` | CRITICAL |
| 2 | `repo_update_engine.py:133` | ZIP path traversal — `extractall(".")` with no path check; added member-by-member traversal guard | HIGH |
| 3 | `data/clubelo_collector.py:9` | Plain HTTP to ClubElo API; changed to HTTPS | HIGH |
| 4 | `ai/feature_generator.py:629` | Silent `except: pass` swallowed MI feature-selection failures; now logs a warning with key name and error | HIGH |
| 5 | `.gitignore` | `settings.json` (contains API keys + Telegram token) was not ignored; added | HIGH |
| 6 | `training/local_automl.py:473` | `pickle.load` fallback had no warning; added `logger.warning` flagging untrusted-source risk | MEDIUM |

---

## 1 — Security

### CRITICAL

**Unsafe pickle deserialization** (`training/local_automl.py:473`)  
`pickle.load()` can execute arbitrary Python on deserialisation. The primary `joblib` path is safe; the pickle fallback now emits a warning. Long-term: remove the fallback entirely or add a SHA-256 checksum guard on the file.

### HIGH

**Plaintext credential storage** (`config/settings_manager.py:47-56`)  
`settings.json` stores `api_football_key` and `telegram_token` in cleartext JSON. `settings.json` was not in `.gitignore` (now fixed). Recommended path: enforce secrets exclusively via environment variables; never write them to disk.

```python
# config/settings_manager.py — never persist secrets
def save_settings(settings):
    data = {
        "league_id": settings.league_id,
        "season": settings.season,
        # api keys/tokens come only from env vars
    }
```

**ZIP path traversal** (`repo_update_engine.py:134`) — **FIXED**.  
**HTTP ClubElo API** (`data/clubelo_collector.py:9`) — **FIXED**.

### MEDIUM

**Telegram token in URL** (`notifications/telegram_notifier.py:164,300,325,349`)  
Token is interpolated into the API URL. If the URL appears in logs or error messages the token is exposed. Mask it: `url.replace(self._token, "***")` before logging.

**No dependency vulnerability scanning**  
Add to CI: `pip-audit` or `safety check`. Neither is present.

---

## 2 — Architecture & Code Quality

### CRITICAL — Duplicate module directories

`analysis/` (14 files) and `analytics/` (4 files, 520+ LOC) both contain `league_predictability.py` and `market_inefficiency` logic with overlapping responsibilities. Import ambiguity and diverging implementations are certain as the codebase grows.

**Action:** consolidate into `analytics/`; remove `analysis/`; update all imports.

### HIGH — God classes

| File | Lines | Problem |
|------|-------|---------|
| `engine/markowitz_optimizer.py` | 801 | Portfolio maths + correlation builder + optimiser fused |
| `engine/gaussian_copula.py` | 863 | Bet builder + copula evaluation + correlation in one class |
| `training/local_automl.py` | 744 | Monolithic (no class wrapper at top level) |
| `dashboard/analytics_dashboard.py` | 742 | UI + computation coupled |
| `engine/bayesian_live.py` | 780 | Live inference + feature prep + state management |

Split each along single-responsibility lines. Target: no module > 400 LOC.

### HIGH — Silent failures

`ai/feature_generator.py:629` — **FIXED** (now logs).  
Multiple `except Exception: pass` patterns remain in `training/local_automl.py` and `ui/offline_import_window.py`. Add at minimum `logger.warning(..., exc_info=True)`.

### HIGH — Type hint coverage

~50 % of functions have return annotations; parameter types are largely absent. No `mypy` config or CI step enforces this.

```toml
# pyproject.toml — add
[tool.mypy]
python_version = "3.10"
disallow_incomplete_defs = true
no_implicit_optional = true
warn_no_return = true
ignore_missing_imports = true
```

### MEDIUM — Loose dependency pins

All packages use `>=` with no upper bound. A pandas 3.0 or requests 3.0 release would silently break builds.

```
# requirements.txt — add upper bounds
pandas>=2.0.0,<3.0
requests>=2.32.0,<3.0
numpy>=1.26.0,<2.0
scikit-learn>=1.4.0,<1.6
```

### MEDIUM — Database layer

- `db_manager.connect()` returns a raw connection with no context manager — callers must manually `commit/rollback/close`.
- No migration framework (Alembic). Schema changes require ad-hoc SQL.
- `config/constants.py`: `DATABASE_NAME` is a bare filename (relative path); should be `Path(__file__).parent.parent / "quant_engine.db"`.

**Recommended:**

```python
# database/db_manager.py
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

### MEDIUM — Magic numbers

`quant/services/quant_engine.py:126-131` contains undocumented thresholds (`0.20`, `0.60`, `0.4`) for xG support scoring. These belong in `config/constants.py` with a short comment explaining derivation.

---

## 3 — Test Coverage & CI/CD

### CRITICAL — Coverage gaps

68 % of modules have **zero tests**:

| Module | Files | Tests |
|--------|-------|-------|
| `analysis/` | 14 | 0 |
| `database/` | 5 | 0 |
| `models/` | 9 | 0 |
| `strategies/` | 4 | 0 |
| `features/` | 6 | 0 |
| `export/` | 3 | 0 |
| `notifications/` | 3 | 0 |
| `simulation/` | 3 | 0 |
| `analytics/` | 4 | 0 |

Quant core: `quant/` has 81 files and only 11 tests (13 % module coverage).

### CRITICAL — Tests in `/test/` and `quant/tests/` never run

`pytest.ini` sets `testpaths = tests`. The 3 files in `/test/` and the 6 files in `quant/tests/` are **never executed by CI**.

**Fix:** either move them into `/tests/` or extend `testpaths`:
```ini
[pytest]
testpaths = tests test quant/tests
```

### CRITICAL — API secrets injected on all branches

`build.yml:5` triggers on `branches: ["**"]` and exposes `API_FOOTBALL_KEY` via `env:`. A fork PR can print it to logs.

```yaml
# build.yml — restrict secret injection
jobs:
  build:
    if: github.ref == 'refs/heads/main' || github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository
    env:
      API_FOOTBALL_KEY: ${{ secrets.API_FOOTBALL_KEY }}
```

### CRITICAL — No dependency lock file

Every build re-resolves transitive dependencies. Non-reproducible builds; cannot audit what was actually installed.

```bash
pip install pip-tools
pip-compile requirements.txt -o requirements.lock
pip-compile requirements-dev.txt -o requirements-dev.lock
```

### HIGH — Automated commits bypass CI

`repo-update.yml:86` pushes with `[skip ci]` to main. This is the highest-risk bypass: untested, auto-generated code lands on main.  
Also: `--force-with-lease` in the same workflow can silently drop commits.

### HIGH — No mocking strategy

Fewer than 2 % of tests mock external dependencies. Tests that call real APIs are environmentally fragile and cannot run offline.

```python
# Example pattern needed throughout
@pytest.fixture
def mock_api_client(mocker):
    return mocker.patch("data.api_client.ApiClient.get_fixtures", return_value=[...])
```

### HIGH — Zero error-path tests

No `pytest.raises()` call found in the entire test suite. No boundary value tests (zero, NaN, infinity). Financial engines must handle extreme inputs without silent wrong answers.

### MEDIUM — Three test directories

`/test/`, `/tests/`, `quant/tests/` — no documented rationale.  
**Target structure:**

```
tests/
  unit/        # isolated, mocked
  integration/ # cross-module
  ui/          # Qt/PySide6
  exe/         # installer smoke
```

---

## 4 — Quant Correctness

### ⚠ Claimed bug INVALIDATED — Poisson defense normalisation

The audit agent claimed `defense_home` and `defense_away` use inverted normalisation denominators. After independent algebraic verification this is **incorrect** — the code is right:

```
defense_home = avg_home_conceded / league_avg_away_goals
```

This is the correct Dixon-Coles-style normalisation. Proof:

```
λ_away = league_avg_away_goals × attack_away × defense_home
       = league_avg_away_goals × (avg_away_scored/league_avg_away_goals)
                               × (avg_home_conceded/league_avg_away_goals)
       = avg_away_scored × avg_home_conceded / league_avg_away_goals
```

For an average team (`avg_home_conceded = league_avg_away_goals`):
```
λ_away = avg_away_scored × 1.0 ✓
```

**Do not change this code.**

### CRITICAL — Temporal leakage in backtest SQL — FIXED

`backtest_engine.py:28-39` joined `MAX(timestamp)` from `odds_history` with no constraint against `match_date`. Post-kickoff odds (live market) could therefore enter backtest evaluation, inflating simulated ROI.

**Fix applied:** correlated subquery now filters `timestamp < f.match_date`.

### HIGH — Naive Kelly ignores bet correlation

`quant/value/kelly_engine.py` implements single-bet Kelly. When two bets on the same match are accepted (e.g., home win + over 2.5), Kelly is applied independently to each, potentially allocating 2× the correct stake.

`engine/markowitz_optimizer.py` exists and handles correlation-aware sizing but `StakePolicy` bypasses it.

**Fix:** route `StakePolicy` through `MarkowitzOptimizer` for any portfolio with > 1 active bet on the same fixture.

### MEDIUM — Covariance matrix not guaranteed PSD

`markowitz_optimizer.py:443-464` builds a correlation matrix using piecewise rules (`same_match_rho = 0.60`, `within_group_rho = 0.35`). Arbitrary piecewise assignment can produce a non-positive-semi-definite matrix, making variance negative or optimisation diverge.

```python
# Add before optimisation
import numpy as np
eigenvals = np.linalg.eigvalsh(cov)
if eigenvals.min() < 1e-8:
    cov += np.eye(len(cov)) * (1e-8 - eigenvals.min())
```

### MEDIUM — Calibration is linear shrinkage, not Platt scaling

`quant/models/calibration.py:3-35` shrinks probabilities 8 % toward uniform 1/3. This is ad-hoc. True calibration requires fitting a logistic (Platt) or isotonic regression on a held-out validation set.

The shrinkage constant `0.08` has no documented origin. It should be cross-validated.

### MEDIUM — Gaussian copula: no tail dependence

`engine/gaussian_copula.py` uses Gaussian copula which assumes zero tail dependence. In football, extreme outcomes (e.g., multiple upsets on the same matchday) exhibit positive tail dependence. The model will underestimate joint loss probability on correlated multi-leg bets.

Long-term solution: offer Clayton or Gumbel copula as an alternative for multi-leg builders.

### LOW — ELO K-factor and home advantage not calibrated

`quant/models/elo_engine.py`: K=20, home_advantage=55 are reasonable starting points but were not derived from data. A 500-match backtest calibration grid-search should determine optimal values per league.

### LOW — Monte Carlo: no variance reduction

`simulation/monte_carlo.py` uses raw Poisson sampling (20 000 simulations). For most probability estimates, direct PMF enumeration (as in `dixon_coles_engine.py`) is both faster and exact. If simulation is needed for complex multi-outcome props, add antithetic variates or control variates.

---

## Remediation Roadmap

### Week 1 — Stop the bleeding
1. Rotate any API keys/tokens that may have been committed or logged (check git history).
2. Fix secret injection scope in `build.yml` (restrict to main branch).
3. Add `requirements.lock` (pin transitive deps).
4. Remove `[skip ci]` + `--force-with-lease` from `repo-update.yml`.
5. Add `pytest.ini` `testpaths` to pick up `/test/` and `quant/tests/`.

### Week 2-3 — High-impact quality
6. Consolidate `analysis/` → `analytics/`.
7. Add context-manager wrapper in `db_manager.py`.
8. Route `StakePolicy` through `MarkowitzOptimizer` when multiple bets per fixture.
9. Add PSD regularisation to Markowitz covariance matrix.
10. Add 30+ error-path tests (`pytest.raises`) for core quant modules.
11. Add mocking fixtures; eliminate live-API dependency from unit tests.

### Week 3-4 — Type safety & calibration
12. Add `mypy.ini` and `mypy` CI step.
13. Replace linear shrinkage in `calibration.py` with Platt scaling fitted on a validation split.
14. Centralise magic numbers (xG thresholds, Kelly scale, ELO params) in `config/constants.py`.
15. Cross-validate ELO K-factor and home_advantage per league.

### Week 5-6 — Coverage sprint
16. Target 50 %+ module coverage for `quant/`, `engine/`, `analytics/`, `database/`.
17. Add `conftest.py` fixtures (sample DataFrames, mock API client, in-memory DB).
18. Add parametrised boundary tests (zero, NaN, infinity, 0.0/1.0 probabilities).

### Week 7+ — Hardening
19. Add SLSA provenance + checksum verification on release artifacts.
20. Add `pip-audit` / `safety` step to CI.
21. Add Clayton/Gumbel copula option to `gaussian_copula.py`.
22. Add Monte Carlo variance reduction or replace with exact PMF enumeration.

---

*Report generated by Claude Code CTO Audit — 4 specialist agents, 211 tool calls.*
