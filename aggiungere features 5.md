# Audit Features da Aggiungere — Set 5 (Vista Consolidata per Categoria)
**Data:** 2026-04-29
**Nota:** Questo file è la vista tematica dei 20 moduli già auditati nei Set 1–4.
Ogni verdetto è basato su ispezione diretta del codice sorgente.
**Verdetti:** ✅ IMPLEMENTATO | ⚠️ PARZIALE | ❌ NON IMPLEMENTATO

---

## Categoria 1 — Modelli Matematici e Spaziali

### 1a. Expected Threat (xT) & Pitch Control
**File:** `engine/expected_threat.py`, `engine/pitch_control.py`
**Dettaglio:** → vedi `aggiungere features 4.md §3`

**Verdetto: ⚠️ PARZIALE**
- Griglia xT 16×12 (Karun Singh 2018) implementata con valori reali; Spearman pitch control con time-to-intercept su griglia 32×20
- **Non wired in produzione** — nessun caller esterno ai due file; richiede coordinate eventi (Opta/StatsBomb) non disponibili da API-Football
- Test thin (solo `test_coverage_boost2.py`)

---

### 1b. Bet Builder Exploiter (Copule Gaussiana, Clayton, Gumbel)
**File:** `engine/gaussian_copula.py`, `engine/copula_math.py`, `engine/copula_types.py`
**Dettaglio:** → vedi `aggiungere features 4.md §1`

**Verdetto: ✅ IMPLEMENTATO con test seri**
- Tre famiglie copula (Gaussian zero-tail, Clayton lower-tail, Gumbel upper-tail), 50k-path Monte Carlo, Cholesky puro Python, sampler stabili in log-space
- `evaluate_bet_builder()` confronta probabilità congiunta modello vs bookmaker → edge%
- **Test seri:** `tests/property/test_copula_properties.py` — 9 Hypothesis property tests, 200–300 esempi ciascuno (comonotonicity, independence limit, unit interval, BetLeg validation)
- Gap: correlazioni `DEFAULT_CORRELATIONS` hardcoded, non calibrate su dati storici

---

### 1c. Player Synergy via Graph Neural Networks (GNN)
**File:** `engine/network_synergy.py`
**Dettaglio:** → vedi `aggiungere features 3.md §4`

**Verdetto: ⚠️ PARZIALE**
- Grafo pass-network con PageRank puro (power-iteration, damping 0.85) implementato correttamente
- **Non è un GNN** — nessun `torch`, `torch_geometric`, nessun training su dati storici, nessun embedding appreso
- Non integrato in nessun pipeline di produzione

---

### 1d. Bayesian Live Updating
**File:** `engine/bayesian_live.py`, `engine/bayesian_runner.py`
**Dettaglio:** → vedi `AUDIT_REPORT.md §20`

**Verdetto: ✅ IMPLEMENTATO**
- Coniugazione Gamma-Poisson corretta: `λ | data ~ Gamma(α₀+k, β₀+t/90)`
- Aggiornamento su gol, tiri, cartellini; quiet-period evidence (nessun tiro in 15 min → riduzione α di `quiet_penalty = 0.08`)
- `BayesianAlert` emesso su shift λ > 25% soglia
- Test: `tests/unit/test_engine.py` — unit tests per Bayesian live
- **Gap critico (architettura):** `live_updater.py` e `bayesian_live.py` non sono orchestrati — i dati live arrivano nel DB ma nessun runner chiama `process_event()` in tempo reale (vedi `aggiungere features 1.md §6`)

---

## Categoria 2 — Asimmetria Informativa (Alternative Data)

### 2a. Lineup Sniper
**File:** `engine/lineup_sniper.py`
**Dettaglio:** → vedi `aggiungere features 3.md §1`

**Verdetto: ⚠️ PARZIALE**
- Algoritmo completo: `register_expected()`, `process_official()`, xG delta, 4 livelli di impatto (NONE/MINOR/MAJOR/CRITICAL), `BET AGAINST / FADE`
- **Nessun caller di produzione** — vive isolato; nessun poller API-Football lineups, nessun Twitter feed
- Zero test dedicati

---

### 2b. Motore NLP Insider (Sentiment)
**File:** `engine/sentiment_engine.py`
**Dettaglio:** → vedi `aggiungere features 3.md §2`

**Verdetto: ⚠️ PARZIALE**
- VADER + keyword scoring + recency decay + Elo adjustment (max −80 pts) implementati
- `NewsApiClient` per newsapi.org funzionante
- **Twitter/X assente** — nessun client; `source: "twitter"` è solo metadato
- Non wired nel pipeline pre-match; coverage test al 37%

---

### 2c. Travel & Fatigue Engine
**File:** `engine/travel_fatigue.py`
**Dettaglio:** → vedi `aggiungere features 1.md §5`

**Verdetto: ✅ IMPLEMENTATO**
- 5 componenti: distanza viaggio, giorni riposo, jet lag, altitudine, congestione fixture
- Output `FatigueReport` con moltiplicatori λ; integrato in `quant/features/feature_builder.py`

---

### 2d. Referee Profiling & Match Aggression
**File:** `engine/referee_analyzer.py`, `quant/models/referee_engine.py`
**Dettaglio:** → vedi `aggiungere features 2.md §9`

**Verdetto: ⚠️ PARZIALE — con bug critico nei dati**
- Modello Poisson cartellini corretto (Over 3.5/4.5/5.5 yellows, red card YES, Over 4.5 total)
- Integrato in `quant_engine.py`
- **Bug critico:** `get_referee_stats()` in `api_football_client.py` non incrementa mai `home_cards`/`away_cards` → `strictness = 0.0` per ogni arbitro → il modello non genera edge reale

---

### 2e. Weather Impact Engine
**File:** `engine/weather_engine.py`
**Dettaglio:** → vedi `aggiungere features 2.md §7`

**Verdetto: ✅ IMPLEMENTATO**
- Chiama OpenWeatherMap API per coordinate stadio; bracket vento/pioggia/temperatura con moltiplicatori λ addtivi; clamp a 0.50
- Integrato in `quant_engine.py`; freddo <5°C boostra peso Skellam nell'ensemble
- Gap: zero test dedicati

---

## Categoria 3 — Microstruttura del Mercato

### 3a. Order Book Spoofing & Imbalance Analysis
**File:** `engine/orderbook_analyzer.py`
**Dettaglio:** → vedi `aggiungere features 4.md §4`

**Verdetto: ⚠️ PARZIALE**
- Full depth back/lay ladder, `imbalance ∈ [−1, 1]`, `back_wall`/`lay_wall`, `compute_greenbook()` con formula esatta, `trend_imbalance()` con regressione lineare, `OrderFlowAlert`
- **Gap:** no UI/notifiche, autenticazione Betfair non gestita (richiede cert SSL), zero test

---

### 3b. Latency Arbitrage (Live Betting HFT)
**File:** `engine/latency_arb.py`
**Dettaglio:** → vedi `aggiungere features 4.md §2`

**Verdetto: ⚠️ PARZIALE**
- Architettura async completa (`asyncio`, `aiohttp`, `websockets`), `BetInstruction`, `LatencyProfiler`
- **Gap critico:** `FeedSimulator` è simulatore in memoria — nessun WebSocket reale a Opta/Radar; nessun client bookmaker per fire automatico
- Test thin

---

### 3c. Generatore di Quote Sintetiche (Cross-Market Arbitrage)
**File:** `engine/synthetic_odds.py`
**Dettaglio:** → vedi `aggiungere features 3.md §3`

**Verdetto: ✅ IMPLEMENTATO**
- `calibrate()` → grid search (λ_h, λ_a) → `exact_score_matrix()` → `btts_odds()` → `ou_odds()` → `find_arbitrage()`
- Matematica corretta; confronta sintetico vs bookmaker; filtra per edge ≥ 4%
- Gap: test thin, nessuna UI combo finder

---

### 3d. Cross-Chain Arbitrage (Web3 vs Fiat)
**File:** `engine/crosschain_arb.py`
**Dettaglio:** → vedi `aggiungere features 4.md §5`

**Verdetto: ⚠️ PARZIALE**
- `SXBetClient` (Polygon, no auth), `PolymarketClient` (CLOB), detection completa, formula stake ottimale corretta
- **Gap:** nessuna esecuzione automatica (no `web3.py`, nessun wallet, nessun `send_transaction()`); leg fiat non automatizzata

---

## Categoria 4 — Gestione Istituzionale del Rischio

### 4a. Markowitz Efficient Frontier (Portfolio Optimization)
**File:** `engine/markowitz_optimizer.py`, `engine/markowitz_math.py`, `engine/markowitz_types.py`
**Dettaglio:** → vedi `AUDIT_BUGS.md BUG-001, BUG-006`

**Verdetto: ⚠️ PARZIALE — con bug critico**
- `MarkowitzOptimizer`: gradient-based mean-variance optimization, efficient frontier, `kelly_naive` e `min_variance`
- `PortfolioAllocation`: variance, std, Sharpe ratio output
- Test: `tests/unit/test_engine.py` — unit tests
- **Bug critico (BUG-001):** `engine/markowitz_math.py` righe 129–130:
  ```python
  except Exception:
      pass   # ← PSD fix silently skipped → optimizer runs on non-PSD matrix
  ```
  Se NumPy non disponibile o la correzione PSD fallisce, la matrice di covarianza non è positiva definita → varianza negativa → stake sbagliati su ogni allocazione
- **Bug alto (BUG-006):** gradiente Sharpe divide per `sqrt(variance)` senza guard → ZeroDivisionError o NaN se variance ≤ 0 per via di BUG-001

---

### 4b. CLV (Closing Line Value) Tracking Matrix
**File:** `ui/clv_tracker_window.py`, `training/closing_line_builder.py`
**Dettaglio:** → vedi `aggiungere features 2.md §11`

**Verdetto: ✅ IMPLEMENTATO con test seri**
- UI PySide6 completa, DB persistence (`clv_bets`), formula `CLV% = (our_odds / closing_odds − 1) × 100`
- Test dedicati: `test_clv_tracker.py`, `test_clv_journal_service.py`, `test_app_clv_controller.py`, `test_clv_runner.py`
- Gap: closing odds inserite manualmente (nessun feed automatico)

---

## Scorecard Finale — tutti i 20 moduli

| Modulo | File | Verdetto | Fonte dettaglio |
|--------|------|----------|-----------------|
| Kelly / Fractional Kelly | `ranking/match_ranker.py`, `quant/value/kelly_engine.py` | **✅** | Set 1 §1 |
| Asian Handicap | `models/asian_handicap.py` | **⚠️** | Set 1 §2 |
| Sharp vs Soft / Steam | `engine/sharp_soft_tracker.py`, `live/odds_stream.py` | **⚠️** | Set 1 §3 |
| XGBoost + Time-Decay | `training/local_automl.py`, `training/time_decay.py` | **⚠️** | Set 1 §4 |
| Feature Eng. (Fatica+Assenze+Motivazione) | `engine/travel_fatigue.py`, `engine/lineup_sniper.py`, `quant/models/standings_engine.py` | **✅** | Set 1 §5 |
| Live In-Play (orchestrazione) | `data/live_updater.py` + `engine/bayesian_live.py` | **⚠️** | Set 1 §6 |
| Weather Engine | `engine/weather_engine.py` | **✅** | Set 2 §7 |
| Luck Index / xPTS | `engine/luck_index.py` | **✅** | Set 2 §8 |
| Referee Profiling | `engine/referee_analyzer.py` | **⚠️ BUG** | Set 2 §9 |
| Smart Money / Betfair Volume | `engine/smart_money_tracker.py` | **⚠️** | Set 2 §10 |
| CLV Dashboard | `ui/clv_tracker_window.py` | **✅** | Set 2 §11 |
| Lineup Sniper | `engine/lineup_sniper.py` | **⚠️** | Set 3 §1 |
| NLP Sentiment | `engine/sentiment_engine.py` | **⚠️** | Set 3 §2 |
| Synthetic Odds | `engine/synthetic_odds.py` | **✅** | Set 3 §3 |
| GNN Player Synergy | `engine/network_synergy.py` | **⚠️** | Set 3 §4 |
| Surebet Scanner | `engine/surebet_scanner.py` | **✅** | Set 3 §5 |
| Copula Bet Builder | `engine/gaussian_copula.py`, `engine/copula_math.py` | **✅** | Set 4 §1 |
| Latency Arbitrage | `engine/latency_arb.py` | **⚠️** | Set 4 §2 |
| xT / Pitch Control | `engine/expected_threat.py`, `engine/pitch_control.py` | **⚠️** | Set 4 §3 |
| Order Book / DoM | `engine/orderbook_analyzer.py` | **⚠️** | Set 4 §4 |
| Cross-Chain Arb | `engine/crosschain_arb.py` | **⚠️** | Set 4 §5 |
| Markowitz Portfolio | `engine/markowitz_optimizer.py`, `engine/markowitz_math.py` | **⚠️ BUG** | Set 4+AUDIT_BUGS |

**Totale: 9 ✅ IMPLEMENTATO · 13 ⚠️ PARZIALE · 0 ❌ NON IMPLEMENTATO**

---

## Priorità Fix per Andare in Produzione

### Fix immediati (ore)
| ID | Problema | File | Impatto |
|----|----------|------|---------|
| BUG-001 | `except Exception: pass` swallows PSD fix | `engine/markowitz_math.py:129` | Bet sizing sbagliato su ogni run |
| BUG-REFEREE | `home_cards`/`away_cards` mai incrementati | `quant/providers/api_football_client.py:928` | Modello cartellini produce sempre 0 |

### Completamento integrazione (giorni)
| Modulo | Gap | Effort |
|--------|-----|--------|
| Time-Decay → AutoML | Passare `TimeDecayWeighter` come `sample_weight` nel `clf.fit()` | 1h |
| Lineup Sniper | Aggiungere `lineup_poller.py` → chiama API-Football lineups | 1–2 giorni |
| Bayesian Live | Aggiungere orchestratore `live_pipeline.py` | 1–2 giorni |
| NLP Sentiment | Wiring nel pre-match pipeline + client Twitter/Nitter | 2–3 giorni |
| Smart Money + Order Book | Auth flow Betfair certificati + collegamento notifiche | 3–4 giorni |

### Nuove implementazioni (settimane)
| Modulo | Gap | Effort |
|--------|-----|--------|
| GNN Player Synergy | PyTorch Geometric + dati StatsBomb/Opta | 1–2 settimane |
| xT / Pitch Control in produzione | Feed dati eventi con coordinate | 1 settimana + contratto dati |
| Latency Arb reale | WebSocket feed Opta/Radar + client bookmaker | 2–3 settimane |
| Cross-Chain esecuzione | `web3.py` + wallet Polygon + client fiat | 1–2 settimane |

---

*Audit condotto: 2026-04-29 | Vista consolidata dei Set 1–4 | Totale: 20 moduli auditati su ispezione diretta codice sorgente*
