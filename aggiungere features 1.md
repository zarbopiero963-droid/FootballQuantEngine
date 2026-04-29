# Audit Features da Aggiungere — Set 1
**Data:** 2026-04-29
**Metodologia:** Ispezione diretta dei file sorgente + verifica test coverage
**Verdetti:** ✅ IMPLEMENTATO | ⚠️ PARZIALE | ❌ NON IMPLEMENTATO

---

## 1. Gestione del Capitale (Kelly / Fractional Kelly)

### Descrizione richiesta
Algoritmo che calcola l'importo esatto da scommettere in base al bankroll e alla dimensione dell'edge. Formula matematica usata dai fondi istituzionali per massimizzare la crescita del capitale a lungo termine evitando la bancarotta.

### Verdetto: ✅ IMPLEMENTATO con test seri

**Cosa c'è:**
- `ranking/match_ranker.py` — `kelly_fraction()`: formula completa `f* = (p×b − q)/b` con `KELLY_SCALE = 0.25` (quarter-Kelly)
- `quant/value/kelly_engine.py` — `kelly_fraction()`, `fractional_kelly()`, `suggested_stake()`: tre metodi distinti
- Kelly stake calcolato e restituito su ogni oggetto `BetRecommendation`

**Test:**
- `tests/property/test_kelly_ev_properties.py`: 8 Hypothesis property tests (bounded, monotone, zero when negative EV, ecc.)
- `tests/test_kelly_engine.py`: unit tests dedicati
- `tests/invariant/test_quant_financial_invariants.py`: invarianti finanziarie
- Test veri, non smoke test.

**Gap:** Nessuno.

---

## 2. Mercati Asiatici (Asian Handicap / Skellam Distribution)

### Descrizione richiesta
Convertitore matematico (Skellam distribution) che trasforma le probabilità Poissoniane del 1X2 in probabilità per le linee asiatiche (es. Vittoria Casa -0.75, Over 2.25). Margini più bassi rispetto al 1X2.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `models/asian_handicap.py`: conversione Poisson matrix → linee AH corretta, quarter-line splitting (±0.25) matematicamente preciso, O/U asiatico implementato
- La matematica è corretta e la probabilità delle tre outcome (win/push/lose) somma a 1.0

**Gap:**
1. **Non usa Skellam** — usa la matrice Poisson joint (tecnicamente più accurata per i mercati AH, ma non è quello descritto). La distribuzione di Skellam esiste nell'ensemble (`engine/meta_learner.py`) come modello separato per il 1X2, non è integrata nel convertitore AH.
2. **Test insufficienti** — solo 1 file generico (`tests/unit/test_simulation_and_analytics.py`), zero property tests dedicati alle invarianti AH (somma probabilità = 1, push handling, quarter-line splitting, ecc.)

**Per completare:**
- Aggiungere property tests: `p_win + p_push + p_lose == 1.0` per qualsiasi linea e λ
- Valutare integrazione Skellam direttamente nel convertitore AH per le linee di gol differenza

---

## 3. Tracker Sharp vs Soft / Steam Chasing / CLV

### Descrizione richiesta
Motore che monitora le quote di Pinnacle in tempo reale. Se la quota crolla (Steam move), manda un alert immediato per scommettere sui bookmaker Soft italiani prima che aggiornino. Genera Closing Line Value (CLV) positivo sistematicamente.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/sharp_soft_tracker.py` (348 righe): algoritmo steam detection completo — soglia 4pp in implied probability, finestra 5 minuti, `SteamAlert`, `SoftOpportunity`, metodo `_check_steam()`
- `live/odds_stream.py`: polling multi-sorgente con supporto Pinnacle
- `ui/clv_tracker_window.py`: CLV tracking UI con DB persistence (`clv_bets` table), calcolo `CLV% = (our_odds / closing_odds − 1) × 100`

**Gap critico:**
- Il feed Pinnacle passa da **the-odds-api.com** (aggregatore terze parti), **non dall'API diretta Pinnacle**. Latenza stimata: 30–60 secondi. Per steam chasing reale servono feed con latenza <5s: con 30–60s di ritardo i soft book si sono già aggiornati prima dell'alert.
- CLV deve essere inserito **manualmente** post-partita: nessun feed closing odds automatico.

**Per completare:**
- Sostituire il client the-odds-api con un feed diretto Pinnacle o con BetFair Exchange API (latenza <2s)
- Automatizzare il caricamento closing odds da fonte esterna (es. OddsPortal scraping o feed dedicato)

---

## 4. Upgrade AI: Gradient Boosting (XGBoost/LightGBM) + Time-Decay

### Descrizione richiesta
Integrare LightGBM o XGBoost per calcolare la probabilità di vittoria imparando da migliaia di partite del passato. Time-decay weighting: una partita di 2 settimane fa vale 100%, una dell'anno scorso vale ~10%.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `training/local_automl.py`: import condizionale di `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`; grid search iperparametri AutoML
- `training/time_decay.py`: `TimeDecayWeighter` con formula `w = exp(−λ × Δdays)` corretta, half-life configurabile (default 90 giorni)

**Gap critico — i due moduli non sono collegati:**

```python
# training/local_automl.py — chiama fit senza sample_weight
clf.fit(X, y)   # ← TimeDecayWeighter mai passato qui

# training/time_decay.py — mai importato da local_automl.py
```

`TimeDecayWeighter` esiste e funziona ma **non viene mai passato come `sample_weight`** nel `.fit()` dell'AutoML. Sono due file indipendenti, non un sistema integrato.

**Ulteriori gap:**
- Zero test per `training/time_decay.py`
- Nessuna verifica che i pesi siano effettivamente applicati durante il training

**Per completare:**
```python
# In local_automl.py — aggiungere:
from training.time_decay import TimeDecayWeighter

weighter = TimeDecayWeighter(half_life_days=90)
weights = weighter.compute(match_dates)
clf.fit(X, y, sample_weight=weights)
```
- Aggiungere property test: pesi decrescenti con l'età, normalizzazione, half-life corretto

---

## 5. Feature Engineering: Fatica, Missing Players, Motivazione

### Descrizione richiesta
Fattore Fatica (riposo, Champions League 3 giorni fa), Missing Players Impact (quanto pesano gli assenti sul totale gol), Motivazione (fine stagione, squadra già salva vs lotta retrocessione).

### Verdetto: ✅ IMPLEMENTATO

**Cosa c'è:**
- `engine/travel_fatigue.py` — `TravelFatigueEngine`: 5 componenti (distanza viaggio, giorni di riposo, jet lag, altitudine, congestione fixture); output `FatigueReport` con moltiplicatori λ
- `engine/lineup_sniper.py` — `LineupSniperEngine`: impatto giocatori assenti con quantificazione xG/Elo delta; scala d'impatto NONE / MINOR / MAJOR / CRITICAL
- `quant/models/standings_engine.py` — `_motivation()` + `get_motivation_diff()`: fattore motivazione basato su posizione in classifica (lotta retrocessione/promozione vs squadra già salva)
- `quant/features/feature_builder.py`: **tutti e 3 i fattori usati come feature** nel pipeline — sono integrati, non standalone

**Test:** `tests/unit/test_simulation_and_analytics.py` copre il pipeline di feature.

**Gap:** Nessuno critico. La motivazione è basata solo sulla posizione in classifica, non su fattori contestuali (ultimo turno di stagione, Europa già acquisita, ecc.) — ma è funzionale.

---

## 6. In-Play / Live Betting Engine

### Descrizione richiesta
Collegare l'engine agli endpoint LIVE di API-Football. Il modello calcola il valore in base al tempo rimanente, chi sta attaccando, cartellini rossi (che distruggono l'xG), scarto reti attuale. Edge > 15% frequente nei campionati minori perché i bookmaker sbagliano nel ricalcolo rapido.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `data/live_updater.py`: polling API-Football ogni 60s, thread in background, salva fixture in DB, notifica callbacks — **funziona**
- `engine/bayesian_live.py` + `engine/bayesian_runner.py`: aggiornamento bayesiano λ_home / λ_away su eventi gol/tiri/cartellini; `BayesianAlert` su shift λ sopra soglia — **funziona**

**Gap critico — i due moduli non sono orchestrati insieme:**

```
live_updater  →  salva fixture in DB
                       ↓
            [nessuno legge e chiama BayesianLiveEngine]
                       ↓
bayesian_live  →  vive isolato, non riceve eventi live reali
```

Manca un orchestratore che faccia: *nuovo evento in DB → feed a BayesianLiveEngine → ricalcola edge → emetti BetAlert → notifica UI*.

**Per completare:**
- Creare `engine/live_pipeline.py`: consumer dei callback di `LiveUpdater` che estrae eventi (gol, cartellini, tiri) e li passa a `BayesianLiveEngine.update()`
- Collegare gli alert bayesiani al sistema di notifica (Telegram / UI)
- Aggiungere integration test end-to-end: fixture live simulata → aggiornamento λ → alert generato

---

## Riepilogo

| # | Feature | Verdetto | Gap principale |
|---|---------|----------|----------------|
| 1 | Kelly / Fractional Kelly | **✅ IMPLEMENTATO** | Nessuno |
| 2 | Asian Handicap | **⚠️ PARZIALE** | No test AH; Skellam non integrato nell'AH |
| 3 | Sharp vs Soft / CLV | **⚠️ PARZIALE** | Feed Pinnacle via aggregatore (latenza 30–60s), CLV manuale |
| 4 | XGBoost + Time-Decay | **⚠️ PARZIALE** | `TimeDecayWeighter` mai passato come `sample_weight` nel fit |
| 5 | Feature Engineering | **✅ IMPLEMENTATO** | Nessuno critico |
| 6 | Live In-Play | **⚠️ PARZIALE** | `live_updater` e `bayesian_live` non orchestrati in pipeline |

**Implementati completamente: 2/6**
**Parziali (logica core presente, gap di integrazione o test): 4/6**
**Non implementati: 0/6**

---

*Audit condotto: 2026-04-29 | Metodologia: ispezione diretta file sorgente, grep delle dipendenze reali, verifica test coverage*
