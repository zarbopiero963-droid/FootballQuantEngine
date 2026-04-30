# Audit Features da Aggiungere — Set 6 (Stack ML + Monte Carlo)
**Data:** 2026-04-29
**Metodologia:** Ispezione diretta dei file sorgente + grep codebase completo + verifica test coverage
**Verdetti:** ✅ IMPLEMENTATO | ⚠️ PARZIALE | ❌ NON IMPLEMENTATO

---

## 1. Poisson + Dixon-Coles

### Descrizione richiesta
Modello Poisson standard + correzione Dixon-Coles per i risultati a basso punteggio (0-0, 1-0, 0-1, 1-1 sono sistematicamente sottostimati dal Poisson puro).

### Verdetto: ✅ IMPLEMENTATO

**Cosa c'è:**

**Poisson base** — `quant/models/poisson_engine.py`:
- `fit(completed_matches)`: stima attack/defence strength per ogni squadra via iterazione
- `score_matrix(home, away)`: matrice NxN delle probabilità di ogni risultato
- `probabilities_1x2()`, `probabilities_ou_btts()`: derivazione da matrice Poisson
- Clamp λ a minimo 0.2 per stabilità numerica

**Dixon-Coles** — `quant/models/dixon_coles_engine.py`:
- Estende `PoissonEngine`; aggiunge correzione τ sui 4 risultati a basso punteggio:
  ```
  τ(0,0) = 1 − λh × λa × ρ
  τ(1,0) = 1 + λh × ρ
  τ(0,1) = 1 + λa × ρ
  τ(1,1) = 1 − ρ
  ```
- `_estimate_rho()`: grid search su ρ ∈ [−1, 1] massimizzando log-likelihood
- `fit()`: chiama il Poisson base poi stima ρ ottimale su dati storici

**Test:** `tests/unit/test_quant_models.py` — unit tests; `tests/unit/test_simulation_and_analytics.py`

---

## 2. Elo Rating Dinamico

### Descrizione richiesta
Rating ELO che si aggiorna dopo ogni partita in base all'esito reale vs quello atteso. Più rapido del Poisson a catturare trend recenti.

### Verdetto: ✅ IMPLEMENTATO con test seri

**Cosa c'è** (`quant/models/elo_engine.py`):
- Formula FIDE standard: `E_a = 1 / (1 + 10^((R_b − R_a)/400))`
- Home advantage aggiunto al rating home **prima** del calcolo expected score
- Update: `ΔR = K × (actual − expected)` per entrambe le squadre
- `calibrate()`: grid search su (K, home_advantage) minimizzando Brier score su split cronologico — previene look-ahead bias
- Integrato in `quant_engine.py` e nel `MetaLearner` come modello "elo"

**Test:** `tests/unit/test_quant_core.py`, `tests/unit/test_quant_market_models.py`

---

## 3. Cluster Analysis

### Descrizione richiesta
Segmentazione delle squadre in cluster (es. squadre simili per stile di gioco, rendimento, forma). Permette di riconoscere pattern tra squadre simili.

### Verdetto: ❌ NON IMPLEMENTATO

**Ricerca exhaustiva nel codebase:**
```bash
grep -r "KMeans\|DBSCAN\|cluster\|Cluster\|silhouette\|centroid\|segment" --include="*.py"
```
**Risultato:** Una sola occorrenza in un commento testuale — `"Draws are persistently underpriced in certain league clusters"` in `analytics/market_inefficiency_scanner.py`. Il termine è usato come metafora, non come implementazione algoritmica.

**Nessun file:**
- Nessun `clustering.py` o `cluster_analysis.py`
- Nessun import di `sklearn.cluster.KMeans`, `DBSCAN`, o equivalenti
- Nessun training o inferenza cluster

**Per implementare:**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Feature: xG, xGA, pressioni, possesso, forma recente, elo
X = feature_matrix(teams)
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42).fit(X_scaled)
# → cluster label per ogni squadra → beta coefficienti per matchup inter-cluster
```

---

## 4. Random Forest

### Descrizione richiesta
Ensemble di alberi decisionali per predire la probabilità di vittoria da feature tabellari (xG, Elo, forma, fatica, meteo).

### Verdetto: ✅ IMPLEMENTATO

**Cosa c'è** (`training/automl_learners.py`, `training/local_automl.py`):
```python
RandomForestClassifier(
    n_estimators = min(300, max(50, n_rows // 5)),
    max_depth    = 8,
    min_samples_leaf = 5,
    random_state = 42,
    n_jobs       = -1,
)
```
- Incluso nella lista candidati di `_sklearn_candidates()`
- `local_automl.py`: include `"sklearn_random_forest"` nel model registry
- Optuna hyperparameter search su `n_estimators ∈ [50, 400]`, `max_depth ∈ [3, 10]`, `min_samples_leaf ∈ [1, 20]`
- `feature_importances_` esposto tramite `get_feature_importances()`

**Test:** `tests/test_local_automl.py` — `test_local_automl_runs()`

**Gap:** Time-decay (`TimeDecayWeighter`) non passato come `sample_weight` nel `fit()` (vedi Set 1 §4 — gap comune a tutti i modelli sklearn).

---

## 5. XGBoost

### Descrizione richiesta
Gradient boosting ad alberi, stato dell'arte per dati tabellari calcistici con non-linearità complesse.

### Verdetto: ✅ IMPLEMENTATO (condizionale su package)

**Cosa c'è** (`training/automl_learners.py`):
```python
if _has("xgboost"):
    from xgboost import XGBClassifier
    candidates.append((
        "xgboost",
        XGBClassifier(
            n_estimators    = min(200, n_est),
            max_depth       = 4,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            eval_metric     = "logloss",
            random_state    = 42,
        )
    ))
```
- LightGBM e CatBoost inclusi con la stessa logica condizionale
- Tutti e 3 partecipano all'Optuna hyperparameter search
- Usati nel `MetaLearner` come modello `"gradient_boost"`

**Gap:**
- `xgboost`, `lightgbm`, `catboost` sono **optional dependencies** — se non installati, il runner fallback su sklearn GBM
- Time-decay non wired come `sample_weight` (stesso gap del Random Forest)
- Zero test che verificano il comportamento con XGBoost installato vs assente

---

## 6. Neural Networks

### Descrizione richiesta
Reti neurali (MLP o deep learning) per catturare pattern non lineari nei dati calcistici.

### Verdetto: ❌ NON IMPLEMENTATO

**Ricerca exhaustiva:**
```bash
grep -r "MLPClassifier\|NeuralNetwork\|keras\|tensorflow\|torch\.nn\|nn\.Module\|LSTM\|deep.learn" --include="*.py"
```
**Risultato:** zero occorrenze in codice non-test (escluso `torch_geometric` referenziato come TODO nell'AUDIT_REPORT.md per il GNN).

`local_automl.py` docstring elenca:
```
xgboost:   XGBClassifier
lightgbm:  LGBMClassifier
catboost:  CatBoostClassifier
sklearn:   RandomForest, GradientBoosting, LogisticRegression
```
**Nessuna rete neurale.**

**Per implementare (opzione minimale):**
```python
from sklearn.neural_network import MLPClassifier
candidates.append((
    "mlp",
    MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
))
```
Alternativa più potente: `torch.nn.Sequential` con layer FC + BatchNorm + Dropout, addestrato su sequenze temporali di partite.

---

## 7. Ensemble Stacking

### Descrizione richiesta
Un meta-learner di secondo livello che prende le predizioni dei modelli base (Poisson, Elo, XGBoost, ecc.) come feature e le combina con un classificatore addestrato.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è** (`engine/meta_learner.py`):
- `MetaLearner`: weighted blend con **14 regole contestuali** dinamiche:
  - Pioggia forte → ↑xG_model ×1.40, ↓Poisson ×0.70
  - Trasferta lontana → ↑gradient_boost ×1.35
  - Derby → ↑Elo ×1.30, ↑gradient_boost ×1.40
  - Freddo <5°C → ↑Skellam ×1.20
  - ecc. (14 regole totali)
- Brier-score adaptive weights: dopo ogni partita, il modello con Brier più basso in quella lega aumenta il suo peso
- `EnsemblePrediction`: output con pesi finali per ogni modello

**Non è stacking vero:**
```
True Stacking:
  Level 0: Poisson(X), Elo(X), XGB(X) → predictions P0
  Level 1: MetaClassifier.fit(P0_holdout, y) → learned blend weights

Actual implementation:
  Level 0: Poisson(X), Elo(X), XGB(X) → predictions P0
  Level 1: weighted_avg(P0, weights=rule_based_context) → no training
```
Nessuno `StackingClassifier` di sklearn, nessun level-2 addestrato su held-out predictions. I pesi Brier si aggiornano online, non su un validation set separato.

**Per completare il vero stacking:**
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking = StackingClassifier(
    estimators=[
        ("poisson_features", poisson_feature_transformer),
        ("elo_features",     elo_feature_transformer),
        ("xgb",              XGBClassifier(**xgb_params)),
    ],
    final_estimator=LogisticRegression(),
    cv=TimeSeriesSplit(n_splits=5),   # no data leakage
)
```

---

## 8. Monte Carlo Simulations (10.000+)

### Descrizione richiesta
Simulazioni Monte Carlo per stimare le probabilità di match outcome con alta precisione statistica.

### Verdetto: ✅ IMPLEMENTATO con test seri

**Cosa c'è** (`simulation/monte_carlo_advanced.py`):
- **100.000 simulazioni** default (`simulations: int = 100_000`)
- **Antithetic variates**: usa metà campioni + i loro complementi → riduce varianza stimata del 30–50% vs MC naive
- **Bivariate Poisson sampling**: campiona gol home e away con correlazione positiva (squadra dominante tende a segnare su azioni coordinate)
- **Numpy path**: vettorizzato con `np.random.Generator` per performance
- **Pure Python fallback**: se numpy non disponibile, loop Python equivalente
- `MCResult`: `home_win_prob`, `draw_prob`, `away_win_prob`, `exact_scores`, `ou_probs`, `btts_prob`, `variance_reduction` ratio
- `engine/markov_gamestate.py`: Monte Carlo da stato di gioco live (50.000 path) per calcolare WDL in-play

**Integrato in `engine/prediction_pipeline.py`:**
```python
"monte_carlo": 0.20   # peso 20% nell'ensemble finale
```
O/U e BTTS vengono dalle simulazioni MC, non dalla formula Poisson analitica.

**Test** (`tests/test_simulation.py`) — seri:
- `test_simulate_sums_to_one` — home + draw + away = 1.0
- `test_simulate_av_sums_to_one` — antithetic variates conserva la somma
- `test_simulate_exact_sums_to_one` — exact score matrix somma a 1.0
- `test_high_home_lambda_home_wins_more` — stochastic ordering corretto
- `test_equal_lambdas_draw_plausible` — draw prob ragionevole con λ uguali
- `test_all_probs_non_negative`
- `test_zero_away_lambda`, `test_very_small_lambdas` — edge case handling

---

## Riepilogo

| Modello | File principale | Verdetto | Note |
|---------|----------------|----------|------|
| Poisson | `quant/models/poisson_engine.py` | **✅** | Base engine solido |
| Dixon-Coles | `quant/models/dixon_coles_engine.py` | **✅** | τ correction + stima ρ MLE |
| Elo Dinamico | `quant/models/elo_engine.py` | **✅** | K-calibration Brier, home advantage |
| Cluster Analysis | — | **❌ NON IMPLEMENTATO** | Solo menzionato in un commento |
| Random Forest | `training/automl_learners.py` | **✅** | Optuna hyperopt; time-decay non wired |
| XGBoost | `training/automl_learners.py` | **✅** | Conditional import; time-decay non wired |
| Neural Networks | — | **❌ NON IMPLEMENTATO** | Zero MLPClassifier/keras/torch.nn |
| Ensemble Stacking | `engine/meta_learner.py` | **⚠️ PARZIALE** | Weighted blend con regole; **non è true stacking** (no level-2 trained) |
| Monte Carlo 100k | `simulation/monte_carlo_advanced.py` | **✅** | Antithetic variates, test seri |

**Implementati: 6/9 · Parziali: 1/9 · Non implementati: 2/9**

### Gap da colmare

| Gap | Effort | Impatto |
|-----|--------|---------|
| Cluster Analysis (KMeans su squadre) | 1 giorno | Feature aggiuntiva per MetaLearner |
| Neural Network (sklearn MLP come baseline) | 2h | Aggiungere a `automl_learners.py` |
| True Stacking (sklearn StackingClassifier + TimeSeriesSplit) | 3–5 giorni | Blend weights appresi invece di rule-based |
| Time-decay `sample_weight` in tutti i modelli sklearn | 1h | Fix trasversale (vedi Set 1 §4) |

---

*Audit condotto: 2026-04-29 | Metodologia: grep exhaustivo codebase, lettura file sorgente, verifica test*
