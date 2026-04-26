# Football Quant Engine

Motore quantitativo professionale per l'analisi delle scommesse calcistiche. Combina 9 modelli statistici, machine learning e dati in tempo reale per identificare value bet, calcolare stakes ottimali e tracciare le performance nel tempo.

---

## Indice

1. [Requisiti & Installazione](#1-requisiti--installazione)
2. [Configurazione iniziale](#2-configurazione-iniziale)
3. [Panoramica interfaccia](#3-panoramica-interfaccia)
4. [Come trovare i pronostici](#4-come-trovare-i-pronostici)
5. [Finestre e pulsanti](#5-finestre-e-pulsanti)
6. [Modelli quantitativi](#6-modelli-quantitativi)
7. [Value Betting & Kelly](#7-value-betting--kelly)
8. [Backtest & Ottimizzazione soglie](#8-backtest--ottimizzazione-soglie)
9. [CLV Tracker](#9-clv-tracker)
10. [Notifiche Telegram](#10-notifiche-telegram)
11. [Export dei dati](#11-export-dei-dati)
12. [API utilizzate](#12-api-utilizzate)
13. [Flusso di lavoro completo](#13-flusso-di-lavoro-completo)
14. [Struttura file di output](#14-struttura-file-di-output)

---

## 1. Requisiti & Installazione

**Requisiti:**
- Python 3.10+
- pip

```bash
pip install -r requirements.txt
```

**Dipendenze principali:**

| Pacchetto | Utilizzo |
|-----------|----------|
| `PySide6` | Interfaccia grafica Qt6 |
| `requests` | Chiamate HTTP alle API |
| `pandas` | Elaborazione dati tabellari |
| `numpy` | Calcolo numerico |
| `scipy` | Distribuzioni statistiche (Poisson) |
| `scikit-learn` | Machine learning e AutoML |
| `openpyxl` | Export Excel |

**Dipendenze opzionali (migliorano AutoML):**
```bash
pip install xgboost lightgbm catboost optuna
```

**Avvio:**
```bash
python -m app.main
```

---

## 2. Configurazione iniziale

Al primo avvio clicca **⚙ Settings** nella barra degli strumenti.

| Campo | Obbligatorio | Descrizione |
|-------|-------------|-------------|
| **API-Football Key** | ✅ | Chiave da [api-sports.io](https://api-sports.io) |
| **Odds API Key** | ❌ | Chiave per quote bookmaker alternativi |
| **Telegram Token** | ❌ | Token bot da @BotFather |
| **Telegram Chat ID** | ❌ | ID della chat/gruppo per gli alert |

**ID leghe principali:**

| Lega | ID |
|------|----|
| Serie A | `135` |
| Premier League | `39` |
| La Liga | `140` |
| Bundesliga | `78` |
| Ligue 1 | `61` |
| Champions League | `2` |

In alternativa usa variabili d'ambiente (hanno precedenza su `settings.json`):
```bash
export API_FOOTBALL_KEY="la_tua_chiave"
export TELEGRAM_TOKEN="token_bot"
export TELEGRAM_CHAT_ID="id_chat"
```

---

## 3. Panoramica interfaccia

La finestra principale è composta da:

- **Barra degli strumenti** in alto con tutti i pulsanti
- **Card statistiche** con riepilogo del ciclo corrente
- **Tabella pronostici** con filtri interattivi
- **Tab inferiori**: Risultati · Log · Storico · Metriche · Grafici

---

## 4. Come trovare i pronostici

### Passo 1 — Prima esecuzione: addestra i modelli

Al primo avvio clicca **▶ Run Cycle**. Il sistema:

1. Recupera le partite completate per la stagione corrente tramite API-Football
2. Addestra i modelli di probabilità sui risultati disponibili
3. Salva i dati nel database SQLite locale
4. Nelle esecuzioni successive aggiorna i dati incrementalmente

> **Nota:** il download automatico di tutte le stagioni storiche e il live tracking sono in arrivo nelle prossime release.

### Passo 2 — Ciclo normale

Clicca **▶ Run Cycle**. Il motore:
1. Aggiorna la stagione corrente con i risultati recenti
2. Addestra i modelli sui dati storici
3. Genera probabilità per ogni partita (1X2, OU 2.5, BTTS)
4. Calcola EV, Kelly ed Edge per ogni selezione
5. Assegna la decisione BET / WATCHLIST / NO_BET
6. Popola la tabella nel tab **Risultati**

### Passo 3 — Leggi la tabella

| Colonna | Significato |
|---------|-------------|
| `home` / `away` | Squadre |
| `market` | Mercato (1x2, ou25, btts) |
| `selection` | Selezione (home, draw, away, over, under) |
| `probability` | Probabilità stimata dal modello |
| `edge` | Vantaggio del modello sul bookmaker |
| `ev` | Expected Value — rendimento atteso per unità |
| `kelly` | Frazione Kelly consigliata (% del bankroll) |
| `stake` | Importo suggerito in € |
| `odds_market` | Quota bookmaker disponibile |
| `confidence` | Confidenza composita del modello (0–1) |
| `tier` | Qualità: S · A · B · C |
| `decision` | **BET** · WATCHLIST · NO_BET |

### Passo 4 — Filtra

- **Cerca** — testo libero su tutte le colonne
- **Decision Filter** → seleziona `BET` per vedere solo le scommesse consigliate
- **Market Filter** → filtra per mercato specifico (1x2, ou25, btts)

### Mercati disponibili

| Codice | Mercato |
|--------|---------|
| `1x2` | 1X2 — Home / Draw / Away |
| `ou25` | Over/Under 2.5 gol |
| `btts` | GG/NG — Both Teams to Score |

> **In arrivo:** Over/Under 1.5 e 3.5, primo/secondo tempo, calci d'angolo, risultati esatti.

### Tier di qualità

| Tier | Kelly | EV | Confidenza |
|------|-------|----|-----------|
| 🟢 **S** Elite | ≥ 6% | ≥ 10% | ≥ 70% |
| 🔵 **A** Forte | ≥ 3% | ≥ 5% | ≥ 55% |
| 🟡 **B** Moderato | ≥ 1% | ≥ 2% | — |
| ⚪ **C** Marginale | > 0% | > 0% | — |
| 🔴 **X** No value | ≤ 0% | — | — |

### Card statistiche in tempo reale

| Card | Contenuto |
|------|-----------|
| BET / WATCHLIST / NO_BET | Conteggio per categoria |
| HOME / DRAW / AWAY | Distribuzione per mercato 1X2 |
| Avg Confidence | Confidenza media delle previsioni |
| Max Edge | Miglior vantaggio trovato nel ciclo |
| ROI / Yield / Hit Rate | Performance storica |
| Bankroll | Capitale attuale tracciato |
| Max Drawdown | Peggiore riduzione dell'equity |

---

## 5. Finestre e pulsanti

### Barra degli strumenti

| Pulsante | Funzione |
|----------|----------|
| **▶ Run Cycle** | Aggiorna stagione corrente + genera previsioni su tutti i mercati |
| **📊 Backtest** | Apre la finestra di analisi storica |
| **⚙ Settings** | Configura API-Football Key, Odds API Key, Telegram |
| **📂 Import CSV** | Importa dati partite/quote da file CSV esterni |
| **📝 Manual Context** | Modifica manuale di infortuni, formazioni, meteo |
| **📁 Outputs** | Visualizza e genera i report HTML |
| **✅ Final Check** | Verifica integrità del progetto (DB, file, API) |
| **📋 Project Summary** | Metadati del codebase |
| **📈 Analytics** | Dashboard analitica avanzata a 4 tab |
| **🌐 HTML Report** | Apre l'ultimo report HTML nel browser |
| **ℹ About** | Informazioni sull'applicazione |

### Tab nella dashboard

| Tab | Contenuto |
|-----|-----------|
| **Risultati** | Tabella interattiva dei pronostici su tutti i mercati |
| **Log** | Messaggi in tempo reale del pipeline |
| **Storico** | Storico cumulativo dei cicli |
| **Metriche** | KPI del backtest (ROI, Yield, Hit Rate, Drawdown, Brier, Log Loss) |
| **Grafici** | Curve SVG: Bankroll · Accuracy · Drawdown |

### Pulsanti nella tabella risultati

| Pulsante | Funzione |
|----------|----------|
| **🔄 Refresh** | Ricarica i dati dall'ultimo ciclo |
| **↕ Toggle Sort** | Ordina per tier/EV decrescente |
| **📄 Export CSV** | Salva le scommesse in CSV |
| **📊 Export Excel** | Salva in Excel con colori e fogli multipli |
| **📂 Open CSV** | Apre l'ultimo CSV esportato |
| **📂 Open Excel** | Apre l'ultimo Excel esportato |

### Finestra Import CSV

Importa dati da fonti esterne in 3 passi:
1. **Seleziona file** — CSV partite (obbligatorio) + CSV quote (opzionale)
2. **Mappa le colonne** — indica la corrispondenza `colonna_csv:colonna_db`
3. **Preview & Import** — anteprima delle prime 10 righe, poi importa nel DB

Colonne obbligatorie CSV partite: `home_team, away_team, league, date, home_goals, away_goals`

### Finestra Manual Context

Modifica `manual_context.json` per iniettare dati manuali:

```json
{
  "lineups":   { "fixture_id": ["player_id", ...] },
  "injuries":  { "fixture_id": ["nome_giocatore", ...] },
  "weather":   { "fixture_id": {"temp": 15, "humidity": 65} }
}
```

### Analytics Dashboard (4 tab)

| Tab | Contenuto |
|-----|-----------|
| **CLV Analysis** | Distribuzione CLV, win rate su CLV positivo, tabella per scommessa |
| **Market Inefficiency** | Classificazione dei mispricing per tipo e lega |
| **League Predictability** | RPS, Brier Score, CBI per lega con curva di calibrazione |
| **Feature Importance** | Top 20 feature per potere predittivo (Mutual Information) |

---

## 6. Modelli quantitativi

Il motore combina **9 modelli statistici** in un ensemble pesato, più modelli dedicati per i mercati speciali.

### Modelli core (1X2)

| Modello | Descrizione |
|---------|-------------|
| **Dixon-Coles** | Modello probabilistico temporale con correzione sulla dipendenza tra gol |
| **Elo Rating** | Forza relativa delle squadre aggiornata partita dopo partita |
| **Poisson Engine** | Stima λ_home e λ_away, genera la matrice score completa |
| **Form Engine** | Trend di performance recente sulle ultime N partite |
| **Head-to-Head** | Storico degli scontri diretti |
| **Goal Momentum** | Pattern di scoring e varianza nelle ultime partite |
| **Rest Engine** | Impatto dei giorni di recupero dall'ultima partita |
| **Standings Engine** | Vantaggio dalla posizione attuale in classifica |
| **Injury Engine** | Penalità per assenze di giocatori chiave |
| **Referee Engine** | Bias storico del direttore di gara |
| **Weather Engine** | Condizioni meteo *(richiede OpenWeather API)* |

### Modelli mercati speciali

| Modello | Mercato | Approccio |
|---------|---------|-----------|
| **CornersModel** | Calci d'angolo | Poisson indipendente per angoli home/away con attack/defense strength calibrati sullo storico |
| **HalftimeModel** | Primo e secondo tempo | Ratio HT/FT calibrato sui dati storici (default ~42%), convolution Poisson |
| **CorrectScoreModel** | Risultati esatti | Top-12 score dalla matrice Poisson normalizzata |

### AutoML

Il modulo `LocalAutoML` testa automaticamente i modelli disponibili:

| Libreria | Modelli testati |
|----------|----------------|
| scikit-learn | RandomForest, GradientBoosting, LogisticRegression |
| xgboost | XGBClassifier |
| lightgbm | LGBMClassifier |
| catboost | CatBoostClassifier |
| optuna | Hyperparameter tuning TPE |
| Python puro | GaussianNB + SGD (fallback senza sklearn) |

Selezione tramite **5-fold StratifiedKFold** minimizzando il log-loss.

---

## 7. Value Betting & Kelly

### Expected Value

```
EV = (probabilità_modello × quota) − 1
```

EV positivo = il bookmaker prezza male la probabilità dell'evento.

### Kelly Criterion

```
Kelly = (p × b − q) / b
        dove  b = quota − 1
              q = 1 − p
```

Il sistema usa il **Kelly frazionario al 25%** per ridurre la varianza:

```
Stake = Kelly × 0.25 × Bankroll
Stake_max = 5% del Bankroll (hard cap)
```

### Decisioni automatiche

| Decisione | Condizione |
|-----------|-----------|
| **BET** | EV ≥ 2%, confidence ≥ 60%, agreement ≥ 55%, odds 1.40–8.00 |
| **WATCHLIST** | EV ≥ 1%, criteri leggermente rilassati |
| **NO_BET** | Non soddisfa i requisiti minimi |

### Ranking composito

```
Score   = EV × confidence × (1 + agreement) × (1 + market_edge × 2)
Sharpe  = EV / √(p × (1 − p))
```

---

## 8. Backtest & Ottimizzazione soglie

Clicca **📊 Backtest** nella barra degli strumenti.

### Filtri

| Campo | Descrizione |
|-------|-------------|
| Data Da | Inizio periodo (YYYY-MM-DD, opzionale) |
| Data A | Fine periodo (YYYY-MM-DD, opzionale) |
| Lega | Nome o ID numerico (opzionale) |

### Metriche restituite

| Metrica | Descrizione |
|---------|-------------|
| **Total Bets** | Numero di scommesse nel periodo |
| **Total Staked** | Totale puntato |
| **Total Profit** | Profitto netto |
| **ROI** | Return on Investment % |
| **Yield** | Profitto / Stakes % |
| **Hit Rate** | % di scommesse vincenti |
| **Max Drawdown** | Massima riduzione dell'equity |
| **Brier Score** | Calibrazione probabilità (↓ meglio) |
| **Log Loss** | Accuratezza probabilistica (↓ meglio) |

### Ottimizzazione automatica soglie

Il pulsante **Optimize Thresholds** esegue una grid-search:

| Parametro | Range |
|-----------|-------|
| `min_edge` | 0.02 – 0.10 |
| `min_confidence` | 0.45 – 0.70 |
| `min_kelly` | 0.01 – 0.05 |

**Obiettivo:** massimizzare `ROI × (1 + hit_rate) × (1 − |max_drawdown|)`
Restituisce le **top 3 configurazioni ottimali**.

---

## 9. CLV Tracker

Il **Closing Line Value (CLV)** misura se stai scommettendo a quote migliori di quelle a cui il mercato chiude — indipendentemente dai risultati.

```
CLV % = (quota_nostra / quota_chiusura − 1) × 100
```

### Operazioni

| Pulsante | Funzione |
|----------|----------|
| **+ Add Bet** | Registra una scommessa: fixture, mercato, quota presa, note |
| **✏ Update Closing** | Inserisci quota di chiusura e risultato (WIN / LOSS) |
| **Export CSV** | Esporta lo storico completo |

### Statistiche

| Metrica | Descrizione |
|---------|-------------|
| **Avg CLV %** | CLV medio — la metrica principale |
| **CLV Hit Rate** | % di scommesse con CLV positivo |
| **ROI** | Profitto medio per scommessa chiusa |
| **Total P/L** | Profitto/perdita cumulativo |

---

## 10. Notifiche Telegram

Configura **Telegram Token** e **Chat ID** nelle Settings per ricevere alert automatici.

### Tipi di notifica

| Tipo | Contenuto |
|------|-----------|
| **Value Bets Alert** | Lista BET con tier emoji, fixture, mercato, quota, EV%, Kelly%, confidence% |
| **Steam Alert** | Movimento di quota rilevante |
| **Cycle Summary** | Report del ciclo: n. previsioni, n. bet, n. watchlist, tempo elaborazione |
| **Daily Report** | Riepilogo giornaliero con metriche aggregate |
| **CSV Attachment** | File CSV dei value bet allegato |

### Tier emoji

| Emoji | Tier |
|-------|------|
| 🟢 | S — Elite |
| 🔵 | A — Forte |
| 🟡 | B — Moderato |
| ⚪ | C — Marginale |
| 🔴 | X — No value |

### Limiti gestiti automaticamente

- Rate limiting: 28 msg/min
- Retry: 3 tentativi con backoff esponenziale (2s → 4s → 8s)
- Chunking messaggi > 4096 caratteri
- Ore di silenzio configurabili

---

## 11. Export dei dati

### CSV

| Mode | Contenuto |
|------|-----------|
| `bets` | Solo BET + WATCHLIST |
| `full` | Tutte le righe |
| `summary` | Aggregazione per lega |

### Excel (.xlsx) — 3 fogli

| Foglio | Contenuto |
|--------|-----------|
| **Bets** | Tabella colorata per tier, filtro automatico |
| **Summary** | Aggregazione per lega |
| **Metrics** | KPI backtest + equity curve |

### Report HTML

| Report | Percorso |
|--------|---------|
| Dashboard | `outputs/dashboard.html` |
| Advanced | `outputs/advanced_report.html` |
| Charts | `outputs/charts_report.html` |
| Temporal | `outputs/temporal/report_DATA.html` |

---

## 12. API utilizzate

### API-Football — Principale

**Endpoint:** `https://v3.football.api-sports.io`
**Auth:** header `x-apisports-key`

| Dato scaricato | Utilizzo |
|----------------|----------|
| Fixture completati (risultati) | Addestramento modelli Dixon-Coles |
| Fixture in programma | Previsioni pre-partita |
| Quote bookmaker (Match Winner) | Edge, EV, Kelly |
| Classifiche | Standings Engine |
| Head-to-head | H2H Engine |
| Infortuni | Injury Engine |
| Arbitri | Referee Engine |

Cache automatica: fixture 24h, quote 1h.

---

### OpenWeatherMap — Opzionale

**Endpoint:** `https://api.openweathermap.org/data/2.5`
Temperatura, umidità, vento → Weather Engine.

---

### Telegram Bot API — Opzionale

**Endpoint:** `https://api.telegram.org/bot{token}/{method}`
Metodi: `sendMessage`, `sendDocument`, `sendPhoto`

---

### ClubElo

**Endpoint:** `https://api.clubelo.com`
Rating Elo storici per confronto e validazione interna.

---

## 13. Flusso di lavoro completo

```
CICLO NORMALE (Run Cycle)
  ├─ get_completed_matches() → scarica risultati stagione corrente
  ├─ PoissonEngine.fit() → calcola forza attacco/difesa per ogni squadra
  ├─ DixonColesModel.fit() → addestra sui dati storici
  ├─ get_fixtures() → fixture in programma da prevedere
  ├─ get_odds() → quote bookmaker (Match Winner)
  ├─ predict() → probabilità 1X2, OU 2.5, BTTS
  ├─ Calcolo edge/EV/Kelly per ogni selezione
  └─ Mostra in dashboard con BET/WATCHLIST/NO_BET

ANALISI
  ├─ Backtest → KPI storici + threshold optimizer
  ├─ CLV Tracker → tracking apertura vs chiusura quote
  └─ Analytics Dashboard → CLV, inefficienze, predictability
```

---

## 14. Struttura file di output

```
outputs/
├── dashboard.html
├── advanced_report.html
├── charts_report.html
├── dashboard_export_YYYYMMDD.csv
├── dashboard_export_YYYYMMDD.xlsx
├── automl_best_model.pkl
├── clv_tracker.db
└── temporal/
    └── report_DATA.html

cache/
└── cache_data/          ← Cache chiamate API

quant_engine.db          ← DB principale (fixtures, stats, predictions)
settings.json            ← Config locale (non committare)
manual_context.json      ← Override manuale infortuni/meteo
```

---

> **Nota:** Il motore non garantisce profitti. Un EV positivo nel lungo periodo non elimina la varianza nel breve. Usa sempre il Kelly frazionario e non superare il 5% del bankroll per scommessa.
