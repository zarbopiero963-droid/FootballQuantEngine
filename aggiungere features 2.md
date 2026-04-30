# Audit Features da Aggiungere — Set 2
**Data:** 2026-04-29
**Metodologia:** Ispezione diretta dei file sorgente + verifica test coverage + lettura codice API
**Verdetti:** ✅ IMPLEMENTATO | ⚠️ PARZIALE | ❌ NON IMPLEMENTATO

---

## 7. Motore Meteo (Weather Impact Engine)

### Descrizione richiesta
L'engine interroga un'API meteo per le coordinate dello stadio. Pioggia forte = meno gol, vantaggio per il gioco fisico. Vento forte = under favoriti. Il modello adatta le stime dei gol al meteo.

### Verdetto: ✅ IMPLEMENTATO

**Cosa c'è:**
- `engine/weather_engine.py`: chiama **OpenWeatherMap API** per coordinate stadio, calcola `lambda_multiplier` con bracket precisi e additivi:
  ```
  Vento:       > 50 kph → −0.20  |  > 35 kph → −0.12  |  > 20 kph → −0.06
  Pioggia:     > 8 mm/h → −0.18  |  > 3 mm/h → −0.10  |  > 0.5 mm/h → −0.05
  Temperatura: < 0°C   → −0.08  |  < 5°C   → −0.05  (tutti additivi)
  ```
  Moltiplicatore finale: `max(0.50, 1.0 + somma_aggiustamenti)` — floor a 0.50 per evitare λ irrisori
- λ_home e λ_away adattati simmetricamente; per-lambda minimum di 0.10
- Integrato in `quant/services/quant_engine.py`: `self.weather_engine = weather_engine` passato a ogni predizione
- Freddo (<5°C) boostra automaticamente il peso del modello Skellam nell'ensemble (`engine/meta_learner.py`)

**Gap:**
- Zero test dedicati per `engine/weather_engine.py` (nessun file `test_weather_engine.py`)
- Nessun test delle invarianti: moltiplicatore sempre in [0.50, 1.0], λ_adj >= 0.10

**Per completare:**
- Aggiungere unit tests: `test_multiplier_in_range`, `test_heavy_rain_reduces_lambda`, `test_extreme_cold_reduces_lambda`

---

## 8. Indice di Fortuna (Luck Index / xPTS / Regression to Mean)

### Descrizione richiesta
Calcola i Punti Attesi (xPTS) di ogni squadra dai suoi xG e li confronta con i punti reali. Squadra con 10 pts ma 20 xPTS = sfortunata → scommettere su di lei (regressione verso la media). Squadra con 20 pts ma 10 xPTS = fortunata → fadearla.

### Verdetto: ✅ IMPLEMENTATO

**Cosa c'è:**
- `engine/luck_index.py`: formula xPTS completa su matrice Poisson joint:
  ```
  xPTS_home = Σ_{h>a} 3·P(h,a) + Σ_{h=a} 1·P(h,a)
  xPTS_away = Σ_{h<a} 3·P(h,a) + Σ_{h=a} 1·P(h,a)
  luck(team) = actual_pts − xPTS
  ```
- Auto-flag: `luck_per_match < −0.5` → `is_regression_candidate` (scommetti su); `luck_per_match > +0.5` → `is_fade_candidate`
- `regression_candidates()` e `fade_candidates()` ordinati per intensità
- Poisson PMF in **log-space** (`log_pmf = k·log(λ) − λ − log Γ(k+1)`) per stabilità numerica su λ grandi

**Gap:** Nessuno critico. I xPTS dipendono dalle stime λ del modello Poisson: la qualità dell'indice è limitata dalla qualità delle λ in input.

---

## 9. Analisi Arbitrale e Mercato Cartellini

### Descrizione richiesta
Scarica lo storico dell'arbitro, misura la sua severità (media cartellini/partita), incrocia con la media falli delle squadre. Genera edge su Over 4.5 Cartellini e Rigore Assegnato: Sì.

### Verdetto: ⚠️ PARZIALE (con bug critico nei dati)

**Cosa c'è:**
- `engine/referee_analyzer.py` (modello statistico completo):
  ```
  expected_yellows = ref_avg_yellows
                     + 0.5 × (home_fouls + away_fouls − league_avg_fouls × 2)
                               × ref_card_rate_per_foul
  ```
  Mercati coperti: Over 3.5 / 4.5 / 5.5 yellows, Red card YES, Over 4.5 total cards (Poisson CDF)
- `quant/models/referee_engine.py`: `fit()`, `get_home_bias()`, `get_strictness()`, `get_lambda_modifiers()` — integrato in `quant_engine.py`
- `quant/providers/api_football_client.py` — `get_referee_stats()`: metodo API esiste, itera le fixture completate

**Bug critico in `get_referee_stats()`:**

```python
# CODICE ATTUALE — bug: home_cards e away_cards non vengono mai incrementati
ref_acc.setdefault(ref, {"home_cards": 0, "away_cards": 0, "n": 0})
ref_acc[ref]["n"] += 1
# ← mancano: ref_acc[ref]["home_cards"] += item["statistics"]["home_cards"]
#              ref_acc[ref]["away_cards"] += item["statistics"]["away_cards"]

# Risultato: strictness = (0 + 0) / n = 0.0 per ogni arbitro
#            home_bias  = (0 − 0) / n = 0.0 per ogni arbitro
```

Il modello riceve dati sempre a zero → le predizioni sui cartellini sono identiche per tutti gli arbitri → nessun edge generato.

**Gap aggiuntivi:**
- Nessuna tabella `referee_stats` nel DB → i dati vengono ricalcolati da zero a ogni sessione (nessuna persistenza)
- Zero test per `engine/referee_analyzer.py`

**Per completare (fix prioritario):**
```python
# In get_referee_stats() — aggiungere lettura statistiche cartellini
stats_list = item.get("statistics") or []
for stat in stats_list:
    team_type = stat.get("team", {}).get("name", "")
    yellows = stat.get("cards", {}).get("yellow", 0) or 0
    if team_type == item["teams"]["home"]["name"]:
        ref_acc[ref]["home_cards"] += yellows
    else:
        ref_acc[ref]["away_cards"] += yellows
```
- Aggiungere tabella `referee_stats` al DB schema con indice su `referee_name`
- Aggiungere test: `strictness > 0` per arbitri con storico

---

## 10. Tracciamento Smart Money (Betfair Exchange Volume)

### Descrizione richiesta
API Betfair Exchange per monitorare la liquidità abbinata. Se la quota scende e c'è volume anomalo in pochi minuti → alert Smart Money: sindacato professionale è entrato.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/smart_money_tracker.py` (algoritmo completo):
  - `BetfairExchangeClient`: client APING REST API con `list_market_catalogue()`, `list_market_book()`, `extract_snapshots()`
  - `SmartMoneyTracker`: rilevamento spike con finestra 5 minuti, soglia **3× baseline**, minimo **£5,000**
  - `SmartMoneyAlert`: emesso con `volume_ratio`, `odds_before`, `odds_after`, stima trend
  - Callback `on_alert` configurabile su detection

**Gap critici:**
1. **Nessuna integrazione UI/notifica** — il backend genera `SmartMoneyAlert` ma nessun componente lo consuma: non c'è connessione a `notifications/telegram_notifier.py` né a una finestra PySide6
2. **Autenticazione Betfair non gestita** — l'API Betfair richiede certificato SSL + `certlogin`; il `session_token` deve essere ottenuto e rinnovato esternamente; non esiste un flusso di autenticazione automatico
3. **Zero test** — nessun file di test per `SmartMoneyTracker` o `BetfairExchangeClient`

**Per completare:**
- Collegare `on_alert` al notificatore Telegram: `tracker = SmartMoneyTracker(on_alert=telegram.send_smart_money_alert)`
- Aggiungere UI panel con feed live degli alert
- Implementare flusso di autenticazione Betfair (cert + session token refresh)
- Aggiungere unit tests con snapshot mock: `test_spike_detected_above_threshold`, `test_no_alert_below_min_volume`

---

## 11. Dashboard e Tracker del CLV (Closing Line Value)

### Descrizione richiesta
Vista UI che confronta la quota a cui l'engine ha consigliato la scommessa con la quota di chiusura dello sharp book. L'unico modo per sapere se il modello funziona davvero nel lungo periodo.

### Verdetto: ✅ IMPLEMENTATO con test seri

**Cosa c'è:**
- `ui/clv_tracker_window.py`: UI PySide6 completa
  - Tabella interattiva: `bet_id`, `fixture_desc`, `market`, `our_odds`, `closing_odds`, `CLV%`, `result`, `profit`
  - Formula: `CLV% = (our_odds / closing_odds − 1) × 100`
  - Inserimento e aggiornamento closing odds direttamente dalla UI
  - DB persistence su tabella `clv_bets` con `sqlite3`
- `training/closing_line_builder.py`: builder separato per costruire la closing line da fonti esterne

**Test — la feature più testata del set:**
- `tests/test_clv_tracker.py` — test funzionale output tracker
- `tests/test_clv_journal_service.py` — test record e settle di una scommessa
- `tests/test_app_clv_controller.py` — test del controller applicativo
- `tests/test_clv_runner.py` — test del runner CLV
- `tests/test_bet_journal.py`, `tests/test_performance_report.py`, `tests/test_pnl_calculator.py` — test metriche correlate

**Gap:**
- Closing odds inserite **manualmente** post-partita — nessun feed automatico da fonte sharp (Pinnacle, Betfair)
- Nessuna API integration per il caricamento automatico della closing line

---

## Riepilogo

| # | Feature | Verdetto | Gap principale |
|---|---------|----------|----------------|
| 7 | Weather Engine | **✅ IMPLEMENTATO** | Zero test dedicati |
| 8 | Luck Index / xPTS | **✅ IMPLEMENTATO** | Nessuno critico |
| 9 | Referee Analysis | **⚠️ PARZIALE** | Bug: `home_cards`/`away_cards` mai incrementati → strictness sempre 0.0 |
| 10 | Smart Money / Betfair | **⚠️ PARZIALE** | Backend solido ma non collegato a UI/notifiche; zero test |
| 11 | CLV Dashboard | **✅ IMPLEMENTATO** | Closing odds manuali, nessun feed automatico |

**Implementati completamente: 3/5**
**Parziali (logica core presente, gap di integrazione o bug dati): 2/5**
**Non implementati: 0/5**

### Bug da fixare subito (prima di andare live)
- **BUG-REFEREE-001** — `get_referee_stats()` in `quant/providers/api_football_client.py` non popola `home_cards`/`away_cards` → strictness = 0.0 → il modello cartellini non genera edge reale

---

*Audit condotto: 2026-04-29 | Metodologia: ispezione diretta file sorgente, lettura API client, grep dipendenze reali, verifica test coverage*
