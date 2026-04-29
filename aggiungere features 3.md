# Audit Features da Aggiungere — Set 3
**Data:** 2026-04-29
**Metodologia:** Ispezione diretta dei file sorgente + grep caller di produzione + verifica test coverage
**Verdetti:** ✅ IMPLEMENTATO | ⚠️ PARZIALE | ❌ NON IMPLEMENTATO

---

## 1. Il "Lineup Sniper" (Algoritmo di Reazione alle Formazioni)

### Descrizione richiesta
Intercetta l'undici titolare ufficiale ~60 min prima del fischio. Confronta con la formazione attesa. Se la star è in panchina, ricalcola l'xG e manda un alert per scommettere prima che il trader del bookmaker aggiorni la quota.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/lineup_sniper.py`: algoritmo completo a 4 step
  1. `register_expected()` — registra la formazione attesa con `xg_contribution` per giocatore
  2. `process_official()` — confronta official vs expected, calcola `xg_delta`
  3. Impatto: `adj_lambda = original_lambda × (1 − xg_delta_fraction)`
  4. Scale d'impatto: `NONE / MINOR (< 0.10) / MAJOR (0.10–0.25) / CRITICAL (> 0.25)` → emette `LineupAlert` con raccomandazione `BET AGAINST / FADE`
- Strutture dati complete: `PlayerProfile`, `ExpectedLineup`, `OfficialLineup`, `LineupDiff`, `LineupAlert`
- Helper `official_lineup_from_dict()` per deserializzare risposta API-Football

**Gap critici:**
1. **Nessun caller di produzione** — `grep -r "lineup_sniper"` restituisce solo il file stesso. Nessun modulo crea un `OfficialLineup` e chiama `process_official()` in risposta a dati reali
2. **Nessuna fonte dati live** — non esiste né un poller API-Football lineups né un Twitter/X scraper. Il docstring cita "Twitter" ma non c'è codice che si connette a nessun feed
3. **Zero test dedicati** — solo `test_coverage_boost.py` con smoke imports

**Per completare:**
- Creare `data/lineup_poller.py`: background thread che chiama `GET /fixtures/lineups?fixture={id}` ogni 5 min nelle 2h pre-match → costruisce `OfficialLineup` → chiama `sniper.process_official()`
- Collegare `on_alert` al notificatore Telegram
- Aggiungere unit tests: `test_critical_player_absent_triggers_alert`, `test_lambda_adjustment_proportional_to_xg_delta`

---

## 2. Motore NLP "Insider" (Analisi Sentiment e Rumors)

### Descrizione richiesta
Analizza tweet dei giornalisti locali e forum tifosi. Cerca "febbre", "infortunio", "tensione". Negative Sentiment Score anomalo → abbassa automaticamente l'Elo della squadra. Scommetti contro prima che le quote riflettano la crisi.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/sentiment_engine.py`: pipeline NLP completa in 4 fasi
  1. **Keyword scan**: pattern multi-categoria (injury, fatigue, conflict, morale) con pesi per categoria
  2. **VADER blend**: se `nltk` disponibile, `final = 0.6 × keyword + 0.4 × vader_compound`
  3. **Recency decay**: `w(t) = exp(−λ × age_hours)`, half-life 24h
  4. **Elo adjustment**: `elo_adj = weighted_score × (−80)` — max −80 Elo punti se score = −1; soglia −0.30 per attivare
- `NewsApiClient`: chiama `newsapi.org/v2/everything` con ricerca per team name, ultimi N giorni → restituisce `List[TextItem]`

**Gap critici:**
1. **Twitter/X assente** — `source: str  # "twitter" | "news" | "forum"` è solo un campo metadato sul `TextItem`; non esiste nessun client Twitter/X API né scraper. Solo NewsAPI.
2. **Non wired nel pipeline di produzione** — `grep -r "SentimentEngine\|sentiment_engine"` in non-test mostra solo imports sparsi; non è chiamato in `quant_engine.py` o nel pre-match pipeline
3. **Coverage al 37%** — riportato esplicitamente in `test_coverage_boost.py` come area da migliorare
4. **Forum tifosi assenti** — nessun client per Reddit, forum italiani (Forza Inter, Juventus Forum, ecc.)

**Per completare:**
- Integrare nell'orchestratore pre-match: `sentiment_report = engine.aggregate_team(team, news_items)` → `adjusted_elo = base_elo + sentiment_report.elo_adjustment`
- Aggiungere Twitter/X Basic API client (o Nitter RSS come fallback gratuito)
- Aggiungere property tests: `elo_adjustment always in [−80, 0]`, `score clampato a [−1, 1]`

---

## 3. Il Creatore di Quote Sintetiche (Synthetic Odds Generator)

### Descrizione richiesta
Usa formule di probabilità condizionata per creare quote sintetiche (es. Risultato Esatto 2-1) partendo dall'1X2 e dall'Over/Under. Confronta con le quote reali del bookmaker → individua mispricing invisibile a occhio nudo.

### Verdetto: ✅ IMPLEMENTATO

**Cosa c'è:**
- `engine/synthetic_odds.py`: pipeline completa
  1. `calibrate(odds)` — grid search (λ_h, λ_a) che minimizza MSE tra probabilità 1X2 derivate da Poisson e quote bookmaker; rimuove overround prima del fitting
  2. `exact_score_matrix(model)` — genera matrice NxN dei risultati esatti con `fair_odds = 1 / model_prob`
  3. `btts_odds(model)` — probabilità BTTS Yes/No dalla matrice
  4. `asian_handicap_odds(model, line)` — linee AH dalla matrice Poisson
  5. `ou_odds(model, line)` — Over/Under con gestione push su linee intere
  6. `find_arbitrage(model, competitor_odds, min_edge=0.04)` — confronta sintetico vs bookmaker, filtra per edge ≥ 4%, ordina per edge decrescente
- `engine/crosschain_arb.py`: estende il concetto al cross-chain arb fiat vs crypto (Betfair vs Polymarket)

**Test:** thin — solo `test_coverage_boost.py` e `test_raises.py` (smoke level)

**Gap:**
- Zero property tests: `sum of exact score probs == 1.0`, `btts_yes + btts_no == 1.0`, `fair_odds == 1/prob`
- Nessun UI combo finder (backend API only — documentato nell'audit)

---

## 4. Network Theory e "Player Synergy" (Grafi di Rete / GNN)

### Descrizione richiesta
GNN (Graph Neural Networks): modella la rete di passaggi come un grafo. Calcola l'impatto dell'assenza di un giocatore specifico sul grafo (effetto non lineare sull'asse chiave Kvaratskhelia-Osimhen). Prevede cali di rendimento invisibili a Poisson/Elo.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/network_synergy.py`: grafo pass-network con **PageRank** puro (nessuna ML)
  - Nodi = giocatori, archi = passaggi con `dangerous_count / total_dangerous` come peso
  - `pagerank()`: algoritmo power-iteration (damping 0.85, 50 iterazioni) — implementato from scratch senza librerie
  - `build_network()`: normalizzazione pesi, centralità PageRank per ogni nodo
  - Quantifica impatto assenza: `synergy_loss = sum of centrality × weight per nodo rimosso`

**Gap critici — non è un GNN:**
1. **Nessun PyTorch / PyTorch Geometric** — zero import di `torch`, `torch_geometric`, `dgl`
2. **Nessun training su dati storici** — i pesi del grafo vengono passati manualmente, non appresi da Opta/StatsBomb
3. **Nessun embedding appreso** — le rappresentazioni dei giocatori sono centralità PageRank calcolate analiticamente, non vettori latenti da GNN
4. **Non wired in produzione** — nessun caller esterno a `engine/network_synergy.py`

**Cosa serve per diventare un vero GNN:**
```
1. Dati:  statsbombpy o Opta → grafi di passaggi storici per partita
2. Model: torch_geometric.nn.GCNConv o SAGEConv
          - nodi: feature player (età, posizione, xG storico)
          - archi: freq. passaggi + passaggi pericolosi
3. Train: predire xG_team da grafo → minimizzare MSE su 10.000+ partite
4. Infer: rimuovi nodo assente → delta in xG_predicted = impatto reale
```
**Effort stimato: 1–2 settimane** (come indicato nell'AUDIT_REPORT.md)

---

## 5. Motore di Arbitraggio Puro (Surebet Scanner)

### Descrizione richiesta
Scarica quote da 10+ bookmaker simultaneamente. Cerca overround < 100%. Calcola gli stake ottimali per ogni leg. Profitto matematico garantito indipendentemente dall'esito.

### Verdetto: ✅ IMPLEMENTATO

**Cosa c'è:**
- `engine/surebet_scanner.py`: implementazione completa
  - `OddsApiClient`: chiama the-odds-api.com, fetch simultaneo multi-bookmaker, parsing 2-way e 3-way
  - `SurebetScanner.scan_fixture()`: rileva arb per singola partita — overround < 1.0 su qualsiasi combinazione di outcome da bookmaker diversi
  - `SurebetScanner.scan_all()`: itera tutte le fixture, aggrega `ScanResult` con timing
  - `optimal_stakes()`: formula matematicamente corretta
    ```
    stake_i = bankroll × (1/odds_i) / overround
    profit  = bankroll × (1/overround − 1)
    ```
  - Filtro: `MIN_PROFIT_PCT = 0.3%` — ignora arb troppo piccoli
  - `SurebetOpportunity`: bookmaker, stake, guaranteed_profit, profit_pct per ogni leg

**Test:** thin — `test_coverage_boost.py` e `test_raises.py` (smoke level senza property tests)

**Gap:**
- Zero property tests: `sum(1/odds_i) < 1.0` implica `profit > 0`, `optimal_stakes produce guaranteed return`
- Nessun test con dati reali mock (es. fixture con overround 0.97 → profit 3.1%)
- Nessuna UI per visualizzare opportunità in tempo reale

---

## Riepilogo

| # | Feature | Verdetto | Gap principale |
|---|---------|----------|----------------|
| 1 | Lineup Sniper | **⚠️ PARZIALE** | Algoritmo completo ma non wired ad alcun dato live; nessun caller di produzione |
| 2 | NLP Sentiment | **⚠️ PARZIALE** | Twitter assente; non integrato nel pipeline pre-match; coverage 37% |
| 3 | Synthetic Odds | **✅ IMPLEMENTATO** | Test thin; nessuna UI combo finder |
| 4 | Network Synergy / GNN | **⚠️ PARZIALE** | PageRank rule-based, **non è un GNN**; nessun training su dati storici |
| 5 | Surebet Scanner | **✅ IMPLEMENTATO** | Test thin; nessuna UI real-time |

**Implementati completamente: 2/5**
**Parziali (logica core presente, gap di integrazione o assenza dati live): 3/5**
**Non implementati: 0/5**

### Priorità di completamento
1. **Lineup Sniper** (effort 1–2 giorni): aggiungere `lineup_poller.py` che chiama API-Football → già tutto pronto dall'altra parte
2. **NLP Sentiment** (effort 2–3 giorni): wiring nel pre-match pipeline + Twitter/Nitter client
3. **GNN Player Synergy** (effort 1–2 settimane): richiede dati Opta/StatsBomb + PyTorch Geometric

---

*Audit condotto: 2026-04-29 | Metodologia: ispezione diretta file sorgente, grep caller produzione, analisi import, verifica test coverage*
