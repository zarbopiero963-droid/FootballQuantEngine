# Audit Features da Aggiungere — Set 4
**Data:** 2026-04-29
**Metodologia:** Ispezione diretta dei file sorgente + grep caller produzione + verifica test coverage
**Verdetti:** ✅ IMPLEMENTATO | ⚠️ PARZIALE | ❌ NON IMPLEMENTATO

---

## 1. Il Distruttore di Bet Builder (Correlated Parlay Exploiter)

### Descrizione richiesta
Il motore calcola la vera correlazione non lineare tra eventi della stessa partita. "Inter vince" + "Lautaro segna" + "Meno di 3.5 Cartellini" — il bookmaker usa Gaussiane fisse; il tuo modello usa correlazioni non lineari con tail dependence. Trova Edge +150%.

### Verdetto: ✅ IMPLEMENTATO con test seri

**Cosa c'è:**
- `engine/gaussian_copula.py`: tre famiglie copula complete
  - `"gaussian"` — Gaussiana standard, zero tail dependence
  - `"clayton"` — lower tail dependence `λ_L = 2^(−1/θ)` — modella joint losses (scommesse che perdono insieme quando il favorito delude)
  - `"gumbel"` — upper tail dependence `λ_U = 2 − 2^(1/θ)` — modella joint wins su favoriti correlati
- `engine/copula_math.py`: 50.000-path Monte Carlo puro Python, Cholesky decomposition, Acklam inverse normal CDF (errore < 1.15e-9), sampler Clayton (Gamma-Marshall-Olkin) e Gumbel (Chambers-Mallows-Stuck stabile in log-space)
- `engine/copula_types.py`: `BetLeg` (validazione market_odds > 1.0, model_prob ∈ (0,1)), `CopulaResult`, `CopulaCorrelation`
- `evaluate_bet_builder()`: confronta probabilità congiunta del modello vs probabilità implicita bookmaker → calcola edge%

**Test — i più seri della codebase:**
- `tests/property/test_copula_properties.py`: 9 Hypothesis property tests, 200–300 esempi ciascuno:
  - joint probability sempre in [0, 1] per tutte e 3 le famiglie
  - `rho=0` Gaussian ≈ prodotto marginali (indipendenza), tolleranza 5σ Monte Carlo
  - comonotonicity: Clayton/Gumbel con θ=50 → `analytical_value − 5σ` (bound statisticamente corretto)
  - `BetLeg` validation: probabilità invalide e quote ≤ 1.0 sollevano `ValueError`
- `tests/invariant/test_quant_financial_invariants.py`: invarianti finanziarie

**Gap:** Nessun UI combo finder — è backend API only (documentato). Zero correlazioni reali apprese da dati storici: le correlazioni tra "home_win" e "btts" sono costanti in `DEFAULT_CORRELATIONS`, non calibrate su dati.

---

## 2. Latency Arbitrage (High-Frequency Live Betting)

### Descrizione richiesta
Modulo Python Async (WebSockets) agganciato al feed dati puro dallo stadio (Opta, Radar). Quando un gol arriva in 0.5s, spara la scommessa nel buco di 1s prima che il bookmaker sospenda il mercato a 1.5s.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/latency_arb.py`: architettura async completa
  - `FeedSimulator`: async generator che riproduce eventi con jitter di latenza configurabile
  - `BookmakerMonitor`: polling loop asincrono con `asyncio.sleep`, rileva sospensione simulata
  - `LatencyArbEngine`: coordina i due stream — quando fast feed rileva un goal/red card, chiama `notify_event()` sul monitor e misura la finestra disponibile
  - `BetInstruction`: bet auto-generata con `fired_at_ms`, stake, tipo evento
  - `LatencyProfiler`: rolling statistics su latency misurata
  - `LatencyArbSession`: `bets_fired`, `successful_windows`, `avg_latency_ms`
  - Import condizionale di `aiohttp` e `websockets` (graceful degradation se non installati)

**Gap critici:**
1. **`FeedSimulator` è un simulatore, non un feed reale** — non esiste nessuna connessione WebSocket a Opta, Stats Perform, Radar, o SportRadar. Il test di latenza gira su eventi fittizi in memoria
2. **Nessun client bookmaker reale** — `BookmakerMonitor._fetch_fn` è un callable astratto; non esiste nessun client Bet365/Snai/William Hill che polla quote live e rileva sospensione
3. **Nessuna esecuzione bet automatica** — `BetInstruction` viene emessa ma nessun modulo la consegna a un'API bookmaker
4. **Test thin** — solo `test_coverage_boost2.py` e `test_raises.py`

**Per completare:**
- Implementare `FeedAdapter` che si connette via WebSocket a Stats Perform o Opta feed (richiede contratto dati)
- Implementare `BookmakerApiClient` per uno specifico bookmaker con endpoint odds + stato mercato
- Collegare `on_instruction` al client di piazzamento scommesse

---

## 3. Expected Threat (xT) & Pitch Control

### Descrizione richiesta
Implementa il modello xT (Karun Singh). Il motore divide il campo in griglia 2D. Ogni passaggio viene valutato per quanto ha aumentato la probabilità di segnare nei prossimi 10 secondi. Trova squadre che "soffocano" l'avversario prima ancora di tirare.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/expected_threat.py`: implementazione completa del modello Karun Singh (2018)
  - Griglia 16×12 (192 zone), valori `_DEFAULT_XT_GRID` derivati da open StatsBomb data
  - `XTModel`: `zone_for(x, y)`, `xt_at(x, y)`, `evaluate_touch()` → xT gain per evento
  - `XTReport`: `total_xt`, `total_xt_conceded`, `xt_dominance_ratio`, `top_zones` (top 5 zone pericolose)
  - `xt_gain = xt_to − xt_from` per passaggi riusciti; penalità per palla persa
- `engine/pitch_control.py`: modello Spearman pitch control
  - Per ogni cella della griglia 32×20 (3.28m × 3.40m), calcola `t_i` (time-to-intercept) per ogni giocatore
  - Output `PitchControlSurface`: `home_control[row][col]` ∈ [0,1], `home_territory_pct`, `dangerous_home_pct`
  - `XTPitchControl`: integra pitch control con griglia xT → `home_xt_control = Σ(pc_home × xt)` per ogni cella

**Gap critici:**
1. **Non wired in nessun pipeline di produzione** — nessun modulo chiama `XTModel` o `run_pitch_control()` durante il pre-match o il live. Nessun alert generato su xT anomalo
2. **Nessuna fonte dati eventi** — richiede `List[TouchEvent]` con coordinate (x, y) precise. API-Football non fornisce coordinate passaggi; serve Opta, StatsBomb, o SkillCorner
3. **Test thin** — solo `test_coverage_boost2.py`
4. **`_DEFAULT_XT_GRID` non aggiornabile** — i valori sono hardcoded; nessun meccanismo di retraining dalla stagione in corso

**Per completare:**
- Collegare a `statsbombpy` (open source) per dati di eventi con coordinate
- Aggiungere xT ratio nell'output di `quant_engine.py` come feature per il modello ML
- Aggiungere property test: `xt_gain ∈ [−1, 1]`, `total_xt >= 0`

---

## 4. Reverse Engineering del Rischio (Order Book / Betfair Depth of Market)

### Descrizione richiesta
Scarica la Depth of Market di Betfair (libro ordini completo con muri di denaro in attesa). Analizza l'imbalance di volume per prevedere in che direzione sta per crollare la quota. Compra prima del crollo, rivendi in greenbook garantito.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/orderbook_analyzer.py`: implementazione backend completa
  - `OrderBookSnapshot`: full depth con `available_to_back` e `available_to_lay` a tutti i livelli di prezzo
  - `OrderBookImbalance`: `imbalance = (back_vol − lay_vol) / (back_vol + lay_vol) ∈ [−1, 1]`, `back_wall` (livello supporto), `lay_wall` (livello resistenza), direzione trend
  - `compute_greenbook()`: formula esatta `lay_stake = back_stake × back_odds / lay_odds` → `(profit_if_wins, profit_if_loses)`
  - `OrderBookAnalyzer.analyse()`: calcola imbalance per snapshot corrente
  - `OrderBookAnalyzer.check_alert()`: emette `OrderFlowAlert` se imbalance > soglia
  - `OrderBookAnalyzer.trend_imbalance()`: regressione lineare su storico imbalance → predice direzione movimento
  - `BetfairDepthClient`: client REST per `listMarketBook` con `priceProjection=EX_ALL_OFFERS` (full ladder)

**Gap critici:**
1. **Nessun collegamento a UI o notifiche** — `OrderFlowAlert` viene emessa ma non consumata da nessun componente
2. **Autenticazione Betfair non gestita** — richiede certificato SSL + `certlogin`; stesso problema di `smart_money_tracker.py`
3. **Zero test** — nessun test dedicato per `orderbook_analyzer.py`
4. **Nessuna esecuzione automatica** — il greenbook viene calcolato ma non c'è integrazione con Betfair API per piazzare la lay bet in automatico

---

## 5. Arbitraggio Cross-Chain (Web3 Crypto vs Fiat)

### Descrizione richiesta
Confronta costantemente le quote di Pinnacle (Fiat/€) con quelle degli Smart Contract su Blockchain (SX.bet, Polymarket). Trova buchi di quota enormi. Scommetti Euro sul bookmaker tradizionale e USDC sul protocollo Web3 simultaneamente.

### Verdetto: ⚠️ PARZIALE

**Cosa c'è:**
- `engine/crosschain_arb.py`: scanner completo
  - `SXBetClient`: client REST `api.sx.bet` (Polygon, no auth richiesto), parsing quote con encoding `implied_prob × 10^20`
  - `PolymarketClient`: client REST `clob.polymarket.com` (prediction market), parsing CLOB orders
  - `CrossChainScanner`: confronta fiat vs crypto — `overround = 1/fiat_odds + 1/crypto_odds`; se < 1.0 → opportunità
  - `CrossChainOpportunity`: `fiat_leg`, `crypto_leg`, `fiat_stake`, `crypto_stake_usd`, `guaranteed_profit_usd`, `profit_pct`
  - Formula stake ottimale: `fiat_stake = bankroll × (1/fiat_odds) / overround` — matematicamente corretta
  - `CrossChainScanResult.summary()`: n_fiat_markets, n_crypto_markets, n_opportunities, profit range

**Gap critici:**
1. **Nessuna esecuzione automatica della leg crypto** — il modulo rileva l'opportunità ma non ha integrazione `web3.py` per firmare e inviare transazioni su Polygon. Nessun `send_transaction()`, nessun wallet configurato
2. **Nessuna esecuzione automatica della leg fiat** — stessa assenza del modulo latency_arb: nessun client bookmaker per piazzare la scommessa fiat automaticamente
3. **Test thin** — solo `test_coverage_boost2.py`
4. **Rischio di frontrunning non gestito** — le transazioni blockchain hanno latenza variabile (mining time); l'opportunità può chiudersi tra rilevamento ed esecuzione

**Per completare:**
- Aggiungere `web3.py` per la leg crypto: connessione RPC Polygon, firma transazione con chiave privata, interazione con smart contract SX.bet
- Collegare la leg fiat a un cliente bookmaker con API (es. Pinnacle API o the-odds-api con esecuzione)
- Aggiungere gestione slippage e frontrunning nel calcolo profit atteso

---

## Riepilogo

| # | Feature | Verdetto | Gap principale |
|---|---------|----------|----------------|
| 1 | Copula Bet Builder | **✅ IMPLEMENTATO** | No correlazioni calibrate su dati storici; no UI combo finder |
| 2 | Latency Arbitrage | **⚠️ PARZIALE** | `FeedSimulator` solo simulazione; nessun WebSocket reale; nessun client bookmaker |
| 3 | xT / Pitch Control | **⚠️ PARZIALE** | Modelli corretti ma non wired in produzione; richiede dati eventi con coordinate |
| 4 | Order Book Betfair | **⚠️ PARZIALE** | Backend completo; no UI/notifiche; no esecuzione automatica; no auth flusso |
| 5 | Cross-Chain Arb | **⚠️ PARZIALE** | Detection ok; no `web3.py` esecuzione; no client fiat automatico |

**Implementati completamente: 1/5**
**Parziali (logica core o detection presente, gap di esecuzione/integrazione/dati): 4/5**
**Non implementati: 0/5**

### Nota comune ai 4 parziali
Tutti condividono lo stesso pattern architetturale: **"backend completo, ultima miglia mancante"**. Il motore di calcolo esiste e funziona; manca il connettore verso il mondo reale (feed live, API bookmaker, wallet crypto, UI). Questo è il lavoro restante.

---

*Audit condotto: 2026-04-29 | Metodologia: ispezione diretta file sorgente, grep caller produzione, analisi import, verifica test coverage*
