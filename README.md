# ⚽ Football Quant Engine

Football Quant Engine è un motore quantitativo per analisi e predizione delle partite di calcio, costruito in Python con architettura modulare, modelli matematici, dataset locale e supporto a workflow desktop/offline-first.

## Caratteristiche principali

- Motore di analisi modulare
- Sistema plugin per modelli predittivi
- Analisi probabilistica delle partite
- Monte Carlo simulation
- Edge detection
- Ranking automatico
- Dataset builder
- Backtest engine
- Export CSV / Excel
- Report HTML
- Dashboard HTML locale
- UI desktop PySide6
- Modalità offline-first con import CSV
- Build automatica EXE con GitHub Actions

## Modelli inclusi

- Poisson Model
- Bivariate Poisson
- Elo Model
- Bayesian Model
- Probability Markets (Over/Under 2.5, BTTS)
- Monte Carlo

## Avvio

UI desktop:

python app/main.py

CLI:

python app/cli.py

## Build locale

python -m PyInstaller football_quant_engine.spec

