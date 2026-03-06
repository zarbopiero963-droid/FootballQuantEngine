# Football Quant Engine

Football Quant Engine è un motore quantitativo per analisi delle partite di calcio basato su modelli matematici.

Funzioni principali:

- Raccolta dati API-Football
- Raccolta quote The Odds API
- Rating ClubElo
- xG Understat
- Feature engineering
- Modelli Poisson
- Bivariate Poisson
- Monte Carlo simulation
- Edge detection
- Value betting detection
- Database SQLite
- Scheduler automatico
- Export CSV / Excel
- Notifiche Telegram

Installazione:

pip install -r requirements.txt

Avvio:

python app/main.py

Build EXE locale:

pyinstaller football_quant_engine.spec

L'EXE viene generato anche automaticamente su GitHub Actions.

