# Orchestration (phase 5)

Un **lot ISO par déclenchement**. `retail replay` et le flow Prefect partagent
`retail.orchestration.replay` : qualité d'**une** semaine, curseur bloqué tant qu'un
lot est `quality_failed`, verrou fichier anti-chevauchement.

Le flow (`flows/weekly_lot.py`) :

- `validate_week_task` : **0** retry ;
- `train_register_task` : 2 retries **uniquement** si l'erreur n'est pas `QualityError` / `TrainError` ;
- déclenchement `interval=600` (config `orchestration.interval_seconds`) ;
- concurrence **1** (`limit=1`) ;
- `on_failure` écrit une alerte (fichier + webhook).

```bash
pip install -r requirements.lock
pip install -e . --no-deps
prefect server start --port 4201
prefect config set PREFECT_API_URL=http://127.0.0.1:4201/api
python flows/weekly_lot.py
```

Sans serveur Prefect : `retail replay`. Alertes dans `data/reports/alerts.jsonl`.
