# Docker

Le serveur MLflow (port **5001**) évite les chemins Windows absolus du stockage fichier
(`mlruns/models/.../meta.yaml`). PostgreSQL (port **5437**) reçoit prédictions, feedbacks,
runs et alertes. Prometheus (**9091**) scrape l'API hôte `:8001`. Grafana (**3001**)
charge sources et tableau de bord depuis `monitoring/` (pas de configuration manuelle).

```bash
docker compose up -d mlflow postgres prometheus grafana
# dans .env :
# RETAIL_MLFLOW_TRACKING_URI=http://127.0.0.1:5001
# RETAIL_DATABASE_URL=postgresql+psycopg://retail:change-me@127.0.0.1:5437/retail
retail db-init   # schéma SQLAlchemy (pas de init.sql : une seule source)
retail train     # le serveur MLflow démarre vide : il faut y enregistrer un champion
retail api --workers 1   # :8001, scrapé par Prometheus via host.docker.internal

# Charge (phase 9) : arrêter l'API hôte, puis
# python load_tests/run.py
# → service compose `api` : python -m retail.api.serve (pas `retail api`)
# spawn Uvicorn réimporte __main__ avant le ping : retail.cli est trop lourd.
```

Les tests unitaires gardent un store fichier temporaire et **SQLite**
(`RETAIL_MLFLOW_TRACKING_URI=file:…`, `RETAIL_DATABASE_URL=sqlite:///…`).
