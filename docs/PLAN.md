# Feuille de route

Principe : une **tranche verticale d'abord** (phases 0 → 4 + API minimale), puis on
épaissit couche par couche. Chaque phase se termine par des tests et une commande qui la démontre.

| # | Phase | Contenu | Preuve |
|---|---|---|---|
| 0 | Socle | structure, `pyproject` + lock, config YAML + `.env`, CLI, logging JSON, ruff, pre-commit, CI | CI verte |
| 1 | Ingestion | téléchargement borné (2 essais), SHA256 figé, Parquet typé, lots hebdo ISO, manifestes | `retail data` reproduit les mêmes empreintes |
| 2 | Qualité | **2 schémas Pandera** : `InvoiceSchema` (factures brutes : colonnes, types, dates dans la semaine, volume, part de prix ≤ 0) puis `PanelSchema` (après nettoyage : 1 ligne/SKU/semaine, quantités ≥ 0, tous les produits restants). Nettoyage tracé : annulations `C`, quantités négatives, codes non-produits, valeurs aberrantes, filtre Royaume-Uni. Rapport JSON. Lot corrompu pour la démo | lot invalide = étape en échec, pas d'entraînement |
| 3 | Features + modèle | panel hebdo, lags −1/−2/−4, moyenne mobile 4 sem., calendrier, prix ; **exactement 50 SKU** figés sur les semaines de démarrage ; pipeline sklearn ; baseline naïve ; MAE + WAPE ; split temporel | test anti-fuite, run MLflow |
| 4 | Registre | MLflow (params, métriques, hash données, commit Git, artefacts). Battre la baseline → enregistré `challenger` ; MAE ≤ champion → promu `champion`. Rollback = ré-alias. Export du champion sur disque | historique des versions, démo rollback |
| 5 | Orchestration | flow Prefect, déclenchement planifié (un lot par tick), retries sur erreurs techniques uniquement, alerte webhook local + logs | UI Prefect : runs OK / KO |
| 6 | API | FastAPI : `/predict {StockCode, week}`, `/predict/batch {week}`, `/predict/features`, `/feedback`, `/health`, `/model`, `/model/reload`, `/metrics`, `/alerts` ; `prediction_id` ; Pydantic ; 404/422 ; `model_version` ; mode dégradé sur champion local puis `.champion.bak` | Swagger, tests API |
| 7 | Persistance | PostgreSQL : `predictions`, `feedbacks`, `pipeline_runs`, `alerts` ; feedback auto au lot N+1 | WAPE en production par version |
| 8 | Supervision | Prometheus + Grafana : RPS, p50/p95/p99, erreurs, version servie, WAPE réel, statut des flows | captures |
| 9 | Charge | Locust, 3 configs (1w/0,5 vCPU, 2w/1 vCPU, 4w/2 vCPU) × rampe/plateau/pic | tableau + recommandation chiffrée |
| 10 | Tests | unitaires, API, intégration | `make test`, CI |
| 11 | Démo + livrables | `make demo` (3 lots : nominal, promotion, invalide), incident « MLflow/Postgres down », rollback, `SYNTHESE_OPS.md` | scénario de 10 min |

Ports locaux (fixés dans `.env.example`, modifiables) : Postgres 5437, MLflow 5001,
Prefect 4201, API 8001, Grafana 3001, Prometheus 9091.
