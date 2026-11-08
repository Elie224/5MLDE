# retail-mlops — prévision des ventes hebdomadaires par produit

Chaîne MLOps locale autour d'un cas de **réassort** : prévoir, pour chacun des **50
produits** les plus vendus d'un e-commerce britannique, la **quantité vendue la
semaine suivante**.

Le modèle (Ridge vs naïf) est volontairement simple : on évalue la chaîne
données → qualité → entraînement → registre → API → supervision → charge.

- Données : [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
  (CC BY 4.0), rejouées **par lots hebdomadaires** — [docs/DATA.md](docs/DATA.md).
- Feuille de route : [docs/PLAN.md](docs/PLAN.md).
- Dimensionnement et incidents : [docs/SYNTHESE_OPS.md](docs/SYNTHESE_OPS.md).

## Architecture

```
 Lots ISO ──► Prefect `weekly_lot` : qualité (Pandera) → train → MLflow (challenger/champion)
                    │ échec données = arrêt, pas de retry métier
                    ▼
             PostgreSQL (prédictions, feedbacks, runs, alertes, WAPE)
                    ▲
 FastAPI /predict ──┤── champion disque (mode dégradé si MLflow down)
                    ▼
         Prometheus ──► Grafana          Locust (3 configs Docker)
```

Ports (`.env.example`) : Postgres **5437**, MLflow **5001**, Prefect **4201**,
API **8001**, Grafana **3001**, Prometheus **9091**.

## Récupération des données

L'archive UCI n'est **pas** dans Git. `retail data` la télécharge, vérifie le
**SHA256 figé** dans `configs/data.yaml`, convertit en Parquet et découpe 51 lots ISO
(trou 2010-W52). Si le site UCI est down : déposer `online_retail.zip` dans
`data/raw/` puis relancer. Une archive dont l'empreinte diffère est **refusée**.

Le filtre Royaume-Uni et le top 50 se font au nettoyage / warmup, pas au téléchargement.

## Depuis un clone

Prérequis : Python 3.11–3.13, Git, Docker. PowerShell : activer `.venv\Scripts\Activate.ps1`.

```bash
python -m venv .venv
pip install -r requirements.lock
pip install -e . --no-deps
cp .env.example .env          # Windows : copy .env.example .env

retail check-config
docker compose up -d postgres mlflow prometheus grafana
# image MLflow : tag GHCR v2.22.2 (le client Python est 2.22.5 ; v2.22.5 n'existe pas)
retail db-init                # schéma SQLAlchemy (pas de init.sql)

retail data                   # UCI + lots hebdo
retail quality                # Pandera bloquant
retail train --until-week 2011-W38
retail train --until-week 2011-W40   # registre Docker :5001 (pas mlruns/)

# API hôte : le conteneur `api` (profil load) occupe aussi :8001
docker compose --profile load stop api
# autre terminal
retail api                    # :8001  (GET /health → source, version, db_backend)
# bind toutes interfaces : retail api --host 0.0.0.0

retail demo                   # nominal, lot invalide, Postgres down, rollback
# avant soutenance (curseur W40/W41, évite l'épuisement vers W48) :
retail demo --reset-cursor
# make demo

# charge : arrêter l'API hôte (le run.py lève l'API Docker)
# recréer le conteneur si /metrics est vide (tmpfs /tmp/prom) :
# docker compose --profile load up -d --force-recreate api
python load_tests/run.py --smoke --only 1w_0p5cpu
python load_tests/run.py
# trafic plat ~1 req/s/user : $env:LOCUST_CONSTANT=1
```

Prefect (optionnel), **deux** terminaux — `prefect server start` ne doit pas
rendre le prompt. IPv4 uniquement (`localhost` peut partir sur IPv6) :

```bash
# terminal 1
$env:PREFECT_API_URL = "http://127.0.0.1:4201/api"   # bash : export …
prefect server start --host 127.0.0.1 --port 4201

# terminal 2
$env:PREFECT_API_URL = "http://127.0.0.1:4201/api"
python flows/weekly_lot.py
prefect deployment run "retail-weekly-lot/retail-weekly-lot"   # run immédiat
```

Sans Prefect : `retail replay`. `.env` : `RETAIL_MLFLOW_TRACKING_URI=http://127.0.0.1:5001`
(obligatoire pour que train, API et Prefect parlent au **même** registre).

Grafana : http://127.0.0.1:3001 (admin / `GRAFANA_ADMIN_PASSWORD`). WAPE : d'abord
prédire N+1, **puis** rejouer N+1 (`retail demo` le fait).

## Commandes

Chaque `make` appelle `retail` (utilisable sans `make` sous Windows).

| make | équivalent | rôle |
|---|---|---|
| `make install` | `pip` lock + `-e .` | environnement |
| `make check-config` | `retail check-config` | YAML |
| `make data` | `retail data` | UCI, SHA256, Parquet, lots |
| `make quality` | `retail quality` | 2 schémas Pandera ; lot invalide = arrêt |
| `make quality-invalid` | `retail quality --invalid-demo` | lot corrompu (code 0 si rejeté) |
| `make train` | `retail train` | split temporel, MLflow, promotion |
| `make rollback` | `retail rollback` | `previous_champion` + export disque |
| `make replay` | `retail replay [--week]` | un lot ISO |
| `make api` | `retail api [--workers N]` | FastAPI :8001 |
| `make postgres` / `mlflow` / `monitor` | `docker compose up -d …` | Postgres, MLflow, Prom+Grafana |
| `make db-init` | `retail db-init` | tables |
| `make demo` | `retail demo` | scénario 10 min |
| `make load` | `python load_tests/run.py` | Locust × 3 configs |
| `make test` | `pytest` | unitaires + intégration |

`retail demo` exige l'**API hôte** (`retail api` sur :8001, pas le conteneur de
charge), Postgres sain, et **deux versions** dans MLflow :5001. Sinon : `retail train
--until-week 2011-W38` puis `--until-week 2011-W40`. La démo avance avec le
**curseur** : prédire N+1 puis livrer N+1 (jamais W48 Locust). `--reset-cursor`
revient à W40 : **une fois** avant la soutenance, pas en boucle (sinon WAPE W41
se remplit). Elle force un challenger le temps du rollback puis **restaure le
champion qui a gagné la MAE**. Captures : `docs/captures/`. Avant captures Grafana :
`DELETE FROM predictions WHERE source = 'loadtest';` (Postgres :5437).

## Règles métier (ne pas casser)

- Config YAML + `.env`, pas de secrets dans Git.
- Qualité **bloquante** avant train ; pas de retry Prefect sur erreur de données.
- Top 50 figé sur `warmup_weeks` uniquement.
- SHA256 UCI figé dès le premier téléchargement réussi.
- Chemins relatifs à la **racine du dépôt**.
- API : `model_version` sur chaque réponse ; champion local si registre down.
- Promotion : battre le naïf pour enregistrer ; MAE ≤ champion pour promouvoir.

**Registre** : alias `challenger` / `champion`, tag `previous_champion` (chaîne, pas
ping-pong). Export atomique vers `models/champion/`. Incident MLflow : `/health`
`degraded: true`, repli disque puis `.champion.bak`.

**API / SQL** : file bornée, `/predict` continue si Postgres est down (`db_ok: false`).
WAPE réelle au **rejeu** du lot, pas à la prédiction.

**Charge** : 2 workers / 1 vCPU / 1 Go tiennent **55,8 req/s** à p95 110 ms
(série 2). Détail : `docs/SYNTHESE_OPS.md`.

## Structure

```
configs/          config.yaml, data.yaml (SHA256)
src/retail/       data, quality, features, training, registry, orchestration, api, demo
tests/            unit/, integration/
flows/            weekly_lot (Prefect)
monitoring/       Prometheus + Grafana provisionnés
load_tests/       Locust + run.py
docs/             DATA.md, PLAN.md, SYNTHESE_OPS.md
```

## Choix techniques

| Besoin | Choix | Pourquoi |
|---|---|---|
| Config | YAML + `.env` | règles en revue, secrets hors Git |
| Qualité | Pandera, 2 schémas | porte bloquante, rapport JSON |
| Modèle | sklearn Ridge vs naïf | baseline honnête, MAE + WAPE |
| Registre | MLflow Docker :5001 | évite les chemins Windows de `mlruns/` |
| Orch. | Prefect, 1 lot / tick | retry technique seulement |
| Serving | FastAPI + champion disque | mode dégradé |
| Traces | PostgreSQL | Grafana SQL, WAPE par version |
| Observabilité | Prometheus + Grafana | RPS, latences, dégradé, WAPE |
| Charge | Locust + cpus Docker | reco chiffrée |
