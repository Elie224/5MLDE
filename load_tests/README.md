# Tests de charge (phase 9)

Mélange Locust : **70 %** `POST /predict`, **20 %** `POST /predict/batch`, **10 %** `GET /health/live`
(sans sonde MLflow). SKU : `GET /model`, sinon `models/champion/top_skus.json`. Semaine :
`LOCUST_WEEK` (défaut `2011-W48`).

L'API Docker sert le champion **disque** (`RETAIL_SKIP_REGISTRY_PROBE=1`, pas de chemins
Windows `mlruns/`). Vérifier `degraded: false` avant Locust.

Protocole **identique** pour chaque configuration : paliers **2 / 5 / 10 / 20 / 40 / 80 /
160 / 320** utilisateurs, **45 s** chacun (~6 min). Capacité = dernier palier avec
**p95 < 200 ms** et **0 % d'erreurs**. Timeout Locust : ``timeout=5`` **sur chaque
requête** (``self.client.timeout`` est ignoré par Locust 2.47).

```bash
python load_tests/run.py --smoke --only 1w_0p5cpu   # 30 s, une config
python load_tests/run.py                            # 3 configs
```

Configurations (API **dans Docker**, CPU/RAM plafonnés) :

| Nom | Workers | vCPU | RAM |
|---|---|---|---|
| `1w_0p5cpu` | 1 | 0,5 | 1 Go (512 Mo : OOM au chargement sklearn) |
| `2w_1cpu` | 2 | 1 | 1 Go |
| `4w_2cpu` | 4 | 2 | 2 Go |

```bash
# Arrêter l'API hôte (port 8001) avant, Postgres déjà levé
docker compose --profile load stop api   # si un vieux conteneur traîne
docker compose --profile load up -d --force-recreate api   # tmpfs /tmp/prom
python load_tests/run.py
# CSV + docker stats : load_tests/results/ (gitignoré)
# palier plat : LOCUST_CONSTANT=1
```

Sans Make sous Windows : la même commande. Avec Make : `make load`.
Grafana : http://127.0.0.1:3001/ pendant le run. Synthèse : `docs/SYNTHESE_OPS.md`.
