# Synthèse ops (dimensionnement)

Mesures Locust : mélange 70 / 20 / 10 (`/predict`, `/predict/batch`, `/health/live`),
paliers 20→320 users × 45 s. L'API Docker sert `models/champion/` (sonde MLflow coupée :
chemins fichier Windows inutilisables dans le conteneur). CSV : `load_tests/results/`
(`--csv-full-history`, chiffres **par palier** et **par endpoint**). Relancer :
`python load_tests/run.py`.

**Limite du banc** : Locust, Postgres, Prometheus, Grafana et l'API tournent **sur le
même poste**. Locust (un cœur) ou la file SQL (`retail_db_dropped_total`) peuvent
saturer avant l'API. `docker stats` CPU% = **part d'un cœur** (100 % = 1 vCPU).

## Série 1 : avant optimisation (7 nov. 2026, 21 h 30 → 21 h 42)

Run `python load_tests/run.py`, fichier `load_tests/results/summary.json`. Les trois
configurations ont bien été servies par le conteneur (`source: disk`, `degraded: false`,
Postgres), 0 échec Locust. Locust n'a pas de délai d'expiration : les requêtes lentes
attendent au lieu d'échouer.

### Débit par palier (req/s, calculé sur la fenêtre de 45 s)

| Config | 20 users | 40 users | 80 users | 160 users | 320 users | **Débit max** |
|---|---|---|---|---|---|---|
| 1w / 0,5 vCPU | 0,96 ¹ | 9,53 | 8,44 | 9,44 | 7,93 | **≈ 9,5** |
| 2w / 1 vCPU | 12,56 | 19,44 | 19,49 | 16,51 | 15,56 | **≈ 19,5** |
| 4w / 2 vCPU | 26,76 | 34,36 | 38,84 | 36,29 | 36,16 | **≈ 38,8** |

¹ Anomalie du premier palier (43 requêtes, p95 40 s) : probablement le démarrage à froid ;
exclu de l'analyse, cause non vérifiée.

### p95 par palier (ms)

| Config | 20 users | 40 users | 80 users | 160 users | 320 users |
|---|---|---|---|---|---|
| 1w / 0,5 vCPU | 40 000 ¹ | 13 000 | 17 000 | 27 000 | 32 000 |
| 2w / 1 vCPU | 1 200 | 2 800 | 5 700 | 16 000 | 25 000 |
| 4w / 2 vCPU | 890 | 1 500 | 2 500 | 6 300 | 16 000 |

### Ressources (`docker stats`, plateau)

| Config | CPU API (% d'un cœur) | RAM API | Lignes `predictions` attendues | Lignes écrites en base |
|---|---|---|---|---|
| 1w / 0,5 vCPU | 48–53 % (limite atteinte) | ≈ 500 Mo | 18 441 | |
| 2w / 1 vCPU | 95–105 % (limite atteinte) | ≈ 690 Mo | 41 865 | |
| 4w / 2 vCPU | 175–214 % (limite atteinte) | ≈ 1,17 Go | 82 308 | |
| **Total** | | | **142 614** | **29 749 (≈ 21 %)** |

Lignes attendues = `/predict` + 50 × `/predict/batch` (Locust). Lignes écrites = comptage
Postgres entre 21 h 30 et 21 h 42.

### Lecture

1. **L'API est limitée par le CPU, et le débit croît linéairement avec les vCPU** :
   9,5 → 19,5 → 38,8 req/s, soit **≈ 19,4 req/s par vCPU** (mélange 70/20/10), donc
   **≈ 51 ms de CPU par requête**. Le CPU est à la limite dans les trois cas.
2. **Aucune configuration ne respecte p95 < 200 ms**, même au palier le plus bas (20 users) :
   la capacité au sens du critère est `null` partout. Les latences suivent la loi de
   Little (20 users / 26,8 req/s ≈ 0,75 s de cycle pour 4w) : c'est de la file d'attente,
   pas des erreurs. Les paliers commencent trop haut pour mesurer la capacité sous 200 ms.
3. **Goulot identifié** : à chaque requête, l'API liste `data/interim/panel` et lit la
   date de ses 51 fichiers (contrôle d'horizon et clé de cache), deux fois. Dans le
   conteneur, ce dossier est monté depuis Windows : mesuré à **≈ 400 ms** pour une
   lecture (mesure prise sous charge, donc majorée).
4. **≈ 79 % des traces de prédiction perdues** sous charge (29 749 lignes écrites sur
   142 614 attendues), alors que `retail_db_dropped_total` affiche 0 : **le compteur
   d'abandons ne fonctionne pas** et l'écrivain SQL ne suit pas.
5. **Mémoire** : ≈ 0,25 Go par worker supplémentaire (0,69 Go → 1,17 Go entre 2 et 4 workers).
6. **Multi-workers** : avec CPU limité, les workers mettaient plus de 5 s à démarrer et
   Uvicorn les tuait en boucle (`timeout_worker_healthcheck = 5` par défaut). Corrigé
   (délai relevé, imports allégés).

### Dimensionnement provisoire (avant optimisation)

Sur cette série, le seul résultat fiable est le **débit par vCPU (≈ 19 req/s)**. Pour
garder une latence raisonnable, on vise un CPU à 70 % au plus :

> vCPU nécessaires ≈ demande (req/s) / (19 × 0,7) ≈ demande / 13,5
>
> Exemple : 20 req/s en pointe → 2 vCPU → **4 workers / 2 vCPU / 1,5 Go**.

Ce chiffre n'atteint pas p95 < 200 ms : il sera remplacé par la série 2.

### Limites connues de cette série

- Banc sur un seul poste (Locust, Postgres, Prometheus, Grafana, API).
- CPU de Locust non mesuré (`locust_cpu_pct` toujours à 0, bug `psutil`).
- Le bloc `endpoints` de `summary.json` reprend la **dernière fenêtre** et non le total :
  utiliser `*_stats.csv` pour les chiffres par endpoint.

## Série 2 : après optimisation (7 nov. 2026, 22 h 05 → 22 h 25)

Corrections appliquées : dossier panel relu au plus toutes les 5 s, `/predict/batch` en
une seule insertion SQL, compteur d'abandons réparé, paliers 2 → 320 users. Les trois
configurations ont été servies par le conteneur (`source: disk`, `degraded: false`,
Postgres), 0 échec Locust.

### Capacité (dernier palier avec p95 < 200 ms et 0 % d'erreurs)

| Config | Palier | **Capacité** | p95 | RAM API | Débit max (saturation) |
|---|---|---|---|---|---|
| 1w / 0,5 vCPU | 5 users | **29,8 req/s** | 120 ms | ≈ 305 Mo | 32,4 req/s (10 users, p95 310 ms) |
| 2w / 1 vCPU | 10 users | **55,8 req/s** | 110 ms | ≈ 550 Mo | 55,8 req/s (10 users) |
| 4w / 2 vCPU | 10 users | **≥ 60,2 req/s** ¹ | 60 ms | ≈ 1,05 Go | 81,6 req/s (20 users, p95 360 ms) |

¹ Pour 4w, le palier de 10 users n'est pas saturé (p95 60 ms, CPU ≈ 100 % sur 200 %) et
le suivant (20 users) dépasse 200 ms : la capacité réelle est **entre 60 et 82 req/s**.
Il faudrait un palier intermédiaire (15 users) pour la préciser.

### Débit et p95 par palier

| Config | 2 | 5 | 10 | 20 | 40 | 80 | 160 | 320 users |
|---|---|---|---|---|---|---|---|---|
| 1w req/s | 12,4 | 29,8 | 32,4 | 29,3 | 23,4 | 21,7 | 22,3 | 21,5 |
| 1w p95 (ms) | 51 | 120 | 310 | 1 500 | 3 800 | 6 000 | 6 800 | 17 000 |
| 2w req/s | 12,9 | 31,0 | 55,8 | 49,9 | 40,0 | 33,3 | 33,5 | 34,7 |
| 2w p95 (ms) | 27 | 49 | 110 | 720 | 1 900 | 4 100 | 9 800 | 16 000 |
| 4w req/s | 12,3 | 31,3 | 60,2 | 81,6 | 77,0 | 80,8 | 68,4 | 58,0 |
| 4w p95 (ms) | 34 | 49 | 60 | 360 | 1 200 | 2 100 | 4 900 | 11 000 |

### Avant / après optimisation

| Config | Débit max série 1 | Débit max série 2 | Gain |
|---|---|---|---|
| 1w / 0,5 vCPU | 9,5 req/s | 32,4 req/s | × 3,4 |
| 2w / 1 vCPU | 19,5 req/s | 55,8 req/s | × 2,9 |
| 4w / 2 vCPU | 38,8 req/s | 81,6 req/s | × 2,1 |
| Traces écrites en base | 29 749 / 142 614 (≈ 21 %) | 474 596 / 468 973 ² (≈ 100 %) | perte supprimée |

² Lignes `predictions` comptées dans Postgres entre 22 h 05 et 22 h 26, comparées à
`/predict` + 50 × `/predict/batch` côté Locust. Le léger excédent vient probablement des
requêtes terminées côté serveur après l'arrêt de Locust (non vérifié).

### Lecture

1. **L'optimisation du dossier panel a triplé le débit** à ressources égales : le goulot
   identifié en série 1 (lecture de fichiers sur un dossier monté depuis Windows) était
   bien le facteur principal.
2. **Le passage à l'échelle n'est plus linéaire** : 65 req/s par vCPU en 1w, 56 en 2w,
   41 en 4w (débit max ÷ vCPU). Au-delà de 2 vCPU, une autre limite apparaît. Hypothèses
   non vérifiées : Locust (un seul cœur, sur le même poste), l'écriture SQL (≈ 230 000
   lignes en 6 min pour 4w), ou le poste lui-même.
3. **Au-delà de la saturation, le débit baisse** (1w : 32 → 21 req/s ; 2w : 56 → 33 ;
   4w : 82 → 58) et les latences explosent par file d'attente. Il faut donc protéger
   l'API contre la surcharge (limite de concurrence ou délai d'expiration côté client).
4. **Mémoire** : ≈ 250 Mo par worker (305 Mo → 550 Mo → 1,05 Go).
5. **Traçabilité** : plus aucune trace perdue sous charge après le passage à une insertion
   par lot.

### Recommandation de dimensionnement

**Unité recommandée : 2 workers / 1 vCPU / 768 Mo à 1 Go.** Elle tient **55,8 req/s avec
p95 = 110 ms et 0 % d'erreurs**, et c'est la meilleure capacité garantie par vCPU mesurée
(4w / 2 vCPU n'apporte qu'entre 60 et 82 req/s pour le double de CPU).

En gardant 30 % de marge sous la capacité mesurée (≈ 39 req/s par instance) :

> instances = ceil(pointe en req/s / 39)
>
> Exemples : 30 req/s → **1 instance** ; 100 req/s → **3 instances** ;
> 200 req/s → **6 instances** de 2 workers / 1 vCPU, derrière un répartiteur de charge.

Pour une démonstration sur un poste, 1 worker / 0,5 vCPU suffit jusqu'à environ 30 req/s.

### Limites connues de la série 2

- **Le délai d'expiration Locust n'a pas été appliqué** : `self.client.timeout = 5` n'a
  aucun effet, car le client HTTP de Locust ne prend le délai qu'en argument de chaque
  requête (`self.client.post(..., timeout=5)`). D'où des p95 de 17 s avec 0 erreur après
  saturation. La capacité n'est pas touchée (elle exige p95 < 200 ms), mais la colonne
  « erreurs » au-delà de la saturation l'est.
- **CPU de Locust toujours à 0** : on ne peut pas exclure que Locust plafonne en 4w.
- Banc sur un seul poste ; granularité des paliers trop grossière entre 10 et 20 users
  pour 4w.

## Cache des variables

Les features sont mises en cache par `(semaine, version du champion, mtime du panel)`.
Un premier `/predict/batch` sur une semaine froide paie le calcul panel ; les appels
suivants (Locust sur une semaine fixe) restent sur le chemin court. Comparer dans
Grafana le p95 de `/predict` en début de rampe vs plateau.

## WAPE (rappel démo)

Le feedback auto n'existe qu'à la **livraison du lot**. Ordre : replay jusqu'à N,
prédire N+1, rejouer N+1. `retail demo` enchaîne cet ordre.

## Décisions techniques

| Brique | Choix | Justification |
|---|---|---|
| Qualité | **Pandera**, `InvoiceSchema` puis `PanelSchema` | Contrôles **bloquants** avant tout train ; rapport JSON par lot ; le lot corrompu de démo ne déclenche pas d'entraînement. |
| Entraînement | Ridge sklearn vs **naïf** (lag 1), MAE + WAPE, split temporel | Le naïf est la barre d'enregistrement ; anti-fuite (top 50 sur `warmup_weeks` seulement). Tag `data_hash` = archive UCI **entière** ; la tranche est `until_week` + `features_hash`. |
| Registre | **MLflow** (serveur Docker :5001), alias `challenger` / `champion` | Promotion si MAE ≤ champion **sur le même test** ; rollback via tag `previous_champion` ; export `models/champion/` pour le mode dégradé. |
| Orchestration | **Prefect** `weekly_lot`, un lot ISO par tick | Retries sur panne technique uniquement ; échec qualité = arrêt + alerte, pas de retry métier. |
| Serving | **FastAPI**, `model_version` partout | Champion registre, repli disque / `.champion.bak` ; `/health` expose `source`, `degraded`, `db_ok`. |
| Traces | **PostgreSQL** (:5437), file bornée | Grafana lit la même base que l'API ; `/predict` survit à une coupure SQL. |
| Supervision | **Prometheus** + **Grafana** provisionnés | RPS, p50/p95/p99, 5xx, version, WAPE, runs, `ALERTS`. |
| Charge | **Locust** + `cpus` / `mem_limit` Docker | Même protocole pour 1 / 2 / 4 workers ; reco : 2w / 1 vCPU / 1 Go. |

Le trafic Locust envoie l'en-tête `X-Load-Test` ; l'API enregistre `source=loadtest`,
exclu du feedback auto et de la WAPE de production.

Config : YAML versionné + `.env` (ports 5437, 5001, 4201, 8001, 3001, 9091). Pas de notebook comme chemin principal.

## Preuves d'intégration

Commande unique : `retail demo` (ou `make demo`), API déjà levée.

| Preuve | Où |
|---|---|
| Santé + Postgres | sortie étape 0 : `source`, `model_version`, `db_backend=postgresql` |
| Nominal + WAPE | étape 1 : N = curseur, `POST /predict/batch` N+1, rejeu N+1, `GET /model` (`production_wape`) |
| Lot invalide | `data/reports/invalid_demo/2099-W01.json` (`status: failed`), `GET /alerts` |
| Postgres down | `/predict` 200, `db_ok: false`, jauge `retail_db_ok=0` **> 15 s**, `GET Prometheus /api/v1/alerts` : `RetailDbDown` firing |
| Rollback | `model_version` change, `models/champion/meta.json` ; restauré en `finally` |
| Grafana | http://127.0.0.1:3001 dashboard `retail-mlops` (WAPE après rejeu N+1, alerte étape 3) |
| Prometheus | http://127.0.0.1:9091/alerts (`RetailDegraded`, `RetailHigh5xx`, `RetailApiDown`, `RetailDbDown`) |
| MLflow | http://127.0.0.1:5001 alias `champion` / `challenger` |
| Prefect | http://127.0.0.1:4201 runs `weekly_lot` |
| Charge | `load_tests/results/summary.json` + séries 1 et 2 ci-dessus |

Joindre au livrable des captures Grafana, MLflow (versions) et Prefect (un run OK / un KO qualité) dans `docs/captures/` si le jury les exige.

## Incidents rencontrés et corrigés

1. **Chemins Windows dans `mlruns/`** : `meta.yaml` du registre fichier est inutilisable après déplacement ou dans Docker. Correctif : serveur MLflow :5001 + champion exporté sur disque ; l'API de charge saute la sonde registre (`RETAIL_SKIP_REGISTRY_PROBE`).
2. **Workers Uvicorn tués** : ping santé 5 s dépassé pendant `spawn` (réimport du CLI). Correctif : `python -m retail.api.serve`, `timeout_worker_healthcheck=60`, imports CLI paresseux.
3. **Goulot `data/interim/panel`** : listing NTFS monté dans le conteneur ≈ 400 ms / requête. Correctif : cache 5 s → débit × 2 à × 3 (série 2).
4. **Traces SQL perdues (79 %)** avec compteur à 0 : flush en échec non compté, inserts 1 à 1. Correctif : insertion par lot + drop sur file pleine **et** flush KO. Série 2 : ~100 % des lignes en base.
5. **Grafana SQL vide** : datasource sans `database` / API encore sur SQLite. Correctif : `${POSTGRES_DB}` + `RETAIL_DATABASE_URL` Postgres, `db_backend` dans `/health`.
6. **Locust `self.client.timeout`** ignoré en 2.47 (p95 17 s, 0 erreur). Correctif pour une série 3 : `timeout=` sur chaque requête. La capacité série 2 (p95 < 200 ms) n'en dépend pas.
7. **`RetailDbDown` absente / non firing** : règles non rechargées, Postgres coupé ~4 s (`for: 30s`). Correctif : `for: 15s`, reload Prometheus dans `retail demo`, attente jusqu'à `firing` via `/api/v1/alerts`.
8. **Démo sur une semaine déjà livrée** (W41 alors que curseur = W41). Correctif : N = curseur, N+1 = semaine suivante ; `retail demo --reset-cursor` avant soutenance.
9. **`/metrics` vide dans l'API Docker** : `PROMETHEUS_MULTIPROC_DIR` sur le volume Windows `./data`. Correctif : `/tmp/prom` en `tmpfs` (hors NTFS).
10. **~500 k prédictions Locust W48** enregistrées `source=disk` (avant `X-Load-Test`). Correctif : `ensure_schema` / `retail db-init` les réétiquette `loadtest` (exclues du WAPE).
11. **MLflow Docker vide** (8 nov.) : `docker start retail-mlflow` alors que le conteneur n'existait plus ; image `v2.22.5` absente de GHCR. Correctif : `ghcr.io/mlflow/mlflow:v2.22.2`, `docker compose up -d mlflow`, `RETAIL_MLFLOW_TRACKING_URI=http://127.0.0.1:5001` dans `.env` (train + API + Prefect), réentraîner W38 puis W40.
12. **Port 8001 partagé** : conteneur `retail-api` (Exited / relancé) et `retail api` hôte. Prometheus scrapait le mauvais process. Correctif : `docker compose --profile load stop api` avant l'API Windows. Charge : `--force-recreate api` pour le tmpfs `/tmp/prom`.
13. **API arrêtée ~01:05–01:30** (WinError 10048 / Docker down) : `RetailApiDown`, timeouts SQL, Locust encore actif. Correctif ops : un seul Uvicorn, Docker Desktop allumé, **arrêter Locust** avant la démo.
14. **Champion v5 aliasé malgré une MAE pire que v1** : `prepare_rollback_chain` forçait le dernier challenger et le `finally` le restaurait. Correctif : rollback vers le champion MAE-valide avant la scène, restauration de **ce** champion.
15. **Prefect `ConnectError` / prompt de profil** : serveur quitté (retour au `PS>`), port 4201 tenu par Anaconda, worker trop tôt. Correctif README : `--host 127.0.0.1`, `PREFECT_API_URL=http://127.0.0.1:4201/api` (pas `localhost`).
16. **`/health/live`** : `db_ok: true` volontaire (liveness sans ping SQL). Le ready est `GET /health`. `/metrics` et `/health/live` exclus du middleware de latence.
17. **`pipeline_runs` / `alerts` vides après `retail replay`** : écriture SQL dans un thread `daemon` tué à la fin du process. Correctif : `persist_run` / `persist_alert` synchrones.

## Limites connues

- Tout le banc (API, Locust, Postgres, Prom, Grafana) est **sur un seul poste**.
- En 4 workers / 2 vCPU, le passage à l'échelle se tasse (41 req/s/vCPU vs 56 en 2w) : Locust, SQL ou le poste, **non départagé**.
- Capacité 4w entre 60 et 82 req/s (pas de palier à 15 users).
- CPU Locust série 2 restait à 0 (`cpu_percent` recréé) ; corrigé par delta `cpu_times`, non remesuré.
- WAPE Grafana vide tant que le lot N+1 n'est pas rejoué.
- Prefect n'est pas embarqué dans `retail demo` (UI à lancer à part).
- Reproductibilité clone : `retail data` dépend d'UCI (repli fichier local) ; le SHA256 dans `configs/data.yaml` fige l'archive. La séquence README « Depuis un clone » n'a pas été rejouée ici sur un dépôt vide.
- La démo s'arrête avant W48 (`pick_demo_weeks`) : après ~6 lancements, utiliser `retail demo --reset-cursor`.
- Captures jury : `docs/captures/` (Grafana étape 3, MLflow, Prefect) à prendre sur la machine de démo.
- **Arrêter Locust avant `retail demo`**, sinon la coupure Postgres se lit « sous trafic ». Les écritures SQL perdues (~1 100–1 500 par coupure) sont **voulues** : l'API continue et incrémente `retail_db_dropped_total`.
- Timeouts Locust (côté client) : **jamais en 5xx**. Le signal est la latence p95/p99 (histogramme jusqu'à 10 s).
- Volume Grafana : `predictions` hors `loadtest` ; le reste Locust (~500 k) est une ligne à part. Purge optionnelle : `DELETE FROM predictions WHERE source = 'loadtest';`
- Latences Grafana / Prometheus : mesurées **après** prise en charge par Uvicorn (file d'attente TCP/WSGI exclue). Vu du client (Locust), la file peut afficher 1–3 s alors que le dashboard affiche ~40 ms. Tests sur un **PC partagé** (Docker Desktop, API, Locust, navigateur) : p95 > 200 ms à ~16 req/s / 4 workers n'est pas une fuite de cache.
- WAPE : un feedback par `(SKU, semaine, version)` (le plus récent). Relancer `retail demo --reset-cursor` ne multiplie plus le score. Une WAPE identique v3/v5 sur W41 peut venir du même Ridge / des mêmes ventes.
