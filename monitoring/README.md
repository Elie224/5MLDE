# Supervision (phase 8)

Prometheus (port **9091**) interroge `http://host.docker.internal:8001/metrics`.
Grafana (port **3001**) est **provisionné** : sources Prometheus + PostgreSQL et le
tableau de bord `retail-mlops` (aucun clic manuel).

```bash
# API déjà sur :8001, Postgres déjà levé
docker compose up -d prometheus grafana
# ou : make monitor
```

Ouvrir http://127.0.0.1:3001 (compte `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`,
défaut `admin` / `admin`, lecture anonyme activée). Le tableau inclut un panneau
**Alertes Prometheus** (`ALERTS{alertstate="firing"}`) : pendant la démo, rester
sur Grafana plutôt que d'ouvrir `/alerts`. Alertes : mode dégradé > 1 min, 5xx > 5 %,
API injoignable 30 s.

Le scrape Docker → API hôte passe par `host.docker.internal` (Windows / Docker Desktop).
Les compteurs RPS / latence **excluent** `/metrics` et `/health`. Avec plusieurs
workers : `retail api --workers 2` (agrégation `PROMETHEUS_MULTIPROC_DIR` ; chaque
processus recharge l'alias `champion` toutes les 30 s). L'API **Docker** (profil
`load`) écrit les fichiers multiprocess dans **`/tmp/prom` (tmpfs)**, pas dans
`./data` monté depuis Windows — sinon `/metrics` reste vide et Grafana n'affiche
pas la charge.
