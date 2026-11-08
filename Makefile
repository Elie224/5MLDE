# Raccourcis : chaque cible appelle la CLI `retail` (utilisable seule sous Windows).
PY ?= python

.PHONY: help install lock lint format test test-unit check-config data download batches quality quality-invalid features train rollback replay api mlflow postgres db-init monitor load demo clean-data

help:
	@echo "install       environnement + dépendances verrouillées"
	@echo "lock          régénérer requirements.lock depuis pyproject.toml (uv)"
	@echo "lint / format qualité du code (ruff)"
	@echo "test          tests unitaires + intégration"
	@echo "check-config  valider configs/*.yaml"
	@echo "data          télécharger UCI, vérifier, convertir, découper en lots"
	@echo "quality       contrôles Pandera bloquants + nettoyage tracé"
	@echo "quality-invalid  lot corrompu (code 0 si rejeté, 1 si la porte le laisse passer)"
	@echo "features      top 50 figé sur warmup + lags (--until-week optionnel)"
	@echo "train         Ridge vs naïf, MLflow, registre ; --until-week pour le rejeu"
	@echo "rollback      ré-alias previous_champion (VERSION=n pour --to-version)"
	@echo "replay        un lot ISO (qualité --week, pas les 51 lots)"
	@echo "api           FastAPI sur le port 8001"
	@echo "mlflow        serveur MLflow Docker (port 5001)"
	@echo "postgres      PostgreSQL Docker (port 5437)"
	@echo "db-init       créer les tables (SQLite ou Postgres)"
	@echo "monitor       Prometheus (9091) + Grafana (3001)"
	@echo "load          Locust × 3 configs API Docker (8 paliers × 3 configs)"
	@echo "demo          nominal, lot invalide, Postgres down, rollback"
	@echo "clean-data    supprimer data/ (re-généré par make data)"

install:
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install -e . --no-deps
	pre-commit install || true

lock:
	uv pip compile pyproject.toml --extra dev --extra orch --extra load --universal --python-version 3.11 -o requirements.lock

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

test:
	$(PY) -m pytest --cov --cov-report=term-missing

test-unit:
	$(PY) -m pytest tests/unit

check-config:
	retail check-config

data:
	retail data

download:
	retail download

batches:
	retail batches

quality:
	retail quality

quality-invalid:
	retail quality --invalid-demo

features:
	retail features

train:
	retail train

rollback:
	retail rollback $(if $(VERSION),--to-version $(VERSION),)

replay:
	retail replay

WORKERS ?= 1
api:
	retail api --workers $(WORKERS)

mlflow:
	docker compose up -d mlflow

postgres:
	docker compose up -d postgres

db-init:
	retail db-init

monitor:
	docker compose up -d postgres prometheus grafana

load:
	$(PY) load_tests/run.py

demo:
	retail demo

clean-data:
	$(PY) -c "import shutil; shutil.rmtree('data', ignore_errors=True)"
