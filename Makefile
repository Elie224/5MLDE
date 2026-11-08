# Raccourcis GNU Make. Sous Windows, `python -m retail` suffit sans Make.
PY ?= python
CLI := $(PY) -m retail

UNTIL_WEEK ?=
WEEK ?=
VERSION ?=
WORKERS ?= 1

.PHONY: help install lock lint format test test-unit check-config \
	data download batches quality quality-invalid features train rollback replay \
	api stack mlflow postgres db-init monitor load demo clean-data

help:
	@echo "install          pip lock + paquet editable"
	@echo "lock             regenerer requirements.lock (uv)"
	@echo "lint / format    ruff"
	@echo "test             pytest (test-unit : tests/unit seulement)"
	@echo "check-config     YAML"
	@echo "data             UCI + lots ISO"
	@echo "quality          Pandera (WEEK=2011-W40 optionnel)"
	@echo "quality-invalid  lot corrompu"
	@echo "train            Ridge vs naif (UNTIL_WEEK=2011-W40)"
	@echo "rollback         VERSION=n optionnel"
	@echo "replay           un lot (WEEK=... optionnel)"
	@echo "api              FastAPI (WORKERS=1)"
	@echo "stack            Postgres + MLflow + Prom + Grafana"
	@echo "db-init          tables SQL"
	@echo "load             Locust (arreter l'API hote avant)"
	@echo "demo             retail demo (RESET=1 pour --reset-cursor)"
	@echo "clean-data       supprime data/"

install:
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install -e . --no-deps

lock:
	uv pip compile pyproject.toml --extra dev --extra orch --extra load --universal --python-version 3.11 -o requirements.lock

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

test:
	$(PY) -m pytest --cov --cov-report=term-missing

test-unit:
	$(PY) -m pytest tests/unit

check-config:
	$(CLI) check-config

data:
	$(CLI) data

download:
	$(CLI) download

batches:
	$(CLI) batches

quality:
	$(CLI) quality $(if $(WEEK),--week $(WEEK),)

quality-invalid:
	$(CLI) quality --invalid-demo

features:
	$(CLI) features $(if $(UNTIL_WEEK),--until-week $(UNTIL_WEEK),)

train:
	$(CLI) train $(if $(UNTIL_WEEK),--until-week $(UNTIL_WEEK),)

rollback:
	$(CLI) rollback $(if $(VERSION),--to-version $(VERSION),)

replay:
	$(CLI) replay $(if $(WEEK),--week $(WEEK),)

api:
	$(CLI) api --workers $(WORKERS)

stack:
	docker compose up -d postgres mlflow prometheus grafana

mlflow:
	docker compose up -d mlflow

postgres:
	docker compose up -d postgres

db-init:
	$(CLI) db-init

monitor:
	docker compose up -d postgres prometheus grafana

load:
	$(PY) load_tests/run.py

demo:
	$(CLI) demo $(if $(RESET),--reset-cursor,)

clean-data:
	$(PY) -c "import shutil; shutil.rmtree('data', ignore_errors=True)"
