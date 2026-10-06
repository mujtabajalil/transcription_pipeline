# Targets carry a `## description` comment; `make help` lists them.
.DEFAULT_GOAL := help
SHELL := bash

COMPOSE := docker compose
INFRA_SERVICES := postgres redis s3
KEY_NAME ?= dev

# Host processes (make api / worker / migrate / key) against the `make infra` containers.
HOST_ENV := \
	TX_ENV=dev \
	TX_LOG_JSON=false \
	TX_DATABASE_URL=postgresql+psycopg://tx:tx@localhost:5432/tx \
	TX_REDIS_URL=redis://localhost:6379/0 \
	TX_S3_BUCKET=transcription-audio \
	TX_S3_ENDPOINT_URL=http://localhost:5050 \
	AWS_ACCESS_KEY_ID=test \
	AWS_SECRET_ACCESS_KEY=test \
	AWS_DEFAULT_REGION=us-east-1 \
	TX_BOOTSTRAP_API_KEY=tx_dev_local_only_key \
	TX_WEBHOOK_ALLOW_PRIVATE_TARGETS=true

.PHONY: help install infra up down logs ps migrate api worker key test test-integration \
	test-all e2e lint fmt typecheck eval demo clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create the venv with dev dependencies (uv sync)
	uv sync --frozen

infra: ## Start Postgres, Redis and S3 (moto) only, for tests and host processes
	$(COMPOSE) up -d --wait $(INFRA_SERVICES)

up: ## Build the image and start everything (migrate, api :8000, worker)
	$(COMPOSE) up -d --build

down: ## Stop all containers (volumes are kept)
	$(COMPOSE) down

logs: ## Follow api and worker logs
	$(COMPOSE) logs -f --tail=100 api worker

ps: ## Show container status
	$(COMPOSE) ps

migrate: ## Apply migrations and create the bucket against `make infra`
	$(HOST_ENV) uv run tx-admin migrate
	$(HOST_ENV) uv run tx-admin init-storage

api: ## Run the API on the host with auto-reload (needs `make infra migrate`)
	$(HOST_ENV) uv run uvicorn transcription.api.app:create_app --factory --reload --reload-dir src --port 8000

worker: ## Run a worker on the host (needs `make infra migrate`)
	$(HOST_ENV) uv run tx-worker

key: ## Create an API key on the host DB (KEY_NAME=dev)
	$(HOST_ENV) uv run tx-admin create-key --name "$(KEY_NAME)"

test: ## Unit tests (ffmpeg only)
	uv run pytest

test-integration: ## Integration tests (needs `make infra`)
	uv run pytest -m integration

test-all: ## Unit + integration tests (needs `make infra`)
	uv run pytest -m "not e2e"

e2e: ## End-to-end tests with a real Whisper model
	uv run pytest -m e2e

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-fix lint findings and format
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy --strict over src/
	uv run mypy

eval: ## Word error rate on the bundled samples (eval/)
	uv run python -m eval.run

demo: ## Walk the HTTP API end to end against `make up` (scripts/demo.sh)
	./scripts/demo.sh

clean: ## Remove caches and build output (keeps .venv and docker volumes)
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
