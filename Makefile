# Svelte frontend + twin-api + worker (programme #262). Linux, macOS and the Jetson; on Windows use
# `python scripts/dev.py` and the pnpm / pytest commands directly.
PY ?= uv run python
PNPM ?= pnpm --dir frontend
RUFF ?= uvx ruff@0.17.2
# Files under the programme's Ruff rules (the rest of the repository keeps black + isort).
RUFF_PATHS := src/api src/services src/worker src/utils/memo.py src/ui/coke_energy/loaders.py \
	src/ui/burden_optimiser/loaders.py scripts/dev.py tests/services_fakes.py tests/test_services_*.py \
	tests/test_utils_memo.py tests/test_api_*.py scripts/export_openapi.py

.PHONY: help dev api worker legacy web install build check lint format test test-web test-py client

help:
	@echo "make dev      SPA (5173) + API (8100) + worker + Streamlit (/legacy, 8501)"
	@echo "make install  uv sync + pnpm install"
	@echo "make build    static SPA into frontend/build"
	@echo "make check    svelte-check (types)"
	@echo "make lint     prettier + eslint + ruff (new packages)"
	@echo "make test     vitest + pytest (BURDEN_ENGINE_FAST=1)"
	@echo "make client   export the OpenAPI schema and regenerate the typed TS client"

dev:
	$(PY) scripts/dev.py

api:
	$(PY) scripts/dev.py api

worker:
	$(PY) scripts/dev.py worker

legacy:
	$(PY) scripts/dev.py legacy

web:
	$(PNPM) dev

install:
	uv sync
	$(PNPM) install

build:
	$(PNPM) build

check:
	$(PNPM) check

lint:
	$(PNPM) lint
	$(RUFF) check $(RUFF_PATHS)
	$(RUFF) format --check $(RUFF_PATHS)

format:
	$(PNPM) format
	$(RUFF) format $(RUFF_PATHS)

test: test-web test-py

test-web:
	$(PNPM) test

test-py:
	BURDEN_ENGINE_FAST=1 uv run pytest -q tests

client:
	$(PY) scripts/export_openapi.py
	$(PNPM) api:types
