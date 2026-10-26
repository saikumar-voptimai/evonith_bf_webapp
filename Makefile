# Svelte frontend + twin-api + worker (programme #262). Linux, macOS and the Jetson; on Windows use
# `python scripts/dev.py` and the pnpm / pytest commands directly.
PY ?= uv run python
PNPM ?= pnpm --dir frontend
SRC_PATH := PYTHONPATH=src:furnace_data

.PHONY: help dev api worker legacy web install build check lint format test test-web test-py

help:
	@echo "make dev      SPA (5173) + API (8100) + worker + Streamlit (/legacy, 8501)"
	@echo "make install  uv sync + pnpm install"
	@echo "make build    static SPA into frontend/build"
	@echo "make check    svelte-check (types)"
	@echo "make lint     prettier + eslint + ruff (new packages)"
	@echo "make test     vitest + pytest (BURDEN_ENGINE_FAST=1)"

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
	uvx ruff@0.17.2 check src/api src/services src/worker scripts/dev.py
	uvx ruff@0.17.2 format --check src/api src/services src/worker scripts/dev.py

format:
	$(PNPM) format
	uvx ruff@0.17.2 format src/api src/services src/worker scripts/dev.py

test: test-web test-py

test-web:
	$(PNPM) test

test-py:
	BURDEN_ENGINE_FAST=1 uv run pytest -q tests
