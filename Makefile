# Svelte frontend + twin-api + worker (programme #262). Linux, macOS and the Jetson; on Windows use
# `python scripts/dev.py` and the pnpm / pytest commands directly.
PY ?= uv run python
PNPM ?= pnpm --dir frontend
RUFF ?= uvx ruff@0.17.2
# uv of the lock rule (CLAUDE.md): keeps uv.lock at revision 2. CI installs this version and sets UV_LOCK=uv.
UV_LOCK ?= uvx --from uv==0.7.13 uv
# Files under the programme's Ruff rules (the rest of the repository keeps black + isort).
RUFF_PATHS := src/api src/services src/worker src/utils/memo.py src/ui/coke_energy/loaders.py \
	src/ui/burden_optimiser/loaders.py scripts/dev.py tests/services_fakes.py tests/test_services_*.py \
	tests/test_utils_memo.py tests/test_api_*.py tests/test_events.py tests/test_jobs.py tests/redis_server.py \
	scripts/export_openapi.py scripts/export_chart_fixtures.py tests/test_chart_fixtures.py src/data/twin alembic/twin \
	tests/pg_server.py tests/test_twin_store.py scripts/export_legacy_pages.py tests/test_legacy_pages.py run_streamlit.py \
	tests/test_caddyfile.py tests/test_burden_optimiser_thermal.py tests/module_stubs.py src/utils/coke_energy/twin.py \
	tests/twin_data.py tests/test_twin_window.py src/utils/burden_engine/strata_window.py tests/test_twin_strata.py \
	scripts/twin_parity.py tests/test_twin_parity_script.py
# The Jetson (SV-6.1): JetPack 6 = Ubuntu 22.04, glibc 2.35, CPython 3.12 from uv. Every locked package needs a wheel
# for it, except these, which have none and install from their source archive (pypika: pure Python; quickjs:
# C extension, built with the Jetson's compiler).
JETSON_PLATFORM := aarch64-manylinux_2_35
JETSON_SOURCE_BUILDS := pypika quickjs

.PHONY: help dev api worker legacy web install build check lint lint-web lint-py format test test-web test-py client \
	fixtures db-upgrade db-downgrade db-current audit-ui lock-check

help:
	@echo "make dev      SPA (5173) + API (8100) + worker + Streamlit (/legacy, 8501)"
	@echo "make install  uv sync + pnpm install"
	@echo "make build    static SPA into frontend/build"
	@echo "make check    svelte-check (types)"
	@echo "make lint     prettier + eslint (lint-web) + ruff on the programme's Python (lint-py)"
	@echo "make test     vitest + pytest (BURDEN_ENGINE_FAST=1)"
	@echo "make client   export the OpenAPI schema and regenerate the typed TS client"
	@echo "make fixtures write the chart-kit parity fixtures (Rist diagram, schematic) and the /legacy page map"
	@echo "make db-upgrade / db-downgrade / db-current   twin schema migrations (BF_TWIN_DB_URL or DATABASE_URL)"
	@echo "make audit-ui build the SPA, then axe-core, layout and keyboard checks with screenshots (Playwright)"
	@echo "make lock-check  uv.lock matches pyproject.toml and installs on the Jetson (linux-aarch64)"

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

lint: lint-web lint-py

lint-web:
	$(PNPM) lint

lint-py:
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

fixtures:
	$(PY) scripts/export_chart_fixtures.py
	$(PY) scripts/export_legacy_pages.py

db-upgrade:
	uv run alembic -n twin upgrade head

db-downgrade:
	uv run alembic -n twin downgrade -1

db-current:
	uv run alembic -n twin current

audit-ui: build
	$(PNPM) audit:ui

lock-check:
	$(UV_LOCK) lock --check
	@set -e; req=$$(mktemp); trap 'rm -f "$$req"' EXIT; \
	$(UV_LOCK) export --frozen --no-hashes --no-emit-project --all-groups --quiet -o "$$req"; \
	$(UV_LOCK) pip compile "$$req" --no-deps --quiet --python-version 3.12 --python-platform $(JETSON_PLATFORM) \
		--only-binary :all: $(addprefix --no-binary ,$(JETSON_SOURCE_BUILDS)) > /dev/null; \
	echo "uv.lock installs on $(JETSON_PLATFORM) (source builds: $(JETSON_SOURCE_BUILDS))"
