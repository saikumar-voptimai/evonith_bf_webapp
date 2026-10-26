# Jetson AGX Orin deployment

Production host for the Svelte frontend, the twin-api, the twin-worker and Streamlit (`/legacy`), programme #262.

| Issue | Content |
|---|---|
| #279 (SV-0.10) | `Caddyfile`: static SPA with `200.html` fallback, `/api` → twin-api, `/legacy` → Streamlit |
| #315 (SV-6.1) | Base system: JetPack 6.x, NVMe at `/var/lib/bf-webapp`, nvpmodel MAXN, chrony, ufw |
| #316 (SV-6.2) | Python 3.12 via uv on aarch64, wheel audit, runtime benchmark |
| #317 (SV-6.3) | systemd units: caddy, twin-api, twin-worker, streamlit, redis, furnace-data-service |
| #318 (SV-6.4) | `release.sh <tag>`: CI artefact, `uv sync --frozen`, `alembic upgrade head`, smoke test, rollback |

Files are added here by those issues.
