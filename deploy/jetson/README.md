# Jetson AGX Orin deployment

Production host for the Svelte frontend, the twin-api, the twin-worker and Streamlit (`/legacy`), programme #262.

| Issue | Content |
|---|---|
| #279 (SV-0.10) | `Caddyfile`: static SPA with `200.html` fallback, `/api` → twin-api, `/legacy` → Streamlit (this file, below) |
| #280 (SV-0.11) | CI release artefacts (`frontend-build.tar.gz`, lock files) and the aarch64 lock check (below) |
| #315 (SV-6.1) | Base system: JetPack 6.x, NVMe at `/var/lib/bf-webapp`, nvpmodel MAXN, chrony, ufw |
| #316 (SV-6.2) | Python 3.12 via uv on aarch64, wheel audit, runtime benchmark |
| #317 (SV-6.3) | systemd units: caddy, twin-api, twin-worker, streamlit, redis, furnace-data-service |
| #318 (SV-6.4) | `release.sh <tag>`: CI artefact, `uv sync --frozen`, `alembic -n twin upgrade head`, smoke test, rollback |

## Reverse proxy (`Caddyfile`, SV-0.10)

Caddy 2.10 serves one HTTPS origin; only ports 80 and 443 are open (ufw, SV-6.1).

| Request | Target | Behaviour |
|---|---|---|
| `/_app/immutable/*` | `BF_SPA_ROOT` | `Cache-Control: public, max-age=31536000, immutable`; a missing file is a 404 |
| any other path | `BF_SPA_ROOT` | the file, else `200.html` (client-side route, so a deep link survives a reload); `Cache-Control: no-cache` |
| `/api/*` | twin-api `127.0.0.1:8100` | `flush_interval -1`: server-sent events of `/api/stream` leave at once; `/api/stream` is not compressed |
| `/legacy`, `/legacy/*` | Streamlit `127.0.0.1:8501` | prefix kept (Streamlit runs with `--server.baseUrlPath legacy`); WebSocket `/legacy/_stcore/stream` |
| `http://…` | | 308 to `https://` with path and query |

Response headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`,
`Referrer-Policy: strict-origin-when-cross-origin`, `Strict-Transport-Security` from `BF_HSTS`; the `Server` header is
removed. Responses other than the event stream are compressed (zstd, gzip).

| Variable | Default | Meaning |
|---|---|---|
| `BF_SITE` | `bf02.local` | Host name of the console |
| `BF_TLS` | `internal` | `internal`: Caddy's own CA signs the certificate. Or `"<cert.pem> <key.pem>"`: certificate of the plant CA |
| `BF_HSTS` | `max-age=0` | Set `max-age=31536000` only once every client trusts the certificate: with HSTS a browser offers no way past a certificate warning |
| `BF_SPA_ROOT` | `/var/lib/bf-webapp/current/frontend/build` | Frontend build of the current release |
| `BF_API_UPSTREAM`, `BF_LEGACY_UPSTREAM` | `127.0.0.1:8100`, `127.0.0.1:8501` | twin-api and Streamlit |
| `BF_HTTP_PORT`, `BF_HTTPS_PORT` | `80`, `443` | Listening ports. Redirects assume the clients use 443 |

Processes behind the proxy (units in SV-6.3), all on 127.0.0.1:

```bash
# twin-api: ONE process (no --workers). It holds the sign-in failure limit and the revoked sessions.
uv run uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8100
# Streamlit under /legacy
uv run python run_streamlit.py --server.address 127.0.0.1 --server.port 8501 --server.baseUrlPath legacy \
    --server.headless true --browser.gatherUsageStats false
```

The twin-api trusts `X-Forwarded-For` from `BF_API_TRUSTED_PROXIES` (default 127.0.0.1, ::1, i.e. Caddy) and sets its
session cookie `Secure` (`BF_API_COOKIE_SECURE=true`), which needs the HTTPS origin.

**Certificate.** With `BF_TLS=internal`, Caddy creates a local CA under its data directory (for the caddy user
normally `/var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt`). Install that root certificate on every
control-room PC (Windows: Trusted Root Certification Authorities). The Caddyfile sets `skip_install_trust`: Caddy
does not add its root to the Jetson's own store.
With a certificate of the plant CA, set `BF_TLS="/etc/bf-webapp/tls/bf02.crt /etc/bf-webapp/tls/bf02.key"` (readable
by the caddy user). The choice is an open item of the plan (TLS certificate source and Jetson host name).

**Check.**

```bash
caddy validate --config deploy/jetson/Caddyfile --adapter caddyfile
BF_CADDY_BIN=$(command -v caddy) uv run pytest tests/test_caddyfile.py   # routes, upstreams, fallback, TLS, redirect
```

Verified on 27 Oct 2026 with Caddy 2.10.2 in front of the twin-api, the SPA build and Streamlit 1.56 (container,
ports 80 / 443, `BF_SITE=localhost`):

| Check | Result |
|---|---|
| HTTP to HTTPS | `http://localhost/trace?x=1` → 308 `https://localhost/trace?x=1`; same for `/legacy/` and `/api/health` |
| Deep links | Signed out, `/trace` → `/login?next=%2Ftrace` → after sign-in `/trace`; `/admin` and `/strata` stay on their page after a reload; `/admin/jobs` → 200 `200.html` |
| Assets | `/_app/immutable/...` 200 with the one-year cache; a missing asset 404; `/_app/version.json` `no-cache` |
| Event stream | 20 of 20 events through Caddy, latency from publish to the browser 3 ms (p50), 4 ms (max) |
| Streamlit | All 11 pages under `/legacy` run to the end with the WebSocket `/legacy/_stcore/stream` connected and the same content as Streamlit without the proxy |

## Release artefacts (CI, SV-0.11)

A tag `v*` runs `.github/workflows/release.yml`: the full CI suite, then the GitHub release of the tag with

| File | Content |
|---|---|
| `frontend-build.tar.gz` | `build/` of the SPA (entries sorted, owner 0, commit time as mtime); unpack into `<release>/frontend/`, giving `BF_SPA_ROOT` |
| `uv.lock`, `pnpm-lock.yaml` | Lock files of the tagged commit (`uv sync --frozen` on the Jetson) |
| `SHA256SUMS` | SHA-256 of the three files; `sha256sum --check SHA256SUMS` before unpacking |

`release.sh` (SV-6.4, #318) consumes these files.

## Python packages on aarch64

`make lock-check` (CI job `static`) resolves `uv.lock` for `aarch64-manylinux_2_35` (JetPack 6: Ubuntu 22.04,
glibc 2.35) and CPython 3.12 from wheels, with two exceptions that publish no aarch64 wheel and build from source
during `uv sync`: `pypika` (pure Python) and `quickjs` (C extension; needs `build-essential`). The check covers wheel
availability only: torch 2.2.2 from PyPI is the CPU build on aarch64, and its use of the Jetson GPU (NVIDIA's
JetPack wheel) belongs to the wheel audit and runtime benchmark of SV-6.2 (#316).
