# Jetson Orin Nano Backend With Hosted Streamlit

> For the live `/opt/evonith-bf` installation, private port 1432, Nginx,
> public-IP testing, logs, updates, backups, and incident handling, use the
> [Jetson backend operations runbook](../operations/jetson-backend-runbook.md).
> This deployment guide retains the preferred future domain-and-TLS topology.

## Recommended topology

Run only the FastAPI backend on the Jetson. Keep Uvicorn private on
`127.0.0.1:1432` and publish only HTTPS port `443` through Caddy. Configure the
hosted Streamlit process to call `https://api.example.com/api/v1`.

```text
Streamlit host -> HTTPS :443 -> router/NAT -> Caddy on Jetson
                                             -> 127.0.0.1:1432 FastAPI
                                             -> /var/lib/evonith-bf
```

Do not expose Uvicorn port `1432` directly to the internet. A raw public IP over
plain HTTP exposes login tokens and application data. Use a DNS hostname and a
valid TLS certificate. If inbound port forwarding is unavailable because of
CGNAT, use an authenticated tunnel or VPN instead of opening additional ports.

## 1. Prepare the Jetson

Install a JetPack release compatible with the CUDA-dependent packages you plan
to use. Put the repository at `/opt/evonith-bf`, create a dedicated service user,
and use persistent storage for runtime data:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin evonith
sudo mkdir -p /opt/evonith-bf /var/lib/evonith-bf /etc/evonith-bf
sudo chown -R evonith:evonith /opt/evonith-bf /var/lib/evonith-bf
```

Add `evonith` to the groups that own the Jetson CUDA device nodes (commonly
`video` and `render`) after confirming them with `ls -l /dev/nvhost-gpu`. The
verification script checks access as the actual service user, not just whether
the device file exists.

Install the conservative backend first:

```bash
cd /opt/evonith-bf
uv sync --no-dev --group edge
sudo -u evonith .venv/bin/python scripts/bootstrap_runtime.py \
  --create --runtime-dir /var/lib/evonith-bf
```

The edge group intentionally does not install Torch, vector models, or a CUDA
XGBoost build. Install only JetPack-compatible ARM64 packages for the features
you actually use. Do not copy an x86 CUDA virtual environment to the Jetson.

## 2. Configure the backend

Create `/etc/evonith-bf/backend.env` from the Jetson device template, owned by
root and readable by the service group:

```bash
sudo install -o root -g evonith -m 0640 infra/env/edge.jetson.env.example \
  /etc/evonith-bf/backend.env
sudo openssl rand -hex 32
sudo editor /etc/evonith-bf/backend.env
```

Set at least:

```bash
EVONITH_BACKEND_ENV=production
EVONITH_RUNTIME_PROFILE=edge
EVONITH_RUNTIME_DIR=/var/lib/evonith-bf
EVONITH_EDGE_MODE=true
EVONITH_EDGE_DEVICE_TYPE=jetson
EVONITH_UVICORN_HOST=127.0.0.1
EVONITH_UVICORN_PORT=1432
EVONITH_UVICORN_WORKERS=1
EVONITH_UVICORN_BIN=/opt/evonith-bf/.venv/bin/uvicorn
EVONITH_AUTH_SECRET_KEY=<set-a-strong-random-secret>
EVONITH_AUTH_REQUIRE_SECRET_IN_PRODUCTION=true
EVONITH_ML_DEVICE=auto
EVONITH_XGBOOST_DEVICE=auto
EVONITH_CUDA_REQUIRED=false
```

One Uvicorn worker avoids duplicating model memory on the 8 GB device. Start in
automatic mode so an incompatible CUDA package falls back safely to CPU. After
GPU execution is verified, set `EVONITH_CUDA_REQUIRED=true` if CPU fallback
would be operationally unacceptable.

`BACKEND_CORS_ORIGINS` is only needed for browser-side JavaScript calls. Normal
Streamlit Python requests are server-to-server and do not depend on CORS. If a
browser client is added, list exact HTTPS origins separated by commas; never use
`*` with authenticated endpoints.

## 3. Validate acceleration and deployment

Run these commands inside the same environment used by systemd:

```bash
set -a
source /etc/evonith-bf/backend.env
set +a
.venv/bin/python scripts/validate_deployment.py --profile edge --offline --strict
.venv/bin/python scripts/verify_jetson_cuda.py
```

For installations that require the optional runtimes:

```bash
.venv/bin/python scripts/verify_jetson_cuda.py \
  --require-torch --require-xgboost
```

Jetson commonly does not provide `nvidia-smi`; that field is diagnostic and is
not itself a failure. The important checks are the CUDA device node and actual
framework execution. The admin API also exposes live selection details at
`GET /api/v1/status/accelerator`.

## 4. Install the backend service

```bash
sudo install -o root -g root -m 0644 \
  infra/systemd/evonith-backend.service.example \
  /etc/systemd/system/evonith-backend.service
sudo systemctl daemon-reload
sudo systemctl enable --now evonith-backend
sudo systemctl status evonith-backend
curl --fail http://127.0.0.1:1432/api/v1/health
curl --fail http://127.0.0.1:1432/api/v1/readiness
```

The example unit assumes the repository is `/opt/evonith-bf` and runtime data is
`/var/lib/evonith-bf`. Adjust both the unit and environment file together if
your paths differ.

## 5. Publish the API with TLS

Create a DNS `A` record such as `api.example.com` pointing to the public IP.
Give the Jetson a stable LAN address, and forward router TCP ports `80` and `443`
to the Jetson. Do not forward `1432`, `8080`, or `8501`.

Install Caddy, then copy and edit the API-only configuration:

```bash
sudo install -o root -g root -m 0644 \
  infra/caddy/Caddyfile.jetson-api.example /etc/caddy/Caddyfile
sudo editor /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
curl --fail https://api.example.com/api/v1/health
```

Allow inbound `80/tcp` and `443/tcp` only. Restrict SSH to a trusted network or
VPN. If the public IP changes, configure a dynamic-DNS updater before relying on
the endpoint.

## 6. Configure hosted Streamlit

Set these variables in the Streamlit deployment's secret/environment settings:

```bash
BACKEND_API_BASE_URL=https://api.example.com/api/v1
BACKEND_API_VERIFY_SSL=true
BACKEND_API_CONNECT_TIMEOUT_SECONDS=5
BACKEND_API_TIMEOUT_SECONDS=60
BACKEND_API_MAX_RETRIES=2
USE_BACKEND_API=true
USE_BACKEND_API_AUTH=true
USE_BACKEND_API_ADMIN=true
USE_BACKEND_API_DATA_EXPLORER=true
USE_BACKEND_API_DATASETS=true
USE_BACKEND_API_FEEDBACK=true
USE_BACKEND_API_MATERIAL_BALANCE=true
USE_BACKEND_API_RECOMMENDATIONS=true
USE_BACKEND_API_BLEND_OPTIMIZER=true
USE_BACKEND_API_COPILOT=true
USE_BACKEND_API_FURNACEMIND=true
USE_BACKEND_API_OPS=true
```

Restart the Streamlit deployment after changing variables. Keep TLS verification
enabled; fixing DNS/certificates is safer than setting it to false.

## 7. End-to-end and operational checks

From a machine outside the Jetson LAN:

```bash
python scripts/smoke_test_deployment.py \
  --backend-url https://api.example.com/api/v1 --skip-auth
```

Before each update, run `scripts/backup_runtime.py`; after an update, validate
health/readiness, inspect `journalctl -u evonith-backend`, and keep the previous
release available for rollback.

```bash
systemctl is-active evonith-backend caddy
journalctl -u evonith-backend --since "15 minutes ago"
curl --fail https://api.example.com/api/v1/health
curl --fail https://api.example.com/api/v1/readiness
```
