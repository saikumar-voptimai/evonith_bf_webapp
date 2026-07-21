# Raspberry Pi Backend Deployment Guide

This guide installs the same Evonith FastAPI backend used on Jetson on a
Raspberry Pi. The application code is shared. Raspberry Pi uses the CPU-only
edge configuration; Jetson can select CUDA when a compatible runtime exists.

Follow the sections in order. Do not configure public routing until local
health and readiness both pass.

## 1. Supported starting point

Use:

- Raspberry Pi 4 or 5, preferably 8 GB RAM
- Raspberry Pi OS 64-bit or Ubuntu Server 64-bit
- `aarch64`/`arm64`, not 32-bit `armv7l`
- SSD/NVMe storage where possible; avoid production writes to an SD card
- Wired Ethernet and a router DHCP reservation
- Python 3.10 or newer

The conservative edge profile excludes local LLM, vector, document, and
provider-AI dependencies. Enable those only after measuring Pi memory and
temperature under realistic load.

## 2. Collect the Pi facts first

Run on the Pi:

```bash
uname -m
cat /etc/os-release
python3 --version
hostname -I
free -h
df -h /
git --version
```

Stop when `uname -m` is not `aarch64` or `arm64`. Reinstall a 64-bit OS before
continuing. Reserve the Pi's LAN address in the router so port forwarding does
not break after a reboot.

## 3. Install operating-system packages

```bash
sudo apt update
sudo apt install -y \
  ca-certificates \
  curl \
  git \
  build-essential \
  nginx \
  openssl \
  ufw
```

Apply normal OS security updates and reboot if the kernel or firmware changed:

```bash
sudo apt upgrade
sudo reboot
```

## 4. Install uv

Install `uv` using Astral's installer while logged in as the account that will
own the production checkout:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

The GitHub runner workflow also sets up its own pinned `uv`, so CI/CD does not
depend on an interactive shell profile.

## 5. Clone the production source

Replace `<GITHUB_REPOSITORY_URL>` with the repository URL. For a private
repository, use an SSH deploy key or GitHub credential helper; never put a token
in a shell command or remote URL.

```bash
sudo install -d -o "$USER" -g "$USER" -m 0755 /opt/evonith-bf
git clone \
  --branch release \
  --single-branch \
  <GITHUB_REPOSITORY_URL> \
  /opt/evonith-bf

cd /opt/evonith-bf
git status -sb
git log -1 --oneline
```

The production checkout must remain clean. Do not edit application files under
`/opt/evonith-bf`.

## 6. Verify the hardware profile

```bash
cd /opt/evonith-bf
python3 scripts/verify_edge_device.py --expect raspberry-pi --json
```

Expected values include:

```json
{
  "architecture": "aarch64",
  "device_family": "raspberry-pi",
  "status": "ok"
}
```

A low-memory warning is advisory. An architecture or device-family failure is
blocking.

## 7. Bootstrap the protected host layout

The bootstrap is idempotent and does not overwrite an existing environment
file. Preview it first:

```bash
cd /opt/evonith-bf
sudo scripts/bootstrap_edge_host.sh \
  --device raspberry-pi \
  --deploy-owner "$USER" \
  --dry-run
```

Apply it:

```bash
sudo scripts/bootstrap_edge_host.sh \
  --device raspberry-pi \
  --deploy-owner "$USER"
```

This creates:

| Purpose | Path |
|---|---|
| Production source | `/opt/evonith-bf` |
| Protected configuration | `/etc/evonith-bf/backend.env` |
| Persistent runtime | `/var/lib/evonith-bf` |
| systemd service | `/etc/systemd/system/evonith-backend.service` |
| Service account | `evonith` |

## 8. Configure the backend

Generate a strong secret without pasting it into Git or chat:

```bash
openssl rand -hex 32
```

Edit the protected file and replace the auth-secret placeholder:

```bash
sudo nano /etc/evonith-bf/backend.env
sudo chown root:evonith /etc/evonith-bf/backend.env
sudo chmod 0640 /etc/evonith-bf/backend.env
sudo stat -c '%U %G %a %n' /etc/evonith-bf/backend.env
```

Expected permissions are `root evonith 640`. Keep these Pi values:

```dotenv
EVONITH_EDGE_DEVICE_TYPE=raspberry-pi
EVONITH_UVICORN_HOST=127.0.0.1
EVONITH_UVICORN_PORT=1432
EVONITH_UVICORN_WORKERS=1
EVONITH_ML_DEVICE=cpu
EVONITH_XGBOOST_DEVICE=cpu
EVONITH_CUDA_REQUIRED=false
```

Never copy the Jetson secret to the Pi unless both devices intentionally need
to issue and accept the same authentication tokens.

## 9. Install locked Python dependencies

```bash
cd /opt/evonith-bf
uv sync \
  --python /usr/bin/python3 \
  --no-dev \
  --group edge \
  --locked

.venv/bin/python --version
.venv/bin/uvicorn --version
.venv/bin/python -c \
  "import fastapi, pandas, sklearn, furnace_data; print('backend imports: OK')"
```

One `edge` dependency group is deliberately shared by Pi and Jetson. This
prevents the two installations from drifting. Hardware selection belongs in
the protected environment, not in separate Python forks.

## 10. Create and validate runtime storage

```bash
cd /opt/evonith-bf
sudo -u evonith env EVONITH_RUNTIME_DIR=/var/lib/evonith-bf \
  .venv/bin/python scripts/bootstrap_runtime.py \
  --create \
  --runtime-dir /var/lib/evonith-bf
```

Validate using the protected environment:

```bash
sudo -u evonith bash -c '
  set -a
  source /etc/evonith-bf/backend.env
  set +a
  cd /opt/evonith-bf
  exec .venv/bin/python scripts/validate_deployment.py \
    --profile edge \
    --offline \
    --strict \
    --runtime-dir /var/lib/evonith-bf
'
```

Resolve every failure before starting systemd. A warning for an optional,
unused integration can be reviewed separately.

## 11. Start the backend service

```bash
sudo systemctl enable --now evonith-backend
sudo systemctl status evonith-backend --no-pager --full

curl --fail --silent --show-error \
  http://127.0.0.1:1432/api/v1/health
curl --fail --silent --show-error \
  http://127.0.0.1:1432/api/v1/readiness
```

Both requests must return HTTP 200. If not:

```bash
sudo journalctl -u evonith-backend -n 100 --no-pager
```

## 12. Configure Nginx

```bash
cd /opt/evonith-bf
sudo install -d -o root -g www-data -m 0755 \
  /var/www/certbot/.well-known/acme-challenge
sudo install -o root -g root -m 0644 \
  infra/nginx/evonith-edge-api.conf.example \
  /etc/nginx/sites-available/evonith-api
sudo unlink /etc/nginx/sites-enabled/default 2>/dev/null || true
sudo ln -sfn /etc/nginx/sites-available/evonith-api \
  /etc/nginx/sites-enabled/evonith-api
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

Test the proxy locally:

```bash
curl --fail --silent --show-error http://127.0.0.1/api/v1/health
curl --silent --output /dev/null --write-out '%{http_code}\n' http://127.0.0.1/
```

Expected results are health HTTP 200 and root HTTP 404.

## 13. Firewall and network routing

Enable SSH before enabling UFW so remote access is not lost:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx HTTP'
sudo ufw enable
sudo ufw status verbose
```

Do not open port 1432. Nginx is the only public listener.

One public IP and one external port can normally forward to only one internal
device. Choose an operating model:

- Jetson active, Pi standby: keep public ports forwarded to Jetson.
- Pi active, Jetson standby: forward ports to Pi's reserved LAN IP.
- Both active: place a real reverse proxy/load balancer in front of both.

Automatic deployment to both devices does not automatically provide traffic
failover. Local uploads, SQLite databases, and runtime files are not replicated
between devices. Use shared PostgreSQL/object storage or a tested replication
plan before treating the Pi as a seamless failover node.

Test public HTTP only for initial health checks. Configure trusted HTTPS before
sending credentials, tokens, uploads, or furnace data.

## 14. Register CI/CD only after manual deployment passes

Complete one manual deployment and reboot test first:

```bash
sudo reboot
sudo systemctl is-active evonith-backend
sudo systemctl is-active nginx
curl --fail http://127.0.0.1/api/v1/health
```

Then follow [Edge CI/CD Guide](edge-cicd-guide.md). Register the Pi runner with
the custom label `evonith-pi`. Do not enable automatic Pi deployment before the
manual installation is healthy and backed up.
