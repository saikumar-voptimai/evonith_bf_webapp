# Jetson Backend Operations Runbook

This is the day-to-day maintenance guide for the Evonith FastAPI backend on the
Jetson Orin Nano. It documents the current installation, where to find files,
how to deploy updates, how to read logs, and what to check when something fails.

Never put the public IP, passwords, tokens, certificate private keys, or the
contents of `backend.env` in Git, screenshots, tickets, or chat messages.

## 1. Current architecture

```text
Hosted Streamlit application
        |
        | Temporary test: HTTP port 80
        | Production: HTTPS port 443
        v
Router public IP / port forwarding
        |
        v
Nginx on Jetson
        |
        | http://127.0.0.1:1432
        v
FastAPI/Uvicorn systemd service
        |
        v
/var/lib/evonith-bf persistent runtime data
```

Nginx is the only public API entrypoint. Uvicorn listens only on
`127.0.0.1:1432`; therefore port 1432 must not be opened in UFW or forwarded by
the router.

## 2. Installation inventory

| Purpose | Location |
|---|---|
| Production source | `/opt/evonith-bf` |
| Deployment branch | `dev-v01` |
| Python environment | `/opt/evonith-bf/.venv` |
| FastAPI application | `/opt/evonith-bf/apps/backend_api/app` |
| Shared data package | `/opt/evonith-bf/packages/furnace-data` |
| Deployment scripts | `/opt/evonith-bf/scripts` |
| Persistent runtime | `/var/lib/evonith-bf` |
| Protected configuration | `/etc/evonith-bf/backend.env` |
| systemd unit | `/etc/systemd/system/evonith-backend.service` |
| systemd template | `/opt/evonith-bf/infra/systemd/evonith-backend.service.example` |
| Nginx site | `/etc/nginx/sites-available/evonith-api` |
| Enabled Nginx link | `/etc/nginx/sites-enabled/evonith-api` |
| Nginx source template | `/opt/evonith-bf/infra/nginx/evonith-api-ip.conf.example` |
| ACME webroot | `/var/www/certbot` |
| Backend service user | `evonith` |

The development checkout under `/home/v-optimaise/evonith_bf_webapp` is not the
production source. Editing it does not update the running API. Production
updates must be applied in `/opt/evonith-bf` using the update procedure below.

## 3. Ports and firewall

| Port | Listener | Exposure | Purpose |
|---|---|---|---|
| `22` | SSH | Restricted/public as configured | Server administration |
| `80` | Nginx | Public | Temporary HTTP test, certificate validation, HTTPS redirect |
| `443` | Nginx | Public after TLS setup | Production Streamlit API traffic |
| `1432` | Uvicorn | Loopback only | Private FastAPI upstream |

Verify listeners:

```bash
sudo ss -ltnp
```

The expected backend line contains:

```text
127.0.0.1:1432
```

It must not contain `0.0.0.0:1432`.

Check UFW:

```bash
sudo ufw status verbose
```

Production UFW should allow OpenSSH, Nginx HTTP, and Nginx HTTPS. It should not
allow 1432, 8080, or 8501.

## 4. Basic health check

Run these checks in order. They identify which layer is failing.

### Backend directly

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:1432/api/v1/health

curl --fail --silent --show-error \
  http://127.0.0.1:1432/api/v1/readiness
```

### Through local Nginx

```bash
curl --fail --silent --show-error \
  http://127.0.0.1/api/v1/health
```

### Through the public endpoint

Temporary HTTP test only:

```bash
curl --fail --silent --show-error \
  http://<PUBLIC_IP>/api/v1/health
```

Production:

```bash
curl --fail --silent --show-error \
  https://<PUBLIC_IP_OR_DOMAIN>/api/v1/health
```

The expected health data contains `"status":"ok"`. Readiness should contain
`"status":"ready"`.

## 5. Backend service commands

Check status:

```bash
sudo systemctl status evonith-backend --no-pager --full
```

Start, stop, or restart:

```bash
sudo systemctl start evonith-backend
sudo systemctl stop evonith-backend
sudo systemctl restart evonith-backend
```

Check whether it starts automatically after reboot:

```bash
sudo systemctl is-enabled evonith-backend
sudo systemctl is-active evonith-backend
```

After every restart, wait for startup and run both health and readiness checks.

## 6. Logs

### Backend logs

Show the latest 100 lines:

```bash
sudo journalctl -u evonith-backend -n 100 --no-pager
```

Follow logs live; stop with `Ctrl+C`:

```bash
sudo journalctl -u evonith-backend -f
```

Show logs from the last 30 minutes:

```bash
sudo journalctl -u evonith-backend --since "30 minutes ago" --no-pager
```

Show only the current boot:

```bash
sudo journalctl -u evonith-backend -b --no-pager
```

Access logs include the request method, path, status code, duration, and
`request_id`. When a user reports an error, ask for the request ID and search:

```bash
sudo journalctl -u evonith-backend --since today --no-pager | \
  grep '<REQUEST_ID>'
```

Do not enable query-parameter logging in production unless it has been reviewed
for sensitive values.

### Nginx logs

```bash
sudo tail -n 100 /var/log/nginx/access.log
sudo tail -n 100 /var/log/nginx/error.log
sudo tail -f /var/log/nginx/error.log
```

### Runtime logs and audit data

Runtime log directory:

```text
/var/lib/evonith-bf/logs
```

Audit database:

```text
/var/lib/evonith-bf/audit/audit.db
```

These paths are intentionally protected. Inspect them with `sudo`; do not
change their ownership to make inspection easier.

## 7. Configuration maintenance

The protected configuration is:

```text
/etc/evonith-bf/backend.env
```

Required ownership and permissions:

```text
root evonith 640
```

Verify without printing secrets:

```bash
sudo stat -c '%U %G %a %n' /etc/evonith-bf/backend.env
```

Edit configuration:

```bash
sudo nano /etc/evonith-bf/backend.env
```

After a change:

```bash
sudo systemctl restart evonith-backend
sudo systemctl status evonith-backend --no-pager --full
```

Never run `cat /etc/evonith-bf/backend.env` in recorded terminals or paste its
contents into support messages. Never regenerate `EVONITH_AUTH_SECRET_KEY`
during a normal deployment; changing it invalidates existing access tokens.

## 8. Validate the installation

Run the deployment validator as the service user:

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

Validate Jetson and CUDA device access:

```bash
sudo -u evonith bash -c '
  set -a
  source /etc/evonith-bf/backend.env
  set +a
  cd /opt/evonith-bf
  exec .venv/bin/python scripts/verify_jetson_cuda.py
'
```

Torch and XGBoost may show as unavailable when the conservative edge profile is
installed. The base deployment is healthy when the architecture is `aarch64`,
the CUDA device is present and accessible, and the overall status is `ok`.

## 9. Deploy an application update

Use this procedure for every update. Do not edit production Python files
directly.

### 9.1 Record the current release

```bash
cd /opt/evonith-bf
git status --short
git branch --show-current
git log -1 --oneline
```

Stop if `git status --short` prints anything. Investigate unexpected production
changes before continuing.

### 9.2 Create a runtime backup

```bash
sudo -u evonith bash -c '
  set -a
  source /etc/evonith-bf/backend.env
  set +a
  cd /opt/evonith-bf
  exec .venv/bin/python scripts/backup_runtime.py \
    --runtime-dir /var/lib/evonith-bf
'
```

List backups:

```bash
sudo ls -lh /var/lib/evonith-bf/backups
```

Copy important backups off the Jetson. A backup stored only on the same SSD is
not sufficient protection against device failure.

### 9.3 Update source and dependencies

```bash
cd /opt/evonith-bf
git fetch origin
git switch dev-v01
git pull --ff-only origin dev-v01
uv sync --python /usr/bin/python3 --no-dev --group edge --locked
```

### 9.4 Validate and restart

Run the deployment validator from section 8, then:

```bash
sudo systemctl restart evonith-backend
sudo systemctl status evonith-backend --no-pager --full
curl --fail http://127.0.0.1:1432/api/v1/health
curl --fail http://127.0.0.1:1432/api/v1/readiness
```

Only declare the deployment successful after the public endpoint and Streamlit
application also pass.

## 10. Roll back application code

Use the commit recorded before the update as `<KNOWN_GOOD_COMMIT>`:

```bash
cd /opt/evonith-bf
git switch --detach <KNOWN_GOOD_COMMIT>
uv sync --python /usr/bin/python3 --no-dev --group edge --locked
sudo systemctl restart evonith-backend
```

Run health and readiness checks. After the incident is resolved, return to the
deployment branch:

```bash
cd /opt/evonith-bf
git switch dev-v01
```

Code rollback does not restore runtime data. Use the restore procedure only
when runtime data is known to be damaged or incompatible.

## 11. Runtime backup and restore

Always test a restore plan before applying it:

```bash
cd /opt/evonith-bf
sudo -u evonith .venv/bin/python scripts/restore_runtime.py \
  --backup /var/lib/evonith-bf/backups/<BACKUP_FILE>.tar.gz \
  --target-runtime-dir /var/lib/evonith-bf-restore-test
```

Review the dry-run output. Applying a restore is an incident operation: stop the
backend, preserve the current runtime directory, and follow
`docs/deployment/backup-restore-guide.md`. Do not add `--apply` during ordinary
testing.

## 12. Nginx maintenance

The active site configuration is:

```text
/etc/nginx/sites-available/evonith-api
```

The version-controlled HTTP template is:

```text
/opt/evonith-bf/infra/nginx/evonith-api-ip.conf.example
```

Compare the live configuration with the template:

```bash
sudo diff -u \
  /opt/evonith-bf/infra/nginx/evonith-api-ip.conf.example \
  /etc/nginx/sites-available/evonith-api
```

No output means they match. Expected TLS additions may create differences after
HTTPS is enabled; document and preserve those additions during updates.

After any Nginx change:

```bash
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl status nginx --no-pager --full
```

Use `reload`, not `restart`, after a valid configuration change so existing
connections can finish normally.

Check the upstream independently when Nginx reports `502 Bad Gateway`:

```bash
curl --fail http://127.0.0.1:1432/api/v1/health
```

If this succeeds, inspect Nginx configuration and error logs. If it fails,
inspect the backend service instead.

## 13. Streamlit Cloud configuration

### Temporary connectivity test only

```toml
BACKEND_API_BASE_URL = "http://<PUBLIC_IP>/api/v1"
BACKEND_API_VERIFY_SSL = false
BACKEND_API_CONNECT_TIMEOUT_SECONDS = 5
BACKEND_API_TIMEOUT_SECONDS = 60
BACKEND_API_MAX_RETRIES = 1
USE_BACKEND_API = true
SHOW_BACKEND_STATUS_BADGE = true
```

HTTP is suitable only for a short health/connectivity test. Do not use real
passwords, tokens, uploads, customer data, or furnace data over this endpoint.

### Production configuration

```toml
BACKEND_API_BASE_URL = "https://<PUBLIC_IP_OR_DOMAIN>/api/v1"
BACKEND_API_VERIFY_SSL = true
BACKEND_API_CONNECT_TIMEOUT_SECONDS = 5
BACKEND_API_TIMEOUT_SECONDS = 60
BACKEND_API_MAX_RETRIES = 2
USE_BACKEND_API = true
SHOW_BACKEND_STATUS_BADGE = true
```

Restart the Streamlit app after changing secrets. CORS is not required for the
normal Python server-to-server connection.

## 14. TLS certificate maintenance

Do not enable production login or real-data traffic until HTTPS works with a
publicly trusted certificate.

For a bare public IP, use Certbot 5.4 or newer and a short-lived Let's Encrypt
IP certificate. The ACME challenge requires public port 80. Production traffic
uses port 443. The router must forward both ports to the Jetson while IP
certificates are used.

Check Certbot and renewal scheduling:

```bash
/snap/bin/certbot --version
snap services certbot
sudo /snap/bin/certbot certificates
```

Certificates for IP addresses expire after approximately six days. Renewal
must be automatic and Nginx must reload after renewal. Test renewal after the
production certificate and Nginx TLS configuration are installed:

```bash
sudo /snap/bin/certbot renew --dry-run
```

If renewal fails, do not disable TLS verification in Streamlit as a permanent
workaround. Fix port forwarding, the ACME webroot, Certbot, or Nginx.

Official references:

- [Let's Encrypt IP certificate availability](https://letsencrypt.org/2026/01/15/6day-and-ip-general-availability.html)
- [Certbot IP certificate support](https://letsencrypt.org/2026/03/11/shorter-certs-certbot)

## 15. Common failures

| Symptom | First checks | Typical cause |
|---|---|---|
| Streamlit says backend unavailable | Public health URL, Nginx logs | Router, firewall, wrong URL, backend stopped |
| Local port 1432 refuses connection | Backend status and journal | Service failed during startup |
| Nginx returns 502 | Direct port 1432 health, Nginx error log | Uvicorn stopped or wrong upstream port |
| Nginx returns 404 for `/api/v1/health` | Active Nginx site and request path | Wrong location block or default site enabled |
| API returns 401 | Streamlit auth/token configuration | Missing, expired, or invalid token |
| API returns 500 | Request ID and backend journal | Application exception or missing dependency |
| Readiness returns 503 | Runtime ownership and disk space | Runtime missing, unwritable, or full disk |
| Service restarts repeatedly | `systemctl status`, current-boot journal | Invalid environment value or import error |
| Public IP stops working | Router WAN IP and forwarding | ISP IP changed or CGNAT |
| HTTPS certificate error | Certbot certificates and renewal logs | Expired IP certificate or failed renewal |
| CUDA status degraded | Accelerator script and device permissions | Missing package, wrong group, incompatible build |
| High memory usage | `systemctl status`, `free -h`, worker count | Multiple workers or large model cache |

## 16. Troubleshooting sequence

Use this order during an incident:

1. Check disk and memory.
2. Check backend service status.
3. Check backend journal logs.
4. Test FastAPI directly on `127.0.0.1:1432`.
5. Test through local Nginx.
6. Check Nginx status and logs.
7. Check UFW.
8. Test the public endpoint from outside the LAN.
9. Check router WAN IP and port forwarding.
10. Check Streamlit secrets and application logs.

Commands:

```bash
df -h /
free -h
sudo systemctl status evonith-backend --no-pager --full
sudo journalctl -u evonith-backend -n 100 --no-pager
curl --fail http://127.0.0.1:1432/api/v1/health
curl --fail http://127.0.0.1/api/v1/health
sudo nginx -t
sudo systemctl status nginx --no-pager --full
sudo tail -n 100 /var/log/nginx/error.log
sudo ufw status verbose
```

## 17. Regular maintenance checklist

### Daily or when alerted

- Check public health and readiness.
- Check Streamlit backend status.
- Review repeated 5xx responses or service restarts.

### Weekly

- Check free disk and memory.
- Confirm backend and Nginx are active.
- Review backend and Nginx error logs.
- Confirm a recent runtime backup exists.
- Confirm Certbot renewal scheduling is enabled.

### Before every deployment

- Confirm the production Git tree is clean.
- Record the current commit.
- Back up runtime data.
- Review release changes and dependency changes.

### After every deployment

- Run deployment validation.
- Restart the backend.
- Check health and readiness locally and publicly.
- Verify Streamlit connectivity.
- Watch logs for at least several requests.

### Monthly

- Copy a backup off the Jetson and test a restore dry run.
- Install reviewed OS security updates.
- Review firewall and router forwarding rules.
- Review service users and SSH access.
- Upgrade Certbot when a supported update is available.

## 18. Reboot verification

After a planned reboot:

```bash
sudo systemctl is-active evonith-backend
sudo systemctl is-active nginx
sudo ss -ltnp
curl --fail http://127.0.0.1:1432/api/v1/health
curl --fail http://127.0.0.1/api/v1/health
```

The backend and Nginx should start automatically. Do not consider reboot
recovery complete until the public endpoint and Streamlit app also work.
