# Jetson AGX Orin deployment

Production host for the Svelte frontend, the twin-api, the twin-worker and Streamlit (`/legacy`), programme #262.

| Issue | Content |
|---|---|
| #279 (SV-0.10) | `Caddyfile`: static SPA with `200.html` fallback, `/api` → twin-api, `/legacy` → Streamlit (this file, below) |
| #280 (SV-0.11) | CI release artefacts (`frontend-build.tar.gz`, lock files) and the aarch64 lock check (below) |
| #315 (SV-6.1) | `provision.sh`, `verify_base.sh`: JetPack 6.x, NVMe at `/var/lib/bf-webapp`, nvpmodel MAXN, chrony, ufw, user `bfapp`, directory layout, logrotate (below) |
| #316 (SV-6.2) | Python 3.12 via uv on aarch64, wheel audit, runtime benchmark |
| #317 (SV-6.3) | systemd units: caddy, twin-api, twin-worker, streamlit, redis, furnace-data-service |
| #318 (SV-6.4) | `release.sh <tag>`: CI artefact, `uv sync --frozen`, `alembic -n twin upgrade head`, smoke test, rollback |

## Base system (`provision.sh`, `verify_base.sh`, SV-6.1)

`provision.sh` sets up the Orin's base system; run it as root, and again whenever an argument changes. Every step reads
the current state and changes only what differs: a second run ends with "no changes". `verify_base.sh` checks the
result and changes nothing but its record of the last passing boot.

**Before the first run (by hand).** Flash JetPack 6.x (L4T R36.x, Ubuntu 22.04) with SDK Manager; the AGX Orin boots
from its eMMC and the NVMe holds the data only. Partition and format the NVMe (destructive, so never done by the script):

```bash
lsblk -o NAME,SIZE,MODEL,FSTYPE,MOUNTPOINT                  # the NVMe, e.g. nvme0n1
sudo parted -s /dev/nvme0n1 mklabel gpt mkpart bf-data ext4 0% 100%
sudo mkfs.ext4 -L bf-data /dev/nvme0n1p1
sudo blkid -s UUID -o value /dev/nvme0n1p1                  # the --nvme-uuid argument
```

The script installs chrony through apt (ufw and logrotate come with Ubuntu), so the first run needs the Ubuntu / NVIDIA
mirrors or a plant mirror.

```bash
sudo deploy/jetson/provision.sh --nvme-uuid <UUID> --ntp-server <plant NTP server> --admin-subnet <CIDR>
sudo deploy/jetson/verify_base.sh --admin-subnet <CIDR>
sudo reboot
sudo deploy/jetson/verify_base.sh --admin-subnet <CIDR>     # "PASS reboot": every check passes on the new boot
```

| Option | Meaning |
|---|---|
| `--nvme-uuid UUID` | Filesystem UUID (ext4 or xfs) of the NVMe partition; written to `/etc/fstab` as `UUID=…`, mounted at `/var/lib/bf-webapp` |
| `--ntp-server HOST` | Plant NTP server; repeat for a second one. chrony uses only these (no Ubuntu pools). Open item of the plan: the server's address |
| `--admin-subnet CIDR` | IPv4 network that may reach SSH (22/tcp), /8 to /32; repeat for more. Open item: the admin subnet |
| `--no-http` | Keep 80/tcp closed. Default: open, for Caddy's redirect of `http://` to `https://` |
| `--allow-ssh-lockout` | Enable the firewall although an SSH client outside the admin subnets is connected (refused otherwise) |

Steps (`main`, in this order):

| Step | Result |
|---|---|
| platform | Stops unless aarch64, L4T R36.x (JetPack 6.0 to 6.2) and Ubuntu 22.04; prints the module and the `nvidia-jetpack` version |
| packages | `chrony`, `ufw`, `logrotate` installed (apt only for the missing ones) |
| user | System user `bfapp`: shell `/usr/sbin/nologin`, home `/var/lib/bf-webapp`, groups `video` and `render` (GPU) |
| nvme | One `/etc/fstab` entry `UUID=<uuid> /var/lib/bf-webapp <fs> defaults,noatime,nofail,x-systemd.device-timeout=30s 0 2` (the first version kept as `/etc/fstab.bf-orig`), mounted. Stops when the UUID is not on an NVMe, is mounted elsewhere, or the unmounted mount point holds files |
| layout | The paths below; `/etc/bf-webapp/env` created once with a new `BF_API_JWT_SECRET` (64 hex characters), afterwards only its owner and mode are corrected |
| power | `nvpmodel -m <id>` of the `MAXN` mode in `/etc/nvpmodel.conf` (`MAXN_SUPER` on an Orin Nano / NX), and `bf-jetson-power.service` (oneshot, after `nvpmodel.service`): nvpmodel MAXN, then `jetson_clocks`, at every boot. The summary says when the mode change needs a reboot |
| clock | `/etc/chrony/chrony.conf` with only the plant servers (`iburst`, `makestep 1 3`, `rtcsync`; the Ubuntu file kept as `chrony.conf.bf-orig`), systemd-timesyncd off, chrony enabled; waits up to 60 s for an offset below 100 ms |
| logs | `/etc/logrotate.d/bf-webapp`: `/var/log/bf-webapp/*.log` daily, `rotate 14`, `maxage 14`, compressed, `copytruncate`, as `bfapp`; checked with `logrotate --debug`. journald (`journald.conf.d/bf-webapp.conf`): persistent, `MaxRetentionSec=14day` |
| firewall | ufw: incoming denied, outgoing allowed, routing denied; allowed in: 443/tcp and 80/tcp from anywhere, 22/tcp from each admin subnet. Rewritten (`ufw --force reset`) only when `ufw show added`, the active state or a default policy differs (routing: `DEFAULT_FORWARD_POLICY="DROP"` in `/etc/default/ufw`, since `ufw status verbose` prints `disabled (routed)` while IP forwarding is off). Stops while an SSH client outside the admin subnets is connected |

| Path | Owner | Mode | Content |
|---|---|---|---|
| `/opt/bf-webapp/releases/<tag>` | root:root | 755 | One directory per release (`release.sh`, SV-6.4) |
| `/opt/bf-webapp/current` | root | symlink | → `releases/<tag>`, set by `release.sh`; absent until the first release. The Caddyfile's `BF_SPA_ROOT` default is `/opt/bf-webapp/current/frontend/build` |
| `/var/lib/bf-webapp` | bfapp:bfapp | 750 | NVMe mount point; `storage/` (application state: the releases' `src/storage`), `cache/` (`UV_CACHE_DIR`, `XDG_CACHE_HOME`) |
| `/var/log/bf-webapp` | bfapp:bfapp | 750 | Log files of the services (`*.log`, 14 days) |
| `/etc/bf-webapp/env` | root:bfapp | 640 | `EnvironmentFile=` of the units (SV-6.3): `BF_API_ENVIRONMENT=prod`, the JWT secret, `BF_REDIS_URL`; `DATABASE_URL` and the InfluxDB, LLM and Qdrant keys are filled in by hand before the first release |

For the units of SV-6.3: name them `bf-*` (`verify_base.sh` then checks them without arguments), and give every unit
that writes to the NVMe `RequiresMountsFor=/var/lib/bf-webapp`. The `nofail` mount lets the Jetson boot, with SSH
reachable, when the NVMe is missing; the services then stay down instead of writing to the eMMC.

**Checks** (`verify_base.sh`; one line per check, exit status 1 on a FAIL):

| Option | Meaning |
|---|---|
| `--max-offset-ms MS` | Largest accepted clock offset (default 100) |
| `--admin-subnet CIDR` | The SSH rules must name exactly these networks (default: any network, never anywhere) |
| `--no-http` | 80/tcp must be closed |
| `--unit NAME` | Another unit that must be enabled and active |

| Check | PASS when |
|---|---|
| platform | L4T R36 (JetPack 6.x) |
| clock | `chronyc tracking`: synchronised to an NTP source (not the local clock) and the System time offset below `--max-offset-ms`; prints the error bound (offset + root dispersion + root delay / 2) and the selected source |
| nvme | `/var/lib/bf-webapp` mounted from `/dev/nvme*`, by its UUID in `/etc/fstab` |
| power | `nvpmodel -q` reports a MAXN mode |
| layout | `bfapp` without login shell; every path of the table with its owner and mode; `current` points into `releases/` (WARN while no release exists) |
| logs | logrotate file with `rotate 14`, accepted by `logrotate --debug`; journald `MaxRetentionSec=14day` |
| firewall | ufw active, incoming denied by default; allowed: 443/tcp (and 80/tcp unless `--no-http`) from anywhere, 22/tcp only from admin subnets; any other rule fails |
| ports | INFO: the open-port list, network and loopback-only listeners (`ss -tulnp`). WARN for a listener beyond loopback other than 22, 80, 443/tcp and 68, 546, 5353/udp (DHCP, mDNS): ufw blocks it, but it should bind 127.0.0.1 |
| units | `ssh`, `chrony`, `ufw`, `bf-jetson-power.service`, `logrotate.timer`, every enabled `bf-*` unit, `caddy` and `redis-server` when installed, and each `--unit`: enabled and active; `systemctl is-system-running` = running |
| reboot | Every other check passes on a boot other than the last passing one (record `/var/lib/bf-webapp/verify/last_pass`): the services came back after the reboot. Make no changes between the reboot and the run |

**External port scan** (the firewall seen from the network; record the output in #315):

```bash
# From a control-room PC outside the admin subnet: only 80 and 443 open; 22 and every other port filtered
nmap -Pn -sT -p- bf02.local
# From the admin subnet: 22, 80, 443 open; 6379 (Redis), 8100 (twin-api), 8501 (Streamlit) filtered
nmap -Pn -sT -p 22,80,443,6379,8100,8501 bf02.local
# UDP: nothing open; 5353 (mDNS, accepted by ufw's default before.rules) may answer from the same subnet
sudo nmap -Pn -sU --top-ports 100 bf02.local
```

Without nmap, on Windows: `Test-NetConnection bf02.local -Port 443` gives `TcpTestSucceeded : True`; ports 22 (outside
the admin subnet), 6379, 8100 and 8501 give `False`.

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
| `BF_SPA_ROOT` | `/opt/bf-webapp/current/frontend/build` | Frontend build of the current release |
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
