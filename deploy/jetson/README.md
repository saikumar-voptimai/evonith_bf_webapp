# Jetson AGX Orin deployment

Production host for the Svelte frontend, the twin-api, the twin-worker and Streamlit (`/legacy`), programme #262.

| Issue | Content |
|---|---|
| #279 (SV-0.10) | `Caddyfile`: static SPA with `200.html` fallback, `/api` → twin-api, `/legacy` → Streamlit (this file, below) |
| #280 (SV-0.11) | CI release artefacts (`frontend-build.tar.gz`, lock files) and the aarch64 lock check (below) |
| #315 (SV-6.1) | `provision.sh`, `verify_base.sh`: JetPack 6.x, NVMe at `/var/lib/bf-webapp`, nvpmodel MAXN, chrony, ufw, user `bfapp`, directory layout, logrotate (below) |
| #316 (SV-6.2) | `install_python.sh`: uv 0.7.13 and uv-managed CPython 3.12, `uv sync --frozen`, import check; wheel audit (`docs/jetson_wheel_audit.md`); `benchmark.sh`: optimiser runtimes against the plan's targets (`docs/jetson_benchmark.md`) (below) |
| #317 (SV-6.3) | `systemd/`: units of Caddy, twin-api, twin-worker, Streamlit, Redis, furnace-data-service, the target `bf-webapp.target` and the health check (`healthcheck.sh`, timer); installed by `provision.sh` (below) |
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
| packages | `chrony`, `ufw`, `logrotate`, `redis-server`, `curl` installed (apt only for the missing ones) |
| user | System user `bfapp`: shell `/usr/sbin/nologin`, home `/var/lib/bf-webapp`, groups `video` and `render` (GPU) |
| nvme | One `/etc/fstab` entry `UUID=<uuid> /var/lib/bf-webapp <fs> defaults,noatime,nofail,x-systemd.device-timeout=30s 0 2` (the first version kept as `/etc/fstab.bf-orig`), mounted. Stops when the UUID is not on an NVMe, is mounted elsewhere, or the unmounted mount point holds files |
| layout | The paths below; `/etc/bf-webapp/env` created once with a new `BF_API_JWT_SECRET` (64 hex characters), afterwards only its owner and mode are corrected |
| power | `nvpmodel -m <id>` of the `MAXN` mode in `/etc/nvpmodel.conf` (`MAXN_SUPER` on an Orin Nano / NX), and `bf-jetson-power.service` (oneshot, after `nvpmodel.service`): nvpmodel MAXN, then `jetson_clocks`, at every boot. The summary says when the mode change needs a reboot |
| clock | `/etc/chrony/chrony.conf` with only the plant servers (`iburst`, `makestep 1 3`, `rtcsync`; the Ubuntu file kept as `chrony.conf.bf-orig`), systemd-timesyncd off, chrony enabled; waits up to 60 s for an offset below 100 ms |
| logs | `/etc/logrotate.d/bf-webapp`: `/var/log/bf-webapp/*.log` daily, `rotate 14`, `maxage 14`, compressed, `copytruncate`, as `bfapp`; checked with `logrotate --debug`. journald (`journald.conf.d/bf-webapp.conf`): persistent, `MaxRetentionSec=14day` |
| services | Caddy 2.10.2 in `/usr/local/bin/caddy` (GitHub release, checked against its checksum file), `redis-server` from apt with its own `redis-server.service` masked (and `caddy.service`, if a package installed one); the units of `systemd/` in `/etc/systemd/system`, the health check in `/usr/local/sbin/bf-healthcheck`, every unit with an `[Install]` section enabled; with a release installed, changed units restarted and `bf-webapp.target` started. Section "Services" |
| firewall | ufw: incoming denied, outgoing allowed, routing denied; allowed in: 443/tcp and 80/tcp from anywhere, 22/tcp from each admin subnet. Rewritten (`ufw --force reset`) only when `ufw show added`, the active state or a default policy differs (routing: `DEFAULT_FORWARD_POLICY="DROP"` in `/etc/default/ufw`, since `ufw status verbose` prints `disabled (routed)` while IP forwarding is off). Stops while an SSH client outside the admin subnets is connected |

| Path | Owner | Mode | Content |
|---|---|---|---|
| `/opt/bf-webapp/releases/<tag>` | root:root | 755 | One directory per release (`release.sh`, SV-6.4) |
| `/opt/bf-webapp/current` | root | symlink | → `releases/<tag>`, set by `release.sh`; absent until the first release. The Caddyfile's `BF_SPA_ROOT` default is `/opt/bf-webapp/current/frontend/build` |
| `/var/lib/bf-webapp` | bfapp:bfapp | 750 | NVMe mount point; `storage/` (application state: the releases' `src/storage`), `cache/` (`UV_CACHE_DIR`, `XDG_CACHE_HOME`) |
| `/var/log/bf-webapp` | bfapp:bfapp | 750 | Log files of the services (`*.log`, 14 days) |
| `/etc/bf-webapp/env` | root:bfapp | 640 | `EnvironmentFile=` of the units (SV-6.3): `BF_API_ENVIRONMENT=prod`, the JWT secret, `BF_REDIS_URL`; `DATABASE_URL` and the InfluxDB, LLM and Qdrant keys are filled in by hand before the first release |

The service units (SV-6.3) are named `bf-*`, so `verify_base.sh` checks them without arguments, and every one has
`RequiresMountsFor=/var/lib/bf-webapp`. The `nofail` mount lets the Jetson boot, with SSH reachable, when the NVMe is
missing; the services then stay down instead of writing to the eMMC.

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

Processes behind the proxy, all on 127.0.0.1 (units of section "Services"; by hand from a checkout):

```bash
# twin-api: 2 uvicorn workers (bf-twin-api.service)
uv run uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8100 --workers 2 --timeout-graceful-shutdown 10
# Streamlit under /legacy (bf-streamlit.service)
uv run python run_streamlit.py --server.address 127.0.0.1 --server.port 8501 --server.baseUrlPath legacy \
    --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
```

The twin-api trusts `X-Forwarded-For` from `BF_API_TRUSTED_PROXIES` (default 127.0.0.1, ::1, i.e. Caddy) and sets its
session cookie `Secure` (`BF_API_COOKIE_SECURE=true`), which needs the HTTPS origin.

**Certificate.** With `BF_TLS=internal`, Caddy creates a local CA under its data directory
(`/var/lib/bf-webapp/caddy/data/caddy/pki/authorities/local/root.crt`, `XDG_DATA_HOME` of `bf-caddy.service`).
Install that root certificate on every control-room PC (Windows: Trusted Root Certification Authorities). The
Caddyfile sets `skip_install_trust`: Caddy does not add its root to the Jetson's own store.
With a certificate of the plant CA, set `BF_TLS="/etc/bf-webapp/tls/bf02.crt /etc/bf-webapp/tls/bf02.key"` (readable
by `bfapp`, the user of `bf-caddy.service`). The choice is an open item of the plan (TLS certificate source and Jetson
host name).

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

## Services (`systemd/`, SV-6.3)

`provision.sh` (step "services") installs the units of `deploy/jetson/systemd/` in `/etc/systemd/system` and enables
`bf-webapp.target` at boot. Every service runs as `bfapp` with `EnvironmentFile=/etc/bf-webapp/env` (the Python units
also read `/etc/bf-webapp/uv.env`), `RequiresMountsFor=/var/lib/bf-webapp`, `PartOf=bf-webapp.target`,
`Restart=always` after 2 s and `StartLimitIntervalSec=0` (systemd never gives up restarting), `MemoryMax`, and
`NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=full`, `ProtectHome`.

| Unit | Process | Listens | MemoryMax | Starts after |
|---|---|---|---|---|
| `bf-redis.service` | `redis-server` (apt), `Type=notify`; append-only file every second in `/var/lib/bf-webapp/redis`, `maxmemory 512mb`, `noeviction` | 127.0.0.1:6379 | 1 GiB | network |
| `bf-furnace-data.service` | furnace-data-service: `uvicorn app.main:app` in `furnace-data-service/`, the release's `.venv` | 127.0.0.1:8000 | 2 GiB | network online |
| `bf-twin-api.service` | twin-api: `uvicorn api.main:app --workers 2 --timeout-graceful-shutdown 10` | 127.0.0.1:8100 | 3 GiB | Redis |
| `bf-twin-worker.service` | twin-worker: `python -m worker`, `BF_WORKER_POOL=8`, `PYTHONPATH=src`; `KillMode=mixed`, `CPUWeight=50` | | 12 GiB | Redis |
| `bf-streamlit.service` | `run_streamlit.py` with `--server.baseUrlPath legacy`, no file watcher | 127.0.0.1:8501 | 6 GiB | Redis, furnace-data-service |
| `bf-caddy.service` | `caddy run` with the release's `Caddyfile`, `Type=notify`; `CAP_NET_BIND_SERVICE` for ports 80 and 443; data in `/var/lib/bf-webapp/caddy` | :443, :80 | 512 MiB | twin-api, Streamlit |
| `bf-healthcheck.timer` | `bf-healthcheck.service` (root, oneshot): `/usr/local/sbin/bf-healthcheck` = `healthcheck.sh`, 30 s after the start, then 10 s after each run | | | |

The memory limits add up to 24.5 GiB (the AGX Orin's 32 GB, shared with the GPU, less the system); measured values
replace them after the soak on the device. The application units start only when a release is installed
(`ConditionPathExists` on `/opt/bf-webapp/current/.venv/bin/python`, Caddy on the release's `Caddyfile`); Redis does
not depend on a release. They find the interpreter, `src` and `run_streamlit.py` relative to
`WorkingDirectory=/opt/bf-webapp/current`, which systemd resolves once at the start: a release switch never mixes two
releases in one process tree, and `release.sh` (SV-6.4) restarts the target after switching.

**Start order.** `After=` and `Wants=`: Redis and furnace-data-service first, then the twin-api, the twin-worker and
Streamlit, then Caddy, so the first request after a boot finds its upstream. `Wants=`, not `Requires=`, between the
services: a start that fails on a dependency is never retried, while a service that starts without its dependency
(arq without Redis exits; the twin-api answers and retries Redis every 15 s) is restarted until it runs. The only hard
dependency is the NVMe (`RequiresMountsFor`).

**Restart and watchdog.** A process that exits or is killed is restarted by systemd after 2 s; a uvicorn worker that
dies is restarted by uvicorn's parent. None of the processes signals systemd's watchdog (`WatchdogSec`), so
`bf-healthcheck.timer` probes the services that run but may no longer answer, and restarts one after two failed probes
in a row (a hung service is restarted 16 to 30 s after it stops answering). A service that is not active (stopped by
hand, starting, waiting for its restart) or still in its start grace is not probed.

| Unit | Probe | Grace |
|---|---|---|
| `bf-redis.service` | `redis-cli -u $BF_REDIS_URL ping` answers `PONG` | 15 s |
| `bf-twin-api.service` | `GET http://127.0.0.1:8100/api/health` 2xx | 90 s |
| `bf-streamlit.service` | `GET http://127.0.0.1:8501/legacy/_stcore/health` 2xx | 90 s |
| `bf-furnace-data.service` | `GET http://127.0.0.1:8000/health` 2xx | 60 s |
| `bf-caddy.service` | any HTTP answer from `https://$BF_SITE:$BF_HTTPS_PORT/` at 127.0.0.1 | 15 s |
| `bf-twin-worker.service` | `twin:worker:alive` (written every 10 s) at most 60 s old; not judged while Redis does not answer | 120 s |

The check logs only failed probes and restarts, at warning level: `journalctl -u bf-healthcheck -p warning`.

**Two twin-api workers.** The revoked sessions are shared through Redis (SV-6.6). Per process remain the sign-in
failure limit (a client can fail up to 2 x `BF_API_LOGIN_MAX_FAILURES` sign-ins per window) and FurnaceMind's tool
state (a follow-up question answered by the other worker starts without the DataFrames the conversation fetched); set
`--workers 1` in the unit if either matters more than the second worker.

```bash
systemctl list-dependencies bf-webapp.target              # every unit and its state
sudo systemctl restart bf-webapp.target                   # all services (release.sh after a switch)
journalctl -u bf-twin-api -u bf-twin-worker -f            # logs (journald keeps 14 days)
sudo systemctl kill -s KILL bf-twin-api; sleep 30; curl -fsS http://127.0.0.1:8100/api/health   # serving again
systemd-analyze critical-chain bf-caddy.service           # boot to serving
```

Acceptance on the device (open until measured, #317): a killed process serves again within 30 s (kill each of the
six; the twin-api imports in about 16 s on the development laptop), and boot to serving is below 3 min
(`systemd-analyze critical-chain bf-caddy.service`, then the first 200 from `https://<site>/api/health`).
`tests/test_jetson_units.py` checks the units: sections and keys, the ports against the Caddyfile and the environment
template, the start order (no cycle), the health check's probes and its decisions.

## Release artefacts (CI, SV-0.11)

A tag `v*` runs `.github/workflows/release.yml`: the full CI suite, then the GitHub release of the tag with

| File | Content |
|---|---|
| `frontend-build.tar.gz` | `build/` of the SPA (entries sorted, owner 0, commit time as mtime); unpack into `<release>/frontend/`, giving `BF_SPA_ROOT` |
| `uv.lock`, `pnpm-lock.yaml` | Lock files of the tagged commit (`uv sync --frozen` on the Jetson) |
| `SHA256SUMS` | SHA-256 of the three files; `sha256sum --check SHA256SUMS` before unpacking |

`release.sh` (SV-6.4, #318) consumes these files.

## Python and uv (`install_python.sh`, SV-6.2)

The project runs on CPython 3.12 installed by uv (python-build-standalone), never on JetPack's system Python, and
installs from `uv.lock` with `uv sync --frozen`. `install_python.sh` sets this up; run it as root after `provision.sh`,
and again for every checkout or release directory to sync. Like `provision.sh` it changes only what differs.

```bash
sudo deploy/jetson/install_python.sh                          # uv, CPython 3.12, /etc/bf-webapp/uv.env
sudo deploy/jetson/install_python.sh --release ~/bf-webapp    # also: uv sync --frozen there, then the import check
```

| Option | Meaning |
|---|---|
| `--uv-version V` | uv release to install (default 0.7.13, the lock rule's version: CLAUDE.md, Makefile `UV_LOCK`) |
| `--python X.Y` | CPython minor version (default 3.12, `requires-python` and the lock check's target) |
| `--release DIR` | Checkout or release directory: `uv sync --frozen` as the directory's owner (bfapp for a release directory, the operator for a checkout; never root), then the import check |
| `--no-check` | Skip the import check after the sync |

| Step | Result |
|---|---|
| platform | Stops unless aarch64, L4T R36 (JetPack 6.x), Ubuntu 22.04, and `provision.sh` has run (`bfapp`, `/etc/bf-webapp`, `/var/lib/bf-webapp/cache`) |
| packages | `build-essential` (quickjs compiles its C extension during `uv sync`), `curl`, `ca-certificates` |
| uv | `/usr/local/bin/uv` and `uvx` from the GitHub release tarball `uv-aarch64-unknown-linux-gnu.tar.gz`, checked against its `.sha256` file; replaced only when the version differs |
| python | `uv python install 3.12` into `/opt/bf-webapp/python` (root-owned, readable by all) |
| env | `/etc/bf-webapp/uv.env` (root:bfapp 640): `UV_PYTHON_INSTALL_DIR=/opt/bf-webapp/python`, `UV_PYTHON=3.12`, `UV_PYTHON_PREFERENCE=only-managed`, `UV_PYTHON_DOWNLOADS=never`, `UV_CACHE_DIR=/var/lib/bf-webapp/cache/uv`, `UV_LINK_MODE=copy`, `UV_FROZEN=1`, `UV_NO_PROGRESS=1`. `EnvironmentFile=` of the units (SV-6.3) and read by `release.sh` (SV-6.4) |
| release | With `--release DIR`: `uv sync --frozen` in DIR with these variables (the operator keeps their own uv cache), then imports torch, scipy, numpy, pandas, psycopg2, duckdb, chromadb, pyarrow, influxdb_client_3, sentence_transformers, xgboost, onnxruntime, quickjs and pypika in the synced `.venv`, and solves one LP with HiGHS through scipy |

The downloads (uv release, python-build-standalone, PyPI) need GitHub and PyPI from the Jetson, or a plant mirror
(`UV_PYTHON_INSTALL_MIRROR`, `UV_INDEX_URL`). After a sync, as the checkout's owner: `uv run --no-sync pytest -q tests/`
(full suite) and the benchmark below.

**Benchmark** (`benchmark.sh`; `docs/jetson_benchmark.md`). Runs `scripts/optimiser_bench.py` in the modes `solve`,
`jacobian --solvers 8`, `step2`, `max_production` and `step1` on the reference plan day, writes the measured table into
`docs/jetson_benchmark.md` and compares each mode with its target of section 10 of the plan (#262): network solve
2.8 s, Jacobian 6 s, Step 2 90 s, maximum production 300 s, Step 1 600 s. Run it as the checkout's owner, not root;
it changes nothing on the system. Exit status 1 when a mode is above its target or gave no result.

```bash
deploy/jetson/benchmark.sh --open-issue                       # all five modes; an issue per runtime above target
deploy/jetson/benchmark.sh --mode solve --mode jacobian       # a partial run leaves the document as it is
```

| Option | Meaning |
|---|---|
| `--release DIR` | Synced checkout or release directory (default: the repository of the script) |
| `--solvers N` | Solver processes of the `jacobian` mode (default 8, the target's pool); the other modes use `BF_STRATA_SOLVERS`, else min(8, cpus) |
| `--mode M` | One of `solve`, `jacobian`, `step2`, `max_production`, `step1`; repeat for several (default all five) |
| `--doc FILE` | Document whose block between `<!-- benchmark:start -->` and `<!-- benchmark:end -->` is replaced (default `docs/jetson_benchmark.md`) |
| `--log-dir DIR` | Output of every mode, the results TSV and the issue bodies (default `/tmp/bf-benchmark-<UTC time>`) |
| `--live-day` | Plan day and blast moisture of today instead of the reference day 2026-10-15 with the shipped moisture |
| `--open-issue` | Open a GitHub issue (`gh`, signed in) for each mode above its target; without it the script prints the `gh issue create` command |
| `--repo OWNER/NAME` | Repository of the issues (default: the checkout's `origin`, else `saikumar-voptimai/evonith_bf_webapp`) |

## Python packages on aarch64

`make lock-check` (CI job `static`) resolves `uv.lock` for `aarch64-manylinux_2_35` (JetPack 6: Ubuntu 22.04,
glibc 2.35) and CPython 3.12 from wheels, with two exceptions that publish no aarch64 wheel and build from source
during `uv sync`: `pypika` (pure Python) and `quickjs` (C extension; needs `build-essential`). The wheel audit
(`docs/jetson_wheel_audit.md`, `python scripts/jetson_wheel_audit.py --write` or `make wheel-audit`) runs the same
recipe and records which file uv installs for each package: on 29 Oct 2026, 244 packages, all from aarch64 or
pure-Python wheels except those two, glibc floor at most 2.34. torch 2.2.2 from PyPI is the CPU build on aarch64 (its
CUDA libraries are locked for x86_64 only); the Orin GPU is not used.
