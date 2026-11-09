# BF-02 webapp: deployment handover (Svelte frontend on the Jetson)

30 Oct 2026 · branch `feature/262_svelte_twin` (head `2520552`; not merged into `main` yet) · programme #262

This guide is for whoever installs the new operator console on the Jetson AGX Orin. It gives the order of the work, the
decisions to get from the project owner first, and the checks to record. Step-by-step detail of every script is in
[`deploy/jetson/README.md`](../deploy/jetson/README.md); this guide points to its sections instead of repeating them.

## 1. What runs where

One HTTPS address (`https://<site>/`) on the Jetson serves everything:

| Process | systemd unit | Port (127.0.0.1) | Role |
|---|---|---|---|
| Caddy | `bf-caddy` | 443, 80 (public) | HTTPS, serves the Svelte build, forwards `/api` and `/legacy` |
| Svelte app (static files) | (served by Caddy) | | The new operator console (`/`, `/trace`, `/forge`, `/strata`, `/status`, `/reports`, `/furnacemind`, `/optimiser`, `/bmo`, `/admin/...`) |
| twin-api | `bf-twin-api` | 8100 | FastAPI backend of the Svelte app, 2 workers |
| twin-worker | `bf-twin-worker` | | Scheduled and on-demand jobs: twin evaluation, forecasts, optimiser runs, reports, calibration |
| Redis | `bf-redis` | 6379 | Job queue and live event stream |
| Streamlit | `bf-streamlit` | 8501 | The existing app under `/legacy`; opens without a second login |
| furnace-data-service | `bf-furnace-data` | 8000 | Data sidecar used by Streamlit |

Everything the app reads is outside the Jetson: the PostgreSQL server in `DATABASE_URL` (plant data and user logins),
InfluxDB Cloud, Qdrant, OpenRouter / OpenAI and Open-Meteo. The Jetson needs outbound access to all of them.

The current Streamlit Cloud deployment is not affected; keep it running until the Svelte pages are signed off
(`docs/retirement.md`).

## 2. Decisions to get from the project owner before the first release

| # | Decision | Why it matters | Default if nobody objects |
|---|---|---|---|
| 1 | **Where the twin tables live** (`BF_TWIN_DB_URL`) | `release.sh` runs the twin migrations against `BF_TWIN_DB_URL`, else against `DATABASE_URL`. Without a decision, the first release creates the `twin` schema on the plant database. See section 3 | Ask; do not release until answered |
| 2 | Host name and TLS certificate (`BF_SITE`, `BF_TLS`) | Clients need a name that resolves and a certificate they trust | `bf02.local` with Caddy's own CA (`internal`); its root certificate is installed on the control-room PCs |
| 3 | Plant NTP server(s) | `provision.sh --ntp-server`; the clock must be within 100 ms | None; required |
| 4 | Admin subnet(s) for SSH | `provision.sh --admin-subnet`; the firewall allows SSH only from them | None; required |
| 5 | Internet access from the Jetson | Install needs GitHub, PyPI and the Ubuntu / NVIDIA mirrors; the app needs the data sources of section 1 | If GitHub is blocked, copy the release files over (`release.sh --artefacts`) |
| 6 | Release tag and build | Releases come from `.github/workflows/release.yml`, which runs only by hand (to save GitHub Actions minutes) and runs the full CI first | The owner tags the commit and runs `gh workflow run release.yml --ref v<x.y.z>` |

## 3. The twin tables, explained

The Svelte app stores things the Streamlit app never kept:
- the hourly twin snapshots (coke rate by model, ω, alarms);
- optimiser runs and the accept / reject decisions;
- job history and the audit log;
- shift handover records;
- fact-register edits, operator inputs, FurnaceMind conversations and saved what-if cases.

They live in their own PostgreSQL schema, `twin`. Nine migrations in `alembic/twin/versions/` (`twin_0001` to
`twin_0009`) create it. The migrations only add the `twin` schema and its tables; they never change existing tables.
The schema's own version table is `twin.alembic_version`.

The app finds the twin tables at `BF_TWIN_DB_URL`, else at `DATABASE_URL`. `BF_TWIN_DB_URL=off` disables them; the app
still starts, but the Live Twin, reports, plan card, admin pages and FurnaceMind history are then mostly empty.

**State on 30 Oct 2026:** the database in the current `DATABASE_URL` has **no `twin` schema** (checked read-only). None
of the nine migrations has been applied there.

The two options for the Jetson:

| Option | Setting | Note |
|---|---|---|
| A. Same server as the plant data | leave `BF_TWIN_DB_URL` unset | `release.sh` creates the `twin` schema next to the existing ones on the first release |
| B. A separate database (e.g. a Neon project or a database on the Jetson) | `BF_TWIN_DB_URL=postgresql://...` in `/etc/bf-webapp/env` | The plant database stays untouched. A database on the Jetson also works, but then the twin history exists only on the device: back it up |

**Only one worker per twin database.** If two machines run the twin-worker against the same twin tables, every
scheduled job runs twice (optimiser runs, twin evaluation, calibration, reports). A test PC must use its own twin
database (section 4).

## 4. Try it on a PC first (recommended)

This runs the full stack from a checkout, with its own twin database, so nothing is written to the Jetson's tables.

**Needs:** Python 3.12 and uv, Node.js 20 or later with pnpm 10, a local PostgreSQL (or any empty PostgreSQL database),
and Docker or WSL for Redis. Get the project `.env` from the owner over a private channel and put it in the repository
root (it holds the database URL and API keys; never commit it).

Windows PowerShell, from the repository root:

```powershell
git switch feature/262_svelte_twin
uv sync
pnpm --dir frontend install --frozen-lockfile

# Own twin database (once): an empty database, then the migrations
createdb -h localhost -U postgres bf_twin            # "C:\Program Files\PostgreSQL\18\bin\createdb.exe" on Windows
$env:BF_TWIN_DB_URL = "postgresql://postgres:<password>@localhost:5432/bf_twin"
uv run alembic -n twin upgrade head                  # again after every pull

# Redis (job buttons and live updates)
docker run -d --name bf-redis -p 6379:6379 redis:7   # first time; later: docker start bf-redis

# Start (SPA 5173, API 8100, worker, Streamlit 8501 under /legacy); Ctrl+C stops all
$env:BF_TWIN_DB_URL = "postgresql://postgres:<password>@localhost:5432/bf_twin"
$env:BF_WORKER_SKIP_SCHEDULES = "optimiser.,calibration.,learn."
uv run python scripts/dev.py
```

Open http://localhost:5173 and sign in with a normal app account; the admin pages need the admin role. On Linux, use
`export BF_TWIN_DB_URL=...` instead of `$env:...`, and `redis-server` from the distribution instead of Docker.

- `$env:` settings last for that terminal only.
- `BF_WORKER_SKIP_SCHEDULES` keeps the scheduled STRATA-ω optimiser, calibration and STRATA-Learn jobs off the PC (each
  can use the CPU for up to an hour); "Run now" still works.
- Without Redis the script leaves the worker out: pages that read data work, but job buttons answer 503 and the Live
  Twin stays empty.
- The Live Twin fills after the worker's first `twin.evaluate` (hourly at HH:10 IST; it backfills 24 h).
- **`/admin/users` (users, hoppers and burden tabs) writes to the real `identity` and `ops_config` tables** in
  `DATABASE_URL`, exactly as the Welcome-page admin tools of Streamlit do. Save only real changes there. Other edits
  (operator inputs, facts, decisions, what-if cases) go to the local twin database.

## 5. Jetson installation, in order

Each step is idempotent unless noted; details in the README section named in the last column.

| Step | What | README section |
|---|---|---|
| 1 | Flash JetPack 6.x (L4T R36, Ubuntu 22.04) with SDK Manager. Partition and format the NVMe by hand (destructive) and note its UUID | Base system, "Before the first run" |
| 2 | Clone the repository onto the Jetson for the scripts: `git clone https://github.com/saikumar-voptimai/evonith_bf_webapp.git ~/bf-webapp && git -C ~/bf-webapp switch feature/262_svelte_twin` | |
| 3 | `sudo deploy/jetson/provision.sh --nvme-uuid <UUID> --ntp-server <NTP> --admin-subnet <CIDR>`: user `bfapp`, NVMe mount, MAXN power mode, chrony, logrotate, Caddy, Redis, the `bf-*` units, firewall | Base system; Services |
| 4 | `sudo deploy/jetson/verify_base.sh --admin-subnet <CIDR>`, `sudo reboot`, run it again: the second run must report `PASS reboot` | Base system, "Checks" |
| 5 | `sudo deploy/jetson/install_python.sh`: uv 0.7.13 and CPython 3.12 (never JetPack's system Python) | Python and uv |
| 6 | Fill `/etc/bf-webapp/env` (root:bfapp 0640, created by step 3 with the JWT secret): `DATABASE_URL`, `INFLUX_ONLINE_TOKEN`, `INFLUX_OFFLINE_TOKEN`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`; `BF_SITE` and `BF_TLS` (decision 2); `BF_TWIN_DB_URL` if decision 1 is option B. `KEY=value` lines, no `export`, no quotes needed | Base system, path table |
| 7 | Release build: the owner tags the commit and runs `gh workflow run release.yml --ref v<x.y.z>` (decision 6) | Release artefacts |
| 8 | `sudo --preserve-env=GH_TOKEN deploy/jetson/release.sh v<x.y.z>`: fetch and verify, unpack, `uv sync --frozen`, **twin migrations**, switch, restart, smoke test, automatic rollback on failure. Without GitHub on the Jetson: download the five files on a connected PC and pass `--artefacts <dir>` | Release and update |
| 9 | With `BF_TLS=internal`: install Caddy's root certificate (`/var/lib/bf-webapp/caddy/data/caddy/pki/authorities/local/root.crt`) on every control-room PC (Windows: Trusted Root Certification Authorities). Leave `BF_HSTS` at `max-age=0` until every client trusts it | Reverse proxy, "Certificate" |
| 10 | Open `https://<site>/` from a control-room PC, sign in, check the footer shows the release tag, open a `/legacy` page | |

Later updates: `sudo /opt/bf-webapp/current/deploy/jetson/release.sh v<new>`. Back to the previous release:
`sudo /opt/bf-webapp/current/deploy/jetson/rollback.sh`. A rollback never downgrades the twin schema.

Useful commands:

```bash
systemctl list-dependencies bf-webapp.target          # every unit and its state
sudo systemctl restart bf-webapp.target               # all services
journalctl -u bf-twin-api -u bf-twin-worker -f        # logs (14 days kept)
journalctl -u bf-healthcheck -p warning               # failed probes and restarts
cat /opt/bf-webapp/release.log                        # updates and rollbacks with their durations
```

## 6. Acceptance checks to record on the device

Post the results as a comment on the issue in the last column.

| Check | How | Target | Issue |
|---|---|---|---|
| Clock | `verify_base.sh` (clock line) | offset < 100 ms | #315 |
| Services after a reboot | `verify_base.sh` after `reboot` | `PASS reboot` | #315 |
| Open ports | `nmap` from a control-room PC and from the admin subnet (README, "External port scan") | only 80 / 443 from outside; 22 only from the admin subnet; 6379, 8100, 8501 filtered | #315 |
| Test suite | as the checkout owner: `uv run --no-sync pytest -q tests/` | all pass (Redis, Caddy and POSIX-only tests, skipped on Windows, run here) | #316 |
| Optimiser runtimes | `deploy/jetson/benchmark.sh --open-issue` | solve 2.8 s, Jacobian 6 s, Step 2 90 s, max production 300 s, **Step 1 600 s** (not yet met on a laptop) | #316, #288 |
| Restart after a kill | `sudo systemctl kill -s KILL <unit>` for each of the six services, then the health URL | serving again within 30 s | #317 |
| Boot to serving | `systemd-analyze critical-chain bf-caddy.service`, then the first 200 from `https://<site>/api/health` | < 3 min | #317 |
| Update and rollback | `release.sh` to a second tag, then `rollback.sh`; seconds in `release.log` | each < 5 min | #318 |
| Failed smoke test | covered on a fake host by `tests/test_jetson_release.py`; on the device only if a failing release is at hand | rolls back without a manual step | #318 |
| Outages | stop network access to the twin database for an hour; block InfluxDB | no hourly snapshot lost after recovery; stale badges within 1 min | #319 |
| Long-running page | Live Twin open 8 h on a control-room PC (browser task manager) | < 50 MB memory growth | #289 |

## 7. Rules and gotchas

1. **Twin tables before the first release** (section 2, decision 1): `release.sh` migrates whatever URL is configured.
2. **One worker per twin database** (section 3). Stop any test worker that points at the Jetson's twin tables.
3. **GitHub Actions run only by hand.** Don't run a workflow or add triggers without the owner's agreement (minutes).
4. **`/admin/users` writes to the plant database** (users, hoppers, burden distribution), as the Streamlit tools do.
5. **Two twin-api workers:** the sign-in failure limit and FurnaceMind's fetched data are per worker. Set `--workers 1`
   in `bf-twin-api.service` if a follow-up FurnaceMind question losing its data matters more than the second worker.
6. **HSTS** (`BF_HSTS=max-age=31536000`) only after every client trusts the certificate; a browser then offers no way
   past a certificate warning.
7. The app's state lives on the NVMe (`/var/lib/bf-webapp`); without the NVMe the services stay down on purpose.

## 8. Known open items

| Item | Effect | Issue |
|---|---|---|
| `/api/health/deep` does not exist yet | The smoke test accepts its 404 | #318 |
| No free-space check before a release download; size of a synced `.venv` on the device not measured | Watch disk space on the first releases | #318 |
| Release directories are owned by `bfapp` (code and `.venv`) | Hardening option for later | #318 |
| Outage handling is partial: optimiser, calibration, learn and report jobs don't check source health; optimiser-run and report writes are not buffered | Those jobs fail (and retry) during an outage instead of skipping | #319 |
| tegrastats parsing checked on sample lines only | Check the host-health tiles in `/admin/jobs` on the device | #311 |
| Ten pages were UI-audited only in their error state (no sample data in the audit) | Look at them with real data once on the device | #274 |
| The Streamlit admin tools store the operator's IST time as UTC (5 h 30 min off); the Svelte admin converts correctly | Existing Streamlit behaviour | #310 |
| Retirement of Streamlit pages: gates, 14 days of shadow use and shift-team sign-off per page | No page is retired yet | #321 |

Each ticket's issue comment on GitHub (#270 to #321) lists its own pending checks under "pending: device / plant data /
sign-off".

## 9. Reference

| File | Content |
|---|---|
| `deploy/jetson/README.md` | Every script, option, unit and check in detail |
| `deploy/jetson/provision.sh`, `verify_base.sh` | Base system and its check |
| `deploy/jetson/install_python.sh`, `benchmark.sh` | Python / uv and the optimiser benchmark |
| `deploy/jetson/release.sh`, `rollback.sh` | Update and rollback |
| `deploy/jetson/Caddyfile`, `deploy/jetson/systemd/` | Reverse proxy and service units |
| `docs/jetson_wheel_audit.md`, `docs/jetson_benchmark.md` | aarch64 package audit; benchmark table (filled on the device) |
| `docs/retirement.md` | Streamlit page retirement list and gates |
| `CLAUDE.md` section 12 | Architecture of the Svelte app, twin-api, worker and twin store |
