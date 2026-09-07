# Scheduled Tasks: complete Jetson deployment and operations runbook

This runbook installs the database-backed Scheduled Tasks system on an NVIDIA
Jetson, connects it to the Streamlit application, installs its systemd units,
and verifies the complete UI-to-FurnaceMind workflow.

The production design is:

```text
Streamlit Cloud
    -> PostgreSQL job definition and control command
    -> Jetson control-sync timer
    -> per-job systemd timer
    -> shared headless runner
    -> restricted FurnaceMind execution
    -> PostgreSQL run/output history
    -> Streamlit "Your tasks" view
```

There is no continuously running database-polling scheduler daemon. The only
periodic control process is a short-lived systemd service that checks a bounded
batch of lifecycle commands and exits. Each scheduled job has one timer and all
jobs share the same `furnacemind-job@.service` template.

## 1. Current production scope

Deploy only the capabilities that the headless executor currently supports:

- Target device: `bf2-jetson-01`.
- Delivery: `in_app` only. Results are stored in PostgreSQL.
- Data: online process data, plus historical furnace data for non-ETA reports.
- ETA CO reports: online process data only and `include_graph=false`.
- Output: bounded text and JSON. Scheduled image/plot artifacts are not yet
  implemented.
- Analysis: deterministic `none`, or OpenRouter-backed `low`, `medium`, and
  `high` reasoning levels.
- Lifecycle: create, activate, pause, resume, edit with immutable revisions,
  clone, and archive.
- Manual **Run now**, permanent deletion, email, WhatsApp, Telegram, and failure
  notification sending are not implemented.

Portable JSON can describe some future delivery and graph options, but the
Jetson rejects those options during production validation. Do not expose them
to operators until their adapters are implemented.

At the time of this runbook, the catalog/schema and Streamlit form still contain
some of those future choices. Before production, hide or disable unsupported
choices in the UI, or add an equally clear creation-time production-policy
validation. Relying on a later Jetson activation failure is not an acceptable
operator experience.

## 2. Deployment responsibilities

Use three separate security boundaries:

| Component | Location | Responsibility |
|---|---|---|
| Streamlit web app | Streamlit Cloud | Authenticate the operator, validate input, store definitions, and enqueue lifecycle commands |
| PostgreSQL | Shared managed/plant database | Store jobs, revisions, commands, runs, logs, and outputs |
| Jetson | Plant network | Consume commands, manage systemd, fetch furnace data, run FurnaceMind, and store results |

Both Streamlit Cloud and the Jetson must connect to the same PostgreSQL
database. Streamlit Cloud does not need network access to systemd or an inbound
connection to the Jetson. The Jetson does not need to expose an HTTP API.

## 3. Values to prepare before installation

Replace the following placeholders throughout this guide:

| Placeholder | Example meaning |
|---|---|
| `<repository-url>` | SSH or HTTPS URL of the private repository |
| `<release-ref>` | Approved Git tag or commit SHA; do not deploy an unreviewed moving branch |
| `<new-release-ref>` | Approved tag or commit SHA used during a later upgrade |
| `<deployment-user>` | Trusted Linux administrator account that owns deployed application files |
| `<database-url>` | SQLAlchemy PostgreSQL URL used by the application runtime |
| `<migration-database-url>` | Optional higher-privilege URL used only for Alembic |
| `<openrouter-api-key>` | OpenRouter key used by model-enabled scheduled tasks |
| `<influx-online-token>` | Token used to read BF2 online process data |
| `<job-uuid>` | Canonical lowercase UUID shown for a saved task |

Before starting, confirm:

- The Jetson runs a supported Ubuntu/JetPack image with systemd as PID 1.
- The Jetson can reach PostgreSQL, InfluxDB, DNS, and the selected model
  provider over the required outbound ports.
- PostgreSQL accepts connections from both the Jetson and Streamlit Cloud.
- A database backup or provider restore point exists before migrations.
- You have `sudo` access on the Jetson and permission to deploy the private
  repository.
- The repository has been reviewed and the Scheduled Tasks tests pass for the
  exact release being installed.

## 4. Prepare the Jetson operating system

Run these commands from an administrator account on the Jetson:

```bash
uname -m
cat /etc/os-release
ps -p 1 -o comm=
timedatectl status
```

Expected results:

- Architecture is normally `aarch64` on Jetson.
- PID 1 reports `systemd`.
- The clock is synchronized. The host timezone may be UTC; each task stores its
  own IANA timezone.

Enable network time synchronization if required:

```bash
sudo timedatectl set-ntp true
```

Install the base deployment packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  git \
  curl \
  ca-certificates \
  build-essential \
  pkg-config \
  libpq-dev \
  rsync
```

The scheduler needs outbound connectivity only:

- PostgreSQL, normally TCP 5432 or the managed provider's configured port.
- InfluxDB and OpenRouter, normally HTTPS/TCP 443.
- DNS and NTP according to the plant network policy.

Do not open an inbound scheduler or FurnaceMind API port. Keep SSH restricted to
approved administration sources.

## 5. Create the unprivileged runtime identity

The job runner must not run as root. Create one system user and group:

```bash
getent group furnacemind >/dev/null || sudo groupadd --system furnacemind
id -u furnacemind >/dev/null 2>&1 || sudo useradd \
    --system \
    --gid furnacemind \
    --home-dir /var/lib/furnacemind \
    --create-home \
    --shell /usr/sbin/nologin \
    furnacemind
```

If either already exists, inspect it instead of recreating it:

```bash
getent passwd furnacemind
getent group furnacemind
id furnacemind
```

The account and its primary group must both resolve to non-root IDs.

Create the application and secret directories. Replace `<deployment-user>` with
the administrator account that will update releases:

```bash
sudo install -d -o <deployment-user> -g furnacemind -m 2750 /opt/furnacemind
sudo install -d -o <deployment-user> -g furnacemind -m 2750 /opt/furnacemind-runtime
sudo install -d -o root -g furnacemind -m 0750 /etc/furnacemind
sudo install -d -o root -g root -m 0750 /run/lock/furnacemind-scheduled-tasks
```

The `/run/lock` directory is temporary and may disappear at reboot. The
root-owned control service safely recreates it when needed.

## 6. Put the repository on the Jetson

Use an SSH deploy key, an approved credential helper, or a release artifact.
Never put a personal access token directly in the clone URL or commit a secret
to the repository.

For an empty `/opt/furnacemind` directory:

```bash
git clone <repository-url> /opt/furnacemind
cd /opt/furnacemind
git fetch --tags --prune
git checkout --detach <release-ref>
git status --short
git rev-parse HEAD
test ! -f /opt/furnacemind/.env
```

`git status --short` should be empty. Record the printed commit SHA in the
deployment record. The repository `.env` file must not be deployed; privileged
scripts deliberately use `/etc/furnacemind/furnacemind.env` instead.

If the repository is delivered as an artifact, extract it into a temporary
release directory, verify its checksum, and use `rsync` to place the reviewed
contents under `/opt/furnacemind`.

## 7. Install `uv`, Python, and project dependencies

The root project requires Python `3.12` as declared in `.python-version` and
`pyproject.toml`. Install `uv` using your organization's approved package
process. One common standalone installation is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv
uv --version
```

If `uv` was installed only in the administrator's local bin directory, install
that verified executable at a stable system path:

```bash
sudo install -o root -g root -m 0755 "$(command -v uv)" /usr/local/bin/uv
/usr/local/bin/uv --version
```

Create the project environment from the committed lock file:

```bash
cd /opt/furnacemind
export UV_PYTHON_INSTALL_DIR=/opt/furnacemind-runtime/python
export UV_CACHE_DIR=/opt/furnacemind-runtime/uv-cache
/usr/local/bin/uv python install 3.12
PYTHON_BIN="$(/usr/local/bin/uv python find 3.12)"
/usr/local/bin/uv sync --frozen --python "$PYTHON_BIN"
/opt/furnacemind/.venv/bin/python --version
```

The managed interpreter is deliberately installed under `/opt`, not under the
deployment administrator's home directory. The generated systemd service uses
`ProtectHome=true`; a virtual environment that links into an administrator's
home can work interactively and then fail under systemd.

The first deployment includes development dependencies so the focused test
suite can run. After validation, an organization may use
`uv sync --frozen --no-dev`, provided the resulting environment is tested again.

### Jetson ARM64 dependency gate

The repository contains compiled dependencies, including numerical and ML
packages. A lock that installs on Windows or x86 Linux is not proof that all
wheels work on Jetson ARM64. Treat `uv sync --frozen` as a deployment gate.

If a package such as PyTorch requires an NVIDIA JetPack-specific wheel:

1. Stop the deployment.
2. Select the wheel/version supported by the installed JetPack release.
3. Record the platform-specific dependency in a reviewed lock or deployment
   constraint.
4. Re-run the complete focused test suite.

Do not silently run `uv sync` without `--frozen` or replace locked production
packages directly on the device.

## 8. Create the protected service environment

Create the file first with its final owner and mode. The check prevents a repeat
installation from truncating an existing secret file and refuses an existing
symlink or non-regular file:

```bash
if sudo test -L /etc/furnacemind/furnacemind.env; then
  echo 'Unsafe FurnaceMind environment symlink'
  exit 1
elif sudo test -e /etc/furnacemind/furnacemind.env; then
  sudo test -f /etc/furnacemind/furnacemind.env || \
    { echo 'Unsafe non-regular FurnaceMind environment path'; exit 1; }
else
  sudo install \
    -o root \
    -g furnacemind \
    -m 0640 \
    /dev/null \
    /etc/furnacemind/furnacemind.env
fi
sudo chown root:furnacemind /etc/furnacemind/furnacemind.env
sudo chmod 0640 /etc/furnacemind/furnacemind.env
sudoedit /etc/furnacemind/furnacemind.env
```

Use this template. Quote values containing special characters and replace all
placeholders. Do not copy the example password or keys.

```dotenv
# Shared PostgreSQL database. URL-encode username/password characters.
DATABASE_URL="<database-url>"

# Online BF2 historian. Required for online_process_data jobs.
INFLUX_ONLINE_TOKEN="<influx-online-token>"

# Current scheduled LLM executor uses OpenRouter for low/medium/high analysis.
# Deterministic no-AI jobs do not make a model request.
LLM_PROVIDER="openrouter"
OPENROUTER_API_KEY="<openrouter-api-key>"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
LLM_MAX_TOKENS="800"

# Optional model overrides. Omit these to use the reviewed code defaults.
# OPENROUTER_LOW_MODEL="provider/model"
# OPENROUTER_MEDIUM_MODEL="provider/model"
# OPENROUTER_HIGH_MODEL="provider/model"
# OPENROUTER_LOW_REASONING_EFFORT="low"
# OPENROUTER_MEDIUM_REASONING_EFFORT="medium"
# OPENROUTER_HIGH_REASONING_EFFORT="high"

# Keep both gates false until Sections 9-12 pass.
SCHEDULED_TASKS_SYSTEMD_ENABLED="false"
SCHEDULED_TASKS_EXECUTOR_READY="false"

# Logical device identity. It must match src/config/scheduled_jobs.yml.
SCHEDULED_TASKS_TARGET_DEVICE_ID="bf2-jetson-01"

# Explicit deployment paths.
SCHEDULED_TASKS_SYSTEMD_UNIT_DIRECTORY="/etc/systemd/system"
SCHEDULED_TASKS_WORKING_DIRECTORY="/opt/furnacemind"
SCHEDULED_TASKS_PYTHON_EXECUTABLE="/opt/furnacemind/.venv/bin/python"
SCHEDULED_TASKS_RUNNER_SCRIPT="/opt/furnacemind/scripts/furnacemind_job_runner.py"
SCHEDULED_TASKS_CONTROL_SYNC_SCRIPT="/opt/furnacemind/scripts/furnacemind_control_sync.py"
SCHEDULED_TASKS_ENVIRONMENT_FILE="/etc/furnacemind/furnacemind.env"
SCHEDULED_TASKS_SYSTEMCTL_PATH="/usr/bin/systemctl"
SCHEDULED_TASKS_SYSTEMD_ANALYZE_PATH="/usr/bin/systemd-analyze"
SCHEDULED_TASKS_SERVICE_USER="furnacemind"
SCHEDULED_TASKS_SERVICE_GROUP="furnacemind"
SCHEDULED_TASKS_LOCK_DIRECTORY="/run/lock/furnacemind-scheduled-tasks"

# Bounded command-queue behavior.
SCHEDULED_TASKS_CONTROL_SYNC_INTERVAL_SECONDS="30"
SCHEDULED_TASKS_COMMAND_LEASE_SECONDS="300"
SCHEDULED_TASKS_COMMAND_RETRY_SECONDS="30"
```

Qdrant, embedding, OpenAI, offline Influx, and document-search credentials are
not required by the current restricted scheduled executor. Add them only if a
separately reviewed scheduled capability starts using them.

Verify the secret file without printing its contents:

```bash
sudo stat -c '%U %G %a %n' /etc/furnacemind/furnacemind.env
sudo test ! -L /etc/furnacemind/furnacemind.env
sudo -u furnacemind test -r /etc/furnacemind/furnacemind.env
```

Expected ownership and mode are `root furnacemind 640`. The file must not be a
symlink, group-writable, executable, or accessible by other users.

Never run `set -x`, `env`, or another command that prints the service
environment while secrets are loaded.

## 9. Set safe repository permissions

The runtime user needs read access to the application and execute access to the
virtual environment, but it should not be able to modify application code:

```bash
sudo chgrp -R furnacemind /opt/furnacemind
sudo chgrp -R furnacemind /opt/furnacemind-runtime
sudo chmod -R g-w,o-rwx /opt/furnacemind
sudo chmod -R g-w,o-rwx /opt/furnacemind-runtime
sudo chmod -R g+rX /opt/furnacemind
sudo chmod -R g+rX /opt/furnacemind-runtime
sudo -u furnacemind test -r /opt/furnacemind/scripts/furnacemind_job_runner.py
sudo -u furnacemind test -x /opt/furnacemind/.venv/bin/python
sudo -u furnacemind /opt/furnacemind/.venv/bin/python --version
```

Keep the repository owned by the deployment administrator or root. Do not make
application source files group-writable by `furnacemind`.

### Historical dataset preparation

Non-ETA jobs using `historical_furnace_data` read the configured static dataset.
The current configuration points to `src/data/V14_df_filtered.csv` and may build
that cache from the configured publisher/database if it is absent. A locked-down
service account must not create Python-package files at runtime.

If historical scheduled jobs are required, prepare and verify the cache during
deployment, before locking the final file permissions:

```bash
cd /opt/furnacemind
sudo -i
set -a
. /etc/furnacemind/furnacemind.env
set +a
/opt/furnacemind/.venv/bin/python - <<'PY'
from data.ml.static_csv import load_static_dataset

frame = load_static_dataset()
if frame.empty:
    raise SystemExit("Historical furnace dataset is empty")
print(f"Historical dataset ready: rows={len(frame)} columns={len(frame.columns)}")
PY
exit
sudo chgrp -R furnacemind /opt/furnacemind/src/data
sudo chmod -R g-w,o-rwx /opt/furnacemind/src/data
sudo chmod -R g+rX /opt/furnacemind/src/data
```

Arrange an explicit maintenance process to refresh that cache. If no validated
refresh process exists, restrict operators to `online_process_data`; otherwise a
historical report can be valid but stale.

## 10. Verify PostgreSQL connectivity

The database must be a shared PostgreSQL database, not SQLite. It must contain
the existing identity records referenced by scheduled-job ownership.

Open a temporary root administration shell. The root shell is used only for
deployment; Streamlit and job execution do not run as root.

```bash
sudo -i
set -a
. /etc/furnacemind/furnacemind.env
set +a
cd /opt/furnacemind
```

Test the runtime database URL:

```bash
/opt/furnacemind/.venv/bin/python - <<'PY'
from sqlalchemy import text
from furnace_data.relational.engine import build_relational_engine

engine = build_relational_engine()
try:
    with engine.connect() as connection:
        database, user = connection.execute(
            text("select current_database(), current_user")
        ).one()
        print(f"PostgreSQL reachable: database={database} user={user}")
finally:
    engine.dispose()
PY
```

If this fails, fix DNS, firewall/allow-list, TLS, credentials, or PostgreSQL
permissions before continuing. Managed providers commonly require
`sslmode=require` in the connection URL.

Use a least-privilege runtime database role. Alembic may require a separate
migration role with DDL permission. If so, load the migration URL only for the
migration command and do not save it in `furnacemind.env`.

At a minimum, the application runtime roles need the following access. Express
the grants through the organization's normal database-management process rather
than copying ad hoc ownership changes into production:

| Runtime | Required database access |
|---|---|
| Streamlit | Read authenticated user identity; create/read owner-scoped jobs and revisions; enqueue/read owner-scoped commands; read run and output history |
| Jetson control sync | Read/update target-device commands; read/update jobs; append revisions during edits |
| Jetson job runner | Read active definitions; create/update runs and leases; append run logs and outputs; update terminal one-time job state |
| Alembic migration role | Create/alter/drop reviewed objects and indexes in the required schemas for the duration of migration only |

Historical furnace jobs also require read access to their configured source
tables. Do not give the headless runtime ownership of the database schemas.

## 11. Back up and apply database migrations

Create a provider snapshot or approved database backup before changing the
schema. Then inspect the current revision:

```bash
/opt/furnacemind/.venv/bin/python -m alembic heads
/opt/furnacemind/.venv/bin/python -m alembic current
```

Upgrade the database:

```bash
/opt/furnacemind/.venv/bin/python -m alembic upgrade head
/opt/furnacemind/.venv/bin/python -m alembic current
```

If migrations use a separate DDL-capable role, read it without echoing it or
placing it in shell history, use it only for Alembic, and then discard it:

```bash
read -rsp 'Migration DATABASE_URL: ' MIGRATION_DATABASE_URL
echo
DATABASE_URL="$MIGRATION_DATABASE_URL" \
  /opt/furnacemind/.venv/bin/python -m alembic upgrade head
unset MIGRATION_DATABASE_URL
/opt/furnacemind/.venv/bin/python -m alembic current
```

For this release, the expected head is:

```text
20260907_0010
```

Verify the scheduling tables:

```bash
/opt/furnacemind/.venv/bin/python - <<'PY'
from sqlalchemy import inspect
from furnace_data.relational.engine import build_relational_engine

expected = {
    "scheduled_jobs",
    "scheduled_job_revisions",
    "scheduled_job_commands",
    "job_runs",
    "job_run_logs",
    "job_outputs",
}
engine = build_relational_engine()
try:
    actual = set(inspect(engine).get_table_names(schema="automation"))
finally:
    engine.dispose()
missing = sorted(expected - actual)
if missing:
    raise SystemExit(f"Missing automation tables: {missing}")
print("Scheduled-task tables are present.")
PY
```

Remain in the root shell until Section 15 is complete. If you leave it, load the
protected environment again before running a provisioning command.

## 12. Run pre-deployment validation

First verify imports and command entry points:

```bash
/opt/furnacemind/.venv/bin/python -m compileall -q src scripts furnace_data/furnace_data
/opt/furnacemind/.venv/bin/python scripts/furnacemind_job_runner.py --help
/opt/furnacemind/.venv/bin/python scripts/furnacemind_control_sync.py --help
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py --help
```

Run the focused scheduling tests while development dependencies are installed:

```bash
env \
  -u DATABASE_URL \
  -u NEON_DATABASE_URL \
  -u NEON_STR \
  -u INFLUX_ONLINE_TOKEN \
  -u INFLUX_OFFLINE_TOKEN \
  -u OPENROUTER_API_KEY \
  OPENAI_API_KEY=unit-test-placeholder \
  LLM_PROVIDER=openai \
  /opt/furnacemind/.venv/bin/python -m pytest -q \
  tests/test_scheduled_* \
  tests/test_furnacemind_runtime_state.py \
  tests/test_furnacemind_graph.py \
  tests/test_page_registry.py \
  tests/test_session_auth.py
```

There must be no failures in the scheduled-task test set. The command removes
production database/data/model credentials from the test process and supplies a
non-working placeholder only to satisfy import-time configuration validation.
Do not run unit tests against the production database.

Render the shared units without modifying the host:

```bash
/opt/furnacemind/.venv/bin/python \
  scripts/furnacemind_systemd_provisioner.py \
  install-service --dry-run

/opt/furnacemind/.venv/bin/python \
  scripts/furnacemind_systemd_provisioner.py \
  install-control-sync --dry-run
```

Confirm the output contains only trusted absolute paths and the expected unit
names:

- `furnacemind-job@.service`
- `furnacemind-control-sync.service`
- `furnacemind-control-sync.timer`

Dry-run is safe while the two deployment gates remain false. Do not continue if
the rendered working directory, Python executable, scripts, service identity,
environment file, or systemd directory is unexpected.

## 13. Validate data and model access

At minimum, verify that DNS and TLS access are available from the Jetson. The
most reliable functional test is a temporary scheduled job created through the
UI in Section 18, because it exercises the same bounded data adapter used in
production.

For model-enabled tasks, `OPENROUTER_API_KEY` must be present and the configured
Low/Medium/High model identifiers must be available to that account. A job using
**No AI analysis** still fetches data but does not call OpenRouter.

For online tasks, `INFLUX_ONLINE_TOKEN` must be authorized for the configured
BF2 database and measurement fields. For historical tasks, verify the cache as
described in Section 9.

## 14. Enable the production safety gates

Only after the database, files, imports, tests, data access, and dry-run checks
have passed, edit the protected environment:

```bash
sudoedit /etc/furnacemind/furnacemind.env
```

Change exactly these values:

```dotenv
SCHEDULED_TASKS_SYSTEMD_ENABLED="true"
SCHEDULED_TASKS_EXECUTOR_READY="true"
```

Reload the values into the root deployment shell:

```bash
set -a
. /etc/furnacemind/furnacemind.env
set +a
```

These flags are independent approvals:

- `SCHEDULED_TASKS_SYSTEMD_ENABLED` permits managed systemd changes.
- `SCHEDULED_TASKS_EXECUTOR_READY` confirms the Jetson runtime, credentials,
  data adapters, and FurnaceMind execution path have been validated.

## 15. Install the systemd components

Install the shared job service template:

```bash
/opt/furnacemind/.venv/bin/python \
  scripts/furnacemind_systemd_provisioner.py \
  install-service
```

Install and start the Streamlit-to-Jetson control bridge:

```bash
/opt/furnacemind/.venv/bin/python \
  scripts/furnacemind_systemd_provisioner.py \
  install-control-sync
```

The second command installs both the one-shot control service and its timer,
enables the timer at boot, and starts it. It does not create a permanently
running scheduler process.

Leave the root shell after installation:

```bash
exit
```

## 16. Verify the installed systemd state

Check the exact managed files:

```bash
sudo ls -l \
  /etc/systemd/system/furnacemind-job@.service \
  /etc/systemd/system/furnacemind-control-sync.service \
  /etc/systemd/system/furnacemind-control-sync.timer
```

Inspect and verify them:

```bash
sudo systemctl cat furnacemind-job@.service
sudo systemctl cat furnacemind-control-sync.service
sudo systemctl cat furnacemind-control-sync.timer
sudo systemd-analyze verify \
  /etc/systemd/system/furnacemind-job@.service \
  /etc/systemd/system/furnacemind-control-sync.service \
  /etc/systemd/system/furnacemind-control-sync.timer
```

Check the control timer:

```bash
sudo systemctl is-enabled furnacemind-control-sync.timer
sudo systemctl is-active furnacemind-control-sync.timer
sudo systemctl status furnacemind-control-sync.timer --no-pager
sudo systemctl list-timers --all 'furnacemind-*'
```

Expected state is `enabled` and `active` for the control-sync timer. The
one-shot control service is normally `inactive (dead)` between successful runs;
that is correct.

Force one bounded control check and inspect its logs:

```bash
sudo systemctl start furnacemind-control-sync.service
sudo journalctl -u furnacemind-control-sync.service -n 100 --no-pager
```

With an empty command queue, a successful run simply processes zero commands.

## 17. Configure Streamlit Cloud

Deploy the same reviewed repository release to Streamlit Cloud. Streamlit Cloud
runs the UI only; it must never attempt to install or control systemd.

Configure the application's normal authentication secrets and add the same
runtime PostgreSQL URL as a root-level Streamlit secret/environment value:

```toml
DATABASE_URL = "<database-url>"
```

Also configure any existing application secrets required for login. The web app
does not need the Jetson's root-controlled environment file, systemd flags,
Influx token, or OpenRouter key merely to create and manage scheduled task
definitions.

Verify that:

- Streamlit Cloud can read and write the `automation` scheduling tables.
- The signed-in application user exists in the shared `identity.users` table.
- The Jetson uses `SCHEDULED_TASKS_TARGET_DEVICE_ID=bf2-jetson-01`.
- Only one production Jetson consumes commands for that logical device ID.

If the database uses an IP allow-list, account for the connectivity model of
both environments. Do not weaken the database to unrestricted public access;
use TLS, strong credentials, and the provider's approved networking controls.

## 18. Complete end-to-end acceptance test

Perform this test before operators create long-running production schedules.

### 18.1 Create a safe test task

In Streamlit:

1. Open **Scheduled Tasks**.
2. Create an in-app task using `online_process_data`.
3. Select **No AI analysis** for the first test. This isolates scheduling and
   data access from the model provider.
4. Choose a one-time execution several minutes in the future.
5. Save the task.
6. Open **Your tasks** and copy the full job UUID.

The initial database/UI status should be waiting for activation or
`pending_provisioning`.

### 18.2 Validate before activation

On the Jetson, load the protected environment in a root shell and run:

```bash
sudo -i
set -a
. /etc/furnacemind/furnacemind.env
set +a
cd /opt/furnacemind
/opt/furnacemind/.venv/bin/python \
  scripts/furnacemind_job_runner.py \
  --job-id <job-uuid> \
  --validate-only
exit
```

This checks the stored JSON and production execution policy. It does not claim
or execute an occurrence.

### 18.3 Activate through the real cloud-to-Jetson bridge

In Streamlit, click **Activate**. Do not manually provision the task for this
acceptance test; the purpose is to validate the database command bridge.

Within approximately one control interval, inspect:

```bash
sudo journalctl -u furnacemind-control-sync.service -n 100 --no-pager
sudo systemctl list-timers --all 'furnacemind-job-*'
sudo systemctl status \
  furnacemind-job-<job-uuid>.timer \
  --no-pager
```

Expected results:

- The command becomes `succeeded`.
- The task becomes `active`.
- `/etc/systemd/system/furnacemind-job-<job-uuid>.timer` exists.
- The timer is enabled and active.
- The next trigger time matches the task timezone and schedule.

### 18.4 Verify execution and stored output

After the scheduled time:

```bash
sudo systemctl status \
  furnacemind-job@<job-uuid>.service \
  --no-pager
sudo journalctl \
  -u furnacemind-job@<job-uuid>.service \
  -n 200 \
  --no-pager
```

Return to **Your tasks** in Streamlit and confirm:

- A completed run appears in **Run history**.
- The scheduled time and attempt number are correct.
- The in-app text/JSON output is available.
- The output does not expose credentials, raw model arguments, or unrestricted
  plant data.

Repeat the acceptance test with **Brief analysis** to verify OpenRouter after
the deterministic path passes.

Finally test **Pause**, **Resume**, **Edit**, and **Archive**. For each action,
confirm that the command succeeds in the UI and that the corresponding timer
state changes on the Jetson.

## 19. Normal operator lifecycle

After installation, normal operators use only Streamlit:

| Operator action | Database command | Jetson result |
|---|---|---|
| Create | No systemd command yet | Definition and initial revision are stored |
| Activate | `provision` | UUID timer is created, enabled, and started |
| Pause | `pause` | Job is made non-runnable; timer is stopped and disabled |
| Resume | `resume` | Timer is reinstalled and activated with a new execution generation |
| Edit | `update` | Validated revision is stored; active timer is safely replaced |
| Clone | None for original | A new pending job and UUID are created |
| Archive | `archive` | Timer and persistent timer stamp are removed; history remains |

Pause and archive stop future executions. They do not kill an occurrence that
already owns a valid execution lease.

## 20. Administrative commands

Run these only from a root deployment shell with
`/etc/furnacemind/furnacemind.env` loaded:

```bash
# Read-only rendering and inspection
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py plan --job-id <job-uuid>
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py provision --job-id <job-uuid> --dry-run
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py reconcile-all --dry-run

# Direct recovery actions
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py provision --job-id <job-uuid>
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py pause --job-id <job-uuid>
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py resume --job-id <job-uuid>
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py archive --job-id <job-uuid>
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py reconcile --job-id <job-uuid>
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py reconcile-all

# Read-only definition/policy validation
/opt/furnacemind/.venv/bin/python scripts/furnacemind_job_runner.py --job-id <job-uuid> --validate-only

# One bounded command-queue pass
/opt/furnacemind/.venv/bin/python scripts/furnacemind_control_sync.py --max-commands 25
```

The direct lifecycle commands are for deployment tests and administrator
recovery. Normal production actions should travel through the UI and command
queue so requester identity and command outcomes remain auditable.

Running the job runner without `--validate-only` is not a general **Run now**
feature. It executes only a logically due occurrence and may correctly exit
without a run when nothing is due.

## 21. Monitoring and health checks

### Control bridge

```bash
sudo systemctl status furnacemind-control-sync.timer --no-pager
sudo systemctl list-timers --all 'furnacemind-*'
sudo journalctl -u furnacemind-control-sync.service --since '1 hour ago' --no-pager
```

### One task

```bash
sudo systemctl status furnacemind-job-<job-uuid>.timer --no-pager
sudo systemctl status furnacemind-job@<job-uuid>.service --no-pager
sudo journalctl -u furnacemind-job@<job-uuid>.service -n 200 --no-pager
```

### Database queue and runs

Use pgAdmin or another approved SQL client:

```sql
SELECT command_id, job_id, action, status, attempt_count,
       available_at, started_at, completed_at, error_message
FROM automation.scheduled_job_commands
ORDER BY created_at DESC
LIMIT 50;

SELECT run_id, job_id, scheduled_for, status, attempt_number,
       started_at, completed_at, error_message
FROM automation.job_runs
ORDER BY created_at DESC
LIMIT 50;
```

Never add credentials, full prompts, raw plant datasets, or model tool arguments
to journald or command error messages.

Recommended alerts include:

- Control-sync timer is not enabled or has not triggered recently.
- Repeated failed commands for `bf2-jetson-01`.
- Active database jobs with missing/inactive timer units.
- Repeated failed or timed-out job runs.
- PostgreSQL, InfluxDB, or OpenRouter connectivity failures.
- Low disk space or incorrect system time.

## 22. Reconciliation after reboot or interruption

The control-sync timer starts 30 seconds after boot. Per-job timers use
`Persistent=true`, so systemd can recover missed calendar triggers according to
the stored misfire policy.

After a reboot or failed deployment, run a finite reconciliation pass:

```bash
sudo -i
set -a
. /etc/furnacemind/furnacemind.env
set +a
cd /opt/furnacemind
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py reconcile-all --dry-run
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py reconcile-all
exit
```

Reconciliation repairs database/systemd drift for the configured target device
and exits. It is not a daemon.

## 23. Troubleshooting

| Symptom | Checks and likely cause |
|---|---|
| UI saves a task but Activate remains pending | Check the control-sync timer and service journal; verify both sides use the same PostgreSQL database and target device ID |
| Command becomes failed | Read its bounded `error_message` and the control-sync journal; run `--validate-only` for the job |
| `Systemd mutations are disabled` | The root-loaded environment still has `SCHEDULED_TASKS_SYSTEMD_ENABLED=false` or was not reloaded after editing |
| `production executor is not ready` | `SCHEDULED_TASKS_EXECUTOR_READY` is false or the deployed code does not advertise executor capability |
| Environment file is unsafe | Require a regular, non-symlink file owned by `root`, group `furnacemind`, mode `0640` |
| Service user/group does not exist | Recheck `getent passwd furnacemind`, `getent group furnacemind`, and `id furnacemind` |
| Database connection refused/timed out | Check hostname, port, provider allow-list, VPN/routing, TLS parameters, and credentials |
| Alembic reports the wrong current revision | Stop activation, confirm the database URL, inspect `alembic heads/current`, and apply reviewed migrations |
| Timer exists but no run appears | Inspect timer next trigger, service journal, task active status, logical timezone, and whether the occurrence was actually due |
| Service exits with status 2 | Permanent validation/configuration/task failure; systemd intentionally does not restart it |
| Service exits with status 1 | Retryable persistence, lease, cancellation, or infrastructure failure; inspect the journal and database run record |
| ETA task is rejected | Use `online_process_data`, in-app delivery, and `include_graph=false` |
| Delivery is rejected | The production executor currently supports `in_app` only |
| Online result is empty | Verify the Influx token, requested time window, configured database/measurements, and Jetson clock |
| Historical result is missing or stale | Prepare and maintain the configured static dataset cache, or use online process data |
| OpenRouter task fails | Verify API key, model availability, outbound HTTPS/DNS, account limits, and reasoning-profile model overrides |
| `uv sync` fails on ARM64 | Resolve the dependency against the installed JetPack/Python combination and update the reviewed lock; do not bypass the lock ad hoc |

For additional systemd diagnostics:

```bash
sudo systemctl daemon-reload
sudo systemctl reset-failed furnacemind-control-sync.service
sudo systemd-analyze verify /etc/systemd/system/furnacemind-*.service
sudo journalctl -b -p warning --no-pager
```

Do not manually edit generated per-job timer files. Fix the definition or
deployment configuration and use the provisioning/reconciliation commands.

## 24. Upgrade procedure

Use a maintenance window for application, dependency, or migration changes.

1. Create a database backup or restore point.
2. Stop the control-sync timer so no new lifecycle command is applied mid-update.
3. Confirm no headless job service is currently running, or wait for it to
   finish. Then stop the per-job timers for the maintenance window. Database
   jobs remain active and reconciliation restores their timers afterward.
4. Fetch and check out the reviewed release.
5. Run `uv sync --frozen` and the focused tests.
6. Reapply the repository group/read permissions from Section 9.
7. Apply Alembic migrations.
8. Reinstall the shared service template.
9. Run `reconcile-all` to restore the expected per-job timer state.
10. Reinstall/start the control-sync units.
11. Perform a deterministic acceptance task.

Commands:

```bash
sudo systemctl stop furnacemind-control-sync.timer
sudo systemctl list-units --all 'furnacemind-job@*.service'
sudo systemctl stop 'furnacemind-job-*.timer'

cd /opt/furnacemind
git fetch --tags --prune
git checkout --detach <new-release-ref>
UV_PYTHON_INSTALL_DIR=/opt/furnacemind-runtime/python \
UV_CACHE_DIR=/opt/furnacemind-runtime/uv-cache \
  /usr/local/bin/uv sync --frozen
sudo chgrp -R furnacemind /opt/furnacemind
sudo chgrp -R furnacemind /opt/furnacemind-runtime
sudo chmod -R g-w,o-rwx /opt/furnacemind
sudo chmod -R g-w,o-rwx /opt/furnacemind-runtime
sudo chmod -R g+rX /opt/furnacemind
sudo chmod -R g+rX /opt/furnacemind-runtime

sudo -i
set -a
. /etc/furnacemind/furnacemind.env
set +a
cd /opt/furnacemind
/opt/furnacemind/.venv/bin/python -m alembic upgrade head
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py install-service
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py reconcile-all
/opt/furnacemind/.venv/bin/python scripts/furnacemind_systemd_provisioner.py install-control-sync
exit
```

Do not update the working tree while a job is executing. For higher-assurance
deployments, build versioned release directories and switch the configured paths
only through a reviewed deployment process.

## 25. Rollback procedure

Database downgrades can remove columns or data. Do not run `alembic downgrade`
automatically. Restore a database backup or use a reviewed forward-fix migration
when schema rollback is required.

For a code-only rollback that remains compatible with the current database:

1. Stop the control-sync timer.
2. Check out the previously recorded commit/tag.
3. Restore its locked environment with `uv sync --frozen`.
4. Run its focused tests.
5. Reinstall the shared units from that release.
6. Reconcile all jobs.
7. Restart and verify the control-sync timer.

If the old release cannot understand the current database revision, keep the
control timer stopped and use the approved database restore/forward-fix plan.

## 26. Disable or uninstall the scheduler

To disable new lifecycle processing without deleting data:

```bash
sudo systemctl disable --now furnacemind-control-sync.timer
```

Archive active jobs through Streamlit and allow the final queued archive
commands to finish before disabling the control timer. If the timer is already
disabled, use the direct archive command for each known job UUID.

Before removing shared unit files, confirm that no per-job timers remain:

```bash
sudo systemctl list-timers --all 'furnacemind-job-*'
sudo find /etc/systemd/system -maxdepth 1 -type f \
  -name 'furnacemind-job-*.timer' -print
```

Inspect the managed marker before removing any file:

```text
# Managed by FurnaceMind scheduled tasks. Do not edit.
```

After all job timers are archived and the exact files have been verified:

```bash
sudo systemctl stop furnacemind-control-sync.service
sudo rm -- /etc/systemd/system/furnacemind-control-sync.timer
sudo rm -- /etc/systemd/system/furnacemind-control-sync.service
sudo rm -- /etc/systemd/system/furnacemind-job@.service
sudo systemctl daemon-reload
```

The uninstall procedure deliberately retains PostgreSQL definitions, revisions,
commands, runs, logs, and outputs. Remove database history or
`/etc/furnacemind/furnacemind.env` only under a separate approved data-retention
and secret-decommissioning procedure.

## 27. Architecture and safety details

### Per-job timers and shared service

Each job uses a UUID-derived timer:

```text
furnacemind-job-<job-uuid>.timer
```

Every timer invokes:

```text
furnacemind-job@<job-uuid>.service
```

The template runs:

```text
<python> <application>/scripts/furnacemind_job_runner.py --job-id <job-uuid>
```

The unit files contain no job name, instructions, database credentials, API
keys, or model prompt. The runner reloads the canonical definition from
PostgreSQL.

### Safety gates

Systemd mutation is disabled by default. Both deployment flags must be true:

```dotenv
SCHEDULED_TASKS_SYSTEMD_ENABLED="true"
SCHEDULED_TASKS_EXECUTOR_READY="true"
```

Planning, JSON validation, and dry-run remain available while they are false.
Changing the flags alone does not install units, migrate the database, or prove
the environment is ready.

The Streamlit process is never run as root and is not granted unrestricted
`sudo systemctl`. The privileged control-sync service runs as root only because
it owns the narrow systemd mutation boundary. Actual FurnaceMind jobs run as the
unprivileged `furnacemind` user with systemd hardening.

### Command bridge

Streamlit writes owner-checked lifecycle commands to
`automation.scheduled_job_commands`. A partial unique index permits only one
pending or processing command for a job. The Jetson claims commands addressed to
its target device with a fenced lease, processes a bounded batch, records the
outcome, and exits.

Active edits are applied as pause, immutable revision update, and resume. If the
replacement timer cannot be installed, the processor attempts to restore the
previous audited definition and active state.

### Restricted FurnaceMind policy

Each execution exposes exactly one read-only fetch tool selected from the stored
data source. The runtime replaces model-provided fetch arguments with the exact
trusted data window, permits one fetch, and rejects every other FurnaceMind
tool. Operator instructions are report content, not authority to change tools,
credentials, policy, delivery, or the data window.

Analysis choices map as follows:

| UI choice | Stored level | Runtime behavior |
|---|---|---|
| No AI analysis | `none` | Deterministic fetch and numeric report; no model request |
| Brief analysis | `low` | FurnaceMind with Low reasoning |
| Standard analysis | `medium` | FurnaceMind with Medium reasoning |
| Detailed analysis | `high` | FurnaceMind with High reasoning |

Model-enabled executions receive aggregate, size-bounded data rather than raw
rows. Persisted metadata is allow-listed and excludes prompts, credentials,
model tool arguments, and raw tool results.

### Schedule translation

Calendar schedules become explicit systemd `OnCalendar` expressions in the
task's IANA timezone. Anchored every-N-hours schedules use safe UTC calendar
probes. The runner derives the actual logical occurrence and PostgreSQL's unique
`(job_id, scheduled_for)` constraint removes duplicate probes.

Per-job timers use `Persistent=true`. The runner applies the stored overlap and
misfire policies when a delayed or recovered timer fires.

### Data windows, leases, and retries

The data window is calculated before model/tool work from the persisted logical
`scheduled_for` timestamp, never from the wall clock at recovery time. Windows
use exact half-open `[start, end)` UTC semantics.

Claims use PostgreSQL's authoritative wall clock and a renewable execution
lease. Activation generation, lease token, and attempt number fence every retry
and terminal write. Retry intent is committed before sleeping, and no database
transaction remains open during data or model calls.

For a one-time task, success moves the job to `completed`; terminal failure,
timeout, or cancellation moves it to `execution_failed`. Recurring jobs retain
their active or paused lifecycle. Reconciliation removes timer files for
terminal one-time tasks.

## 28. Final production checklist

- [ ] Reviewed Git commit/tag recorded.
- [ ] Jetson architecture, systemd, clock, DNS, and network verified.
- [ ] Unprivileged `furnacemind` user/group created.
- [ ] Repository and virtual environment readable but not writable by the
      runtime user.
- [ ] `uv sync --frozen` completed on Jetson ARM64.
- [ ] Protected environment is `root:furnacemind` mode `0640` and not a symlink.
- [ ] Runtime PostgreSQL connection tested.
- [ ] Database backup completed and Alembic is at `20260907_0010`.
- [ ] Scheduled-task focused tests pass.
- [ ] Streamlit hides or blocks unsupported delivery, graph, and ETA data-source
      combinations before task creation.
- [ ] Historical cache prepared or historical tasks disabled.
- [ ] Influx access validated.
- [ ] OpenRouter access validated for model-enabled tasks.
- [ ] Both deployment safety gates explicitly enabled.
- [ ] Shared job service installed and verified.
- [ ] Control-sync service/timer installed, enabled, and active.
- [ ] Streamlit Cloud uses the same PostgreSQL database.
- [ ] Deterministic end-to-end test completed.
- [ ] Model-enabled end-to-end test completed if AI analysis is enabled.
- [ ] Pause, resume, edit, and archive verified.
- [ ] Monitoring, backup, update, rollback, and ownership responsibilities
      assigned.
