# Edge CI/CD Guide for Jetson and Raspberry Pi

This guide enables automatic backend deployment to a user-selected Jetson,
Raspberry Pi, or both after a push to `release`. Pushes to `dev` and `dev-v01`
run CI but never deploy. The implementation uses
`.github/workflows/edge-ci-cd.yml` and `scripts/deploy_edge_release.sh`.

## 1. Deployment flow

```text
Push or merge to release
          |
          v
GitHub-hosted CI: locked install + complete tests
          |
          v
EVONITH_DEPLOY_TARGET (default: pi)
          |
          +-----------------------+
          |                       |
          v                       v
Jetson self-hosted runner    Pi self-hosted runner
label: evonith-jetson       label: evonith-pi
          |                       |
          v                       v
Deploy exact tested SHA, restart, health/readiness, automatic code rollback
```

The target can be `pi`, `jetson`, `both`, or `none`. When the repository
variable is missing, `pi` is selected. The two device jobs are independent.
Deployment does not copy secrets, runtime data, databases, uploads, or TLS
certificates.

## 2. Security requirements

Before registering a runner:

1. Keep the repository private.
2. Protect `release`; require pull-request review and the CI check.
3. Do not run pull-request jobs on the device runners.
4. Give the runner no shell secrets and no unrestricted sudo.
5. Keep `/etc/evonith-bf/backend.env` unreadable by the runner.
6. Restrict repository administration and workflow-file changes.

Self-hosted runners execute repository code on the device. GitHub recommends
using them only with private repositories because untrusted pull requests can
otherwise execute dangerous code. This repository routes only post-test
deployment jobs—not pull-request tests—to device labels.

## 3. Prepare each device account

Choose the Linux account that owns `/opt/evonith-bf` as the runner account, or
transfer ownership to a dedicated runner account. The runner must be able to
update the production checkout but must not read `backend.env`.

Create the narrowly scoped deploy group:

```bash
sudo groupadd --force evonith-deploy
sudo usermod --append --groups evonith-deploy <RUNNER_USER>
sudo install -o root -g root -m 0440 \
  /opt/evonith-bf/infra/sudoers/evonith-edge-deploy.example \
  /etc/sudoers.d/evonith-edge-deploy
sudo visudo --check --file=/etc/sudoers.d/evonith-edge-deploy
```

Log out and back in so new group membership applies. Confirm the production
checkout is owned by the runner account:

```bash
sudo chown -R <RUNNER_USER>:<RUNNER_USER> /opt/evonith-bf
sudo -u <RUNNER_USER> git -C /opt/evonith-bf status --short
```

The status command must print nothing. The sudoers rule allows only backend
restart and active-status commands.

## 4. Register the GitHub runner

In the private GitHub repository:

1. Open **Settings → Actions → Runners**.
2. Select **New self-hosted runner**.
3. Choose **Linux** and **ARM64**.
4. Run GitHub's generated download and configuration commands as
   `<RUNNER_USER>`. Registration tokens expire quickly; never save or commit
   them.
5. Add one custom label during configuration:
   - Jetson: `evonith-jetson`
   - Raspberry Pi: `evonith-pi`
6. Install and start the runner using GitHub's generated service commands.

When using `config.sh`, the label option is:

```bash
./config.sh --url <REPOSITORY_URL> --token <ONE_TIME_TOKEN> \
  --labels evonith-jetson
```

Use `evonith-pi` on the Pi. Do not copy a runner directory or registration
token between devices. GitHub should show both runners as **Idle**.

## 5. Select the deployment device

In **Settings → Secrets and variables → Actions → Variables**, create one
repository variable:

| Variable | Value | Effect after CI on `release` |
|---|---|---|
| `EVONITH_DEPLOY_TARGET` | `pi` | Deploy Raspberry Pi (default when missing) |
| `EVONITH_DEPLOY_TARGET` | `jetson` | Deploy Jetson |
| `EVONITH_DEPLOY_TARGET` | `both` | Deploy both devices independently |
| `EVONITH_DEPLOY_TARGET` | `none` | Run tests without deployment |

Set it to `none` while initially creating the `release` branch or maintaining
runners. Once the manually installed Pi runner is online, change it to `pi`.
An invalid value fails the CI target-validation step instead of silently
deploying somewhere unexpected.

This is a non-secret selector. Do not put passwords, tokens, public IPs, or
environment-file contents in it.

### First-time release-branch creation

Before the Pi runner exists, create `EVONITH_DEPLOY_TARGET=none` in GitHub.
After the CI/CD implementation is committed and pushed to `dev-v01`, create the
release branch from that tested commit:

```bash
git switch dev-v01
git pull --ff-only origin dev-v01
git switch -c release
git push -u origin release
git switch dev-v01
```

Protect `release` in GitHub immediately. Its first workflow run tests only
because the target is `none`. Do not set the target to `pi` until the manual Pi
deployment is healthy and its `evonith-pi` runner is online.

The workflow uses the GitHub environments `jetson-production` and
`raspberry-pi-production` for deployment history. Optional environment approval
rules turn fully automatic deployment into approved deployment.

## 6. First deployment test

Before enabling automatic deployment, preview the deploy script on each device:

```bash
cd /opt/evonith-bf
scripts/deploy_edge_release.sh \
  --ref "$(git rev-parse HEAD)" \
  --branch release \
  --device raspberry-pi \
  --dry-run
```

Use `--device jetson` on Jetson. Then use **Actions → Edge CI/CD → Run
workflow** and select one device. Watch:

```bash
sudo journalctl -u evonith-backend -f
```

After success, verify:

```bash
git -C /opt/evonith-bf log -1 --oneline
sudo systemctl is-active evonith-backend
curl --fail http://127.0.0.1:1432/api/v1/health
curl --fail http://127.0.0.1:1432/api/v1/readiness
```

The deployed commit must equal the workflow commit SHA.

## 7. Normal release process

1. Develop and test on a feature branch or `dev-v01`.
2. Open a pull request from `dev-v01` to protected `release`.
3. Wait for the Edge CI/CD test job and review the change.
4. Confirm `EVONITH_DEPLOY_TARGET` selects the intended device.
5. Merge into `release`.
6. CI tests the release commit again.
7. The selected device job deploys that exact SHA.
8. Check local/public health and Streamlit after deployment.

Pushes to `dev` and `dev-v01` never run device jobs. Only the exact
`refs/heads/release` reference is eligible for automatic deployment.

## 8. Deployment safety behavior

`deploy_edge_release.sh`:

- obtains an exclusive deployment lock;
- refuses a dirty production checkout;
- fetches only the configured deployment branch;
- verifies the requested commit belongs to `origin/release`;
- verifies Jetson/Pi identity and 64-bit ARM architecture;
- installs only the locked `edge` dependency group;
- checks critical imports before restart;
- restarts the backend through narrowly scoped sudo;
- waits for health and readiness;
- restores the previous code and dependencies when deployment fails.

Runtime data is deliberately outside Git and is not rolled back. Changes that
require a database or runtime-data migration need a separately reviewed backup
and migration plan.

## 9. Disable or stop automatic deployment

Set `EVONITH_DEPLOY_TARGET=none` before maintenance. This does not stop the
running API; it only prevents new automatic deployments.

For an incident:

```bash
sudo systemctl status evonith-backend --no-pager --full
sudo journalctl -u evonith-backend -n 150 --no-pager
git -C /opt/evonith-bf log -2 --oneline
```

If automatic rollback succeeded, the previous commit will be checked out. If
both deployment and rollback failed, use the manual rollback procedure in the
device runbook and keep the deployment variable disabled until resolved.

## 10. Runner maintenance

Monthly and after runner alerts:

- confirm each enabled runner is online and idle;
- apply OS security updates;
- allow the runner service to update itself;
- review runner and backend service logs;
- verify the sudoers file with `visudo`;
- remove obsolete/offline runner registrations from GitHub;
- test a manual workflow dispatch on the standby device.

Official GitHub references:

- [Adding self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [Using labels with self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/apply-labels)
- [Using self-hosted runners in workflows](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow)
