# Jetson runtime benchmark (SV-6.2, #316)

Runtime of the optimiser on the BF-02 Jetson AGX Orin against the targets of section 10 of the programme plan (#262,
AGX Orin column). `deploy/jetson/benchmark.sh` runs `scripts/optimiser_bench.py` in five modes, as the worker runs the
scheduled jobs, and writes the measured table between the markers below.

| Mode | `optimiser_bench.py` | Measured value | Target |
|---|---|---|---|
| `solve` | `--mode solve` | the slowest of three network solves after the first (the first builds the STRATA-Flow context) | 2.8 s |
| `jacobian` | `--mode jacobian --solvers 8` | the 17 solves of a new calibration's Jacobian on 8 started solver processes (their start is timed apart: once per run) | 6 s |
| `step2` | `--mode step2` | wall time of Step 2, STRATA-ω and Fixed-ω in a 2-process pool, value of limits included (`optimiser.step2`, 3 per day) | 90 s |
| `max_production` | `--mode max_production` | wall time of the maximum-production search, STRATA-ω | 300 s |
| `step1` | `--mode step1` | wall time of Step 1, cheapest production level and frontier, STRATA-ω (`optimiser.step1`, 3 per day) | 600 s |

The plans are solved on the reference plan day (`COKE_ENERGY_TODAY=2026-10-15`, `COKE_ENERGY_WEATHER_LIVE=0`), so
runs compare across hosts and days; the yard and calibration are the live ones when Neon and InfluxDB answer, else the
reference yard and the shipped calibration (the first line of each mode's log says which). Network solves run on
`BF_STRATA_SOLVERS` processes (default min(8, cpus)).

## On the device

In a checkout of the operator synced by `install_python.sh` (`deploy/jetson/README.md`, section "Python and uv"), as
the operator; `uv run --no-sync` uses the checkout's `.venv`:

```bash
sudo deploy/jetson/install_python.sh --release "$PWD"         # uv, CPython 3.12, uv sync --frozen, import check
uv run --no-sync pytest -q tests/                             # full suite (BURDEN_ENGINE_FAST unset: slow cases too)
deploy/jetson/benchmark.sh --open-issue                       # 20 min or more; one issue per runtime above target
git add docs/jetson_benchmark.md && git commit -m "docs(deploy): Jetson benchmark (SV-6.2) [skip ci]"
```

Tests that need PostgreSQL, Redis or Caddy skip until those run on the device (SV-6.3); with them installed, set
`BF_TEST_REQUIRE_SERVICES=1` so that a missing service fails the run, as in CI.

`benchmark.sh` exits 1 when a mode is above its target or gave no result; without `--open-issue` it prints the `gh issue
create` command of each mode above its target instead. Logs of every mode stay in `--log-dir` (default `/tmp/bf-benchmark-<UTC
time>`).

## Results

<!-- benchmark:start -->
Not measured yet (pending: device). `deploy/jetson/benchmark.sh` replaces this block with the measured table.

| Operation | Target (AGX Orin) | Measured | Status | Notes |
|---|---|---|---|---|
| One STRATA-Flow network solve (worst of 3 repeats after the first) | 2.8 s | — | pending: device | — |
| Jacobian at a new calibration (17 solves) on the solver pool | 6.0 s | — | pending: device | — |
| Step 2, STRATA-ω and Fixed-ω in parallel, with the value of limits | 90.0 s | — | pending: device | — |
| Maximum production (STRATA-ω) | 300 s (5.0 min) | — | pending: device | — |
| Step 1, cheapest production level and frontier (STRATA-ω) | 600 s (10.0 min) | — | pending: device | — |
<!-- benchmark:end -->

## Reference hosts (not the Orin)

Same script and reference case on x86_64 hosts, for comparison with the device's table:

| Operation | Development host (SV-1.8, Linux, 2 vCPU, 2 solver processes) | Laptop (29 Oct 2026, Windows, 4 cores / 8 threads shared with two other jobs, 8 solver processes) |
|---|---|---|
| One network solve | about 1.1 s | 1.35–1.70 s (first 1.64 s) |
| Jacobian, 17 solves | 21 s serial | 28.3 s serial; 14.7 s on 8 processes (started in 12.2 s, spawn) |
| Step 2 | 48 s (73 solves) | 181.4 s (90 solves; above 90 s) |
| Maximum production | 246 s (347 solves) | not run |
| Step 1 | 19.3 min (1,260 solves; above 10 min) | not run |

Step 1 was above its target on the development host (SV-1.8, #288); its runtime on the Orin decides whether the
scheduled `optimiser.step1` (time limit 3,600 s) needs the follow-ups of #288.
