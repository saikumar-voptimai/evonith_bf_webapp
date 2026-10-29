"""Host health of the admin view (SV-4.8, #311): CPU, RAM, disks and temperatures of the machine the API runs on
(the Jetson runs the API, the worker and Streamlit).

On the Jetson the figures come from one line of ``tegrastats`` (RAM, swap, the load of each online core and the
thermal zones, e.g. ``cpu@46.9C gpu@42.7C tj@46.9C``); elsewhere, and when ``tegrastats`` fails, from ``psutil``
(CPU over 0.2 s, memory, swap, ``sensors_temperatures`` on Linux; Windows reports none). Disks always come from
``psutil.disk_usage`` for the file systems of the repository and of ``src/storage``. Each figure carries a state
against ``LIMITS`` (warning, critical); the host's state is the worst. ``snapshot`` keeps a reading for 5 s.
No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.memo import memo

log = logging.getLogger("services.host")

TEGRASTATS = "tegrastats"
TEGRASTATS_INTERVAL_MS = 500
TEGRASTATS_TIMEOUT_S = 3.0
#: (warning, critical) per figure: CPU load %, RAM used %, disk used %, temperature °C. The Jetson Orin throttles near
#: a junction temperature of 99 °C.
LIMITS: dict[str, tuple[float, float]] = {
    "cpu_pct": (85.0, 95.0),
    "ram_pct": (85.0, 95.0),
    "disk_pct": (85.0, 95.0),
    "temp_c": (80.0, 95.0),
}
#: tegrastats zones that report a fixed placeholder, not a measurement (PMIC@50C / PMIC@100C on several modules).
IGNORED_ZONES = frozenset({"pmic"})
REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = REPO_ROOT / "src" / "storage"
RANK = {"ok": 0, "warning": 1, "critical": 2}

_RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
_SWAP = re.compile(r"\bSWAP (\d+)/(\d+)MB")
_CPU = re.compile(r"\bCPU \[([^\]]*)\]")
_CORE = re.compile(r"^(\d+(?:\.\d+)?)%")
_TEMP = re.compile(r"(?<![\w%])([A-Za-z][\w-]*)@(-?\d+(?:\.\d+)?)C\b")


def state_of(value: float | None, limit: str) -> str | None:
    """``ok``, ``warning`` or ``critical`` against ``LIMITS[limit]``; None without a value."""
    if value is None:
        return None
    warn, crit = LIMITS[limit]
    return "critical" if value >= crit else "warning" if value >= warn else "ok"


def worst(states: Iterable[str | None]) -> str:
    present = [s for s in states if s]
    return max(present, key=RANK.__getitem__) if present else "ok"


def _pct(used: float | None, total: float | None) -> float | None:
    return None if used is None or not total else 100.0 * used / total


# ---------------------------------------------------------------------------------------------------- tegrastats
def parse_tegrastats(line: str) -> dict[str, Any]:
    """RAM and swap (MB), the mean load of the online cores (%), the cores (None = off) and the thermal zones (°C) of
    one ``tegrastats`` line; a figure the line lacks is None (an empty list for zones)."""
    out: dict[str, Any] = {
        "ram_used_mb": None,
        "ram_total_mb": None,
        "swap_used_mb": None,
        "swap_total_mb": None,
        "cpu_pct": None,
        "cores_pct": [],
        "temperatures": [],
    }
    if m := _RAM.search(line):
        out["ram_used_mb"], out["ram_total_mb"] = float(m[1]), float(m[2])
    if m := _SWAP.search(line):
        out["swap_used_mb"], out["swap_total_mb"] = float(m[1]), float(m[2])
    if m := _CPU.search(line):
        cores: list[float | None] = []
        for item in m[1].split(","):
            core = _CORE.match(item.strip())
            cores.append(float(core[1]) if core else None)
        online = [c for c in cores if c is not None]
        out["cores_pct"] = cores
        out["cpu_pct"] = sum(online) / len(online) if online else None
    temps = []
    for name, value in _TEMP.findall(line):
        if name.lower() not in IGNORED_ZONES:
            temps.append({"sensor": name, "c": float(value)})
    out["temperatures"] = temps
    return out


def read_tegrastats(timeout_s: float = TEGRASTATS_TIMEOUT_S) -> str | None:
    """One line of ``tegrastats`` (it prints a line every interval until stopped); None when it is not installed,
    prints nothing within ``timeout_s`` or fails."""
    exe = shutil.which(TEGRASTATS)
    if exe is None:
        return None
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed arguments, the path found by shutil.which
            [exe, "--interval", str(TEGRASTATS_INTERVAL_MS)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError as exc:
        log.warning("tegrastats not started (%s)", exc)
        return None
    line: list[str] = []
    reader = threading.Thread(target=lambda: line.append(proc.stdout.readline() if proc.stdout else ""), daemon=True)
    reader.start()
    reader.join(timeout_s)
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    text = line[0].strip() if line else ""
    return text or None


# -------------------------------------------------------------------------------------------------------- psutil
def _psutil_figures() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    temps = []
    sensors = getattr(psutil, "sensors_temperatures", None)
    if sensors is not None:
        try:
            for chip, entries in (sensors() or {}).items():
                for i, e in enumerate(entries):
                    if e.current is None:
                        continue
                    label = f"{chip} {e.label}" if e.label else chip if len(entries) == 1 else f"{chip} {i + 1}"
                    temps.append({"sensor": label, "c": float(e.current)})
        except OSError as exc:  # pragma: no cover - sysfs not readable
            log.debug("temperatures not read (%s)", exc)
    return {
        "cpu_pct": float(psutil.cpu_percent(interval=0.2)),
        "cores_pct": [],
        "ram_used_mb": (vm.total - vm.available) / 1e6,
        "ram_total_mb": vm.total / 1e6,
        "swap_used_mb": swap.used / 1e6,
        "swap_total_mb": swap.total / 1e6,
        "temperatures": temps,
    }


def disks(paths: Iterable[Path | str] | None = None) -> list[dict[str, Any]]:
    """Used and total space (GB) of the file systems of ``paths`` (default: the repository and ``src/storage``), one
    row per file system."""
    import psutil

    rows, seen = [], set()
    for path in paths if paths is not None else (REPO_ROOT, STORAGE_DIR):
        p = Path(path)
        if not p.exists():
            continue
        try:
            device = os.stat(p).st_dev
            if device in seen:
                continue
            usage = psutil.disk_usage(str(p))
        except OSError as exc:
            log.debug("disk %s not read (%s)", p, exc)
            continue
        seen.add(device)
        pct = float(usage.percent)
        rows.append(
            {
                "path": str(p),
                "used_gb": usage.used / 1e9,
                "total_gb": usage.total / 1e9,
                "pct": pct,
                "state": state_of(pct, "disk_pct"),
            }
        )
    return rows


def _uptime_s() -> float | None:
    try:
        import psutil

        return max(0.0, time.time() - psutil.boot_time())
    except Exception:  # noqa: BLE001 - uptime is informative
        return None


def _load_1m() -> float | None:
    getloadavg = getattr(os, "getloadavg", None)  # not on Windows
    return float(getloadavg()[0]) if getloadavg is not None else None


# ---------------------------------------------------------------------------------------------------- snapshot
def collect(
    *,
    tegrastats: Callable[[], str | None] = read_tegrastats,
    figures: Callable[[], Mapping[str, Any]] = _psutil_figures,
    disk_rows: Callable[[], list[dict[str, Any]]] = disks,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One reading of the host: ``tegrastats`` when it answers, else ``psutil``; disks from ``psutil``."""
    notes: list[str] = []
    line = tegrastats()
    data: Mapping[str, Any] | None = None
    source = "psutil"
    if line:
        parsed = parse_tegrastats(line)
        if parsed["ram_total_mb"] is not None or parsed["cpu_pct"] is not None:
            data, source = parsed, "tegrastats"
        else:
            notes.append("tegrastats printed a line without RAM or CPU figures; psutil used instead.")
    elif shutil.which(TEGRASTATS) is not None and tegrastats is read_tegrastats:
        notes.append("tegrastats did not answer; psutil used instead.")
    if data is None:
        data = figures()
    temps = [{**t, "state": state_of(t["c"], "temp_c")} for t in data.get("temperatures") or []]
    if not temps:
        notes.append("No temperature sensor readable on this host.")
    ram_pct = _pct(data.get("ram_used_mb"), data.get("ram_total_mb"))
    cpu_pct = data.get("cpu_pct")
    disk = disk_rows()
    states = {
        "cpu_state": state_of(cpu_pct, "cpu_pct"),
        "ram_state": state_of(ram_pct, "ram_pct"),
    }
    overall = worst([*states.values(), *(d["state"] for d in disk), *(t["state"] for t in temps)])
    return {
        "collected_at": now or datetime.now(UTC),
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "source": source,
        "state": overall,
        "cpu_pct": cpu_pct,
        "cpu_count": os.cpu_count(),
        "cores_pct": list(data.get("cores_pct") or []),
        "load_1m": _load_1m(),
        "ram_used_mb": data.get("ram_used_mb"),
        "ram_total_mb": data.get("ram_total_mb"),
        "ram_pct": ram_pct,
        "swap_used_mb": data.get("swap_used_mb"),
        "swap_total_mb": data.get("swap_total_mb"),
        "disks": disk,
        "temperatures": temps,
        "uptime_s": _uptime_s(),
        "limits": {k: {"warning": w, "critical": c} for k, (w, c) in LIMITS.items()},
        "notes": notes,
        **states,
    }


@memo(ttl_seconds=5, max_entries=1, copy=True)
def snapshot() -> dict[str, Any]:
    """``collect()``, kept for 5 s (several admin tabs refresh at once; tegrastats takes ~0.5 s)."""
    return collect()


__all__ = [
    "IGNORED_ZONES",
    "LIMITS",
    "collect",
    "disks",
    "parse_tegrastats",
    "read_tegrastats",
    "snapshot",
    "state_of",
    "worst",
]
