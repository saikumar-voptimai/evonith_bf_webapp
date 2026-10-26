"""Start the Svelte dev server, the twin-api, the twin-worker and Streamlit (/legacy) in one terminal.

Cross-platform (Windows, Linux, Jetson). Run from anywhere:

    python scripts/dev.py                 # all four processes
    python scripts/dev.py api web         # a subset

Ports: SPA 5173 (open http://localhost:5173), API 8100, Streamlit 8501 under /legacy. The SPA dev server proxies
/api and /legacy, so the browser sees one origin, as it does behind Caddy in production. Ctrl+C stops everything.
"""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_KEY_FILE = ROOT / "src" / "storage" / "twin_api" / "jwt_secret.dev"
API_PORT = os.environ.get("BF_API_PORT", "8100")
LEGACY_PORT = os.environ.get("BF_LEGACY_PORT", "8501")
IS_WINDOWS = os.name == "nt"

STREAMLIT_ARGS = [
    "run",
    "src/app.py",
    "--server.port",
    LEGACY_PORT,
    "--server.baseUrlPath",
    "legacy",
    "--server.headless",
    "true",
]


def _streamlit_cmd() -> list[str]:
    if IS_WINDOWS:
        # torch must be imported before Streamlit on Windows (DLL load order, see run_streamlit.py).
        code = (
            "import sys, torch; from streamlit.web.cli import main; "
            f"sys.argv = ['streamlit', *{STREAMLIT_ARGS!r}]; main()"
        )
        return [sys.executable, "-c", code]
    return [sys.executable, "-m", "streamlit", *STREAMLIT_ARGS]


def _pnpm() -> str:
    exe = shutil.which("pnpm") or shutil.which("pnpm.cmd")
    if not exe:
        sys.exit("pnpm not found: install Node.js >= 20 and run `corepack enable` (or `npm i -g pnpm`).")
    return exe


PROCESSES = {
    "api": lambda: [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--app-dir",
        "src",
        "--host",
        "127.0.0.1",
        "--port",
        API_PORT,
        "--reload",
        "--reload-dir",
        "src/api",
        "--reload-dir",
        "src/services",
    ],
    "worker": lambda: [sys.executable, "-m", "worker"],
    "legacy": _streamlit_cmd,
    "web": lambda: [_pnpm(), "--dir", "frontend", "dev"],
}
COLOURS = {"api": "\033[33m", "worker": "\033[35m", "legacy": "\033[36m", "web": "\033[32m"}


def _pump(name: str, proc: subprocess.Popen) -> None:
    tag = f"{COLOURS.get(name, '')}{name:>6}\033[0m |"
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(f"{tag} {line}")
        sys.stdout.flush()


def _dev_jwt_secret() -> str:
    """Session signing key for development, kept in a git-ignored file so sessions survive API reloads."""
    if DEV_KEY_FILE.is_file():
        key = DEV_KEY_FILE.read_text(encoding="utf-8").strip()
        if len(key) >= 32:
            return key
    DEV_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    fd = os.open(DEV_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key)
    return key


def _sigterm(signum: int, frame: object) -> None:
    raise KeyboardInterrupt  # stop the children exactly as Ctrl+C does


def main(argv: list[str]) -> int:
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, _sigterm)
    names = argv or list(PROCESSES)
    unknown = [n for n in names if n not in PROCESSES]
    if unknown:
        sys.exit(f"unknown process(es): {', '.join(unknown)}; choose from {', '.join(PROCESSES)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "furnace_data"), env.get("PYTHONPATH", "")])
    env.setdefault("PYTHONUNBUFFERED", "1")
    if "api" in names and not env.get("BF_API_JWT_SECRET"):
        env["BF_API_JWT_SECRET"] = _dev_jwt_secret()
    procs: dict[str, subprocess.Popen] = {}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    for name in names:
        procs[name] = subprocess.Popen(
            PROCESSES[name](),
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=flags,
            start_new_session=not IS_WINDOWS,
        )
        threading.Thread(target=_pump, args=(name, procs[name]), daemon=True).start()
    print(f"started: {', '.join(names)} - SPA http://localhost:5173, API :{API_PORT}, Streamlit :{LEGACY_PORT}/legacy")
    try:
        while True:
            for name, proc in procs.items():
                code = proc.poll()
                if code is not None:
                    print(f"{name} exited with code {code}; stopping the others")
                    return code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                if IS_WINDOWS:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
