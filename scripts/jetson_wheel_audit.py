"""Wheel audit of ``uv.lock`` for the Jetson AGX Orin (SV-6.2, #316).

    python scripts/jetson_wheel_audit.py [--uv "uvx --from uv==0.7.13 uv"] [--write]

Runs the recipe of ``make lock-check`` by hand: ``uv lock --check``, ``uv export`` of every group, and ``uv pip
compile`` of the export for ``JETSON_PLATFORM`` (aarch64-manylinux_2_35: JetPack 6 = Ubuntu 22.04, glibc 2.35) and
CPython 3.12 with ``--only-binary :all:`` except ``JETSON_SOURCE_BUILDS`` (both read from the Makefile). For every
resolved package it then picks the file uv installs on the Jetson: the locked wheel whose best tag ranks first in
packaging's order for CPython 3.12 on manylinux_2_35_aarch64 (CPython ABI before abi3 before pure Python, the newest
manylinux first), else the sdist. Prints the block between the markers of ``docs/jetson_wheel_audit.md`` (``--write``
replaces it there): the counts per kind, the packages without an aarch64 resolution and the table of the audited
packages. Exit 1 when the recipe fails, a resolved package has neither a matching wheel nor a source-build exception,
or a source-build exception has a wheel after all (the exception is stale).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import NamedTuple

from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "jetson_wheel_audit.md"
MARK_START = "<!-- wheel-audit:start -->"
MARK_END = "<!-- wheel-audit:end -->"
PYTHON = (3, 12)
UV_DEFAULT = "uvx --from uv==0.7.13 uv"

# (package, required by, note): the packages of the ticket and the compiled dependencies they pull in
AUDIT: tuple[tuple[str, str, str], ...] = (
    ("torch", "project, sentence-transformers", "CPU build: its CUDA libraries (`nvidia-*-cu12`) are locked for "
     "x86_64 Linux only, so none installs here. The Orin GPU would need NVIDIA's JetPack wheel (not used)"),
    ("highspy", "—", "Not a dependency. The LPs (burden engine, BMO) call HiGHS through "
     "`scipy.optimize.linprog(method=\"highs\")`; HiGHS is compiled into the scipy wheel"),
    ("scipy", "project, scikit-learn, sentence-transformers", "Carries HiGHS (see highspy)"),
    ("numpy", "project and 23 packages", ""),
    ("psycopg2-binary", "project, furnace-data", "libpq and OpenSSL bundled in the wheel"),
    ("duckdb", "project, pygwalker", ""),
    ("pyarrow", "influxdb3-python, project, streamlit, pygwalker", "Arrow Flight client of InfluxDB 3"),
    ("influxdb3-python", "project, furnace-data", "Pure Python on pyarrow"),
    ("chromadb", "project", "Pure Python; its compiled dependencies are the next rows"),
    ("chroma-hnswlib", "chromadb", ""),
    ("onnxruntime", "chromadb", "CPU execution provider (default embedding function)"),
    ("tokenizers", "chromadb, transformers, voyageai", ""),
    ("pypika", "chromadb", "No wheel at all: sdist, pure Python (Makefile `JETSON_SOURCE_BUILDS`)"),
    ("sentence-transformers", "project", "Pure Python on torch and transformers"),
    ("safetensors", "transformers (sentence-transformers)", ""),
    ("scikit-learn", "project, sentence-transformers", ""),
    ("quickjs", "project, pygwalker", "No aarch64 wheel: C extension compiled from the sdist during `uv sync` "
     "(needs build-essential; Makefile `JETSON_SOURCE_BUILDS`)"),
)  # fmt: skip

KIND_LABELS = {
    "cpython": "CPython 3.12 aarch64 wheel",
    "abi3": "abi3 aarch64 wheel",
    "pure": "pure-Python wheel",
    "sdist": "source build (sdist)",
    "missing": "no installable file",
}


class Pick(NamedTuple):
    """The file uv installs for one package on the target, and its kind (``KIND_LABELS``)."""

    name: str
    version: str
    file: str | None
    kind: str
    glibc: str | None  # glibc floor of the matched manylinux tag ("2.28"), None for a pure wheel or an sdist


# ---- configuration --------------------------------------------------------------------------------------------------


def makefile_settings(makefile: Path = ROOT / "Makefile") -> tuple[str, list[str]]:
    """``JETSON_PLATFORM`` and ``JETSON_SOURCE_BUILDS`` of the Makefile (the lock-check recipe's inputs)."""
    text = makefile.read_text(encoding="utf-8")
    values = dict(re.findall(r"^(JETSON_PLATFORM|JETSON_SOURCE_BUILDS)\s*:=\s*(.*?)\s*$", text, re.M))
    if set(values) != {"JETSON_PLATFORM", "JETSON_SOURCE_BUILDS"}:
        raise ValueError(f"{makefile}: JETSON_PLATFORM / JETSON_SOURCE_BUILDS not found")
    return values["JETSON_PLATFORM"], values["JETSON_SOURCE_BUILDS"].split()


def target_tags(platform: str, python: tuple[int, int] = PYTHON) -> list[Tag]:
    """Tags a CPython ``python`` interpreter accepts on ``platform`` (``aarch64-manylinux_2_35``), best first."""
    match = re.fullmatch(r"aarch64-manylinux_2_(\d+)", platform)
    if not match or int(match.group(1)) < 17:
        raise ValueError(f"not an aarch64 manylinux platform (aarch64-manylinux_2_17 or later): {platform}")
    platforms: list[str] = []
    for minor in range(int(match.group(1)), 16, -1):  # aarch64 starts at manylinux2014 = manylinux_2_17
        platforms.append(f"manylinux_2_{minor}_aarch64")
    platforms.append("manylinux2014_aarch64")
    interpreter = f"cp{python[0]}{python[1]}"
    return [
        *cpython_tags(python, abis=[interpreter], platforms=platforms),
        *compatible_tags(python, interpreter, platforms),
    ]


def _glibc(platform: str) -> str | None:
    if platform == "manylinux2014_aarch64":
        return "2.17"
    match = re.fullmatch(r"manylinux_(\d+)_(\d+)_aarch64", platform)
    return f"{match.group(1)}.{match.group(2)}" if match else None


# ---- lock -----------------------------------------------------------------------------------------------------------


def load_lock(path: Path = ROOT / "uv.lock") -> dict[str, list[dict]]:
    """The lock's packages by canonical name (a forked resolution locks several versions of one name)."""
    with path.open("rb") as fh:
        lock = tomllib.load(fh)
    packages: dict[str, list[dict]] = {}
    for package in lock.get("package", []):
        packages.setdefault(canonicalize_name(package["name"]), []).append(package)
    return packages


def pick(package: dict, tags: list[Tag]) -> Pick:
    """The wheel of ``package`` (a ``[[package]]`` table of uv.lock) that ranks first on ``tags``, else its sdist."""
    rank = {tag: i for i, tag in enumerate(tags)}
    best: tuple[int, str, Tag] | None = None
    for wheel in package.get("wheels", []):
        filename = wheel["url"].rsplit("/", 1)[-1]
        matched = [(rank[t], t) for t in parse_wheel_filename(filename)[3] if t in rank]
        if matched:
            position, tag = min(matched)
            if best is None or position < best[0]:
                best = (position, filename, tag)
    name, version = package["name"], package["version"]
    if best is not None:
        _, filename, tag = best
        kind = "pure" if tag.platform == "any" else "abi3" if tag.abi == "abi3" else "cpython"
        return Pick(name, version, filename, kind, _glibc(tag.platform))
    sdist = package.get("sdist")
    if sdist:
        filename = sdist.get("url", "").rsplit("/", 1)[-1] or f"{name}-{version}.tar.gz"
        return Pick(name, version, filename, "sdist", None)
    return Pick(name, version, None, "missing", None)


def locked_package(packages: dict[str, list[dict]], name: str, version: str | None = None) -> dict | None:
    """The locked entry of ``name`` (at ``version`` when given); None when the lock does not hold it."""
    entries = packages.get(canonicalize_name(name), [])
    if version is not None:
        entries = [p for p in entries if p["version"] == version]
    return entries[0] if len(entries) == 1 else None


# ---- recipe ---------------------------------------------------------------------------------------------------------


def parse_resolved(text: str) -> dict[str, str]:
    """``name==version`` lines of a ``uv pip compile`` output, by canonical name."""
    found = re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", text, re.M)
    return {canonicalize_name(name): version for name, version in found}


def run_recipe(uv: list[str], platform: str, source_builds: list[str], cwd: Path = ROOT) -> tuple[dict[str, str], str]:
    """The lock-check recipe of the Makefile: the packages it resolves for ``platform`` and the uv version."""

    def run(*args: str) -> str:
        done = subprocess.run([*uv, *args], cwd=cwd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"uv {' '.join(args[:2])} failed (exit {done.returncode}):\n{done.stderr.strip()}")
        return done.stdout

    version = run("--version").split()[1]
    run("lock", "--check")
    with tempfile.TemporaryDirectory() as tmp:
        req, out = Path(tmp) / "requirements.txt", Path(tmp) / "resolved.txt"
        run("export", "--frozen", "--no-hashes", "--no-emit-project", "--all-groups", "--quiet", "-o", str(req))
        no_binary = [arg for name in source_builds for arg in ("--no-binary", name)]
        run("pip", "compile", str(req), "--no-deps", "--quiet", "--python-version", f"{PYTHON[0]}.{PYTHON[1]}",
            "--python-platform", platform, "--only-binary", ":all:", *no_binary, "-o", str(out))  # fmt: skip
        return parse_resolved(out.read_text(encoding="utf-8")), version


# ---- audit and rendering --------------------------------------------------------------------------------------------


def audit(
    packages: dict[str, list[dict]], resolved: dict[str, str], tags: list[Tag], source_builds: list[str]
) -> tuple[list[Pick], list[str]]:
    """Picks of every resolved package and the problems (no file, a source build that is not an exception, a stale
    exception)."""
    picks: list[Pick] = []
    problems: list[str] = []
    exceptions = {canonicalize_name(n) for n in source_builds}
    for name, version in sorted(resolved.items()):
        package = locked_package(packages, name, version)
        if package is None:
            problems.append(f"{name}=={version} is resolved but not locked")
            continue
        chosen = pick(package, tags)
        picks.append(chosen)
        if name in exceptions:
            if chosen.kind != "sdist":
                problems.append(f"{name} has an aarch64 wheel now ({chosen.file}): drop it from JETSON_SOURCE_BUILDS")
        elif chosen.kind in ("sdist", "missing"):
            problems.append(f"{name}=={version}: no wheel for the Jetson and not in JETSON_SOURCE_BUILDS")
    return picks, problems


def table_rows(packages: dict[str, list[dict]], tags: list[Tag]) -> list[str]:
    """The markdown rows of the audited packages (``AUDIT``), from the lock alone."""
    rows = []
    for name, required_by, note in AUDIT:
        package = locked_package(packages, name)
        if package is None:
            rows.append(f"| {name} | not locked | {required_by} | — | — | — | {note or '—'} |")
            continue
        chosen = pick(package, tags)
        file = f"`{chosen.file}`" if chosen.file else "—"
        rows.append(
            f"| {name} | {chosen.version} | {required_by} | {KIND_LABELS[chosen.kind]} | {file} | "
            f"{chosen.glibc or '—'} | {note or '—'} |"
        )
    return rows


def render_block(
    packages: dict[str, list[dict]],
    resolved: dict[str, str],
    tags: list[Tag],
    platform: str,
    source_builds: list[str],
    uv_version: str,
    day: dt.date,
) -> str:
    picks, problems = audit(packages, resolved, tags, source_builds)
    counts = {kind: sum(p.kind == kind for p in picks) for kind in KIND_LABELS}
    floors = {p.name: p.glibc for p in picks if p.glibc}
    top_glibc = max(floors.values(), key=lambda v: tuple(map(int, v.split("."))), default="—")
    at_top = sorted(name for name, floor in floors.items() if floor == top_glibc)
    locked = {name for name, entries in packages.items() for p in entries if "virtual" not in p.get("source", {})}
    absent = sorted(locked - set(resolved) - {canonicalize_name(n) for n in _local_packages(packages)})
    lines = [
        MARK_START,
        f"Recipe of `make lock-check` run by hand on {day:%d %b %Y} with uv {uv_version}: `uv lock --check` passed; "
        f"`uv pip compile` resolved {len(resolved)} packages for `{platform}` and CPython {PYTHON[0]}.{PYTHON[1]} "
        f"with `--only-binary :all:` except {', '.join(source_builds) or 'none'}"
        + (": no problem." if not problems else f": {len(problems)} problem(s)."),
        "",
        "| File installed on the Jetson | Packages |",
        "|---|---|",
        *(f"| {KIND_LABELS[k]} | {n} |" for k, n in counts.items() if n or k != "missing"),
        "",
        f"Highest glibc floor of the selected wheels: {top_glibc} ({', '.join(at_top) or '—'}); the Jetson has 2.35. "
        f"Locked but not installed on linux-aarch64 (markers for other platforms): {', '.join(absent) or 'none'}. "
        f"Local packages (built from this repository): {', '.join(_local_packages(packages)) or 'none'}.",
        *(f"- PROBLEM: {problem}" for problem in problems),
        "",
        "| Package | Version | Required by | File kind | File installed on the Jetson | glibc floor | Notes |",
        "|---|---|---|---|---|---|---|",
        *table_rows(packages, tags),
        MARK_END,
    ]
    return "\n".join(lines) + "\n"


def _local_packages(packages: dict[str, list[dict]]) -> list[str]:
    return sorted(p["name"] for entries in packages.values() for p in entries if "editable" in p.get("source", {}))


def replace_block(text: str, block: str) -> str:
    """``text`` with the lines from the start marker to the end marker replaced by ``block``."""
    start, end = text.find(MARK_START), text.find(MARK_END)
    if start < 0 or end < start:
        raise ValueError(f"markers {MARK_START} / {MARK_END} missing")
    return text[:start] + block + text[end + len(MARK_END) :].removeprefix("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--uv", default=UV_DEFAULT, help=f"uv command of the lock rule (default: {UV_DEFAULT})")
    ap.add_argument("--write", action="store_true", help=f"replace the block of {DOC.relative_to(ROOT)}")
    args = ap.parse_args(argv)

    platform, source_builds = makefile_settings()
    tags = target_tags(platform)
    packages = load_lock()
    try:
        resolved, uv_version = run_recipe(shlex.split(args.uv), platform, source_builds)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    block = render_block(packages, resolved, tags, platform, source_builds, uv_version, dt.date.today())
    print(block, end="")
    if args.write:
        DOC.write_text(replace_block(DOC.read_text(encoding="utf-8"), block), encoding="utf-8", newline="\n")
        print(f"written: {DOC.relative_to(ROOT)}")
    return 1 if "- PROBLEM:" in block else 0


if __name__ == "__main__":
    raise SystemExit(main())
