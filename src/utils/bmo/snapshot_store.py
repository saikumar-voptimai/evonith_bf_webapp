"""Where snapshots live: one JSON file each, in the repository.

``src/storage/bmo_snapshots/<YYYYmmdd_HHMMSS>_<source>.json``, beside the other
JSON state already kept under ``src/storage``. One file per snapshot keeps them
individually readable, diffable and deletable, and a corrupt file costs one
snapshot rather than the whole history.

Writes are atomic - written to a temporary name and renamed into place - so a
crash mid-write cannot leave a half file that breaks the listing.

ON STREAMLIT CLOUD the container filesystem is ephemeral: files written at
runtime survive until the app restarts and are never pushed back to git. Locally
they persist and can be committed. Swapping this module for a database-backed one
later needs no change to the panel or the page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STORE_DIR = Path(__file__).resolve().parents[2] / "storage" / "bmo_snapshots"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "snapshot"


def _stamp(created_at: str) -> str:
    digits = re.sub(r"[^0-9]", "", created_at)[:14]
    return f"{digits[:8]}_{digits[8:14]}" if len(digits) >= 14 else "undated"


def save(snapshot: dict[str, Any], directory: Path | None = None) -> Path:
    """Write a snapshot, assigning it an id that does not collide with any other."""

    folder = Path(directory or STORE_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{_stamp(snapshot.get('created_at', ''))}_{_slug(snapshot.get('source', ''))}"
    snapshot_id, n = base, 1
    while (folder / f"{snapshot_id}.json").exists():
        n += 1
        snapshot_id = f"{base}_{n}"
    snapshot["id"] = snapshot_id

    target = folder / f"{snapshot_id}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return target


def load(snapshot_id: str, directory: Path | None = None) -> dict[str, Any]:
    path = Path(directory or STORE_DIR) / f"{Path(snapshot_id).stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def delete(snapshot_id: str, directory: Path | None = None) -> bool:
    path = Path(directory or STORE_DIR) / f"{Path(snapshot_id).stem}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_snapshots(directory: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Every readable snapshot's header, newest first, plus files that failed.

    Only the header fields are returned - id, time, source, label, summary - so
    the table stays cheap however many snapshots accumulate. Unreadable files are
    reported by name instead of being skipped silently.
    """

    folder = Path(directory or STORE_DIR)
    if not folder.is_dir():
        return [], []
    rows: list[dict[str, Any]] = []
    broken: list[str] = []
    for path in sorted(folder.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "id": doc.get("id") or path.stem,
                "created_at": doc.get("created_at", ""),
                "source": doc.get("source", ""),
                "label": doc.get("label", ""),
                "summary": doc.get("summary") or {},
                "path": str(path),
            })
        except (OSError, ValueError):
            broken.append(path.name)
    rows.sort(key=lambda r: (r["created_at"], r["id"]), reverse=True)
    return rows, broken
