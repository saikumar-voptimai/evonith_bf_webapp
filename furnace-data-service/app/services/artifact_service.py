"""Runtime-backed artifact storage for data and dataset downloads."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from furnace_data.runtime_paths import get_dataset_results_dir


_ARTIFACT_DIRNAME = "artifacts"
_SAFE_ID = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_id: str
    filename: str
    content_type: str
    row_count: int | None
    created_at: str
    expires_at: str | None = None


def _artifact_root() -> Path:
    path = get_dataset_results_dir() / _ARTIFACT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._") or "artifact"
    return cleaned[:120]


def _metadata_path(artifact_id: str) -> Path:
    return _artifact_root() / f"{artifact_id}.json"


def get_artifact_path(artifact_id: str) -> Path:
    if not _SAFE_ID.match(artifact_id):
        raise ValueError("Invalid artifact id")
    metadata = get_artifact_metadata(artifact_id)
    path = (_artifact_root() / metadata.filename).resolve()
    root = _artifact_root().resolve()
    if root not in path.parents and path != root:
        raise ValueError("Invalid artifact path")
    return path


def get_artifact_metadata(artifact_id: str) -> ArtifactMetadata:
    if not _SAFE_ID.match(artifact_id):
        raise ValueError("Invalid artifact id")
    path = _metadata_path(artifact_id)
    if not path.exists():
        raise FileNotFoundError(artifact_id)
    return ArtifactMetadata(**json.loads(path.read_text(encoding="utf-8")))


def create_csv_artifact(
    dataframe: pd.DataFrame,
    filename_prefix: str,
    *,
    ttl_hours: int = 24,
) -> ArtifactMetadata:
    artifact_id = uuid.uuid4().hex
    filename = f"{artifact_id}_{sanitize_filename(filename_prefix)}.csv"
    path = _artifact_root() / filename
    dataframe.to_csv(path, index=True)
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(hours=ttl_hours) if ttl_hours > 0 else None
    metadata = ArtifactMetadata(
        artifact_id=artifact_id,
        filename=filename,
        content_type="text/csv",
        row_count=len(dataframe),
        created_at=created_at.isoformat(),
        expires_at=expires_at.isoformat() if expires_at else None,
    )
    _metadata_path(artifact_id).write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return metadata


def delete_expired_artifacts(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    deleted = 0
    for metadata_file in _artifact_root().glob("*.json"):
        try:
            metadata = ArtifactMetadata(**json.loads(metadata_file.read_text(encoding="utf-8")))
            if not metadata.expires_at:
                continue
            expires_at = datetime.fromisoformat(metadata.expires_at)
            if expires_at > now:
                continue
            artifact_path = _artifact_root() / metadata.filename
            if artifact_path.exists():
                artifact_path.unlink()
            metadata_file.unlink()
            deleted += 1
        except Exception:
            continue
    return deleted
