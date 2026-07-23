"""Runtime-backed compute artifact storage."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from furnace_data.runtime_paths import runtime_path

_SAFE_ID = re.compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class ComputeArtifactMetadata:
    artifact_id: str
    filename: str
    content_type: str
    workflow: str
    row_count: int | None
    created_at: str
    expires_at: str | None = None
    owner_user_id: str | None = None
    calculation_id: str | None = None
    size_bytes: int | None = None
    checksum_sha256: str | None = None


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip())
    return (cleaned.strip("._") or "compute_artifact")[:120]


class ComputeArtifactService:
    """Write and resolve compute artifacts under EVONITH_RUNTIME_DIR."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    @property
    def root(self) -> Path:
        path = runtime_path("compute", "artifacts")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _metadata_path(self, artifact_id: str) -> Path:
        return self.root / f"{artifact_id}.json"

    def _validate_id(self, artifact_id: str) -> None:
        if not _SAFE_ID.match(str(artifact_id or "")):
            raise ValueError("Invalid compute artifact id")

    def _write_metadata(self, metadata: ComputeArtifactMetadata) -> None:
        self._metadata_path(metadata.artifact_id).write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8")

    def create_csv_artifact(self, *, workflow: str, filename_prefix: str, rows: list[dict[str, Any]], ttl_hours: int | None = None, owner_user_id: str | None = None, calculation_id: str | None = None) -> ComputeArtifactMetadata:
        artifact_id = uuid.uuid4().hex
        filename = f"{artifact_id}_{sanitize_filename(filename_prefix)}.csv"
        path = (self.root / filename).resolve()
        path.relative_to(self.root.resolve())
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return self._finalize_metadata(artifact_id, filename, "text/csv", workflow, len(rows), ttl_hours, owner_user_id, calculation_id)

    def create_json_artifact(self, *, workflow: str, filename_prefix: str, payload: dict[str, Any], ttl_hours: int | None = None, owner_user_id: str | None = None, calculation_id: str | None = None) -> ComputeArtifactMetadata:
        artifact_id = uuid.uuid4().hex
        filename = f"{artifact_id}_{sanitize_filename(filename_prefix)}.json"
        path = (self.root / filename).resolve()
        path.relative_to(self.root.resolve())
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return self._finalize_metadata(artifact_id, filename, "application/json", workflow, None, ttl_hours, owner_user_id, calculation_id)

    def _finalize_metadata(self, artifact_id: str, filename: str, content_type: str, workflow: str, row_count: int | None, ttl_hours: int | None, owner_user_id: str | None, calculation_id: str | None) -> ComputeArtifactMetadata:
        path = (self.root / filename).resolve()
        created_at = datetime.now(timezone.utc)
        ttl = self.settings.compute_artifact_ttl_hours if ttl_hours is None else ttl_hours
        expires_at = created_at + timedelta(hours=ttl) if ttl and ttl > 0 else None
        metadata = ComputeArtifactMetadata(
            artifact_id=artifact_id,
            filename=filename,
            content_type=content_type,
            workflow=sanitize_filename(workflow),
            row_count=row_count,
            created_at=created_at.isoformat(),
            expires_at=expires_at.isoformat() if expires_at else None,
            owner_user_id=owner_user_id,
            calculation_id=calculation_id,
            size_bytes=path.stat().st_size,
            checksum_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self._write_metadata(metadata)
        return metadata

    def get_metadata(self, artifact_id: str) -> ComputeArtifactMetadata:
        self._validate_id(artifact_id)
        path = self._metadata_path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(artifact_id)
        return ComputeArtifactMetadata(**json.loads(path.read_text(encoding="utf-8")))

    def get_path(self, artifact_id: str) -> Path:
        metadata = self.get_metadata(artifact_id)
        path = (self.root / metadata.filename).resolve()
        path.relative_to(self.root.resolve())
        return path

    @staticmethod
    def is_expired(metadata: ComputeArtifactMetadata) -> bool:
        if not metadata.expires_at:
            return False
        expires_at = datetime.fromisoformat(metadata.expires_at)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= expires_at.astimezone(timezone.utc)

    @staticmethod
    def response(metadata: ComputeArtifactMetadata, route_prefix: str) -> dict[str, Any]:
        expires_at = datetime.fromisoformat(metadata.expires_at) if metadata.expires_at else None
        return {
            "artifact_id": metadata.artifact_id,
            "filename": metadata.filename,
            "content_type": metadata.content_type,
            "download_url": f"{route_prefix}/artifacts/{metadata.artifact_id}/download",
            "expires_at": expires_at,
        }