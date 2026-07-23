"""Dataclasses used by the shared V-Board domain layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


SectionStatus = Literal["ok", "partial", "empty", "unavailable"]


@dataclass(frozen=True)
class ResolvedRange:
    """A requested V-Board range resolved to UTC timestamps."""

    start: datetime
    end: datetime
    requested_kind: str
    preset_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "start": _iso_z(self.start),
            "end": _iso_z(self.end),
            "requested_kind": self.requested_kind,
            "preset_id": self.preset_id,
        }


@dataclass(frozen=True)
class ResolutionWindow:
    """Server-allowlisted time-series resolution."""

    id: str
    label: str
    seconds: int


def _iso_z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
