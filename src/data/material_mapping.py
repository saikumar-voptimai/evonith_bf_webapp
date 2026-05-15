"""Client material-name mapping for hopper/admin workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "materials_map.yml"


def normalize_client_material_name(value: str) -> str:
    """Normalize a client material display name for exact map lookup."""
    value = str(value).replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value.strip())
    return value.casefold()


@dataclass(frozen=True)
class MaterialMapEntry:
    client_name: str
    material_name: str
    is_primary: bool = False


class MaterialNameMapper:
    """Map client display names to canonical plant_master material names."""

    def __init__(self, entries: Iterable[MaterialMapEntry]) -> None:
        self.entries = list(entries)
        by_client: dict[str, MaterialMapEntry] = {}
        primary_by_material: dict[str, MaterialMapEntry] = {}

        for entry in self.entries:
            normalized = normalize_client_material_name(entry.client_name)
            if normalized in by_client:
                raise ValueError(f"Duplicate client material mapping: {entry.client_name}")
            by_client[normalized] = entry
            if entry.is_primary:
                existing = primary_by_material.get(entry.material_name)
                if existing is not None:
                    raise ValueError(
                        "Multiple primary client names for "
                        f"{entry.material_name}: {existing.client_name}, {entry.client_name}"
                    )
                primary_by_material[entry.material_name] = entry

        self._by_client = by_client
        self._primary_by_material = primary_by_material

    @classmethod
    def from_file(cls, path: str | Path = _DEFAULT_PATH) -> "MaterialNameMapper":
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh) or {}
        entries = [
            MaterialMapEntry(
                client_name=str(row["client_name"]),
                material_name=str(row["material_name"]),
                is_primary=bool(row.get("is_primary", False)),
            )
            for row in data.get("materials", [])
        ]
        return cls(entries)

    @property
    def client_names(self) -> list[str]:
        return [entry.client_name for entry in self.entries]

    def material_name_for_client(self, client_name: str) -> str:
        normalized = normalize_client_material_name(client_name)
        entry = self._by_client.get(normalized)
        if entry is None:
            raise ValueError(f"Unknown client material name: {client_name}")
        return entry.material_name

    def primary_client_name_for_material(self, material_name: str) -> str:
        entry = self._primary_by_material.get(material_name)
        return entry.client_name if entry else material_name

    def validate_material_names(self, active_material_names: set[str]) -> None:
        missing = sorted(
            {entry.material_name for entry in self.entries}
            - set(active_material_names)
        )
        if missing:
            raise ValueError(
                "materials_map.yml references inactive/missing material_name(s): "
                + ", ".join(missing)
            )
