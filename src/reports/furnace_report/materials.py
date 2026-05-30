"""Material-code alias resolution for furnace reports."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Mapping, Sequence

import pandas as pd

from data.material_mapping import MaterialNameMapper

logger = logging.getLogger(__name__)


def material_code_candidates(
    charge_col: str,
    configured: Mapping[str, Sequence[str]],
) -> list[str]:
    base = charge_col.removesuffix("_mt")
    return list(dict.fromkeys(configured.get(charge_col, (base, base.replace("_", "")))))


def display_material_code(material_code: str) -> str:
    return material_code.replace("_", "")


@dataclass(frozen=True)
class MaterialAliasResolver:
    """Resolve charge-column material aliases to client-facing material names."""

    material_names: Mapping[str, str]
    mapper: MaterialNameMapper | None = None

    @classmethod
    def from_dataframe(cls, materials_df: pd.DataFrame) -> "MaterialAliasResolver":
        return cls(_material_name_lookup(materials_df), _load_mapper())

    def display_name(self, material_codes: Sequence[str]) -> str:
        for code in material_codes:
            material_name = self.material_names.get(code) or self.material_names.get(
                code.casefold()
            )
            if material_name:
                return self._display_material_name(material_name)
        return display_material_code(material_codes[0]) if material_codes else "-"

    def _display_material_name(self, material_name: str) -> str:
        if self.mapper is None:
            return material_name
        return self.mapper.primary_client_name_for_material(material_name)


@lru_cache(maxsize=1)
def _load_mapper() -> MaterialNameMapper | None:
    try:
        return MaterialNameMapper.from_file()
    except Exception:
        logger.warning("Unable to load materials_map.yml", exc_info=True)
        return None


def _material_name_lookup(materials_df: pd.DataFrame) -> dict[str, str]:
    if materials_df.empty or not {"material_code", "material_name"}.issubset(
        materials_df.columns
    ):
        return {}

    df = materials_df.copy()
    if "is_active" in df.columns:
        df = df[df["is_active"].fillna(True).astype(bool)]

    lookup: dict[str, str] = {}
    for _, row in df.dropna(subset=["material_code", "material_name"]).iterrows():
        code = str(row["material_code"])
        name = str(row["material_name"])
        code_without_underscores = code.replace("_", "")
        aliases = (
            code,
            code.casefold(),
            code_without_underscores,
            code_without_underscores.casefold(),
        )
        for alias in dict.fromkeys(aliases):
            lookup.setdefault(alias, name)
    return lookup
