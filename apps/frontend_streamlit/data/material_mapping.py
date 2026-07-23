"""Compatibility re-export for shared material-name mapping."""

from furnace_data.material_mapping import (
    MaterialMapEntry,
    MaterialNameMapper,
    normalize_client_material_name,
)

__all__ = [
    "MaterialMapEntry",
    "MaterialNameMapper",
    "normalize_client_material_name",
]
