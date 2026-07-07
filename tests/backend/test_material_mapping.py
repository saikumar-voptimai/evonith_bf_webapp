import pytest

from apps.frontend_streamlit.data.material_mapping import (
    MaterialMapEntry,
    MaterialNameMapper,
    normalize_client_material_name,
)


def test_client_names_resolve_many_to_one_material_name() -> None:
    mapper = MaterialNameMapper(
        [
            MaterialMapEntry("Lloyds CLO", "lloyds_ore", True),
            MaterialMapEntry(
                "Lloyds Metals Iron Fines Oversize (+10MM)",
                "lloyds_oversize",
                True,
            ),
        ]
    )

    assert mapper.material_name_for_client(" Lloyds\u00a0CLO ") == "lloyds_ore"
    assert (
        mapper.material_name_for_client("lloyds metals iron fines oversize (+10mm)")
        == "lloyds_oversize"
    )
    assert mapper.primary_client_name_for_material("lloyds_ore") == "Lloyds CLO"
    assert (
        mapper.primary_client_name_for_material("lloyds_oversize")
        == "Lloyds Metals Iron Fines Oversize (+10MM)"
    )


def test_unknown_client_name_fails_clearly() -> None:
    mapper = MaterialNameMapper([MaterialMapEntry("MN Ore", "mn_ore", True)])

    with pytest.raises(ValueError, match="Unknown client material name"):
        mapper.material_name_for_client("mystery ore")


def test_missing_material_names_fail_validation() -> None:
    mapper = MaterialNameMapper([MaterialMapEntry("MN Ore", "mn_ore", True)])

    with pytest.raises(ValueError, match="mn_ore"):
        mapper.validate_material_names({"lloyds_ore"})


def test_normalization_collapses_nbsp_and_case() -> None:
    assert normalize_client_material_name(" Lloyds\u00a0CLO ") == "lloyds clo"
