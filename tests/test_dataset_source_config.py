from __future__ import annotations

import config.config_loader as config_loader


def test_dataset_url_is_disabled_when_fetch_url_is_false(monkeypatch) -> None:
    monkeypatch.setattr(
        config_loader,
        "load_config",
        lambda name="setting_ds_dv.yml": {
            "furnacemind.yml": {"fetch_url": "false"},
            "setting_ds_dv.yml": {"DATA_URL": "http://example.test/data.csv"},
        }[name],
    )

    assert config_loader.get_furnace_dataset_url() is None


def test_dataset_url_is_returned_when_fetch_url_is_true(monkeypatch) -> None:
    monkeypatch.setattr(
        config_loader,
        "load_config",
        lambda name="setting_ds_dv.yml": {
            "furnacemind.yml": {"fetch_url": "true"},
            "setting_ds_dv.yml": {"DATA_URL": "http://example.test/data.csv"},
        }[name],
    )

    assert config_loader.get_furnace_dataset_url() == (
        "http://example.test/data.csv"
    )


def test_dataset_url_override_is_also_gated_by_fetch_url(monkeypatch) -> None:
    monkeypatch.setattr(
        config_loader,
        "load_config",
        lambda _name="setting_ds_dv.yml": {"fetch_url": False},
    )

    assert (
        config_loader.get_furnace_dataset_url(
            {"DATA_URL": "http://example.test/data.csv"},
            override_url="http://example.test/override.csv",
        )
        is None
    )
