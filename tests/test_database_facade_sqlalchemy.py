"""Behavioral tests for the SQLAlchemy-backed legacy Database facade."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from src.data.db import Database


def _sqlite_url() -> tuple[str, Path]:
    db_name = f"db_facade_test_{uuid4().hex}.db"
    db_path = Path.cwd() / db_name
    return f"sqlite:///{db_path.as_posix()}", db_path


def test_auth_seed_and_user_registration_flow() -> None:
    """Database should seed admin and enforce unique usernames."""
    db_url, db_path = _sqlite_url()
    db = Database(db_url=db_url)
    try:

        admin_row = db.validate_user("admin", "admin123")
        assert admin_row == ("admin", "admin")

        db.add_user("qa_user", "pass123", "user")
        assert db.validate_user("qa_user", "pass123") == ("qa_user", "user")

        with pytest.raises(ValueError):
            db.add_user("qa_user", "pass123", "user")
    finally:
        db.dispose()
        if db_path.exists():
            db_path.unlink()


def test_hopper_history_scd_type2_update() -> None:
    """Hopper material update should close old row and insert new active row."""
    db_url, db_path = _sqlite_url()
    db = Database(db_url=db_url)
    try:
        assert db.hoppers, "Expected hopper list from materials.yml."
        hopper = db.hoppers[0]

        current_map = db.get_current_hopper_materials()
        original_material = current_map[hopper]

        next_material = (
            db.materials[0]
            if db.materials and db.materials[0] != original_material
            else "UNASSIGNED"
        )
        from_time = datetime.utcnow()

        db.update_hopper_material_with_time(
            hopper=hopper,
            material=next_material,
            from_time=from_time,
            modifier="qa",
            ip_address="127.0.0.1",
        )

        updated_map = db.get_current_hopper_materials()
        assert updated_map[hopper] == next_material

        at_new_time = db.get_hopper_material_at(hopper, from_time + timedelta(seconds=1))
        assert at_new_time == next_material

        history = [
            row for row in db.get_hopper_material_history() if row["hopper"] == hopper
        ]
        assert len(history) >= 2
        assert history[0]["material"] == next_material
        assert history[1]["material"] == original_material
        assert history[1]["valid_upto"] is not None
    finally:
        db.dispose()
        if db_path.exists():
            db_path.unlink()


def test_burden_history_update_current_values_and_delete() -> None:
    """Burden field update should keep current value query and support deletion."""
    db_url, db_path = _sqlite_url()
    db = Database(db_url=db_url)
    try:
        now = datetime.utcnow()

        db.update_burden_field(
            field_name="COKE_CHARGE_PATTERN",
            value="1,2,3",
            valid_from=now,
            modifier="qa",
            ip="127.0.0.1",
        )
        db.update_burden_field(
            field_name="TOP_CHARGES/HRS.",
            value=8.5,
            valid_from=now + timedelta(minutes=1),
            modifier="qa",
            ip="127.0.0.1",
        )

        values = db.get_all_current_burden_values(now + timedelta(minutes=2))
        assert values["COKE_CHARGE_PATTERN"] == "1,2,3"
        assert float(values["TOP_CHARGES/HRS."]) == 8.5

        history = db.get_burden_history()
        delete_ids = [row["id"] for row in history]
        db.delete_burden_history(delete_ids)
        assert db.get_burden_history() == []
    finally:
        db.dispose()
        if db_path.exists():
            db_path.unlink()
