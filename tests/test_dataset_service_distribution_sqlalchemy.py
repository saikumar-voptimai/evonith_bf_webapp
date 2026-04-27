"""Regression tests for DatasetService burden-distribution SQLAlchemy query path."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from furnace_data.dataset.service import DatasetService
from furnace_data.relational import (
    Base,
    BurdenDistributionHistory,
    build_relational_engine,
)


def _sqlite_url() -> tuple[str, Path]:
    db_name = f"dataset_distribution_test_{uuid4().hex}.db"
    db_path = Path.cwd() / db_name
    return f"sqlite:///{db_path.as_posix()}", db_path


def test_fetch_distribution_data_returns_expected_windowed_rows() -> None:
    """DatasetService should return burden rows built from ORM query results."""
    db_url, db_path = _sqlite_url()
    engine = None
    try:
        engine = build_relational_engine(db_url)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    BurdenDistributionHistory(
                        field_name="COKE_RINGS_1",
                        field_value_float=4.0,
                        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        valid_upto=datetime(2026, 1, 3, tzinfo=timezone.utc),
                    ),
                    BurdenDistributionHistory(
                        field_name="COKE_ANGLES_1",
                        field_value_float=30.0,
                        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        valid_upto=datetime(2026, 1, 3, tzinfo=timezone.utc),
                    ),
                    BurdenDistributionHistory(
                        field_name="BURDEN_CHANGING_PURPOSE",
                        field_value_text="stability",
                        valid_from=datetime(2026, 1, 2, tzinfo=timezone.utc),
                        valid_upto=None,
                    ),
                ]
            )
            session.commit()

        service = DatasetService(db_url=db_url)
        output = service.fetch_distribution_data(
            start_date=date(2026, 1, 2),
            end_date=date(2026, 1, 3),
        )

        assert not output.empty
        assert "COKE_RINGS_1" in output.columns
        assert "COKE_ANGLES_1" in output.columns
        assert "BURDEN_CHANGING_PURPOSE" in output.columns
        assert "TOTAL_COKE_PORTIONS" in output.columns
        assert "WEIGHTED_COKE_ANGLE" in output.columns

        day = output.loc[datetime(2026, 1, 2)]
        assert float(day["COKE_RINGS_1"]) == 4.0
        assert float(day["COKE_ANGLES_1"]) == 30.0
        assert str(day["BURDEN_CHANGING_PURPOSE"]) == "stability"
        assert float(day["TOTAL_COKE_PORTIONS"]) == 4.0
        assert float(day["WEIGHTED_COKE_ANGLE"]) == 30.0
    finally:
        if engine is not None:
            engine.dispose()
        if db_path.exists():
            db_path.unlink()
