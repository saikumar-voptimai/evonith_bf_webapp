# FurnaceMind/memory/structured_store.py
# Purpose: SQLite-backed structured store for operational summaries.
# Fixed: Replaced JSON files with SQLite for concurrency safety,
#        atomic writes, and O(1) lookups instead of full-file scans.

import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timezone

from FurnaceMind.memory.schemas import ShiftSummary


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StructuredStore:
    """
    SQLite-based structured store for operational summaries.
    Thread-safe, atomic, and performant at scale.
    """

    def __init__(self, base_dir: str = "src/FurnaceMind/data/structured"):
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.base_path / "furnacemind.db"
        self._local = threading.local()
        self._init_tables()

    # ------------------------------------------------------------------
    # Connection management (thread-safe)
    # ------------------------------------------------------------------
    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30,
                check_same_thread=False,
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_tables(self):
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS shift_summaries (
                shift_id TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                shift_start TEXT,
                shift_end TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_summaries (
                window_id TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS weekly_summaries (
                window_id TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS biweekly_summaries (
                window_id TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_shift_start
                ON shift_summaries(shift_start);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Migration helper: import existing JSON data
    # ------------------------------------------------------------------
    def migrate_from_json(self):
        """
        One-time migration from old JSON files to SQLite.
        Safe to call multiple times (uses INSERT OR IGNORE).
        """
        json_files = {
            "shift_summaries": self.base_path / "shift_summaries.json",
            "daily_summaries": self.base_path / "daily_summaries.json",
            "weekly_summaries": self.base_path / "weekly_summaries.json",
            "biweekly_summaries": self.base_path / "biweekly_summaries.json",
        }

        conn = self._conn

        # Shifts
        path = json_files["shift_summaries"]
        if path.exists():
            records = json.loads(path.read_text())
            for r in records:
                sid = r.get("shift_id")
                if sid:
                    conn.execute(
                        "INSERT OR IGNORE INTO shift_summaries (shift_id, data_json, shift_start, shift_end) VALUES (?, ?, ?, ?)",
                        (sid, json.dumps(r, default=str), r.get("shift_start"), r.get("shift_end")),
                    )

        # Generic summaries
        for table, key_field in [
            ("daily_summaries", "window_id"),
            ("weekly_summaries", "window_id"),
            ("biweekly_summaries", "window_id"),
        ]:
            path = json_files[table]
            if path.exists():
                records = json.loads(path.read_text())
                for r in records:
                    wid = r.get(key_field)
                    if wid:
                        conn.execute(
                            f"INSERT OR IGNORE INTO {table} (window_id, data_json) VALUES (?, ?)",
                            (wid, json.dumps(r, default=str)),
                        )

        conn.commit()

    # ------------------------------------------------------------------
    # Shift write operations
    # ------------------------------------------------------------------
    def save_shift_summary(self, summary: ShiftSummary) -> None:
        data = summary.model_dump(exclude_none=True)
        self._conn.execute(
            """INSERT OR IGNORE INTO shift_summaries
               (shift_id, data_json, shift_start, shift_end)
               VALUES (?, ?, ?, ?)""",
            (
                summary.shift_id,
                json.dumps(data, default=str),
                summary.shift_start.isoformat() if summary.shift_start else None,
                summary.shift_end.isoformat() if summary.shift_end else None,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Shift read operations
    # ------------------------------------------------------------------
    def load_shift_summary(self, shift_id: str) -> Optional[ShiftSummary]:
        return self.get_shift_by_id(shift_id)

    def load_latest_shift_summary(self) -> Optional[ShiftSummary]:
        row = self._conn.execute(
            "SELECT data_json FROM shift_summaries ORDER BY shift_start DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return ShiftSummary(**json.loads(row["data_json"]))

    def load_all_shift_summaries(self) -> List[ShiftSummary]:
        rows = self._conn.execute(
            "SELECT data_json FROM shift_summaries ORDER BY shift_start ASC"
        ).fetchall()
        return [ShiftSummary(**json.loads(r["data_json"])) for r in rows]

    def load_last_n_shift_summaries(self, n: int) -> List[ShiftSummary]:
        rows = self._conn.execute(
            "SELECT data_json FROM shift_summaries ORDER BY shift_start DESC LIMIT ?",
            (n,),
        ).fetchall()
        # Return in ascending order
        summaries = [ShiftSummary(**json.loads(r["data_json"])) for r in rows]
        summaries.reverse()
        return summaries

    def get_shift_by_id(self, shift_id: str) -> Optional[ShiftSummary]:
        row = self._conn.execute(
            "SELECT data_json FROM shift_summaries WHERE shift_id = ?",
            (shift_id,),
        ).fetchone()
        if not row:
            return None
        return ShiftSummary(**json.loads(row["data_json"]))

    def get_shift(self, shift_id: str) -> Optional[ShiftSummary]:
        return self.get_shift_by_id(shift_id)

    def get_shifts_for_day(self, day_id: str) -> List[ShiftSummary]:
        rows = self._conn.execute(
            "SELECT data_json FROM shift_summaries WHERE shift_start LIKE ? ORDER BY shift_start ASC",
            (f"{day_id}%",),
        ).fetchall()
        return [ShiftSummary(**json.loads(r["data_json"])) for r in rows]

    def list_shifts_for_date(self, d: date) -> List[str]:
        day_id = d.isoformat()
        shifts = self.get_shifts_for_day(day_id)
        return [s.shift_id for s in shifts]

    # ------------------------------------------------------------------
    # Daily summaries
    # ------------------------------------------------------------------
    def daily_exists(self, day_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM daily_summaries WHERE window_id = ?", (day_id,)
        ).fetchone()
        return row is not None

    def save_daily_summary(
        self,
        *,
        day_id: str,
        summary_text: str,
        structured: Dict[str, Any],
    ) -> None:
        record = {
            "window_id": day_id,
            "generated_at": _utc_now_iso(),
            "summary_text": summary_text,
            **structured,
        }
        self._conn.execute(
            "INSERT OR IGNORE INTO daily_summaries (window_id, data_json) VALUES (?, ?)",
            (day_id, json.dumps(record, default=str)),
        )
        self._conn.commit()

    def load_all_daily_summaries(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM daily_summaries ORDER BY window_id ASC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def get_daily_by_id(self, day_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT data_json FROM daily_summaries WHERE window_id = ?", (day_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def get_daily_for_week(self, week_id: str) -> List[Dict[str, Any]]:
        all_daily = self.load_all_daily_summaries()
        out: List[Dict[str, Any]] = []
        for r in all_daily:
            try:
                d = date.fromisoformat(r["window_id"])
            except Exception:
                continue
            iso_year, iso_week, _ = d.isocalendar()
            if f"{iso_year}-W{iso_week:02d}" == week_id:
                out.append(r)
        out.sort(key=lambda r: r.get("window_id", ""))
        return out

    # ------------------------------------------------------------------
    # Weekly summaries
    # ------------------------------------------------------------------
    def weekly_exists(self, week_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM weekly_summaries WHERE window_id = ?", (week_id,)
        ).fetchone()
        return row is not None

    def save_weekly_summary(
        self,
        *,
        week_id: str,
        summary_text: str,
        structured: Dict[str, Any],
    ) -> None:
        record = {
            "window_id": week_id,
            "generated_at": _utc_now_iso(),
            "summary_text": summary_text,
            **structured,
        }
        self._conn.execute(
            "INSERT OR IGNORE INTO weekly_summaries (window_id, data_json) VALUES (?, ?)",
            (week_id, json.dumps(record, default=str)),
        )
        self._conn.commit()

    def load_all_weekly_summaries(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM weekly_summaries ORDER BY window_id ASC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def get_weekly_by_id(self, week_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT data_json FROM weekly_summaries WHERE window_id = ?", (week_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def get_weeks_for_biweek(self, biweek_id: str) -> List[Dict[str, Any]]:
        all_weekly = self.load_all_weekly_summaries()
        out: List[Dict[str, Any]] = []
        for r in all_weekly:
            week_id = r.get("window_id")
            if not week_id or "-W" not in week_id:
                continue
            year_str, w_str = week_id.split("-W")
            year = int(year_str)
            week = int(w_str)
            biweek_idx = (week + 1) // 2
            if f"{year}-BW{biweek_idx:02d}" == biweek_id:
                out.append(r)
        out.sort(key=lambda r: r.get("window_id", ""))
        return out

    # ------------------------------------------------------------------
    # Biweekly summaries
    # ------------------------------------------------------------------
    def biweekly_exists(self, biweek_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM biweekly_summaries WHERE window_id = ?", (biweek_id,)
        ).fetchone()
        return row is not None

    def save_biweekly_summary(
        self,
        *,
        biweek_id: str,
        summary_text: str,
        structured: Dict[str, Any],
    ) -> None:
        record = {
            "window_id": biweek_id,
            "generated_at": _utc_now_iso(),
            "summary_text": summary_text,
            **structured,
        }
        self._conn.execute(
            "INSERT OR IGNORE INTO biweekly_summaries (window_id, data_json) VALUES (?, ?)",
            (biweek_id, json.dumps(record, default=str)),
        )
        self._conn.commit()

    def load_all_biweekly_summaries(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM biweekly_summaries ORDER BY window_id ASC"
        ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def get_biweekly_by_id(self, biweek_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT data_json FROM biweekly_summaries WHERE window_id = ?", (biweek_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    # ------------------------------------------------------------------
    # Unified read API (UI uses this)
    # ------------------------------------------------------------------
    def get_report(
        self,
        *,
        level: str,
        window_id: str,
    ) -> Optional[Dict[str, Any]]:
        if level == "shift":
            shift = self.get_shift_by_id(window_id)
            if not shift:
                return None
            return {
                "window_id": shift.shift_id,
                "summary_text": getattr(shift, "summary_text", None),
                "generated_at": shift.generated_at.isoformat() if shift.generated_at else None,
                "structured": shift.model_dump(),
            }

        if level == "day":
            return self.get_daily_by_id(window_id)
        if level == "week":
            return self.get_weekly_by_id(window_id)
        if level == "biweek":
            return self.get_biweekly_by_id(window_id)

        raise ValueError(f"Unknown report level: {level}")