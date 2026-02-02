# memory/structured_store.py

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, date

from FurnaceMind.memory.schemas import ShiftSummary


class StructuredStore:
    """
    File-based structured store for operational summaries.
    Stores shift summaries + aggregated daily/weekly/bi-weekly summaries.
    """

    def __init__(self, base_dir: str = "FurnaceMind/data/structured"):
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.shift_file = self.base_path / "shift_summaries.json"
        self.daily_file = self.base_path / "daily_summaries.json"
        self.weekly_file = self.base_path / "weekly_summaries.json"
        self.biweekly_file = self.base_path / "biweekly_summaries.json"

        for f in [
            self.shift_file,
            self.daily_file,
            self.weekly_file,
            self.biweekly_file,
        ]:
            if not f.exists():
                self._write_file(f, [])



    # Shift write operations
    def save_shift_summary(self, summary: ShiftSummary) -> None:
        records = self._read_file(self.shift_file)

        # Idempotency
        if any(r.get("shift_id") == summary.shift_id for r in records):
            return

        records.append(
            summary.model_dump(exclude_none=True)
        )

        self._write_file(self.shift_file, records)




    # Shift read operations
    def load_shift_summary(self, shift_id: str) -> Optional[ShiftSummary]:
        """
        Compatibility alias for callers expecting load_shift_summary().
        """
        return self.get_shift_by_id(shift_id)
    

    def load_latest_shift_summary(self) -> Optional[ShiftSummary]:
        """
        Load the most recent shift summary by shift_start time.
        """
        shifts = self.load_all_shift_summaries()
        if not shifts:
            return None
        return max(shifts, key=lambda s: s.shift_start)


    def load_all_shift_summaries(self) -> List[ShiftSummary]:
        records = self._read_file(self.shift_file)
        return [ShiftSummary(**r) for r in records]
    

    def load_last_n_shift_summaries(self, n: int) -> List[ShiftSummary]:
        """
        Load the most recent N shift summaries by shift_start time.
        """
        shifts = self.load_all_shift_summaries()
        # Guard: empty store
        if not shifts:
            return []
        
        # Sort by shift_start (ascending)
        shifts.sort(key=lambda s: s.shift_start)
        # Return last N
        return shifts[-n:]


    def get_shift_by_id(self, shift_id: str) -> Optional[ShiftSummary]:
        records = self._read_file(self.shift_file)
        for r in records:
            if r.get("shift_id") == shift_id:
                return ShiftSummary(**r)
        return None


    # UI-friendly alias
    def get_shift(self, shift_id: str) -> Optional[ShiftSummary]:
        return self.get_shift_by_id(shift_id)

    def get_shifts_for_day(self, day_id: str) -> List[ShiftSummary]:
        """
        day_id format: YYYY-MM-DD
        """
        shifts = [
            s for s in self.load_all_shift_summaries()
            if s.shift_start.date().isoformat() == day_id
        ]
        shifts.sort(key=lambda s: s.shift_start)
        return shifts

    def list_shifts_for_date(self, d: date) -> List[str]:
        """
        Returns shift_ids for a given date, sorted by start time.
        """
        day_id = d.isoformat()
        shifts = self.get_shifts_for_day(day_id)
        return [s.shift_id for s in shifts]



    # Daily summaries
    def daily_exists(self, day_id: str) -> bool:
        return any(
            r.get("window_id") == day_id
            for r in self._read_file(self.daily_file)
        )


    def save_daily_summary(
        self,
        *,
        day_id: str,
        summary_text: str,
        structured: Dict[str, Any],
    ) -> None:
        records = self._read_file(self.daily_file)
        if any(r.get("window_id") == day_id for r in records):
            return

        records.append(
            {
                "window_id": day_id,
                "generated_at": datetime.utcnow().isoformat(),
                "summary_text": summary_text,
                **structured,
            }
        )
        self._write_file(self.daily_file, records)


    def load_all_daily_summaries(self) -> List[Dict[str, Any]]:
        records = self._read_file(self.daily_file)
        records.sort(key=lambda r: r.get("window_id", ""))
        return records


    def get_daily_by_id(self, day_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read_file(self.daily_file):
            if r.get("window_id") == day_id:
                return r
        return None


    def get_daily_for_week(self, week_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for r in self.load_all_daily_summaries():
            try:
                d = date.fromisoformat(r["window_id"])
            except Exception:
                continue

            iso_year, iso_week, _ = d.isocalendar()
            if f"{iso_year}-W{iso_week:02d}" == week_id:
                out.append(r)

        out.sort(key=lambda r: r.get("window_id", ""))
        return out


 
    # Weekly summaries
    def weekly_exists(self, week_id: str) -> bool:
        return any(
            r.get("window_id") == week_id
            for r in self._read_file(self.weekly_file)
        )

    def save_weekly_summary(
        self,
        *,
        week_id: str,
        summary_text: str,
        structured: Dict[str, Any],
    ) -> None:
        records = self._read_file(self.weekly_file)
        if any(r.get("window_id") == week_id for r in records):
            return

        records.append(
            {
                "window_id": week_id,
                "generated_at": datetime.utcnow().isoformat(),
                "summary_text": summary_text,
                **structured,
            }
        )
        self._write_file(self.weekly_file, records)

    def load_all_weekly_summaries(self) -> List[Dict[str, Any]]:
        records = self._read_file(self.weekly_file)
        records.sort(key=lambda r: r.get("window_id", ""))
        return records

    def get_weekly_by_id(self, week_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read_file(self.weekly_file):
            if r.get("window_id") == week_id:
                return r
        return None

    def get_weeks_for_biweek(self, biweek_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for r in self.load_all_weekly_summaries():
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



    # Bi-weekly summaries
    def biweekly_exists(self, biweek_id: str) -> bool:
        return any(
            r.get("window_id") == biweek_id
            for r in self._read_file(self.biweekly_file)
        )


    def save_biweekly_summary(
        self,
        *,
        biweek_id: str,
        summary_text: str,
        structured: Dict[str, Any],
    ) -> None:
        records = self._read_file(self.biweekly_file)
        if any(r.get("window_id") == biweek_id for r in records):
            return

        records.append(
            {
                "window_id": biweek_id,
                "generated_at": datetime.utcnow().isoformat(),
                "summary_text": summary_text,
                **structured,
            }
        )
        self._write_file(self.biweekly_file, records)


    def load_all_biweekly_summaries(self) -> List[Dict[str, Any]]:
        records = self._read_file(self.biweekly_file)
        records.sort(key=lambda r: r.get("window_id", ""))
        return records


    def get_biweekly_by_id(self, biweek_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read_file(self.biweekly_file):
            if r.get("window_id") == biweek_id:
                return r
        return None



    # Unified read API (UI uses this)
    def get_report(
        self,
        *,
        level: str,
        window_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        level: shift | day | week | biweek
        """
        if level == "shift":
            shift = self.get_shift_by_id(window_id)
            if not shift:
                return None
            return {
                "window_id": shift.shift_id,
                "summary_text": getattr(shift, "summary_text", None),
                "generated_at": shift.generated_at.isoformat(),
                "structured": shift.model_dump(),
            }

        if level == "day":
            return self.get_daily_by_id(window_id)

        if level == "week":
            return self.get_weekly_by_id(window_id)

        if level == "biweek":
            return self.get_biweekly_by_id(window_id)

        raise ValueError(f"Unknown report level: {level}")



    # Internal helpers
    
    @staticmethod
    def _read_file(path: Path) -> List[dict]:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _write_file(path: Path, data: List[dict]) -> None:
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.replace(path)