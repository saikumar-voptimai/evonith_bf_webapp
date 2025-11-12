import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from utils.helper_functions_AD_system.vector_store import fetch_latest_record_by_namespace



class BFAnomalyController:
    """
    Manages anomaly reports in the vector database.
    Provides:
      • Fetching latest record by namespace
      • 8-hour shift eligibility check
      • Remaining-time calculation
      • Streamlit feedback messages
    """

    def __init__(self, shift_hours=8):

        self.shift_hours = shift_hours

    
    def get_last_report_time(self, namespace: str):
        record = fetch_latest_record_by_namespace(namespace)
        if not record or "timestamp" not in record:
            return None

        ts = str(record["timestamp"]).strip()
        ts_norm = ts.replace("Z", "+00:00")

        # Convert hyphens in time part to colons
        import re
        ts_norm = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", ts_norm)

        try:
            dt = datetime.fromisoformat(ts_norm)
        except Exception:
            for fmt in ("%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%dT%H-%M-%S"):
                try:
                    dt = datetime.strptime(ts_norm, fmt)
                    break
                except Exception:
                    dt = None
            if dt is None:
                print(f"⚠️ Unparseable timestamp '{ts}' even after normalization.")
                return None

        # ✅ Assign Asia/Kolkata if tzinfo missing
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))

        # Already local time → no conversion
        return dt


    
    # def get_last_report_time(self, namespace: str):
    #     """
    #     Reads the latest record for the namespace and returns a timezone-aware
    #     datetime in Asia/Kolkata. Handles ISO-8601 with/without subseconds and with Z/offset.
    #     """
    #     record = fetch_latest_record_by_namespace(namespace)
    #     if not record or "timestamp" not in record:
    #         return None

    #     ts = str(record["timestamp"]).strip()

    #     # Normalize 'Z' to '+00:00' so fromisoformat can parse it
    #     ts_norm = ts.replace("Z", "+00:00")

    #     try:
    #         dt = datetime.fromisoformat(ts_norm)  # supports subseconds and offsets
    #     except Exception:
    #         # Fallbacks for a few common shapes
    #         for fmt in ("%Y-%m-%dT%H:%M:%S",
    #                     "%Y-%m-%d %H:%M:%S",
    #                     "%Y-%m-%dT%H:%M:%S%z"):
    #             try:
    #                 dt = datetime.strptime(ts, fmt)
    #                 break
    #             except Exception:
    #                 dt = None
    #         if dt is None:
    #             # Last resort: assume UTC naive if everything else fails
    #             try:
    #                 dt = datetime.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    #             except Exception as e:
    #                 print(f"⚠️ Unparseable timestamp '{ts}': {e}")
    #                 return None

    #     # Ensure tz-aware in UTC
    #     if dt.tzinfo is None:
    #         dt = dt.replace(tzinfo=timezone.utc)

    #     # Convert to Asia/Kolkata for your shift logic
    #     return dt.astimezone(ZoneInfo("Asia/Kolkata"))

    
    def can_generate_new_report(self, last_time: datetime):
        """
        Returns (True/False, remaining_timedelta)
        True if >= shift_hours have elapsed since last report.
        """
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        diff = now - last_time
        if diff >= timedelta(hours=self.shift_hours):
            return True, timedelta(0)
        remaining = timedelta(hours=self.shift_hours) - diff
        return False, remaining

  
    def display_wait_message(self, remaining: timedelta):
        """
        Display user-friendly wait message in Streamlit.
        """
        hours, remainder = divmod(remaining.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        st.warning(
            f"⏳ Please try again after {int(hours)} h {int(minutes)} m.\n"
            f"A new anomaly report can be generated only once every {self.shift_hours} hours."
        )



default_controller = BFAnomalyController()

def get_last_report_time(namespace: str):
    return default_controller.get_last_report_time(namespace)

def can_generate_new_report(last_time: datetime):
    return default_controller.can_generate_new_report(last_time)

def display_wait_message(remaining: timedelta):
    return default_controller.display_wait_message(remaining)