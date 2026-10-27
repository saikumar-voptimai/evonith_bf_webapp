from reports.furnace_report.timeframe import REPORT_TYPES, get_report_timeframe

__all__ = ["REPORT_TYPES", "ShiftReportService", "get_report_timeframe"]


def __getattr__(name: str):
    # The service imports the data sources (InfluxDB, Neon clients): loaded on first use, so that the timeframe, data,
    # analyser and renderer modules stay light for the twin-api (services.reports, SV-2.3).
    if name == "ShiftReportService":
        from reports.furnace_report.service import ShiftReportService

        return ShiftReportService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
