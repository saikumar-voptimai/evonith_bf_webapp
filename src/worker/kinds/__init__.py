"""Worker implementations of the job kinds in ``services.jobs.CATALOG``; importing this package registers them."""

from worker.kinds import calibration, forecasts, learn, optimiser, system, twin  # noqa: F401
