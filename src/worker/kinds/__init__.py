"""Worker implementations of the job kinds in ``services.jobs.CATALOG``; importing this package registers them."""

from worker.kinds import forecasts, optimiser, system  # noqa: F401
