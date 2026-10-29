"""Worker implementations of the job kinds in ``services.jobs.CATALOG``; importing this package registers them."""

from worker.kinds import (  # noqa: F401
    bmo,
    calibration,
    forecasts,
    forge,
    learn,
    optimiser,
    reports,
    strata,
    system,
    trace,
    twin,
    whatif,
)
