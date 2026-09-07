"""Run one bounded Jetson pass over queued scheduled-job control commands.

This command is intended for ``furnacemind-control-sync.timer``. It opens the
shared PostgreSQL database, claims commands addressed to this Jetson, delegates
them to the existing systemd provisioning service, records outcomes, and exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# The privileged timer must consume only its root-controlled EnvironmentFile.
# Set this before importing application modules so a repository-local developer
# ``.env`` file cannot influence systemd control operations.
os.environ["FURNACEMIND_DISABLE_DOTENV"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data.scheduled_tasks import (  # noqa: E402
    ScheduledJobCommandConflictError,
    ScheduledJobCommandProcessor,
    ScheduledJobCommandService,
    ScheduledJobPersistenceError,
    ScheduledJobProvisioningError,
    ScheduledJobService,
    ScheduledJobValidationError,
)
from scripts.furnacemind_systemd_provisioner import (  # noqa: E402
    build_provisioning_service_from_environment,
    database_url_from_environment,
)

LOGGER = logging.getLogger("furnacemind.scheduled_job_control_sync")


def _positive_environment_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Return a bounded positive integer from deployment configuration."""

    raw = environment.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _argument_parser() -> argparse.ArgumentParser:
    """Build the stable one-shot command synchronization interface."""

    parser = argparse.ArgumentParser(
        prog="furnacemind-control-sync",
        description="Process queued scheduled-job controls for this Jetson and exit.",
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=25,
        help="Maximum commands processed in this invocation (1-100)",
    )
    parser.add_argument(
        "--target-device-id",
        help="Override SCHEDULED_TASKS_TARGET_DEVICE_ID",
    )
    parser.add_argument(
        "--worker-id",
        help="Stable worker label used only for command lease diagnostics",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Process one bounded queue batch and return a stable process exit code."""

    args = _argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    values = os.environ if environment is None else environment
    target_device_id = args.target_device_id or values.get(
        "SCHEDULED_TASKS_TARGET_DEVICE_ID", "bf2-jetson-01"
    )
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"
    command_service: ScheduledJobCommandService | None = None
    job_service: ScheduledJobService | None = None
    provisioner = None
    try:
        if not 1 <= args.max_commands <= 100:
            raise ValueError("--max-commands must be between 1 and 100.")
        database_url = database_url_from_environment(values)
        command_service = ScheduledJobCommandService(db_url=database_url)
        job_service = ScheduledJobService(db_url=database_url)
        provisioner = build_provisioning_service_from_environment(values)
        processor = ScheduledJobCommandProcessor(
            command_service=command_service,
            job_service=job_service,
            provisioning_service=provisioner,
            target_device_id=target_device_id,
            worker_id=worker_id,
            lease_seconds=_positive_environment_integer(
                values,
                "SCHEDULED_TASKS_COMMAND_LEASE_SECONDS",
                300,
                minimum=5,
                maximum=3600,
            ),
            retry_delay_seconds=_positive_environment_integer(
                values,
                "SCHEDULED_TASKS_COMMAND_RETRY_SECONDS",
                30,
                minimum=1,
                maximum=3600,
            ),
        )
        result = processor.process_batch(max_commands=args.max_commands)
    except (
        ScheduledJobCommandConflictError,
        ScheduledJobPersistenceError,
        ScheduledJobProvisioningError,
        ScheduledJobValidationError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Scheduled-job control synchronization failed: %s", exc)
        return 1
    finally:
        if provisioner is not None:
            provisioner.dispose()
        if job_service is not None:
            job_service.dispose()
        if command_service is not None:
            command_service.dispose()

    print(
        json.dumps(
            {
                "target_device_id": result.target_device_id,
                "claimed": result.claimed,
                "succeeded": result.succeeded,
                "retrying": result.retrying,
                "failed": result.failed,
                "clean": result.clean,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
