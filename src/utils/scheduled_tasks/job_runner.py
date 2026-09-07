"""One-shot production runner for database-backed scheduled jobs.

Each systemd activation starts this module for exactly one job identifier. The
runner validates the stored definition, calculates the latest due occurrence,
claims that occurrence with a database lease, executes it through the restricted
FurnaceMind adapter, and stores one terminal result. It deliberately contains no
polling loop and has no Streamlit dependency.

Leases are renewed in a small background heartbeat while model, data-source, or
retry-wait work is in progress. Every retry and terminal write is fenced by the
current opaque lease token and attempt number, so a delayed process cannot write
after another worker has recovered the occurrence.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import FrameType
from typing import Protocol

from yaml import YAMLError

from data.scheduled_tasks import (
    ScheduledJobExecutionOutput,
    ScheduledJobExecutionService,
    ScheduledJobNotActiveError,
    ScheduledJobNotFoundError,
    ScheduledJobOccurrenceAlreadyClaimedError,
    ScheduledJobPersistenceError,
    ScheduledJobRunLeaseBusyError,
    ScheduledJobRunStateError,
    ScheduledJobRunView,
    ScheduledJobValidationError,
    ScheduledJobView,
    safe_scheduled_job_error_message,
)
from furnace_data.relational import ScheduledJobRunStatus
from utils.scheduled_tasks.furnacemind_executor import (
    FurnaceMindScheduledTaskExecutor,
    ScheduledTaskProductionValidationError,
    validate_production_definition,
)
from utils.scheduled_tasks.schedule_occurrence import (
    ScheduledOccurrenceError,
    ScheduledOccurrenceNotDueError,
    latest_scheduled_occurrence,
)
from utils.scheduled_tasks.scheduled_task_definition import validate_task_definition

LOGGER = logging.getLogger("furnacemind.scheduled_job_runner")

_DEFAULT_LEASE_SECONDS = 90
_DEFAULT_HEARTBEAT_SECONDS = 30.0
_MAX_RETRY_CLOCK_PROBES = 3


class ScheduledTaskExecutor(Protocol):
    """Execution adapter used by the one-shot runner."""

    def execute(
        self,
        *,
        job: ScheduledJobView,
        scheduled_for: datetime,
    ) -> ScheduledJobExecutionOutput:
        """Execute one claimed occurrence and return serializable output."""


class ValidationOnlyExecutor:
    """Validate a stored definition without claiming or executing an occurrence."""

    def execute(
        self,
        *,
        job: ScheduledJobView,
        scheduled_for: datetime,
    ) -> ScheduledJobExecutionOutput:
        """Validate both the JSON contract and the production execution policy."""

        if not isinstance(job.definition, dict):
            raise ValueError("Stored scheduled-job definition must be a JSON object.")
        errors = list(validate_task_definition(job.definition))
        errors.extend(validate_production_definition(job.definition))
        unique_errors = tuple(dict.fromkeys(errors))
        if unique_errors:
            raise ScheduledTaskProductionValidationError(unique_errors)
        return ScheduledJobExecutionOutput(
            output_type="validation_receipt",
            content="Scheduled-job definition and production runner wiring validated.",
            content_json={
                "mode": "validation_only",
                "job_id": job.job_id,
                "job_type": job.definition.get("job_type"),
                "schema_version": job.schema_version,
                "validated_at": scheduled_for.isoformat(),
                "definition_valid": True,
                "production_policy_valid": True,
            },
            metadata={"executor": "validation_only"},
        )


class ScheduledTaskAttemptTimedOut(TimeoutError):
    """Raised when an executor exceeds its configured attempt deadline."""


class ScheduledTaskTerminationRequested(RuntimeError):
    """Raised when the service receives a cooperative termination request."""


class ScheduledTaskExecutionFailed(RuntimeError):
    """Raised after all configured executor attempts have failed."""

    def __init__(self, run: ScheduledJobRunView) -> None:
        """Store the terminal failed run for CLI reporting."""

        self.run = run
        super().__init__(
            f"Scheduled run {run.run_id} failed after "
            f"{run.attempt_number} attempt(s)."
        )


class ScheduledTaskExecutionCancelled(RuntimeError):
    """Raised after a termination request has durably cancelled an owned run."""

    def __init__(self, run: ScheduledJobRunView) -> None:
        """Store the cancelled run for process-level reporting."""

        self.run = run
        super().__init__(f"Scheduled run {run.run_id} was cancelled.")


@dataclass(frozen=True, slots=True)
class ScheduledJobRunnerResult:
    """Outcome returned after completion or an idempotent duplicate claim."""

    run: ScheduledJobRunView
    output_id: str | None
    duplicate: bool = False


def _execute_with_timeout(
    executor: ScheduledTaskExecutor,
    job: ScheduledJobView,
    scheduled_for: datetime,
    timeout_seconds: int,
) -> ScheduledJobExecutionOutput:
    """Execute with a POSIX alarm on the Linux systemd target.

    Windows does not provide ``SIGALRM``; unit-test execution on Windows runs
    directly. Production deployment is Linux/systemd, where the alarm interrupts
    a blocked executor in the main thread and the database deadline remains the
    authoritative recovery boundary.
    """

    if (
        not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return executor.execute(job=job, scheduled_for=scheduled_for)

    def _raise_timeout(_signum: int, _frame: FrameType | None) -> None:
        """Interrupt the current attempt when its real-time alarm expires."""

        raise ScheduledTaskAttemptTimedOut(
            f"Execution attempt exceeded {timeout_seconds} seconds."
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_alarm = signal.alarm(timeout_seconds)
    try:
        return executor.execute(job=job, scheduled_for=scheduled_for)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_alarm:
            signal.alarm(previous_alarm)


def _utc_timestamp(value: datetime) -> datetime:
    """Normalize a database timestamp, treating SQLite-naive values as UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _lease_token(run: ScheduledJobRunView) -> str:
    """Return the current fencing token or reject a non-owned run view."""

    if not run.lease_token:
        raise ScheduledJobRunStateError(
            "The scheduled-job run does not have an active execution lease."
        )
    return run.lease_token


def scheduled_job_result_exit_code(result: ScheduledJobRunnerResult) -> int:
    """Return 2 for terminal work failures so systemd does not retry them."""

    failure_statuses = {
        ScheduledJobRunStatus.FAILED.value,
        ScheduledJobRunStatus.TIMED_OUT.value,
        ScheduledJobRunStatus.CANCELLED.value,
    }
    return 2 if result.run.status in failure_statuses else 0


def _duplicate_result(run: ScheduledJobRunView) -> ScheduledJobRunnerResult:
    """Return an idempotent result or surface its existing terminal failure."""

    result = ScheduledJobRunnerResult(run=run, output_id=None, duplicate=True)
    if scheduled_job_result_exit_code(result):
        raise ScheduledTaskExecutionFailed(run)
    return result


class _RunLeaseKeeper:
    """Renew one fenced run lease while slow work happens outside transactions."""

    def __init__(
        self,
        *,
        execution_service: ScheduledJobExecutionService,
        run: ScheduledJobRunView,
        lease_seconds: int,
        heartbeat_seconds: float,
        cancellation_requested: Callable[[], bool],
    ) -> None:
        """Capture immutable ownership and heartbeat dependencies for one stage."""

        _lease_token(run)
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive.")
        if heartbeat_seconds <= 0 or heartbeat_seconds >= lease_seconds:
            raise ValueError(
                "heartbeat_seconds must be positive and shorter than the lease."
            )
        self._execution_service = execution_service
        self._run = run
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._cancellation_requested = cancellation_requested
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._failure: Exception | None = None
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"scheduled-job-lease-{run.run_id}",
            daemon=True,
        )

    def start(self) -> None:
        """Start renewal after checking cancellation and existing ownership."""

        self.checkpoint()
        self._thread.start()

    def checkpoint(self) -> None:
        """Raise promptly when cancellation or heartbeat ownership loss occurred."""

        if self._cancellation_requested():
            raise ScheduledTaskTerminationRequested(
                "Scheduled-job service termination was requested."
            )
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def snapshot(self) -> ScheduledJobRunView:
        """Return the latest safely published lease view."""

        with self._lock:
            return self._run

    def stop(self) -> ScheduledJobRunView:
        """Stop renewal, wait briefly for an in-flight renewal, and verify state."""

        self._stop_event.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise ScheduledJobPersistenceError(
                "The scheduled-job lease heartbeat did not stop cleanly."
            )
        self.checkpoint()
        return self.snapshot()

    def _heartbeat_loop(self) -> None:
        """Renew ownership periodically until stopped or a transition fails."""

        while not self._stop_event.wait(self._heartbeat_seconds):
            if self._cancellation_requested():
                with self._lock:
                    self._failure = ScheduledTaskTerminationRequested(
                        "Scheduled-job service termination was requested."
                    )
                return
            run = self.snapshot()
            try:
                renewed = self._execution_service.renew_lease(
                    run_id=run.run_id,
                    expected_lease_token=_lease_token(run),
                    expected_attempt_number=run.attempt_number,
                    lease_seconds=self._lease_seconds,
                )
            except Exception as exc:
                with self._lock:
                    self._failure = exc
                return
            with self._lock:
                self._run = renewed


class ScheduledJobRunner:
    """Run one active scheduled-job occurrence and persist its lifecycle."""

    def __init__(
        self,
        *,
        execution_service: ScheduledJobExecutionService,
        executor: ScheduledTaskExecutor,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], datetime] | None = None,
        execute_attempt: Callable[
            [ScheduledTaskExecutor, ScheduledJobView, datetime, int],
            ScheduledJobExecutionOutput,
        ] = _execute_with_timeout,
        cancellation_requested: Callable[[], bool] | None = None,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
        heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        """Create a runner with injected persistence, execution, and timing hooks."""

        self._execution_service = execution_service
        self._executor = executor
        self._sleep = sleep
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._execute_attempt = execute_attempt
        self._cancellation_requested = cancellation_requested or (lambda: False)
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds

    def run(
        self,
        job_id: str,
        *,
        now: datetime | None = None,
        triggered_by: str = "systemd",
    ) -> ScheduledJobRunnerResult:
        """Execute the latest due occurrence of one active scheduled job."""

        job = self._execution_service.load_active_job(job_id)
        production_errors = validate_production_definition(job.definition)
        if production_errors:
            raise ScheduledTaskProductionValidationError(production_errors)
        scheduled_for = latest_scheduled_occurrence(job.definition, at=now)
        assert job.activated_at is not None
        if _utc_timestamp(scheduled_for) < _utc_timestamp(job.activated_at):
            raise ScheduledOccurrenceNotDueError(
                "No scheduled occurrence is due since this job was activated."
            )

        retry = job.definition["retry"]
        assert isinstance(retry, dict)
        maximum_attempts = retry["maximum_attempts"]
        retry_interval_seconds = retry["retry_interval_seconds"]
        timeout_seconds = retry["timeout_seconds"]
        assert isinstance(maximum_attempts, int)
        assert isinstance(retry_interval_seconds, int)
        assert isinstance(timeout_seconds, int)
        reference = now or self._wall_clock()
        stale_before = _utc_timestamp(reference) - timedelta(seconds=timeout_seconds)

        run = self._resolve_or_claim_run(
            job=job,
            scheduled_for=scheduled_for,
            stale_before=stale_before,
            triggered_by=triggered_by,
            maximum_attempts=maximum_attempts,
            retry_interval_seconds=retry_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
        if run.status != ScheduledJobRunStatus.RUNNING.value:
            return _duplicate_result(run)
        scheduled_for = _utc_timestamp(run.scheduled_for)

        try:
            while run.attempt_number <= maximum_attempts:
                if run.retry_not_before_at is not None:
                    run = self._begin_retry_when_due(
                        run,
                        timeout_seconds=timeout_seconds,
                    )
                try:
                    run, output = self._execute_owned_attempt(
                        run=run,
                        job=job,
                        scheduled_for=scheduled_for,
                        timeout_seconds=timeout_seconds,
                    )
                except (
                    ScheduledJobPersistenceError,
                    ScheduledJobRunStateError,
                    ScheduledJobNotFoundError,
                ):
                    raise
                except ScheduledTaskTerminationRequested:
                    raise
                except Exception as exc:
                    LOGGER.error(
                        "Execution attempt failed: job_id=%s run_id=%s "
                        "attempt=%s error_type=%s error=%s",
                        job.job_id,
                        run.run_id,
                        run.attempt_number,
                        type(exc).__name__,
                        safe_scheduled_job_error_message(exc),
                    )
                    if run.attempt_number < maximum_attempts:
                        if not isinstance(exc, ScheduledTaskAttemptTimedOut):
                            run = self._renew_owned_lease(run)
                        run = self._execution_service.schedule_retry(
                            run_id=run.run_id,
                            expected_lease_token=_lease_token(run),
                            expected_attempt_number=run.attempt_number,
                            previous_error=exc,
                            retry_interval_seconds=retry_interval_seconds,
                            lease_seconds=self._lease_seconds,
                        )
                        continue
                    if not isinstance(exc, ScheduledTaskAttemptTimedOut):
                        run = self._renew_owned_lease(run)
                    run = self._execution_service.fail_run(
                        run_id=run.run_id,
                        expected_lease_token=_lease_token(run),
                        expected_attempt_number=run.attempt_number,
                        error=exc,
                        timed_out=isinstance(exc, ScheduledTaskAttemptTimedOut),
                    )
                    raise ScheduledTaskExecutionFailed(run) from exc

                run = self._renew_owned_lease(run)
                completion = self._execution_service.complete_run(
                    run_id=run.run_id,
                    expected_lease_token=_lease_token(run),
                    expected_attempt_number=run.attempt_number,
                    output=output,
                )
                LOGGER.info(
                    "Execution completed: job_id=%s run_id=%s output_id=%s",
                    job.job_id,
                    completion.run.run_id,
                    completion.output_id,
                )
                return ScheduledJobRunnerResult(
                    run=completion.run,
                    output_id=completion.output_id,
                )
        except ScheduledTaskTerminationRequested as exc:
            cancelled = self._execution_service.cancel_run(
                run_id=run.run_id,
                expected_lease_token=_lease_token(run),
                expected_attempt_number=run.attempt_number,
                reason=exc,
            )
            raise ScheduledTaskExecutionCancelled(cancelled) from exc

        raise AssertionError("Validated retry configuration produced no attempt.")

    def _resolve_or_claim_run(
        self,
        *,
        job: ScheduledJobView,
        scheduled_for: datetime,
        stale_before: datetime,
        triggered_by: str,
        maximum_attempts: int,
        retry_interval_seconds: int,
        timeout_seconds: int,
    ) -> ScheduledJobRunView:
        """Reuse, reclaim, or create the single run eligible for this process."""

        running_resolution = self._execution_service.recover_or_get_running_run(
            job_id=job.job_id,
            stale_before=stale_before,
            triggered_by=triggered_by,
            maximum_attempts=maximum_attempts,
            timeout_seconds=timeout_seconds,
            lease_seconds=self._lease_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        if running_resolution is not None:
            run, recovered = running_resolution
            if not recovered:
                self._raise_live_lease(run)
            return run

        try:
            claim = self._execution_service.claim_run(
                job_id=job.job_id,
                scheduled_for=scheduled_for,
                triggered_by=triggered_by,
                timeout_seconds=timeout_seconds,
                lease_seconds=self._lease_seconds,
            )
            return claim.run
        except ScheduledJobOccurrenceAlreadyClaimedError as exc:
            existing = exc.existing_run
            if existing.status != ScheduledJobRunStatus.RUNNING.value:
                return existing
            try:
                recovered = self._execution_service.recover_stale_run(
                    run_id=existing.run_id,
                    stale_before=stale_before,
                    triggered_by=triggered_by,
                    maximum_attempts=maximum_attempts,
                    timeout_seconds=timeout_seconds,
                    lease_seconds=self._lease_seconds,
                    retry_interval_seconds=retry_interval_seconds,
                )
            except ScheduledJobRunStateError:
                self._raise_live_lease(existing)
            return recovered

    def _raise_live_lease(self, run: ScheduledJobRunView) -> None:
        """Raise transient contention using the persisted lease deadline."""

        lease_expires_at = run.lease_expires_at or (
            self._wall_clock() + timedelta(seconds=self._lease_seconds)
        )
        raise ScheduledJobRunLeaseBusyError(_utc_timestamp(lease_expires_at))

    def _new_lease_keeper(self, run: ScheduledJobRunView) -> _RunLeaseKeeper:
        """Construct a heartbeat owner for one execution or retry-wait stage."""

        return _RunLeaseKeeper(
            execution_service=self._execution_service,
            run=run,
            lease_seconds=self._lease_seconds,
            heartbeat_seconds=self._heartbeat_seconds,
            cancellation_requested=self._cancellation_requested,
        )

    def _set_executor_activity_callback(
        self,
        callback: Callable[[], None] | None,
    ) -> None:
        """Attach a lease checkpoint when the executor supports the runtime hook."""

        setter = getattr(self._executor, "set_activity_callback", None)
        if callable(setter):
            setter(callback)

    def _execute_owned_attempt(
        self,
        *,
        run: ScheduledJobRunView,
        job: ScheduledJobView,
        scheduled_for: datetime,
        timeout_seconds: int,
    ) -> tuple[ScheduledJobRunView, ScheduledJobExecutionOutput]:
        """Execute one attempt while renewing and checking its fenced lease."""

        keeper = self._new_lease_keeper(run)
        keeper.start()
        self._set_executor_activity_callback(keeper.checkpoint)
        output: ScheduledJobExecutionOutput | None = None
        execution_error: Exception | None = None
        try:
            output = self._execute_attempt(
                self._executor,
                job,
                scheduled_for,
                timeout_seconds,
            )
            self._execution_service.validate_output(output)
            keeper.checkpoint()
        except Exception as exc:
            execution_error = exc
        finally:
            self._set_executor_activity_callback(None)

        try:
            current = keeper.stop()
        except Exception as heartbeat_error:
            if execution_error is not None:
                raise heartbeat_error from execution_error
            raise
        if execution_error is not None:
            raise execution_error
        assert output is not None
        return current, output

    def _renew_owned_lease(self, run: ScheduledJobRunView) -> ScheduledJobRunView:
        """Synchronously prove ownership immediately before a state transition."""

        return self._execution_service.renew_lease(
            run_id=run.run_id,
            expected_lease_token=_lease_token(run),
            expected_attempt_number=run.attempt_number,
            lease_seconds=self._lease_seconds,
        )

    def _wait_with_lease(
        self,
        run: ScheduledJobRunView,
        *,
        not_before: datetime,
    ) -> ScheduledJobRunView:
        """Wait outside the database while a heartbeat preserves ownership."""

        keeper = self._new_lease_keeper(run)
        keeper.start()
        wait_error: Exception | None = None
        try:
            remaining = max(
                0.0,
                (
                    _utc_timestamp(not_before) - _utc_timestamp(self._wall_clock())
                ).total_seconds(),
            )
            if remaining:
                self._sleep(remaining)
            keeper.checkpoint()
        except Exception as exc:
            wait_error = exc

        try:
            current = keeper.stop()
        except Exception as heartbeat_error:
            if wait_error is not None:
                raise heartbeat_error from wait_error
            raise
        if wait_error is not None:
            raise wait_error
        return current

    def _begin_retry_when_due(
        self,
        run: ScheduledJobRunView,
        *,
        timeout_seconds: int,
    ) -> ScheduledJobRunView:
        """Honor a durable retry time, tolerating small app/database clock skew."""

        not_before = run.retry_not_before_at
        assert not_before is not None
        latest_busy: ScheduledJobRunLeaseBusyError | None = None
        for _ in range(_MAX_RETRY_CLOCK_PROBES):
            run = self._wait_with_lease(run, not_before=not_before)
            try:
                return self._execution_service.begin_retry_attempt(
                    run_id=run.run_id,
                    expected_lease_token=_lease_token(run),
                    expected_attempt_number=run.attempt_number,
                    timeout_seconds=timeout_seconds,
                    lease_seconds=self._lease_seconds,
                )
            except ScheduledJobRunLeaseBusyError as exc:
                latest_busy = exc
                not_before = exc.lease_expires_at
        assert latest_busy is not None
        raise latest_busy


def _argument_parser() -> argparse.ArgumentParser:
    """Build the stable command-line contract used by the service template."""

    parser = argparse.ArgumentParser(
        prog="furnacemind-job-runner",
        description="Run one database-backed FurnaceMind scheduled job.",
    )
    parser.add_argument("--job-id", required=True, help="Scheduled job UUID")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the stored definition without claiming an occurrence",
    )
    return parser


def _validation_document(
    *,
    job: ScheduledJobView,
    validated_at: datetime,
) -> dict[str, object]:
    """Return safe read-only CLI output for a validated stored definition."""

    return {
        "job_id": job.job_id,
        "validated_at": validated_at.isoformat(),
        "status": "validated",
        "schema_version": job.schema_version,
        "job_status": job.status,
        "production_policy_valid": True,
        "database_write": False,
    }


def _run_result_document(result: ScheduledJobRunnerResult) -> dict[str, object]:
    """Return a content-free production receipt suitable for journald."""

    return {
        "job_id": result.run.job_id,
        "run_id": result.run.run_id,
        "scheduled_for": _utc_timestamp(result.run.scheduled_for).isoformat(),
        "status": result.run.status,
        "attempt_number": result.run.attempt_number,
        "duplicate": result.duplicate,
        "output_id": result.output_id,
    }


@contextmanager
def _termination_signal_scope(
    termination_event: threading.Event,
) -> Iterator[None]:
    """Convert SIGTERM/SIGINT into a cooperative runner cancellation request."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    supported = tuple(
        candidate
        for candidate in (getattr(signal, "SIGTERM", None), signal.SIGINT)
        if candidate is not None
    )
    previous = {candidate: signal.getsignal(candidate) for candidate in supported}

    def _request_termination(
        signum: int,
        _frame: FrameType | None,
    ) -> None:
        """Mark cancellation and interrupt blocking Python work promptly."""

        termination_event.set()
        raise ScheduledTaskTerminationRequested(
            f"Scheduled-job service received signal {signum}."
        )

    try:
        for candidate in supported:
            signal.signal(candidate, _request_termination)
        yield
    finally:
        for candidate, handler in previous.items():
            signal.signal(candidate, handler)


def main(
    argv: Sequence[str] | None = None,
    *,
    execution_service: ScheduledJobExecutionService | None = None,
    executor: ScheduledTaskExecutor | None = None,
    now: datetime | None = None,
) -> int:
    """Run validation or one production occurrence and return a stable exit code."""

    args = _argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    owns_service = execution_service is None
    termination_event = threading.Event()
    validation_document: dict[str, object] | None = None
    run_result: ScheduledJobRunnerResult | None = None
    try:
        if execution_service is None:
            execution_service = ScheduledJobExecutionService()
        if args.validate_only:
            job = execution_service.load_job(args.job_id)
            validated_at = now or datetime.now(timezone.utc)
            if validated_at.tzinfo is None or validated_at.utcoffset() is None:
                raise ScheduledJobValidationError(
                    ("validation time: timezone-aware timestamp is required",)
                )
            validated_at = validated_at.astimezone(timezone.utc)
            output = ValidationOnlyExecutor().execute(
                job=job,
                scheduled_for=validated_at,
            )
            execution_service.validate_output(output)
            validation_document = _validation_document(
                job=job,
                validated_at=validated_at,
            )
        else:
            production_executor = executor or FurnaceMindScheduledTaskExecutor()
            runner = ScheduledJobRunner(
                execution_service=execution_service,
                executor=production_executor,
                cancellation_requested=termination_event.is_set,
            )
            with _termination_signal_scope(termination_event):
                run_result = runner.run(args.job_id, now=now)
    except ScheduledOccurrenceNotDueError as exc:
        LOGGER.info("Scheduled timer probe had no due occurrence: %s", exc)
        return 0
    except (
        ScheduledJobNotActiveError,
        ScheduledJobNotFoundError,
        ScheduledJobValidationError,
        ScheduledOccurrenceError,
        ScheduledTaskProductionValidationError,
    ) as exc:
        LOGGER.error("Scheduled job was not runnable: %s", exc)
        return 2
    except ScheduledTaskExecutionFailed as exc:
        LOGGER.error("Scheduled job reached a terminal task failure: %s", exc)
        return 2
    except (
        ScheduledJobPersistenceError,
        ScheduledJobRunLeaseBusyError,
        ScheduledJobRunStateError,
        ScheduledTaskExecutionCancelled,
        ScheduledTaskTerminationRequested,
    ) as exc:
        LOGGER.error("Scheduled job stopped with a retryable runtime error: %s", exc)
        return 1
    except (KeyError, OSError, TypeError, ValueError, YAMLError) as exc:
        LOGGER.error(
            "Scheduled job configuration could not be processed (%s).",
            type(exc).__name__,
        )
        return 2
    finally:
        if owns_service and execution_service is not None:
            execution_service.dispose()

    document = validation_document or (
        _run_result_document(run_result) if run_result is not None else None
    )
    assert document is not None
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ScheduledJobRunner",
    "ScheduledJobRunnerResult",
    "ScheduledTaskAttemptTimedOut",
    "ScheduledTaskExecutionCancelled",
    "ScheduledTaskExecutionFailed",
    "ScheduledTaskExecutor",
    "ScheduledTaskTerminationRequested",
    "ValidationOnlyExecutor",
    "main",
    "scheduled_job_result_exit_code",
]
