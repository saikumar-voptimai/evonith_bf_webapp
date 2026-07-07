"""
In-memory task registry for background dataset operations.

Tasks are stored in a dict — lost on restart, which is acceptable
for a fetch-and-download service on a Raspberry Pi.
"""

import logging
import traceback
import uuid
from datetime import date, datetime
from pathlib import Path
from threading import Thread
from typing import Callable, Optional

import httpx
import pandas as pd

from apps.backend_api.app.config import settings
from apps.backend_api.app.models.schemas import TaskStatus

log = logging.getLogger(__name__)


class TaskState:
    def __init__(self, task_id: str, callback_url: Optional[str] = None):
        self.task_id = task_id
        self.status: TaskStatus = TaskStatus.pending
        self.progress: Optional[str] = None
        self.created_at: datetime = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.result_path: Optional[Path] = None
        self.rows: Optional[int] = None
        self.columns: Optional[int] = None
        self.error: Optional[str] = None
        self.callback_url: Optional[str] = callback_url


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, TaskState] = {}

    def create_task(self, callback_url: Optional[str] = None) -> TaskState:
        task_id = uuid.uuid4().hex[:12]
        task = TaskState(task_id, callback_url)
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    def run_in_background(self, task: TaskState, fn: Callable, *args, **kwargs):
        """Run fn in a background thread, updating task state on completion."""

        def _wrapper():
            task.status = TaskStatus.running
            try:
                fn(task, *args, **kwargs)
                task.status = TaskStatus.completed
                task.completed_at = datetime.now(datetime.UTC)
                log.info("Task %s completed: %d rows", task.task_id, task.rows or 0)
            except Exception as e:
                task.status = TaskStatus.failed
                task.completed_at = datetime.now(datetime.UTC)
                task.error = str(e)
                log.error("Task %s failed: %s\n%s", task.task_id, e, traceback.format_exc())
            finally:
                self._cleanup_old_results()
                self._notify_callback(task)

        thread = Thread(target=_wrapper, daemon=True)
        thread.start()

    def _notify_callback(self, task: TaskState):
        """POST task status to callback URL if configured."""
        if not task.callback_url:
            return
        try:
            payload = {
                "task_id": task.task_id,
                "status": task.status.value,
                "rows": task.rows,
                "columns": task.columns,
                "error": task.error,
            }
            with httpx.Client(timeout=10) as client:
                resp = client.post(task.callback_url, json=payload)
                log.info("Callback to %s returned %d", task.callback_url, resp.status_code)
        except Exception as e:
            log.warning("Callback notification failed: %s", e)

    def _cleanup_old_results(self):
        """Keep only the N most recent result CSVs."""
        results_dir = settings.results_dir
        if not results_dir.exists():
            return

        csv_files = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

        for old_file in csv_files[settings.max_result_files:]:
            try:
                old_file.unlink()
                log.info("Cleaned up old result: %s", old_file.name)
            except OSError as e:
                log.warning("Failed to delete %s: %s", old_file.name, e)

    def save_result(self, task: TaskState, df: pd.DataFrame) -> Path:
        """Save a DataFrame as CSV and update task metadata."""
        settings.results_dir.mkdir(parents=True, exist_ok=True)
        path = settings.results_dir / f"{task.task_id}.csv"
        df.to_csv(path, index=True)
        task.result_path = path
        task.rows = len(df)
        task.columns = len(df.columns)
        return path


# Singleton
task_manager = TaskManager()
