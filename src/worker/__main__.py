"""twin-worker entry point: ``PYTHONPATH=src python -m worker`` (Redis from ``BF_REDIS_URL``)."""

from __future__ import annotations

import logging

from arq import run_worker


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("arq").setLevel(logging.WARNING)  # job lines come from worker.runtime
    from worker.settings import WorkerSettings

    run_worker(WorkerSettings)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
