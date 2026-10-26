"""Worker entry point. Skeleton of #270: idles until the job queue and schedule of #277 are registered."""

from __future__ import annotations

import logging
import time

log = logging.getLogger("worker")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log.info("twin-worker skeleton: no jobs registered yet (job queue and schedule: #277)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log.info("twin-worker stopped")


if __name__ == "__main__":
    main()
