"""
Run every Hyperliquid download in sequence.

Each step is independent: if one fails the rest still run, and the exit code is
non-zero so a scheduler notices. Run it as `python scripts/update_data.py`,
which puts this directory on sys.path so the sibling imports below resolve.
"""

import argparse
import logging

from download_candles import update_candles
from download_funding import update_funding

logger = logging.getLogger(__name__)

# Funding runs first so that the candle files, which the delisting cross-check
# in load_funding reads, end up at least as fresh as the funding they validate.
#
# 1h candles run last but matter most: MAX_CANDLES makes them a rolling ~208 day
# window and the API cannot backfill, so a stretch missed for long enough is
# gone permanently. 1d re-downloads its whole history every run and heals
# itself, and funding resumes from the last stored entry per symbol. If the run
# ever has to be cut short, this is the order to reconsider.
STEPS = (
    ("funding", update_funding),
    ("candles 1d", lambda: update_candles("1d")),
    ("candles 1h", lambda: update_candles("1h")),
)


def run_all():
    """
    Run every step, carrying on past failures. Returns {name: rows written},
    with None for a step that raised. KeyboardInterrupt is not caught, so
    Ctrl-C still stops the run.
    """
    results = {}

    for name, step in STEPS:
        logger.info("--- %s ---", name)
        try:
            results[name] = step()
        except Exception:
            logger.exception("%s failed, continuing with the remaining steps", name)
            results[name] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Update funding and candle data, one step at a time"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-symbol debug logs"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run_all()
    failed = [name for name, rows in results.items() if rows is None]

    logger.info("--- summary ---")
    for name, rows in results.items():
        logger.info("%-11s %s", name, "failed" if rows is None else f"{rows:,} rows")

    if failed:
        logger.error(
            "%d of %d steps failed: %s", len(failed), len(STEPS), ", ".join(failed)
        )
        return 1

    logger.info("All %d steps succeeded", len(STEPS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
