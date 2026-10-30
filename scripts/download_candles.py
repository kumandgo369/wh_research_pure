import argparse
import logging
from pathlib import Path

import polars as pl

from wh_research_pure.data_util.hyperliquid_data import (
    INTERVALS,
    download_all_symbols_candles_max,
)

logger = logging.getLogger(__name__)

# Project root is one level above scripts/
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KEY = ["symbol", "bar_end"]


def merge_candles(old, new):
    """Combine old and new candles, keeping the new version of any duplicate."""
    return (
        pl.concat([old, new])
        .unique(subset=KEY, keep="last", maintain_order=True)
        .sort(KEY)
    )


def warn_on_gaps(old, new, interval):
    """Warn when new data doesn't reach back to the end of existing data."""
    step = pl.duration(minutes=INTERVALS[interval])
    gaps = (
        old.group_by("symbol")
        .agg(pl.col("bar_end").max().alias("old_last"))
        .join(
            new.group_by("symbol").agg(pl.col("bar_end").min().alias("new_first")),
            on="symbol",
        )
        .filter(pl.col("new_first") > pl.col("old_last") + step)
    )
    for row in gaps.iter_rows(named=True):
        logger.warning(
            "%s: gap between %s and %s",
            row["symbol"],
            row["old_last"],
            row["new_first"],
        )


def write_atomic(df, path):
    """Write to a temp file first, so a crash never leaves a half-written file."""
    tmp_path = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def update_candles(interval):
    """
    Download the latest candles for one interval and merge them into data/.
    Returns the number of rows written, or 0 if there was nothing to write.
    Callers that orchestrate several downloads use this instead of main(), so
    they configure logging once themselves.
    """
    new = download_all_symbols_candles_max(interval)

    if new.is_empty():
        logger.warning("No candles downloaded, nothing written")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"hl_candles_{interval}.parquet"

    if out_path.exists():
        old = pl.read_parquet(out_path)
        warn_on_gaps(old, new, interval)
        df = merge_candles(old, new)
        logger.info(
            "Merged: %d existing + %d downloaded -> %d rows",
            old.height,
            new.height,
            df.height,
        )
    else:
        df = new
        logger.info("No existing file, creating new one")

    write_atomic(df, out_path)
    logger.info(
        "Wrote %d rows (%d symbols) to %s", df.height, df["symbol"].n_unique(), out_path
    )
    return df.height


def main():
    parser = argparse.ArgumentParser(
        description="Download max Hyperliquid candles for all perp symbols"
    )
    parser.add_argument("--interval", default="1h", choices=INTERVALS.keys())
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-symbol debug logs"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    update_candles(args.interval)


if __name__ == "__main__":
    main()
