import argparse
import logging
from pathlib import Path

import polars as pl

from wh_research_pure.data_util.hyperliquid_data import download_all_symbols_funding

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = DATA_DIR / "hl_funding.parquet"
KEY = ["symbol", "time"]


def write_atomic(df, path):
    tmp_path = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def update_funding():
    """
    Download funding entries newer than what is stored and merge them into
    data/. Returns the number of rows written, or 0 if there was nothing new.
    Callers that orchestrate several downloads use this instead of main(), so
    they configure logging once themselves.
    """
    old = pl.read_parquet(OUT_PATH) if OUT_PATH.exists() else None

    # Last stored timestamp per symbol, so we only fetch what's new
    last_times = {}
    if old is not None:
        last_times = dict(old.group_by("symbol").agg(pl.col("time").max()).iter_rows())

    new = download_all_symbols_funding(last_times)

    if new.is_empty():
        logger.info("No new funding data")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if old is not None:
        df = (
            pl.concat([old, new])
            .unique(subset=KEY, keep="last", maintain_order=True)
            .sort(KEY)
        )
        logger.info(
            "Merged: %d existing + %d new -> %d rows", old.height, new.height, df.height
        )
    else:
        df = new

    write_atomic(df, OUT_PATH)
    logger.info(
        "Wrote %d rows (%d symbols) to %s", df.height, df["symbol"].n_unique(), OUT_PATH
    )
    return df.height


def main():
    parser = argparse.ArgumentParser(
        description="Download Hyperliquid funding history for all perp symbols"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-symbol debug logs"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    update_funding()


if __name__ == "__main__":
    main()
