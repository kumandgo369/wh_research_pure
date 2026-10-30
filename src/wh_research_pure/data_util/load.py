"""
Read the raw Parquet files and clean them on the way out.

The files in data/ are never modified, so this is the only place that knows how
a clean frame is built: which cleaning functions to run, in what order, and
which rows to drop. Reach for these instead of pl.read_parquet, so that
knowledge lives in one place rather than in every notebook.
"""

import logging
from pathlib import Path

import polars as pl

from .hyperliquid_clean import (
    flag_after_delisting,
    flag_invalid_candles,
    flag_invalid_funding,
    trim_leading_zero_volume,
)

logger = logging.getLogger(__name__)

# Project root is three levels above this file: data_util -> wh_research_pure -> src
DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Columns the flag functions add, dropped unless the caller wants them.
# funding_hour is not in here: it is the key candles join to funding on.
CANDLE_FLAGS = (
    "bad_high_low",
    "bad_high",
    "bad_low",
    "bad_volume",
    "has_null",
    "is_valid",
)
FUNDING_FLAGS = (
    "funding_offset",
    "has_null",
    "exceeds_cap",
    "off_hour",
    "duplicate_hour",
    "is_valid",
    "after_delisting",
)

# Interval used to cross-check delistings. It has to be one that spans the whole
# history: hl_candles_1h.parquet only holds the last MAX_CANDLES hours, so every
# delisted symbol is missing from it and it would confirm nothing.
DELISTING_CHECK_INTERVAL = "1d"


def _resolve_keep_flags(keep_flags, clean):
    """Default to keeping flags only when rows are not being dropped."""
    return (not clean) if keep_flags is None else keep_flags


def load_candles(interval="1h", clean=True, keep_flags=None, data_dir=DATA_DIR):
    """
    Load candles for one interval, trimmed and checked.

    Each symbol's leading zero-volume candles are dropped first - on the daily
    file those are pre-launch oracle prices, which would otherwise be flagged as
    invalid rather than removed - and the OHLCV checks run after. clean drops
    anything still invalid, which is currently nothing in either file.

    keep_flags defaults to the opposite of clean: a clean frame comes back tidy,
    and an unfiltered one keeps the flags that say what is wrong with it.
    """
    keep_flags = _resolve_keep_flags(keep_flags, clean)

    df = pl.read_parquet(data_dir / f"hl_candles_{interval}.parquet")
    raw_height = df.height

    df = trim_leading_zero_volume(df)
    trimmed = raw_height - df.height

    df = flag_invalid_candles(df)
    invalid = (~df["is_valid"]).sum()

    if clean:
        df = df.filter(pl.col("is_valid"))

    logger.info(
        "%s candles: %d rows, trimmed %d leading zero-volume, %d invalid%s",
        interval,
        df.height,
        trimmed,
        invalid,
        " (dropped)" if clean else "",
    )
    return df if keep_flags else df.drop(CANDLE_FLAGS)


def load_funding(clean=True, keep_flags=None, data_dir=DATA_DIR):
    """
    Load funding history, checked and with post-delisting rows marked.

    Funding has two independent problems. is_valid covers integrity, and
    after_delisting covers relevance: once a perp is delisted the API keeps
    returning an entry every hour forever with funding_rate = 0, and those rows
    pass every integrity check. clean drops both kinds, because leaving the
    placeholders in puts hundreds of thousands of artificial zeros into any
    cross-sectional statistic.

    The delisting check is cross-referenced against daily candles, which stop
    when a perp stops trading. If that file is missing the check falls back to
    the funding data alone, which is weaker, so it says so.

    funding_hour survives in both cases: it is the grid-aligned stamp candles
    join to, on bar_end == funding_hour.
    """
    keep_flags = _resolve_keep_flags(keep_flags, clean)

    df = pl.read_parquet(data_dir / "hl_funding.parquet")

    candles_path = data_dir / f"hl_candles_{DELISTING_CHECK_INTERVAL}.parquet"
    candles = None
    if candles_path.exists():
        candles = pl.read_parquet(candles_path)
    else:
        logger.warning(
            "%s not found, flagging delistings from funding alone", candles_path
        )

    df = flag_after_delisting(flag_invalid_funding(df), candles)
    invalid = (~df["is_valid"]).sum()
    delisted = df["after_delisting"].sum()

    if clean:
        df = df.filter(pl.col("is_valid") & ~pl.col("after_delisting"))

    logger.info(
        "funding: %d rows, %d invalid, %d post-delisting%s",
        df.height,
        invalid,
        delisted,
        " (both dropped)" if clean else "",
    )
    return df if keep_flags else df.drop(FUNDING_FLAGS)
