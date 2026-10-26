import logging
import time
from datetime import UTC, datetime, timedelta

import polars as pl
import requests
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL
from hyperliquid.utils.error import ClientError, ServerError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

# Create info instance
info = Info(base_url=MAINNET_API_URL, skip_ws=True, timeout=10)

# Intervals for candle download (minutes)
INTERVALS = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}

# Schema of the candles df returned by the download functions
CANDLE_SCHEMA = {
    "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "close_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "symbol": pl.String,
    "interval": pl.String,
    "open": pl.Float64,
    "close": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "volume": pl.Float64,
    "trades": pl.Int64,
}

# Maximum candles API can retrieve
MAX_CANDLES = 5000

# Column map for renaming columns in candles polars df
COLUMN_MAP = {
    "t": "open_time",
    "T": "close_time",
    "s": "symbol",
    "i": "interval",
    "o": "open",
    "c": "close",
    "h": "high",
    "l": "low",
    "v": "volume",
    "n": "trades",
}


def _is_retryable(exc):
    """Retry only transient errors: network issues, 5xx, and rate limits (429)."""
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ServerError)):
        return True
    return isinstance(exc, ClientError) and exc.status_code == 429


api_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@api_retry
def _fetch_meta():
    return info.meta()


@api_retry
def _fetch_candles(symbol, interval, start, end):
    return info.candles_snapshot(name=symbol, interval=interval, startTime=start, endTime=end)


def _get_universe_symbols_list():
    """
    Returns a list of all symbol names (listed and delisted)
    in the hyperliquid perp universe
    """
    symbols = []
    universe = _fetch_meta()["universe"]
    for _ in universe:
        symbols.append(_["name"])
    return symbols


def _download_symbol_candles_start(symbol, interval, start_date):
    """
    Download candles for a single symbol
    from start_date to now
    """

    now = datetime.now(tz=UTC)

    if interval not in INTERVALS:
        raise ValueError(f"Chosen interval {interval} is not supported")

    candles = _fetch_candles(
        symbol=symbol,
        interval=interval,
        start=start_date,
        end=int(now.timestamp() * 1000),
    )

    if not candles:
        return pl.DataFrame()

    return (
        pl.DataFrame(candles)
        .rename(COLUMN_MAP)
        .with_columns(
            pl.col("open", "close", "high", "low", "volume").cast(pl.Float64),
            pl.from_epoch(
                pl.col("open_time", "close_time"), time_unit="ms"
            ).dt.replace_time_zone(time_zone="UTC"),
        )
        .filter(pl.col("close_time") <= now)
        .sort("open_time")
    )


def _download_symbol_candles_max(symbol, interval):
    """
    Downloads the maximum number of candles until
    current date for a single symbol
    """

    if interval not in INTERVALS:
        raise ValueError(f"Chosen interval {interval} is not supported")

    now = datetime.now(tz=UTC)
    start_date = now - timedelta(minutes=INTERVALS[interval] * MAX_CANDLES)

    return _download_symbol_candles_start(
        symbol=symbol, interval=interval, start_date=int(start_date.timestamp() * 1000)
    )


def download_all_symbols_candles_max(interval):
    """
    Downloads the maximum number of candles until
    current date for all symbols in the hyperliquid
    perp universe
    """

    if interval not in INTERVALS:
        raise ValueError(f"Chosen interval {interval} is not supported")

    symbols = _get_universe_symbols_list()
    logger.info("Downloading %s candles for %d symbols", interval, len(symbols))

    frames = []
    failed = []

    for i, symbol in enumerate(symbols, start=1):
        try:
            df_1 = _download_symbol_candles_max(symbol=symbol, interval=interval)
        except Exception:
            logger.exception("[%d/%d] %s: failed, skipping", i, len(symbols), symbol)
            failed.append(symbol)
            continue

        if not df_1.is_empty():
            frames.append(df_1)
            logger.debug("[%d/%d] %s: %d candles", i, len(symbols), symbol, df_1.height)

        time.sleep(0.5)  # gentle pacing to avoid rate limits

    logger.info("Done: %d symbols with data, %d failed", len(frames), len(failed))
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))

    if not frames:
        return pl.DataFrame(schema=CANDLE_SCHEMA)

    return pl.concat(frames).sort("symbol", "open_time")