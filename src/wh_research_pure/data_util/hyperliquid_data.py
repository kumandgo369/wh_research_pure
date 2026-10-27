import logging
import time
from datetime import UTC, datetime, timedelta

import polars as pl
import requests
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL
from hyperliquid.utils.error import ClientError, ServerError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
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

# Hyperliquid returns at most 500 funding entries per request
FUNDING_PAGE_SIZE = 500

# Earlier than Hyperliquid's launch, so a full download covers all history
FUNDING_START = datetime(2022, 1, 1, tzinfo=UTC)

# Seconds between funding pages; full pages are heavy on the rate limit
FUNDING_PAGE_DELAY = 2.5

FUNDING_SCHEMA = {
    "time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "symbol": pl.String,
    "funding_rate": pl.Float64,
    "premium": pl.Float64,
}


def _is_retryable(exc):
    """Retry only transient errors: network issues, 5xx, and rate limits (429)."""
    if isinstance(
        exc,
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError, ServerError),
    ):
        return True
    return isinstance(exc, ClientError) and exc.status_code == 429


def _log_retry(retry_state):
    """Short one-line retry message instead of the full exception with headers."""
    exc = retry_state.outcome.exception()
    reason = (
        f"HTTP {exc.status_code}"
        if isinstance(exc, (ClientError, ServerError))
        else type(exc).__name__
    )
    target = retry_state.kwargs.get("symbol") or (
        retry_state.args[0] if retry_state.args else ""
    )
    logger.warning(
        "%s %s: %s, retry %d in %.1fs",
        retry_state.fn.__name__,
        target,
        reason,
        retry_state.attempt_number,
        retry_state.next_action.sleep,
    )


api_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=5, max=60),
    stop=stop_after_attempt(8),
    before_sleep=_log_retry,
    reraise=True,
)


@api_retry
def _fetch_meta():
    return info.meta()


@api_retry
def _fetch_candles(symbol, interval, start, end):
    return info.candles_snapshot(
        name=symbol, interval=interval, startTime=start, endTime=end
    )


@api_retry
def _fetch_funding(symbol, start, end):
    return info.funding_history(name=symbol, startTime=start, endTime=end)


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


def _download_symbol_funding(symbol, start_ms):
    """
    Download all funding entries for a single symbol
    from start_ms to now, paginating 500 entries at a time
    """
    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    rows = []
    cursor = start_ms

    while cursor < now_ms:
        batch = _fetch_funding(symbol, cursor, now_ms)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < FUNDING_PAGE_SIZE:
            break  # last page
        cursor = batch[-1]["time"] + 1
        time.sleep(FUNDING_PAGE_DELAY)

    if not rows:
        return pl.DataFrame(schema=FUNDING_SCHEMA)

    return (
        pl.DataFrame(rows)
        .select(
            pl.from_epoch(pl.col("time"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .cast(FUNDING_SCHEMA["time"]),
            pl.col("coin").alias("symbol"),
            pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
            pl.col("premium").cast(pl.Float64),
        )
        .sort("time")
    )


def download_all_symbols_funding(last_times=None):
    """
    Downloads funding history for all symbols in the
    hyperliquid perp universe. last_times maps symbol -> last
    stored datetime; those symbols only fetch newer entries.
    """
    last_times = last_times or {}
    symbols = _get_universe_symbols_list()
    logger.info("Downloading funding for %d symbols", len(symbols))

    frames, failed = [], []

    for i, symbol in enumerate(symbols, start=1):
        last = last_times.get(symbol)
        start = last if last is not None else FUNDING_START
        start_ms = int(start.timestamp() * 1000) + (1 if last is not None else 0)

        try:
            df_1 = _download_symbol_funding(symbol, start_ms)
        except Exception:
            logger.exception("[%d/%d] %s: failed, skipping", i, len(symbols), symbol)
            failed.append(symbol)
            continue

        if not df_1.is_empty():
            frames.append(df_1)
            logger.debug("[%d/%d] %s: %d entries", i, len(symbols), symbol, df_1.height)

        time.sleep(0.5)

    logger.info("Done: %d symbols with new data, %d failed", len(frames), len(failed))
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))

    if not frames:
        return pl.DataFrame(schema=FUNDING_SCHEMA)

    return pl.concat(frames).sort("symbol", "time")
