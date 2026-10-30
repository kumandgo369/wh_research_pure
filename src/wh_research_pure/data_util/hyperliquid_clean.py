from datetime import UTC, datetime

import polars as pl


def flag_invalid_candles(df):
    """
    Add boolean flag columns for OHLCV sanity checks, plus
    an is_valid column that is True only if every check passes.
    """
    checks = {
        "bad_high_low": pl.col("high") < pl.col("low"),
        "bad_high": pl.col("high") < pl.max_horizontal("open", "close"),
        "bad_low": pl.col("low") > pl.min_horizontal("open", "close"),
        "bad_volume": pl.col("volume") < 0,
        "has_null": pl.any_horizontal(
            pl.col("open", "high", "low", "close", "volume").is_null()
            | pl.col("open", "high", "low", "close", "volume").is_nan()
        ),
    }
    return df.with_columns(
        **{name: expr.fill_null(True) for name, expr in checks.items()}
    ).with_columns(
        is_valid=~pl.any_horizontal(list(checks)),
    )


def trim_leading_zero_volume(df):
    """
    Per symbol, drop candles before the first candle with volume > 0.
    Zero-volume candles after trading has started are kept.
    """
    return df.sort("symbol", "bar_end").filter(
        (pl.col("volume") > 0).cast(pl.Int32).cum_sum().over("symbol") > 0
    )


FUNDING_CAP = 0.04  # Hyperliquid's hourly funding cap (verify against current docs)

# Settlements normally land a few ms into their hour, but the exchange sometimes
# records one late, and observed offsets reach 24 minutes. Those are real
# settlements, so the tolerance only has to catch rows so late that they can no
# longer be assigned to an hour with confidence.
OFF_HOUR_TOLERANCE_MINUTES = 30


def flag_invalid_funding(df):
    """
    Assign each funding entry to its hour and add flag columns.
    is_valid is True only if every check passes.

    funding_hour truncates rather than rounds. Every settlement in the data is
    stamped at or after the hour it belongs to - never before - so truncating
    assigns it to that hour and funding_offset measures how late it was
    recorded. Rounding would instead push anything more than half an hour late
    into the *following* hour, silently mis-assigning it and colliding with that
    hour's own entry, while capping the measured offset at 30 minutes so the
    mis-assignment could not be seen.

    A row recorded more than an hour late is still indistinguishable from an
    on-time entry for the next hour; it shows up as a duplicate_hour or a gap
    rather than as a large funding_offset.
    """
    df = df.with_columns(
        pl.col("time").dt.truncate("1h").alias("funding_hour")
    ).with_columns(
        (pl.col("time") - pl.col("funding_hour")).alias("funding_offset"),
    )

    checks = {
        "has_null": pl.any_horizontal(
            pl.col("funding_rate", "premium").is_null()
            | pl.col("funding_rate", "premium").is_nan()
        ),
        "exceeds_cap": pl.col("funding_rate").abs() > FUNDING_CAP,
        # funding_offset is never negative, so this needs no abs()
        "off_hour": pl.col("funding_offset")
        > pl.duration(minutes=OFF_HOUR_TOLERANCE_MINUTES),
        "duplicate_hour": pl.struct("symbol", "funding_hour").is_duplicated(),
    }
    return df.with_columns(
        **{name: expr.fill_null(True) for name, expr in checks.items()}
    ).with_columns(
        is_valid=~pl.any_horizontal(list(checks)),
    )


# Shorter trailing runs of zero funding are a quiet market, not a delisting.
# The shortest real tail observed so far is ~1500h, so this leaves them all alone.
MIN_DELISTED_TAIL_HOURS = 24


def _last_trade_time(candles):
    """
    Per symbol, the bar_end of the last candle that actually traded - the last
    instant the symbol is known to have been trading. A funding row stamped
    later than this has no traded bar ending at or after it.
    """
    return (
        candles.filter(pl.col("volume") > 0)
        .group_by("symbol")
        .agg(pl.col("bar_end").max().alias("last_trade"))
    )


def flag_after_delisting(df, candles=None, min_tail_hours=MIN_DELISTED_TAIL_HOURS):
    """
    Flag each symbol's post-delisting placeholder rows. Once a perp is delisted
    the API keeps returning an entry every hour indefinitely, with funding_rate
    = 0 and premium either 0 or NaN, so these rows look fine to every check in
    flag_invalid_funding. They are real API output, not corrupt data, which is
    why this is a flag of its own rather than part of is_valid.

    A row is flagged when it falls in the run of zero funding_rate that reaches
    the end of its symbol's history, and that run is at least min_tail_hours
    long. A null rate ends the run, so an unknown rate is never treated as a
    delisting.

    Pass candles (any interval, 1d covers the most history) to cross-check: a
    row is then only flagged if its symbol has no traded candle at or after it,
    so a symbol that demonstrably kept trading is never flagged no matter what
    its funding prints. Symbols missing from candles fall back to the run alone.
    """
    df = df.sort("symbol", "time")

    # Counts non-zero rates at or after each row; 0 means every rate from here
    # to the symbol's last entry is zero.
    in_tail = (
        pl.col("funding_rate")
        .ne(0)
        .fill_null(True)
        .cast(pl.Int32)
        .cum_sum(reverse=True)
        .over("symbol")
        == 0
    )

    df = df.with_columns(_in_tail=in_tail).with_columns(
        _tail_hours=pl.col("_in_tail").sum().over("symbol")
    )
    flag = pl.col("_in_tail") & (pl.col("_tail_hours") >= min_tail_hours)
    helpers = ["_in_tail", "_tail_hours"]

    if candles is not None:
        df = df.join(_last_trade_time(candles), on="symbol", how="left")
        flag = flag & (
            pl.col("last_trade").is_null() | (pl.col("time") > pl.col("last_trade"))
        )
        helpers.append("last_trade")

    return df.with_columns(after_delisting=flag).drop(helpers)


def delisting_summary(df, candles=None, min_tail_hours=MIN_DELISTED_TAIL_HOURS):
    """
    One row per delisted symbol, for eyeballing what flag_after_delisting found
    against when the symbol last traded. lag_hours is the time between the last
    traded candle and the start of the tail; it should be small.
    """
    flagged = flag_after_delisting(df, candles, min_tail_hours).filter(
        pl.col("after_delisting")
    )
    out = flagged.group_by("symbol").agg(
        tail_hours=pl.len(),
        tail_start=pl.col("time").min(),
        tail_end=pl.col("time").max(),
    )

    if candles is not None:
        out = out.join(_last_trade_time(candles), on="symbol", how="left").with_columns(
            lag_hours=(pl.col("tail_start") - pl.col("last_trade")).dt.total_hours()
        )

    return out.sort("tail_start")


# Hyperliquid has not always settled funding hourly: it paid every 8 hours until
# 2023-06-08 00:00 UTC and switched to hourly at 01:00 the same day. The rates
# confirm the switch rather than just the spacing - BTC paid 0.0001 per 8h before
# it and 0.0000125 per hour after, exactly one eighth. Each entry is (start of
# the stretch, hours between settlements); None means the start of history, and
# entries run in ascending order.
FUNDING_SCHEDULE = (
    (None, 8),
    (datetime(2023, 6, 8, 1, tzinfo=UTC), 1),
)


def _expected_interval(schedule):
    """Hours between settlements in force at prev_hour, as an expression."""
    expr = pl.lit(schedule[0][1], dtype=pl.Int64)
    for start, hours in schedule[1:]:
        expr = (
            pl.when(
                pl.col("prev_hour") >= pl.lit(start, dtype=pl.Datetime("us", "UTC"))
            )
            .then(pl.lit(hours, dtype=pl.Int64))
            .otherwise(expr)
        )
    return expr


def funding_gaps(df, schedule=FUNDING_SCHEDULE):
    """
    List every place a symbol's funding history skips a scheduled settlement.

    A fixed one-hour step would report the entire 8-hourly era as missing, so
    the expected step comes from schedule instead: expected_hours is the
    interval that applied at prev_hour. Pass a different schedule for another
    venue, or if Hyperliquid changes again.

    missing_settlements counts settlements, not hours - during the 8-hourly era
    one skipped settlement leaves an 8-hour hole. gap_hours keeps the raw
    duration. A step longer than its interval but shorter than two of them
    skips nothing and is not reported, which is what the single 9-hour step
    across the hourly switch is.

    Run this on rows that are both valid and still listed: a gap inside a
    delisting tail is meaningless, and rows dropped as invalid look like gaps.
    """
    return (
        df.sort("symbol", "funding_hour")
        .with_columns(
            prev_hour=pl.col("funding_hour").shift().over("symbol"),
        )
        .with_columns(
            gap_hours=(pl.col("funding_hour") - pl.col("prev_hour")).dt.total_hours(),
            expected_hours=_expected_interval(schedule),
        )
        .with_columns(
            missing_settlements=pl.col("gap_hours") // pl.col("expected_hours") - 1,
        )
        .filter(pl.col("missing_settlements") > 0)
        .select(
            "symbol",
            "prev_hour",
            "funding_hour",
            "gap_hours",
            "expected_hours",
            "missing_settlements",
        )
    )
