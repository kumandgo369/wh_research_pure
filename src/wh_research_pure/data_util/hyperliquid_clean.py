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
    return df.sort("symbol", "open_time").filter(
        (pl.col("volume") > 0).cast(pl.Int32).cum_sum().over("symbol") > 0
    )


FUNDING_CAP = 0.04  # Hyperliquid's hourly funding cap (verify against current docs)


def flag_invalid_funding(df):
    """
    Round funding times to the hour and add flag columns.
    is_valid is True only if every check passes.
    """
    df = df.with_columns(pl.col("time").dt.round("1h").alias("funding_hour"))

    checks = {
        "has_null": pl.any_horizontal(
            pl.col("funding_rate", "premium").is_null()
            | pl.col("funding_rate", "premium").is_nan()
        ),
        "exceeds_cap": pl.col("funding_rate").abs() > FUNDING_CAP,
        "off_hour": (pl.col("time") - pl.col("funding_hour")).abs()
        > pl.duration(minutes=1),
        "duplicate_hour": pl.struct("symbol", "funding_hour").is_duplicated(),
    }
    return df.with_columns(
        **{name: expr.fill_null(True) for name, expr in checks.items()}
    ).with_columns(
        is_valid=~pl.any_horizontal(list(checks)),
    )


def funding_gaps(df):
    """List every place a symbol's funding history skips one or more hours."""
    return (
        df.sort("symbol", "funding_hour")
        .with_columns(
            prev_hour=pl.col("funding_hour").shift().over("symbol"),
        )
        .filter(pl.col("funding_hour") - pl.col("prev_hour") > pl.duration(hours=1))
        .with_columns(
            missing_hours=(
                (pl.col("funding_hour") - pl.col("prev_hour")).dt.total_hours() - 1
            ),
        )
        .select("symbol", "prev_hour", "funding_hour", "missing_hours")
    )
