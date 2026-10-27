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
