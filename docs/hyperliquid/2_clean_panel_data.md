# Cleaning panel data

The files in `data/` are never modified. Cleaning happens at load time, and
`load.py` is the only place that knows how a clean frame is built.

```Python
from wh_research_pure.data_util.load import load_candles, load_funding

candles = load_candles("1h")  # or "1d"
funding = load_funding()
```

| | columns returned | sorted by |
|---|---|---|
| `load_candles` | `open_time, close_time, symbol, interval, open, close, high, low, volume, trades` | `symbol, open_time` |
| `load_funding` | `time, symbol, funding_rate, premium, funding_hour` | `symbol, time` |

Arguments, both loaders:

| arg | default | effect |
|---|---|---|
| `clean` | `True` | drop the bad rows. `False` keeps everything |
| `keep_flags` | `not clean` | keep the flag columns. Set explicitly to override |
| `data_dir` | `data/` | read from somewhere else |

`load_candles` also takes `interval` (default `"1h"`).

---

## What `load_candles(interval)` does

1. `pl.read_parquet(data_dir / f"hl_candles_{interval}.parquet")`
2. `trim_leading_zero_volume` — sorts by `symbol, open_time`, then per symbol
   drops every row before the first with `volume > 0`. **This is also what sets
   the output sort order.**
3. `flag_invalid_candles` — adds six boolean columns
4. if `clean`: `.filter(pl.col("is_valid"))`
5. if not `keep_flags`: drop the six flag columns

Flags added in step 3:

| column | true when |
|---|---|
| `bad_high_low` | `high < low` |
| `bad_high` | `high < max(open, close)` |
| `bad_low` | `low > min(open, close)` |
| `bad_volume` | `volume < 0` |
| `has_null` | any OHLCV value is null or NaN |
| `is_valid` | none of the above |

Each check is `fill_null(True)`, so a null in a comparison fails the check
rather than silently passing.

Current effect:

| file | raw rows | trimmed | invalid | clean rows |
|---|---|---|---|---|
| `hl_candles_1d` | 233,944 | 55,060 | 0 | 178,884 |
| `hl_candles_1h` | 904,722 | 0 | 0 | 904,722 |

The 1d trim is pre-launch oracle prices. Nothing is invalid in either file
today, so `clean=True` only matters for the trim.

---

## What `load_funding()` does

1. `pl.read_parquet(data_dir / "hl_funding.parquet")`
2. read `hl_candles_1d.parquet` if it exists; if not, log a warning and carry on
   with `candles=None`
3. `flag_invalid_funding` — adds `funding_hour`, `funding_offset` and five flags
4. `flag_after_delisting(..., candles)` — sorts by `symbol, time`, adds
   `after_delisting`. **This is also what sets the output sort order.**
5. if `clean`: `.filter(pl.col("is_valid") & ~pl.col("after_delisting"))`
6. if not `keep_flags`: drop `funding_offset` and the six flags

`funding_hour` is **not** dropped — it is the key candles join on.

Columns added in step 3:

| column | meaning |
|---|---|
| `funding_hour` | `time` truncated to the hour |
| `funding_offset` | `time - funding_hour`, i.e. how late the settlement was recorded |
| `has_null` | `funding_rate` or `premium` is null or NaN |
| `exceeds_cap` | `abs(funding_rate) > 0.04` (`FUNDING_CAP`) |
| `off_hour` | `funding_offset > 30 min` (`OFF_HOUR_TOLERANCE_MINUTES`) |
| `duplicate_hour` | `(symbol, funding_hour)` appears more than once |
| `is_valid` | none of the five above |

Step 4 flags a row when **all** of these hold:

- it is inside the run of `funding_rate == 0` that reaches the end of its
  symbol's history (a null rate ends the run),
- that run is at least 24 hours long (`MIN_DELISTED_TAIL_HOURS`),
- and the symbol has no traded candle at or after it — or is missing from
  `candles` entirely.

Current effect:

| | rows |
|---|---|
| raw | 4,881,846 |
| `~is_valid` | 4,427 |
| `after_delisting` | 623,566 (12.8%, 56 symbols) |
| clean | 4,258,273 |

The two filters barely overlap: only 7 invalid rows sit outside a delisting
tail.

---

## Doing it without `load.py`

Both paths below return a frame **identical** to the loader's — verified with
`.equals()`.

### Option A — the cleaning helpers, by hand

```Python
import polars as pl
from pathlib import Path
from wh_research_pure.data_util.hyperliquid_clean import (
    flag_after_delisting,
    flag_invalid_candles,
    flag_invalid_funding,
    trim_leading_zero_volume,
)

DATA = Path("data")
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

# candles == load_candles("1d")
candles = pl.read_parquet(DATA / "hl_candles_1d.parquet")
candles = trim_leading_zero_volume(candles)  # order matters, see below
candles = flag_invalid_candles(candles)
candles = candles.filter(pl.col("is_valid")).drop(CANDLE_FLAGS)

# funding == load_funding()
funding = pl.read_parquet(DATA / "hl_funding.parquet")
daily = pl.read_parquet(DATA / "hl_candles_1d.parquet")  # 1d, not 1h
funding = flag_after_delisting(flag_invalid_funding(funding), daily)
funding = funding.filter(pl.col("is_valid") & ~pl.col("after_delisting")).drop(
    FUNDING_FLAGS
)
```

### Option B — pure polars, no helpers

For a notebook, another language, or when you want the expressions in front of
you.

```Python
import polars as pl
from pathlib import Path

DATA = Path("data")
FUNDING_CAP, OFF_HOUR_MIN, MIN_TAIL_H = 0.04, 30, 24

# candles == load_candles("1d")
candles = (
    pl.read_parquet(DATA / "hl_candles_1d.parquet")
    .sort("symbol", "open_time")
    .filter((pl.col("volume") > 0).cast(pl.Int32).cum_sum().over("symbol") > 0)
    .filter(
        ~(
            (pl.col("high") < pl.col("low")).fill_null(True)
            | (pl.col("high") < pl.max_horizontal("open", "close")).fill_null(True)
            | (pl.col("low") > pl.min_horizontal("open", "close")).fill_null(True)
            | (pl.col("volume") < 0).fill_null(True)
            | pl.any_horizontal(
                pl.col("open", "high", "low", "close", "volume").is_null()
                | pl.col("open", "high", "low", "close", "volume").is_nan()
            ).fill_null(True)
        )
    )
)

# funding == load_funding()
last_trade = (
    pl.read_parquet(DATA / "hl_candles_1d.parquet")
    .filter(pl.col("volume") > 0)
    .group_by("symbol")
    .agg(last_trade=pl.col("open_time").max())
)

funding = (
    pl.read_parquet(DATA / "hl_funding.parquet")
    .with_columns(funding_hour=pl.col("time").dt.truncate("1h"))
    .with_columns(funding_offset=pl.col("time") - pl.col("funding_hour"))
    .with_columns(
        is_valid=~(
            pl.any_horizontal(
                pl.col("funding_rate", "premium").is_null()
                | pl.col("funding_rate", "premium").is_nan()
            ).fill_null(True)
            | (pl.col("funding_rate").abs() > FUNDING_CAP).fill_null(True)
            | (pl.col("funding_offset") > pl.duration(minutes=OFF_HOUR_MIN)).fill_null(
                True
            )
            | pl.struct("symbol", "funding_hour").is_duplicated().fill_null(True)
        )
    )
    .sort("symbol", "time")
    .with_columns(
        _in_tail=pl.col("funding_rate")
        .ne(0)
        .fill_null(True)
        .cast(pl.Int32)
        .cum_sum(reverse=True)
        .over("symbol")
        == 0
    )
    .with_columns(_tail_hours=pl.col("_in_tail").sum().over("symbol"))
    .join(last_trade, on="symbol", how="left")
    .with_columns(
        after_delisting=pl.col("_in_tail")
        & (pl.col("_tail_hours") >= MIN_TAIL_H)
        & (pl.col("last_trade").is_null() | (pl.col("time") > pl.col("last_trade")))
    )
    .filter(pl.col("is_valid") & ~pl.col("after_delisting"))
    .select("time", "symbol", "funding_rate", "premium", "funding_hour")
)
```

### Three things that break if you roll your own

- **Order.** `trim_leading_zero_volume` must run *before* `flag_invalid_candles`,
  or the daily file's 55,060 pre-launch oracle rows are flagged invalid instead
  of removed.
- **The delisting cross-check needs the *daily* candles.** Pass the 1h file and
  it silently falls back to the weaker signal, because every delisted symbol is
  missing from it. `load_funding` reads the 1d file itself, and warns if it is
  absent.
- **Both filters, not one.** Dropping only `~is_valid` leaves 623,566 artificial
  zeros in the frame.

---

## Inspecting rather than analysing

```Python
funding = load_funding(clean=False)  # keeps flags automatically
funding.filter(~pl.col("is_valid") | pl.col("after_delisting"))
```

```Python
from wh_research_pure.data_util.hyperliquid_clean import delisting_summary, funding_gaps

# One row per delisted symbol. lag_hours = last traded candle -> start of the
# zero-funding tail. Currently 3-82h, median 11h, which is what 1d candle
# granularity implies. A large value means that symbol needs a look.
delisting_summary(raw_funding, daily_candles)

# Missing settlements. Run on the CLEAN frame: a gap inside a delisting tail is
# meaningless, and rows dropped as invalid look like gaps.
funding_gaps(funding)
```

`funding_gaps` reads the expected interval per row from `FUNDING_SCHEDULE`, so
the 8-hourly era is not reported as missing, and counts `missing_settlements`
rather than missing hours. Pass `schedule=` for another venue.

---

## What the data actually looks like

**Candles have no gaps.** Every step in both files is exactly one interval — the
API emits a bar even when nothing trades (1,701 such zero-volume bars in the 1h
file, 74 in the 1d). There are no *trailing* zero-volume candles, because the
candle API stops at delisting; `trim_leading_zero_volume` has no counterpart.

**`funding_hour` truncates, it does not round.** Every settlement is stamped at
or after the hour it belongs to (minimum offset exactly 0, no negatives), so
truncating is always correct and `funding_offset` is always non-negative. The
exchange records a settlement late now and then — 571 rows, the latest 23.9
minutes past the hour, in bursts hitting every listed symbol at once. These are
real and are kept; the 30-minute tolerance currently fires on nothing.

A row recorded more than an hour late is undetectable from its timestamp alone.
It surfaces as a `duplicate_hour` or a gap, not as a large offset.

**Funding was 8-hourly before 2023-06-08 01:00 UTC**, at 8× the magnitude — BTC
paid `0.0001` per 8h before and `0.0000125` per hour after. `FUNDING_SCHEDULE`
records the switch. A step longer than one interval but shorter than two skips
nothing and is not reported, which is what the 9-hour step across the switch is.

**Missing settlements: 208 rows, 211 settlements, all in the hourly era.**

| hour missed | symbols | settlements | what |
|---|---|---|---|
| 2024-08-15 13:00 | 132 of 132 listed | 1 | exchange outage |
| 2023-08-23 20:00 | 44 of 44 | 1 | exchange outage |
| 2023-07-02 20:00 | 28 of 28 | 1 | exchange outage |
| 2024-06-20 14:00–17:00 | ZRO | 4 | dropped as invalid |
| 2024-06-20 16:00 | NFTI, OX, RLB | 1 each | dropped as invalid |

Three genuine one-hour outages; 7 of the 211 are rows our own NaN-premium check
removes. Gaps are exchange-wide, never per-symbol.

**Why the delisting flag is the zero-rate run, not the missing premium.**
`premium` is `NaN` rather than null for part of the history (Data Wrangler shows
both as "missing"), but NaN premium stops after 2024-07-01 12:00, so a
missing-premium test catches only 4,420 of the 623,566 rows. The zero-rate run
has no such cutoff and is precise: of 623,584 rows with `funding_rate == 0` it
flags 623,566 and leaves 18 alone — genuine zero-funding hours on live symbols
that still carry a real premium. The shortest real tail is 1,512 hours, so the
24-hour threshold is nowhere near binding.

---

## Joining funding to candles

A funding row stamped `T` describes the hour **before** `T`; a candle row stamped
`T` describes the hour **after** it. So:

```Python
panel = candles.join(
    funding.select("symbol", "funding_hour", "funding_rate", "premium"),
    left_on=["symbol", "open_time"],
    right_on=["symbol", "funding_hour"],
    how="left",
)
```

`funding_hour` is unique per `(symbol, hour)`, so this cannot fan out. Joining on
`close_time` matches **zero rows** — `close_time` is `:59:59.999`, never on an
hour boundary.

Full treatment, including the asof variant and the look-ahead traps:
[3_funding_lookahead_bias.md](3_funding_lookahead_bias.md).
