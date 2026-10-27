# Cleaning panel data guide

The files in `data/` are never modified. Cleaning happens when you load them,
and `load.py` is the only place that knows how a clean frame is built.

## Just give me the data

```Python
from wh_research_pure.data_util.load import load_candles, load_funding

candles = load_candles("1h")  # trimmed, checked, invalid rows dropped
funding = load_funding()  # checked, delisting placeholders dropped
```

Both return a tidy frame with the flag columns removed. The funding frame keeps
`time` (the raw settlement stamp) and `funding_hour` (that stamp truncated to the
hour), because `funding_hour` is what candles join on.

Pass `clean=False` to get everything back with the flags intact, which is what
you want when inspecting rather than analysing:

```Python
funding = load_funding(clean=False)
funding.filter(~pl.col("is_valid") | pl.col("after_delisting"))
```

`keep_flags` defaults to the opposite of `clean`, so a clean frame comes back
tidy and an unfiltered one keeps the columns that explain it. Set it explicitly
to override. `data_dir` points somewhere other than `data/`.

Three things the loaders get right that are easy to get wrong by hand:

- **Order.** `trim_leading_zero_volume` must run *before* `flag_invalid_candles`,
  or the daily file's 55,060 pre-launch oracle rows get flagged as invalid
  instead of removed.
- **The delisting cross-check needs the *daily* candles.** `load_funding` reads
  them itself. Hand `flag_after_delisting` the 1h file instead and it silently
  falls back to the weaker signal, because every delisted symbol is missing from
  that file. If `hl_candles_1d.parquet` is absent the loader falls back too, but
  it logs a warning rather than doing it quietly.
- **Both filters, not one.** Dropping only `~is_valid` leaves 622,166 artificial
  zeros in the frame.

---

## Timestamps and alignment

Read this before joining funding to candles or building features.

**A funding entry describes the hour *before* its timestamp.** The entry stamped
`T` is the settlement paid at `T` for holding through `[T-1h, T)`. It is a
realised, backward-looking quantity, known as of `T`.

That is not in Hyperliquid's docs, so it is established from the data: for every
perp listed since the 1h file begins, the first funding entry arrives exactly one
hour after the symbol's first traded 1h candle, and the last entry lands at or
one hour after the last traded candle.

**A candle describes the hour *after* its `open_time`.** The bar with
`open_time = T` covers `[T, T+1h)`.

So `funding_hour == open_time` is the correct join, and it is safe:

```Python
panel = candles.join(
    funding.select("symbol", "funding_hour", "funding_rate", "premium"),
    left_on=["symbol", "open_time"],
    right_on=["symbol", "funding_hour"],
    how="left",
)
```

The funding value on a row is known at that row's `open_time`; the row's own
return is the hour that follows it. Features from `funding_rate` on that row,
labelled with that row's return, carry no look-ahead. `funding_hour` is unique
per `(symbol, hour)` — `duplicate_hour` fires on nothing — so the join cannot
fan out.

Four ways to reintroduce look-ahead anyway:

- **Joining funding at `T` to the bar opening at `T-1h`.** That bar *is* the hour
  the funding was measured over, so a model given both is reading the settlement
  of a return it is trying to predict.
- **Executing at the same `open_time` you read funding from.** 571 settlements
  are recorded late, up to 23.9 minutes past the hour, so at `T` exactly those
  values do not exist yet. Lag the funding feature one bar, or trade the next
  open, if the strategy is sensitive to it.
- **Forward-filling across the 8-hourly era** (before 2023-06-08, below). One 8h
  rate spread over 8 hours is both a fabricated observation and 8× too large.
  Start the sample at 2023-06-08, or divide pre-switch rates by 8.
- **Building the panel on 1h candles.** `hl_funding.parquet` holds 235 symbols and
  `hl_candles_1d.parquet` 234, but the 1h file holds only the 196 listed in the
  last ~208 days. Delisted perps are absent from it entirely, so a 1h panel is
  survivorship-biased by construction. Use daily candles for anything spanning
  history, and treat the 1h file as a recent window.

Everything below is the manual path, for when you are checking the cleaning
itself rather than using it.

---

## Cleaning OHLCV data

```Python
from wh_research_pure.data_util.hyperliquid_clean import (
    flag_invalid_candles,
    trim_leading_zero_volume,
)

df = pl.read_parquet(DATA_PATH)
df = trim_leading_zero_volume(df)  # must come first, see above
df = flag_invalid_candles(df)
print(df.filter(~pl.col("is_valid")))  # currently empty in both files
```

Candles have **no gaps**: every step in both files is exactly one interval,
because the API emits a bar even when nothing trades (1,697 such zero-volume
bars in the 1h file, 74 in the 1d). There are no *trailing* zero-volume candles
either — the candle API stops at delisting, unlike funding — so
`trim_leading_zero_volume` has no counterpart.

---

## Cleaning funding data

Funding has two separate problems, and they need two separate flags.

`is_valid` is about **integrity**: the row is damaged or contradictory.
`after_delisting` is about **relevance**: the row is fine as API output, but it
describes a perp that no longer trades. Once a perp is delisted Hyperliquid keeps
returning an entry every hour forever, with `funding_rate = 0` and `premium`
either `0` or `NaN`. Those rows pass every integrity check, so nothing but
`after_delisting` will catch them — and they are **12.8% of the file**
(622,166 rows across 56 symbols), every one of them a zero that would drag down
any cross-sectional funding average.

```Python
from wh_research_pure.data_util.hyperliquid_clean import (
    flag_invalid_funding,
    flag_after_delisting,
    delisting_summary,
    funding_gaps,
)

# Use 1d candles for the cross-check: they span the full history, while
# hl_candles_1h.parquet only holds the last ~208 days (MAX_CANDLES), so delisted
# symbols are missing from it entirely and it confirms nothing.
df = pl.read_parquet(FUNDING_PATH)
candles = pl.read_parquet(CANDLES_1D_PATH)

df = flag_invalid_funding(df)  # integrity -> is_valid
df = flag_after_delisting(df, candles)  # relevance -> after_delisting

# Eyeball the cross-check before trusting it. lag_hours is the time from the
# symbol's last traded candle to the start of its tail; it should be small
# (currently 3-82h, median 11h, which is what 1d candle granularity implies).
# A large value means the tail started well before trading stopped.
print(delisting_summary(df, candles))

# Integrity failures that are NOT just delisting placeholders
print(df.filter(~pl.col("is_valid") & ~pl.col("after_delisting")))

# Keep only rows that are both sound and still relevant
clean = df.filter(pl.col("is_valid") & ~pl.col("after_delisting"))

# Gaps. Run on `clean`, never on the raw frame: a gap inside a delisting tail is
# meaningless, and rows dropped as invalid look like gaps. funding_gaps reads
# the settlement schedule, so the 8-hourly era is not reported as missing.
print(funding_gaps(clean))
```

### Hour assignment

`funding_hour` **truncates** `time`, and `funding_offset` is how late the
settlement was recorded. Every entry is stamped at or after the hour it belongs
to (minimum offset exactly 0, no negative offsets anywhere), so truncating is
always the correct assignment and the offset is always non-negative. Rounding
would push a late settlement into the *following* hour and collide with that
hour's own entry.

The exchange records a settlement late now and then — 571 rows, the latest 23.9
minutes past the hour, in bursts that hit every listed symbol at once, so they
look like recovery after an outage. These are real settlements and are kept.
`OFF_HOUR_TOLERANCE_MINUTES` (30) only catches rows so late that their hour
becomes genuinely ambiguous; it currently fires on nothing.

A row recorded more than an hour late is undetectable from its timestamp alone —
it is indistinguishable from an on-time entry for the next hour, and surfaces as
a `duplicate_hour` or a gap instead of a large offset.

### Funding was 8-hourly before 2023-06-08

Hyperliquid settled funding **every 8 hours until 2023-06-08 00:00 UTC** and
switched to hourly at 01:00 the same day. The rates confirm the switch rather
than just the spacing: BTC pays 0.0001 per 8h before it and 0.0000125 per hour
after, exactly one eighth.

`FUNDING_SCHEDULE` records this, and `funding_gaps` reads the expected interval
from it per row, based on `prev_hour`. The output carries `expected_hours` so you
can see which era each gap sits in, and counts `missing_settlements` rather than
missing hours — during the 8-hourly era one skipped settlement leaves an 8-hour
hole, so hours are the wrong unit. Pass a different `schedule` for another venue,
or if Hyperliquid changes again.

A step longer than its interval but shorter than two of them skips nothing and
is not reported. That is what the 9-hour step across the switch is, for symbols
whose last 8-hourly settlement was at 16:00 on 2023-06-07.

### The gap picture

With the schedule applied, the gap table is **208 rows / 211 missing
settlements**, all in the hourly era, in five events:

| hour missed | symbols | settlements | what |
|---|---|---|---|
| 2024-08-15 13:00 | 132 of 132 listed | 1 | exchange outage |
| 2023-08-23 20:00 | 44 of 44 | 1 | exchange outage |
| 2023-07-02 20:00 | 28 of 28 | 1 | exchange outage |
| 2024-06-20 14:00–17:00 | ZRO | 4 | we dropped these as invalid |
| 2024-06-20 16:00 | NFTI, OX, RLB | 1 each | we dropped these as invalid |

So the genuine record for the entire hourly history is **three one-hour outages**,
and 7 of the 211 cells are self-inflicted — rows the NaN-premium check removes.

Gaps are **exchange-wide, never per-symbol**: when funding is missing it is
missing for every listed symbol at once. There is no pagination bug in
`_download_symbol_funding` to chase.

### Why the flag is the zero-rate run and not the missing premium

`premium` is `NaN` rather than null for part of the history, which is easy to
misread (VS Code Data Wrangler reports both as "missing"). But `NaN` premium
stops after `2024-07-01 12:00`, so a missing-premium test only catches tails that
were already running before that date — 4,420 rows of the 622,166. The zero-rate
run has no such cutoff.

The run is also precise, not just complete: of the 622,184 rows in the file with
`funding_rate == 0`, it flags 622,166 and correctly leaves 18 alone. Those 18 are
genuine zero-funding hours mid-history on live symbols, which still carry a real
premium.

`MIN_DELISTED_TAIL_HOURS` (24) keeps a brief quiet patch in a live market from
being read as a delisting. The shortest real tail so far is ~1,500 hours, so the
threshold is nowhere near binding today.
