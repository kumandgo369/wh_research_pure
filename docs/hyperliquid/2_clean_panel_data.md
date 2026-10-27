# Cleaning panel data guide

The files in `data/` are never modified. Cleaning happens when you load them,
and `load.py` is the only place that knows how a clean frame is built.

## Just give me the data

```Python
from wh_research_pure.data_util.load import load_candles, load_funding

candles = load_candles("1h")  # trimmed, checked, invalid rows dropped
funding = load_funding()  # checked, delisting placeholders dropped
```

Both return a tidy frame with the flag columns removed. `funding_hour` stays on
the funding frame, because that is what candles join on.

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

Everything below is the manual path, for when you are checking the cleaning
itself rather than using it.

---

## Cleaning OHLCV data

```Python
# Step 1, import the cleaning functions
from wh_research_pure.data_util.hyperliquid_clean import (
    flag_invalid_candles,
    trim_leading_zero_volume,
)

# Step 2, read the data file
df = pl.read_parquet(DATA_PATH)

# Step 3, remove rows where volume = 0 until the first row where volume != 0
df = trim_leading_zero_volume(df)

# Step 4, perform data check
df = flag_invalid_candles(df)

# Step 5, check for invalid rows
print(df.filter(pl.col("is_valid") == False))

# Step 6, decide what you want to do with invalid rows
```

Note that candles have **no gaps**: every step in both files is exactly one
interval, because the API emits a bar even when nothing trades (1,697 such bars
in the 1h file, 74 in the 1d). There are also no *trailing* zero-volume candles
— the candle API stops at delisting, unlike funding — so `trim_leading_zero_volume`
has no counterpart.

---

## Cleaning Funding data

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
# Step 1, import the cleaning functions
from wh_research_pure.data_util.hyperliquid_clean import (
    flag_invalid_funding,
    flag_after_delisting,
    delisting_summary,
    funding_gaps,
)

# Step 2, read the funding data and the daily candles.
# Use 1d candles for the cross-check: they span the full history, while
# hl_candles_1h.parquet only holds the last ~208 days (MAX_CANDLES), so delisted
# symbols are missing from it entirely and it confirms nothing.
df = pl.read_parquet(FUNDING_PATH)
candles = pl.read_parquet(CANDLES_1D_PATH)

# Step 3, integrity check -> is_valid
df = flag_invalid_funding(df)

# Step 4, relevance check -> after_delisting.
# The flag is the run of zero funding_rate reaching the end of a symbol's
# history. Passing candles cross-checks it: a row is only flagged if the symbol
# has no traded candle at or after it, so a symbol that kept trading is never
# flagged however its funding prints.
df = flag_after_delisting(df, candles)

# Step 5, eyeball the cross-check before trusting it.
# lag_hours is the time from the symbol's last traded candle to the start of its
# tail. It should be small (currently 3-82h, median 11h, which is what 1d candle
# granularity implies). A large value means the tail started well before trading
# stopped, so that symbol needs a look.
print(delisting_summary(df, candles))

# Step 6, check the integrity failures that are NOT just delisting placeholders
print(df.filter(~pl.col("is_valid") & ~pl.col("after_delisting")))

# Step 7, keep only rows that are both sound and still relevant
clean = df.filter(pl.col("is_valid") & ~pl.col("after_delisting"))

# Step 8, gaps. Run this on `clean`, never on the raw frame: a gap inside a
# delisting tail is meaningless, and rows dropped as invalid look like gaps.
# funding_gaps knows the settlement schedule, so the 8-hourly era is not
# reported as missing; see below.
print(funding_gaps(clean))
```

### Hour assignment and late settlements

`funding_hour` **truncates**, and `funding_offset` is how late the settlement was
recorded. Every entry in the data is stamped at or after the hour it belongs to
(minimum offset exactly 0, no negative offsets anywhere), so truncating is
always the correct assignment and the offset is always non-negative.

Rounding, which this used to do, was wrong in two ways at once: anything more
than 30 minutes late was pushed into the *following* hour, colliding with that
hour's own entry, and the measured distance was capped at 30 minutes so the
mis-assignment was invisible. It happened to be harmless on the current data
only because nothing is more than 24 minutes late.

The exchange records a settlement late now and then — 571 rows, at 1.8, 1.9,
4.2, 14.8 and 23.9 minutes, in bursts that hit every listed symbol at once, so
they look like recovery after an outage. These are real settlements and are
kept. `OFF_HOUR_TOLERANCE_MINUTES` (30) only catches rows so late that the hour
they belong to becomes genuinely ambiguous; it currently fires on nothing.

A row recorded more than an hour late is still undetectable from its timestamp
alone — it is indistinguishable from an on-time entry for the next hour, and
surfaces as a `duplicate_hour` or a gap instead of a large offset.

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

Measured per consecutive step across the whole history: **4,251,566** one-hour,
**1,789** eight-hour, **207** two-hour, and a single five-hour.

### The real gap picture

Fixing the interval drops the gap table from 1,997 rows / 12,734 missing hours to
**208 rows / 211 missing settlements** — and all 211 are in the hourly era, in
five events:

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
misread (VS Code Data Wrangler reports both as "missing"). But the API switched
from `NaN` premium to `premium = 0.0` at `2024-07-01 12:00`, so a missing-premium
test only catches tails that were already running before that date — 4,420 rows
of the 622,166. The zero-rate run has no such cutoff.

The run is also precise, not just complete: of the 622,184 rows in the file with
`funding_rate == 0`, it flags 622,166 and correctly leaves 18 alone. Those 18 are
genuine zero-funding hours mid-history on live symbols (AIXBT, BNB, TURBO, and
the CANTO and ZRO glitch windows), which still carry a real premium.

`MIN_DELISTED_TAIL_HOURS` (24) keeps a brief quiet patch in a live market from
being read as a delisting. The shortest real tail so far is ~1,500 hours, so the
threshold is nowhere near binding today.
