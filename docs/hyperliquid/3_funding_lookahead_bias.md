# Funding timestamps and look-ahead bias

Working notes on how to timestamp Hyperliquid funding for supervised learning.
Every number and every code block here was run against the files in `data/`,
snapshot 2026-10-29. Row counts grow with every download, so proportions are
quoted where the proportion is the point; re-measure before citing an absolute.

**The rule:** every row in both files is stamped at the instant its contents
become known, and describes the hour that just ended. A funding row stamped `T`
settles the hour `[T-1h, T)`; a candle row stamped `T` *is* the bar `[T-1h, T)`.
So a panel row holds nothing but realised information, `bar_end == funding_hour`
lines the two series up, and the label is the single thing you build forward.
Everything else in this document is a consequence of that one sentence.

Jump to [§3 The recipe](#3-the-recipe-funding-as-an-ml-feature) if you just want
the pipeline.

---

## 1. One clock: a stamp is a knowledge time

Both series are hourly, both are UTC, both stamp a row on the hour — and both
stamps mean the same thing.

```
                      T-1h                  T                 T+1h
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶

   FUNDING row stamped T
     premium sampled    [═══════════════════)
     settled / knowable                     ●  time = T + 48 ms
     funding_hour                           │  = T
                                            │
   CANDLE row stamped T                     │
     bar span           [═══════════════════)
     bar_end                                ●  = T

                                            │
                        both rows describe [T-1h, T)
                        both are realised and known at T
```

| series | a row stamped `T` describes | direction | status at `T` |
|---|---|---|---|
| funding (`funding_hour`) | the hour `[T-1h, T)` that just ended | **backward** | realised, known |
| candles (`bar_end`) | the hour `[T-1h, T)` that just ended | **backward** | realised, known |

A funding entry is a *settlement*: at the top of hour `T` the exchange charges
holders for the hour they have just held through. A candle is a *bucket*, stamped
at the end of the bucket rather than the start, so that reading it at its own
stamp is never look-ahead.

Everything in a row is therefore usable at the row's own timestamp, and the label
is the only thing that reaches past it:

```
                      T-1h                  T                 T+1h
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶
                                            │
   row stamped T        [═══════════════════)  realised, usable
   label    y(T)                            [═══════════════════)  realised after T
                                            │
                                     decision point
```

### 1.1 Why candles are stamped at the end

A bar's open, high, low, close, volume and trade count are not all known until
the bar finishes. Stamping the row at its start would put a timestamp on data
that does not exist yet, and every candle-derived feature — momentum, realised
vol, a volume z-score — would need a `.shift(1)` to be legal. One forgotten shift
is a silent leak that looks like alpha, and §6.1 shows you cannot find it by
checking correlations.

Stamping at the end makes the safe reading the default one: **any row may be used
at its own stamp.** The failure mode flips from "lookahead" to "one bar stale",
which is the direction you want to be wrong in.

Two things it costs, both paid once and visibly:

- The label is no longer the row's own return. It has to be shifted forward
  ([Step 6](#step-6--the-label-the-one-thing-that-looks-forward)), and under the
  old start-stamped convention the correct label is now a leak — see
  [§6.6](#66-using-the-rows-own-return-as-the-label).
- A daily bar is stamped at the *following* midnight. The 2026-04-02 daily bar
  has `bar_end = 2026-04-03 00:00`, so reporting by calendar day needs
  `bar_end - 1d`.

---

## 2. Why we believe the funding direction

Hyperliquid's docs say the premium is "sampled every 5 seconds and averaged over
the hour" but never say *which* hour, and they do not define the `time` field of
`fundingHistory` at all. So it is established from the data, using perps whose
entire life sits inside the 1h candle window:

```Python
import polars as pl
from wh_research_pure.data_util.load import load_candles, load_funding

f = load_funding()
c = load_candles("1h")

first_bar = (
    c.filter(pl.col("volume") > 0)
    .group_by("symbol")
    .agg(first_bar=pl.col("bar_end").min())
)
first_fund = f.group_by("symbol").agg(first_fh=pl.col("funding_hour").min())

print(
    first_bar.join(first_fund, on="symbol")
    .filter(pl.col("first_bar") > c["bar_end"].min())  # genuinely new listings
    .with_columns(delta_h=(pl.col("first_fh") - pl.col("first_bar")).dt.total_hours())
    .select("symbol", "first_bar", "first_fh", "delta_h")
    .sort("first_bar")
)
```

For the six perps listed since the 1h file begins — CHIP, GRAM, CASHCAT, PONS,
USELESS, XTEN — `delta_h` is **exactly 0** in every case:

```
     first traded bar          first funding entry
        03:00 [════════════) 04:00 ●  04:00
                                   ↑
                        the first full hour of holding,
                        settled at its end — the same instant
                        the bar itself is stamped
```

A perp that starts trading at 03:00 has its first bar stamped 04:00 and gets its
first funding at 04:00, because 04:00 is when the first full hour of holding gets
settled. If funding at `T` summarised `[T, T+1h)` the first entry would be at
03:00. It is not. The tail agrees: a live symbol's last funding entry is stamped
exactly at its last `bar_end`.

That coincidence is the whole point of end-stamping. Under the old convention the
bar was stamped 03:00 and this test read `delta_h = 1`; the two series had to be
nudged into line by hand at every join.

(Symbols with large negative `delta_h` in that output are old listings whose
funding predates the rolling 1h window. They are the window's left edge, not
listings.)

**Do not try to re-derive this from correlations.** Funding against hourly
returns gives |ρ| ≈ 0.02 at every lag from −4 to +4 with no peak clearing the
noise. The listing-boundary test is deterministic; the correlation test has no
power. See [§6.1](#61-the-leak-is-invisible-to-a-correlation-check).

---

## 3. The recipe: funding as an ML feature

The whole pipeline, in order. Each step exists to close one specific hole.

```Python
import polars as pl
from wh_research_pure.data_util.load import load_candles, load_funding

HOURLY_FROM = pl.datetime(2023, 6, 8, 1, time_zone="UTC")  # end of the 8-hourly era
LATENCY = pl.duration(seconds=1)  # how fast you assume you can act
HORIZON = 1  # label length, in bars
```

### Step 1 — load through `load.py`, never `read_parquet`

```Python
funding = load_funding()  # drops 622,929 post-delisting zeros + integrity failures
candles = load_candles("1h")  # trims pre-launch oracle rows, drops invalid bars
```

Reading the parquet directly leaves 12.8% of the funding file as artificial
zeros from delisted perps, which drags down every cross-sectional statistic.

### Step 2 — restrict to the hourly era

```Python
funding = funding.filter(pl.col("funding_hour") >= HOURLY_FROM).sort("time")
candles = candles.sort("symbol", "bar_end")
```

Before 2023-06-08 funding settled every 8 hours at 8× the magnitude. That era is
1,814 rows across 25 symbols — **0.043% of the clean file**. Not worth modelling;
dropping it removes a whole class of scaling bug. See [§6.4](#64-forward-filling-across-the-8-hourly-era).

### Step 3 — make the decision timestamp explicit

```Python
candles = candles.with_columns(decision=pl.col("bar_end") + LATENCY)
```

This is the single most useful line in the pipeline. It turns "when do I know
things?" from an assumption buried in a join into a column you can assert on.
`bar_end` is when the bar is complete; `LATENCY` is your claim about how fast you
can react to the settlement stamped at that same instant. Everything downstream
is only as honest as that number.

### Step 4 — asof join, backward, with a tolerance

```Python
panel = (
    candles.sort("decision")
    .join_asof(
        funding,
        left_on="decision",
        right_on="time",
        by="symbol",
        strategy="backward",
        tolerance="1h",
    )
    .sort("symbol", "bar_end")
)
```

Join on raw `time`, not `funding_hour`. `strategy="backward"` is mandatory;
`"forward"` and `"nearest"` are look-ahead by definition. `tolerance` is
mandatory too — see [§5](#5-choosing-the-join) for what each argument buys and
what happens without it.

### Step 5 — features, every window ending at the current row

```Python
panel = (
    panel.with_columns(
        f_now=pl.col("funding_rate"),
        f_mean_24h=pl.col("funding_rate")
        .rolling_mean(24, min_samples=24)
        .over("symbol"),
        f_sum_168h=pl.col("funding_rate")
        .rolling_sum(168, min_samples=168)
        .over("symbol"),
        f_age_s=(pl.col("decision") - pl.col("time")).dt.total_seconds(),
    )
    .with_columns(
        f_z_720h=(
            (
                pl.col("funding_rate")
                - pl.col("funding_rate").rolling_mean(720, min_samples=720)
            )
            / pl.col("funding_rate").rolling_std(720, min_samples=720)
        ).over("symbol"),
    )
    .with_columns(
        f_xs_rank=(pl.col("funding_rate").rank() / pl.len()).over("bar_end"),
    )
)
```

Three kinds of transform, three different rules:

- **Rolling, trailing** (`f_mean_24h`, `f_sum_168h`) — safe. The window ends at
  the current row and every input is already settled. Always set `min_samples`
  equal to the window, or the first rows are computed from a shorter history and
  are not comparable with the rest.
- **Time-series normalisation** (`f_z_720h`) — must be **rolling or expanding**.
  A full-sample z-score leaks the future distribution into every row, including
  the training rows that precede it. This is the most commonly missed leak in the
  whole pipeline and it does not involve the join at all.
- **Cross-sectional normalisation** (`f_xs_rank`) — safe, and worth preferring.
  A rank or z-score computed `.over("bar_end")` uses only data from that instant,
  so it cannot reach forward in time no matter how the panel is sorted.

`f_age_s` is a feature in its own right, not just diagnostics: on a sub-hourly
grid, funding that settled 55 minutes ago means something different from funding
that settled 5 minutes ago.

**Candle-derived features belong here too, and they need no shift.** The row's
own `close / open`, its high-low range, a rolling realised vol over the last 24
bars — all of it is realised as of `bar_end`. This is what end-stamping buys: the
legality of a candle feature is no longer something you have to remember.

### Step 6 — the label, the one thing that looks forward

```Python
panel = panel.sort("symbol", "bar_end").with_columns(
    y=(pl.col("close") / pl.col("open")).log().shift(-HORIZON).over("symbol")
)
```

The label is the *next* bar's `open`→`close` return — the only quantity in the
panel that is not realised at `bar_end`. Sort before the shift: the loaders
return sorted frames, but a shift that depends on upstream ordering is a bug
waiting for the first reordered join.

The nulls this leaves are load-bearing. There is exactly one per symbol — 196 on
the current file — and each is that symbol's most recent bar. **That row is the
one you predict on live:** its features are complete as of now, its label does
not exist yet, and a null `y` is how you find it. The old start-stamped
convention could not represent that row at all: the bar it would have been
stamped with had not happened yet, so the completeness filter that keeps only
finished bars had already removed it.

Do **not** build the label from the row's own `open`→`close`. Under end-stamping
that return is realised at `bar_end`, so it is a feature; using it as a label
hands the model its own answer. See [§6.6](#66-using-the-rows-own-return-as-the-label).

### Step 7 — realised cost, as a separate column

If you hold through the label window `[T, T+1h)`, the funding you actually
**pay** is the one settled at `T+1h`. That is correct for PnL and a leak for
prediction — which is exactly why they get different names. Never let `cost` into
the feature list.

Take it from `funding`, by the settlement hour:

```Python
panel = panel.with_columns(
    settle=pl.col("bar_end") + pl.duration(hours=HORIZON)
).join(
    funding.select(
        "symbol",
        pl.col("funding_hour").alias("settle"),
        pl.col("funding_rate").alias("cost"),
    ),
    on=["symbol", "settle"],
    how="left",
)
```

**Prefer the join over `shift(-HORIZON)`.** The obvious version,

```Python
cost = pl.col("funding_rate").shift(-HORIZON).over("symbol")
```

is nearly equivalent, and the gap is narrower than it used to be. Now that both
files are stamped the same way they end on the same hour — the last `funding_hour`
and the last `bar_end` are both 2026-10-28 11:00 — so neither method can reach a
settlement past the final bar. Measured on the current file the join nulls 178 and
`shift(-1)` nulls 196, one per symbol.

Write the join anyway, because it *names* the settlement it wants instead of
trusting that the row below is the next hour. That trust holds today — the 1h
panel has no internal gaps, zero within-symbol steps longer than one hour — but
that is a property of this window, not a guarantee. A single missing bar turns
`shift(-1)` into a silently wrong pairing rather than a null.

### Step 8 — audit before training

See [§7](#7-the-audit). Run it every time the panel is rebuilt.

### Step 9 — split by time, with a purge

```Python
cut = panel["bar_end"].quantile(0.8, "nearest")
train = panel.filter(pl.col("bar_end") + pl.duration(hours=HORIZON) <= cut)
test = panel.filter(pl.col("bar_end") > cut)
assert train["bar_end"].max() < test["bar_end"].min()
```

```
         train bars          purge                 test bars
   ════════════════════════╗ ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╔════════════════════
                           ║                ║
           last train label                   first test bar
           closes here                        opens here
```

Never shuffle. A random split puts bar `T+1` in train and bar `T` in test, and
since funding is highly persistent the model effectively reads its own answers.
The purge removes the `HORIZON` bars whose labels would straddle the cut; with a
1-bar horizon that is one bar, but it grows with the horizon.

This is also why **cross-sectional panels need grouped splits**: all symbols at
the same `bar_end` must land on the same side of the cut, or a symbol in test
leaks the market-wide funding regime that a symbol in train is being asked to
predict. Splitting on `bar_end`, as above, does this automatically.

---

## 4. Universe and survivorship

Not a timestamp bug, but it lands in the same place and it is the most expensive
mistake on this page.

| file | symbols | history |
|---|---|---|
| `hl_funding.parquet` | 235 | 2023-05-12 → 2026-10-28 11:00 |
| `hl_candles_1d.parquet` | 235 | 2023-02-27 → 2026-10-28 |
| `hl_candles_1h.parquet` | **196** | 2026-04-02 04:00 →, rolling ~208 days only |

`MAX_CANDLES` makes the 1h file a rolling window and the API cannot backfill past
it, so **a perp that delisted before the window opened is absent from it
entirely**. An inner join of funding to 1h candles silently drops them.

That is specifically poisonous here: extreme funding carry is exactly what
precedes a perp dying, so a carry signal trained on the 1h file never sees its
own worst outcomes. Use **daily** candles for anything spanning history; treat
the 1h file as a recent window.

A perp that delists *during* the window is a different case — it is present, but
its candles stop while funding keeps printing. Spotting them is a one-liner:

```Python
(
    candles.group_by("symbol")
    .agg(last_bar=pl.col("bar_end").max())
    .filter(pl.col("last_bar") < pl.col("last_bar").max() - pl.duration(hours=2))
)
```

**17 symbols are in that state today**, frozen between 2026-05-05 08:00 and
2026-08-26 11:00. All 17 are already flagged `after_delisting`, so `load_funding`
drops their funding — but their candles stay. A left join leaves them with null
funding from their delisting onward, which is correct; the hazard is treating
the candle file alone as the universe, where a `last()` or forward-fill will
carry months-old prices to the present.

They are not survivors, so their presence does not soften the bias above.

The same logic applies to any universe filter — "top 50 by volume" must be
computed from data available at `T`, not from the full sample. End-stamping helps
here: a rolling volume rank `.over("bar_end")` is point-in-time by construction,
where the same expression over a start-stamped grid would include the bar being
ranked.

---

## 5. Choosing the join

Five things you might write. Two are correct, one is stale, two are wrong.

| join | result | verdict |
|---|---|---|
| `funding_hour == bar_end` | **904,722 of 904,722** matched | **correct**, simplest |
| asof backward on `time`, from `bar_end + LATENCY` | identical to the exact key | **correct**, generalises |
| `funding_hour == bar_end + 1h` | the funding settled at the end of the label window | **leak** (but correct for cost) |
| `funding_hour == bar_end - 1h` | the hour before the bar | stale by one bar, not a leak |
| asof backward from `bar_end` exactly | 98.5% of rows one hour stale | wrong, and silent |

### 5.1 Exact key on `bar_end`

```Python
panel = candles.join(
    funding.select("symbol", "funding_hour", "funding_rate", "premium"),
    left_on=["symbol", "bar_end"],
    right_on=["symbol", "funding_hour"],
    how="left",
)
```

`funding_hour` is unique per `(symbol, hour)` — `duplicate_hour` fires on nothing
in the whole file — so the join cannot fan out and duplicate labels. Coverage is
total: **904,722 of 904,722** rows match, with nothing unmatched anywhere in the
1h panel.

That completeness is new. Under the old start-stamped convention this join
reached for the funding of the hour *before* the bar, which does not exist for a
symbol's first bar, and six rows went unmatched — one for each perp listed inside
the window. Same-hour pairing has no such edge.

### 5.2 There is no second timestamp to get wrong

The candle files used to carry both `open_time` and `close_time`, where
`close_time` was `open_time + interval − 1ms`:

```
   open_time    2026-04-02 03:00:00.000
   close_time   2026-04-02 03:59:59.999
   funding_hour 2026-04-02 04:00:00.000   ← always exactly on the hour
```

It never landed on an hour boundary, so joining on it matched **zero rows**.
`how="inner"` gave 0 rows; `how="left"` kept every row with **all** funding values
null, which is the nastier failure — a full-height frame with an all-null feature
column looks like missing data rather than a key that is wrong by one
millisecond.

`close_time` also carried no information: it was exactly `open_time + interval −
1ms` on all 1,138,666 rows of both files. Both columns are gone. `bar_end` is
grid-aligned by construction, because it is derived from the bar's start plus one
interval rather than from the exchange's own close stamp, and the download logs a
warning if Hyperliquid's `T` ever stops being `bar_end − 1ms`.

### 5.3 asof — the one that generalises

```Python
panel = (
    candles.with_columns(decision=pl.col("bar_end") + LATENCY)
    .sort("decision")
    .join_asof(
        funding.sort("time"),
        left_on="decision",
        right_on="time",
        by="symbol",
        strategy="backward",
        tolerance="1h",
    )
)
```

On the 1h grid this reproduces the exact join **exactly**: the same matches, no
nulls on either side, zero disagreements. It costs nothing to adopt now and it is
the only thing that works on a finer grid.

**Trap A — asof from `bar_end` exactly loses an hour.** The funding stamp is
`T + 48ms`, so "the latest entry at or before `T`" is the entry from `T−1h`:

```
   funding stamps       ●                   ●
                    T-1h+48ms             T+48ms
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶
                      T-1h                  T                 T+1h

   decision = T         ╎                   ╎  48 ms too early
                        └────── picked ─────┘    ✗  1 h stale

   decision = T + 1s                        ╎
                                            └──► picked   ✓  current hour
```

| decision point | current hour | **1h stale** | no match |
|---|---|---|---|
| `bar_end` | 1.5% | **98.5%** | 6 rows |
| `bar_end + 1s` | 100.0% | 0 | 0 rows |

Being *too* literal about point-in-time correctness silently shifts 98.5% of the
feature column back an hour. Only the rows whose offset is exactly 0 ms escape.

**Trap B — without `tolerance`, asof fabricates carry.** Unbounded backward fill
is the default and it is silent:

```
   funding stamps       ●                   ✗                   ●
                                     no settlement
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶
                      T-1h                  T                 T+1h

   no tolerance         └───────────────► carries T-1h into T   ✗ invented
   tolerance="1h"                         null at T             ✓ explicit
```

It makes no difference to the current 1h panel, where same-hour coverage is
complete — which is exactly why it has to be demonstrated somewhere it bites. On
a complete hourly grid spanning all of funding history (4,271,595 rows):

| | no tolerance | `tolerance="1h"` |
|---|---|---|
| rows carrying a stale value | **14,103** | 0 |
| max staleness | **8 hours** | — |
| nulls | 0 | 14,103 |

Of those 14,103: **12,523 in the 8-hourly era** (one 8h rate smeared across 8
bars, arrived at silently) and **1,580 in the hourly era** (the three outages,
each hitting every listed symbol at once).

Zero nulls is the tell. That grid has holes; a join reporting none has filled
them for you. Set `tolerance` even where today's panel does not need it — the
panel's completeness is a property of the current window, not of the join.

A nice property of `tolerance`: when the current hour's settlement is one of the
late ones, the join returns **null** rather than a stale value, because the
previous hour's entry falls outside tolerance. At `T + 1s` you genuinely do not
know the funding yet, and the join says so instead of guessing.

**Gotcha.** Polars emits:

```
UserWarning: Sortedness of columns cannot be checked when 'by' groups provided
```

It cannot verify sortedness once `by` is passed, so a wrongly sorted input
produces silently wrong output rather than an error. Sort both frames by the asof
key immediately before the join.

### 5.4 Where asof actually earns its place

On the 1h grid asof and the exact join are the same frame. It matters the moment
you leave that grid. Measured on a 5,000-row 5m BTC series, the exact key matches
only the bars that happen to end on the hour — **393 rows, 7.9%** — while asof
matches every row that falls inside the funding file's range:

```
   funding   ●T                                                          ●T+1h
   UTC ──────┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┼────▶
   bar_end  :00  :05  :10  :15  :20  :25  :30  :35  :40  :45  :50  :55
   f_age_min  0    5   10   15   20   25   30   35   40   45   50   55
```

393 or 394 rows at each of the twelve ages, cycling cleanly. `f_age_min` is a
real feature on this grid, and the exact-key join cannot express it at all.

---

## 6. Failure modes

### 6.1 The leak is invisible to a correlation check

**This is the trap in leak-hunting itself.** You cannot catch a funding
misalignment by looking for suspiciously high feature IC:

| alignment | corr(premium, the label's own return) |
|---|---|
| the row's own funding, `[T-1h, T)` (safe) | **+0.0168** |
| funding settled at `T+1h`, `[T, T+1h)` (leaking) | **−0.0018** |

The *leaking* alignment correlates **less** with the bar's return. At ρ ≈ 0.02
noise on hourly crypto returns the leak is simply invisible — average basis over
an hour and that hour's return are contemporaneous but only weakly related.

Two consequences. First, timestamp assertions are the only reliable detector
([§7](#7-the-audit)), and they catch it cleanly. Second, the leak's size depends
entirely on your label: against returns it is near-invisible, but if your target
is next-hour funding, the basis, or realised volatility, pulling the label's own
hour over hands the answer directly.

### 6.2 Joining funding measured over the label's own window

The one people write by accident, by thinking of `funding_hour` as "the hour this
was earned over" and joining it to the bar that covers that hour:

```Python
# WRONG
panel = candles.join(
    funding.with_columns(earned=pl.col("funding_hour") - pl.duration(hours=1)),
    left_on=["symbol", "bar_end"],
    right_on=["symbol", "earned"],
    how="left",
)
```

```
                      T-1h                  T                 T+1h
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶
                                            │
   row stamped T        [═══════════════════)  realised
   label   y(T)                             [═══════════════════)
   funding settled T+1h                     [═══════════════════)●
                                            │                    ↑
                                     decision point         60 min in the future,
                                                            measured over the SAME
                                                            span as the label
```

The label window `[T, T+1h)` *is* the window that premium was averaged over. A
model handed both is being shown a summary of its own label. The audit catches it
on 904,544 rows — see [§7](#7-the-audit).

### 6.3 The same mistake, reached from the other side

"The funding that settles at the end of the label window" (`bar_end + 1h`) feels
correct and produces a **byte-identical column** to §6.2 — verified with
`.equals()`. That stamp is exactly 60 minutes after `bar_end` (min, median and
max all 60), so the value does not exist at the decision point.

It is, however, the right column for **cost accounting** — see
[Step 7](#step-7--realised-cost-as-a-separate-column). Keep both, named
differently, and never let the cost column into the feature list.

### 6.4 Forward-filling across the 8-hourly era

Hyperliquid settled funding every 8 hours until **2023-06-08 00:00 UTC**, hourly
from 01:00 the same day. Confirmed by magnitude, not just spacing: BTC's modal
rate is `0.0001` per 8h before and `0.0000125` per hour after — exactly one
eighth — and mean `|funding_rate|` is 7.08× higher in the old era.

Forward-filling invents 7 observations out of every 8 **and** labels each with a
rate 8× too large. Not look-ahead in the strict sense, but it corrupts any
cross-era scaling, which produces the same symptom: a backtest that works and a
live book that does not. Drop the era (Step 2). If you ever need it, divide
pre-switch rates by 8 and keep an era dummy — never fill forward.

### 6.5 Assuming you can trade at the open tick

Funding is *recorded* slightly after the top of the hour: median **48 ms**, p99
**420 ms**, with a thin tail to **23.9 minutes** (571 rows, in bursts hitting
every listed symbol at once — they look like recovery after an outage).

```Python
off = (f["time"] - f["funding_hour"]).dt.total_milliseconds()
print(off.min(), off.median(), off.quantile(0.99), off.max())  # 0, 48, 420, 1433040
```

So "known at `T`" holds up to a recording latency, not at the exact tick. All 571
late rows predate the current 1h window; inside it nothing is more than **1
second** late. Setting `LATENCY` honestly in Step 3 is what makes this explicit —
and if you want the strict guarantee, lag the feature a full bar:

```Python
panel = panel.sort("symbol", "bar_end").with_columns(
    f_lag1=pl.col("funding_rate").shift(1).over("symbol")
)
```

Sort explicitly before any `shift`. The loaders happen to return sorted frames,
but a `shift` that depends on upstream ordering is a bug waiting for the first
reordered join.

### 6.6 Using the row's own return as the label

The one this convention creates, and the one to watch for in code written against
the old start-stamped files. Under the old scheme a candle row stamped `T` was the
hour `[T, T+1h)`, so `y = log(close / open)` of the same row was the correct,
look-ahead-free label. Under end-stamping that same expression is the return of
`[T-1h, T)` — already realised at `bar_end`, already available as a feature, and
as a label it simply hands the model its answer.

```Python
y = (pl.col("close") / pl.col("open")).log()                       # WRONG now
y = (pl.col("close") / pl.col("open")).log().shift(-1).over("symbol")  # right
```

The tell is the right edge. The correct label is null on each symbol's last bar;
the wrong one is not, which is what the fifth audit check looks at. A label
shifted the wrong way (`shift(+1)`) fails the same check, so one assertion covers
both.

---

## 7. The audit

Run this every time the panel is rebuilt. It is the only reliable detector of
§6.1–6.3.

```Python
def audit(df, latency=LATENCY, era_start=HOURLY_FROM):
    d = df.drop_nulls("time")
    return {
        "funding stamped after the decision point": d.filter(
            pl.col("time") > pl.col("decision")
        ).height,
        "funding older than one settlement interval": d.filter(
            pl.col("decision") - pl.col("time") >= pl.duration(hours=1) + latency
        ).height,
        "rows before the hourly era": df.filter(pl.col("bar_end") < era_start).height,
        "duplicate (symbol, bar_end)": df.select(
            pl.struct("symbol", "bar_end").is_duplicated().sum()
        ).item(),
        "label present on a symbol's last bar": df.filter(
            pl.col("bar_end") == pl.col("bar_end").max().over("symbol")
        )["y"].drop_nulls().len(),
    }


for k, v in audit(panel).items():
    print(f"{'OK  ' if v == 0 else 'FAIL'} {k}: {v:,}")
```

On the Step-6 panel, all five pass. On the leaking panel from §6.2–6.3:

```
FAIL funding stamped after the decision point: 904,544
OK   funding older than one settlement interval: 0
OK   rows before the hourly era: 0
OK   duplicate (symbol, bar_end): 0
OK   label present on a symbol's last bar: 0
```

That is every row the leaking join matched. The first check catches on
essentially the whole panel what the correlation check could not see at all. This
is why `time` must survive the join — `load_funding` returns it alongside
`funding_hour`, and `clean=True` does not strip it.

The fifth check guards the label instead of the join, and it is the one
end-stamping made possible. It fires on 196 rows — one per symbol — for both
label mistakes:

```
# y = log(close / open), the old convention's correct label
FAIL label present on a symbol's last bar: 196

# y = log(close / open).shift(+1), the shift pointing the wrong way
FAIL label present on a symbol's last bar: 196
```

It replaces an earlier check that failed whenever the funding column had **zero**
nulls, on the reasoning that a join reporting no holes had filled the exchange's
real ones. That check now fires on a correct panel: same-hour pairing on the
hourly era genuinely has 100% coverage, because every candle hour has a
settlement. Keeping it would have trained you to ignore a FAIL. `tolerance` is
what actually prevents invented carry ([§5.3](#53-asof--the-one-that-generalises)),
and [Trap B](#53-asof--the-one-that-generalises) is where that is measured.

---

## 8. Quick reference

| question | answer |
|---|---|
| ML timestamp? | `bar_end` (or `funding_hour` — same value) |
| what does a stamp mean? | everything in the row is known at it, and describes the hour before it |
| join? | asof, `strategy="backward"`, on raw `time`, from `bar_end + LATENCY`, with `tolerance="1h"` |
| exact-key equivalent? | `funding_hour == bar_end` — same result on a 1h grid, 100% coverage |
| label? | **next** bar's `open`→`close` return, `shift(-HORIZON).over("symbol")` |
| candle features? | safe unshifted — the row's own OHLCV is realised at `bar_end` |
| funding actually paid? | join on `bar_end + HORIZON` — cost only, never a feature |
| normalise across time? | rolling or expanding window only |
| normalise across symbols? | `.over("bar_end")` — safe |
| candles for history? | 1d (235 symbols). 1h is 196 survivors only |
| sample start? | 2023-06-08 01:00 UTC |
| CV? | time-ordered, purged by `HORIZON`, grouped on `bar_end`, never shuffled |
| calendar day for a 1d bar? | `bar_end - 1d` — the bar is stamped at the following midnight |

---

## Open items

- **How late does funding arrive live?** The live counterpart to §6.5. The
  history says median 48 ms with a 23.9-minute tail, but that is the *recorded*
  stamp in a backfill. Whether the websocket delivers it that promptly is a
  separate measurement, and it is the one that sets `LATENCY` honestly.
- Nothing here covers HIP-3 perps, which use a different premium definition.
- The 5m path is sketched in §5.4 but not built. The funding side needs no change;
  only the candle grid and `f_age_min` do.
