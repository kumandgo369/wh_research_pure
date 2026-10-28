# Funding timestamps and look-ahead bias

Working notes on how to timestamp Hyperliquid funding for supervised learning.
Every number and every code block here was run against the files in `data/`,
snapshot 2026-10-28. Row counts grow with every download, so proportions are
quoted where the proportion is the point; re-measure before citing an absolute.

**The rule:** a funding row stamped `T` describes the hour *before* `T`. A candle
row stamped `T` describes the hour *after* `T`. Line them up on the same label
and the funding is known at the moment the bar opens, which is exactly what
supervised learning wants. Everything else in this document is a consequence of
that one sentence.

Jump to [§3 The recipe](#3-the-recipe-funding-as-an-ml-feature) if you just want
the pipeline.

---

## 1. The two clocks point opposite ways

Both series are hourly, both are UTC, both label rows with a timestamp on the
hour — and the label means opposite things.

```
                      T-1h                  T                 T+1h
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶

   FUNDING row labelled T
     premium sampled    [═══════════════════)
     settled / knowable                     ●  time = T + 48 ms
     funding_hour                           │  = T
                                            │
   CANDLE row labelled T                    │
     bar span                               [═══════════════════)
     open_time                              ●  = T
     close_time                                                ●  = T+1h-1ms

                                            │
                            the same label T means
                            "just ended"     for funding
                            "about to start" for candles
```

| series | a row stamped `T` describes | direction | status at `T` |
|---|---|---|---|
| funding (`funding_hour`) | the hour `[T-1h, T)` that just ended | **backward** | realised, known |
| candles (`open_time`) | the hour `[T, T+1h)` about to start | **forward** | unknown |

A funding entry is a *settlement*: at the top of hour `T` the exchange charges
holders for the hour they have just held through. A candle is a *bucket*: the bar
labelled `T` is the hour that starts at `T`.

So the natural alignment is also the safe one:

```
                      T-1h                  T                 T+1h
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶
                                            │
   feature  f(T)        ────────────────────●  known at T
   label    y(T)                            [═══════════════════)  realised after T
                                            │
                                     decision point
```

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
    .agg(first_bar=pl.col("open_time").min())
)
first_fund = f.group_by("symbol").agg(first_fh=pl.col("funding_hour").min())

print(
    first_bar.join(first_fund, on="symbol")
    .filter(pl.col("first_bar") > c["open_time"].min())  # genuinely new listings
    .with_columns(delta_h=(pl.col("first_fh") - pl.col("first_bar")).dt.total_hours())
    .select("symbol", "first_bar", "first_fh", "delta_h")
    .sort("first_bar")
)
```

For the six perps listed since the 1h file begins — CHIP, GRAM, CASHCAT, PONS,
USELESS, XTEN — `delta_h` is **exactly 1** in every case:

```
     first traded bar         first funding entry
        03:00  ●────────────────●  04:00
               [════════════════)
               the first full hour of holding,
               settled at its end
```

A perp that starts trading at 03:00 gets its first funding at 04:00, because
04:00 is when the first full hour of holding gets settled. If funding at `T`
summarised `[T, T+1h)` the first entry would be at 03:00. It is not. The tail
agrees: a live symbol's last funding entry is at, or one hour past, its last
traded candle.

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
funding = load_funding()  # drops 622,166 post-delisting zeros + integrity failures
candles = load_candles("1h")  # trims pre-launch oracle rows, drops invalid bars
```

Reading the parquet directly leaves 12.8% of the funding file as artificial
zeros from delisted perps, which drags down every cross-sectional statistic.

### Step 2 — restrict to the hourly era

```Python
funding = funding.filter(pl.col("funding_hour") >= HOURLY_FROM).sort("time")
candles = candles.sort("symbol", "open_time")
```

Before 2023-06-08 funding settled every 8 hours at 8× the magnitude. That era is
1,814 rows across 25 symbols — **0.043% of the clean file**. Not worth modelling;
dropping it removes a whole class of scaling bug. See [§6.4](#64-forward-filling-across-the-8-hourly-era).

### Step 3 — make the decision timestamp explicit

```Python
candles = candles.with_columns(decision=pl.col("open_time") + LATENCY)
```

This is the single most useful line in the pipeline. It turns "when do I know
things?" from an assumption buried in a join into a column you can assert on.
`LATENCY` is your claim about how fast you can react to a settlement; everything
downstream is only as honest as that number.

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
    .sort("symbol", "open_time")
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
        f_xs_rank=(pl.col("funding_rate").rank() / pl.len()).over("open_time"),
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
  A rank or z-score computed `.over("open_time")` uses only data from that
  instant, so it cannot reach forward in time no matter how the panel is sorted.

`f_age_s` is a feature in its own right, not just diagnostics: on a sub-hourly
grid, funding that settled 55 minutes ago means something different from funding
that settled 5 minutes ago.

### Step 6 — the label, from the same row

```Python
panel = panel.with_columns(y=(pl.col("close") / pl.col("open")).log())
```

Build the target from `open`→`close` of the *same* row. A close-to-close
`pct_change` assigns the return of `[T-1h, T)` to the row labelled `T`, which
puts the label one bar behind the features and recreates [§6.2](#62-joining-funding-at-t-to-the-bar-it-was-measured-over) from the other side.

### Step 7 — realised cost, as a separate column

If you hold through `[T, T+1h)`, the funding you actually **pay** is the one
settled at `T+1h`. So close-alignment is correct for PnL and wrong for
prediction — which is exactly why they get different names. Never let `cost`
into the feature list.

Take it from `funding`, by the settlement hour:

```Python
panel = panel.with_columns(
    settle=pl.col("open_time") + pl.duration(hours=HORIZON)
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

**Not `shift(-HORIZON)`.** The obvious version,

```Python
cost = pl.col("funding_rate").shift(-HORIZON).over("symbol")  # loses the last bar
```

looks equivalent and is, except at the right edge. The panel is indexed by
*candles*, and funding runs one settlement ahead of the last candle (§1) — so the
settlement that pays for the final bar has no candle row for `shift` to reach.
It silently returns null for the most recent bar of **every** symbol, which is
the bar you care about most when the panel is rebuilt to trade today. Measured on
the current file: `shift(-1)` nulls 196 of 196 last bars; the join nulls 1.

### Step 8 — audit before training

See [§7](#7-the-audit). Run it every time the panel is rebuilt.

### Step 9 — split by time, with a purge

```Python
cut = panel["open_time"].quantile(0.8, "nearest")
train = panel.filter(pl.col("open_time") + pl.duration(hours=HORIZON) <= cut)
test = panel.filter(pl.col("open_time") > cut)
assert train["open_time"].max() < test["open_time"].min()
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
the same `open_time` must land on the same side of the cut, or a symbol in test
leaks the market-wide funding regime that a symbol in train is being asked to
predict. Splitting on `open_time`, as above, does this automatically.

---

## 4. Universe and survivorship

Not a timestamp bug, but it lands in the same place and it is the most expensive
mistake on this page.

| file | symbols | history |
|---|---|---|
| `hl_funding.parquet` | 235 | 2023-05-12 → now |
| `hl_candles_1d.parquet` | 235 | full |
| `hl_candles_1h.parquet` | **196** | rolling ~208 days only |

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
    .agg(last_bar=pl.col("open_time").max())
    .filter(pl.col("last_bar") < pl.col("last_bar").max() - pl.duration(hours=2))
)
```

**17 symbols are in that state today**, frozen between 2026-05-05 and
2026-08-26. All 17 are already flagged `after_delisting`, so `load_funding`
drops their funding — but their candles stay. A left join leaves them with null
funding from their delisting onward, which is correct; the hazard is treating
the candle file alone as the universe, where a `last()` or forward-fill will
carry months-old prices to the present.

They are not survivors, so their presence does not soften the bias above.

The same logic applies to any universe filter — "top 50 by volume" must be
computed from data available at `T`, not from the full sample.

---

## 5. Choosing the join

Five things you might write. Two are correct, one is a no-op, two are wrong.

| join | result | verdict |
|---|---|---|
| `funding_hour == open_time` | all but 6 rows matched | **correct**, simplest |
| `funding_hour == close_time` | **0 rows** matched | broken, fails silently |
| `funding_hour == close_time.truncate("1h")` | identical to `open_time` | same thing, longer |
| asof backward on `time`, from `decision` | identical to `open_time` | **correct**, generalises |
| `funding_hour == open_time + 1h` | the funding settled at the bar's close | **leak** (but correct for cost) |

### 5.1 Exact key on `open_time`

```Python
panel = candles.join(
    funding.select("symbol", "funding_hour", "funding_rate", "premium"),
    left_on=["symbol", "open_time"],
    right_on=["symbol", "funding_hour"],
    how="left",
)
```

`funding_hour` is unique per `(symbol, hour)` — `duplicate_hour` fires on nothing
in the whole file — so the join cannot fan out and duplicate labels. Coverage is
near-total: 6 unmatched rows in the entire 1h panel.

### 5.2 Why `close_time` does not work

`close_time` is always `open_time + 1h − 1ms`, so it never lands on an hour
boundary — zero rows in the file do:

```
   open_time    2026-04-02 03:00:00.000
   close_time   2026-04-02 03:59:59.999
   funding_hour 2026-04-02 03:00:00.000   ← always exactly on the hour
```

`how="inner"` gives **0 rows**; `how="left"` keeps every row with **all** funding
values null. The left join is the nastier failure: a full-height frame
with an all-null feature column looks like missing data rather than a key that is
wrong by one millisecond.

Truncating first (`close_time.dt.truncate("1h")`) just recovers `open_time`.
Verified identical with `.equals()`.

### 5.3 asof — the one that generalises

```Python
panel = (
    candles.with_columns(decision=pl.col("open_time") + LATENCY)
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

On the 1h grid this reproduces the exact join **exactly**: the same matches, the
same 6 nulls, zero disagreements. It costs nothing to adopt now and it is the
only thing that works on a finer grid.

**Trap A — asof from `open_time` exactly loses an hour.** The funding stamp is
`T + 48ms`, so "the latest entry at or before `T`" is the entry from `T−1h`:

```
   funding stamps       ●                   ●
                    T-1h+48ms             T+48ms
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶
                      T-1h                  T                 T+1h

   decision = T         ╎                   ╎  48 ms too late
                        └────── picked ─────┘    ✗  1 h stale

   decision = T + 1s                        ╎
                                            └──► picked   ✓  current hour
```

| decision point | current hour | **1h stale** | no match |
|---|---|---|---|
| `open_time` | 1.5% | **98.5%** | 12 rows |
| `open_time + 1s` | 100.0% | 0 | 6 rows |

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

Measured on a full-history hourly grid:

| | no tolerance | `tolerance="1h"` |
|---|---|---|
| rows carrying a stale value | **14,124** | 21 |
| max staleness | **8 hours** | 1 hour |
| nulls | 0 | 14,103 |

Of those 14,124: **12,544 in the 8-hourly era** (one 8h rate smeared across 8
bars, arrived at silently) and **1,580 in the hourly era** (the three outages,
each hitting every listed symbol at once).

Zero nulls is the tell. The data has holes; a join reporting none has filled them
for you.

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
you leave that grid. Measured on a 5,000-row 5m candle series, the exact-key join
matches nothing at all while asof matches every row:

```
   funding   ●T                                                          ●T+1h
   UTC ──────┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┼────▶
            :00  :05  :10  :15  :20  :25  :30  :35  :40  :45  :50  :55
   f_age_min 0    5   10   15   20   25   30   35   40   45   50   55
```

~417 rows at each age, cycling cleanly. `f_age_min` is a real feature on this
grid, and the exact-key join cannot even express it.

---

## 6. Failure modes

### 6.1 The leak is invisible to a correlation check

**This is the trap in leak-hunting itself.** You cannot catch a funding
misalignment by looking for suspiciously high feature IC:

| alignment | corr(premium, that bar's own return) |
|---|---|
| open-aligned (safe) | **+0.0169** |
| close-aligned (leaking) | **−0.0016** |

The *leaking* alignment correlates **less** with the bar's return. At ρ ≈ 0.02
noise on hourly crypto returns the leak is simply invisible — average basis over
an hour and that hour's return are contemporaneous but only weakly related.

Two consequences. First, timestamp assertions are the only reliable detector
([§7](#7-the-audit)), and they catch it cleanly. Second, the leak's size depends
entirely on your label: against returns it is near-invisible, but if your target
is next-hour funding, the basis, or realised volatility, close-alignment hands
over the answer directly.

### 6.2 Joining funding at `T` to the bar it was measured over

The one people write by accident, by thinking of `funding_hour` as "the hour this
was earned over" and joining it to that bar:

```Python
# WRONG
panel = candles.join(
    funding.with_columns(earned=pl.col("funding_hour") - pl.duration(hours=1)),
    left_on=["symbol", "open_time"],
    right_on=["symbol", "earned"],
    how="left",
)
```

```
                      T-1h                  T                 T+1h
   UTC ─────────────────┼───────────────────┼───────────────────┼────────▶
                                            │
   label   y(T)                             [═══════════════════)
   funding at T+1h                          [═══════════════════)●
                                            │                    ↑
                                     decision point         60 min in the future,
                                                            measured over the SAME
                                                            span as the label
```

The bar `[T, T+1h)` *is* the window the premium was averaged over. A model handed
both is being shown a summary of its own label.

### 6.3 The same mistake, reached from the other side

"The funding that settles at the end of this bar" (`open_time + 1h`) feels
correct and produces a **byte-identical column** to §6.2 — verified with
`.equals()`. The funding stamp is exactly 60 minutes after the bar opens (min,
median and max all 60), so the value does not exist at the decision point.

It is, however, the right column for **cost accounting** — see
[Step 7](#step-7-realised-cost-as-a-separate-column). Keep both, named
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
panel = panel.sort("symbol", "open_time").with_columns(
    f_lag1=pl.col("funding_rate").shift(1).over("symbol")
)
```

Sort explicitly before any `shift`. The loaders happen to return sorted frames,
but a `shift` that depends on upstream ordering is a bug waiting for the first
reordered join.

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
        "rows before the hourly era": df.filter(pl.col("open_time") < era_start).height,
        "duplicate (symbol, open_time)": df.select(
            pl.struct("symbol", "open_time").is_duplicated().sum()
        ).item(),
        "no missing funding at all (suspicious)": int(
            df["funding_rate"].null_count() == 0
        ),
    }


for k, v in audit(panel).items():
    print(f"{'OK  ' if v == 0 else 'FAIL'} {k}: {v:,}")
```

On the Step-4 panel, all five pass. On the close-aligned panel from §6.3:

```
FAIL funding stamped after the decision point: 904,721
OK   funding older than one settlement interval: 0
OK   rows before the hourly era: 0
OK   duplicate (symbol, open_time): 0
OK   no missing funding at all (suspicious): 0
```

That is every row but one. The first check catches on essentially the whole panel
what the correlation check could not see at all. This is why `time` must survive
the join — `load_funding` returns
it alongside `funding_hour`, and `clean=True` does not strip it.

The last check is deliberately inverted: a funding column with **zero** nulls
means the join filled the exchange's real holes, so it fails.

---

## 8. Quick reference

| question | answer |
|---|---|
| ML timestamp? | `funding_hour` (or `open_time` — same value) |
| join? | asof, `strategy="backward"`, on raw `time`, from `open_time + LATENCY`, with `tolerance="1h"` |
| exact-key equivalent? | `funding_hour == open_time` — same result on a 1h grid |
| label? | same row's `open`→`close` return |
| funding actually paid? | `funding_rate.shift(-HORIZON)` — cost only, never a feature |
| normalise across time? | rolling or expanding window only |
| normalise across symbols? | `.over("open_time")` — safe |
| candles for history? | 1d (234 symbols). 1h is 196 survivors only |
| sample start? | 2023-06-08 01:00 UTC |
| CV? | time-ordered, purged by `HORIZON`, grouped on `open_time`, never shuffled |

---

## Open items

- **How late does funding arrive live?** The live counterpart to §6.5. The
  history says median 48 ms with a 23.9-minute tail, but that is the *recorded*
  stamp in a backfill. Whether the websocket delivers it that promptly is a
  separate measurement, and it is the one that sets `LATENCY` honestly.
- Nothing here covers HIP-3 perps, which use a different premium definition.
- The 5m path is sketched in §5.4 but not built. The funding side needs no change;
  only the candle grid and `f_age_min` do.
