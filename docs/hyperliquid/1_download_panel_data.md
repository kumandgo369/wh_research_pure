# Downloading panel data guide

---

## Update everything in one go

```Shell
uv run python scripts/update_data.py --verbose
```

Runs funding, then 1d candles, then 1h candles. Each step is independent: if one
fails the rest still run, the traceback is logged, and the exit code is 1 so a
scheduler notices. A step that downloads nothing reports 0 rows, which is not a
failure. Ctrl-C still stops the run.

**Run this regularly, for the 1h candles.** `MAX_CANDLES` makes the candle
download a rolling window — about 208 days at 1h — and the API cannot backfill
beyond it, so any stretch missed for longer than that is gone permanently. The
other two heal themselves: 1d re-downloads its whole history every run (5,000
days is longer than Hyperliquid has existed), and funding resumes from the last
stored entry per symbol.

Funding runs first so the candle files, which `load_funding` reads to cross-check
delistings, end up at least as fresh as the funding they validate.

---

## Run the following command to download / update OHLCV panel data

```Shell
uv run python scripts/download_candles.py --interval 1h --verbose
uv run python scripts/download_candles.py --interval 1d --verbose
```

- Change the interval to one of the following supported intervals: "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w"

## Run the following command to download / update hourly funding panel data

```Shell
uv run python scripts/download_funding.py --verbose
```

---

Both scripts expose their work as `update_candles(interval)` and
`update_funding()`, which return the number of rows written and leave logging to
the caller. `update_data.py` uses those; `main()` is only the CLI wrapper.
