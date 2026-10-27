# Downloading panel data guide

---

## Run the following command to download / update OHLCV panel data

```Shell
uv run python scripts/download_candles.py --interval 1h --verbose 
uv run python scripts/download_candles.py --interval 1d --verbose
```

- Change the interval to one of the following supported intervals: "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w"

## Run the following command to download / update hourly funding panel data

```
uv run python scripts/download_funding.py --verbose
```

---
