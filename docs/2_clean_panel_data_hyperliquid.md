# Cleaning panel data guide

---

## Cleaning OHLCV data

```Python
# Step 1, import the cleaning functions
from wh_research_pure.data_util.hyperliquid_clean.py import flag_invalid_candles, trim_leading_zero_volume

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

---

## Cleaning Funding data

```Python
```
