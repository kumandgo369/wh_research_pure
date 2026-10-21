import random
import sys
import time

from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError, ServerError

from wh_research_pure.data.db import get_conn, init_db

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000,"4h": 14_400_000, "8h": 28_000_000, "12h": 43_200_000, "1d": 86_400_000
}

MAX_CANDLES = 5000
BUDGET_PER_SEC = (1200 / 60) * 0.9         # 1200 weight/min per IP, with a 10% safety margin

def request_weight(n_items: int) -> int:
    """candleSnapshot: base weight 20, plus 1 per 60 items returned."""
    return 20 + n_items // 60

def fetch_with_retry(info, symbol, interval, start, end, tries=6, base_wait=10, max_wait=120):
    wait = base_wait
    for attempt in range(1, tries + 1):
        try:
            return info.candles_snapshot(symbol, interval, start, end)
        except ClientError as e:        # Save ClientError class as e
            if e.status_code != 429:    # ClientError.status_code is the error status code, 429 is a rate limit
                raise                   # Non 429 error status code errors are "real" errors, e.g. non rate limit errors
            reason = "rate limited"
        except ServerError as e:        # Save ServerError class as e
            reason = f"server error {e.status_code}"

        if attempt == tries:
            raise RuntimeError(f"{symbol}: giving up after {tries} attempts ({reason})")