import sqlite3
from pathlib import Path

# db.py -> data -> wh_research_pure -> src -> project root
# Path(__file__).resolve().parent gets the path of the folder one folder up
# .parent moves up once, .parent[1] moves up an additional time, .parent[3] moves up four times
PROJECT_ROOT = Path(__file__).resolve().parent[3]

# path for database file to be stored
DB_PATH = PROJECT_ROOT / "data"/ "hyperliquid.db"

def get_conn():
    """Open a connection to the Hyperliquid database, creating it if missing."""
    # Ensure the parent directory exists before establishing the connection.
    DB_PATH.parent.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # WAL mode is enabled to improve concurrent read/write performance.
    conn.execute("PRAGMA journal_mode=WAL;") 

    return conn

def init_db():
    with get_conn as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol     TEXT    NOT NULL,
                interval   TEXT    NOT NULL,
                open_time  INTEGER NOT NULL,   -- ms since epoch
                close_time INTEGER NOT NULL,   -- ms since epoch
                open       REAL,
                high       REAL,
                low        REAL,
                close      REAL,
                volume     REAL,
                trades     INTEGER,
                PRIMARY KEY (symbol, interval, open_time)
                )
        """)