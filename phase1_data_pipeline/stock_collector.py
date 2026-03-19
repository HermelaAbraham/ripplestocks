"""
Phase 1 — Script 1: stock_collector.py
Pulls daily OHLCV price data for benchmark tickers over the Covid-19 window
(Jan 1 – Apr 30, 2020) and saves one CSV per ticker.
"""

import os
import yfinance as yf

TICKERS = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]
START_DATE = "2020-01-01"
END_DATE = "2020-04-30"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "stocks")


def fetch_and_save(ticker: str) -> None:
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if df.empty:
        print(f"[WARN]  {ticker}: no data returned — skipping.")
        return

    # Flatten MultiIndex columns produced by yfinance when auto_adjust=True
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    out_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
    df.to_csv(out_path)
    print(f"[OK]    {ticker}: {len(df)} rows saved → {out_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Fetching daily OHLCV data  |  {START_DATE} → {END_DATE}\n")
    for ticker in TICKERS:
        fetch_and_save(ticker)
    print("\nDone.")


if __name__ == "__main__":
    main()
