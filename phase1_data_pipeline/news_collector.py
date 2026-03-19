"""
Phase 1 — Script 2: news_collector.py
Pulls news headlines for 10 benchmark tickers from the Alpha Vantage
News & Sentiment API for the Covid-19 window (2020-01-01 – 2020-04-30).
Saves structured JSON to phase1_data_pipeline/data/news/covid_news.json.
"""

import json
import os
import time
import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY    = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL   = "https://www.alphavantage.co/query"
TIME_FROM  = "20200101T0000"
TIME_TO    = "20200430T2359"
DELAY_SECS = 12   # free tier: 5 calls/min

TICKERS = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "news")
OUTPUT_JSON = os.path.join(DATA_DIR, "covid_news.json")

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_articles(ticker: str) -> list[dict]:
    """Call Alpha Vantage NEWS_SENTIMENT for one ticker and return raw feed items."""
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ticker,
        "time_from": TIME_FROM,
        "time_to":   TIME_TO,
        "limit":     1000,
        "apikey":    API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "feed" not in data:
        # Surface any API-level error messages
        msg = data.get("Information") or data.get("Note") or str(data)
        print(f"  [WARN] {ticker}: no feed returned — {msg}")
        return []

    return data["feed"]

# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_article(item: dict, ticker: str) -> dict:
    """Map one Alpha Vantage feed item to our standard article schema."""
    # AV date format: "20200311T120000" → "2020-03-11"
    raw_date = item.get("time_published", "")
    date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) >= 8 else ""

    return {
        "date":      date,
        "headline":  item.get("title", ""),
        "publisher": item.get("source", ""),
        "ticker":    ticker,
        "url":       item.get("url", ""),
        "entities":  [],
        "sentiment": None,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY or API_KEY == "paste_your_key_here":
        raise SystemExit("ERROR: ALPHAVANTAGE_API_KEY not set in .env")

    os.makedirs(DATA_DIR, exist_ok=True)

    all_articles: list[dict] = []
    counts: dict[str, int] = {}

    print(f"Fetching news for {len(TICKERS)} tickers  |  {TIME_FROM[:8]} → {TIME_TO[:8]}")
    print(f"Rate-limit delay: {DELAY_SECS}s between calls\n")

    for i, ticker in enumerate(TICKERS):
        print(f"[{i+1}/{len(TICKERS)}] {ticker} ...", end=" ", flush=True)
        items = fetch_articles(ticker)
        articles = [parse_article(item, ticker) for item in items]
        all_articles.extend(articles)
        counts[ticker] = len(articles)
        print(f"{len(articles)} articles")

        if i < len(TICKERS) - 1:
            time.sleep(DELAY_SECS)

    # Save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nSaved {len(all_articles)} total articles → {OUTPUT_JSON}\n")
    print("Articles per ticker:")
    for ticker in TICKERS:
        count = counts.get(ticker, 0)
        bar = "#" * (count // 10)
        print(f"  {ticker:>5}:  {count:>4}  {bar}")


if __name__ == "__main__":
    main()
