#!/usr/bin/env python3
import os, hashlib, sys
import feedparser
import psycopg2, psycopg2.extras
from email.utils import parsedate_to_datetime
from datetime import datetime

FEED_URL = "http://feeds.marketwatch.com/marketwatch/topstories/"
SOURCE = "MarketWatch"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ripplestocks:ripplestocks@localhost:5433/ripplestocks"
)

def make_hash(title, source, pubdate):
    return hashlib.sha256(f"{title}|{source}|{pubdate}".encode()).hexdigest()

def main():
    # DB connect
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print("DB connection failed:", e); sys.exit(1)

    # fetch & parse RSS
    feed = feedparser.parse(FEED_URL)
    if feed.bozo:
        print("RSS parse warning:", feed.bozo_exception)

    inserted = 0; skipped = 0
    with conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        for entry in feed.entries:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            pub_raw = entry.get("published") or entry.get("updated") or ""
            if not (title and link and pub_raw):
                skipped += 1; continue

            try:
                published_at = parsedate_to_datetime(pub_raw)
            except Exception:
                skipped += 1; continue

            h = make_hash(title, SOURCE, pub_raw)
            cur.execute(
                """
                INSERT INTO news_raw (source, headline, url, published_at, hash, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (hash) DO NOTHING
                RETURNING id;
                """,
                (SOURCE, title, link, published_at, h, datetime.now())
            )
            row = cur.fetchone()
            inserted += 1 if row else 0
            skipped  += 0 if row else 1

    conn.close()
    print(f"Done. Inserted: {inserted}  Skipped: {skipped}")

if __name__ == "__main__":
    main()