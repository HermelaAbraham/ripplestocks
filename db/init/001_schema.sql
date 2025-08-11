CREATE TABLE IF NOT EXISTS news_raw (
  id SERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  headline TEXT NOT NULL,
  url TEXT NOT NULL,
  published_at TIMESTAMPTZ NOT NULL,
  hash TEXT UNIQUE NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS signals (
  id SERIAL PRIMARY KEY,
  news_id INT REFERENCES news_raw(id) ON DELETE CASCADE,
  primary_ticker TEXT,
  sentiment_score DOUBLE PRECISION,
  ripple_score DOUBLE PRECISION,
  direction TEXT,
  computed_at TIMESTAMPTZ
);