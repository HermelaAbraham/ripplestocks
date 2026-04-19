"""
Phase 2 — Script 1: entity_extractor.py
Extracts named entities (ORG) and ticker mentions from news headlines
using spaCy (en_core_web_sm).

Input:  phase1_data_pipeline/data/news/covid_news.json
Output: phase2_nlp_layer/data/covid_news_entities.json

Each article's `entities` field is populated with a list of dicts:
    {"text": str, "label": str, "ticker": str | None}

  text   — the matched surface form (e.g. "Microsoft", "ZM")
  label  — spaCy NER label (e.g. "ORG") or "TICKER" for direct symbol matches
  ticker — resolved ticker symbol when known, else None
"""

import json
import os
import re
import spacy

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSON  = os.path.join(ROOT_DIR, "phase1_data_pipeline", "data", "news", "covid_news.json")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "covid_news_entities.json")

# ── Ticker alias map ──────────────────────────────────────────────────────────
# Maps common company name variations → canonical ticker symbol.
# Ordered longest-first within groups so "American Airlines" beats "American".

TICKER_ALIASES: dict[str, str] = {
    # ZM — Zoom Video Communications
    "Zoom Video":            "ZM",
    "Zoom Video Communications": "ZM",
    "Zoom":                  "ZM",

    # MSFT — Microsoft
    "Microsoft":             "MSFT",

    # AMZN — Amazon
    "Amazon Web Services":   "AMZN",
    "Amazon":                "AMZN",
    "AWS":                   "AMZN",

    # NFLX — Netflix
    "Netflix":               "NFLX",

    # MRNA — Moderna
    "Moderna":               "MRNA",

    # AAL — American Airlines
    "American Airlines Group": "AAL",
    "American Airlines":     "AAL",

    # UAL — United Airlines
    "United Airlines Holdings": "UAL",
    "United Airlines":       "UAL",

    # MGM — MGM Resorts International
    "MGM Resorts International": "MGM",
    "MGM Resorts":           "MGM",
    "MGM":                   "MGM",

    # XOM — ExxonMobil
    "ExxonMobil":            "XOM",
    "Exxon Mobil":           "XOM",
    "Exxon":                 "XOM",

    # SPY — S&P 500 ETF / references
    "S&P 500":               "SPY",
    "S&P500":                "SPY",
    "SPDR":                  "SPY",
}

# All tracked tickers (exact uppercase symbols that may appear in headlines)
TRACKED_TICKERS = {"ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"}

# Pre-compile a regex that matches bare ticker symbols as whole words
# e.g. "ZM" but not "ZMQ" or "FZM"
_TICKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(TRACKED_TICKERS, key=len, reverse=True)) + r")\b"
)


# ── Core extraction ───────────────────────────────────────────────────────────

def resolve_ticker(text: str) -> str | None:
    """Return a known ticker for an entity surface form, or None."""
    # 1. Direct alias lookup (case-sensitive, longest-match order in dict)
    if text in TICKER_ALIASES:
        return TICKER_ALIASES[text]

    # 2. Case-insensitive alias scan (covers 'ZOOM', 'zoom', etc.)
    text_lower = text.lower()
    for alias, ticker in TICKER_ALIASES.items():
        if alias.lower() == text_lower:
            return ticker

    # 3. Bare ticker symbol
    if text.upper() in TRACKED_TICKERS:
        return text.upper()

    return None


def extract_entities(headline: str, nlp: spacy.language.Language) -> list[dict]:
    """
    Return a deduplicated list of entity dicts for one headline.

    Strategy (applied in order, duplicates removed by (text, label)):
    1. spaCy NER — keep ORG entities and resolve them to tickers where possible.
    2. Alias scan — scan for company-name aliases not caught by NER.
    3. Bare-ticker regex — catch uppercase ticker symbols in the text.
    """
    doc = nlp(headline)
    seen: set[tuple[str, str]] = set()   # (text, label) dedup key
    entities: list[dict] = []

    def add(text: str, label: str, ticker: str | None) -> None:
        key = (text, label)
        if key not in seen:
            seen.add(key)
            entities.append({"text": text, "label": label, "ticker": ticker})

    # 1. spaCy NER — ORG entities
    for ent in doc.ents:
        if ent.label_ == "ORG":
            add(ent.text, "ORG", resolve_ticker(ent.text))

    # 2. Alias scan for names spaCy may have missed
    for alias, ticker in TICKER_ALIASES.items():
        # Use word-boundary-aware search to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        if pattern.search(headline):
            # Prefer "ORG" if the alias is a company name, else use the ticker
            add(alias, "ORG", ticker)

    # 3. Bare ticker symbols  (e.g. headline literally contains "ZM" or "MSFT")
    for m in _TICKER_RE.finditer(headline):
        symbol = m.group(1)
        add(symbol, "TICKER", symbol)

    return entities


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 2 — entity_extractor.py")
    print("=" * 50)

    # Load spaCy model
    print("Loading spaCy en_core_web_sm …")
    nlp = spacy.load("en_core_web_sm")

    # Load input
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input not found: {INPUT_JSON}")

    with open(INPUT_JSON, encoding="utf-8") as f:
        articles: list[dict] = json.load(f)

    print(f"Loaded {len(articles)} articles from {INPUT_JSON}\n")

    # Extract entities
    total_entities = 0
    ticker_counts: dict[str, int] = {}

    for article in articles:
        headline = article.get("headline", "")
        entities = extract_entities(headline, nlp)
        article["entities"] = entities
        total_entities += len(entities)

        for ent in entities:
            t = ent["ticker"]
            if t:
                ticker_counts[t] = ticker_counts.get(t, 0) + 1

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"Entities extracted : {total_entities}")
    print(f"Articles processed : {len(articles)}")
    avg = total_entities / len(articles) if articles else 0
    print(f"Avg per article    : {avg:.2f}")
    print(f"\nOutput saved → {OUTPUT_JSON}\n")

    print("Ticker mention counts (resolved):")
    for ticker in sorted(ticker_counts, key=ticker_counts.get, reverse=True):
        bar = "#" * (ticker_counts[ticker] // 5 or 1)
        print(f"  {ticker:>5}: {ticker_counts[ticker]:>4}  {bar}")

    # Show a few example extractions
    print("\nSample extractions (first 5 articles with entities):")
    shown = 0
    for article in articles:
        if article["entities"] and shown < 5:
            print(f"\n  [{article['ticker']}] {article['headline'][:80]}")
            for ent in article["entities"]:
                t = f" → {ent['ticker']}" if ent["ticker"] else ""
                print(f"    • {ent['text']!r:<30} [{ent['label']}]{t}")
            shown += 1


if __name__ == "__main__":
    main()
