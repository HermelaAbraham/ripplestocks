"""
Phase 2 — Script 2: sentiment_scorer.py
Scores each news headline with FinBERT (ProsusAI/finbert) and writes a
signed sentiment value into the `sentiment` field of every article.

Scoring convention
------------------
  score = P(positive) - P(negative)   ∈ [-1.0, +1.0]

  • Positive dominant → score ∈ (0, +1]
  • Negative dominant → score ∈ [-1,  0)
  • Neutral dominant  → score ≈  0   (P(pos) ≈ P(neg))

The raw FinBERT label and per-class probabilities are stored alongside
the scalar score so later phases can threshold differently if needed.

    "sentiment": {
        "score":      float,          # signed scalar
        "label":      "positive" | "negative" | "neutral",
        "positive":   float,          # raw P(positive)
        "negative":   float,          # raw P(negative)
        "neutral":    float           # raw P(neutral)
    }

Input:  phase2_nlp_layer/data/covid_news_entities.json
Output: phase2_nlp_layer/data/covid_news_enriched.json
"""

import json
import os
import warnings

# Suppress the benign position_ids UNEXPECTED key warning from the old checkpoint
warnings.filterwarnings("ignore", message=".*position_ids.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # silence tokenizer fork warning

from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error()   # suppress INFO-level HuggingFace logs

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON  = os.path.join(_HERE, "data", "covid_news_entities.json")
OUTPUT_JSON = os.path.join(_HERE, "data", "covid_news_enriched.json")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 32      # process headlines in batches for speed
MAX_TOKENS  = 512     # FinBERT hard limit; headlines are short, but guard anyway


# ── Scoring helpers ───────────────────────────────────────────────────────────

def build_pipeline() -> pipeline:
    """Load FinBERT as a text-classification pipeline (returns all 3 class scores)."""
    print(f"Loading FinBERT model: {MODEL_NAME} …")
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,                 # return scores for all three labels
        truncation=True,
        max_length=MAX_TOKENS,
    )


def scores_to_sentiment(raw: list[dict]) -> dict:
    """
    Convert FinBERT's list of {label, score} dicts into our sentiment object.

    raw example:
      [{"label": "positive", "score": 0.95},
       {"label": "neutral",  "score": 0.03},
       {"label": "negative", "score": 0.02}]
    """
    probs = {item["label"]: item["score"] for item in raw}
    pos   = probs.get("positive", 0.0)
    neg   = probs.get("negative", 0.0)
    neu   = probs.get("neutral",  0.0)

    # Signed scalar: positive win → positive score, negative win → negative score
    signed = round(pos - neg, 6)

    # Primary label = argmax across the three classes
    label = max(probs, key=probs.get)

    return {
        "score":    signed,
        "label":    label,
        "positive": round(pos, 6),
        "negative": round(neg, 6),
        "neutral":  round(neu, 6),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 2 — sentiment_scorer.py")
    print("=" * 50)

    # Load input
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input not found: {INPUT_JSON}\nRun entity_extractor.py first.")

    with open(INPUT_JSON, encoding="utf-8") as f:
        articles: list[dict] = json.load(f)

    print(f"Loaded {len(articles)} articles\n")

    # Build pipeline
    finbert = build_pipeline()
    print(f"Model ready — scoring {len(articles)} headlines in batches of {BATCH_SIZE} …\n")

    # Extract headlines (guard against empty strings)
    headlines = [a.get("headline") or "[no headline]" for a in articles]

    # Batch inference
    all_raw: list[list[dict]] = []
    total_batches = (len(headlines) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(headlines), BATCH_SIZE):
        batch      = headlines[i : i + BATCH_SIZE]
        batch_num  = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num:>3}/{total_batches}  ({i+1}–{min(i+BATCH_SIZE, len(headlines))})", end="\r")
        results    = finbert(batch)
        all_raw.extend(results)

    print()  # newline after carriage-return progress

    # Attach sentiment to each article
    for article, raw in zip(articles, all_raw):
        article["sentiment"] = scores_to_sentiment(raw)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Saved → {OUTPUT_JSON}\n")

    # ── Summary ───────────────────────────────────────────────────────────────

    label_counts = {"positive": 0, "negative": 0, "neutral": 0}
    ticker_scores: dict[str, list[float]] = {}

    for article in articles:
        s = article["sentiment"]
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

        ticker = article.get("ticker", "UNKNOWN")
        ticker_scores.setdefault(ticker, []).append(s["score"])

    total = len(articles)
    pos_c = label_counts["positive"]
    neg_c = label_counts["negative"]
    neu_c = label_counts["neutral"]

    print("─" * 50)
    print(f"Articles scored     : {total}")
    print(f"  Positive          : {pos_c:>4}  ({pos_c/total*100:5.1f}%)")
    print(f"  Negative          : {neg_c:>4}  ({neg_c/total*100:5.1f}%)")
    print(f"  Neutral           : {neu_c:>4}  ({neu_c/total*100:5.1f}%)")
    print()

    all_scores = [a["sentiment"]["score"] for a in articles]
    print(f"Overall avg score   : {sum(all_scores)/len(all_scores):+.4f}")
    print(f"Min score           : {min(all_scores):+.4f}")
    print(f"Max score           : {max(all_scores):+.4f}")
    print()

    print("Average sentiment score per ticker:")
    print(f"  {'Ticker':<6}  {'Avg Score':>10}  {'Count':>6}  {'Distribution'}")
    print(f"  {'------':<6}  {'---------':>10}  {'-----':>6}  {'------------'}")

    TICKERS = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]
    for ticker in TICKERS:
        scores = ticker_scores.get(ticker, [])
        if not scores:
            continue
        avg   = sum(scores) / len(scores)
        # Simple ASCII bar: scale [-1,+1] → bar of + or - signs
        bar_len = min(abs(int(avg * 20)), 20)
        bar_chr = "+" if avg >= 0 else "-"
        bar     = bar_chr * bar_len
        print(f"  {ticker:<6}  {avg:>+10.4f}  {len(scores):>6}  {bar}")

    print()
    print("Sample scored headlines:")
    examples = {"positive": None, "negative": None, "neutral": None}
    for article in articles:
        lbl = article["sentiment"]["label"]
        if examples[lbl] is None:
            examples[lbl] = article
        if all(v is not None for v in examples.values()):
            break

    for lbl, article in examples.items():
        if article:
            s = article["sentiment"]["score"]
            h = article["headline"][:72]
            print(f"  [{lbl.upper():<8} {s:+.4f}]  {h}")


if __name__ == "__main__":
    main()
