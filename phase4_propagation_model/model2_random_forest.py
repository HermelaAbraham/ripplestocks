"""
Phase 4 — Script 2: model2_random_forest.py
Random Forest classifier that predicts price direction (UP / DOWN)
for each ticker using four hand-crafted features.

Features (per ticker)
---------------------
  1. avg_sentiment   — mean FinBERT score from Phase 2
  2. edge_to_aal     — graph edge weight to AAL (Covid epicenter) from Phase 3
  3. volatility      — std of daily log-returns from Phase 1 stock CSVs
  4. article_count   — total news articles in Phase 2 corpus

Label
-----
  1 = price UP  (Jan 2 → Apr 30 2020 return > 0)
  0 = price DOWN

Validation
----------
  Leave-One-Out cross-validation (LOO-CV).
  With only 10 samples LOO gives the most unbiased estimate — every
  ticker gets to be the held-out test set exactly once.

Output: phase4_propagation_model/data/results_random_forest.json
"""

import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_HERE)
NEWS_JSON   = os.path.join(ROOT_DIR, "phase2_nlp_layer",       "data", "covid_news_enriched.json")
GRAPH_PATH  = os.path.join(ROOT_DIR, "phase3_graph_construction", "data", "market_graph.gpickle")
STOCKS_DIR  = os.path.join(ROOT_DIR, "phase1_data_pipeline",    "data", "stocks")
OUTPUT_PATH = os.path.join(_HERE, "data", "results_random_forest.json")

TICKERS  = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]
EPICENTER = "AAL"

# Ground-truth labels derived from actual Jan 2 → Apr 30 2020 price changes
GROUND_TRUTH_RETURN = {
    "ZM":   +1.132, "MSFT": +0.108, "AMZN": +0.250, "NFLX": +0.249,
    "MRNA": +1.411, "AAL":  -0.564, "UAL":  -0.652, "MGM":  -0.477,
    "XOM":  -0.321, "SPY":  -0.092,
}


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(articles: list[dict], G, stocks_dir: str) -> pd.DataFrame:
    """Return a DataFrame with one row per ticker and four feature columns."""

    # Sentiment aggregation
    sent_scores: dict[str, list[float]] = defaultdict(list)
    art_counts:  dict[str, int]         = defaultdict(int)
    for a in articles:
        t = a.get("ticker", "")
        if t in TICKERS:
            sent_scores[t].append(a["sentiment"]["score"])
            art_counts[t] += 1

    rows = []
    for ticker in TICKERS:
        # Feature 1: average FinBERT sentiment
        scores = sent_scores.get(ticker, [0.0])
        avg_sent = float(np.mean(scores))

        # Feature 2: edge weight to epicenter (AAL)
        edge_to_aal = float(G[ticker][EPICENTER]["weight"]) if ticker != EPICENTER else 1.0

        # Feature 3: daily return volatility
        df   = pd.read_csv(os.path.join(stocks_dir, f"{ticker}.csv"),
                           index_col="Date", parse_dates=True).sort_index()
        ret  = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        vol  = float(ret.std())

        # Feature 4: article count
        n_articles = art_counts.get(ticker, 0)

        rows.append({
            "ticker":        ticker,
            "avg_sentiment": avg_sent,
            "edge_to_aal":   edge_to_aal,
            "volatility":    vol,
            "article_count": n_articles,
            "label":         1 if GROUND_TRUTH_RETURN[ticker] > 0 else 0,
        })

    return pd.DataFrame(rows).set_index("ticker")


# ── LOO-CV training ───────────────────────────────────────────────────────────

def run_loo_cv(df: pd.DataFrame) -> list[dict]:
    """
    Leave-One-Out cross-validation.
    Returns a list of per-ticker prediction records.
    """
    feature_cols = ["avg_sentiment", "edge_to_aal", "volatility", "article_count"]
    X = df[feature_cols].values
    y = df["label"].values
    tickers = df.index.tolist()

    loo     = LeaveOneOut()
    results = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]
        test_ticker     = tickers[test_idx[0]]

        # Scale features
        scaler   = StandardScaler()
        X_train  = scaler.fit_transform(X_train)
        X_test   = scaler.transform(X_test)

        # Train RF — n_estimators=200, random_state fixed for reproducibility
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)

        prob        = rf.predict_proba(X_test)[0]   # [P(DOWN), P(UP)]
        pred_label  = int(rf.predict(X_test)[0])
        true_label  = int(y[test_idx[0]])
        confidence  = float(prob[pred_label])

        # Signed score: P(UP) - P(DOWN)  →  ∈ [-1, +1]
        signed_score = float(prob[1] - prob[0])

        results.append({
            "ticker":      test_ticker,
            "pred_label":  pred_label,
            "true_label":  true_label,
            "direction":   "UP" if pred_label == 1 else "DOWN",
            "confidence":  round(confidence, 4),
            "norm_score":  round(signed_score, 4),
            "correct":     pred_label == true_label,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 4 — Model 2: Random Forest (LOO-CV)")
    print("=" * 50)

    # Load data
    with open(NEWS_JSON, encoding="utf-8") as f:
        articles = json.load(f)
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # Build feature matrix
    df = build_features(articles, G, STOCKS_DIR)
    print("\nFeature matrix:")
    print(f"  {'Ticker':<7} {'AvgSent':>9} {'EdgeAAL':>9} {'Volatility':>11} {'Articles':>9} {'Label':>6}")
    print(f"  {'------':<7} {'-------':>9} {'-------':>9} {'----------':>11} {'--------':>9} {'-----':>6}")
    for ticker, row in df.iterrows():
        print(
            f"  {ticker:<7} {row['avg_sentiment']:>+9.4f} {row['edge_to_aal']:>9.4f} "
            f"{row['volatility']:>11.4f} {int(row['article_count']):>9} {int(row['label']):>6}"
        )

    # LOO-CV
    print("\nRunning Leave-One-Out CV …")
    results = run_loo_cv(df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {OUTPUT_PATH}\n")

    # Summary table
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"{'Ticker':<7} {'Predicted':>10} {'Actual':>8} {'Confidence':>11} {'Correct':>8}")
    print(f"{'------':<7} {'---------':>10} {'------':>8} {'----------':>11} {'-------':>8}")
    for r in results:
        actual_dir = "UP" if r["true_label"] == 1 else "DOWN"
        mark = "✓" if r["correct"] else "✗"
        print(
            f"{r['ticker']:<7} {r['direction']:>10} {actual_dir:>8} "
            f"{r['confidence']:>10.1%}  {mark}"
        )
    print(f"\nDirectional accuracy (LOO-CV): {accuracy:.0%}  ({sum(r['correct'] for r in results)}/{len(results)})")


if __name__ == "__main__":
    main()
