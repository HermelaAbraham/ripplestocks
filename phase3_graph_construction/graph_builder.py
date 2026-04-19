"""
Phase 3 — graph_builder.py
Builds a weighted undirected graph of the 10 RippleStocks tickers using
three complementary edge sources, then combines them by averaging.

Edge sources
------------
1. Domain knowledge  — hardcoded economic relationships (weight 0/1 → normalised)
2. Co-mention freq   — Pearson-normalised count of articles mentioning two tickers
3. Price correlation — |Pearson r| of daily log-returns across the 82-day window

Final edge weight = mean of the three source weights (sources without a direct
observation for a pair contribute 0 to the average, diluting rather than blocking).

Output
------
  phase3_graph_construction/data/market_graph.gpickle
  (NetworkX Graph, nodes = ticker strings, edges carry 'weight' + per-source weights)
"""

import json
import os
import pickle
import itertools
from collections import defaultdict

import networkx as nx
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(_HERE)
NEWS_JSON    = os.path.join(ROOT_DIR, "phase2_nlp_layer", "data", "covid_news_enriched.json")
STOCKS_DIR   = os.path.join(ROOT_DIR, "phase1_data_pipeline", "data", "stocks")
OUTPUT_DIR   = os.path.join(_HERE, "data")
OUTPUT_GRAPH = os.path.join(OUTPUT_DIR, "market_graph.gpickle")

TICKERS = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]

# All unique unordered pairs — the universe of possible edges
ALL_PAIRS = list(itertools.combinations(TICKERS, 2))


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — Domain Knowledge
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple is (ticker_a, ticker_b, relationship_description)
# Weight 1.0 = confirmed strong economic link; 0.6 = moderate/indirect link
DOMAIN_EDGES: list[tuple[str, str, str, float]] = [
    # Airline sector — direct competitors sharing cost drivers
    ("AAL",  "UAL",  "airline competitors",                    1.0),
    # Airlines depend heavily on jet fuel → oil price sensitivity
    ("AAL",  "XOM",  "fuel cost dependency (jet fuel / oil)",  1.0),
    ("UAL",  "XOM",  "fuel cost dependency (jet fuel / oil)",  1.0),
    # Remote-work tech — both benefited from WFH, compete in video/collab
    ("ZM",   "MSFT", "remote-work / video-collab competitors", 1.0),
    # Cloud hyperscalers — compete in enterprise cloud
    ("MSFT", "AMZN", "cloud platform competitors (Azure/AWS)", 1.0),
    # Streaming / consumer digital — overlapping subscribers
    ("AMZN", "NFLX", "streaming competitors (Prime/Netflix)",  0.8),
    # Biotech bellwether vs broad market
    ("MRNA", "SPY",  "biotech sentiment → broad market",       0.8),
    # Travel & hospitality — both crushed by lockdowns
    ("AAL",  "MGM",  "travel & hospitality sector peers",      0.8),
    ("UAL",  "MGM",  "travel & hospitality sector peers",      0.8),
    # Big-tech weight in S&P 500 index
    ("MSFT", "SPY",  "mega-cap tech drives index (MSFT/SPY)",  0.8),
    ("AMZN", "SPY",  "mega-cap tech drives index (AMZN/SPY)",  0.8),
    # Energy vs broad market
    ("XOM",  "SPY",  "energy sector weight in S&P 500",        0.6),
    # ZM surge reflected in tech-heavy index
    ("ZM",   "SPY",  "high-growth tech vs index",              0.6),
    # MRNA news often moved broader biotech/market
    ("MRNA", "AMZN", "pandemic beneficiaries (vaccine/e-comm)",0.6),
]


def build_domain_weights() -> dict[tuple[str, str], float]:
    """Return {canonical_pair: weight} for all domain-knowledge edges."""
    weights: dict[tuple[str, str], float] = {}
    for a, b, _, w in DOMAIN_EDGES:
        pair = tuple(sorted([a, b]))
        weights[pair] = w
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — News Co-mention Frequency
# ─────────────────────────────────────────────────────────────────────────────

def build_comention_weights(articles: list[dict]) -> dict[tuple[str, str], float]:
    """
    For each article collect all ticker signals (article.ticker + entity tickers),
    then count co-occurrences of every pair.  Normalise to [0, 1].
    """
    raw_counts: dict[tuple[str, str], int] = defaultdict(int)

    for article in articles:
        # Gather all tickers referenced in this article
        present: set[str] = set()

        # The article's own ticker
        t = article.get("ticker", "")
        if t in TICKERS:
            present.add(t)

        # Any entity-resolved tickers
        for ent in article.get("entities", []):
            if ent.get("ticker") in TICKERS:
                present.add(ent["ticker"])

        # Count every pair that co-appears
        for pair in itertools.combinations(sorted(present), 2):
            raw_counts[pair] += 1

    if not raw_counts:
        return {}

    max_count = max(raw_counts.values())
    return {pair: count / max_count for pair, count in raw_counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — Price Correlation
# ─────────────────────────────────────────────────────────────────────────────

def load_daily_returns() -> pd.DataFrame:
    """Load all stock CSVs and compute daily log-returns aligned on trading dates."""
    series: dict[str, pd.Series] = {}
    for ticker in TICKERS:
        path = os.path.join(STOCKS_DIR, f"{ticker}.csv")
        df   = pd.read_csv(path, index_col="Date", parse_dates=True)
        close = df["Close"].sort_index()
        # Log-returns: ln(P_t / P_{t-1})
        log_ret = np.log(close / close.shift(1)).dropna()
        series[ticker] = log_ret

    returns_df = pd.DataFrame(series).dropna()   # inner join on common dates
    return returns_df


def build_correlation_weights(returns_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Compute |Pearson r| for every ticker pair.  Values already in [0, 1]."""
    corr_matrix = returns_df.corr(method="pearson")
    weights: dict[tuple[str, str], float] = {}
    for a, b in ALL_PAIRS:
        r = corr_matrix.loc[a, b]
        weights[tuple(sorted([a, b]))] = round(abs(r), 6)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def combine_weights(
    domain:    dict[tuple[str, str], float],
    comentions: dict[tuple[str, str], float],
    correlations: dict[tuple[str, str], float],
) -> dict[tuple[str, str], dict]:
    """
    For every pair, average the three source weights.
    A missing source contributes 0 (not excluded), so pairs with only
    one signal still get a proportionally lower combined weight.
    """
    combined: dict[tuple[str, str], dict] = {}
    for pair in ALL_PAIRS:
        key = tuple(sorted(pair))
        w_domain  = domain.get(key, 0.0)
        w_comention = comentions.get(key, 0.0)
        w_corr    = correlations.get(key, 0.0)
        w_combined = round((w_domain + w_comention + w_corr) / 3.0, 6)
        combined[key] = {
            "weight":      w_combined,
            "w_domain":    round(w_domain,    6),
            "w_comention": round(w_comention, 6),
            "w_corr":      round(w_corr,      6),
        }
    return combined


def build_graph(edge_data: dict[tuple[str, str], dict]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(TICKERS)
    for (a, b), attrs in edge_data.items():
        G.add_edge(a, b, **attrs)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 3 — graph_builder.py")
    print("=" * 55)

    # ── Source 1: Domain knowledge ────────────────────────────────
    print("\n[1/3] Domain knowledge edges …")
    domain = build_domain_weights()
    print(f"      {len(domain)} pre-seeded relationships loaded")

    # ── Source 2: Co-mention frequency ───────────────────────────
    print("\n[2/3] News co-mention frequency …")
    if not os.path.exists(NEWS_JSON):
        raise FileNotFoundError(f"Missing: {NEWS_JSON}\nRun Phase 2 scripts first.")
    with open(NEWS_JSON, encoding="utf-8") as f:
        articles = json.load(f)
    comentions = build_comention_weights(articles)
    print(f"      {len(articles)} articles scanned → {len(comentions)} co-mention pairs found")

    # ── Source 3: Price correlation ───────────────────────────────
    print("\n[3/3] Price correlation (daily log-returns) …")
    returns_df = load_daily_returns()
    print(f"      {len(returns_df)} trading days × {len(returns_df.columns)} tickers")
    correlations = build_correlation_weights(returns_df)
    print(f"      {len(correlations)} correlation pairs computed")

    # ── Combine & build graph ─────────────────────────────────────
    print("\nCombining edge sources (averaging weights) …")
    edge_data = combine_weights(domain, comentions, correlations)
    G = build_graph(edge_data)

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_GRAPH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Graph saved → {OUTPUT_GRAPH}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")
    print()

    # Top 10 edges by combined weight
    edges_sorted = sorted(
        G.edges(data=True),
        key=lambda e: e[2]["weight"],
        reverse=True,
    )

    print("Top 10 strongest edges (combined weight):")
    print(f"  {'Edge':<12}  {'Combined':>8}  {'Domain':>8}  {'CoMention':>10}  {'PriceCorr':>10}")
    print(f"  {'----':<12}  {'--------':>8}  {'------':>8}  {'---------':>10}  {'---------':>10}")
    for a, b, d in edges_sorted[:10]:
        edge_label = f"{a}↔{b}"
        print(
            f"  {edge_label:<12}  {d['weight']:>8.4f}  "
            f"{d['w_domain']:>8.4f}  {d['w_comention']:>10.4f}  {d['w_corr']:>10.4f}"
        )

    # Domain-knowledge relationships with their resolved combined weights
    print("\nDomain-knowledge relationships (with resolved combined weights):")
    for a, b, desc, dw in DOMAIN_EDGES:
        pair = tuple(sorted([a, b]))
        combined = edge_data[pair]["weight"]
        print(f"  {a}↔{b:<6}  combined={combined:.4f}  ({desc})")

    # Correlation heatmap snapshot (top 5 correlated pairs)
    print("\nTop 5 price-correlated pairs (|Pearson r|):")
    corr_sorted = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for (a, b), r in corr_sorted[:5]:
        print(f"  {a}↔{b:<6}  |r| = {r:.4f}")


if __name__ == "__main__":
    main()
