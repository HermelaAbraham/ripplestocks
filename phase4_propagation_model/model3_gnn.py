"""
Phase 4 — Script 3: model3_gnn.py
Graph Neural Network (GNN) using PyTorch Geometric.

Architecture: 2-layer GCN (Graph Convolutional Network)
  Input node features (3):
    1. avg_sentiment   — mean FinBERT score (Phase 2)
    2. news_volume     — log-normalised article count (Phase 2)
    3. volatility      — std of daily log-returns (Phase 1)

  Edge weights: combined weights from Phase 3 market graph

  Output: binary classification per node (UP=1 / DOWN=0)

Training strategy
-----------------
  With only 10 nodes, standard train/test split is not meaningful.
  We use Leave-One-Out: train 9 nodes, predict the held-out 1, repeat
  for all 10.  This mirrors the RF approach for fair comparison.

  Epochs: 300 per fold  |  LR: 0.01  |  Hidden dim: 16

Output: phase4_propagation_model/data/results_gnn.json
"""

import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_HERE)
NEWS_JSON   = os.path.join(ROOT_DIR, "phase2_nlp_layer",        "data", "covid_news_enriched.json")
GRAPH_PATH  = os.path.join(ROOT_DIR, "phase3_graph_construction","data", "market_graph.gpickle")
STOCKS_DIR  = os.path.join(ROOT_DIR, "phase1_data_pipeline",    "data", "stocks")
OUTPUT_PATH = os.path.join(_HERE, "data", "results_gnn.json")

TICKERS  = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]
T_INDEX  = {t: i for i, t in enumerate(TICKERS)}   # ticker → node index

GROUND_TRUTH_LABEL = {
    "ZM": 1, "MSFT": 1, "AMZN": 1, "NFLX": 1, "MRNA": 1,
    "AAL": 0, "UAL": 0, "MGM": 0, "XOM": 0, "SPY": 0,
}

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ── GCN model ─────────────────────────────────────────────────────────────────

class MarketGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels,  hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x   # raw logits, shape [N, 2]


# ── Data preparation ──────────────────────────────────────────────────────────

def build_node_features(articles: list[dict], stocks_dir: str) -> np.ndarray:
    """Return (10, 3) array: [avg_sentiment, log_news_volume, volatility]."""
    sent_scores: dict[str, list[float]] = defaultdict(list)
    art_counts:  dict[str, int]         = defaultdict(int)
    for a in articles:
        t = a.get("ticker", "")
        if t in TICKERS:
            sent_scores[t].append(a["sentiment"]["score"])
            art_counts[t] += 1

    rows = []
    for ticker in TICKERS:
        avg_sent   = float(np.mean(sent_scores.get(ticker, [0.0])))
        log_vol    = float(np.log1p(art_counts.get(ticker, 0)))
        df         = pd.read_csv(os.path.join(stocks_dir, f"{ticker}.csv"),
                                 index_col="Date", parse_dates=True).sort_index()
        ret        = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        volatility = float(ret.std())
        rows.append([avg_sent, log_vol, volatility])

    X = np.array(rows, dtype=np.float32)
    # Normalise each feature to zero mean / unit std
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X


def build_graph_tensors(G) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert NetworkX graph to edge_index and edge_weight tensors."""
    src_list, dst_list, wgt_list = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = T_INDEX[u], T_INDEX[v]
        w = float(d["weight"])
        # Undirected: add both directions
        src_list += [i, j]
        dst_list += [j, i]
        wgt_list += [w, w]

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(wgt_list, dtype=torch.float)
    return edge_index, edge_weight


# ── LOO-CV training ───────────────────────────────────────────────────────────

def train_loo(
    X: np.ndarray,
    labels: list[int],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    epochs: int = 300,
    lr: float   = 0.01,
    hidden: int = 16,
) -> list[dict]:
    """
    Leave-One-Out CV: for each fold, mask one node as test, train on rest.
    Returns per-ticker prediction records.
    """
    n = len(TICKERS)
    y = torch.tensor(labels, dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)

    results = []

    for test_idx in range(n):
        train_mask = torch.zeros(n, dtype=torch.bool)
        test_mask  = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            if i != test_idx:
                train_mask[i] = True
        test_mask[test_idx] = True

        model     = MarketGCN(in_channels=3, hidden=hidden, out_channels=2)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # Training loop
        model.train()
        for _ in range(epochs):
            optimiser.zero_grad()
            out  = model(x, edge_index, edge_weight)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimiser.step()

        # Prediction on held-out node
        model.eval()
        with torch.no_grad():
            out      = model(x, edge_index, edge_weight)
            logits   = out[test_mask]            # shape [1, 2]
            probs    = F.softmax(logits, dim=1)  # [1, 2]
            pred     = int(probs.argmax(dim=1).item())
            conf     = float(probs[0, pred].item())
            signed   = float(probs[0, 1].item() - probs[0, 0].item())  # P(UP)-P(DOWN)

        ticker     = TICKERS[test_idx]
        true_label = labels[test_idx]
        results.append({
            "ticker":     ticker,
            "pred_label": pred,
            "true_label": true_label,
            "direction":  "UP" if pred == 1 else "DOWN",
            "confidence": round(conf,   4),
            "norm_score": round(signed, 4),
            "correct":    pred == true_label,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 4 — Model 3: GNN (2-layer GCN, LOO-CV)")
    print("=" * 50)

    # Load data
    with open(NEWS_JSON, encoding="utf-8") as f:
        articles = json.load(f)
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # Build inputs
    X           = build_node_features(articles, STOCKS_DIR)
    edge_index, edge_weight = build_graph_tensors(G)
    labels      = [GROUND_TRUTH_LABEL[t] for t in TICKERS]

    print(f"\nNode feature matrix shape : {X.shape}  (10 nodes × 3 features)")
    print(f"Edge index shape           : {edge_index.shape}")
    print(f"Running LOO-CV (10 folds × 300 epochs each) …\n")

    results = train_loo(X, labels, edge_index, edge_weight)

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
