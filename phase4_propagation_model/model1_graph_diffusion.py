"""
Phase 4 — Script 1: model1_graph_diffusion.py
Graph Diffusion model: simulates a shock propagating outward from an
epicenter node across the weighted market graph.

Propagation rule (per hop)
--------------------------
  shock_received(v) += shock_sent(u) × edge_weight(u, v)

The shock attenuates naturally with each hop because edge_weight ≤ 1.
Three hops are run; a node accumulates shocks from ALL paths that reach it,
not just the shortest one, so the final score reflects total exposure.

Epicenter: AAL  (airlines = direct Covid victim; shock = -1.0)

Output: phase4_propagation_model/data/results_graph_diffusion.json
"""

import json
import os
import pickle
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(_HERE)
GRAPH_PATH   = os.path.join(ROOT_DIR, "phase3_graph_construction", "data", "market_graph.gpickle")
OUTPUT_PATH  = os.path.join(_HERE, "data", "results_graph_diffusion.json")

EPICENTER    = "AAL"
INITIAL_SHOCK = -1.0
N_HOPS       = 3
TICKERS      = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]


def propagate(G, epicenter: str, initial_shock: float, n_hops: int) -> dict[str, float]:
    """
    Multi-hop diffusion.  Each hop sends accumulated shock from every
    active node outward along weighted edges.

    Returns accumulated impact score per ticker (including the epicenter).
    """
    # impact[node] = total shock accumulated so far
    impact: dict[str, float] = defaultdict(float)
    impact[epicenter] = initial_shock

    # active = {node: shock_to_propagate_THIS_hop}
    active: dict[str, float] = {epicenter: initial_shock}

    for hop in range(1, n_hops + 1):
        next_active: dict[str, float] = defaultdict(float)
        for sender, shock in active.items():
            for neighbor, edge_data in G[sender].items():
                if neighbor == epicenter:
                    continue   # don't propagate back to epicenter
                received = shock * edge_data["weight"]
                next_active[neighbor] += received
                impact[neighbor]      += received
        active = dict(next_active)
        # Stop early if shock has fully attenuated
        if not active or max(abs(v) for v in active.values()) < 1e-6:
            break

    return dict(impact)


def main() -> None:
    print("Phase 4 — Model 1: Graph Diffusion")
    print("=" * 50)

    # Load graph
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Epicenter : {EPICENTER}  (initial shock = {INITIAL_SHOCK:+.1f})\n")

    # Run diffusion
    impact = propagate(G, EPICENTER, INITIAL_SHOCK, N_HOPS)

    # Ensure all tickers present (even unaffected ones score 0)
    for t in TICKERS:
        impact.setdefault(t, 0.0)

    # Rank by absolute impact (most affected first)
    ranked = sorted(impact.items(), key=lambda x: x[1])

    # Normalise to [-1, +1] for comparability with other models
    max_abs = max(abs(v) for v in impact.values()) or 1.0
    normalised = {t: round(v / max_abs, 6) for t, v in impact.items()}

    # Build output records
    results = []
    for ticker, raw_score in ranked:
        results.append({
            "ticker":     ticker,
            "raw_score":  round(raw_score, 6),
            "norm_score": normalised[ticker],
            "direction":  "UP" if raw_score > 0 else "DOWN",
        })

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {OUTPUT_PATH}\n")

    # Print ranked table
    print(f"{'Rank':<5} {'Ticker':<7} {'Raw Score':>10} {'Norm Score':>11} {'Direction'}")
    print(f"{'----':<5} {'------':<7} {'---------':>10} {'----------':>11} {'---------'}")
    for i, rec in enumerate(results, 1):
        print(
            f"{i:<5} {rec['ticker']:<7} {rec['raw_score']:>+10.4f} "
            f"{rec['norm_score']:>+11.4f} {rec['direction']}"
        )


if __name__ == "__main__":
    main()
