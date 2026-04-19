"""
Phase 4 — Script 4: model_comparison.py
Runs all three models sequentially and compares their predictions against
the Covid-19 ground truth in a single consolidated table.

Ground truth
------------
  MRNA +141%  ZM +113%  AMZN +25%  NFLX +25%  MSFT +11%
  UAL  -65%   AAL -56%  MGM  -48%  XOM  -32%  SPY   -9%

Metrics
-------
  • Directional accuracy — did the model predict UP/DOWN correctly?
  • Per-ticker comparison across all three models
  • Side-by-side accuracy scores

Output: phase4_propagation_model/data/comparison_results.json
"""

import json
import os
import sys
import importlib.util
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_HERE)
DATA_DIR    = os.path.join(_HERE, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "comparison_results.json")

# Result files written by each model script
RESULT_FILES = {
    "Graph Diffusion": os.path.join(DATA_DIR, "results_graph_diffusion.json"),
    "Random Forest":   os.path.join(DATA_DIR, "results_random_forest.json"),
    "GNN":             os.path.join(DATA_DIR, "results_gnn.json"),
}

TICKERS = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]

# Ground truth: actual Jan 2 → Apr 30 2020 percentage returns
GROUND_TRUTH = {
    "ZM":   +1.132, "MSFT": +0.108, "AMZN": +0.250, "NFLX": +0.249,
    "MRNA": +1.411, "AAL":  -0.564, "UAL":  -0.652, "MGM":  -0.477,
    "XOM":  -0.321, "SPY":  -0.092,
}
GT_DIRECTION = {t: ("UP" if r > 0 else "DOWN") for t, r in GROUND_TRUTH.items()}
GT_LABEL     = {t: (1 if r > 0 else 0)         for t, r in GROUND_TRUTH.items()}


# ── Model runner ──────────────────────────────────────────────────────────────

def run_model_script(script_name: str) -> None:
    """Dynamically import and run a model script's main() function."""
    path = os.path.join(_HERE, script_name)
    spec = importlib.util.spec_from_file_location(script_name[:-3], path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def load_results(path: str) -> dict[str, dict]:
    """Load a model's JSON result file into a {ticker: record} dict."""
    with open(path) as f:
        records = json.load(f)
    return {r["ticker"]: r for r in records}


# ── Graph Diffusion direction inference ───────────────────────────────────────

def gd_direction_from_score(score: float, ticker: str) -> str:
    """
    Graph Diffusion produces a continuous shock score, not a binary label.
    Convert to UP/DOWN:
      • Epicenter (AAL) and any ticker receiving negative shock → DOWN
      • Tickers with positive or zero accumulated shock → UP
    Special case: AAL is the epicenter → always DOWN by definition.
    """
    if ticker == "AAL":
        return "DOWN"
    return "DOWN" if score < 0 else "UP"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 4 — Model Comparison")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 1: Run all three models ─────────────────────────────────────────
    for label, fname in [
        ("Graph Diffusion", "model1_graph_diffusion.py"),
        ("Random Forest",   "model2_random_forest.py"),
        ("GNN",             "model3_gnn.py"),
    ]:
        sep = "─" * 50
        print(f"\n{sep}\nRunning {label} …\n{sep}")
        run_model_script(fname)

    # ── Step 2: Load results ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("CONSOLIDATED COMPARISON")
    print("═" * 70)

    gd_res = load_results(RESULT_FILES["Graph Diffusion"])
    rf_res = load_results(RESULT_FILES["Random Forest"])
    gn_res = load_results(RESULT_FILES["GNN"])

    # ── Step 3: Build comparison table ───────────────────────────────────────
    comparison = []
    for ticker in TICKERS:
        gt_dir    = GT_DIRECTION[ticker]
        gt_ret    = GROUND_TRUTH[ticker]

        # Graph Diffusion
        gd_score  = gd_res[ticker]["raw_score"]
        gd_dir    = gd_direction_from_score(gd_score, ticker)
        gd_ok     = gd_dir == gt_dir

        # Random Forest
        rf_dir    = rf_res[ticker]["direction"]
        rf_conf   = rf_res[ticker]["confidence"]
        rf_ok     = rf_dir == gt_dir

        # GNN
        gn_dir    = gn_res[ticker]["direction"]
        gn_conf   = gn_res[ticker]["confidence"]
        gn_ok     = gn_dir == gt_dir

        comparison.append({
            "ticker":          ticker,
            "ground_truth":    {"direction": gt_dir, "return_pct": round(gt_ret * 100, 1)},
            "graph_diffusion": {"direction": gd_dir, "score": round(gd_score, 4), "correct": gd_ok},
            "random_forest":   {"direction": rf_dir, "confidence": rf_conf,       "correct": rf_ok},
            "gnn":             {"direction": gn_dir, "confidence": gn_conf,       "correct": gn_ok},
        })

    # ── Step 4: Print side-by-side table ─────────────────────────────────────
    # Header
    print(f"\n{'Ticker':<6}  {'Actual':>8}  {'Return':>8}  "
          f"{'GD Pred':>8}  {'GD':>3}  "
          f"{'RF Pred':>8}  {'RF':>3}  "
          f"{'GNN Pred':>9}  {'GNN':>4}")
    print(f"{'------':<6}  {'------':>8}  {'------':>8}  "
          f"{'-------':>8}  {'--':>3}  "
          f"{'-------':>8}  {'--':>3}  "
          f"{'--------':>9}  {'---':>4}")

    for row in comparison:
        t   = row["ticker"]
        gt  = row["ground_truth"]
        gd  = row["graph_diffusion"]
        rf  = row["random_forest"]
        gn  = row["gnn"]
        print(
            f"{t:<6}  {gt['direction']:>8}  {gt['return_pct']:>+7.1f}%  "
            f"{gd['direction']:>8}  {'✓' if gd['correct'] else '✗':>3}  "
            f"{rf['direction']:>8}  {'✓' if rf['correct'] else '✗':>3}  "
            f"{gn['direction']:>9}  {'✓' if gn['correct'] else '✗':>4}"
        )

    # ── Step 5: Accuracy scores ───────────────────────────────────────────────
    n = len(TICKERS)
    gd_acc = sum(row["graph_diffusion"]["correct"] for row in comparison) / n
    rf_acc = sum(row["random_forest"]["correct"]   for row in comparison) / n
    gn_acc = sum(row["gnn"]["correct"]             for row in comparison) / n

    gd_correct = sum(row["graph_diffusion"]["correct"] for row in comparison)
    rf_correct = sum(row["random_forest"]["correct"]   for row in comparison)
    gn_correct = sum(row["gnn"]["correct"]             for row in comparison)

    print()
    print("─" * 70)
    print(f"{'Model':<22}  {'Correct':>8}  {'Accuracy':>10}")
    print(f"{'-----':<22}  {'-------':>8}  {'--------':>10}")
    print(f"{'Graph Diffusion':<22}  {gd_correct:>5}/{n:<2}  {gd_acc:>9.0%}")
    print(f"{'Random Forest (LOO)':<22}  {rf_correct:>5}/{n:<2}  {rf_acc:>9.0%}")
    print(f"{'GNN (LOO)':<22}  {gn_correct:>5}/{n:<2}  {gn_acc:>9.0%}")
    print("─" * 70)

    best_model = max(
        [("Graph Diffusion", gd_acc), ("Random Forest", rf_acc), ("GNN", gn_acc)],
        key=lambda x: x[1]
    )
    print(f"\nBest model: {best_model[0]}  ({best_model[1]:.0%} directional accuracy)")

    # Notable misses analysis
    misses = {
        "Graph Diffusion": [r["ticker"] for r in comparison if not r["graph_diffusion"]["correct"]],
        "Random Forest":   [r["ticker"] for r in comparison if not r["random_forest"]["correct"]],
        "GNN":             [r["ticker"] for r in comparison if not r["gnn"]["correct"]],
    }
    print("\nMispredictions per model:")
    for model, missed in misses.items():
        if missed:
            miss_str = ", ".join(missed)
            print(f"  {model:<22}: {miss_str}")
        else:
            print(f"  {model:<22}: none — perfect accuracy!")

    # ── Step 6: Save consolidated results ────────────────────────────────────
    output = {
        "accuracy": {
            "graph_diffusion": round(gd_acc, 4),
            "random_forest":   round(rf_acc, 4),
            "gnn":             round(gn_acc, 4),
        },
        "best_model":  best_model[0],
        "per_ticker":  comparison,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nConsolidated results saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
