"""
Phase 6 — dashboard.py
RippleStocks Streamlit dashboard: event input → ripple network → impact leaderboard.

Run with:
    streamlit run phase6_dashboard/dashboard.py
"""

import json
import os
import pickle
import tempfile
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from pyvis.network import Network

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(_HERE)

GRAPH_PATH   = os.path.join(ROOT_DIR, "phase3_graph_construction", "data", "market_graph.gpickle")
RF_PATH      = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "results_random_forest.json")
GD_PATH      = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "results_graph_diffusion.json")
GNN_PATH     = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "results_gnn.json")
COMPARE_PATH = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "comparison_results.json")

TICKERS = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]

GROUND_TRUTH_RETURNS = {
    "ZM":   +113.2, "MSFT":  +10.8, "AMZN": +25.0, "NFLX": +24.9,
    "MRNA": +141.1, "AAL":   -56.4, "UAL":  -65.2, "MGM":  -47.7,
    "XOM":  -32.1,  "SPY":    -9.2,
}

SECTOR_LABELS = {
    "ZM":   "Video Comms",  "MSFT": "Cloud/Tech",  "AMZN": "E-Commerce",
    "NFLX": "Streaming",    "MRNA": "Biotech",     "AAL":  "Airlines",
    "UAL":  "Airlines",     "MGM":  "Hospitality", "XOM":  "Energy",
    "SPY":  "Broad Market",
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RippleStocks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #666; font-size: 1rem; margin-top: 0; }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #1f77b4;
        border-left: 4px solid #1f77b4; padding-left: 10px;
        margin: 1.5rem 0 0.8rem 0;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
        border: 1px solid #e0e0e0; text-align: center;
    }
    .anomaly-banner {
        background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
        padding: 10px 14px; margin: 8px 0; font-size: 0.9rem;
    }
    .correct-tag   { color: #2ca02c; font-weight: 600; }
    .incorrect-tag { color: #d62728; font-weight: 600; }
    .phase-complete { color: #2ca02c; }
    .phase-pending  { color: #ff7f0e; }
    div[data-testid="stDataFrame"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource
def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_rf_results():
    with open(RF_PATH) as f:
        records = json.load(f)
    return {r["ticker"]: r for r in records}

@st.cache_data
def load_comparison():
    with open(COMPARE_PATH) as f:
        return json.load(f)

@st.cache_data
def load_model_results():
    gd  = {r["ticker"]: r for r in json.load(open(GD_PATH))}
    rf  = {r["ticker"]: r for r in json.load(open(RF_PATH))}
    gnn = {r["ticker"]: r for r in json.load(open(GNN_PATH))}
    return gd, rf, gnn

# ── Node colour logic ─────────────────────────────────────────────────────────

def ticker_colour(ticker: str, rf_results: dict) -> str:
    """
    Green  = RF predicts UP   with confidence ≥ 65%
    Red    = RF predicts DOWN with confidence ≥ 65%
    Yellow = low confidence (< 65%) or neutral
    """
    r = rf_results.get(ticker)
    if r is None:
        return "#aaaaaa"
    conf = r["confidence"]
    if conf < 0.65:
        return "#f5c518"   # yellow — uncertain
    return "#2ca02c" if r["direction"] == "UP" else "#d62728"

def ticker_size(ticker: str, rf_results: dict) -> int:
    """Scale node size 20–55 by RF confidence magnitude."""
    r = rf_results.get(ticker)
    if r is None:
        return 25
    return int(20 + r["confidence"] * 35)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 RippleStocks")
    st.markdown("*Market shock propagation system*")
    st.divider()

    st.markdown("**Test Case**")
    st.markdown("🦠 Covid-19 pandemic  \n📅 Jan – Apr 2020  \n🎯 10 benchmark tickers")

    st.divider()
    st.markdown("**Phase Completion**")
    phases = [
        ("✅", "Phase 1", "Data Pipeline"),
        ("✅", "Phase 2", "NLP / FinBERT"),
        ("✅", "Phase 3", "Market Graph"),
        ("✅", "Phase 4", "Propagation Models"),
        ("✅", "Phase 5", "Validation"),
        ("✅", "Phase 6", "Dashboard"),
    ]
    for icon, phase, label in phases:
        st.markdown(f"{icon} **{phase}** — {label}")

    st.divider()
    st.markdown("**Model Accuracy**")
    comp = load_comparison()
    acc  = comp["accuracy"]
    st.markdown(f"🥇 Random Forest: **{acc['random_forest']:.0%}**")
    st.markdown(f"🥈 GNN:           **{acc['gnn']:.0%}**")
    st.markdown(f"🥉 Graph Diffusion: **{acc['graph_diffusion']:.0%}**")

    st.divider()
    st.caption("IEEE Conference Paper  \nRippleStocks v1.0")

# ── Main header ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">RippleStocks</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Macro-economic shock propagation through financial networks '
    '— Covid-19 test case</p>',
    unsafe_allow_html=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Event Input Panel
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">① Event Input Panel</p>', unsafe_allow_html=True)

col_input, col_select, col_btn = st.columns([3, 1.5, 1])

with col_input:
    event_text = st.text_input(
        "Macro event description",
        value="WHO declares Covid-19 a global pandemic — March 11, 2020",
        placeholder="Describe the macro event (e.g. Fed raises rates by 75bp)…",
        label_visibility="collapsed",
    )

with col_select:
    epicenter = st.selectbox(
        "Epicenter ticker",
        options=TICKERS,
        index=TICKERS.index("AAL"),
        format_func=lambda t: f"{t} — {SECTOR_LABELS[t]}",
        label_visibility="collapsed",
    )

with col_btn:
    run_analysis = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# Show event confirmation banner once triggered
if event_text:
    st.info(
        f"**Event:** {event_text}  \n"
        f"**Epicenter:** {epicenter} ({SECTOR_LABELS[epicenter]})  \n"
        f"**Model:** Random Forest (best — 80% accuracy, F1 0.80)",
        icon="🎯",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Ripple Network Visualization
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">② Ripple Network Visualization</p>', unsafe_allow_html=True)

G          = load_graph()
rf_results = load_rf_results()

# ── pyvis interactive graph ───────────────────────────────────────────────────

def build_pyvis_graph(G, rf_results: dict, epicenter: str) -> str:
    net = Network(
        height="520px", width="100%",
        bgcolor="#0e1117", font_color="white",
        notebook=False,
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=180)

    for ticker in G.nodes():
        colour = "#FFD700" if ticker == epicenter else ticker_colour(ticker, rf_results)
        size   = 55       if ticker == epicenter else ticker_size(ticker, rf_results)
        r      = rf_results.get(ticker, {})
        direction = r.get("direction", "?")
        confidence = r.get("confidence", 0)
        label  = f"{ticker}\n{SECTOR_LABELS[ticker]}"
        title  = (
            f"<b>{ticker}</b> — {SECTOR_LABELS[ticker]}<br>"
            f"RF prediction: {direction} ({confidence:.0%} confidence)<br>"
            f"Actual return: {GROUND_TRUTH_RETURNS[ticker]:+.1f}%"
        )
        border = "#FFD700" if ticker == epicenter else "#ffffff"
        net.add_node(
            ticker, label=label, title=title,
            color={"background": colour, "border": border,
                   "highlight": {"background": colour, "border": "#ffffff"}},
            size=size, font={"size": 13, "face": "Arial", "color": "white"},
            borderWidth=3 if ticker == epicenter else 1,
        )

    for u, v, d in G.edges(data=True):
        w = float(d["weight"])
        if w < 0.05:
            continue   # skip near-zero edges for clarity
        width  = max(1.0, w * 7)
        alpha  = int(80 + w * 175)
        colour = f"rgba(180,180,180,{alpha/255:.2f})"
        net.add_edge(
            u, v, width=width,
            color=colour,
            title=f"{u}↔{v}: weight={w:.3f}",
            smooth={"type": "curvedCW", "roundness": 0.1},
        )

    # Write to temp file and return HTML string
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    net.save_graph(tmp.name)
    tmp.close()
    return open(tmp.name).read()

graph_html = build_pyvis_graph(G, rf_results, epicenter)
st.components.v1.html(graph_html, height=540, scrolling=False)

# Legend
leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
with leg_col1:
    st.markdown("🟢 **Green** — Predicted UP (high conf)")
with leg_col2:
    st.markdown("🔴 **Red** — Predicted DOWN (high conf)")
with leg_col3:
    st.markdown("🟡 **Yellow** — Low confidence / uncertain")
with leg_col4:
    st.markdown("🟠 **Gold border** — Event epicenter")

st.caption(
    "Node size = RF confidence magnitude  •  "
    "Edge thickness = combined graph weight  •  "
    "Hover nodes/edges for details"
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Impact Leaderboard
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="section-header">③ Impact Leaderboard</p>', unsafe_allow_html=True)

comp            = load_comparison()
gd_res, rf_res, gnn_res = load_model_results()

# ── Build leaderboard dataframe ───────────────────────────────────────────────

rows = []
for entry in comp["per_ticker"]:
    t        = entry["ticker"]
    gt       = entry["ground_truth"]
    rf       = entry["random_forest"]
    gd       = entry["graph_diffusion"]
    gnn_e    = entry["gnn"]
    rf_full  = rf_res.get(t, {})
    conf     = rf_full.get("confidence", 0)

    rows.append({
        "Ticker":      t,
        "Sector":      SECTOR_LABELS[t],
        "RF Direction": rf["direction"],
        "Confidence":  conf,
        "Actual Return": gt["return_pct"],
        "Actual Direction": gt["direction"],
        "RF Correct":  rf["correct"],
        "GD Correct":  gd["correct"],
        "GNN Correct": gnn_e["correct"],
        "ZM Anomaly":  t == "ZM",
    })

df = pd.DataFrame(rows)
df_sorted = df.sort_values("Actual Return", ascending=False).reset_index(drop=True)

# ── Styled leaderboard ────────────────────────────────────────────────────────

def style_leaderboard(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row):
        if row["ZM Anomaly"]:
            return ["background-color: #fff8e1; font-weight: bold"] * len(row)
        elif row["Actual Direction"] == "UP":
            return ["background-color: #f0fff0"] * len(row)
        else:
            return ["background-color: #fff0f0"] * len(row)

    display = df[[
        "Ticker", "Sector", "RF Direction", "Confidence",
        "Actual Return", "Actual Direction", "RF Correct", "GD Correct", "GNN Correct"
    ]].copy()

    display["Confidence"]    = display["Confidence"].map(lambda x: f"{x:.0%}")
    display["Actual Return"] = display["Actual Return"].map(lambda x: f"{x:+.1f}%")
    display["RF Correct"]    = display["RF Correct"].map(lambda x: "✓" if x else "✗")
    display["GD Correct"]    = display["GD Correct"].map(lambda x: "✓" if x else "✗")
    display["GNN Correct"]   = display["GNN Correct"].map(lambda x: "✓" if x else "✗")

    return display.style.apply(row_style, axis=1)

st.dataframe(
    style_leaderboard(df_sorted),
    use_container_width=True,
    hide_index=True,
    height=420,
)

# ── ZM anomaly callout ────────────────────────────────────────────────────────

st.markdown("""
<div class="anomaly-banner">
⚠️ <b>ZM Anomaly</b> — All three models misclassified Zoom Video Communications.<br>
<b>Sentiment signal:</b> FinBERT score <b>−0.75</b> (negative) — Zoom-bombing scandals,
encryption disputes, institutional bans dominated headlines.<br>
<b>Actual outcome:</b> <b>+113.2%</b> price surge driven by unprecedented remote-work adoption.<br>
<b>Key insight:</b> Headline tone completely diverged from underlying economic fundamentals.
Sentiment-only models cannot detect demand-driven outliers — this is the core motivation
for graph-based propagation that encodes structural sector relationships.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Model comparison metrics ──────────────────────────────────────────────────

st.markdown("**Model Comparison — Covid-19 Test Case**")

# Detailed per-model metrics from validation report
metrics_data = {
    "Model":     ["Graph Diffusion", "Random Forest", "GNN"],
    "Accuracy":  ["50%", "80% 🥇", "70%"],
    "Precision": ["0%",  "80% 🥇", "75%"],
    "Recall":    ["0%",  "80% 🥇", "60%"],
    "F1 Score":  ["0.00", "0.80 🥇", "0.67"],
    "TP": [0, 4, 3],
    "FP": [0, 1, 1],
    "TN": [5, 4, 4],
    "FN": [5, 1, 2],
    "Mispredictions": ["ZM MSFT AMZN NFLX MRNA", "ZM SPY", "ZM MRNA SPY"],
}

metrics_df = pd.DataFrame(metrics_data)

def style_metrics(df):
    def highlight(row):
        if row["Model"] == "Random Forest":
            return ["background-color: #e8f5e9; font-weight: bold"] * len(row)
        elif row["Model"] == "Graph Diffusion":
            return ["background-color: #fafafa; color: #888"] * len(row)
        return [""] * len(row)
    return df.style.apply(highlight, axis=1)

st.dataframe(
    style_metrics(metrics_df),
    use_container_width=True,
    hide_index=True,
)

# ── Four metric summary cards ─────────────────────────────────────────────────

st.markdown("**Best model performance (Random Forest)**")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Accuracy",  "80%", delta="vs 50% baseline")
with m2:
    st.metric("Precision", "80%")
with m3:
    st.metric("Recall",    "80%")
with m4:
    st.metric("F1 Score",  "0.80", delta="+0.80 vs Graph Diffusion")

st.divider()

# ── Matplotlib confusion matrix grid ─────────────────────────────────────────

st.markdown("**Confusion Matrices**")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
fig.patch.set_facecolor("#0e1117")

model_cm = {
    "Graph Diffusion": {"TP": 0, "FP": 0, "TN": 5, "FN": 5},
    "Random Forest":   {"TP": 4, "FP": 1, "TN": 4, "FN": 1},
    "GNN":             {"TP": 3, "FP": 1, "TN": 4, "FN": 2},
}

for ax, (name, cm) in zip(axes, model_cm.items()):
    mat   = np.array([[cm["TP"], cm["FP"]], [cm["FN"], cm["TN"]]])
    cmap  = "Greens" if name == "Random Forest" else "Blues"
    im    = ax.imshow(mat, cmap=cmap, vmin=0, vmax=5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted UP", "Predicted DOWN"], color="white", fontsize=9)
    ax.set_yticklabels(["Actual UP", "Actual DOWN"],       color="white", fontsize=9)
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    for (r, c), val in np.ndenumerate(mat):
        lbl = {(0,0):"TP", (0,1):"FP", (1,0):"FN", (1,1):"TN"}[(r,c)]
        ax.text(c, r, f"{lbl}\n{val}", ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
    acc_str = {"Graph Diffusion": "50%", "Random Forest": "80%", "GNN": "70%"}[name]
    ax.set_title(f"{name}\nAccuracy: {acc_str}", color="white",
                 fontsize=10, fontweight="bold", pad=8)

plt.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "RippleStocks — Phase 6 Dashboard  |  "
    "Covid-19 test case (Jan–Apr 2020)  |  "
    "Models: Graph Diffusion · Random Forest · GNN  |  "
    "IEEE Conference Paper"
)
