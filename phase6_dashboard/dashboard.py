"""
Phase 6 — dashboard.py
RippleStocks Streamlit dashboard.

Mode 1 — Validated Demo   : proven Covid-19 results (default on load)
Mode 2 — Live Experimental: type any macro event, fetch live news via
                             Alpha Vantage, score with FinBERT, predict
                             with the saved Random Forest.

Run from project root:
    streamlit run phase6_dashboard/dashboard.py
"""

import json
import os
import pickle
import time
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Path resolution (works regardless of CWD) ─────────────────────────────────

_DASH_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(_DASH_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

GRAPH_PATH   = os.path.join(ROOT_DIR, "phase3_graph_construction", "data", "market_graph.gpickle")
RF_PATH      = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "results_random_forest.json")
GD_PATH      = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "results_graph_diffusion.json")
GNN_PATH     = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "results_gnn.json")
COMPARE_PATH = os.path.join(ROOT_DIR, "phase4_propagation_model",  "data", "comparison_results.json")
NEWS_JSON    = os.path.join(ROOT_DIR, "phase2_nlp_layer",          "data", "covid_news_enriched.json")
STOCKS_DIR   = os.path.join(ROOT_DIR, "phase1_data_pipeline",      "data", "stocks")

# ── Constants ─────────────────────────────────────────────────────────────────

TICKERS = ["ZM", "MSFT", "AMZN", "NFLX", "MRNA", "AAL", "UAL", "MGM", "XOM", "SPY"]

GROUND_TRUTH_RETURNS = {
    "ZM":   113.2, "MSFT":  10.8, "AMZN": 25.0, "NFLX": 24.9,
    "MRNA": 141.1, "AAL":  -56.4, "UAL": -65.2, "MGM": -47.7,
    "XOM":  -32.1, "SPY":   -9.2,
}
GROUND_TRUTH_LABEL = {
    "ZM": 1, "MSFT": 1, "AMZN": 1, "NFLX": 1, "MRNA": 1,
    "AAL": 0, "UAL": 0, "MGM": 0,  "XOM": 0,  "SPY": 0,
}

SECTOR_LABELS = {
    "ZM":   "Video Comms", "MSFT": "Cloud / Tech", "AMZN": "E-Commerce",
    "NFLX": "Streaming",   "MRNA": "Biotech",      "AAL":  "Airlines",
    "UAL":  "Airlines",    "MGM":  "Hospitality",   "XOM":  "Energy",
    "SPY":  "Broad Market",
}

# Mode 1 node colours — fixed by research finding
MODE1_COLOURS = {
    "MRNA": "#2ca02c", "AMZN": "#2ca02c", "NFLX": "#2ca02c", "MSFT": "#2ca02c",
    "AAL":  "#d62728", "UAL":  "#d62728", "MGM":  "#d62728", "XOM":  "#d62728",
    "ZM":   "#ff7f0e",   # amber — anomaly
    "SPY":  "#aaaaaa",   # gray — neutral
}

# Sector map for Mode 2 keyword matching
EVENT_SECTOR_MAP = {
    "interest rate": ["JPM", "BAC", "GS", "VNQ", "TLT", "ARKK"],
    "oil":           ["XOM", "CVX", "BP",  "HAL", "AAL", "UAL"],
    "pandemic":      ["ZM",  "MRNA","TDOC","AAL", "UAL", "MGM"],
    "inflation":     ["WMT", "TGT", "AMZN","TIP", "GLD", "XOM"],
    "bank":          ["JPM", "BAC", "GS",  "MS",  "SIVB","FRC"],
    "tech":          ["MSFT","GOOGL","NVDA","META","AMZN","AAPL"],
    "war":           ["LMT", "RTX", "NOC", "XOM", "CVX", "GLD"],
    "ai":            ["NVDA","MSFT","GOOGL","AMD", "META","AMZN"],
}

AV_BASE = "https://www.alphavantage.co/query"


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RippleStocks",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Base font size ── */
  html, body, [class*="css"] { font-size: 16px; }

  /* ── Main title ── */
  .main-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
  }
  .subtitle {
    color: #555; font-size: 1.15rem; margin-top: 0; margin-bottom: 0;
  }

  /* ── Cards ── */
  .event-card {
    background: #f0f4ff; border: 1px solid #c8d8ff; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 8px; font-size: 1.05rem; line-height: 1.7;
  }
  .event-card b { color: #1f77b4; }

  /* ── Anomaly banner ── */
  .anomaly-box {
    background: #fff8e1; border: 1px solid #ffc107; border-radius: 8px;
    padding: 16px 20px; margin: 14px 0;
    font-size: 1.05rem; line-height: 1.75;
  }

  /* ── Experimental warning ── */
  .exp-banner {
    background: #fff3cd; border: 2px solid #ff9800; border-radius: 8px;
    padding: 14px 20px; margin-bottom: 16px; font-size: 1.05rem; line-height: 1.6;
  }

  /* ── Sidebar ── */
  .phase-done    { color: #2ca02c; font-size: 1rem; line-height: 1.8; }
  .phase-pending { color: #ff7f0e; font-size: 1rem; line-height: 1.8; }
  section[data-testid="stSidebar"] { font-size: 1rem; }
  section[data-testid="stSidebar"] p  { font-size: 1rem; }
  section[data-testid="stSidebar"] li { font-size: 1rem; }

  /* ── Dataframe / table text ── */
  [data-testid="stDataFrame"] * { font-size: 15px !important; }
  [data-testid="stDataFrame"] th {
    font-size: 15px !important; font-weight: 700 !important;
    padding-top: 10px !important; padding-bottom: 10px !important;
  }
  [data-testid="stDataFrame"] td {
    padding-top: 10px !important; padding-bottom: 10px !important;
  }

  /* ── Metric cards ── */
  [data-testid="stMetricValue"]  { font-size: 2rem !important; font-weight: 700 !important; }
  [data-testid="stMetricLabel"]  { font-size: 1rem !important; }
  [data-testid="stMetricDelta"]  { font-size: 0.95rem !important; }

  /* ── Section spacing ── */
  .section-gap { margin-top: 2.2rem; }

  /* ── Footer ── */
  .footer {
    text-align: center; color: #888; font-size: 0.95rem;
    margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid #e0e0e0;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────

if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []
if "mode2_results" not in st.session_state:
    st.session_state.mode2_results = None


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADERS  (all cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading market graph…")
def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_phase4_results():
    rf  = {r["ticker"]: r for r in json.load(open(RF_PATH))}
    gd  = {r["ticker"]: r for r in json.load(open(GD_PATH))}
    gnn = {r["ticker"]: r for r in json.load(open(GNN_PATH))}
    comp = json.load(open(COMPARE_PATH))
    return rf, gd, gnn, comp


@st.cache_data(show_spinner=False)
def build_training_features():
    """
    Build the same 4-feature matrix used in Phase 4 model2_random_forest.py.
    Returns X (10×4), y (10,), scaler, feature list, ticker order.
    """
    articles = json.load(open(NEWS_JSON))
    G        = load_graph()

    sent_scores: dict[str, list] = defaultdict(list)
    art_counts:  dict[str, int]  = defaultdict(int)
    for a in articles:
        t = a.get("ticker", "")
        if t in TICKERS:
            sent_scores[t].append(a["sentiment"]["score"])
            art_counts[t] += 1

    rows, labels = [], []
    for ticker in TICKERS:
        avg_sent   = float(np.mean(sent_scores.get(ticker, [0.0])))
        edge_to_aal = float(G[ticker]["AAL"]["weight"]) if ticker != "AAL" else 1.0
        df         = pd.read_csv(
            os.path.join(STOCKS_DIR, f"{ticker}.csv"),
            index_col="Date", parse_dates=True,
        ).sort_index()
        ret  = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        vol  = float(ret.std())
        n_articles = art_counts.get(ticker, 0)
        rows.append([avg_sent, edge_to_aal, vol, n_articles])
        labels.append(GROUND_TRUTH_LABEL[ticker])

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y, TICKERS[:]


@st.cache_resource(show_spinner="Training Random Forest model…")
def build_rf_model():
    """Train RF on ALL 10 tickers (full dataset, not LOO) for Mode 2 inference."""
    X, y, _ = build_training_features()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    rf.fit(X_scaled, y)
    return rf, scaler


@st.cache_resource(show_spinner="Loading FinBERT sentiment model (first load ~30s)…")
def load_finbert():
    from transformers import pipeline, logging as hf_logging
    hf_logging.set_verbosity_error()
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        truncation=True,
        max_length=512,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_network(
    G,
    node_colours: dict,      # {ticker: hex_colour}
    node_sizes:   dict,      # {ticker: int 300-2000}
    epicenter:    str | None = None,
    title:        str        = "",
) -> plt.Figure:
    """
    Draw the market graph with matplotlib on a white background.
    Returns the Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("white")
    fig.patch.set_linewidth(1.5)
    fig.patch.set_edgecolor("#dddddd")
    ax.set_facecolor("white")

    pos = nx.spring_layout(G, seed=42, k=2.2, weight="weight")

    # Edge widths proportional to combined weight
    edges   = list(G.edges(data=True))
    weights = [d["weight"] for _, _, d in edges]
    max_w   = max(weights) if weights else 1

    for (u, v, d), weight in zip(edges, weights):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        width = max(0.6, (weight / max_w) * 5.5)
        alpha = max(0.18, weight / max_w * 0.65)
        ax.plot([x0, x1], [y0, y1],
                color="#888888", linewidth=width, alpha=alpha, zorder=1)

    # Nodes
    colours_list = [node_colours.get(t, "#aaaaaa") for t in G.nodes()]
    sizes_list   = [node_sizes.get(t, 900)          for t in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colours_list,
        node_size=sizes_list,
        alpha=0.92, linewidths=2.5,
        edgecolors=["#FFD700" if t == epicenter else "#444444" for t in G.nodes()],
    )

    # Labels — dark text on white background
    for ticker, (x, y) in pos.items():
        ax.text(
            x, y, ticker,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#1a1a1a", zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
        sector = SECTOR_LABELS.get(ticker, "")
        ax.text(
            x, y - 0.13, sector,
            ha="center", va="top",
            fontsize=7, color="#555555", zorder=5,
        )

    if title:
        ax.set_title(title, color="#1a1a1a", fontsize=12, pad=10, fontweight="600")

    ax.axis("off")
    plt.tight_layout(pad=0.8)
    return fig


def mode1_legend(ax=None):
    patches = [
        matplotlib.patches.Patch(color="#2ca02c", label="Predicted UP (winner)"),
        matplotlib.patches.Patch(color="#d62728", label="Predicted DOWN (loser)"),
        matplotlib.patches.Patch(color="#ff7f0e", label="Anomaly (ZM)"),
        matplotlib.patches.Patch(color="#aaaaaa", label="Neutral / uncertain"),
        matplotlib.patches.Patch(color="#FFD700", label="Epicenter node"),
    ]
    return patches


def build_leaderboard_df(rf_results: dict, comp_data: dict) -> pd.DataFrame:
    rows = []
    for entry in comp_data["per_ticker"]:
        t   = entry["ticker"]
        gt  = entry["ground_truth"]
        rf  = entry["random_forest"]
        gd  = entry["graph_diffusion"]
        gnn = entry["gnn"]
        conf = rf_results.get(t, {}).get("confidence", 0)
        rows.append({
            "Ticker":       t,
            "Sector":       SECTOR_LABELS[t],
            "RF Prediction":rf["direction"],
            "Confidence":   f"{conf:.0%}",
            "Actual Return":f"{gt['return_pct']:+.1f}%",
            "Correct?":     "✓" if rf["correct"] else "✗",
            "Note":         "⚠ Anomaly — negative sentiment masked demand surge"
                            if t == "ZM" else "",
            "_correct":     rf["correct"],
            "_zm":          t == "ZM",
            "_return":      gt["return_pct"],
        })
    df = pd.DataFrame(rows).sort_values("_return", ascending=False)
    return df


def style_leaderboard(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    display = df.drop(columns=["_correct", "_zm", "_return"]).reset_index(drop=True)

    def row_colour(row):
        idx = row.name
        orig = df.reset_index(drop=True).iloc[idx]
        if orig["_zm"]:
            return ["background-color:#fff8e1; font-weight:bold"] * len(row)
        elif orig["_correct"]:
            return ["background-color:#f0fff4"] * len(row)
        else:
            return ["background-color:#fff0f0"] * len(row)

    return display.style.apply(row_colour, axis=1)


def build_model_comparison_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Model":       ["Graph Diffusion", "Random Forest ★", "GNN"],
        "Accuracy":    ["50%",   "80%",   "70%"],
        "Precision":   ["0%",    "80%",   "75%"],
        "Recall":      ["0%",    "80%",   "60%"],
        "F1 Score":    ["0.00",  "0.80",  "0.67"],
        "TP/FP/TN/FN": ["0/0/5/5","4/1/4/1","3/1/4/2"],
        "Misses":      ["ZM MSFT AMZN NFLX MRNA", "ZM · SPY", "ZM · MRNA · SPY"],
    })


def style_model_comparison(df: pd.DataFrame):
    def row_colour(row):
        if "★" in str(row["Model"]):
            return ["background-color:#e8f5e9; font-weight:bold"] * len(row)
        elif row["Model"] == "Graph Diffusion":
            return ["color:#999"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_colour, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA VANTAGE  (Mode 2)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_av_news(tickers: list[str], limit: int = 50) -> list[dict]:
    """
    Fetch recent headlines for a list of tickers from Alpha Vantage.
    Returns list of {ticker, headline} dicts.
    Silently handles API errors and rate limits.
    """
    if not AV_KEY or AV_KEY in ("", "paste_your_key_here", "demo"):
        return []

    all_articles = []
    for i, ticker in enumerate(tickers[:6]):      # cap at 6 to avoid rate limits
        try:
            resp = requests.get(
                AV_BASE,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers":  ticker,
                    "limit":    limit,
                    "apikey":   AV_KEY,
                },
                timeout=15,
            )
            data = resp.json()
            if "feed" not in data:
                continue
            for item in data["feed"][:10]:
                all_articles.append({
                    "ticker":   ticker,
                    "headline": item.get("title", ""),
                })
        except Exception:
            pass
        if i < len(tickers) - 1:
            time.sleep(12)   # free tier: 5 calls/min
    return all_articles


def score_headlines(headlines: list[str]) -> list[float]:
    """Run FinBERT on a batch of headlines, return signed scores."""
    if not headlines:
        return []
    finbert = load_finbert()
    results = finbert(headlines)
    scores  = []
    for raw in results:
        probs = {item["label"]: item["score"] for item in raw}
        scores.append(probs.get("positive", 0) - probs.get("negative", 0))
    return scores


def map_event_to_tickers(event_text: str) -> list[str]:
    """Match keywords in event_text to EVENT_SECTOR_MAP, return unique tickers."""
    text_lower = event_text.lower()
    matched    = []
    for keyword, tickers in EVENT_SECTOR_MAP.items():
        if keyword in text_lower:
            for t in tickers:
                if t not in matched:
                    matched.append(t)
    return matched


def predict_live(
    tickers:     list[str],
    avg_sents:   dict[str, float],
    art_counts:  dict[str, int],
    G,
    rf_model,
    scaler,
) -> list[dict]:
    """
    Predict UP/DOWN for each ticker using the trained RF.
    For tickers not in our Phase 3 graph, use default feature values.
    """
    results = []
    for ticker in tickers:
        avg_sent   = avg_sents.get(ticker, 0.0)
        # Edge to AAL proxy: use graph value if available, else low default
        if ticker in G.nodes() and "AAL" in G.nodes():
            edge_to_aal = float(G[ticker]["AAL"]["weight"]) if ticker != "AAL" else 1.0
        else:
            edge_to_aal = 0.10   # unknown — weak default
        vol        = 0.05        # market-average default for unknown tickers
        n_articles = art_counts.get(ticker, 0)

        X_raw    = np.array([[avg_sent, edge_to_aal, vol, n_articles]], dtype=float)
        X_scaled = scaler.transform(X_raw)
        pred     = int(rf_model.predict(X_scaled)[0])
        proba    = rf_model.predict_proba(X_scaled)[0]
        conf     = float(proba[pred])
        signed   = float(proba[1] - proba[0])

        results.append({
            "ticker":     ticker,
            "direction":  "UP" if pred == 1 else "DOWN",
            "confidence": conf,
            "norm_score": signed,
            "avg_sentiment": avg_sent,
            "article_count": n_articles,
        })
    return results


def mode2_node_colours(results: list[dict]) -> dict[str, str]:
    colours = {}
    for r in results:
        conf = r["confidence"]
        if conf < 0.60:
            colours[r["ticker"]] = "#f5c518"   # yellow — uncertain
        elif r["direction"] == "UP":
            colours[r["ticker"]] = "#2ca02c"   # green
        else:
            colours[r["ticker"]] = "#d62728"   # red
    return colours


def mode2_node_sizes(results: list[dict]) -> dict[str, int]:
    return {r["ticker"]: int(400 + r["confidence"] * 1200) for r in results}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 RippleStocks")
    st.markdown("*Macro market ripple predictor*")
    st.divider()

    mode = st.radio(
        "**Dashboard Mode**",
        options=["Mode 1 — Validated Demo", "Mode 2 — Live Experimental"],
        index=0,
        help="Mode 1 shows the proven Covid-19 validation. "
             "Mode 2 accepts any macro event description and queries live data.",
    )

    st.divider()
    st.markdown("**Build Status**")
    phases = [
        ("✅", "Phase 1", "Data Pipeline"),
        ("✅", "Phase 2", "NLP / FinBERT"),
        ("✅", "Phase 3", "Market Graph"),
        ("✅", "Phase 4", "Propagation Models"),
        ("✅", "Phase 5", "Validation"),
        ("✅", "Phase 6", "Dashboard"),
    ]
    for icon, phase, label in phases:
        st.markdown(f"<span class='phase-done'>{icon} **{phase}** — {label}</span>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("**Model in use**")
    st.markdown("🥇 Random Forest  \n80% accuracy · F1 0.80")

    st.divider()
    st.markdown("**Covid-19 ground truth**")
    st.markdown("📅 Jan 2 – Apr 30, 2020  \n🦠 WHO pandemic — Mar 11, 2020")

    # Recent searches (Mode 2 only)
    if st.session_state.recent_searches:
        st.divider()
        st.markdown("**Recent searches**")
        for s in st.session_state.recent_searches[-5:][::-1]:
            st.markdown(f"• {s[:42]}{'…' if len(s) > 42 else ''}")

    st.divider()
    st.caption("RippleStocks v1.0  |  Graduate research")


# ── Load shared data ──────────────────────────────────────────────────────────

G                        = load_graph()
rf_results, gd_res, gnn_res, comp = load_phase4_results()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER  (both modes)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">RippleStocks</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Macro Market Ripple Predictor — '
    'Covid-19 validation: 80% accuracy (Random Forest)</p>',
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — VALIDATED DEMO
# ══════════════════════════════════════════════════════════════════════════════

if "Mode 1" in mode:

    # ── Section 1: Event Summary ──────────────────────────────────────────────
    st.markdown("## ① Event Summary")

    c1, c2, c3 = st.columns([3, 1.2, 1.2])
    with c1:
        st.markdown("""
        <div class="event-card">
          <b>Event:</b> WHO declares Covid-19 a global pandemic<br>
          <b>Date:</b> March 11, 2020<br>
          <b>Analysis window:</b> January 2 – April 30, 2020
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="event-card" style="text-align:center">
          <b>Epicenter</b><br>
          <span style="font-size:1.4rem">✈️ AAL</span><br>
          <span style="color:#888;font-size:0.85rem">Airlines — direct victim</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="event-card" style="text-align:center">
          <b>Model</b><br>
          <span style="font-size:1.1rem">Random Forest</span><br>
          <span style="color:#2ca02c;font-size:0.85rem;font-weight:bold">80% accuracy</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Section 2: Ripple Network ─────────────────────────────────────────────
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("## ② Ripple Network Visualization")

    node_sizes_m1 = {
        t: int(400 + rf_results[t]["confidence"] * 1400)
        for t in TICKERS
    }
    node_sizes_m1["AAL"] = 1800   # epicenter always large

    fig = draw_network(
        G,
        node_colours=MODE1_COLOURS,
        node_sizes=node_sizes_m1,
        epicenter="AAL",
        title="Market Shock Propagation — Covid-19 Epicenter: AAL",
    )

    # Inject legend into figure
    legend_ax = fig.axes[0]
    legend_ax.legend(
        handles=mode1_legend(),
        loc="lower left",
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
        labelcolor="#333333",
        fontsize=9,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Node size = RF confidence magnitude  ·  "
        "Edge thickness = combined graph weight  ·  "
        "Gold border = epicenter (AAL)"
    )

    # ── Section 3: Impact Leaderboard ─────────────────────────────────────────
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("## ③ Impact Leaderboard")

    df = build_leaderboard_df(rf_results, comp)
    st.dataframe(
        style_leaderboard(df),
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    st.markdown("""
    <div class="anomaly-box">
      ⚠️ <b>ZM Anomaly</b> — Zoom Video Communications was misclassified by
      <b>all three models</b>.<br>
      <b>FinBERT sentiment:</b> −0.75 (negative) — Zoom-bombing scandals,
      encryption disputes and institutional bans dominated headlines.<br>
      <b>Actual price outcome:</b> <b style="color:#2ca02c">+113.2%</b> —
      driven by unprecedented remote-work adoption.<br>
      <b>Key insight:</b> Headline tone completely diverged from underlying
      economic fundamentals. Sentiment-only models cannot detect demand-driven
      outliers without structural sector-relationship graph features.
    </div>
    """, unsafe_allow_html=True)

    # ── Section 4: Model Comparison ───────────────────────────────────────────
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("## ④ Model Comparison")

    st.dataframe(
        style_model_comparison(build_model_comparison_df()),
        use_container_width=True,
        hide_index=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Best Accuracy",   "80%",  delta="RF vs 50% baseline")
    with m2: st.metric("Best F1 Score",   "0.80", delta="+0.80 vs Graph Diffusion")
    with m3: st.metric("Best Precision",  "80%")
    with m4: st.metric("Best Recall",     "80%")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — LIVE EXPERIMENTAL SEARCH
# ══════════════════════════════════════════════════════════════════════════════

else:
    # Experimental banner — always visible in Mode 2
    st.markdown("""
    <div class="exp-banner">
      ⚠️ <b>Experimental mode</b> — model trained on Covid-19 data only (10 tickers,
      Jan–Apr 2020). Results are indicative, not validated. Predictions for tickers
      outside the original training set use approximate graph features.
      Do not use for financial decisions.
    </div>
    """, unsafe_allow_html=True)

    rf_model, scaler = build_rf_model()

    # ── Section 1: Search ─────────────────────────────────────────────────────
    st.markdown("## ① Describe a Macro Event")

    col_txt, col_btn = st.columns([5, 1])
    with col_txt:
        event_input = st.text_input(
            "Macro event",
            placeholder='e.g. "Fed raises interest rates by 75bps" or "oil price crash"',
            label_visibility="collapsed",
        )
    with col_btn:
        run_btn = st.button("▶ Predict Ripple", type="primary", use_container_width=True)

    # ── Section 2: Sector mapper ──────────────────────────────────────────────
    matched_tickers: list[str] = []
    if event_input:
        matched_tickers = map_event_to_tickers(event_input)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("## ② Sector Mapper")
        if matched_tickers:
            st.markdown(
                f"**Matched tickers** from event keywords: "
                f"`{'  ·  '.join(matched_tickers)}`"
            )
            # Show which keywords fired
            fired = [kw for kw in EVENT_SECTOR_MAP if kw in event_input.lower()]
            st.caption(f"Keywords matched: {', '.join(fired)}")
        else:
            st.warning(
                "No keyword match found. Try words like: "
                "*interest rate, oil, pandemic, inflation, bank, tech, war, ai*"
            )

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run_btn and event_input and matched_tickers:
        # Save to recent searches
        if event_input not in st.session_state.recent_searches:
            st.session_state.recent_searches.append(event_input)
            st.session_state.recent_searches = st.session_state.recent_searches[-5:]

        with st.spinner("Fetching live news from Alpha Vantage…"):
            articles = fetch_av_news(matched_tickers)

        if not articles:
            st.warning(
                "No recent news found for these tickers. "
                "Try a different event description, or check your Alpha Vantage API key.",
                icon="📭",
            )
        else:
            # Score with FinBERT
            with st.spinner(f"Scoring {len(articles)} headlines with FinBERT…"):
                headlines   = [a["headline"] for a in articles if a["headline"]]
                scores_list = score_headlines(headlines)

            # Aggregate sentiment per ticker
            ticker_sents:  dict[str, list] = defaultdict(list)
            ticker_counts: dict[str, int]  = defaultdict(int)
            for article, score in zip(articles, scores_list):
                t = article["ticker"]
                ticker_sents[t].append(score)
                ticker_counts[t] += 1

            avg_sents = {
                t: float(np.mean(s)) for t, s in ticker_sents.items()
            }

            # Predict
            live_results = predict_live(
                matched_tickers, avg_sents, ticker_counts, G, rf_model, scaler
            )
            st.session_state.mode2_results = {
                "event":   event_input,
                "results": live_results,
                "n_articles": len(articles),
            }

    # ── Section 3: Results ────────────────────────────────────────────────────
    if st.session_state.mode2_results:
        res_data  = st.session_state.mode2_results
        live_res  = res_data["results"]
        n_arts    = res_data["n_articles"]

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("## ③ Ripple Prediction Results")

        st.info(
            f"**Event:** {res_data['event']}  \n"
            f"**Headlines analysed:** {n_arts}  \n"
            f"**Tickers assessed:** {len(live_res)}  \n"
            "⚠️ Predictions use approximate features for non-training tickers.",
            icon="🔬",
        )

        # Network visualization — build a sub-graph for matched tickers
        ticker_set = [r["ticker"] for r in live_res]
        live_colours = mode2_node_colours(live_res)
        live_sizes   = mode2_node_sizes(live_res)

        # Build a temporary graph limited to matched tickers
        G_live = nx.Graph()
        G_live.add_nodes_from(ticker_set)
        for u, v, d in G.edges(data=True):
            if u in ticker_set and v in ticker_set:
                G_live.add_edge(u, v, **d)

        if G_live.number_of_nodes() > 1:
            fig2 = draw_network(
                G_live,
                node_colours=live_colours,
                node_sizes=live_sizes,
                epicenter=None,
                title=f"Predicted Ripple — {res_data['event'][:60]}",
            )
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
            st.caption(
                "🟢 Green = UP  ·  🔴 Red = DOWN  ·  🟡 Yellow = uncertain (<60% confidence)  "
                "·  Node size = confidence magnitude"
            )
        else:
            st.info("Graph view requires at least 2 matched tickers with connections.")

        # Leaderboard for live results
        live_df = pd.DataFrame([{
            "Ticker":      r["ticker"],
            "Prediction":  r["direction"],
            "Confidence":  f"{r['confidence']:.0%}",
            "Avg Sentiment": f"{r['avg_sentiment']:+.3f}",
            "Articles":    r["article_count"],
        } for r in sorted(live_res, key=lambda x: x["norm_score"], reverse=True)])

        def style_live(df):
            def row_colour(row):
                if "UP" in str(row["Prediction"]):
                    return ["background-color:#f0fff4"] * len(row)
                return ["background-color:#fff0f0"] * len(row)
            return df.style.apply(row_colour, axis=1)

        st.dataframe(style_live(live_df), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="exp-banner" style="margin-top:10px">
          ⚠️ <b>Reminder:</b> These predictions are experimental. The model was trained
          on 10 tickers from the Covid-19 period only. Tickers outside the original
          training set use default graph and volatility features.
        </div>
        """, unsafe_allow_html=True)

    elif run_btn and not matched_tickers:
        st.info("Please describe a macro event with recognizable keywords to get predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER  (both modes)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  RippleStocks — Built as part of graduate research.
  Covid-19 validation: 80% accuracy (Random Forest).<br>
  Models: Graph Diffusion · Random Forest · GNN  |
  Data: Alpha Vantage · yfinance  |  NLP: FinBERT (ProsusAI)
</div>
""", unsafe_allow_html=True)
