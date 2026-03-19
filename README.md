# RippleStocks

**RippleStocks** is a market analysis system that predicts how news events ripple through the financial ecosystem using NLP and graph-based propagation modeling.

When a major news event breaks — a pandemic, a merger, a regulation — its impact does not stop at one ticker. It cascades: through supply chains, competitors, correlated sectors, and macro indicators. RippleStocks models that cascade.

---

## Project Vision

Most financial NLP tools classify sentiment on a stock-by-stock basis. RippleStocks goes further: it treats the financial market as a **graph** where entities (companies, sectors, indices) are nodes connected by real-world relationships (supply chain, competition, ownership, correlation). When a news event hits one node, the system propagates the expected impact across the graph — weighted by relationship strength, sentiment magnitude, and event type — to produce ranked impact predictions across the entire ecosystem.

The end goal is a system that, given a breaking news event, answers:
> *"Which stocks will move, in what direction, by how much, and why?"*

---

## 6-Phase Build Plan

| Phase | Name | Description |
|-------|------|-------------|
| **1** | Data Pipeline | Ingest news (APIs, RSS, scrapers) and market data (prices, volume, fundamentals). Normalize, deduplicate, and store. |
| **2** | NLP Layer | Extract sentiment, named entities (companies, people, sectors), event type (earnings, M&A, macro, regulatory), and inter-entity relationships from raw news text. |
| **3** | Graph Construction | Build a financial knowledge graph. Nodes = companies, sectors, indices. Edges = supply chain links, competition, ownership, price correlation. |
| **4** | Propagation Model | Run a graph-based diffusion algorithm that propagates NLP-derived signals from affected nodes outward, decaying by distance, edge weight, and time. |
| **5** | Validation | Backtest predictions against historical price movements. Measure directional accuracy, magnitude error, and ranking quality. The Covid-19 case (March 2020) is the primary benchmark. |
| **6** | Dashboard UI | Interactive interface showing the event feed, the propagation graph, and predicted impact heatmaps by sector and ticker. |

---

## Validation Benchmark: Covid-19, March 2020

The primary validation case is the Covid-19 market shock of **March 2020** — one of the sharpest, most sector-differentiated crashes in modern market history. This event is ideal because:

- The originating signal (pandemic news) was textually clear and dateable
- Market reactions were **highly heterogeneous** across tickers — some crashed, some soared
- The propagation logic is domain-interpretable (travel → airlines; remote work → videoconferencing; vaccines → biotech)

### Benchmark Tickers

| Ticker | Company | Expected Direction | Propagation Rationale |
|--------|---------|-------------------|----------------------|
| **ZM** | Zoom Video | ↑ Strong positive | Remote work demand surge |
| **MSFT** | Microsoft | ↑ Moderate positive | Cloud/Teams adoption, diversified revenue |
| **AAL** | American Airlines | ↓ Strong negative | Travel collapse, direct operational impact |
| **AMZN** | Amazon | ↑ Positive | E-commerce + AWS surge offsetting logistics costs |
| **MRNA** | Moderna | ↑ Strong positive | Direct vaccine development opportunity |

The system should correctly predict the **direction** of movement for all five tickers and rank their magnitudes in a sensible order, using only the news signal and the graph — no lookahead price data.

---

## Project Structure

```
RippleStocks/
├── README.md
├── docs/
│   ├── paper_notes.md       # Notes from relevant research papers
│   └── architecture.md      # System design and technology decisions
├── phase1_data_pipeline/
├── phase2_nlp_layer/
├── phase3_graph_construction/
├── phase4_propagation_model/
├── phase5_validation/
└── phase6_dashboard_ui/
```
