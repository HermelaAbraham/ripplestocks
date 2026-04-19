# RippleStocks — Master Briefing Document

## What is RippleStocks?
A 6-phase market analysis system that predicts how macro-economic 
news events ripple through the financial ecosystem using NLP, 
economic dependency graphs, and a propagation model.
Validated against the Covid-19 test case (March 2020).

## Key Details
- Test case: WHO declares pandemic March 11, 2020
- Tickers: ZM, MSFT, AMZN, NFLX, MRNA, AAL, UAL, MGM, XOM, SPY
- Professor feedback: Compare 3 models in Phase 4
  (Graph Diffusion, Random Forest, GNN)
- Final deliverable: IEEE conference paper

## Ground Truth (Phase 5 validation targets)
- MRNA +141%, ZM +113%, AMZN +25%, NFLX +25%, MSFT +11%
- UAL -65%, AAL -56%, MGM -48%, XOM -32%, SPY -9%

## Build Status
- ✅ Phase 1 — Data pipeline COMPLETE
  - stock_collector.py → 82 days OHLCV, 10 tickers
  - news_collector.py → 426 articles, Alpha Vantage API
- ✅ Phase 2 — NLP layer COMPLETE
  - entity_extractor.py → 534 entities, 3-layer strategy
  - sentiment_scorer.py → FinBERT, 426 articles scored
  - Key finding: ZM anomaly (-0.75 sentiment, +113% price)
    due to security scandals masking demand surge
- ✅ Phase 3 — Graph construction COMPLETE
  - graph_builder.py → NetworkX, 3 edge sources
  - Sources: domain knowledge + co-mentions + price correlation
- ✅ Phase 4 — Propagation model COMPLETE
  - model1_graph_diffusion.py → 50% accuracy (predicts all DOWN, misses winners)
  - model2_random_forest.py → 80% accuracy (best model, misses ZM + SPY)
  - model3_gnn.py → 70% accuracy (misses ZM, MRNA, SPY)
  - model_comparison.py → comparison table generated
  - Key finding: Random Forest wins
  - Key anomaly: ZM (-0.75 sentiment vs +113% price) — ALL models get it wrong
  - Core paper argument: sentiment alone insufficient, network propagation needed
- ✅ Phase 5 — Validation COMPLETE
  - validator.py → formal metrics against Covid-19 ground truth
  - Graph Diffusion: 50% accuracy, F1 0.00 (predicts all DOWN)
  - Random Forest:   80% accuracy, F1 0.80 (best model)
  - GNN:             70% accuracy, F1 0.67
  - ZM anomaly confirmed: ALL three models misclassified ZM
    (-0.75 sentiment vs +113% price — headline tone ≠ fundamentals)
  - Outputs: validation_report.txt + model_comparison.csv
- ⏳ Phase 6 — Dashboard UI UP NEXT

## Working Method
- CoCo (Claude chat) = architect, teacher, paper notes
- Claude Code (terminal) = builder, file manager
- GitHub = version control and backup
- Workflow: design here → build in Claude Code → push to GitHub

## Preferences
- Call Claude: CoCo
- Call user: Herms
- Explain everything in plain English
- Log paper notes after every major step
- Always include 3-line project summary when starting Claude Code

## IEEE Paper Notes

### System Overview
RippleStocks is a 6-layer market analysis pipeline designed to 
predict the cascading effects of macro-economic events on financial 
markets. Unlike traditional sentiment analysis tools that evaluate 
stocks in isolation, RippleStocks models the market as an 
interconnected economic dependency graph, enabling prediction of 
secondary and tertiary ripple effects from a single event epicenter.

### Phase 1 — Data Pipeline
Phase 1 establishes a dual data ingestion pipeline. Financial 
time-series data was collected via the yfinance library producing 
82 trading days of OHLCV data across 10 carefully selected tickers 
representing Covid-19 market winners and losers. News data was 
collected via the Alpha Vantage News API, yielding 426 articles 
spanning January through April 2020. Coverage variability across 
tickers reflects real-world media attention patterns prior to the 
pandemic peak, and itself constitutes an observable signal 
consistent with the hypothesis that ripple effects propagate from 
well-covered epicenters outward to less-covered secondary stocks.

### Phase 1 — Ground Truth Validation
Sanity validation of the collected financial dataset confirms the 
expected Covid-19 market impact pattern. Direct beneficiaries 
(MRNA +141%, ZM +113%) significantly outperformed indirect 
beneficiaries (AMZN +25%, NFLX +25%), while direct victims 
(UAL -65%, AAL -56%) experienced more severe losses than indirect 
victims (XOM -32%). This tiered impact pattern forms the ground 
truth against which the Phase 4 propagation model will be 
evaluated in Phase 5.

### Phase 1 — Data Source Strategy
News data is sourced via the Alpha Vantage News & Sentiment API, 
selected for its finance-specific coverage, historical depth, and 
pre-extracted sentiment scores. The news collector structures each 
article into a standardized JSON schema with placeholder fields for 
entity extraction and sentiment scoring, populated in Phase 2. 
The ingestion layer is designed to be source-agnostic, enabling 
seamless substitution with alternative feeds such as NewsAPI or GDELT.

### Phase 2 — Entity Extraction
Phase 2 Script 1 implements a three-layer entity extraction strategy 
combining spaCy Named Entity Recognition, exhaustive alias scanning 
for company name variants, and bare ticker symbol matching. 
Processing all 426 articles yielded 534 total entity mentions with 
an average of 1.25 entities per article. 72% of articles contained 
at least one resolvable entity. Notable entity ambiguity was observed 
for the MGM ticker symbol — a known limitation addressed through 
sentiment filtering in Script 2.

### Phase 2 — Sentiment Scoring
Phase 2 Script 2 applies FinBERT sentiment scoring using the formula 
P(positive) − P(negative), yielding scores on [−1, +1]. Overall 
corpus sentiment averaged −0.0383, consistent with net negative but 
recovering market conditions of March–April 2020. Per-ticker scores 
showed strong directional alignment: MRNA (+0.49, +141% price), 
NFLX (+0.21, +25%), AAL (−0.96, −56%), XOM (−0.16, −32%). 
A notable anomaly was observed for ZM (−0.75 sentiment, +113% price), 
attributable to concurrent security and privacy coverage masking 
underlying demand surge. This finding motivates the graph-based 
propagation approach — sentiment alone is insufficient when 
network-level demand signals override headline tone.

### Key Research Contribution
By shifting focus from isolated sentiment analysis to network-based 
impact prediction, RippleStocks addresses a gap in existing market 
analysis tools. The system models economic dependencies as a directed 
weighted graph, enabling simulation of shock propagation from an 
event epicenter through 1st, 2nd, and 3rd order affected entities. 
Model comparison across Graph Diffusion, Random Forest, and GNN 
provides empirical evidence for the most effective propagation 
architecture. This represents the core novel contribution of the work.

### Tiered Impact Framework
Analysis of the Covid-19 test case reveals a tiered impact structure:
(1) Direct beneficiaries — core product became essential overnight
    (MRNA +141%, ZM +113%)
(2) Indirect beneficiaries — gained from behavioral shifts
    (AMZN +25%, NFLX +25%)
(3) Cushioned performers — partial exposure (MSFT +11%)
(4) Direct victims — core business suspended (UAL -65%, AAL -56%)
(5) Indirect victims — supply chain dependencies (XOM -32%)
(6) Systemic baseline — broad market impact (SPY -9%)

### Professor Feedback
Faculty advisor recommended comparative evaluation of multiple ML 
models for the propagation prediction task. RippleStocks implements 
and benchmarks three models — Graph Diffusion, Random Forest, and 
GNN — comparing performance on directional accuracy, precision, 
and recall against the Covid-19 ground truth. This comparison forms 
the core experimental contribution of the paper.

### Phase 5 — Validation Results
Formal validation of all three propagation models against the 
Covid-19 ground truth (10 tickers, Jan 2 – Apr 30 2020) yielded 
the following results. Graph Diffusion achieved 50% directional 
accuracy with F1 = 0.00, exposing a structural limitation: the 
model propagates negative shocks outward but has no mechanism to 
flip a signal positive for sectors that benefit from a crisis, 
resulting in all-DOWN predictions. Random Forest achieved 80% 
accuracy with F1 = 0.80, correctly classifying 8 of 10 tickers 
using four hand-crafted features (sentiment, graph proximity to 
epicenter, volatility, article count). GNN achieved 70% accuracy 
with F1 = 0.67, performing strongly on clear losers but 
misclassifying outliers with sparse news coverage (ZM, MRNA, SPY). 
Confusion matrix summary: GD (TP=0, FP=0, TN=5, FN=5), 
RF (TP=4, FP=1, TN=4, FN=1), GNN (TP=3, FP=1, TN=4, FN=2).

### Phase 5 — ZM Anomaly Analysis
The ZM ticker was misclassified by all three models, representing 
the most significant finding of the validation phase. ZM achieved 
a FinBERT sentiment score of −0.75 (negative) due to concurrent 
Zoom-bombing, encryption disputes, and institutional bans dominating 
headlines. Despite this, ZM stock surged +113% over the same period 
driven by unprecedented remote-work adoption. The Random Forest 
predicted DOWN with only 51% confidence — effectively a coin flip — 
while the GNN predicted DOWN with 100% confidence, demonstrating 
that graph-propagated negative signals can overwhelm correct 
structural reasoning when node features are sentiment-dominated. 
This anomaly constitutes the central empirical motivation for the 
paper's argument: network-level propagation models require structural 
benefit-relationship features (which sectors gain from a crisis) 
in addition to sentiment features to correctly classify demand-driven 
outliers. The ZM case is presented as a named limitation and 
direction for future work.

## IEEE Paper Draft Status
- Abstract → IN PROGRESS
- Introduction → IN PROGRESS  
- Related Work → pending
- Methodology → pending
- Experiments → pending
- Results → pending
- Discussion → pending
- Future Work → pending
- Conclusion → pending

## Paper Writing Notes
- Herms writes personal understanding sections
- CoCo provides technical draft per section
- Target venue: IEEE conference paper
- Core contribution: network-based propagation 
  vs isolated sentiment analysis
- Key result: Random Forest 80% accuracy, F1 0.80
- Star finding: ZM anomaly motivates entire approach

## 3-Line Project Summary (use at start of every Claude session)
"I'm building RippleStocks, a 6-phase market analysis system that 
predicts how macro-economic news events ripple through the financial 
ecosystem using NLP, economic dependency graphs, and a propagation 
model. The system is validated against the Covid-19 test case 
(March 2020) tracking tickers ZM, MSFT, AAL, AMZN, MRNA and others. 
We follow a strict phase-by-phase build: Phase 1 (data pipeline) → 
Phase 2 (NLP/entity extraction) → Phase 3 (graph construction) → 
Phase 4 (propagation model — comparing Graph Diffusion, Random 
Forest, and GNN) → Phase 5 (validation) → Phase 6 (dashboard UI)."
