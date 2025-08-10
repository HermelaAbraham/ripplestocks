# RippleStocks — MVP Specification

## Overview
RippleStocks is a market sentiment analysis tool that tracks how news events might trigger ripple effects on related stocks within 1–3 days.  
The MVP focuses on automating headline ingestion, mapping news to relevant tickers, applying sentiment analysis, and delivering results through an API and simple dashboard.

---

## Problem Statement
Investors often miss secondary market movements caused by indirect news impacts.  
RippleStocks provides a quick, explainable view of potential ripple effects so users can spot opportunities earlier.

---

## Target Users
- Retail traders seeking edge through news sentiment  
- Financial analysts looking for ripple-effect insights  
- Students and hobbyists learning market reaction patterns  

---

## MVP Goals
- **Ingest** top finance news headlines from one reliable free source  
- **Extract** companies/entities and map them to stock tickers  
- **Analyze** sentiment using a baseline model (VADER or FinBERT)  
- **Estimate** ripple effect score using simple heuristics  
- **Store** data in a Postgres database  
- **Expose** results via a REST API (FastAPI)  
- **Display** them in a minimal React/Next.js dashboard  

---

## MVP Scope
✅ Single news API or RSS feed ingestion  
✅ Basic entity → ticker dictionary mapping  
✅ Sentiment scoring (positive, negative, neutral)  
✅ Simple scoring model for ripple probability & direction  
✅ API endpoints for fetching results  
✅ Minimal UI with ticker search & latest impacts  

---

## Out of Scope (for MVP)
❌ Multiple news sources  
❌ Complex ML-based impact forecasting  
❌ Real-time trading signals  
❌ Historical backtesting at scale  

---

## Tech Stack
**Backend:** Python, FastAPI, SQLAlchemy, Pandas, VADER/FinBERT  
**Frontend:** React (Next.js), Tailwind CSS  
**Database:** PostgreSQL  
**Infrastructure:** Docker, Docker Compose, GitHub Actions (CI/CD)  

---

## Success Criteria
- End-to-end pipeline from news ingestion → sentiment → ripple score in < 30 seconds  
- Minimum 55% directional hit-rate vs. naive baseline on a small labeled dataset  
- Clean, readable code and reproducible local setup with Docker  

---

## Quickstart
```bash
# Clone the repo
git clone https://github.com/HermsCodes/ripplestocks.git
cd ripplestocks

# Backend setup
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend setup
cd ../frontend
npm install
npm run dev

