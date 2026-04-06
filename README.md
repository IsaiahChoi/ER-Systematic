# Systematic Equity Research: US Semiconductors

> *A narrative-driven quantitative research project simulating the workflow of a junior equity analyst at a fundamental long/short hedge fund.*

---

## Overview

This repository implements a **complete factor-based investment research framework** for the US semiconductor sector, covering the period 2014–2024. It is structured as both a reusable Python library (`src/`) and an end-to-end research notebook (`notebooks/semiconductor_equity_research.ipynb`) that reads like a professional sell-side or hedge-fund research note.

The project follows the full analyst workflow:

1. **Universe Construction** — 30 US semiconductor stocks spanning fabless designers, IDMs, equipment makers, and materials suppliers.
2. **Data Collection** — Prices via `yfinance`; fundamentals via `yfinance` quarterly financials; macro context via the St. Louis Fed's FRED API.
3. **Factor Construction** — Six alpha factors with economic intuition drawn from the semiconductor industry cycle.
4. **Cross-Sectional Analysis** — Information Coefficient (IC) analysis, quintile portfolio construction, and factor correlation.
5. **Portfolio Construction** — Composite signal → long-only top-10 and long-short top-10 / bottom-10 strategies.
6. **Backtesting** — Rigorous performance attribution, transaction-cost sensitivity, and regime analysis.

---

## Preview

> Run the notebook end-to-end to generate all charts and tables. See [Setup](#setup) below.

Expected outputs include:
- Cumulative return charts comparing the strategy vs. SMH and SPY
- IC time-series and quintile analysis for all 6 factors
- Monthly return heatmaps
- Regime-conditional Sharpe ratio tables
- Fama-French 5-factor exposure regressions

---

## Repository Structure

```
systematic-equity-research-semiconductors/
├── README.md
├── requirements.txt
├── config.py                   ← All parameters, tickers, FRED series
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← Price, fundamental, and macro data fetching
│   ├── universe.py             ← Dynamic universe filtering at each rebalance
│   ├── factors.py              ← Six alpha factors (momentum, quality, value, etc.)
│   ├── portfolio.py            ← Composite scoring, long-only, long-short construction
│   ├── analytics.py            ← Backtesting, performance statistics, regime analysis
│   └── utils.py                ← Shared helpers (z-score, winsorize, rebalance dates)
├── notebooks/
│   └── semiconductor_equity_research.ipynb   ← PRIMARY DELIVERABLE
└── tests/
    └── test_factors.py         ← Pytest unit tests for factor calculations
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/systematic-equity-research-semiconductors.git
cd systematic-equity-research-semiconductors
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Obtain a FRED API Key

This project uses macroeconomic data from the [Federal Reserve Bank of St. Louis FRED](https://fred.stlouisfed.org/). A free API key is required.

1. Register at: https://fred.stlouisfed.org/docs/api/api_key.html
2. Set your key in one of two ways:

   **Option A — Environment variable (recommended):**
   ```bash
   export FRED_API_KEY="your_key_here"
   ```

   **Option B — Edit `config.py` directly:**
   ```python
   FRED_API_KEY: str = "your_key_here"
   ```

### 4. Launch the notebook

```bash
jupyter notebook notebooks/semiconductor_equity_research.ipynb
```

Then run all cells (`Kernel → Restart & Run All`). The full run takes approximately **5–10 minutes** depending on network speed (yfinance data fetching is the bottleneck).

---

## Methodology

### Universe

30 US-listed semiconductor companies spanning four sub-segments:

| Sub-segment | Description | Examples |
|---|---|---|
| Fabless | Design chips; outsource fabrication to foundries | NVDA, AMD, QCOM, AVGO |
| IDM (Integrated Device Manufacturer) | Design and fabricate in-house | INTC, TXN, MU, ON |
| Equipment | Manufacture capital equipment for fabs | LRCX, AMAT, KLAC, ACLS |
| Materials/Other | Probe cards, test equipment, substrates | FORM, COHU, ONTO |

### Factors

| Factor | Construction | Economic Rationale |
|---|---|---|
| `momentum_6_1` | 6-month return, skip last 1 month | Price momentum reflects earnings revision momentum and analyst attention |
| `earnings_revision` | YoY EPS growth (proxy for analyst revision) | Fundamental improvement leads stock price over ~3–6 months |
| `gross_margin_trend` | 4-quarter change in gross margin (bps) | Expanding margins signal pricing power and mix shift |
| `rd_intensity` | R&D / Revenue, cross-sectional percentile rank | Innovation pipeline quality; semis are IP-intensive |
| `relative_value` | EV/Sales sector-relative z-score (inverted) | Valuation discipline; mean-reversion in cyclical sector |
| `quality_composite` | Average z-score of gross margin, ROIC, revenue growth | High-quality companies survive semiconductor down-cycles |

### Portfolio Construction

- **Composite score**: Equal-weight z-score average of all 6 factors.
- **Long-only**: Top 10 stocks, equal-weighted, rebalanced monthly.
- **Long-short**: Long top 10 / short bottom 10, dollar-neutral.
- **Transaction costs**: 15 bps one-way (applied to portfolio turnover).

### Backtesting Conventions

- **No look-ahead bias**: All fundamental data lagged by 45 days from quarter-end to approximate 10-Q/10-K filing dates.
- **Rebalancing**: Month-end, using prices available at close.
- **Returns**: Based on adjusted close prices from yfinance.
- **Benchmarks**: SMH (VanEck Semiconductor ETF) and SPY (S&P 500 ETF).

---

## Important Disclaimers

> **Simulated Data**: The semiconductor sector revenue growth series used in the macro analysis is **simulated** as a noisy proxy based on FRED's Industrial Production: Business Equipment series (IPBUSEQ). Real-world implementation should use SIA monthly sales data or Bloomberg Intelligence estimates. All simulated series are clearly labeled in the notebook.

> **Survivorship Bias**: The universe is a fixed list of 30 stocks selected as of 2024. Companies that were delisted, acquired, or went bankrupt between 2014 and 2024 are excluded. This introduces upward bias in historical returns. Real-world research would use a point-in-time database.

> **Not Investment Advice**: This project is for educational and research purposes only. Nothing in this repository constitutes investment advice. Past performance of any modeled strategy does not guarantee future results.

> **Analyst Estimates Proxy**: The `earnings_revision` factor uses a trailing EPS growth rate as a proxy for analyst estimate revisions. Real implementation would use IBES/I/B/E/S consensus estimates. This simplification introduces noise relative to a production factor.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Suggested Extensions

- **IBES analyst estimates** via Refinitiv/FactSet for a true earnings revision factor
- **Option-implied volatility** as a sentiment/uncertainty factor
- **Supply chain data** (SEMI equipment book-to-bill, Taiwan export data) as a leading indicator
- **Short interest ratio** as a contrarian signal
- **ESG scores** given increasing regulatory focus on semiconductor supply chains
- **Alternative data**: job postings (signal R&D ramp), patent filings, conference call NLP sentiment

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with Python 3.10+, pandas, yfinance, fredapi, statsmodels, and matplotlib.*
