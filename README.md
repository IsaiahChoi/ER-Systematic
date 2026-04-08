# Systematic Equity Research: US Semiconductors

![Strategy Dashboard](assets/images/executive_summary_dashboard.png)

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

## Key Results

### Performance Summary

| Metric | Long-Only Top-10 | SMH Benchmark | SPY |
|--------|------------------|---------------|-----|
| CAGR | 27.3% | 44.6% | 19.7% |
| Sharpe Ratio | 0.83 | 1.15 | 0.98 |
| Max Drawdown | -25.4% | -32.6% | -18.8% |
| Volatility (Ann.) | 33.2% | 36.1% | 18.0% |

### Cumulative Returns: Strategy vs Benchmarks

![Cumulative Returns](assets/images/cumulative_returns.png)

*The factor portfolio tracks the semiconductor sector while providing modestly lower drawdowns during corrections.*

### Semiconductor Sector vs Broad Market (2014–2024)

![SMH vs SPY](assets/images/smh_vs_spy.png)

*Semiconductors (SMH) delivered 44.6% CAGR vs 19.7% for the S&P 500, driven by the AI compute boom of 2023-2024.*

---

## Universe

30 US-listed semiconductor companies spanning four sub-segments:

| Sub-segment | Description | Examples |
|---|---|---|
| Fabless | Design chips; outsource fabrication to foundries | NVDA, AMD, QCOM, AVGO |
| IDM (Integrated Device Manufacturer) | Design and fabricate in-house | INTC, TXN, MU, ON |
| Equipment | Manufacture capital equipment for fabs | LRCX, AMAT, KLAC, ACLS |
| Materials/Other | Probe cards, test equipment, substrates | FORM, COHU, ONTO |

### Universe Composition

![Universe Composition](assets/images/universe_composition.png)

*By stock count, fabless companies represent 53% of the universe. By market cap, they dominate at 78%—driven largely by NVIDIA's $4.3 trillion valuation.*

---

## Factors

| Factor | Construction | Economic Rationale |
|---|---|---|
| `momentum_6_1` | 6-month return, skip last 1 month | Price momentum reflects earnings revision momentum and analyst attention |
| `earnings_revision` | YoY EPS growth (proxy for analyst revision) | Fundamental improvement leads stock price over ~3–6 months |
| `gross_margin_trend` | 4-quarter change in gross margin (bps) | Expanding margins signal pricing power and mix shift |
| `rd_intensity` | R&D / Revenue, cross-sectional percentile rank | Innovation pipeline quality; semis are IP-intensive |
| `relative_value` | EV/Sales sector-relative z-score (inverted) | Valuation discipline; mean-reversion in cyclical sector |
| `quality_composite` | Average z-score of gross margin, ROIC, revenue growth | High-quality companies survive semiconductor down-cycles |

### Factor Performance (Information Coefficients)

![Factor IC Analysis](assets/images/factor_ic_timeseries.png)

*Monthly Spearman IC between factor scores and forward 1-month returns. Quality composite shows the most consistent positive predictive power.*

#### IC Summary Statistics

| Factor | Mean IC | IC IR | Hit Rate | t-Stat |
|--------|---------|-------|----------|--------|
| quality_composite | **+0.079** | 0.37 | 80.0% | 1.18 |
| momentum_6_1 | -0.024 | -0.13 | 62.5% | -0.36 |
| relative_value | -0.072 | -0.48 | 50.0% | -0.96 |
| rd_intensity_rank | -0.178 | -1.19 | 10.0% | -3.77 |

*Note: Only quality_composite meets conventional thresholds (ICIR > 0.3) for factor viability in this sample.*

### Factor Correlation Matrix

![Factor Correlations](assets/images/factor_correlation_matrix.png)

*Low correlations between most factors confirm diversification benefits. Quality and value are negatively correlated (-0.61), as expected—high-quality companies trade at premium valuations.*

### Quintile Analysis

![Quintile Returns](assets/images/quintile_analysis.png)

*Average monthly returns by factor quintile. A monotonically increasing pattern from Q1 to Q5 indicates factor efficacy.*

---

## Regime Analysis

Strategy performance varies significantly across macroeconomic regimes:

![Regime Conditional Sharpe](assets/images/regime_sharpe.png)

| Regime | Portfolio Sharpe | SMH Sharpe |
|--------|-----------------|------------|
| High Yield Curve (expansion) | 1.43 | 2.21 |
| Low Yield Curve (contraction) | 0.19 | 0.41 |
| Low Oil Price | 1.68 | 2.17 |
| High Oil Price | -0.90 | -0.07 |
| High Industrial Production | 1.20 | 1.78 |
| Low Industrial Production | 0.16 | 0.42 |

*The strategy works best during expansionary regimes characterized by steep yield curves, low oil prices, and above-trend industrial production.*

---

## Risk Analysis

### Drawdown Profile

![Drawdown Analysis](assets/images/drawdown_profile.png)

*The factor portfolio experienced a maximum drawdown of -25.4% vs -32.6% for SMH, demonstrating modest downside protection.*

### Monthly Return Heatmap

![Monthly Returns Heatmap](assets/images/monthly_return_heatmap.png)

---

## Robustness Checks

### NVDA Exclusion Test

![NVDA Exclusion](assets/images/nvda_exclusion_test.png)

| Portfolio | Sharpe Ratio |
|-----------|--------------|
| Base Case (includes NVDA) | 0.83 |
| Ex-NVDA | 0.75 |

*Strategy retains value without NVIDIA, though the AI-driven returns from 2023-2024 are partially attributable to NVDA selection.*

### Transaction Cost Sensitivity

| Cost (bps) | CAGR | Sharpe |
|------------|------|--------|
| 0 | 27.8% | 0.84 |
| 15 (base) | 27.3% | 0.83 |
| 30 | 26.9% | 0.82 |
| 50 | 26.3% | 0.81 |

*Alpha persists across reasonable transaction cost assumptions.*

### Number of Holdings Sensitivity

![Holdings Sensitivity](assets/images/vary_n_holdings.png)

*Testing top-5, top-10, and top-15 portfolios. Top-5 shows highest Sharpe (1.18) but with higher concentration risk.*

---

## Portfolio Construction

- **Composite score**: Equal-weight z-score average of all 6 factors.
- **Long-only**: Top 10 stocks, equal-weighted, rebalanced monthly.
- **Long-short**: Long top 10 / short bottom 10, dollar-neutral.
- **Transaction costs**: 15 bps one-way (applied to portfolio turnover).

### Sub-Segment Allocation Over Time

![Segment Allocation](assets/images/segment_allocation.png)

*Portfolio composition by semiconductor sub-segment at each monthly rebalance.*

---

## Backtesting Conventions

- **No look-ahead bias**: All fundamental data lagged by 45 days from quarter-end to approximate 10-Q/10-K filing dates.
- **Rebalancing**: Month-end, using prices available at close.
- **Returns**: Based on adjusted close prices from yfinance.
- **Benchmarks**: SMH (VanEck Semiconductor ETF) and SPY (S&P 500 ETF).

---

## Repository Structure
