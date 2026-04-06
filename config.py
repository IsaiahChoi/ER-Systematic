"""
config.py
=========
Central configuration for the systematic semiconductor equity research project.
All parameters, tickers, FRED series, and strategy settings live here so that
every module pulls from a single source of truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

TICKERS: list[str] = [
    "NVDA",  # NVIDIA – fabless GPU / AI compute
    "AMD",   # Advanced Micro Devices – fabless CPU/GPU
    "INTC",  # Intel – IDM (CPU)
    "AVGO",  # Broadcom – fabless (networking, storage)
    "QCOM",  # Qualcomm – fabless (mobile SoC, RF)
    "TXN",   # Texas Instruments – IDM (analog, embedded)
    "ADI",   # Analog Devices – IDM (analog signal processing)
    "MRVL",  # Marvell Technology – fabless (networking, storage)
    "MU",    # Micron Technology – IDM (DRAM, NAND memory)
    "LRCX",  # Lam Research – equipment (etch, deposition)
    "AMAT",  # Applied Materials – equipment (deposition, etch, inspection)
    "KLAC",  # KLA Corporation – equipment (process control / inspection)
    "ON",    # ON Semiconductor – IDM (power, analog, image sensors)
    "NXPI",  # NXP Semiconductors – fabless (automotive, IoT)
    "MCHP",  # Microchip Technology – IDM (microcontrollers, analog)
    "SWKS",  # Skyworks Solutions – fabless (RF / wireless)
    "QRVO",  # Qorvo – fabless (RF components)
    "MPWR",  # Monolithic Power Systems – fabless (power management)
    "WOLF",  # Wolfspeed – IDM (SiC power, RF)
    "ALGM",  # Allegro MicroSystems – fabless (magnetic sensors, power)
    "SLAB",  # Silicon Laboratories – fabless (IoT, wireless)
    "MTSI",  # MACOM Technology Solutions – fabless (RF, microwave)
    "RMBS",  # Rambus – fabless (memory interface chips, IP)
    "CRUS",  # Cirrus Logic – fabless (audio chips, power)
    "DIOD",  # Diodes Incorporated – fabless (discrete, analog)
    "POWI",  # Power Integrations – fabless (power conversion ICs)
    "ACLS",  # Axcelis Technologies – equipment (ion implant)
    "ONTO",  # Onto Innovation – equipment (process control, metrology)
    "FORM",  # FormFactor – equipment / materials (probe cards)
    "COHU",  # Cohu – equipment (test handlers, thermal subsystems)
]

# Sub-segment classification for each ticker
# Segments: "Fabless", "IDM", "Equipment", "Materials/Other"
TICKER_SEGMENTS: dict[str, str] = {
    "NVDA": "Fabless",
    "AMD":  "Fabless",
    "INTC": "IDM",
    "AVGO": "Fabless",
    "QCOM": "Fabless",
    "TXN":  "IDM",
    "ADI":  "IDM",
    "MRVL": "Fabless",
    "MU":   "IDM",
    "LRCX": "Equipment",
    "AMAT": "Equipment",
    "KLAC": "Equipment",
    "ON":   "IDM",
    "NXPI": "Fabless",
    "MCHP": "IDM",
    "SWKS": "Fabless",
    "QRVO": "Fabless",
    "MPWR": "Fabless",
    "WOLF": "IDM",
    "ALGM": "Fabless",
    "SLAB": "Fabless",
    "MTSI": "Fabless",
    "RMBS": "Fabless",
    "CRUS": "Fabless",
    "DIOD": "Fabless",
    "POWI": "Fabless",
    "ACLS": "Equipment",
    "ONTO": "Equipment",
    "FORM": "Materials/Other",
    "COHU": "Equipment",
}

# Human-readable company names for display in tables
TICKER_NAMES: dict[str, str] = {
    "NVDA": "NVIDIA Corporation",
    "AMD":  "Advanced Micro Devices",
    "INTC": "Intel Corporation",
    "AVGO": "Broadcom Inc.",
    "QCOM": "Qualcomm Inc.",
    "TXN":  "Texas Instruments",
    "ADI":  "Analog Devices",
    "MRVL": "Marvell Technology",
    "MU":   "Micron Technology",
    "LRCX": "Lam Research",
    "AMAT": "Applied Materials",
    "KLAC": "KLA Corporation",
    "ON":   "ON Semiconductor",
    "NXPI": "NXP Semiconductors",
    "MCHP": "Microchip Technology",
    "SWKS": "Skyworks Solutions",
    "QRVO": "Qorvo Inc.",
    "MPWR": "Monolithic Power Systems",
    "WOLF": "Wolfspeed Inc.",
    "ALGM": "Allegro MicroSystems",
    "SLAB": "Silicon Laboratories",
    "MTSI": "MACOM Technology Solutions",
    "RMBS": "Rambus Inc.",
    "CRUS": "Cirrus Logic",
    "DIOD": "Diodes Incorporated",
    "POWI": "Power Integrations",
    "ACLS": "Axcelis Technologies",
    "ONTO": "Onto Innovation",
    "FORM": "FormFactor Inc.",
    "COHU": "Cohu Inc.",
}

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

BENCHMARK_TICKERS: list[str] = ["SMH", "SPY"]

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

START_DATE: str = "2014-01-01"
END_DATE: str   = "2024-12-31"

# ---------------------------------------------------------------------------
# FRED macro series
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "industrial_prod": "INDPRO",    # Industrial Production Index (total)
    "yield_curve":     "T10Y2Y",    # 10-Year minus 2-Year Treasury spread
    "oil":             "DCOILWTICO", # WTI crude oil price (daily)
    "rate_10y":        "DGS10",     # 10-Year Treasury constant maturity rate
    "biz_equipment":   "IPBUSEQ",   # IP: Business Equipment (proxy for semi demand)
}

# FRED API key – set this before running.
# Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
# Alternatively set the environment variable FRED_API_KEY.
FRED_API_KEY: str = ""  # ← set your key here or via os.environ["FRED_API_KEY"]

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

REBALANCE_FREQ: str     = "M"          # Monthly rebalancing ('M' = month-end)
TRANSACTION_COST_BPS: int = 15         # One-way transaction cost in basis points
N_LONG: int             = 10           # Number of long positions
N_SHORT: int            = 10           # Number of short positions (long-short)
RISK_FREE_RATE: float   = 0.02         # Annual risk-free rate for Sharpe calculation
FILING_LAG_DAYS: int    = 45           # Days lag to avoid look-ahead on financials

# Factor weights for composite score (None = equal weight)
FACTOR_WEIGHTS: dict[str, float] | None = None

# Minimum filter thresholds
MIN_MARKET_CAP: float   = 1e9          # $1 billion minimum market cap
MIN_PRICE_HISTORY: int  = 252          # ~1 year of daily price history

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

COLOR_PALETTE: dict[str, str] = {
    "primary":    "#1B3A6B",   # Deep navy blue
    "secondary":  "#2E86AB",   # Teal
    "accent":     "#E07A2F",   # Burnt orange
    "negative":   "#C0392B",   # Red
    "neutral":    "#7F8C8D",   # Grey
    "light_blue": "#AED6F1",   # Light blue (fills)
    "light_orange": "#FAD7A0", # Light orange (fills)
    "green":      "#1E8449",   # Green (positive)
}

FIGURE_DPI: int    = 120
FIGURE_SIZE: tuple[int, int] = (14, 6)
