"""Asset universes used in the study, with the group (asset class / sector) of each asset.

Groups drive the group-level constraints (e.g. "no more than 30% in any one sector").
"""

ETF_MULTI_ASSET = {
    # Equity
    "SPY": "Equity", "QQQ": "Equity", "IWM": "Equity", "EFA": "Equity",
    "EEM": "Equity", "VGK": "Equity", "EWJ": "Equity",
    # Fixed income
    "TLT": "Bonds", "IEF": "Bonds", "SHY": "Bonds", "LQD": "Bonds",
    "HYG": "Bonds", "TIP": "Bonds", "EMB": "Bonds",
    # Commodities
    "GLD": "Commodities", "SLV": "Commodities", "DBC": "Commodities", "USO": "Commodities",
    # Real estate
    "VNQ": "RealEstate", "RWX": "RealEstate",
}

NIFTY_STOCKS = {
    "HDFCBANK.NS": "Financials", "ICICIBANK.NS": "Financials", "KOTAKBANK.NS": "Financials",
    "AXISBANK.NS": "Financials", "SBIN.NS": "Financials", "INDUSINDBK.NS": "Financials",
    "BAJFINANCE.NS": "Financials", "BAJAJFINSV.NS": "Financials",
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT", "TECHM.NS": "IT",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "NTPC.NS": "Utilities", "POWERGRID.NS": "Utilities", "COALINDIA.NS": "Energy",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "MARUTI.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto",
    "TATASTEEL.NS": "Metals", "HINDALCO.NS": "Metals", "JSWSTEEL.NS": "Metals",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma",
    "ULTRACEMCO.NS": "Materials", "GRASIM.NS": "Materials", "ASIANPAINT.NS": "Materials",
    "LT.NS": "Industrials", "ADANIPORTS.NS": "Industrials",
    "BHARTIARTL.NS": "Telecom", "TITAN.NS": "Consumer",
}

US_LARGE_CAPS = {
    "AAPL": "Tech", "MSFT": "Tech", "ORCL": "Tech", "INTC": "Tech", "CSCO": "Tech", "IBM": "Tech",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials", "AXP": "Financials",
    "JNJ": "Health", "PFE": "Health", "MRK": "Health", "UNH": "Health", "ABT": "Health", "AMGN": "Health",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "WMT": "Staples", "COST": "Staples",
    "HD": "Discretionary", "MCD": "Discretionary", "NKE": "Discretionary", "AMZN": "Discretionary",
    "SBUX": "Discretionary",
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials", "MMM": "Industrials",
    "UNP": "Industrials", "LMT": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "T": "Telecom", "VZ": "Telecom",
    "LIN": "Materials", "APD": "Materials", "NEM": "Materials",
    "SPG": "RealEstate", "PLD": "RealEstate",
}

UNIVERSES = {
    "etf": {"assets": ETF_MULTI_ASSET, "start": "2008-01-01", "label": "Multi-asset ETFs"},
    "nifty": {"assets": NIFTY_STOCKS, "start": "2011-01-01", "label": "NIFTY 50 stocks"},
    "us": {"assets": US_LARGE_CAPS, "start": "2006-01-01", "label": "US large caps"},
}
