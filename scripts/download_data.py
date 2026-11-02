"""Download adjusted daily close prices for every universe and save them under data/.

Usage:  python scripts/download_data.py
Outputs:
  data/raw/<universe>_prices.csv      wide table of adjusted closes (dates x tickers)
  data/raw/<universe>_groups.csv      ticker -> group map
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp.universes import UNIVERSES  # noqa: E402

END = "2026-10-31"


def fetch(ticker: str, start: str) -> pd.Series:
    for attempt in range(5):
        try:
            df = yf.download(ticker, start=start, end=END, progress=False,
                             auto_adjust=True, threads=False)
            if df is not None and len(df) > 0:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close.rename(ticker)
        except Exception as exc:  # network hiccups / rate limits
            print(f"  {ticker}: attempt {attempt + 1} failed ({exc})")
        time.sleep(3 + 4 * attempt)
    print(f"  {ticker}: giving up")
    return pd.Series(dtype=float, name=ticker)


def main():
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in UNIVERSES.items():
        print(f"[{name}] {len(spec['assets'])} tickers from {spec['start']}")
        series = []
        for t in spec["assets"]:
            s = fetch(t, spec["start"])
            print(f"  {t}: {len(s)} rows")
            series.append(s)
            time.sleep(0.5)
        prices = pd.concat(series, axis=1).sort_index()
        prices.index.name = "Date"
        prices.to_csv(out / f"{name}_prices.csv", float_format="%.6f")
        pd.Series(spec["assets"], name="group").rename_axis("ticker").to_csv(
            out / f"{name}_groups.csv")


if __name__ == "__main__":
    main()
