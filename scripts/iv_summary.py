"""Table and figure for the implied-volatility extension (CA-HRP-IV)."""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp.metrics import perf, sharpe_diff_ci  # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
OUT = ROOT / "paper" / "tables"
VALID_END = {"etf": "2016-12-31"}
STRESS = {"GFC": ("2008-09-01", "2009-03-31"), "COVID": ("2020-02-15", "2020-04-30"),
          "Rates 2022": ("2022-01-01", "2022-10-31")}
LAB = {"etf": "Multi-asset ETFs", "india_etf": "India ETFs"}


def main():
    rows = []
    for u in ["etf", "india_etf"]:
        m6 = pickle.load(open(RAW / f"main6_{u}.pkl", "rb"))
        iv = pickle.load(open(RAW / f"iv_{u}.pkl", "rb"))
        res = {"HRP-Proj": m6["HRP-Proj"], "MinVar-C": m6["MinVar-C"], "MinVar-C-IV": iv["MinVar-C-IV"],
               "CA-HRP": m6["CA-HRP"], "CA-HRP-IV": iv["CA-HRP-IV"]}
        ref = res["CA-HRP-IV"].net(10)
        for m, r in res.items():
            net = r.net(10)
            p = perf(net)
            row = {"universe": u, "method": m, "Sharpe": p["Sharpe"], "Vol": p["Vol"], "MaxDD": p["MaxDD"],
                   "Turnover": r.turnover.iloc[1:].mean() * 12, "ViolRate": r.viol["any"].mean(),
                   "identity_gap": iv["_identity_gap"]}
            row["Sharpe_test"] = perf(net.loc[VALID_END[u]:].iloc[1:])["Sharpe"] if u in VALID_END else np.nan
            for s, (a, b) in STRESS.items():
                seg = net.loc[a:b]
                row[s] = (1 + seg).prod() - 1 if len(seg) > 20 else np.nan
            if m != "CA-HRP-IV":
                dd, lo, hi, _ = sharpe_diff_ci(ref, net.reindex(ref.index).fillna(0))
                row.update({"iv_minus_this": dd, "lo": lo, "hi": hi})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "implied_vol.csv", index=False)

    def f(x, d=1, p=True):
        return "--" if pd.isna(x) else (f"{100 * x:.{d}f}" if p else f"{x:.{d}f}")

    lines = [r"\begin{table*}[!htbp]\centering\small",
             r"\caption{Forward-looking risk with option-implied volatility (net of 10\,bps). CA-HRP-IV "
             r"and MinVar-C-IV use historical correlations and volatilities rescaled by implied/realised "
             r"ratios of VIX, VXN, GVZ, OVX, MOVE and India VIX. Vol, MaxDD and stress returns in \%; "
             r"$\Delta$SR = Sharpe of CA-HRP-IV minus that method, with 95\% block-bootstrap "
             r"interval.}\label{tab:iv}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{ll rrrrr rrr c}\toprule",
             r"Universe & Method & Sharpe & Test SR & Vol & MaxDD & Turn. & GFC & COVID & 2022 & "
             r"$\Delta$SR [95\% CI]\\\midrule"]
    for u in ["etf", "india_etf"]:
        s = df[df.universe == u]
        for i, (_, r) in enumerate(s.iterrows()):
            ci = "--" if r.method == "CA-HRP-IV" else f"{r.iv_minus_this:+.3f} [{r.lo:+.2f}, {r.hi:+.2f}]"
            name = r"\textbf{" + r.method + "}" if r.method == "CA-HRP-IV" else r.method
            first = (r"\multirow{5}{*}{" + LAB[u] + "}") if i == 0 else ""
            lines.append(f"{first} & {name} & {r.Sharpe:.3f} & {f(r.Sharpe_test, 3, False)} & {f(r.Vol)} & "
                         f"{f(r.MaxDD)} & {r.Turnover:.2f} & {f(r.GFC)} & {f(r.COVID)} & "
                         f"{f(r['Rates 2022'])} & {ci}\\\\")
        lines.append(r"\midrule" if u == "etf" else r"\bottomrule")
    lines += [r"\end{tabular}}\end{table*}"]
    (OUT / "iv.tex").write_text("\n".join(lines))

    iv = pickle.load(open(RAW / "iv_etf.pkl", "rb"))["_ratios"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    names = {"^VIX": "VIX (equities)", "^VXN": "VXN (Nasdaq)", "^GVZ": "GVZ (gold)", "^OVX": "OVX (oil)",
             "^MOVE": "MOVE (Treasuries)"}
    for k in iv.columns:
        ax.plot(iv.index, iv[k], lw=1.1, label=names.get(k, k))
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("implied / realised volatility")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "iv_ratios.pdf")
    fig.savefig(FIG / "iv_ratios.png", dpi=150)
    plt.close(fig)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
