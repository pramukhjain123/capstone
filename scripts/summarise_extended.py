"""Tables and figures for the extended experiments (six universes, CA-HRP-R, frequency,
constraint decomposition, Monte Carlo, runtime)."""
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
from cahrp.backtest import all_universe_names, load_universe  # noqa: E402
from cahrp.metrics import perf, sharpe_diff_ci, structure  # noqa: E402
from cahrp.universes import ALL_UNIVERSES  # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
COST = 10
VALID_END = {"etf": "2016-12-31", "nifty": "2018-12-31", "us": "2015-12-31"}
STRESS = {"GFC (2008-09)": ("2008-09-01", "2009-03-31"),
          "COVID (2020)": ("2020-02-15", "2020-04-30"),
          "Rates shock (2022)": ("2022-01-01", "2022-10-31")}
CORE = ["EW", "HRP", "MinVar-C", "MVO-C", "HRP-Clip", "HRP-Proj", "CA-HRP", "CA-HRP-R"]


def load(name):
    with open(RAW / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main6():
    rows, ci_rows, stress_rows, test_rows = [], [], [], []
    for u in all_universe_names():
        res = load(f"main6_{u}")
        rets, _ = load_universe(u)
        hrp_ref = res["HRP"].net(COST)
        for m, r in res.items():
            net = r.net(COST)
            p = perf(net)
            p.update(structure(r.weights, rets))
            p.update({"universe": u, "method": m, "Turnover": r.turnover.iloc[1:].mean() * 12,
                      "ViolRate": r.viol["any"].mean(),
                      "TE_HRP": (net - hrp_ref.reindex(net.index)).std() * np.sqrt(252),
                      "start": net.index.min().date(), "end": net.index.max().date()})
            rows.append(p)
            for lab, (a, b) in STRESS.items():
                seg = net.loc[a:b]
                if len(seg) > 20:
                    stress_rows.append({"universe": u, "method": m, "period": lab,
                                        "Return": (1 + seg).prod() - 1})
            if u in VALID_END:
                tnet = net.loc[VALID_END[u]:].iloc[1:]
                test_rows.append({"universe": u, "method": m, "Sharpe": perf(tnet)["Sharpe"],
                                  "Vol": perf(tnet)["Vol"], "MaxDD": perf(tnet)["MaxDD"]})
        ca = res["CA-HRP"].net(COST)
        for other in ["HRP-Proj", "HRP-Clip", "MinVar-C", "EW", "CA-HRP-R"]:
            o = res[other].net(COST).reindex(ca.index).fillna(0)
            d, lo, hi, pp = sharpe_diff_ci(ca, o)
            ci_rows.append({"universe": u, "vs": other, "diff": d, "lo": lo, "hi": hi, "p_pos": pp})
    pd.DataFrame(rows).to_csv(TAB / "main6_results.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(TAB / "main6_sharpe_ci.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(TAB / "main6_stress.csv", index=False)
    pd.DataFrame(test_rows).to_csv(TAB / "main6_test_period.csv", index=False)

    # figure: Sharpe difference CIs (CA-HRP minus competitor) across six universes
    ci = pd.DataFrame(ci_rows)
    comps = ["HRP-Proj", "HRP-Clip", "MinVar-C", "EW"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    us = all_universe_names()
    for j, comp in enumerate(comps):
        s = ci[ci.vs == comp].set_index("universe").loc[us]
        x = np.arange(len(us)) + (j - 1.5) * 0.18
        ax.errorbar(x, s["diff"], yerr=[s["diff"] - s["lo"], s["hi"] - s["diff"]], fmt="o", capsize=3,
                    label=f"CA-HRP minus {comp}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(us)))
    ax.set_xticklabels([ALL_UNIVERSES[u]["label"].replace(" ", "\n", 1) for u in us], fontsize=8)
    ax.set_ylabel("Sharpe ratio difference (95% CI)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "sharpe_ci.pdf")
    fig.savefig(FIG / "sharpe_ci.png", dpi=150)
    plt.close(fig)

    # wealth curves for the external universes
    ext = ["india_etf", "sector", "country"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, u in zip(axes, ext):
        res = load(f"main6_{u}")
        for m in ["EW", "HRP", "HRP-Clip", "HRP-Proj", "MinVar-C", "CA-HRP", "CA-HRP-R"]:
            w = (1 + res[m].net(COST)).cumprod()
            ax.plot(w.index, w.values, lw=2.2 if m == "CA-HRP" else 1.0,
                    color="black" if m == "CA-HRP" else None,
                    ls="--" if m == "CA-HRP-R" else "-", label=m)
        ax.set_yscale("log")
        ax.set_title(ALL_UNIVERSES[u]["label"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Wealth (log), net of 10 bps")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "wealth_external.pdf")
    fig.savefig(FIG / "wealth_external.png", dpi=150)
    plt.close(fig)


def frequency():
    rows = []
    for freq in ["W", "Q"]:
        for u in all_universe_names():
            res = load(f"freq{freq}_{u}")
            for m, r in res.items():
                p = perf(r.net(COST))
                rows.append({"freq": freq, "universe": u, "method": m, "Sharpe": p["Sharpe"],
                             "Vol": p["Vol"], "ViolRate": r.viol["any"].mean(),
                             "Turnover": r.turnover.iloc[1:].sum() / (len(r.gross) / 252)})
    # monthly from main6
    for u in all_universe_names():
        res = load(f"main6_{u}")
        for m in CORE:
            r = res[m]
            p = perf(r.net(COST))
            rows.append({"freq": "M", "universe": u, "method": m, "Sharpe": p["Sharpe"], "Vol": p["Vol"],
                         "ViolRate": r.viol["any"].mean(),
                         "Turnover": r.turnover.iloc[1:].sum() / (len(r.gross) / 252)})
    pd.DataFrame(rows).to_csv(TAB / "frequency.csv", index=False)


def decompose():
    rows = []
    for u in all_universe_names():
        res = load(f"decompose_{u}")
        for (sname, m), r in res.items():
            p = perf(r.net(COST))
            rows.append({"universe": u, "subset": sname, "method": m, "Sharpe": p["Sharpe"],
                         "Vol": p["Vol"], "MaxDD": p["MaxDD"], "ViolRate": r.viol["any"].mean()})
    pd.DataFrame(rows).to_csv(TAB / "decomposition.csv", index=False)


def montecarlo():
    df = pd.read_csv(RAW / "montecarlo.csv")
    g = df.groupby(["dist", "T", "method"]).agg(
        true_vol=("true_vol", "mean"), true_te_hrp=("true_te_hrp", "mean"),
        true_sharpe=("true_sharpe", "mean"), sharpe_sd=("true_sharpe", "std"),
        viol_rate=("violation", lambda x: float((x > 1e-6).mean())), eff_n=("eff_n", "mean")).reset_index()
    g.to_csv(TAB / "montecarlo.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, lab in [(axes[0], "true_te_hrp", "true tracking error to true-covariance HRP (% p.a.)"),
                         (axes[1], "true_vol", "true portfolio volatility (% p.a.)")]:
        s = g[g.dist == "t5"].pivot(index="T", columns="method", values=col) * 100
        for m in s.columns:
            ax.plot(s.index, s[m], marker="o", lw=2.2 if m == "CA-HRP" else 1.2,
                    color="black" if m == "CA-HRP" else None, ls="--" if m == "CA-HRP-R" else "-", label=m)
        ax.set_xlabel("estimation window T (days)")
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "montecarlo.pdf")
    fig.savefig(FIG / "montecarlo.png", dpi=150)
    plt.close(fig)


def runtime():
    df = pd.read_csv(RAW / "runtime.csv")
    df.to_csv(TAB / "runtime.csv", index=False)
    s = df.pivot(index="N", columns="method", values="seconds") * 1000
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in s.columns:
        ax.plot(s.index, s[m], marker="o", lw=2.2 if m == "CA-HRP" else 1.2,
                color="black" if m == "CA-HRP" else None, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of assets N")
    ax.set_ylabel("time per rebalance (ms)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "runtime.pdf")
    fig.savefig(FIG / "runtime.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main6()
    frequency()
    decompose()
    montecarlo()
    runtime()
    print("done")
