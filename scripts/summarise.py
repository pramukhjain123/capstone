"""Turn raw backtest results into the tables and figures used in the paper."""
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
from cahrp.backtest import load_universe, universe_names  # noqa: E402
from cahrp.metrics import perf, sharpe_diff_test, structure  # noqa: E402
from cahrp.universes import UNIVERSES  # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
COST = 10  # bps per unit of two-sided turnover

STRESS = {"GFC (2008-09)": ("2008-09-01", "2009-03-31"),
          "COVID (2020)": ("2020-02-15", "2020-04-30"),
          "Rates shock (2022)": ("2022-01-01", "2022-10-31")}


def load(tag, u):
    with open(RAW / f"{tag}_{u}.pkl", "rb") as f:
        return pickle.load(f)


VALID_END = {"etf": "2016-12-31", "nifty": "2018-12-31", "us": "2015-12-31"}


def summary_table(tag, with_structure=True, period="full"):
    rows = []
    for u in universe_names():
        res = load(tag, u)
        rets, _ = load_universe(u)
        hrp_ref = load("main", u)["HRP"].net(COST)
        for name, r in res.items():
            net = r.net(COST)
            if period == "test":
                net = net.loc[VALID_END[u]:].iloc[1:]
            p = perf(net)
            te = (net - hrp_ref.reindex(net.index)).std() * np.sqrt(252)
            p["TE_HRP"] = te
            p.update({"universe": u, "method": name,
                      "Turnover": r.turnover.iloc[1:].mean() * 12,
                      "ViolRate": r.viol["any"].mean(),
                      "ViolMag": (r.viol["box"] + r.viol["group"] + r.viol["turnover"]).mean()})
            if with_structure:
                p.update(structure(r.weights, rets))
            rows.append(p)
    return pd.DataFrame(rows)


def main():
    main_tab = summary_table("main")
    main_tab.to_csv(TAB / "main_results.csv", index=False)
    summary_table("main", with_structure=False, period="test").to_csv(TAB / "test_period.csv", index=False)

    # Sharpe difference p-values vs CA-HRP
    pv = []
    for u in universe_names():
        res = load("main", u)
        ref = res["CA-HRP"].net(COST)
        for name, r in res.items():
            if name == "CA-HRP":
                continue
            pv.append({"universe": u, "method": name,
                       "p_value": sharpe_diff_test(ref, r.net(COST).reindex(ref.index).fillna(0))})
    pd.DataFrame(pv).to_csv(TAB / "sharpe_pvalues.csv", index=False)

    # cost sensitivity
    rows = []
    for u in universe_names():
        res = load("main", u)
        for name, r in res.items():
            for c in [0, 10, 25, 50]:
                rows.append({"universe": u, "method": name, "cost_bps": c,
                             "Sharpe": perf(r.net(c))["Sharpe"]})
    pd.DataFrame(rows).to_csv(TAB / "cost_sensitivity.csv", index=False)

    # stress periods
    rows = []
    for u in universe_names():
        res = load("main", u)
        for name, r in res.items():
            net = r.net(COST)
            for lab, (a, b) in STRESS.items():
                seg = net.loc[a:b]
                if len(seg) < 20:
                    continue
                rows.append({"universe": u, "method": name, "period": lab,
                             "Return": (1 + seg).prod() - 1,
                             "MaxDD": ((1 + seg).cumprod() / (1 + seg).cumprod().cummax() - 1).min()})
    pd.DataFrame(rows).to_csv(TAB / "stress.csv", index=False)

    # ablations and lookbacks (if present)
    for tag in ["ablation", "lookback504", "lookback126"]:
        if (RAW / f"{tag}_etf.pkl").exists():
            summary_table(tag, with_structure=(tag == "ablation")).to_csv(TAB / f"{tag}.csv", index=False)

    # ---------------- figures ----------------
    show = ["EW", "HRP", "HRP-Clip", "HRP-Proj", "MinVar-C", "MVO-C", "CA-HRP-B", "CA-HRP"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, u in zip(axes, universe_names()):
        res = load("main", u)
        for name in show:
            w = (1 + res[name].net(COST)).cumprod()
            ax.plot(w.index, w.values, lw=2.2 if name == "CA-HRP" else 1.0,
                    color="black" if name == "CA-HRP" else None, label=name)
        ax.set_yscale("log")
        ax.set_title(UNIVERSES[u]["label"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Wealth (log scale), net of 10 bps costs")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "wealth_curves.pdf")
    fig.savefig(FIG / "wealth_curves.png", dpi=150)
    plt.close(fig)

    # group weights through time: HRP vs CA-HRP (ETF)
    rets, groups = load_universe("etf")
    res = load("main", "etf")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, ["HRP", "CA-HRP"]):
        W = res[name].weights
        gw = W.T.groupby(pd.Series(groups, index=W.columns)).sum().T
        ax.stackplot(gw.index, gw.T.values, labels=gw.columns, alpha=0.85)
        ax.axhline(0.40, color="k", ls="--", lw=1)
        ax.set_title(f"{name}: asset-class weights (dashed = 40% cap)")
        ax.set_ylim(0, 1)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "etf_group_weights.pdf")
    fig.savefig(FIG / "etf_group_weights.png", dpi=150)
    plt.close(fig)

    if (RAW / "tightness.csv").exists():
        t = pd.read_csv(RAW / "tightness.csv")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, u in zip(axes, universe_names()):
            s = t[t.universe == u].groupby(["upper_x_n", "method"])["te_hrp"].mean().unstack() * 100
            for m in s.columns:
                ax.plot(s.index, s[m], marker="o", lw=2.2 if m == "CA-HRP" else 1.2,
                        color="black" if m == "CA-HRP" else None, label=m)
            ax.set_xlabel("per-asset cap  (multiples of 1/N)")
            ax.invert_xaxis()
            ax.set_title(UNIVERSES[u]["label"])
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("ex-ante tracking error to HRP (% p.a.)")
        axes[-1].legend()
        fig.tight_layout()
        t.groupby(["universe", "upper_x_n", "method"])[["violation", "te_hrp", "port_vol"]].mean()\
            .unstack().to_csv(TAB / "tightness.csv")
        fig.savefig(FIG / "tightness.pdf")
        fig.savefig(FIG / "tightness.png", dpi=150)
        plt.close(fig)

    if (RAW / "stability.csv").exists():
        s = pd.read_csv(RAW / "stability.csv")
        s.groupby(["universe", "method"])["mean_abs_dev"].mean().unstack(0).to_csv(TAB / "stability.csv")


if __name__ == "__main__":
    main()
