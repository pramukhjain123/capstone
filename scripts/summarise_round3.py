"""Figures and summary CSVs for round 3: compliance-cost attribution, noise damping,
random mandates, statistical checks and verification."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cahrp.universes import ALL_UNIVERSES  # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FAM = ["shap_asset caps", "shap_group limits", "shap_turnover"]
FAM_LAB = ["asset caps", "group limits", "turnover"]


def compliance():
    d = pd.read_csv(RAW / "compliance_cost.csv", parse_dates=["date"])
    s = d.groupby("universe")[FAM].sum()
    share = s.div(s.sum(axis=1), axis=0)
    share.columns = FAM_LAB
    g = d.groupby("universe").agg(te_mean=("te_ann", "mean"), te_median=("te_ann", "median"),
                                  te_max=("te_ann", "max"),
                                  zero_cost=("fstar", lambda x: float((x < 1e-10).mean())),
                                  turnover_binding=("turnover_dual", lambda x: float((x > 1e-7).mean())),
                                  w_match=("w_match", "max"))
    pd.concat([g, share], axis=1).to_csv(TAB / "compliance_cost.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex="col",
                             gridspec_kw={"height_ratios": [1.2, 1]})
    for j, u in enumerate(["etf", "india_etf", "sector"]):
        e = d[d.universe == u].set_index("date")
        axes[0, j].plot(e.index, e.te_ann * 100, color="black", lw=1.2)
        axes[0, j].set_title(ALL_UNIVERSES[u]["label"])
        axes[0, j].grid(alpha=0.3)
        sh = e[FAM].clip(lower=0)
        tot = sh.sum(axis=1).replace(0, np.nan)
        frac = sh.div(tot, axis=0).fillna(0).rolling(6, min_periods=1).mean()
        axes[1, j].stackplot(frac.index, frac.T.values, labels=FAM_LAB, alpha=0.85)
        axes[1, j].set_ylim(0, 1)
    axes[0, 0].set_ylabel("price of compliance:\nex-ante TE to HRP (% p.a.)")
    axes[1, 0].set_ylabel("Shapley share\n(6-month average)")
    axes[1, 2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "compliance_cost.pdf")
    fig.savefig(FIG / "compliance_cost.png", dpi=150)
    plt.close(fig)


def noise():
    n = pd.read_csv(RAW / "noise_damping.csv")
    agg = n.groupby("universe").agg(
        proj_mean=("mean_ratio_TE_proj", "mean"), proj_max=("ratio_TE_proj", "max"),
        euclid_max=("ratio_euclid", "max"),
        clip_mean_S=("mean_ratio_clip_Snorm", "mean"), clip_max_S=("ratio_clip_Snorm", "max"),
        clip_mean_E=("mean_ratio_clip_euclid", "mean"), clip_max_E=("ratio_clip_euclid", "max"),
        clip_frac_expand=("frac_pairs_clip_expands", "mean"))
    agg.to_csv(TAB / "noise_damping.csv")


def random_mandates():
    r = pd.read_csv(RAW / "random_mandates.csv")
    piv = r.pivot_table(index=["universe", "draw", "placebo"], columns="method", values="Sharpe").reset_index()
    vol = r.pivot_table(index=["universe", "draw", "placebo"], columns="method", values="Vol").reset_index()
    hv = r[r.method == "CA-HRP"].set_index(["universe", "draw"])["HRP_viol"]
    piv["HRP_viol"] = hv.reindex(pd.MultiIndex.from_frame(piv[["universe", "draw"]])).values
    rows = []
    for (u, plc), g in list(piv.groupby(["universe", "placebo"])) + [(("all", "all"), piv)]:
        v = vol.loc[g.index]
        rows.append({"universe": u, "placebo": plc, "draws": len(g),
                     "win_vs_proj": float((g["CA-HRP"] > g["HRP-Proj"]).mean()),
                     "win_vs_clip": float((g["CA-HRP"] > g["HRP-Clip"]).mean()),
                     "win_vs_minvar": float((g["CA-HRP"] > g["MinVar-C"]).mean()),
                     "dSR_proj_mean": float((g["CA-HRP"] - g["HRP-Proj"]).mean()),
                     "dSR_proj_median": float((g["CA-HRP"] - g["HRP-Proj"]).median()),
                     "dSR_minvar_mean": float((g["CA-HRP"] - g["MinVar-C"]).mean()),
                     "lower_vol_than_proj": float((v["CA-HRP"] <= v["HRP-Proj"] + 1e-12).mean()),
                     "clip_viol_mean": float(r[(r.method == "HRP-Clip") & r.draw.isin(g.draw)
                                               & r.universe.isin(g.universe)].ViolRate.mean())})
    pd.DataFrame(rows).to_csv(TAB / "random_mandates.csv", index=False)
    # bind intensity vs gain
    piv["gain"] = piv["CA-HRP"] - piv["HRP-Proj"]
    piv["bin"] = pd.cut(piv.HRP_viol, [-0.01, 0.25, 0.5, 0.75, 1.0])
    piv.groupby("bin", observed=True).agg(draws=("gain", "size"), mean_gain=("gain", "mean"),
                                          win_rate=("gain", lambda x: float((x > 0).mean()))
                                          ).to_csv(TAB / "random_mandates_by_binding.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for plc, mk in [(False, "o"), (True, "x")]:
        g = piv[piv.placebo == plc]
        axes[0].scatter(g.HRP_viol, g.gain, marker=mk, alpha=0.7,
                        label="scrambled groups (placebo)" if plc else "true groups")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("share of rebalances at which HRP breaches the mandate")
    axes[0].set_ylabel("Sharpe(CA-HRP) - Sharpe(HRP-Proj)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].scatter(piv["MinVar-C"], piv["CA-HRP"], c=piv.HRP_viol, cmap="viridis", s=18)
    lim = [piv[["MinVar-C", "CA-HRP"]].min().min(), piv[["MinVar-C", "CA-HRP"]].max().max()]
    axes[1].plot(lim, lim, color="k", lw=0.8)
    axes[1].set_xlabel("Sharpe, constrained minimum variance")
    axes[1].set_ylabel("Sharpe, CA-HRP")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "random_mandates.pdf")
    fig.savefig(FIG / "random_mandates.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    compliance()
    noise()
    if (RAW / "random_mandates.csv").exists():
        random_mandates()
    print("done")
