"""Write LaTeX tables for the paper directly from results/tables/*.csv (no hand-copied numbers)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
UNI = {"etf": "Multi-asset ETFs", "nifty": "NIFTY 50 stocks", "us": "US large caps"}
ORDER = ["EW", "IVP", "MinVar", "ERC", "HRP", "MinVar-C", "MVO-C", "HRP-Clip", "HRP-Proj",
         "CA-HRP-B", "CA-HRP"]
CONSTRAINED = {"MinVar-C", "MVO-C", "HRP-Clip", "HRP-Proj", "CA-HRP-B", "CA-HRP"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def main_table():
    t = pd.read_csv(TAB / "main_results.csv")
    p = pd.read_csv(TAB / "sharpe_pvalues.csv").set_index(["universe", "method"])["p_value"]
    lines = [r"\begin{table*}[!htbp]\centering\small",
             r"\caption{Full-sample out-of-sample results, net of 10\,bps costs. CAGR, Vol, MaxDD, "
             r"CVaR and TE$_{\text{HRP}}$ in \%; turnover annualised; Viol = share of rebalances "
             r"breaching any constraint; $p$ = bootstrap $p$-value of the Sharpe difference with CA-HRP. "
             r"Unconstrained methods (top block) are shown for reference only. Best constrained Sharpe "
             r"ratio in bold.}\label{tab:main}",
             r"\setlength{\tabcolsep}{4.5pt}\begin{tabular}{llrrrrrrrrrr}\toprule",
             r"Universe & Method & CAGR & Vol & Sharpe & $p$ & MaxDD & CVaR$_{5}$ & Turn. & Viol & "
             r"TE$_{\text{HRP}}$ & Eff.\,N\\\midrule"]
    for u in UNI:
        s = t[t.universe == u].set_index("method").loc[ORDER]
        best = s.loc[list(CONSTRAINED), "Sharpe"].idxmax()
        for i, (m, r) in enumerate(s.iterrows()):
            sh = f"{r.Sharpe:.3f}"
            if m == best:
                sh = r"\textbf{" + sh + "}"
            pv = "--" if m == "CA-HRP" else f"{p[(u, m)]:.2f}"
            name = r"\textbf{CA-HRP}" if m == "CA-HRP" else m
            lab = (r"\multirow{11}{*}{\rotatebox{90}{" + UNI[u] + "}}") if i == 0 else ""
            lines.append(f"{lab} & {name} & {pct(r.CAGR)} & {pct(r.Vol)} & {sh} & {pv} & "
                         f"{pct(r.MaxDD)} & {pct(r.CVaR5, 2)} & {r.Turnover:.2f} & {pct(r.ViolRate, 0)}\\% & "
                         f"{pct(r.TE_HRP)} & {r.EffN:.1f}\\\\")
            if m == "HRP":
                lines.append(r"\cmidrule{2-12}")
        lines.append(r"\midrule" if u != "us" else r"\bottomrule")
    lines += [r"\end{tabular}\end{table*}"]
    (OUT / "main.tex").write_text("\n".join(lines))


def test_table():
    t = pd.read_csv(TAB / "test_period.csv")
    full = pd.read_csv(TAB / "main_results.csv")
    ms = ["EW", "HRP", "MinVar-C", "MVO-C", "HRP-Clip", "HRP-Proj", "CA-HRP-B", "CA-HRP"]
    lines = [r"\begin{table}[!htbp]\centering\small",
             r"\caption{Sharpe ratio (net of 10\,bps) on the held-out test period, which was never used "
             r"to choose the CA-HRP design, next to the full sample. Vol and MaxDD (\%) on the test "
             r"period.}\label{tab:test}",
             r"\begin{tabular}{l rrr rrr}\toprule",
             r" & \multicolumn{3}{c}{Test-period Sharpe} & \multicolumn{3}{c}{ETFs, test period}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & ETFs & NIFTY & US & Vol & MaxDD & Full-sample SR\\\midrule"]
    for m in ms:
        row = [t[(t.universe == u) & (t.method == m)].iloc[0] for u in UNI]
        f = full[(full.universe == "etf") & (full.method == m)].iloc[0]
        name = r"\textbf{CA-HRP}" if m == "CA-HRP" else m
        lines.append(f"{name} & {row[0].Sharpe:.3f} & {row[1].Sharpe:.3f} & {row[2].Sharpe:.3f} & "
                     f"{pct(row[0].Vol)} & {pct(row[0].MaxDD)} & {f.Sharpe:.3f}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "test.tex").write_text("\n".join(lines))


def ablation_table():
    a = pd.read_csv(TAB / "ablation.csv")
    order = list(dict.fromkeys(a.method))
    lines = [r"\begin{table}[!htbp]\centering\small",
             r"\caption{Ablation of CA-HRP components (full sample, net of 10\,bps). Each row changes "
             r"one component of the full method.}\label{tab:ablation}",
             r"\begin{tabular}{l rrr rrr}\toprule",
             r" & \multicolumn{3}{c}{Sharpe ratio} & \multicolumn{3}{c}{ETFs}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Variant & ETFs & NIFTY & US & Vol & Turn. & TE$_{\text{HRP}}$\\\midrule"]
    for m in order:
        rows = [a[(a.universe == u) & (a.method == m)].iloc[0] for u in UNI]
        lines.append(f"{m} & {rows[0].Sharpe:.3f} & {rows[1].Sharpe:.3f} & {rows[2].Sharpe:.3f} & "
                     f"{pct(rows[0].Vol)} & {rows[0].Turnover:.2f} & {pct(rows[0].TE_HRP)}\\\\")
        if m == order[0]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "ablation.tex").write_text("\n".join(lines))


def robustness_table():
    c = pd.read_csv(TAB / "cost_sensitivity.csv")
    l5 = pd.read_csv(TAB / "lookback504.csv")
    l1 = pd.read_csv(TAB / "lookback126.csv")
    st = pd.read_csv(TAB / "stability.csv").set_index("method")
    ms = ["EW", "HRP", "MinVar-C", "MVO-C", "HRP-Clip", "HRP-Proj", "CA-HRP"]
    lines = [r"\begin{table*}[!htbp]\centering\small",
             r"\caption{Robustness. Left: Sharpe ratio in the multi-asset universe at 50\,bps costs and "
             r"with 126- and 504-day estimation windows (all universes). Right: weight instability, the "
             r"mean $L_1$ distance of bootstrapped weights from their average (lower = more stable).}"
             r"\label{tab:robust}",
             r"\begin{tabular}{l r rrr rrr rrr}\toprule",
             r" & ETFs & \multicolumn{3}{c}{Window 126 days} & \multicolumn{3}{c}{Window 504 days} & "
             r"\multicolumn{3}{c}{Weight instability}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Method & 50\,bps & ETFs & NIFTY & US & ETFs & NIFTY & US & ETFs & NIFTY & US\\\midrule"]
    for m in ms:
        cc = c[(c.universe == "etf") & (c.method == m) & (c.cost_bps == 50)].Sharpe.iloc[0]
        a = [l1[(l1.universe == u) & (l1.method == m)].Sharpe.iloc[0] for u in UNI]
        b = [l5[(l5.universe == u) & (l5.method == m)].Sharpe.iloc[0] for u in UNI]
        s = [f"{st.loc[m, u]:.3f}" if m in st.index else "--" for u in UNI]
        name = r"\textbf{CA-HRP}" if m == "CA-HRP" else m
        lines.append(f"{name} & {cc:.3f} & " + " & ".join(f"{x:.3f}" for x in a + b) + " & "
                     + " & ".join(s) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    (OUT / "robust.tex").write_text("\n".join(lines))


def stress_table():
    s = pd.read_csv(TAB / "stress.csv")
    ms = ["EW", "HRP", "MinVar-C", "MVO-C", "HRP-Clip", "HRP-Proj", "CA-HRP-B", "CA-HRP"]
    cols = [("etf", "GFC (2008-09)"), ("etf", "COVID (2020)"), ("etf", "Rates shock (2022)"),
            ("nifty", "COVID (2020)"), ("us", "GFC (2008-09)"), ("us", "COVID (2020)"),
            ("us", "Rates shock (2022)")]
    lines = [r"\begin{table}[!htbp]\centering\small",
             r"\caption{Cumulative return (\%) in stress windows: GFC = Sep 2008--Mar 2009, COVID = "
             r"15 Feb--30 Apr 2020, Rates = Jan--Oct 2022.}\label{tab:stress}",
             r"\setlength{\tabcolsep}{4pt}\begin{tabular}{l rrr r rrr}\toprule",
             r" & \multicolumn{3}{c}{ETFs} & NIFTY & \multicolumn{3}{c}{US}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-8}",
             r"Method & GFC & COVID & Rates & COVID & GFC & COVID & Rates\\\midrule"]
    for m in ms:
        vals = []
        for u, per in cols:
            v = s[(s.universe == u) & (s.method == m) & (s.period == per)].Return
            vals.append(pct(v.iloc[0]) if len(v) else "--")
        name = r"\textbf{CA-HRP}" if m == "CA-HRP" else m
        lines.append(f"{name} & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "stress.tex").write_text("\n".join(lines))


def design_table():
    d = pd.read_csv(TAB / "design_study_validation.csv")
    order = list(dict.fromkeys(d.variant))
    lines = [r"\begin{table}[h]\centering\small",
             r"\caption{Design study on the validation period only (net of 10\,bps). ``cap'' = "
             r"capacity-aware bisection; ``tree'' = tree-weighted projection; ``TE'' = tracking-error "
             r"projection. LW covariance unless stated.}\label{tab:design}",
             r"\begin{tabular}{l rrr rr}\toprule",
             r" & \multicolumn{3}{c}{Validation Sharpe} & \multicolumn{2}{c}{ETFs}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Variant & ETFs & NIFTY & US & Vol & Turn.\\\midrule"]
    for m in order:
        rows = [d[(d.universe == u) & (d.variant == m)].iloc[0] for u in UNI]
        lines.append(f"{m} & {rows[0].Sharpe:.3f} & {rows[1].Sharpe:.3f} & {rows[2].Sharpe:.3f} & "
                     f"{pct(rows[0].Vol)} & {rows[0].Turnover:.2f}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (OUT / "design.tex").write_text("\n".join(lines).replace("(=HRP-Proj, LW)", "(LW)"))


if __name__ == "__main__":
    main_table()
    test_table()
    ablation_table()
    robustness_table()
    stress_table()
    design_table()
    print("tables written to", OUT)
