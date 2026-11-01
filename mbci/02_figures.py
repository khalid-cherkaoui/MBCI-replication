"""
02_figures.py -- Figures 1-7 and 9 of the revised article (Elo-Davidson surprise).
(Figure 8, attendance, is produced by the notebook attendance/validation_affluence_mbci.ipynb.)

  fig_distribution_<League>.pdf : Figures 1-5, MBCI*_k by season third (Section 7.2)
  fig_mbci_seasons.pdf         : Figure 6, MBCI_s by league and season (Section 7.3)
  fig_wk_progress.pdf          : Figure 7, mean W_k by season progress t/T (Section 7.4)
  fig_uk_calibration.pdf       : Figure 9, U_k against the market surprise (Section 7.7)

Outputs in outputs/figures/ (or in the folder given as argument).
Copy these PDFs to the Figures/ folder of the LaTeX source if they are regenerated.
Run: python 02_figures.py   (about 30 seconds)
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbci_core as mc, elo_davidson as ed

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else mc.FIG
OUT.mkdir(parents=True, exist_ok=True)
m = mc.load_master(); proc = mc.process(m)
P = ed.league_parameters(proc)
d = ed.index(ed.attach_surprise(proc, ed.run_elo(ed.load_elo_matches(True), P)))
d = mc.add_market(d)
d["terc"] = pd.cut(d.Gameweek / d["T"], [0, 1 / 3, 2 / 3, 1.01], labels=["Early", "Mid", "Late"])
d["tT"] = d.Gameweek / d["T"]

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
def style(ax):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color("#555555")
    ax.tick_params(colors="#333333", length=2.5)

# ---------------- Figures 1-5: distribution by league and tercile
LEAGUES = [("EPL", "EPL"), ("LaLiga", "La Liga"), ("Bundesliga", "Bundesliga"), ("SerieA", "Serie A"), ("Ligue1", "Ligue 1")]
STY = {"Early": dict(color="#86b6ef", ls=":"), "Mid": dict(color="#2a78d6", ls="--"), "Late": dict(color="#0d366b", ls="-")}
BINS = np.arange(0, 0.85 + 1e-9, 0.025)
ymax = 0
for code, _ in LEAGUES:
    for t in STY:
        h, _ = np.histogram(d[(d.League == code) & (d.terc == t)].MBCI_star, bins=BINS, density=True); ymax = max(ymax, h.max())
ymax = np.ceil(ymax * 2) / 2 + 0.25
W1 = 345 / 72 * 0.75
for code, name in LEAGUES:
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.6))
    for t, s in STY.items():
        ax.hist(d[(d.League == code) & (d.terc == t)].MBCI_star, bins=BINS, density=True, histtype="step", lw=1.3, label=t, **s)
    ax.set_xlim(0, 0.85); ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$MBCI^*_k$", labelpad=1); ax.set_ylabel("Density", labelpad=2); style(ax)
    ax.legend([Line2D([0], [0], lw=1.3, **st) for st in STY.values()], list(STY), frameon=False, loc="upper right",
              handlelength=2.4, borderaxespad=0.1, labelspacing=0.3)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / f"fig_distribution_{code}.pdf"); plt.close(fig)

# ---------------- Figure 6 / 7: stakes by progress, seasons
LC = [("EPL", "EPL", "#2a78d6", "o"), ("LaLiga", "La Liga", "#eb6834", "s"), ("Bundesliga", "Bundesliga", "#1baf7a", "^"),
      ("SerieA", "Serie A", "#eda100", "D"), ("Ligue1", "Ligue 1", "#e87ba4", "v")]
W2 = 345 / 72 * 0.85
d["bin"] = pd.cut(d.tT, np.linspace(0, 1, 21), include_lowest=True)
fig, ax = plt.subplots(figsize=(W2, W2 * 0.58))
ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="#777777", label=r"Upper bound $W_k = t/T$", zorder=1)
for code, name, col, mk in LC:
    bb = d[d.League == code].groupby("bin", observed=True).agg(x=("tT", "mean"), y=("W_k", "mean")).dropna()
    ax.plot(bb.x, bb.y, color=col, lw=1.3, marker=mk, ms=3.2, label=name, zorder=2)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xlabel(r"Season progress $t/T$"); ax.set_ylabel(r"Mean $W_k$")
style(ax); ax.legend(frameon=False, loc="upper left", handlelength=2.2)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_wk_progress.pdf"); plt.close(fig)

S = d.groupby(["League", "Season"]).MBCI_star.mean().unstack(0)
seasons = sorted(set(d.Season)); S = S.reindex(seasons)
fig, ax = plt.subplots(figsize=(W2, W2 * 0.62)); x = np.arange(len(seasons))
for code, name, col, mk in LC:
    ax.plot(x, S[code].values, color=col, lw=1.3, marker=mk, ms=3.6, label=name)
ax.set_xticks(x); ax.set_xticklabels([s.replace("-", "/") for s in seasons], rotation=45, ha="right")
ax.set_ylabel(r"$MBCI_s$"); ax.set_xlabel("Season"); style(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5, handlelength=2.0, columnspacing=1.2)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_mbci_seasons.pdf"); plt.close(fig)

# ---------------- Figure 9: calibration of U_k against the betting market
v = d[d.h_mkt.notna()].copy(); v["m_surp"] = 1 - np.exp(-v.surprisal)
OC = [("H", "Home win", "#2a78d6", "o"), ("D", "Draw", "#1baf7a", "s"), ("A", "Away win", "#eb6834", "^")]
fig, ax = plt.subplots(figsize=(W2 * 0.82, W2 * 0.62))
ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color="#777777", label="Perfect agreement", zorder=1)
for o, name, col, mk in OC:
    x_ = v[v.FTR == o].copy(); x_["q"] = pd.qcut(x_.U_k, 10, labels=False)
    g = x_.groupby("q").agg(u=("U_k", "mean"), m=("m_surp", "mean"))
    ax.plot(g.u, g.m, color=col, lw=1.2, marker=mk, ms=3.4, label=name, zorder=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel(r"Outcome surprise $U_k$ (decile means)"); ax.set_ylabel(r"Market surprise $1-\hat p_{o_k}$")
style(ax); ax.legend(frameon=False, loc="upper left", handlelength=2.2)
fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_uk_calibration.pdf"); plt.close(fig)
print("figures ecrites dans", OUT)
for f in sorted(OUT.glob("*.pdf")): print("  ", f.name)
