"""
01_article_numbers.py -- every number and table of the revised article (Elo-Davidson surprise).

Computes, in the order of the article: parameters (Sections 4 and 7), Table 3, Table 4,
Section 7.2 (by league and by season third), Table 5 and Figure 6 (text), Section 7.4,
Section 7.5 (HHI), statistical uncertainty (bootstrap and permutations, Sections 7.2, 7.3, 7.5),
Section 7.7 (validation against the betting market), Section 7.8 and Table 7.

Outputs in outputs/:
  article_numbers_report.txt   the full log (what is printed on screen)
  article_numbers_check.csv    automatic check: each number of the article compared with the computed value
  numbers.json                 all computed numbers
  mbci_match_level.xlsx        one match per row (components, Elo-Davidson probabilities, market odds)
  mbci_season_level.xlsx       one league-season per row (MBCI_s, components, HHI)
  table3_descriptive.csv, table4_top_matches.csv, table5_leagues.csv,
  table7_robustness.csv, grid_alpha_lambda_c.csv, validation_calibration.csv,
  uncertainty_league_means.csv, uncertainty_league_pairs.csv, uncertainty_terciles.csv (bootstrap, permutations)

Run: python 01_article_numbers.py   (about 2 minutes)
"""
import sys, json, itertools, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind, ks_2samp, linregress
import statsmodels.formula.api as smf
import mbci_core as mc, elo_davidson as ed

OUT = mc.OUT; OUT.mkdir(parents=True, exist_ok=True)


class Tee:                                            # screen + log file
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8"); self.s = sys.stdout
    def write(self, x):
        self.s.write(x); self.f.write(x)
    def flush(self):
        self.s.flush(); self.f.flush()
sys.stdout = Tee(OUT / "article_numbers_report.txt")

N = {}
def put(k, v): N[k] = v if not isinstance(v, (np.floating, np.integer)) else v.item()
pd.set_option("display.width", 240)
ABBR = {"Bundesliga": "BL", "Ligue1": "L1", "EPL": "EPL", "SerieA": "SA", "LaLiga": "LL"}

# ======================================================================== data and index
m = mc.load_master(); H = mc.hhi_table(m)
base_proc = mc.process(m)                                       # standings and stakes W_k
P = ed.league_parameters(base_proc)                             # nu_L and h_L (equation 8)
ELO = ed.run_elo(ed.load_elo_matches(True), P)                  # Elo-Davidson ratings, 2014/15 warm-up season
ELO0 = ed.run_elo(ed.load_elo_matches(False), P)                # variant without the warm-up season
dproc = ed.attach_surprise(base_proc, ELO)                      # U_k = 1 - P(observed outcome)
b = ed.index(dproc)                                             # baseline MBCI*_k (alpha = 0.5, lambda = 1, c = 0.5)
b = mc.add_market(b)
b["tg"] = b.FTHG + b.FTAG; b["gdiff"] = (b.FTHG - b.FTAG).abs()
b = mc.add_tercile(b)
E, M_, L_ = b.terc == "early", b.terc == "mid", b.terc == "late"
LG = mc.LEAGUES
print("Matchs analyses :", len(b), "| ligues-saisons :", b.groupby(["League", "Season"]).ngroups,
      "| matchs avec cotes :", int(b.h_mkt.notna().sum()))
print("Poids AHP :", mc.ahp_weights())
print("League parameters:\n", P.round(3))
put("params", P.round(4).to_dict("index"))
put("n_elo", len(ELO)); put("n_warmup", int((ed.load_elo_matches(True).Season == ed.WARMUP).sum()))

# ======================================================================== Table 3
t3 = b[["tg", "gdiff", "G_k", "U_k", "W_k", "MBCI_star"]].agg(["mean", "std", "min", "median", "max"]).T
print("\nTABLE 3\n", t3.round(4)); put("t3", t3.round(4).to_dict("index")); put("mu", b.mu.iloc[0])
put("U_by_outcome", b.groupby("FTR").U_k.mean().round(3).to_dict())
print("U_k moyen par issue :", N["U_by_outcome"])
t3.rename(index={"tg": "Total goals", "gdiff": "Goal difference", "G_k": "Match intensity G_k", "U_k": "Outcome surprise U_k",
                 "W_k": "Stakes weight W_k", "MBCI_star": "Normalised index MBCI*_k"}).round(3).to_csv(OUT / "table3_descriptive.csv")

# ======================================================================== Table 4
cols = ["League", "Season", "Gameweek", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "rank_home", "rank_away", "P_H", "P_N", "P_A", "G_k", "U_k", "W_k", "W_cat", "MBCI_star"]
top = b.sort_values("MBCI_star", ascending=False)
print("\nTOP 10\n", top.head(10)[cols].round(3).to_string(index=False))
top.head(10)[cols].round(3).to_csv(OUT / "table4_top_matches.csv", index=False)
put("top", top.head(10).MBCI_star.round(4).tolist())
put("top10_relg", int((top.head(10).W_cat == "relg").sum())); put("top10_draws", int((top.head(10).FTR == "D").sum()))
put("top10_fav_not_win", int((((top.head(10).FTR == "H") & (top.head(10).P_H < top.head(10).P_A)) | ((top.head(10).FTR == "A") & (top.head(10).P_A < top.head(10).P_H)) | (top.head(10).FTR == "D")).sum()))
print("top10: relg", N["top10_relg"], "draws", N["top10_draws"], "favourite did not win", N["top10_fav_not_win"])

# ======================================================================== 7.2 by league and by season third
mm = b.groupby(["League", "terc"], observed=True).MBCI_star.mean().unstack()
sdl = b.groupby(["League", "terc"], observed=True).MBCI_star.std().unstack()
gt5 = b.assign(x=b.MBCI_star > 0.5).groupby(["League", "terc"], observed=True).x.mean().unstack() * 100
lt2 = b.assign(x=b.MBCI_star < 0.2).groupby(["League", "terc"], observed=True).x.mean().unstack() * 100
Ug = b.groupby(["League", "terc"], observed=True).U_k.mean().unstack()
Gg = b.groupby(["League", "terc"], observed=True).G_k.mean().unstack()
Wg = b.groupby(["League", "terc"], observed=True).W_k.mean().unstack()
gLE, gME, gLM = 100 * (mm.late / mm.early - 1), 100 * (mm.mid / mm.early - 1), 100 * (mm.late / mm.mid - 1)
print("\nMEANS\n", mm.round(3), "\nSD\n", sdl.round(3), "\n>0.5 %\n", gt5.round(2), "\n<0.2 %\n", lt2.round(2))
print("U\n", Ug.round(3), "\nG\n", Gg.round(3), "\nW\n", Wg.round(3))
print("gain L/E", gLE.round(1).to_dict(), "\ngain M/E", gME.round(1).to_dict(), "\ngain L/M", gLM.round(1).to_dict())
ks_el = {L: ks_2samp(b[(b.League == L) & E].MBCI_star, b[(b.League == L) & L_].MBCI_star) for L in LG}
print("KS early-late D", {L: round(v[0], 3) for L, v in ks_el.items()})
sd_all = b.groupby("terc", observed=True).MBCI_star.std(); print("overall sd by tercile", sd_all.round(4).to_dict())
z = b.tg == 0
low = b.MBCI_star < 0.2
print("MBCI < 0.2 : %.2f%% des matchs | dont victoires du favori %.0f%% | au plus un but %.0f%% | 1er tiers %.0f%%" % (
    100 * low.mean(), 100 * (((b.FTR == "H") & (b.P_H > b.P_A)) | ((b.FTR == "A") & (b.P_A > b.P_H)))[low].mean(),
    100 * (b.tg <= 1)[low].mean(), 100 * E[low].mean()))
print("0-0 : MBCI moyen %.3f" % b.MBCI_star[z].mean())
bind_late = b[L_].groupby("League").W_cat.value_counts(normalize=True).unstack() * 100
print("enjeu determinant, dernier tiers (%):\n", bind_late.round(1))
put("by_league_terc", dict(mean=mm.round(4).to_dict("index"), sd=sdl.round(4).to_dict("index"), gt05=gt5.round(2).to_dict("index"),
                           lt02=lt2.round(2).to_dict("index"), U=Ug.round(4).to_dict("index"), G=Gg.round(4).to_dict("index"), W=Wg.round(4).to_dict("index")))
put("gains", dict(LE=gLE.round(2).to_dict(), ME=gME.round(2).to_dict(), LM=gLM.round(2).to_dict()))
put("ks_D", {L: v[0] for L, v in ks_el.items()}); put("sd_terc", sd_all.round(4).to_dict())
favwin = ((b.FTR == "H") & (b.P_H > b.P_A)) | ((b.FTR == "A") & (b.P_A > b.P_H))
put("low", dict(share=100 * low.mean(), fav=100 * favwin[low].mean(), tg_le1=100 * (b.tg <= 1)[low].mean(), early=100 * E[low].mean()))
put("mean_00", b.MBCI_star[z].mean()); put("bind_late", bind_late.round(2).to_dict("index"))
for L in LG:
    t2 = b[b.League == L].sort_values("MBCI_star", ascending=False).head(2)
    print(L, t2[["Season", "Gameweek", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "rank_home", "rank_away", "W_cat", "MBCI_star"]].round(3).values.tolist())
sa_best = b[b.League == "SerieA"].MBCI_star.max(); print("SerieA best", round(sa_best, 3), "overall rank", int((b.MBCI_star > sa_best).sum()) + 1)
put("sa_best_rank", int((b.MBCI_star > sa_best).sum()) + 1)

# ======================================================================== Table 5 and Figure 6
S = b.groupby(["League", "Season"]).agg(MBCI_s=("MBCI_star", "mean"), G=("G_k", "mean"), U=("U_k", "mean"), W=("W_k", "mean")).reset_index()
t5 = S.groupby("League").agg(mean=("MBCI_s", "mean"), sd=("MBCI_s", "std"), G=("G", "mean"), U=("U", "mean"), W=("W", "mean")).sort_values("mean", ascending=False)
t5["goals"] = b.groupby("League").tg.mean()
print("\nTABLE 5\n", t5.round(4)); put("t5", t5.round(4).to_dict("index"))
t5.round(4).to_csv(OUT / "table5_leagues.csv")
order = list(t5.index)
welch = {}
for a_, c_ in itertools.combinations(order, 2):
    t_, p_ = ttest_ind(S[S.League == a_].MBCI_s, S[S.League == c_].MBCI_s, equal_var=False); welch[f"{a_}-{c_}"] = (round(t_, 2), round(p_, 4))
print("Welch", welch); put("welch", welch)
Sw = S.pivot(index="Season", columns="League", values="MBCI_s")
first = Sw.idxmax(axis=1); print("first by season", first.to_dict())
st = Sw.stack(); print("max", st.idxmax(), round(st.max(), 4), "| min", st.idxmin(), round(st.min(), 4))
span = Sw.max() - Sw.min(); print("range per league", span.round(4).to_dict())
o4 = t5["mean"].drop(order[0]); print("max gap among other four", round(o4.max() - o4.min(), 4))
tr = {}
for L in Sw.columns:
    y = Sw[L].dropna(); lr = linregress([int(s[:4]) for s in y.index], y.values); tr[L] = (lr.slope * 1e3, lr.pvalue)
print("trends (x1e-3, p)", {k: (round(v[0], 2), round(v[1], 3)) for k, v in tr.items()})
cc = Sw.corr(); cu = cc.values[np.triu_indices(5, 1)]
print("pairwise corr season values", round(cu.min(), 2), round(cu.max(), 2), "| L1-SA", round(cc.loc["Ligue1", "SerieA"], 2))
put("fig6", dict(first=first.to_dict(), max=[list(st.idxmax()), st.max()], min=[list(st.idxmin()), st.min()], span=span.round(4).to_dict(),
                 gap4=o4.max() - o4.min(), trends={k: [v[0], v[1]] for k, v in tr.items()}, corr=[cu.min(), cu.max()],
                 corr_L1_SA=cc.loc["Ligue1", "SerieA"]))

# ======================================================================== 7.4 season thirds
tt = b.groupby("terc", observed=True)[["MBCI_star", "G_k", "U_k", "W_k", "W_raw"]].mean()
print("\nTERCILES\n", tt.round(4))
nb = ed.index(dproc, use_tT=False); tn = mc.add_tercile(nb).groupby("terc", observed=True).MBCI_star.mean()
print("no t/T terciles", tn.round(3).to_dict(), "change %.1f%%" % (100 * (tn["late"] / tn["early"] - 1)))
put("terciles", tt.round(4).to_dict("index")); put("no_tT_change", 100 * (tn["late"] / tn["early"] - 1))
b["tT"] = b.Gameweek / b["T"]
put("maxsig_tT_le_02", b.W_raw[b.tT <= 0.2].mean()); put("maxsig_04_06", b.W_raw[(b.tT > 0.4) & (b.tT <= 0.6)].mean())
fin = b.Gameweek == b["T"]
put("final_round", dict(W0=100 * (b.W_k[fin] == 0).mean(), W09=100 * (b.W_k[fin] >= 0.9).mean()))
print("max sigma t/T<=0.2 : %.2f | 0.4<t/T<=0.6 : %.2f | derniere journee : W=0 %.0f%%, W>=0.9 %.0f%%" % (
    N["maxsig_tT_le_02"], N["maxsig_04_06"], N["final_round"]["W0"], N["final_round"]["W09"]))

# ======================================================================== 7.5 HHI
sb = mc.season_level(b)
hp = mc.hhi_corr(sb, H, "HHI_pts"); hw = mc.hhi_corr(sb, H, "HHI_wins")
print("\nHHI pts", {k: round(v, 3) for k, v in hp.items()}, "\nHHI wins", {k: round(v, 3) for k, v in hw.items()})
x = S.merge(H, on=["League", "Season"])
comp = {c_: pearsonr(x[c_], -x.HHI_pts) for c_ in ["G", "U", "W"]}
print("components vs HHI-:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in comp.items()})
a0 = mc.season_level(ed.index(dproc, alpha=0)); a1 = mc.season_level(ed.index(dproc, alpha=1))
r_a0 = mc.hhi_corr(a0, H, "HHI_pts"); r_a1 = mc.hhi_corr(a1, H, "HHI_pts")
print("alpha=0 HHI r %.3f p %.3f | alpha=1 r %.3f p %.3f" % (r_a0["r"], r_a0["p"], r_a1["r"], r_a1["p"]))
put("hhi", dict(pts=hp, wins=hw, comp={k: [v[0], v[1]] for k, v in comp.items()}, a0=r_a0, a1=r_a1))
x.to_excel(OUT / "mbci_season_level.xlsx", index=False)

# ======================================================================== statistical uncertainty (7.2, 7.3, 7.5)
# Resampling units: seasons (Table 5, HHI) and league-seasons (season thirds),
# because matches of the same season are not independent. Fixed seeds: reproducible results.
B_BOOT, B_KS = 10000, 2000
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); k = len(p); adj = np.empty(k); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (k - i) * p[j])); adj[j] = run
    return adj
def perm_exact(ya, yc):
    """Exact permutation test of the difference in means (all allocations of the seasons)."""
    z = np.concatenate([ya, yc]); na = len(ya)
    cb = np.array(list(itertools.combinations(range(len(z)), na)))
    sa = z[cb].sum(1); d = sa / na - (z.sum() - sa) / (len(z) - na)
    return float(np.mean(np.abs(d) >= abs(ya.mean() - yc.mean()) - 1e-12))
def hedges_g(ya, yc):
    na, nc = len(ya), len(yc)
    sp = np.sqrt(((na - 1) * ya.var(ddof=1) + (nc - 1) * yc.var(ddof=1)) / (na + nc - 2))
    return (ya.mean() - yc.mean()) / sp * (1 - 3 / (4 * (na + nc) - 9))
rng = np.random.default_rng(2026)
ci_mean = {}
for L in order:                                                   # bootstrap CIs of the Table 5 means
    y = S[S.League == L].MBCI_s.values
    bs = rng.choice(y, (B_BOOT, len(y))).mean(1); ci_mean[L] = np.percentile(bs, [2.5, 97.5])
pairs = []
for a_, c_ in itertools.combinations(order, 2):                  # 10 pairwise comparisons
    ya, yc = S[S.League == a_].MBCI_s.values, S[S.League == c_].MBCI_s.values
    bs = rng.choice(ya, (B_BOOT, len(ya))).mean(1) - rng.choice(yc, (B_BOOT, len(yc))).mean(1)
    dpair = (Sw[a_] - Sw[c_]).dropna().values                    # paired variant by season (sign flips)
    flips = np.array(list(itertools.product([1, -1], repeat=len(dpair))))
    p_pair = float(np.mean(np.abs((flips * dpair).mean(1)) >= abs(dpair.mean()) - 1e-12))
    pairs.append(dict(pair=f"{a_}-{c_}", diff=ya.mean() - yc.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                      g=hedges_g(ya, yc), p_perm=perm_exact(ya, yc), p_paired=p_pair))
PR = pd.DataFrame(pairs); PR["p_holm"] = holm(PR.p_perm); PR["p_paired_holm"] = holm(PR.p_paired)
print("\nINCERTITUDE : moyennes par ligue (IC bootstrap 95 %, saisons reechantillonnees)")
print({L: [round(t5.loc[L, "mean"], 4)] + list(np.round(v, 4)) for L, v in ci_mean.items()})
print(PR.round(4).to_string(index=False))
ks_boot = {}
rng = np.random.default_rng(2027)
for L in LG:                                                      # season thirds: bootstrap by league-season
    d_ = b[b.League == L]; seas = d_.Season.unique()
    gg = {s_: (d_[(d_.Season == s_) & E[d_.index]].MBCI_star.values, d_[(d_.Season == s_) & L_[d_.index]].MBCI_star.values) for s_ in seas}
    Ds, chs = [], []
    for _ in range(B_KS):
        dr = rng.choice(seas, len(seas))
        e_ = np.concatenate([gg[s_][0] for s_ in dr]); l_ = np.concatenate([gg[s_][1] for s_ in dr])
        Ds.append(ks_2samp(e_, l_).statistic); chs.append(100 * (l_.mean() / e_.mean() - 1))
    ks_boot[L] = dict(D=ks_el[L][0], D_lo=np.percentile(Ds, 2.5), D_hi=np.percentile(Ds, 97.5),
                      gain=gLE[L], gain_lo=np.percentile(chs, 2.5), gain_hi=np.percentile(chs, 97.5))
KB = pd.DataFrame(ks_boot).T
print("\nKS et hausse 1er -> 3e tiers, IC bootstrap par ligue-saison\n", KB.round(3))
xh = S.merge(H, on=["League", "Season"]); xh["h"] = -xh.HHI_pts   # relationship with the HHI: panel structure
rx = smf.ols("MBCI_s ~ C(League) + C(Season)", xh).fit().resid; rh = smf.ols("h ~ C(League) + C(Season)", xh).fit().resid
r_2w = float(np.corrcoef(rx, rh)[0, 1]); df_2w = len(xh) - 1 - xh.League.nunique() - xh.Season.nunique() + 1
from scipy.stats import t as tdist
p_2w = float(2 * (1 - tdist.cdf(abs(r_2w) * np.sqrt(df_2w / (1 - r_2w ** 2)), df_2w)))
rng = np.random.default_rng(2028); seas = xh.Season.unique(); idx = {s_: np.where(xh.Season.values == s_)[0] for s_ in seas}
r_bs = []
for _ in range(B_BOOT):
    ii = np.concatenate([idx[s_] for s_ in rng.choice(seas, len(seas))])
    r_bs.append(np.corrcoef(xh.MBCI_s.values[ii], xh.h.values[ii])[0, 1])
hhi_boot = dict(r_2w=r_2w, p_2w=p_2w, df_2w=df_2w, lo_season=float(np.percentile(r_bs, 2.5)), hi_season=float(np.percentile(r_bs, 97.5)))
print("HHI : r apres effets fixes ligue et saison %.3f (p = %.3f, ddl %d) | IC bootstrap par saison du r groupe [%.3f, %.3f]" % (
    r_2w, p_2w, df_2w, hhi_boot["lo_season"], hhi_boot["hi_season"]))
put("uncertainty", dict(ci_mean={k: list(v) for k, v in ci_mean.items()}, pairs=PR.to_dict("records"),
                        ks=KB.to_dict("index"), hhi=hhi_boot, B=B_BOOT, B_ks=B_KS))
pd.DataFrame({L: dict(mean=t5.loc[L, "mean"], ci_lo=v[0], ci_hi=v[1]) for L, v in ci_mean.items()}).T.round(4).to_csv(OUT / "uncertainty_league_means.csv")
PR.round(5).to_csv(OUT / "uncertainty_league_pairs.csv", index=False)
KB.round(4).to_csv(OUT / "uncertainty_terciles.csv")

# ======================================================================== 7.7 validation against the betting market
v = b[b.h_mkt.notna()].copy(); v["LS"] = v.League + v.Season
r_ = lambda col, mm_: np.corrcoef(v.loc[mm_, col], v.loc[mm_, "surprisal"])[0, 1]
allm, dec, drw = v.FTR.notna(), v.FTR != "D", v.FTR == "D"
v["U_rank"] = mc.add_index_rank(base_proc).set_index("MID").loc[v.MID, "U_k"].values
val = {col: dict(all=r_(col, allm), dec=r_(col, dec), draw=r_(col, drw)) for col in ["U_k", "U_rank"]}
print("\nVALIDATION corr", {k: {kk: round(vv, 3) for kk, vv in d_.items()} for k, d_ in val.items()})
vd = v[v.FTR != "D"].copy(); vd["away_win"] = (vd.FTR == "A").astype(int)
reg = {}
for col in ["U_k", "U_rank"]:
    a = smf.ols(f"surprisal ~ {col}", vd).fit(cov_type="cluster", cov_kwds={"groups": vd.LS})
    bb = smf.ols(f"surprisal ~ {col} + away_win", vd).fit(cov_type="cluster", cov_kwds={"groups": vd.LS})
    c = smf.ols(f"surprisal ~ {col}", v).fit(cov_type="cluster", cov_kwds={"groups": v.LS})
    reg[col] = dict(dec_b=a.params[col], dec_t=a.tvalues[col], dec_R2=a.rsquared, away_R2=bb.rsquared, away_b=bb.params["away_win"], away_t=bb.tvalues["away_win"],
                    all_b=c.params[col], all_t=c.tvalues[col], all_R2=c.rsquared)
    print(col, {k: round(vv, 3) for k, vv in reg[col].items()})
put("validation", dict(corr=val, reg=reg, n=len(v), n_dec=len(vd)))
fh = v.P_H >= v.P_A
v["E_fav"] = np.where(fh, v.P_H + .5 * v.P_N, v.P_A + .5 * v.P_N)
v["bin"] = pd.cut(v.E_fav, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
cal = pd.DataFrame({"n": v.groupby("bin", observed=True).size(),
                    "m_fav": v.assign(x=1 - np.where(fh, v.pH, v.pA)).groupby("bin", observed=True).x.mean(),
                    "m_draw": v.assign(x=1 - v.pD).groupby("bin", observed=True).x.mean(),
                    "m_dog": v.assign(x=1 - np.where(fh, v.pA, v.pH)).groupby("bin", observed=True).x.mean(),
                    "u_fav": v.assign(x=1 - np.where(fh, v.P_H, v.P_A)).groupby("bin", observed=True).x.mean(),
                    "u_draw": v.assign(x=1 - v.P_N).groupby("bin", observed=True).x.mean(),
                    "u_dog": v.assign(x=1 - np.where(fh, v.P_A, v.P_H)).groupby("bin", observed=True).x.mean()})
print(cal.round(3)); put("calibration", {str(k): r.to_dict() for k, r in cal.round(4).iterrows()})
cal.round(4).to_csv(OUT / "validation_calibration.csv")
put("logloss", dict(model=float(-np.log(1 - v.U_k).mean()), market=float(v.surprisal.mean())))
print("logloss model %.4f market %.4f" % (N["logloss"]["model"], N["logloss"]["market"]))
hm = {c_: float(v[c_].corr(v.h_mkt)) for c_ in ["MBCI_star", "G_k", "U_k", "W_k"]}; print("corr with H_mkt", {k: round(x_, 3) for k, x_ in hm.items()}); put("hmkt", hm)

# ======================================================================== 7.8 robustness and Table 7
def summary(d):
    s_ = mc.season_level(d); lr = mc.league_rank(s_); hc = mc.hhi_corr(s_, H, "HHI_pts"); tc = mc.terciles(d)
    return dict(ranking=" > ".join(ABBR[k] for k in lr.index), r_hhi=hc["r"], p_hhi=hc["p"],
                gain=100 * (tc.loc["late", "MBCI_star"] / tc.loc["early", "MBCI_star"] - 1), means=lr.round(4).to_dict())
VAR = [("baseline", "Baseline (alpha = 0.5, lambda = 1, c = 0.5)", lambda: b),
       ("ties_avg", "Tied clubs at average rank", lambda: ed.index(ed.attach_surprise(mc.process(m, rank_mode="avg"), ELO))),
       ("calendar", "Calendar-based standings", lambda: ed.index(ed.attach_surprise(cal_proc, ELO))),
       ("eps_unant", "European Performance Spots unanticipated", lambda: ed.index(ed.attach_surprise(eps_proc, ELO))),
       ("w_equal", "Equal European weights (w_b = 1)", lambda: ed.index(ed.attach_surprise(mc.process(m, w=mc.WTS_EQUAL), ELO))),
       ("w_kg", "Prestige weights of Kringstad and Gerrard", lambda: ed.index(ed.attach_surprise(mc.process(m, w=mc.WTS_KG), ELO))),
       ("no_tT", "No temporal factor (W_k = max sigma)", lambda: ed.index(dproc, use_tT=False)),
       ("alpha0", "alpha = 0 (no intensity)", lambda: ed.index(dproc, alpha=0)),
       ("alpha1", "alpha = 1 (no surprise)", lambda: ed.index(dproc, alpha=1)),
       ("lam0", "lambda = 0 (no stakes)", lambda: ed.index(dproc, lam=0)),
       ("lam2", "lambda = 2", lambda: ed.index(dproc, lam=2)),
       ("additive", "Additive form (G_k + U_k + W_k)/3", lambda: ed.index(dproc, form="add")),
       ("geometric", "Geometric form (G_k U_k W_k)^(1/3)", lambda: ed.index(dproc, form="geo")),
       ("rank_surprise", "Rank-based surprise", lambda: mc.add_index_rank(base_proc)),
       ("no_warmup", "Ratings without the 2014/15 initialisation", lambda: ed.index(ed.attach_surprise(base_proc, ELO0)))]
cal_proc = mc.process(m, stand="calendar"); eps_proc = mc.process(m, eps=False)
var = {k: summary(f()) for k, _, f in VAR}
print("\nTABLE 6")
for k, lab, _ in VAR:
    s_ = var[k]; print(f"  {lab:45s} {s_['ranking']:28s} r_HHI={s_['r_hhi']:.2f} (p={s_['p_hhi']:.3f}) gain={s_['gain']:+.0f}%")
put("variants", var)
pd.DataFrame([dict(variant=lab, ranking=var[k]["ranking"], r_HHI=round(var[k]["r_hhi"], 2), gain_pct=round(var[k]["gain"]))
              for k, lab, _ in VAR]).to_csv(OUT / "table7_robustness.csv", index=False)
put("cal_W_changed", 100 * (np.abs(cal_proc.W_k.values - base_proc.W_k.values) > 1e-9).mean())
put("eps_W_changed", 100 * (np.abs(eps_proc.W_k.values - base_proc.W_k.values) > 1e-9).mean())
put("ties_W_maxdiff", float(np.abs(mc.process(m, rank_mode="avg").W_k.values - base_proc.W_k.values).max()))
print("calendar: W differs for %.1f%% of matches | EPS: %.2f%% | ties at average rank: max |dW| = %g" % (
    N["cal_W_changed"], N["eps_W_changed"], N["ties_W_maxdiff"]))
bind = (base_proc.W_cat.value_counts(normalize=True) * 100).round(1).to_dict(); print("binding stake shares", bind); put("binding", bind)
F = {k: ed.index(dproc, form=k) for k in ["mult", "add", "geo"]}
z2 = (b.tg == 0) & (b.W_k >= 0.8)
ff = {}
for k, d_ in F.items():
    pct = d_.MBCI_star.rank(pct=True)
    ff[k] = dict(rho=spearmanr(F["mult"].MBCI_star, d_.MBCI_star)[0], top100=len(set(F["mult"].MBCI_star.nlargest(100).index) & set(d_.MBCI_star.nlargest(100).index)),
                 zero_share=100 * (d_.MBCI_star < 1e-12).mean(), pct00=100 * pct[z2].mean(), mean00=d_.MBCI_star[z2].mean())
rk = mc.add_index_rank(base_proc); rk_pct = rk.MBCI_star.rank(pct=True)
ff["rank_old"] = dict(pct00=100 * rk_pct[z2].mean(), mean00=rk.MBCI_star[z2].mean())
print("functional forms", {k: {kk: round(vv, 3) for kk, vv in d_.items()} for k, d_ in ff.items()}); put("forms", ff)
g = []
for a_, l_, c_ in itertools.product([0, .25, .5, .75, 1], [0, .5, 1, 2], [.25, .5, .75]):
    s_ = mc.season_level(ed.index(dproc, c=c_, alpha=a_, lam=l_)); lr = mc.league_rank(s_)
    g.append(dict(alpha=a_, lam=l_, c=c_, first=lr.index[0], ranking=" > ".join(ABBR[k] for k in lr.index)))
G_ = pd.DataFrame(g); G_.to_csv(OUT / "grid_alpha_lambda_c.csv", index=False)
print("grid: BL first in %d/60 ; by alpha:" % (G_["first"] == "Bundesliga").sum(), G_.groupby("alpha")["first"].value_counts().unstack(fill_value=0).to_dict("index"))
put("grid_bl_first", int((G_["first"] == "Bundesliga").sum()))

# ======================================================================== exports
mcols = ["League", "Season", "Gameweek", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "rank_home", "rank_away", "n", "T",
         "G_k", "R_home", "R_away", "P_H", "P_N", "P_A", "U_k", "W_raw", "W_cat", "W_k", "MBCI_star", "terc",
         "B365H", "B365D", "B365A", "pH", "pD", "pA", "surprisal", "h_mkt"]
b[mcols].to_excel(OUT / "mbci_match_level.xlsx", index=False)
json.dump(N, open(OUT / "numbers.json", "w", encoding="utf-8"), default=float, indent=1)

# ======================================================================== check against the article
# (section, label, value in the article, computed value, decimals); "lt" = the article says "less than"
V = var; R = reg["U_k"]; C = val["U_k"]; cl = cal.iloc[3]; T5 = t5
CHECKS = [
    ("4.5", "w_UCL", 0.713, mc.ahp_weights()["UCL"], 3), ("4.5", "w_UEL", 0.165, mc.ahp_weights()["UEL"], 3),
    ("4.5", "w_UECL", 0.121, mc.ahp_weights()["UECL"], 3),
    ("6", "matchs", 17732, len(b), 0), ("6", "ligues-saisons", 49, b.groupby(["League", "Season"]).ngroups, 0),
    ("6", "matchs 2014/15", 1826, N["n_warmup"], 0), ("6", "matchs avec cotes", 17726, len(v), 0),
    ("7", "mu", 0.347, N["mu"], 3), ("7", "nu EPL", 0.61, P.loc["EPL", "nu"], 2), ("7", "nu LaLiga", 0.71, P.loc["LaLiga", "nu"], 2),
    ("7", "h SerieA", 50, P.loc["SerieA", "h"], 0), ("7", "h LaLiga", 81, P.loc["LaLiga", "h"], 0),
    ("T3", "buts moyenne", 2.79, t3.loc["tg", "mean"], 2), ("T3", "buts ecart-type", 1.68, t3.loc["tg", "std"], 2),
    ("T3", "ecart moyenne", 1.39, t3.loc["gdiff", "mean"], 2), ("T3", "ecart ecart-type", 1.24, t3.loc["gdiff", "std"], 2),
    ("T3", "G moyenne", 0.456, t3.loc["G_k", "mean"], 3), ("T3", "G ecart-type", 0.240, t3.loc["G_k", "std"], 3),
    ("T3", "G mediane", 0.500, t3.loc["G_k", "median"], 3), ("T3", "G max", 0.956, t3.loc["G_k", "max"], 3),
    ("T3", "U moyenne", 0.586, t3.loc["U_k", "mean"], 3), ("T3", "U ecart-type", 0.182, t3.loc["U_k", "std"], 3),
    ("T3", "U min", 0.098, t3.loc["U_k", "min"], 3), ("T3", "U mediane", 0.607, t3.loc["U_k", "median"], 3),
    ("T3", "U max", 0.962, t3.loc["U_k", "max"], 3),
    ("T3", "W moyenne", 0.384, t3.loc["W_k", "mean"], 3), ("T3", "W ecart-type", 0.214, t3.loc["W_k", "std"], 3),
    ("T3", "W mediane", 0.400, t3.loc["W_k", "median"], 3),
    ("T3", "MBCI* moyenne", 0.360, t3.loc["MBCI_star", "mean"], 3), ("T3", "MBCI* ecart-type", 0.114, t3.loc["MBCI_star", "std"], 3),
    ("T3", "MBCI* min", 0.074, t3.loc["MBCI_star", "min"], 3), ("T3", "MBCI* mediane", 0.349, t3.loc["MBCI_star", "median"], 3),
    ("T3", "MBCI* max", 0.806, t3.loc["MBCI_star", "max"], 3),
    ("7.1", "U apres victoire dom.", 0.49, N["U_by_outcome"]["H"], 2), ("7.1", "U apres victoire ext.", 0.58, N["U_by_outcome"]["A"], 2),
    ("7.1", "U apres nul", 0.77, N["U_by_outcome"]["D"], 2),
    ("T4", "1er match (PSG-Le Havre)", 0.806, N["top"][0], 3), ("T4", "2e match (Dortmund-Schalke)", 0.795, N["top"][1], 3),
    ("T4", "5e match", 0.765, N["top"][4], 3), ("7.1", "top 10 : enjeu relegation", 8, N["top10_relg"], 0),
    ("7.1", "top 10 : nuls", 5, N["top10_draws"], 0), ("7.1", "top 10 : favori n'a pas gagne", 10, N["top10_fav_not_win"], 0),
    ("7.2", "KS D min", 0.28, min(N["ks_D"].values()), 2), ("7.2", "KS D max", 0.35, max(N["ks_D"].values()), 2),
    ("7.2", "ecart-type 1er tiers", 0.085, N["sd_terc"]["early"], 3), ("7.2", "ecart-type 3e tiers", 0.126, N["sd_terc"]["late"], 3),
    ("7.2", "matchs < 0.2 (%)", 6.5, N["low"]["share"], 1), ("7.2", "< 0.2 : victoires du favori (%)", 93, N["low"]["fav"], 0),
    ("7.2", "< 0.2 : au plus un but (%)", 72, N["low"]["tg_le1"], 0), ("7.2", "< 0.2 : 1er tiers (%)", 53, N["low"]["early"], 0),
    ("7.2", "0-0 : MBCI moyen", 0.27, N["mean_00"], 2),
    ("7.2", "hausse 1->3 min (%)", 26, gLE.min(), 0), ("7.2", "hausse 1->3 max (%)", 30, gLE.max(), 0),
    ("7.2", "hausse 1->2 min (%)", 21, gME.min(), 0), ("7.2", "hausse 1->2 max (%)", 24, gME.max(), 0),
    ("7.2", "hausse 2->3 min (%)", 2, gLM.min(), 0), ("7.2", "hausse 2->3 max (%)", 6, gLM.max(), 0),
    ("7.2", "EPL : hausse (%)", 26, gLE["EPL"], 0), ("7.2", "EPL : Europe determinante au 3e tiers (%)", 27, bind_late.loc["EPL", "eur"], 0),
    ("7.2", "EPL : U au 3e tiers", 0.557, Ug.loc["EPL", "late"], 3), ("7.2", "LaLiga : tiers 2", 0.372, mm.loc["LaLiga", "mid"], 3),
    ("7.2", "LaLiga : W au 3e tiers", 0.499, Wg.loc["LaLiga", "late"], 3), ("7.2", "LaLiga : titre determinant au 3e tiers (%)", 22, bind_late.loc["LaLiga", "title"], 0),
    ("7.2", "Bundesliga : tiers 2", 0.389, mm.loc["Bundesliga", "mid"], 3), ("7.2", "Serie A : hausse (%)", 29, gLE["SerieA"], 0),
    ("7.2", "Ligue 1 : relegation determinante au 3e tiers (%)", 62, bind_late.loc["Ligue1", "relg"], 0),
    ("7.2", "Ligue 1 : titre determinant au 3e tiers (%)", 15, bind_late.loc["Ligue1", "title"], 0),
    ("7.2", "EPL : >0.5 au 1er tiers (%)", 0.5, gt5.loc["EPL", "early"], 1),
    ("7.2", "EPL : tiers 1 -> 3", 0.304, mm.loc["EPL", "early"], 3), ("7.2", "EPL : tiers 3", 0.384, mm.loc["EPL", "late"], 3),
    ("7.2", "LaLiga : tiers 1", 0.301, mm.loc["LaLiga", "early"], 3), ("7.2", "LaLiga : tiers 3", 0.378, mm.loc["LaLiga", "late"], 3),
    ("7.2", "LaLiga : <0.2 au 1er tiers (%)", 13.3, lt2.loc["LaLiga", "early"], 1),
    ("7.2", "Bundesliga : tiers 1", 0.322, mm.loc["Bundesliga", "early"], 3), ("7.2", "Bundesliga : tiers 3", 0.413, mm.loc["Bundesliga", "late"], 3),
    ("7.2", "Bundesliga : >0.5 au 3e tiers (%)", 27.0, gt5.loc["Bundesliga", "late"], 1),
    ("7.2", "Serie A : >0.5 au 3e tiers (%)", 21.8, gt5.loc["SerieA", "late"], 1), ("7.2", "Serie A : rang du meilleur match", 12, N["sa_best_rank"], 0),
    ("7.2", "Ligue 1 : U tiers 1", 0.610, Ug.loc["Ligue1", "early"], 3),
    ("T5", "BL", 0.3755, T5.loc["Bundesliga", "mean"], 4), ("T5", "L1", 0.3627, T5.loc["Ligue1", "mean"], 4),
    ("T5", "EPL", 0.3567, T5.loc["EPL", "mean"], 4), ("T5", "SA", 0.3563, T5.loc["SerieA", "mean"], 4),
    ("T5", "LL", 0.3514, T5.loc["LaLiga", "mean"], 4),
    ("T5", "IC BL", "0.372-0.380", "%.3f-%.3f" % tuple(ci_mean["Bundesliga"]), "str"), ("T5", "IC L1", "0.358-0.367", "%.3f-%.3f" % tuple(ci_mean["Ligue1"]), "str"),
    ("T5", "IC EPL", "0.353-0.361", "%.3f-%.3f" % tuple(ci_mean["EPL"]), "str"), ("T5", "IC SA", "0.351-0.361", "%.3f-%.3f" % tuple(ci_mean["SerieA"]), "str"),
    ("T5", "IC LL", "0.344-0.358", "%.3f-%.3f" % tuple(ci_mean["LaLiga"]), "str"),
    ("T5", "SD BL", 0.0068, T5.loc["Bundesliga", "sd"], 4), ("T5", "SD LL", 0.0112, T5.loc["LaLiga", "sd"], 4),
    ("T5", "G BL", 0.491, T5.loc["Bundesliga", "G"], 3), ("T5", "G LL", 0.434, T5.loc["LaLiga", "G"], 3),
    ("T5", "U L1", 0.599, T5.loc["Ligue1", "U"], 3), ("T5", "U EPL", 0.575, T5.loc["EPL", "U"], 3),
    ("T5", "buts BL", 3.06, T5.loc["Bundesliga", "goals"], 2), ("T5", "buts LL", 2.62, T5.loc["LaLiga", "goals"], 2),
    ("7.3", "permutation BL : p Holm max", 0.005, PR[PR.pair.str.startswith("Bundesliga")].p_holm.max(), 3),
    ("7.3", "BL - L1 : ecart", 0.013, PR.set_index("pair").loc["Bundesliga-Ligue1", "diff"], 3),
    ("7.3", "BL - L1 : IC bas", 0.007, PR.set_index("pair").loc["Bundesliga-Ligue1", "lo"], 3),
    ("7.3", "BL - L1 : IC haut", 0.019, PR.set_index("pair").loc["Bundesliga-Ligue1", "hi"], 3),
    ("7.3", "g de Hedges BL min", 1.7, PR[PR.pair.str.startswith("Bundesliga")].g.min(), 1),
    ("7.3", "g de Hedges BL max", 2.5, PR[PR.pair.str.startswith("Bundesliga")].g.max(), 1),
    ("7.3", "4 autres : p Holm min", 0.12, PR[~PR.pair.str.startswith("Bundesliga")].p_holm.min(), 2),
    ("7.3", "4 autres : g max", 1.1, PR[~PR.pair.str.startswith("Bundesliga")].g.max(), 1),
    ("7.3", "BL - L1 apparie : p Holm", 0.055, PR.set_index("pair").loc["Bundesliga-Ligue1", "p_paired_holm"], 3),
    ("7.2", "IC bootstrap D min", 0.23, KB.D_lo.min(), 2), ("7.2", "IC bootstrap D max", 0.40, KB.D_hi.max(), 2),
    ("7.2", "IC bootstrap hausse min (%)", 20, KB.gain_lo.min(), 0), ("7.2", "IC bootstrap hausse max (%)", 34, KB.gain_hi.max(), 0),
    ("7.5", "r effets fixes ligue et saison", 0.20, hhi_boot["r_2w"], 2), ("7.5", "p effets fixes", 0.25, hhi_boot["p_2w"], 2),
    ("7.5", "IC bootstrap par saison bas", 0.10, hhi_boot["lo_season"], 2), ("7.5", "IC bootstrap par saison haut", 0.55, hhi_boot["hi_season"], 2),
    ("7.3", "BL 2022/23", 0.393, Sw.loc["2022-23", "Bundesliga"], 3), ("7.3", "LL 2017/18", 0.329, Sw.loc["2017-18", "LaLiga"], 3),
    ("7.3", "amplitude min", 0.022, span.min(), 3), ("7.3", "amplitude max", 0.035, span.max(), 3),
    ("7.3", "ecart max des 4 autres", 0.011, o4.max() - o4.min(), 3), ("7.3", "correlation L1-SA", 0.78, cc.loc["Ligue1", "SerieA"], 2),
    ("7.3", "saisons avec BL premier", 9, int((first == "Bundesliga").sum()), 0), ("7.3", "premier en 2024/25", "Ligue1", first["2024-25"], "str"),
    ("7.3", "tendance min (x1e-3)", -0.04, min(v_[0] for v_ in tr.values()), 2), ("7.3", "tendance max (x1e-3)", 1.2, max(v_[0] for v_ in tr.values()), 1),
    ("7.3", "tendance p min", 0.17, min(v_[1] for v_ in tr.values()), 2), ("7.3", "correlation min entre ligues", -0.16, cu.min(), 2),
    ("7.4", "tiers 1", 0.307, tt.loc["early", "MBCI_star"], 3), ("7.4", "tiers 2", 0.377, tt.loc["mid", "MBCI_star"], 3),
    ("7.4", "tiers 3", 0.392, tt.loc["late", "MBCI_star"], 3), ("7.4", "hausse (%)", 28, V["baseline"]["gain"], 0),
    ("7.4", "W tiers 1", 0.168, tt.loc["early", "W_k"], 3), ("7.4", "W tiers 3", 0.513, tt.loc["late", "W_k"], 3),
    ("7.4", "max sigma tiers 1", 0.98, tt.loc["early", "W_raw"], 2), ("7.4", "max sigma tiers 3", 0.63, tt.loc["late", "W_raw"], 2),
    ("7.4", "sans t/T (%)", -19, N["no_tT_change"], 0), ("7.4", "max sigma t/T<=0.2", 0.99, N["maxsig_tT_le_02"], 2),
    ("7.4", "max sigma 0.4-0.6", 0.92, N["maxsig_04_06"], 2), ("7.4", "derniere journee W=0 (%)", 56, N["final_round"]["W0"], 0),
    ("7.4", "derniere journee W>=0.9 (%)", 4, N["final_round"]["W09"], 0),
    ("7.5", "r(MBCI_s, HHI-)", 0.33, hp["r"], 2), ("7.5", "p", 0.02, hp["p"], 2), ("7.5", "IC bas", 0.06, hp["lo"], 2),
    ("7.5", "IC haut", 0.56, hp["hi"], 2), ("7.5", "r au sein des ligues", 0.27, hp["r_within"], 2),
    ("7.5", "p au sein des ligues", 0.07, hp["p_within"], 2), ("7.5", "r HHI des victoires", 0.30, hw["r"], 2),
    ("7.5", "r U_s", 0.83, comp["U"][0], 2), ("7.5", "r G_s", -0.27, comp["G"][0], 2), ("7.5", "r W_s", 0.28, comp["W"][0], 2),
    ("7.5", "r alpha=0", 0.79, r_a0["r"], 2), ("7.5", "r alpha=1", -0.17, r_a1["r"], 2),
    ("7.7", "corr toutes issues", 0.91, C["all"], 2), ("7.7", "corr victoires", 0.90, C["dec"], 2), ("7.7", "corr nuls", 0.78, C["draw"], 2),
    ("7.7", "matchs decisifs", 13286, len(vd), 0), ("7.7", "pente", 2.45, R["all_b"], 2), ("7.7", "t", 135, R["all_t"], 0),
    ("7.7", "R2 toutes issues", 0.83, R["all_R2"], 2), ("7.7", "R2 victoires", 0.80, R["dec_R2"], 2),
    ("7.7", "R2 + victoire ext.", 0.81, R["away_R2"], 2), ("7.7", "coef. victoire ext.", 0.06, R["away_b"], 2),
    ("7.7", "R2 rangs victoires", 0.43, reg["U_rank"]["dec_R2"], 2), ("7.7", "R2 rangs toutes issues", 0.05, reg["U_rank"]["all_R2"], 2),
    ("7.7", "U favori (0.8-0.9)", 0.24, cl.u_fav, 2), ("7.7", "U nul (0.8-0.9)", 0.84, cl.u_draw, 2), ("7.7", "U outsider (0.8-0.9)", 0.92, cl.u_dog, 2),
    ("7.7", "marche favori", 0.22, cl.m_fav, 2), ("7.7", "marche nul", 0.85, cl.m_draw, 2), ("7.7", "marche outsider", 0.92, cl.m_dog, 2),
    ("7.7", "log-perte Elo", 0.985, N["logloss"]["model"], 3), ("7.7", "log-perte marche", 0.965, N["logloss"]["market"], 3),
    ("7.7", "r(MBCI*, H_mkt)", 0.21, hm["MBCI_star"], 2), ("7.7", "r(U, H_mkt)", 0.50, hm["U_k"], 2),
    ("7.7", "r(W, H_mkt)", -0.02, hm["W_k"], 2), ("7.7", "r(G, H_mkt)", -0.08, hm["G_k"], 2),
    ("7.8", "grille : BL premier", 48, N["grid_bl_first"], 0), ("7.8", "W change (calendrier, %)", 26.5, N["cal_W_changed"], 1),
    ("7.8", "W change (EPS, %)", 1, N["eps_W_changed"], "lt"), ("7.8", "max |dW| (rang moyen)", 0, N["ties_W_maxdiff"], 6),
    ("7.8", "additive : rho", 0.98, ff["add"]["rho"], 2), ("7.8", "additive : top 100", 95, ff["add"]["top100"], 0),
    ("7.8", "geometrique : zeros (%)", 9, ff["geo"]["zero_share"], 0), ("7.8", "0-0 fort enjeu : percentile", 53, ff["mult"]["pct00"], 0),
    ("7.8", "0-0 fort enjeu : percentile (rangs)", 7, ff["rank_old"]["pct00"], 0),
    ("7.8", "enjeu determinant relegation (%)", 64, bind["relg"], 0), ("7.8", "titre (%)", 28, bind["title"], 0),
    ("7.8", "Europe (%)", 8, bind["eur"], 0)]
T6 = [("baseline", "BL > L1 > EPL > SA > LL", 0.33, 28), ("ties_avg", "BL > L1 > EPL > SA > LL", 0.33, 28),
      ("calendar", "BL > L1 > SA > EPL > LL", 0.33, 28), ("eps_unant", "BL > L1 > EPL > SA > LL", 0.32, 28),
      ("w_equal", "BL > L1 > EPL > SA > LL", 0.35, 37), ("w_kg", "BL > L1 > EPL > SA > LL", 0.18, 7),
      ("no_tT", "BL > L1 > EPL > SA > LL", 0.31, -19), ("alpha0", "L1 > BL > LL > SA > EPL", 0.79, 26),
      ("alpha1", "BL > EPL > SA > L1 > LL", -0.17, 30), ("lam0", "BL > L1 > EPL > SA > LL", 0.19, -1),
      ("lam2", "BL > L1 > EPL > SA > LL", 0.36, 49), ("additive", "BL > L1 > EPL > SA > LL", 0.34, 27),
      ("geometric", "BL > L1 > EPL > SA > LL", 0.20, 39), ("rank_surprise", "BL > EPL > L1 > SA > LL", 0.18, 27),
      ("no_warmup", "BL > L1 > EPL > SA > LL", 0.35, 28)]
for k, rk_, r_t, g_t in T6:
    CHECKS += [("T6", f"{k} : classement", rk_, V[k]["ranking"], "str"), ("T6", f"{k} : r_HHI", r_t, V[k]["r_hhi"], 2),
               ("T6", f"{k} : hausse (%)", g_t, V[k]["gain"], 0)]

rows = []
for sec, lab, art, calc, dec in CHECKS:
    if dec == "str": ok = art == calc; shown = calc
    elif dec == "lt": ok = calc < art; shown = round(float(calc), 4)
    else: ok = abs(float(calc) - art) <= 0.5 * 10 ** (-dec) + 1e-9; shown = round(float(calc), max(dec, 0) + 2)
    rows.append(dict(section=sec, chiffre=lab, article=art, calcul=shown, statut="OK" if ok else "ECART"))
chk = pd.DataFrame(rows); chk.to_csv(OUT / "article_numbers_check.csv", index=False, encoding="utf-8-sig")
n_ok = (chk.statut == "OK").sum()
print(f"\nCONTROLE CONTRE L'ARTICLE : {n_ok}/{len(chk)} chiffres conformes")
if n_ok < len(chk): print(chk[chk.statut != "OK"].to_string(index=False))
print("Sorties ecrites dans outputs/")
