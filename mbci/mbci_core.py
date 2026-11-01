"""
mbci_core.py -- data, standings, stakes and shared tools of the MBCI.

Article: "A Match-Based Competitiveness Index for Football Leagues: Integrating
Goal Intensity, Outcome Surprise, and Season Stakes" (Sports, MDPI), revised version.

This module provides everything that does not depend on the surprise component: data loading,
reconstructed standings, stakes W_k (Section 4.5), HHI, market probabilities (Shin's method),
terciles. The surprise U_k of the article (Elo-Davidson, Section 4.4) and the index
MBCI*_k are computed in elo_davidson.py (functions attach_surprise and index).

add_index_rank() and baseline_rank() compute the OLD rank-based surprise of the
first version; it is now used only as a robustness variant (row
"Rank-based surprise" of Table 7) and as a benchmark in the validation (Section 7.7).

Conventions (Section 6 of the article)
- Ligue 1 2019-20 (curtailed season) and the 2025-26 season (no official rounds) are excluded.
- Missing rounds (2 matches) are reconstructed from the fixture calendar.
- Villarreal-Vallecano (LaLiga 2024-25) is moved to its official round (12 instead of 24).
- Standings before official round t: points, then goal difference,
  then goals scored, then alphabetical order (no effect on the index, since W_k
  depends only on point gaps); no point deductions.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

try:                                    # Windows consoles: avoid encoding errors
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DATA_XLSX = DATA / "global_5_leagues.xlsx"
UEFA_XLSX = DATA / "UEFA_Simple_1_decale.xlsx"      # berths at stake in the final standings of the season
RELEG_XLSX = DATA / "Relegation_Simple.xlsx"

EXCLUDE_LS = {("Ligue1", "2019-20")}
EXCLUDE_SEASONS = {"2025-26"}
C0, LAM0, A0 = 0.5, 1.0, 0.5            # c, lambda, alpha (beta = 1 - alpha): baseline of the article
FIX_GW = True

LMAP = {"Premier League": "EPL", "LaLiga": "LaLiga", "Serie A": "SerieA", "Bundesliga": "Bundesliga", "Ligue 1": "Ligue1"}
ORDER = ["Bundesliga", "EPL", "Ligue1", "SerieA", "LaLiga"]          # league ranking in the article (Table 5)
LEAGUES = ["EPL", "LaLiga", "Bundesliga", "SerieA", "Ligue1"]        # order of presentation (Section 7.2)


# ------------------------------------------------------------------ AHP (Section 4.5)
FEES = {"UCL": 18.62, "UEL": 4.31, "UECL": 3.17}    # EUR M participation fee per club, UEFA 2024-27 cycle

def ahp_weights(fees=FEES, verbose=False):
    """Principal eigenvector of the matrix a_bb' = fee_b / fee_b' (Saaty, 1980).
    The matrix is perfectly consistent (CR = 0): the weights are the normalised fees."""
    v = np.array(list(fees.values()), float)
    A = np.outer(v, 1 / v)
    w = (A / A.sum(0)).mean(1)
    n = len(v); lam_max = float((A @ w) @ (1 / w) / n)
    CI = (lam_max - n) / (n - 1); CR = CI / 0.58
    if verbose:
        print("AHP weights: " + ", ".join(f"{k}={x:.4f}" for k, x in zip(fees, w)) + f" | lambda_max={lam_max:.4f} CR={CR:.4f}")
    # rounded to 4 decimals, as in the computation that produced the published results
    return {k: round(float(x), 4) for k, x in zip(fees, w)}

W_FEE = ahp_weights()
WTS_BASE = dict(title=1.0, relg=1.0, **W_FEE)
WTS_EQUAL = dict(title=1.0, relg=1.0, UCL=1.0, UEL=1.0, UECL=1.0)
# Kringstad & Gerrard (2005), prestige weights also used by Wagner et al. (2020)
WTS_KG = dict(title=1.0, UCL=1/1.5**2, UEL=1/2**2, UECL=1/2**2, relg=1/3**2)


# ------------------------------------------------------------------ European cut-offs and relegation
def load_eur(eps=True):
    """{(league, season): [(cut-off rank, competition), ...]}. eps=False removes the European Performance Spots."""
    b = {}
    for _, r in pd.read_excel(UEFA_XLSX, sheet_name="Places UEFA").iterrows():
        lg = LMAP.get(str(r["Championnat"]).strip())
        if lg is None:
            continue
        nU = int(r["UCL"]) - (0 if eps else int(r.get("UCL dont EPS", 0) or 0))
        nE, nC = int(r["UEL"]), int(r["UECL"])
        L = [(nU, "UCL")]
        if nE > 0: L.append((nU + nE, "UEL"))
        if nC > 0: L.append((nU + nE + nC, "UECL"))
        b[(lg, str(r["Saison"]).strip())] = L
    return b

EUR = {True: load_eur(True), False: load_eur(False)}
RELEG = {(LMAP[str(r["Championnat"]).strip()], str(r["Saison"]).strip()): int(r["Zone_relegation"])
         for _, r in pd.read_excel(RELEG_XLSX).iterrows() if str(r["Championnat"]).strip() in LMAP}

def eur_bounds(lg, se, eps=True):
    E_ = EUR[eps]
    if (lg, se) in E_:
        return E_[(lg, se)]
    raise KeyError(f"no European boundaries for {lg} {se}")


# ------------------------------------------------------------------ data
def _parse_dates(s):
    """Dates of the source file in dd/mm/yy or dd/mm/yyyy format (explicit format, no guessing)."""
    s = s.astype(str).str.strip()
    d = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    long_, short_ = s.str.len() == 10, s.str.len() == 8
    d[long_] = pd.to_datetime(s[long_], format="%d/%m/%Y")
    d[short_] = pd.to_datetime(s[short_], format="%d/%m/%y")
    if d.isna().any():
        raise ValueError(f"{int(d.isna().sum())} unparsed dates")
    return d

def load_master(fix_gw=FIX_GW, verbose=False):
    m = pd.read_excel(DATA_XLSX, sheet_name="Global").rename(columns={"GW": "Gameweek"})
    m["FTHG"] = pd.to_numeric(m["FTHG"], errors="coerce").fillna(0).astype(int)
    m["FTAG"] = pd.to_numeric(m["FTAG"], errors="coerce").fillna(0).astype(int)
    m["Date"] = _parse_dates(m["Date"])
    m["FTR"] = np.where(m.FTHG > m.FTAG, "H", np.where(m.FTHG < m.FTAG, "A", "D"))
    n0 = len(m)
    m = m[~m.Season.isin(EXCLUDE_SEASONS)]
    m = m[~m.set_index(["League", "Season"]).index.isin(EXCLUDE_LS)].reset_index(drop=True)
    parts, recon = [], []
    for (lg, se), g in m.groupby(["League", "Season"], sort=False):
        g = g.sort_values("Date").copy(); played, rec = {}, []
        for h, a in zip(g.HomeTeam, g.AwayTeam):
            t = max(played.get(h, 0), played.get(a, 0)) + 1; rec.append(t)
            played[h] = played.get(h, 0) + 1; played[a] = played.get(a, 0) + 1
        prov = pd.to_numeric(g["Gameweek"], errors="coerce").values
        miss = np.isnan(prov)
        for (h, a, t) in zip(g.HomeTeam[miss], g.AwayTeam[miss], np.array(rec)[miss]):
            recon.append((lg, se, h, a, int(t)))
        g["Gameweek"] = np.where(miss, rec, prov).astype(int); parts.append(g)
    m = pd.concat(parts).reset_index(drop=True)
    m["MID"] = m.League + "|" + m.Season + "|" + m.HomeTeam + "|" + m.AwayTeam
    # Villarreal-Vallecano (LaLiga 2024-25, played on 18/12/2024) is labelled round 24 in the source
    # but belongs to round 12 (round 12 contains neither club; round 24 contains both twice).
    fixed = False
    fix = m.MID == "LaLiga|2024-25|Villarreal|Vallecano"
    if fix_gw and fix.any() and int(m.loc[fix, "Gameweek"].iloc[0]) == 24:
        m.loc[fix, "Gameweek"] = 12; fixed = True
    if verbose:
        print(f"Source rows: {n0} | excluded: {n0 - len(m)} | kept: {len(m)}")
        for r in recon: print(f"  round reconstructed from calendar: {r[0]} {r[1]} {r[2]}-{r[3]} -> round {r[4]}")
        print(f"  Villarreal-Vallecano 2024-25 moved from round 24 to 12: {fixed}")
    m.attrs["reconstructed"] = recon; m.attrs["villarreal_fix"] = fixed
    return m


# ------------------------------------------------------------------ standings and stakes (Section 4.5)
def standings(teams, pts, gd, gf, mode):
    key = {t: (-pts[t], -gd[t], -gf[t]) for t in teams}
    st = sorted(teams, key=lambda x: key[x])          # teams sorted alphabetically -> stable sort = alphabetical order
    if mode == "order":
        return st, {tm: i + 1 for i, tm in enumerate(st)}
    rank, i = {}, 0                                    # mode "avg": average rank for ties
    while i < len(st):
        j = i
        while j + 1 < len(st) and key[st[j + 1]] == key[st[i]]: j += 1
        for k in range(i, j + 1): rank[st[k]] = (i + j + 2) / 2
        i = j + 1
    return st, rank

def stakes(tm, st, rank, pts, t, T, n, safe, eurb, w):
    """(sigma_title, sigma_relegation, sigma_europe) of team tm before round t."""
    mx = 3 * (T - t + 1)
    lead, second = pts[st[0]], pts[st[1]]
    last_safe, first_zone = pts[st[safe - 1]], pts[st[safe]]
    pbr = [pts[x] for x in st]; r = rank[tm]
    d = (lead - second) if r == 1 else (lead - pts[tm])
    s_t = w["title"] * max(0.0, 1 - d / mx)
    gap = (pts[tm] - first_zone) if r <= safe else (last_safe - pts[tm])
    s_r = w["relg"] * max(0.0, 1 - gap / mx)
    s_e = 0.0
    for br, lab in eurb:
        p_b, p_b1 = pbr[br - 1], (pbr[br] if br < n else 0)
        g2 = (pts[tm] - p_b1) if r <= br else (p_b - pts[tm])
        s_e = max(s_e, w[lab] * max(0.0, 1 - g2 / mx))
    return s_t, s_r, s_e

def process(m, rank_mode="order", stand="round", w=WTS_BASE, eps=True):
    """Pre-match ranks and W_k. stand='round': standings after official round t-1;
    stand='calendar': standings on the match date. Adds W_raw (= max sigma) and W_cat (decisive stake)."""
    res = []
    for (lg, se), df in m.groupby(["League", "Season"]):
        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam)); n = len(teams); T = int(df.Gameweek.max())
        safe = n - RELEG.get((lg, se), 3); eurb = eur_bounds(lg, se, eps)
        pts = {t: 0 for t in teams}; gd = dict(pts); gf = dict(pts)
        df = df.sort_values(["Gameweek", "Date"] if stand == "round" else ["Date", "Gameweek"]).reset_index(drop=True)
        gkey = "Gameweek" if stand == "round" else "Date"
        rows = []
        for _, grp in df.groupby(gkey, sort=True):
            st, rank = standings(teams, pts, gd, gf, rank_mode)
            for i in grp.index:
                h, a, t = df.at[i, "HomeTeam"], df.at[i, "AwayTeam"], int(df.at[i, "Gameweek"])
                sh, sa = stakes(h, st, rank, pts, t, T, n, safe, eurb, w), stakes(a, st, rank, pts, t, T, n, safe, eurb, w)
                allv = sh + sa; raw = max(allv); cat = ["title", "relg", "eur"][int(np.argmax(allv)) % 3]
                rows.append((df.at[i, "MID"], rank[h], rank[a], (t / T) * raw, raw, cat, n, T))
            for i in grp.index:
                h, a, hg, ag = df.at[i, "HomeTeam"], df.at[i, "AwayTeam"], int(df.at[i, "FTHG"]), int(df.at[i, "FTAG"])
                if hg > ag: pts[h] += 3
                elif ag > hg: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
                gd[h] += hg - ag; gd[a] += ag - hg; gf[h] += hg; gf[a] += ag
        res += rows
    r = pd.DataFrame(res, columns=["MID", "rank_home", "rank_away", "W_k", "W_raw", "W_cat", "n", "T"])
    return m.drop(columns=[c for c in r.columns if c in m.columns and c != "MID"]).merge(r, on="MID", how="left")


# ------------------------------------------------------------------ index with the OLD rank-based surprise
def add_index_rank(d, c=C0, alpha=A0, lam=LAM0, use_tT=True):
    """MBCI with the rank-based surprise of the first version (robustness variant).
    The index of the revised article is elo_davidson.index()."""
    d = d.copy()
    S = (d.FTHG + d.FTAG) - c * (d.FTHG - d.FTAG).abs()
    mu = np.log(2) / S.median()                        # a single global mu (sample median -> G_k = 0.5)
    d["S_k"] = S; d["G_k"] = 1 - np.exp(-mu * S)
    den = 2 * (d.n - 1)
    d["U_k"] = np.where(d.FTR == "D", (d.rank_home - d.rank_away).abs() / den,
               np.where(d.FTR == "H", (d.rank_home - d.rank_away + d.n - 1) / den,
                                      (d.rank_away - d.rank_home + d.n - 1) / den))
    W = d.W_k if use_tT else d.W_raw
    d["MBCI_k"] = (alpha * d.G_k + (1 - alpha) * d.U_k) * (1 + lam * W)
    d["MBCI_star"] = d["MBCI_k"] / (1 + lam)
    d["mu"] = mu
    return d

def baseline_rank(verbose=False):
    """(data, index of the first version, rank-based surprise)."""
    m = load_master(verbose=verbose)
    return m, add_index_rank(process(m))

# names of the first version, kept for the archived scripts (ancienne_version/)
add_index = add_index_rank
baseline = baseline_rank


# ------------------------------------------------------------------ aggregation, HHI, terciles
def season_level(d):
    return d.groupby(["League", "Season"]).MBCI_star.mean().rename("MBCI_s").reset_index()

def league_rank(s):
    return s.groupby("League").MBCI_s.mean().sort_values(ascending=False)

def hhi_table(m):
    """HHI of final points and wins, normalised as HHI x n x 100 (Michie & Oughton, 2004)."""
    rows = []
    for (L, S), g in m.groupby(["League", "Season"]):
        teams = sorted(set(g.HomeTeam) | set(g.AwayTeam)); pts = {t: 0 for t in teams}; wins = dict(pts)
        for h, a, hg, ag in zip(g.HomeTeam, g.AwayTeam, g.FTHG, g.FTAG):
            if hg > ag: pts[h] += 3; wins[h] += 1
            elif ag > hg: pts[a] += 3; wins[a] += 1
            else: pts[h] += 1; pts[a] += 1
        n = len(teams); p = np.array(list(pts.values())) / sum(pts.values()); w_ = np.array(list(wins.values())) / sum(wins.values())
        rows.append((L, S, (p ** 2).sum() * n * 100, (w_ ** 2).sum() * n * 100))
    return pd.DataFrame(rows, columns=["League", "Season", "HHI_pts", "HHI_wins"])

def final_table(d, league, season):
    """Reconstructed final standings (points): list [(club, points), ...]."""
    x = d[(d.League == league) & (d.Season == season)]; pts = {}
    for h, a, hg, ag in zip(x.HomeTeam, x.AwayTeam, x.FTHG, x.FTAG):
        pts.setdefault(h, 0); pts.setdefault(a, 0)
        if hg > ag: pts[h] += 3
        elif ag > hg: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    return sorted(pts.items(), key=lambda kv: -kv[1])

def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

def hhi_corr(s, H, col):
    from scipy.stats import pearsonr, spearmanr, t as tdist
    x = s.merge(H, on=["League", "Season"])
    r, p = pearsonr(x.MBCI_s, -x[col]); lo, hi = fisher_ci(r, len(x))
    rs, ps = spearmanr(x.MBCI_s, -x[col])
    xd = x.copy(); xd["m"] = xd.MBCI_s - xd.groupby("League").MBCI_s.transform("mean")
    xd["h"] = -xd[col] + xd.groupby("League")[col].transform("mean")
    rw = np.corrcoef(xd.m, xd.h)[0, 1]; dfw = len(x) - 5 - 1
    tw = rw * np.sqrt(dfw / (1 - rw ** 2)); pw = 2 * (1 - tdist.cdf(abs(tw), dfw))
    return {k: float(v) for k, v in dict(r=r, p=p, lo=lo, hi=hi, rho=rs, p_rho=ps, r_within=rw, p_within=pw).items()}

def add_tercile(d):
    d = d.copy(); d["terc"] = pd.cut(d.Gameweek / d["T"], [0, 1 / 3, 2 / 3, 1.01], labels=["early", "mid", "late"])
    d["tT"] = d.Gameweek / d["T"]
    return d

def terciles(d):
    return add_tercile(d).groupby("terc", observed=True)[["MBCI_star", "W_k", "W_raw", "tT", "G_k", "U_k"]].mean()


# ------------------------------------------------------------------ postponed matches (Section 6)
def postponed(m, days=3):
    """Matches played more than `days` days after the start of a later round."""
    post = []
    for (lg, se), gg in m.groupby(["League", "Season"]):
        first = gg.groupby("Gameweek").Date.min()
        for gw, dt, mid in zip(gg.Gameweek, gg.Date, gg.MID):
            later = first[first.index > gw]
            if len(later) and dt > later.min() + pd.Timedelta(days=days):
                post.append((lg, se, mid))
    return pd.DataFrame(post, columns=["League", "Season", "MID"])


# ------------------------------------------------------------------ betting odds (Section 7.7)
ODDS = ["B365H", "B365D", "B365A"]

def shin(o):
    """Shin (1993) probabilities from 1X2 odds."""
    bb = 1 / np.array(o, float); B = bb.sum()
    pi = lambda z: (np.sqrt(z * z + 4 * (1 - z) * bb * bb / B) - z) / (2 * (1 - z))
    try:
        z = brentq(lambda z: pi(z).sum() - 1, 1e-9, 0.45)
    except Exception:
        return bb / B
    return pi(z)

def add_market(d):
    """pH, pD, pA (Shin), normalised entropy h_mkt and surprisal s_k = -ln p(result)."""
    d = d.copy()
    for col in ["pH", "pD", "pA", "h_mkt", "p_real", "surprisal"]: d[col] = np.nan
    mask = d[ODDS].notna().all(1)
    P = np.array([shin(o) for o in d.loc[mask, ODDS].values])
    Pc = np.clip(P, 1e-12, 1); fr = d.loc[mask, "FTR"].values
    p_real = np.where(fr == "H", P[:, 0], np.where(fr == "A", P[:, 2], P[:, 1]))
    d.loc[mask, ["pH", "pD", "pA"]] = P
    d.loc[mask, "h_mkt"] = -(Pc * np.log(Pc)).sum(1) / np.log(3)
    d.loc[mask, "p_real"] = p_real
    d.loc[mask, "surprisal"] = -np.log(np.clip(p_real, 1e-12, 1))
    return d
