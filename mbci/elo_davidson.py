"""
elo_davidson.py -- surprise component U_k (Elo-Davidson), same definitions as MBCI_elo_davidson.ipynb.

U_k = 1 - P_k(observed outcome), with P(H), P(N), P(A) from Elo-Davidson ratings
(Davidson 1970; Szczecinski & Djebbi 2020), K_k = 10 (1 + |g_i - g_j|) (Hvattum & Arntzen 2010),
carry-over rho = 2/3, promoted clubs = mean rating of the relegated clubs, warm-up season 2014/15.
League parameters from the analysed sample: nu_L = 2 d_L / (1 - d_L), h_L from the home points share.
"""
import numpy as np
import pandas as pd
import mbci_core as mc

MATCH_ALL = mc.DATA / "global_5_leagues_2014_26.xlsx"
WARMUP = "2014-15"
ANALYSIS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2025)]
R0, SCALE, K0, RHO = 1500.0, 400.0, 10.0, 2 / 3


def load_elo_matches(warmup=True):
    raw = pd.read_excel(MATCH_ALL, sheet_name="Global")
    raw["Date"] = mc._parse_dates(raw["Date"])
    raw["MID"] = raw.League + "|" + raw.Season + "|" + raw.HomeTeam + "|" + raw.AwayTeam
    seasons = ([WARMUP] if warmup else []) + ANALYSIS
    e = raw[raw.Season.isin(seasons)].copy()
    return e.sort_values(["League", "Season", "Date"], kind="stable").reset_index(drop=True)


def league_parameters(master):
    out = {}
    for lg, df in master.groupby("League"):
        d = (df.FTHG == df.FTAG).mean()
        s = ((df.FTHG > df.FTAG) + 0.5 * (df.FTHG == df.FTAG)).mean()
        nu = 2 * d / (1 - d)
        y = (-nu * (0.5 - s) + np.sqrt(nu ** 2 * (0.5 - s) ** 2 + 4 * s * (1 - s))) / (2 * (1 - s))
        out[lg] = dict(draw_share=d, nu=nu, home_share=s, h=800 * np.log10(y))
    return pd.DataFrame(out).T


def davidson(q, nu):
    den = q + 1 + nu * np.sqrt(q)
    return q / den, nu * np.sqrt(q) / den, 1 / den


def run_elo(elo_matches, params, k0=K0, rho=RHO):
    rows = []
    for lg, f in elo_matches.groupby("League", sort=False):
        nu, h = params.loc[lg, "nu"], params.loc[lg, "h"]
        R, prev = {}, set()
        for se, g in f.groupby("Season", sort=True):
            teams = set(g.HomeTeam) | set(g.AwayTeam)
            if not R:
                R = {t: R0 for t in teams}
            else:
                rel = prev - teams
                r_rel = np.mean([R[t] for t in rel]) if rel else min(R[t] for t in prev)
                R = {t: R0 + rho * ((R[t] if t in prev else r_rel) - R0) for t in teams}
            for r in g.itertuples():
                Ri, Rj = R[r.HomeTeam], R[r.AwayTeam]
                q = 10 ** ((Ri - Rj + h) / SCALE)
                pH, pN, pA = davidson(q, nu)
                E = pH + 0.5 * pN
                Y = 1.0 if r.FTHG > r.FTAG else (0.5 if r.FTHG == r.FTAG else 0.0)
                K = k0 * (1 + abs(r.FTHG - r.FTAG))
                rows.append((r.MID, Ri, Rj, q, pH, pN, pA))
                R[r.HomeTeam] = Ri + K * (Y - E); R[r.AwayTeam] = Rj - K * (Y - E)
            prev = teams
    return pd.DataFrame(rows, columns=["MID", "R_home", "R_away", "q", "P_H", "P_N", "P_A"])


def attach_surprise(d, elo):
    d = d.drop(columns=[c for c in ["R_home", "R_away", "q", "P_H", "P_N", "P_A", "U_elo"] if c in d.columns])
    d = d.merge(elo, on="MID", how="left")
    p = np.where(d.FTR == "H", d.P_H, np.where(d.FTR == "D", d.P_N, d.P_A))
    d["U_elo"] = 1 - p
    return d


def index(d, U="U_elo", c=mc.C0, alpha=mc.A0, lam=mc.LAM0, use_tT=True, form="mult"):
    """MBCI* with the chosen surprise column. form: mult (baseline), add, geo."""
    d = d.copy()
    S = (d.FTHG + d.FTAG) - c * (d.FTHG - d.FTAG).abs()
    mu = np.log(2) / S.median(); d["G_k"] = 1 - np.exp(-mu * S); d["mu"] = mu
    d["U_k"] = d[U]
    Wv = d.W_k if use_tT else d.W_raw
    if form == "mult":
        d["MBCI_star"] = (alpha * d.G_k + (1 - alpha) * d.U_k) * (1 + lam * Wv) / (1 + lam)
    elif form == "add":
        d["MBCI_star"] = (d.G_k + d.U_k + Wv) / 3
    elif form == "geo":
        d["MBCI_star"] = (d.G_k * d.U_k * Wv) ** (1 / 3)
    return d
