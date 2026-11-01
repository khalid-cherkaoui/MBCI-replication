"""
00_fetch_2014_15.py -- Elo warm-up season: Big Five 2014/15 from football-data.co.uk.

football-data.co.uk has no API: each season is one CSV file per league,
https://www.football-data.co.uk/mmz4281/1415/<Div>.csv  (E0, SP1, D1, I1, F1).
It is the same source as global_5_leagues.xlsx: same club names, same date format,
same Bet365 odds. Club names must be identical from one season to the next so that
Elo ratings carry over and the promoted/relegated rule works.

Usage (from the "mbci" folder):
    python 00_fetch_2014_15.py                  # direct download
    python 00_fetch_2014_15.py --local CSV_DIR  # if the 5 CSV files were downloaded manually

Outputs in data/ (global_5_leagues.xlsx is NOT modified):
    football_data_2014_15.xlsx        the 2014/15 season alone, sheet "Global"
    global_5_leagues_2014_26.xlsx     global_5_leagues.xlsx + 2014/15, sheet "Global"

The 2014/15 season is used only to initialise the Elo ratings: it must be excluded from the analysis
(EXCLUDE_SEASONS = {"2014-15", "2025-26"} in mbci_core.py). The source has no official
rounds: GW stays empty (GW_source = "none"), as for 2025/26; Elo uses only the dates.
"""
import argparse, io, sys, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEASON, CODE = "2014-15", "1415"
NEXT_SEASON = "2015-16"
DIVS = {"E0": "EPL", "SP1": "LaLiga", "D1": "Bundesliga", "I1": "SerieA", "F1": "Ligue1"}
N_TEAMS = {"E0": 20, "SP1": 20, "D1": 18, "I1": 20, "F1": 20}
URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
COLS = ["League", "Season", "GW", "GW_source", "Div", "Date", "Time",
        "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A"]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def read_bytes(div, local_dir):
    if local_dir:
        return (Path(local_dir) / f"{div}.csv").read_bytes()
    req = urllib.request.Request(URL.format(code=CODE, div=div), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def parse_csv(raw):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError("encodage du CSV non reconnu")


def parse_dates(s):
    """dd/mm/yy or dd/mm/yyyy dates, read with an explicit format (as in mbci_core.py)."""
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[s.str.len() == 10] = pd.to_datetime(s[s.str.len() == 10], format="%d/%m/%Y")
    out[s.str.len() == 8] = pd.to_datetime(s[s.str.len() == 8], format="%d/%m/%y")
    return out


def build_division(div, local_dir):
    raw = parse_csv(read_bytes(div, local_dir))
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    out = pd.DataFrame({
        "League": DIVS[div], "Season": SEASON, "GW": float("nan"), "GW_source": "none", "Div": div,
        "Date": raw["Date"].astype(str).str.strip(),
        "Time": raw["Time"].astype(str).str.strip() if "Time" in raw else float("nan"),
        "HomeTeam": raw["HomeTeam"].astype(str).str.strip(),
        "AwayTeam": raw["AwayTeam"].astype(str).str.strip(),
        "FTHG": pd.to_numeric(raw["FTHG"], errors="raise").astype(int),
        "FTAG": pd.to_numeric(raw["FTAG"], errors="raise").astype(int),
    })
    for c in ["B365H", "B365D", "B365A"]:
        out[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw else float("nan")
    return out[COLS].reset_index(drop=True)


def check_division(div, df, problems):
    n = N_TEAMS[div]; teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    expected = n * (n - 1)
    if len(teams) != n: problems.append(f"{div}: {len(teams)} clubs au lieu de {n}")
    if len(df) != expected: problems.append(f"{div}: {len(df)} matchs au lieu de {expected}")
    if df.duplicated(["HomeTeam", "AwayTeam"]).any(): problems.append(f"{div}: affiche en double")
    home, away = df.HomeTeam.value_counts(), df.AwayTeam.value_counts()
    bad = [t for t in teams if home.get(t, 0) != n - 1 or away.get(t, 0) != n - 1]
    if bad: problems.append(f"{div}: nombre de matchs domicile/exterieur incorrect pour {bad}")
    dates = parse_dates(df.Date)
    if dates.isna().any(): problems.append(f"{div}: {int(dates.isna().sum())} dates illisibles")
    elif not (dates.min() >= pd.Timestamp("2014-07-01") and dates.max() <= pd.Timestamp("2015-07-01")):
        problems.append(f"{div}: dates hors de la saison 2014/15 ({dates.min().date()} -> {dates.max().date()})")
    miss = int(df[["B365H", "B365D", "B365A"]].isna().any(axis=1).sum())
    print(f"  {div:3s} {DIVS[div]:10s} {len(df)} matchs, {len(teams)} clubs, "
          f"{dates.min().date()} -> {dates.max().date()}, cotes manquantes : {miss}")


def check_continuity(new, glob):
    """Clubs 2014/15 vs 2015/16: those that disappear must be the relegated clubs, those that appear the promoted ones."""
    print("\nContinuite des noms avec 2015/16 (a verifier : relegues 2014/15 et promus 2015/16) :")
    for div, lg in DIVS.items():
        a = set(new.loc[new.League == lg, "HomeTeam"])
        b = set(glob.loc[(glob.League == lg) & (glob.Season == NEXT_SEASON), "HomeTeam"])
        print(f"  {lg:10s} restent {len(a & b):2d} | relegues {sorted(a - b)} | promus {sorted(b - a)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="dossier contenant E0.csv, SP1.csv, D1.csv, I1.csv, F1.csv")
    args = ap.parse_args()

    print(f"Saison {SEASON} ({'fichiers locaux' if args.local else 'telechargement football-data.co.uk'})")
    parts, problems = [], []
    for div in DIVS:
        df = build_division(div, args.local)
        check_division(div, df, problems)
        parts.append(df)
    new = pd.concat(parts, ignore_index=True)

    glob = pd.read_excel(DATA / "global_5_leagues.xlsx", sheet_name="Global")
    if list(glob.columns) != COLS:
        problems.append(f"colonnes de global_5_leagues.xlsx inattendues : {list(glob.columns)}")
    if (glob.Season == SEASON).any():
        problems.append(f"global_5_leagues.xlsx contient deja la saison {SEASON}")
    check_continuity(new, glob)

    if problems:
        print("\nPROBLEMES DETECTES -- fichiers non ecrits :"); [print("  -", p) for p in problems]
        sys.exit(1)

    order = {lg: i for i, lg in enumerate(glob.League.drop_duplicates())}
    merged = pd.concat([new, glob], ignore_index=True)
    merged = merged.sort_values(["League", "Season"], key=lambda s: s.map(order) if s.name == "League" else s,
                                kind="stable").reset_index(drop=True)
    new.to_excel(DATA / "football_data_2014_15.xlsx", sheet_name="Global", index=False)
    merged.to_excel(DATA / "global_5_leagues_2014_26.xlsx", sheet_name="Global", index=False)
    print(f"\nEcrit : data/football_data_2014_15.xlsx ({len(new)} matchs)")
    print(f"Ecrit : data/global_5_leagues_2014_26.xlsx ({len(merged)} matchs = {len(glob)} + {len(new)})")


if __name__ == "__main__":
    main()
