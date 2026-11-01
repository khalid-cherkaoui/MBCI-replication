# MBCI: replication code and data

Code and data for the article

> Cherkaoui Rbati, K.; Lahby, M. **A Match-Based Competitiveness Index for Football Leagues: Integrating Goal Intensity, Outcome Surprise, and Season Stakes.** *Sports* (MDPI), 2026 (revised version).

The Match-Based Competitiveness Index (MBCI) scores every league match from three normalised components:
match intensity $G_k$ (from the scoreline), outcome surprise $U_k$ (one minus the pre-match Elo–Davidson
probability of the observed result) and a season stakes weight $W_k$ (importance of the match for the title,
European qualification and relegation). The repository reproduces every number, table and figure of the article
for the 17,732 matches of the Big Five leagues, 2015/16–2024/25.

## Repository structure

```
MBCI-replication/
├── mbci/                          index, Sections 4–7 of the article (except 7.6)
│   ├── mbci_core.py               data loading, standings, stakes W_k, HHI, market odds (Shin), terciles
│   ├── elo_davidson.py            Elo–Davidson ratings and outcome surprise U_k, index MBCI*_k
│   ├── 00_fetch_2014_15.py        downloads the 2014/15 season (rating initialisation); already done
│   ├── 01_article_numbers.py      every number and table of the article, with an automatic check
│   ├── 02_figures.py              Figures 1–7 and 9
│   ├── run_all.py                 runs 01 and 02
│   ├── README_fr.md               detailed documentation (French)
│   ├── data/                      input data (see "Data sources")
│   └── outputs/                   reference results produced by run_all.py
└── attendance/                    Section 7.6: validation against stadium attendance
    ├── validation_affluence_mbci.ipynb   data collection, data verification and analysis (Table 6, Figure 8)
    └── sorties_affluence/         stadium capacities (input) and reference results
```

## Reproducing the results

Python 3.11 or later. Tested with Python 3.13 and the package versions in `requirements.txt`.

```bash
pip install -r requirements.txt
cd mbci
python run_all.py          # about 3 minutes
```

`run_all.py` regenerates `mbci/outputs/` and compares 232 numbers of the article with the computed values;
the log ends with `CONTROLE CONTRE L'ARTICLE : 232/232 chiffres conformes` (all numbers match).
Bootstrap and permutation results use fixed seeds and are identical from one run to the next.

The attendance analysis is a notebook, run from its own folder:



It needs the `mbci/` folder next to it. On the first run it downloads the fixture lists of FBref with the
`soccerdata` package (Google Chrome is required by soccerdata's FBref reader; the download takes about an hour
because of FBref's rate limits). The raw download is then cached in
`attendance/sorties_affluence/fbref_calendrier_brut.xlsx` and re-read on later runs (about 10 minutes).
The Tobit model of Table 6 is estimated by numerical optimisation, so its last decimals can differ
slightly between runs; the values reported in the article are unaffected.

## Where to find each result of the article

| Article | File | Output |
|---|---|---|
| Section 4 (index), 4.4 (Elo–Davidson parameters) | `mbci/elo_davidson.py`, `mbci/mbci_core.py` | `outputs/numbers.json` |
| Section 6 (data) | `mbci/data/`, `mbci/00_fetch_2014_15.py`; `attendance/` notebook, Part 2 | `attendance/sorties_affluence/controle_donnees.xlsx` |
| Table 3, Table 4 (Section 7.1) | `01_article_numbers.py` | `table3_descriptive.csv`, `table4_top_matches.csv` |
| Section 7.2, Figures 1–5 (cluster bootstrap of the season profile) | `01`, `02` | `uncertainty_terciles.csv`, `figures/fig_distribution_*.pdf` |
| Section 7.3, Table 5, Figure 6 (bootstrap intervals, permutation tests) | `01`, `02` | `table5_leagues.csv`, `uncertainty_league_means.csv`, `uncertainty_league_pairs.csv`, `figures/fig_mbci_seasons.pdf` |
| Section 7.4, Figure 7 | `01`, `02` | `figures/fig_wk_progress.pdf` |
| Section 7.5 (HHI, panel specification, season bootstrap) | `01` | `article_numbers_report.txt`, `numbers.json` |
| Section 7.6, Table 6, Figure 8 (attendance) | `attendance/validation_affluence_mbci.ipynb` | `tableau_article_affluence.xlsx`, `fig_attendance_stakes_by_fill.pdf`, `resultats_validation.xlsx` |
| Section 7.7, Figure 9 (betting market) | `01`, `02` | `validation_calibration.csv`, `figures/fig_uk_calibration.pdf` |
| Section 7.8, Table 7 (robustness) | `01` | `table7_robustness.csv`, `grid_alpha_lambda_c.csv` |
| Match-level and league-season datasets | `01` | `mbci_match_level.xlsx`, `mbci_season_level.xlsx` |
| Check of every number against the article | `01` | `article_numbers_check.csv` |

Unless stated otherwise, output files are in `mbci/outputs/` or `attendance/sorties_affluence/`.

## Data sources

| Data | Source | In this repository |
|---|---|---|
| Results, dates, rounds and Bet365 pre-match odds, 2014/15–2025/26 | [football-data.co.uk](https://www.football-data.co.uk) | `mbci/data/global_5_leagues*.xlsx`, `football_data_2014_15.xlsx` |
| European berths and relegation-zone sizes by league-season | compiled by the authors from the UEFA access lists and national league regulations | `mbci/data/UEFA_Simple_1_decale.xlsx`, `Relegation_Simple.xlsx` |
| Match attendance, venue and kick-off time | [FBref](https://fbref.com), retrieved with [soccerdata](https://github.com/probberechts/soccerdata) 2026.43.0 on 27 October 2026 | not redistributed (see below); downloaded by the notebook |
| Stadium capacity by club and season | English Wikipedia season articles, tables "Stadiums and locations", accessed 31 October 2026 | `attendance/sorties_affluence/capacites_stades_wikipedia.xlsx` (source URL on every row) |

FBref's data-use policy restricts the redistribution of its data, so the match-level attendance files are not
included: `fbref_calendrier_brut.xlsx` (raw download) and `donnees_affluence_mbci.xlsx` (merged dataset) are
recreated by the notebook and excluded by `.gitignore`. The verification tables and results derived from them
are included.


## Licence and citation

The code is released under the MIT licence (`LICENSE`). Third-party data remain subject to their providers'
terms; the stadium-capacity table, derived from Wikipedia, is under CC BY-SA 4.0. If you use this code or data,
please cite the article (see `CITATION.cff`).
