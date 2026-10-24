
These are the scripts behind the figures and numbers. They are included so the
methods can be inspected directly — every statistic here is hand-written rather
than delegated to a library.

## Why no scipy / statsmodels / matplotlib

| Instead of | This repo uses |
|---|---|
| `scipy.stats.spearmanr` | `rank()` + `corrcoef` |
| `statsmodels.Logit` | numpy IRLS with ridge penalty |
| `sklearn.roc_auc_score` | Mann-Whitney rank statistic |
| `matplotlib` | Pillow, via `chartkit.py` / `mapkit.py` / `flowkit.py` |

Also note: **do not use `pandas.corr(method="spearman")`** — it requires scipy and
will raise `ModuleNotFoundError` in this environment.

## Reading order

| File | What it does |
|---|---|
| `01_profile.py` | Profiles all 17 tables, audits data quality, documents six traps |
| `02_features.py` | Builds the per-business feature table (~125 columns) |
| `03_findings.py` | Survival analysis: univariate contrasts, logistic model, AUC, risk deciles |
| `03b_density.py` | Haversine neighbour counts on real coordinates |
| `11_traffic.py` | Switches the dependent variable to traffic; excludes all volume predictors |
| `12_traffic_charts.py` | Traffic figures |
| `13_profit_scenarios.py` | Cumulative-profit scenario model and break-even scan |
| `15_sensitivity.py` | Joint sweep over the unobservable parameters |
| `10_method_chain.py` | Renders the methodology diagram |
| `chartkit.py` / `mapkit.py` / `flowkit.py` | Plotting primitives (Pillow) |
| `tiles.py` / `streetcache.py` | OpenStreetMap tile stitching and Overpass caching |

## Running it

The scripts expect BigQuery-exported CSVs under `data/bigquery/`, which are not
published. They are included for transparency, not as a one-command reproduction.
Statistical sections (`03_findings.py`, `03b_density.py`, `11_traffic.py`,
`13_profit_scenarios.py`, `15_sensitivity.py`) read only from the feature table.
