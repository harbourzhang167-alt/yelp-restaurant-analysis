# What Keeps a Restaurant Alive — and What Makes It Profitable?

[![checks](https://github.com/harbourzhang167-alt/yelp-restaurant-analysis/actions/workflows/checks.yml/badge.svg)](https://github.com/harbourzhang167-alt/yelp-restaurant-analysis/actions/workflows/checks.yml)

**A survival + traffic + profitability analysis of 52,286 US restaurants.**
Yelp academic dataset · 17 tables · 42.4 million rows · Python (no scipy/statsmodels)

<p align="center">
  <img src="charts/fig23_density_tension.png" width="760">
</p>

---

## The question

Restaurant owners are usually told to chase two things: better ratings, and busier
locations. This project tests both against the one outcome that cannot be argued
with — **did the restaurant survive?** — and then asks a harder question:
*is survival even the right measure of success?*

I ran the same dataset three times with three different dependent variables:

| # | Dependent variable | What it answers |
|---|---|---|
| 1 | `is_open` (survived or not) | What separates restaurants that last from ones that close? |
| 2 | Check-ins per active month | What actually brings customers through the door? |
| 3 | Modelled cumulative profit | Is a long life worth more than a busy one? |

## Three findings

**1. Reviews drive traffic; delivery drives survival — and the two barely overlap.**

| Variable (+1 SD) | Effect on traffic | Effect on survival |
|---|---:|---|
| Star rating | **+21.8%** | ≈ 0 (Spearman +0.038, sign unstable across samples) |
| 1 km competitor count | **+24.0%** | closure 23.7% → 45.2% |
| Offers delivery | **−0.5% (not significant)** | **−28.6 pp** closure rate |

Delivery money never walks through the door, so it generates no check-ins — yet it is
the single strongest survival factor in the dataset. Star ratings are the
opposite: they move traffic, but say almost nothing about whether a restaurant lasts.
The full model reaches **AUC 0.763**; ratings alone give **0.523** (a coin flip).

**2. Density buys about five times the traffic, and nearly doubles the closure rate.**

| Density decile | Median traffic/month | Closure rate |
|---|---:|---:|
| D1 (sparsest) | 0.49 | 23.7% |
| D10 (densest) | 2.47 | 45.2% |

So "how busy am I" is a bad metric. **Traffic divided by competitive intensity** is
the one that matters.

**3. Survival and profitability can point in opposite directions.**

Under an explicitly parameterised profit model, the *lowest*-density decile shows
**−$276k** cumulative profit while the *highest*-density decile shows **+$1,075k** —
even though the low-density decile has the **lowest** closure rate. Long-lived,
low-traffic restaurants can be quietly losing money for years.

<p align="center">
  <img src="charts/fig27_survival_vs_profit.png" width="700">
</p>

## What this demonstrates

- **Two dependent variables, one dataset.** Reframing the target forced the whole
  feature set to change — volume variables had to leave the right-hand side.
- **Six data traps found and documented** before any modelling, e.g. a variable that
  is mechanically circular with the outcome (correlation 0.352), and a snapshot field
  repeated across historical rows that fabricates a "survival rate rising 64% → 99.7%".
- **Right-censoring corrected.** An early finding ("dense locations die younger") turned
  out to be a censoring artefact; restricted to closed stores the lifespan is flat
  across density (43–45 months).
- **Everything hand-rolled.** Spearman via rank transforms, logistic regression via
  IRLS in numpy, charts drawn with Pillow — no scipy, statsmodels or matplotlib.
- **Assumptions made visible.** Where the data is silent (rent, ticket price, margin),
  the model reports a *break-even condition* plus a joint sensitivity surface rather
  than pretending to estimate profit.

<p align="center">
  <img src="charts/fig20_method_chain.png" width="880">
</p>

## A street-level look

The points are real per-business coordinates; the basemap is stitched OpenStreetMap.

<p align="center">
  <img src="charts/maps/philadelphia-block.jpg" width="620">
</p>

## How it was built

```
BigQuery export → data audit → feature engineering → survival model
                                                     → density measurement (Haversine)
                → traffic model (volume variables excluded)
                → profit scenario + joint sensitivity → figures → report
```

The 29 figures and the full write-up are in this repository. Statistical work:

| Step | Method |
|---|---|
| Correlation | Pearson **and** Spearman side by side (relationships are non-monotonic and heavy-tailed) |
| Survival model | Logistic regression, numpy IRLS with ridge; AUC via Mann-Whitney ranks |
| Uncertainty | Percentile bootstrap, 600 replicates, clustered by business |
| Spatial density | Haversine neighbour counts at 0.5 / 1 / 2 km on real GPS coordinates |
| Profit model | Parameter grid over customers-per-check-in × margin × fixed cost × rent ratio |

## Documents

| File | Contents |
|---|---|
| [FINDINGS.md](FINDINGS.md) | Main analysis (survival) — six documented data traps, 12 limitations |
| [FINDINGS_客流分析.md](FINDINGS_客流分析.md) | Traffic as the dependent variable |
| [FINDINGS_盈利口径.md](FINDINGS_盈利口径.md) | Survival vs profit, break-even model, joint sensitivity |
| [01_profile.md](01_profile.md) | Data profiling and quality audit |
| [report-zh.docx](report-zh.docx) | 15-page explanatory report (Chinese), 11 figures |

## Honest framing

This was a **four-person academic group project** (BDAO, University of Auckland).
<!-- TODO: replace the line below with your actual contribution -->
*My contribution: [e.g. designed the two-dependent-variable framing, built the
survival and traffic models, and wrote the report].*

All findings are **correlational, not causal**. The strongest threat to the traffic
analysis is that check-in behaviour is itself uneven across venue types, so part of
the density effect may be measurement, not real footfall. The profit model's
monetary parameters are external assumptions, not estimates — that is stated
explicitly wherever its output appears.

## Data

Raw data is **not** included: it is 11.9 GB and the Yelp Data Use Agreement
prohibits redistribution. The repository contains aggregate results, figures and
the analysis code only. Per-business tables are deliberately excluded.

Map imagery © OpenStreetMap contributors, [ODbL](https://opendatacommons.org/licenses/odbl/).

## Repository checks

Three CI jobs run on every push (see [`.github/workflows/checks.yml`](.github/workflows/checks.yml)):

| Job | What it enforces |
|---|---|
| `secret-scan` | no credential material in tracked files — it matches key *signatures*, not the word "key", so the check does not flag its own source |
| `integrity` | every relative link and embedded figure resolves; no tracked file exceeds 50 MB |
| `syntax` | all analysis scripts byte-compile on Python 3.10 and 3.12 |

Both checkers also run locally:

```bash
python scripts/check_no_secrets.py
python scripts/check_links.py
```

## Licence

Code: MIT (see [LICENSE](LICENSE)).
