# Switching the Dependent Variable: Customer Traffic as the Outcome

> **English edition.** The Chinese original is
> [FINDINGS_客流分析.md](FINDINGS_客流分析.md); both cover the same analysis and numbers.

**English summary.** The main analysis asks what predicts *survival*. This one asks what predicts
*traffic*, using check-ins per active month as the outcome. All volume-derived variables are removed
from the right-hand side, otherwise the model would predict traffic from traffic. Result: reputation
drives traffic (rating +21.8% per SD), spatial density raises both traffic and closure, and the two
dependent variables turn out to explain almost disjoint sets of behaviour.

---

## 1. Why this is not just swapping Y

The original dependent variable was `is_open`. Switching to traffic creates one dominant technical
risk: **circularity**. Review counts, check-in counts and tip counts *are* constituents of traffic,
so leaving them on the right-hand side would amount to predicting traffic from traffic.

A hard boundary was therefore enforced:

| Allowed as predictors | Excluded (volume measures) |
|---|---|
| Listing attributes: delivery, takeout, TV, reservations, outdoor seating, catering, good for kids, credit cards, bike parking, alcohol, WiFi | `n_reviews` |
| Price tier, hours completeness, open late | `n_tips` |
| **Competitors within 1 km**, city restaurant count | `total_checkins` (the outcome's numerator) |
| Rating, rating volatility | `mean_useful`, `engagement_index` |
| Category dummies (20), active-period midpoint year (control) | `tip_per_review`, `checkin_per_review` |

**Rule**: a predictor's value must not grow mechanically with the number of customers served.
"Share of negative reviews" is a composition measure (a quality signal) and may stay; "review count"
is volume and must go.

## 2. How traffic is defined

```
traffic intensity = total check-in events / active months
active months     = (last check-in - first check-in) / 30.44, floored at 1
```

- Source: `cleaned_checkin_expanded`, **8,515,582 timestamped check-in events** (Dec 2009 – Jan 2022)
- Covers 51,318 stores; median active span **68.7 months**; 88.4% of stores span ≥365 days
- Median 1.0 per month, mean 2.14, maximum 278 → heavy tail, so the model uses logs
- Values are capped at the 99.5th percentile (22.8/month) so a handful of venues cannot drive the fit

Dividing by active months is essential: **cumulative check-ins grow with store age**, so without
normalisation "open longer" would be misread as "busier".

## 3. Two traps that had to be ruled out first

### Trap A — the decline in check-ins is a platform effect, not a traffic decline

| Year | Check-ins | Stores with check-ins | Per store |
|---|---:|---:|---:|
| 2013 | 951,990 | 27,744 | 34.3 |
| **2015** | **1,103,125** | 30,332 | **36.4** |
| 2017 | 892,496 | 30,809 | 29.0 |
| 2019 | 692,079 | 30,620 | 22.6 |
| 2021 | 318,933 | 25,644 | 12.4 |

Check-ins peak in 2015 and fall to **29% of that peak by 2021**. That is **Yelp's check-in feature
losing usage**, not restaurants losing customers. Therefore:

- ✗ "check-ins in the last 12 months" cannot measure current traffic
- ✗ the time trend of check-ins cannot be read as a traffic trend
- ✓ only **average intensity within the store's own active period**, with the active-period midpoint
  year included as a time control

Check: `corr(log traffic intensity, midpoint year) = 0.069` — the platform trend barely contaminates
the intensity measure.

### Trap B — midpoint year is mechanically tied to survival

Stores whose active period centres on 2010–2012 show survival rates of only **2.5%–4.6%**, because
the check-in record simply stops when the store closes. This is **exactly the same trap** as
`last_review_date` in the main analysis. So `mid_year` is used **only as a control, never as a
predictor**.

## 4. Findings

### T1 — Structural factors explain 41% of the variation in traffic

OLS with log traffic intensity as the dependent variable, 43 features: **R² = 0.410**, n = 51,318.
For cross-sectional data that is a substantial share, so traffic is not a purely random quantity.

**A methodological issue that had to be handled first.** `corr(star rating, share of negative
reviews) = −0.937` — the two are near-collinear. With both in the model, the rating coefficient
became **−5.7%**, which is not "higher ratings mean fewer customers" but a **suppression artefact**
of collinearity. The model now keeps the rating only (interpretable and comparable with the survival
analysis), with the negative-review share as a sensitivity check: **rating +21.8% vs negative-share
−20.0%** — same magnitude, opposite sign, so the conclusion does not depend on which is chosen.

### T2 — The density paradox: denser means busier *and* more likely to close (core finding)

![Figure 23](charts/fig23_density_tension.png)

| Decile | Competitors within 1 km | Median traffic (per month) | Closure |
|---|---:|---:|---:|
| D1 (sparsest) | 4 | 0.49 | 23.7% |
| D5 | 30 | 0.97 | 31.0% |
| D8 | 76 | 1.53 | 37.4% |
| **D10 (densest)** | **558** | **2.47** | **45.2%** |

- In the model, `competitors within 1 km +1 SD → traffic +24.0%` (CI [+22.8, +25.3])
- On the same axis, closure rises from **23.7% to 45.2%** (about 1.9×)

**Business meaning**: a dense location does buy traffic (**5×**), but at roughly double the closure
rate. So absolute traffic is a bad metric; **traffic divided by competitive intensity** is the one
that matters. This explains why many owners feel "the place is always full yet we cannot survive".

### T3 — Reputation decides how many come; accessibility decides how long you last

![Figure 22](charts/fig22_traffic_drivers.png)

This is the most useful synthetic result of changing the dependent variable:

| Variable (+1 SD) | Effect on **traffic** | Effect on **survival** (main analysis) |
|---|---:|---|
| Star rating | **+21.8%** (CI [+20.4, +23.2]) | **≈ 0** (Spearman +0.038, sign unstable) |
| Negative-review share (sensitivity) | **−20.0%** | weak |
| Competitors within 1 km | +24.0% | closure 23.7% → 45.2% |
| Offers delivery | **−0.5%, not significant** (CI [−1.4, +0.5]) | **−28.6 pp closure** (strongest factor) |
| Price tier | +16.1% | — |

**The two columns barely overlap**, and that is the point:

> **Reputation (rating, negative reviews) significantly raises traffic but has almost no effect on
> survival; delivery accessibility has no effect on traffic at all (even slightly negative) yet is
> the strongest survival factor.**

The reason is straightforward: delivery revenue never walks through the door, so it produces no
check-ins — but it provides a revenue line independent of dine-in, which greatly improves survival.
Conversely, the rating affects "people willing to come in", but more people coming in does not equal
survival; rent, competition and cost structure decide that.

In the main analysis, "ratings barely predict survival" looked counter-intuitive. This resolves it:
**ratings are not useless — their effect shows up in traffic, not in survival.**

### T4 — At category level, high traffic goes with high competition and higher closure

![Figure 24](charts/fig24_category_traffic_closure.png)

Category-level Spearman **ρ = +0.66** (weighted +0.665) across 20 categories:

| Category | Relative traffic | Closure |
|---|---:|---:|
| Nightlife | **+143%** | 33.5% |
| Bars | +137% | 32.9% |
| American (New) | +108% | 40.0% |
| Korean | +91% | 40.0% |
| **Fast Food** | **−56%** | **14.7% (lowest)** |
| Pizza | −46% | 28.2% |
| Chinese | −42% | 27.2% |
| Burgers | −17% | 24.1% |

**This does not mean fast food is safer**: fast food has the lowest closure but also the lowest
traffic and the lowest average price — a different risk/return curve. The real counter-example is
**Indian**: traffic only +20% yet closure 45.4%. Category is not destiny; competitive position is.

### T5 — Traffic and survival form an inverted U, not "busier is better"

![Figure 25](charts/fig25_traffic_deciles_closure.png)

| Traffic decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Closure | 15% | 23% | 28% | 33% | 37% | 38% | 38% | **40%** | 38% | 36% |

The lowest band (stable small operators) closes at only 15%; mid-high bands peak at 40%; the top band
eases to 36%. **Structurally identical to the rating finding** — both extremes are clearly positioned
(very small neighbourhood stores, or genuine destination venues), while the middle is homogeneous and
most directly contested.

## 5. Side-by-side with the main analysis

| | Main analysis | This analysis |
|---|---|---|
| Dependent variable | `is_open` | traffic intensity (per month) |
| Strongest factors | delivery (−28.6 pp closure) | density (+24.0%), rating (+21.8%) |
| Role of rating | ≈ 0, **no survival value** | **+21.8%, real traffic value** |
| Role of delivery | **−28.6 pp, strongest survival factor** | −0.5% n.s., **no traffic value** |
| Model performance | AUC 0.763 | R² 0.410 |
| Serves | the owner deciding on delivery and site density | the owner deciding whether to chase traffic or margin |

**Combined recommendations**
1. Judge a site on **traffic ÷ competition**, not absolute traffic — dense districts deliver 5× the
   traffic but nearly 2× the closure.
2. **Survival comes from accessibility** (delivery); **traffic comes from reputation** (ratings,
   fewer negative reviews). They are different problems; raising your rating will not make you last
   longer.
3. Do not use "business is good lately" as a safety signal — high-traffic districts are precisely
   the high-risk ones.

## 6. Limitations

1. **Check-ins ≠ real footfall.** They are one voluntary Yelp action, skewed toward younger, more
   urban, more nightlife-oriented customers; coverage of Chinese restaurants and neighbourhood fast
   food is systematically thin.
2. **No time trends.** Check-in volume is dominated by the feature's life cycle (Trap A).
3. **Not causal.** Attributes come from listings; "has a full bar" marks what kind of venue it is,
   not an intervention worth +32.6% traffic.
4. **The denominator comes from the dependent variable itself.** Active months derive from first and
   last check-in, and closed stores' active periods end at closure. That is a reasonable definition
   of intensity *while operating*, but not a store's trading lifetime.
5. **Category effects mix location and clientele.** A category effect cannot be separated from "this
   kind of venue usually opens there".
6. **`mid_year` can only be a control** — it is mechanically tied to survival (Trap B).
7. **Missing attributes are information.** The Unknown group has systematically lower traffic;
   mean-filling dilutes the real difference, hence the careful phrasing "stores with more complete
   listings have higher traffic".
8. **No population denominator.** The 1 km neighbour count is a per-km² density, not a per-capita
   penetration rate.
