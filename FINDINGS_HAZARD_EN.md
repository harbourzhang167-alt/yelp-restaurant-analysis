# Discrete-Time Hazard Model: When Do Restaurants Go Quiet?

> **Chinese edition:** [FINDINGS_HAZARD.md](FINDINGS_HAZARD.md) — same analysis, same numbers.

**Why this exists.** The main analysis used the `is_open` snapshot, which can only say
"still open" or "closed" — it **cannot tell a store that lasted three years from one
that lasted fifteen**. The quarterly trajectory table provides an observed review
series per store, which makes a genuine time-to-event design possible.

**Headline.** A **sliding rating does not predict a store going quiet**
(HR 0.997, CI [0.980, 1.017]). What does predict it is the **rating level**
(HR 0.809), the **review-volume trend** (HR 0.777) and **offering delivery**
(HR 0.606). This is a hypothesis the data **refuted**, including one of my own.

---

## 1. First, a seventh data trap: `final_warning_data` is not store × month

Only one of the two time-series tables is usable:

| Table | Rows | Unique (store, period) | Duplicates | Usable |
|---|---:|---:|---:|---|
| `final_warning_data` | 4,724,683 | 1,825,219 | **2,899,464** | ❌ |
| `restaurant_quarterly_trajectory` | 986,088 | 986,088 | **0** | ✅ |

The "duplicates" in `final_warning_data` are **not exact copies** — for the same store
and month, `rolling_avg_stars` appears as both 4.5 and 4.8. That means the table is
actually at **review-event grain** (one row per review, carrying that month's rolling
statistic) despite being documented as store × month. Aggregating it as store × month
would double-count. It is therefore not used here.

## 2. Specification

**Discrete-time hazard**: pooled logistic regression on person-period rows, with the
event set to 1 only in the quarter a store's series stops. The fitted probability is
the conditional hazard $h(t\mid x)$. The baseline hazard enters as **period dummies**,
so no parametric shape is imposed.

| Element | Specification |
|---|---|
| Unit | store × observed quarter |
| Periods observed | median 15 quarters, max 68 |
| Event | **the review series stops in that quarter** ("goes quiet") |
| Events | **17,235** (33.0% of stores, matching the 33.05% closure rate) |
| Censoring | still open at the snapshot (35,051 stores), right-censored |
| Time axis | quarters since first observation (by sequence, tolerating gaps) |

### How this relates to `is_open`: validation only, never an input

The observable event is defined **from the review series itself**, while `is_open` is
mechanically tied to when the series stops (closed stores necessarily stop earlier).
Using `is_open` as a predictor would be predicting the outcome from the outcome — the
same discipline that made the main analysis drop `last_review_date`.

So:

- every predictor uses only information **strictly earlier** than the current quarter
  (one-quarter lags, a trailing four-quarter window, and a prior-four-quarter contrast)
- `business_stars` and `n_reviews` are deliberately excluded: they are computed over the
  **entire** review history, including the future
- `is_open` is used only at the end, as an **external check**

## 3. Results

![Figure 30](charts/fig30_hazard_ratios.png)

| Variable (+1 SD) | Hazard ratio | 95% CI | Significant |
|---|---:|---|---|
| **offers delivery** | **0.606** | [0.599, 0.615] | ✅ |
| review-volume trend (last 4 vs prior 4) | 0.777 | [0.767, 0.790] | ✅ |
| recent mean rating | 0.809 | [0.795, 0.824] | ✅ |
| enough history for a trend | 0.877 | [0.851, 0.899] | ✅ |
| recent review volume (log) | 0.926 | [0.911, 0.939] | ✅ |
| price tier | 0.996 | [0.983, 1.011] | ❌ |
| **rating trend (change)** | **0.997** | **[0.980, 1.017]** | ❌ |
| 1 km competitor count (log) | **1.176** | [1.158, 1.194] | ✅ |

### The core result: the rating trend carries no early-warning value

**This is the one place where the analysis refuted its own hypothesis.**

The main report listed "rate of rating decline" as a natural early-warning factor. It
was tested here and **does not hold**: `HR 0.997`, with the confidence interval tight
around 1.

![Figure 31](charts/fig31_hazard_survival_curves.png)

In the left panel, the three rating-trend curves (rising / flat / falling)
**coincide exactly** — identical survival at quarter 8 and quarter 24
(87.7% and 68.7%). In the right panel, delivery separates cleanly
(77.7% vs 47.2% at quarter 24, a **30.5 pp** gap).

> **Correct reading**: what matters for survival is the **absolute level** of the
> rating, not its **direction of change**. "My rating is falling" is not an early
> warning; "my rating is low" is.

### Two results that corroborate the earlier analysis

| This model (hazard) | Main analysis (logistic / descriptive) | Agreement |
|---|---|---|
| delivery HR **0.606** (strongest protective factor) | closure 19.5% vs 48.1% (−28.6 pp) | ✅ direction and magnitude |
| 1 km competitors HR **1.176** (risk factor) | closure 23.7% → 46.5% across density deciles | ✅ consistent |

Delivery is the strongest protective factor under **two entirely different modelling
approaches**, which raises confidence in that finding.

### Baseline hazard: not "newer is riskier"

![Figure 32](charts/fig32_baseline_hazard.png)

Risk peaks in the **5–8 quarter** band (2.73×) and then falls monotonically to the
49+ reference level. The first four quarters are *lower* (1.15×), which looks more like
a **left-truncation effect** — a store enters observation when it receives its first
review, so "newly observed" is not the same as "newly opened".

## 4. Fit and validation

| Metric | Value |
|---|---:|
| Person-period observations | 986,088 |
| Events | 17,235 |
| Log-likelihood | −82,617.2 |
| McFadden pseudo-R² | 0.0485 |
| In-sample AUC | 0.6944 |
| **AUC against `is_open` (a label the model never saw)** | **0.7045** |

A pseudo-R² of 0.0485 is low, but normal here: the model uses only **lagged
first- and second-order time signals** and carries no store-identity features, so most
of the variation is individual heterogeneity rather than observable trajectory.

The external validation AUC of **0.7045** shows that even without ever seeing
`is_open`, the stopping pattern of the review series alone recovers "which stores
eventually close" reasonably well.

## 5. Limitations

1. **The event is "going quiet", not "closing".** The two are strongly related but not
   the same thing: a closed store may still receive late reviews, and a trading store
   may go a long time without any.
2. **No true opening date.** The time axis is "quarters since first observed", so the
   early baseline hazard is affected by left truncation (see §3).
3. **Gaps are collapsed.** Only quarters with at least one review are observed, so the
   "quarter" axis is closer to active quarters than to calendar time.
4. **Static attributes are snapshots.** Delivery and price tier are taken from the
   collection-time business record; for stores that closed early, they may postdate
   closure.
5. **Still not causal.** The delivery HR of 0.606 must not be read as "adding delivery
   cuts risk by 39%", because better-run stores may be the ones able to offer it.
6. **The closure time is unknown.** The event quarter is when the series stops, while
   closure may come later — this is **interval censoring**, treated here as a point
   event.
7. **No time-varying macro controls.** The 2020 collapse in review volume raises that
   year's event rate; the model uses period dummies but no year fixed effects.

## 6. Outputs

| File | Contents |
|---|---|
| `analysis/18_hazard.py` | person-period construction, including the grain check |
| `analysis/19_hazard_model.py` | discrete-time hazard fit plus cluster bootstrap |
| `analysis/21_hazard_curves.py` | survival curves per dimension |
| `analysis/20_hazard_charts.py` | figures 30–32 |
| `08_hazard_person_period.csv.gz` | 986,088 person-period rows |
| `08_hazard_coefficients.csv` | hazard ratios with 95% CI |
| `08_hazard_curves.csv` | cumulative survival curves by dimension |
| `08_hazard_fit.md` | fit and validation metrics |
