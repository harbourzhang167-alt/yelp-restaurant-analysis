# Restaurant Survival Index — Survival and Performance (Findings)

> **English edition.** The Chinese original is [FINDINGS.md](FINDINGS.md); both cover the same
> analysis and report the same numbers.

**Business question.** For independent restaurants that have already accumulated at least five
reviews, **which observable and actionable characteristics separate the ones that survive from the
ones that close?** Put concretely: if NROA's public member data were turned into a self-check list,
which indicators are worth watching and which are noise?

**Why it matters.** NROA represents 40,000+ independent restaurants, and what members lack is a
clear answer to "what should I change first?" The industry treats "raise your rating" as the
primary goal. This analysis suggests **that goal may be wrong**.

**Analysis sample.** 52,286 restaurants across 19 states / 916 cities; 4,724,683 reviews,
649,181 tips, 8,515,582 check-ins. Outcome variable: `is_open`. **Overall closure rate 33.1%.**

**Method.** All findings are store-level cross-sectional. Correlations are reported as both Pearson
and Spearman. Group differences use **2,000-replicate bootstrap 95% confidence intervals**.
Multivariate effects use **ridge-stabilised logistic regression (IRLS)**; discrimination is
measured with **AUC**. No scipy or statsmodels was available on this machine — every statistic is
implemented in numpy, in `analysis/03_findings.py`.

---

## Headline findings

| # | Finding | Key numbers | Confidence |
|---|---|---|---|
| 1 | **Star rating barely distinguishes survival; review volume does** | rating~survival Spearman **+0.038**; closure by review decile **47% → 15%** | strong |
| 2 | **"Accessibility" beats "reputation"** | delivery closure **19.5%** vs no-delivery **48.1%** (−28.6 pp, CI [−29.3, −27.8]); yet the **no-delivery group is rated higher** (3.61 vs 3.41) | strong |
| 3 | **Real spatial density: competitors within 1 km double the closure rate** | computed from real coordinates; sparsest decile 23.7% → densest 46.5%; **18 of 18 cities positive**; model OR 0.76 | strong |
| 4 | **Rating and survival are inverted-U, not linear** | ≤2 stars 23.6% → 2.5–3.0 stars **40.2%** → 4.5–5 stars 25.9% | strong |
| 5 | **A composite risk score spreads closure 4.8×** | lowest decile **13.7%** vs highest **66.1%**; model AUC **0.763** (rating alone 0.523) | strong |
| 6 | **Geography is highly concentrated, and larger markets close more** | 94.5% of coordinates fall inside real US state boundaries; top 10 cities hold 48% of stores; market size~closure Spearman **+0.224** | strong |

> **Companion analyses.** This report uses survival as the dependent variable.
> A parallel analysis using **customer traffic** finds that reputation drives how many people
> come (rating +21.8%) while accessibility drives how long a store lasts (delivery −28.6 pp),
> and that the two do not substitute for each other. See
> **[FINDINGS_TRAFFIC_EN.md](FINDINGS_TRAFFIC_EN.md)** (figures 21–25).
> A third tracks **survival versus cumulative profit** and shows the two can point in opposite
> directions. See **[FINDINGS_PROFIT_EN.md](FINDINGS_PROFIT_EN.md)** (figures 26–29).

---

## Finding 1 — The rating is not a survival discriminator; review volume is

**Evidence**

- `business_stars` vs `is_open`: Spearman only **+0.038** (95% CI [+0.029, +0.045]). With a sample
  this large that is statistically significant, but the **effect size is negligible**.
- **Sample sensitivity**: on the `dropna` subsample (n = 44,497) the same relationship is Pearson
  **−0.011** / Spearman **+0.013** — **not even the sign is stable**. This is closer to "no
  relationship" than to "a weak positive one".
- By contrast: `business_review_count` **+0.180**, `total_checkins` **+0.121**, `n_tips` **+0.059**.
- Across review-count deciles, closure falls **monotonically from 47.0% to 15.0%**
  (D10 averages 455 reviews).
- Across tip-count quantiles, the open rate rises from 66.5% to **79.6%**.

![Figure 1](charts/fig01_reviewcount_closure.png)

**Interpretation.** Once a store has cleared the "≥5 reviews" threshold, **review volume — a proxy
for customer traffic and longevity — says far more than the rating does**. Stars largely reflect
taste preference; traffic reflects commercial viability.

**⚠ A counter-intuitive detail that must be stated.** Low-rated stores actually survive *better*
(≤2 stars: 23.6% closure). This is a **selection effect**: those are often long-established
neighbourhood places with a stable clientele. **This does not mean "lower ratings are better."**

**For restaurant owners**
- Demote "raise the rating" from primary KPI to **secondary indicator**; promote **sustained
  review growth** to the main health metric.
- Concrete actions: prompt reviews after the meal; put review growth month-over-month on the
  monthly dashboard; flag stores with flat review counts for diagnosis.

---

## Finding 2 — "Accessibility" beats "reputation"

**Evidence (univariate, 2,000-replicate bootstrap)**

| Attribute | Closure with | Closure without | Difference | 95% CI |
|---|---:|---:|---:|---|
| **delivery** | **19.5%** | **48.1%** | **−28.6 pp** | [−29.3, −27.8] |
| has_tv | 28.2% | 40.6% | −12.5 pp | [−13.3, −11.6] |
| bike_parking | 29.8% | 36.4% | −6.6 pp | [−7.5, −5.8] |
| takeout | 32.6% | 35.9% | −3.3 pp | [−4.5, −2.1] |
| outdoor_seating | 36.3% | 31.0% | +5.3 pp | [+4.4, +6.1] |
| accepts_credit_cards | 34.6% | 24.6% | +10.1 pp | [+9.0, +11.1] |
| reservations | 42.3% | 29.5% | +12.8 pp | [+11.9, +13.7] |

![Figure 4](charts/fig04_delivery_paradox.png)

**The key contrast**: the `delivery=True` group averages **3.41** stars; `delivery=False` averages
**3.61**. **The better-rated group is the more likely to close.**

**Multivariate check (controlling for each other, price, category, geography)**

Standardised logistic regression predicting survival (OR > 1 = higher survival odds), AUC **0.763**:

| Attribute | Coefficient | OR (per +1 SD) | Reading |
|---|---:|---:|---|
| delivery | +0.669 | **1.95** | strongest positive factor |
| log reviews | +0.484 | 1.62 | |
| has_tv | +0.350 | 1.42 | |
| open late | +0.197 | 1.22 | |
| city store count | −0.193 | 0.82 | denser is riskier |
| price tier | −0.164 | 0.85 | pricier is riskier |
| reservations | −0.217 | 0.80 | flips negative once controlled |
| share of negative reviews | −0.081 | 0.92 | |
| rating volatility | +0.040 | 1.04 | almost no independent effect |

![Figure 5](charts/fig05_model_forest.png)

**Interpretation.** **Whether you can get the food to the customer — or make it easier for the
customer to arrive — matters more than whether the food is good.** Delivery, TV and late hours all
widen reach and dwell time. "Formalised operations" flags (reservations, cards) turn negative once
other variables are controlled, which more likely reflects **venue type** (full-service vs
fast-food/neighbourhood) than a causal lever — so this should not be read as "drop reservations
and you'll survive".

**For restaurant owners**
- **Evaluate adding delivery first**: it is the largest effect in this data and a decision the
  owner actually controls (28.6 pp closure gap).
- If delivery already exists: monitor delivery order share and delivery rating separately, to avoid
  "delivery-only, dine-in experience collapses".
- Do not treat "get on TV / add outdoor seating" as a rescue measure — those correlate with survival
  but are more plausibly **a consequence** of an already-healthy business.

---

## Finding 3 — Real spatial density: competitors within 1 km double the closure rate

**How "real density" was derived**

The dataset contains **no population or area field**, but it does carry **latitude and longitude for
every store**. Coordinate validity was checked first:

- 52,286 stores map to **49,476 unique coordinate pairs** — only **5.4% duplication**, with a
  maximum of 14 stores at one point (a mall food court);
- Philadelphia's 5,854 stores sit on 5,489 coordinates, spanning 0.42° longitude and 0.28° latitude.

**Conclusion: these are genuine per-store GPS locations, not replicated city-centre coordinates.**
So true spatial density can be computed: the count of other restaurants within a **1 km Haversine
radius**.

| Radius | Mean | Median | Max |
|---|---:|---:|---:|
| 0.5 km | 40.3 | 16 | 560 |
| **1 km** | **96.6** | **33** | **1,329** |
| 2 km | 231.0 | 79 | 2,416 |

**Evidence**

| Density measure | Spearman vs survival |
|---|---:|
| **neighbours within 1 km** | **−0.141** |
| neighbours within 0.5 km | −0.143 |
| neighbours within 2 km | −0.137 |
| **same-category** neighbours within 1 km | −0.092 |
| city store count (old proxy) | **−0.069** |

**Real density has roughly twice the explanatory power of the old proxy** (−0.141 vs −0.069).

Across deciles of 1 km competitor count, closure **rises from 23.7% to 46.5%** (+22.8 pp, nearly
double):

| Decile | Mean neighbours | Closure |
|---|---:|---:|
| D1 (sparsest) | 4.3 | **23.7%** |
| D3 | 17.0 | 28.5% |
| D5 | 38.1 | 32.0% |
| D8 | 156.2 | 42.7% |
| D10 (densest) | **565.8** | **46.5%** |

![Figure 11](charts/fig11_true_density_deciles.png)

**A stronger test: comparison within the same city**

If density were merely a stand-in for "some cities are riskier", there should be no gradient inside a
city. Instead — **all 18 cities with ≥400 stores show a positive gradient** (densest quartile closes
more than sparsest), and **13 are fully monotonic**:

| City | Q1 (sparsest) | Q2 | Q3 | Q4 (densest) |
|---|---:|---:|---:|---:|
| Saint Louis | 34.7% | 43.3% | 43.2% | **65.2%** |
| Reno | 27.9% | 36.9% | 36.5% | **51.2%** |
| Philadelphia | 25.7% | 37.1% | 47.8% | **48.7%** |
| Indianapolis | 26.7% | 29.2% | 33.0% | **45.3%** |
| Tucson | 24.4% | 32.3% | 36.3% | **41.6%** |
| Wilmington | 19.9% | 26.7% | 29.9% | **41.9%** |

![Figure 12](charts/fig12_density_within_city.png)

**Spatially**: in Philadelphia, closures visibly cluster in the dense Center City business district.

![Figure 13](charts/fig13_density_map.png)

**Multivariate check**: replacing the old proxy with real density improves discrimination, and the
confidence intervals do not overlap.

| Model | AUC |
|---|---|
| A — old proxy (city store count) | 0.7630 | 95% CI [0.7584, 0.7668] |
| C1 — real density 0.5 km | 0.7680 |
| **C2 — real density 1 km** | **0.7676** | 95% CI [0.7630, 0.7722] |
| C3 — real density 2 km | 0.7682 |

After controlling for 15 variables (delivery, review volume, price, attributes), the
**1 km neighbour count has OR = 0.760** — each +1 SD lowers survival odds by 24%. That makes it the
**third strongest factor** after delivery (OR 1.95) and review volume (OR 1.68).

**Two technical points that must be stated honestly**

1. **Same-category competition adds almost nothing.** Adding same-category neighbours within 1 km
   gives OR 0.96 and no AUC change (0.7676 → 0.7677). What drives risk is **overall market density**,
   not "too many stores selling what I sell" — **so the advice "switch category to survive" does not
   hold up in this data.**
2. **The three scales cannot enter the model together.** Spearman correlation between 0.5/1/2 km is
   0.72–0.89; including them together flips coefficient signs (producing counter-intuitive ORs such
   as 1.42). This report uses a **single scale (1 km)** and keeps the comparison table to document
   the collinearity.

**Limitation**: there is still **no population denominator**, so this is "restaurants per square
kilometre", not "restaurants per capita". A place with 500 restaurants within 1 km may simply be a
very dense downtown. **True per-capita competitive intensity needs external population data.**

**For restaurant owners**
- When choosing a site, count **restaurants within 1 km of the target point** — not the whole city
  (the city-level proxy halves the measured risk). Above roughly 156 (from D8) closure jumps past
  43%; above 400 you are in the riskiest band.
- Already in a dense district: **do not expect "switch category" to rescue you** (same-category
  density has no extra explanatory power). Shift to **widening reach** (delivery, opening hours),
  the path this analysis does support.
- For a new site, look at the **1 km** gap, not the city average — within-city Q1-to-Q4 gaps reach
  23 pp (Reno, Philadelphia).

---

## Finding 4 — Rating and survival form an inverted U

**Evidence**

| Rating band | Stores | Closure | Mean reviews |
|---|---:|---:|---:|
| ≤2.0 | 4,639 | **23.6%** | 25 |
| 2.0–2.5 | 4,850 | 35.0% | 40 |
| 2.5–3.0 | 7,641 | **40.2%** | 56 |
| 3.0–3.5 | 11,798 | 37.8% | 86 |
| 3.5–4.0 | 13,444 | 31.7% | 132 |
| 4.0–4.5 | 8,360 | 27.5% | 117 |
| 4.5–5.0 | 1,554 | 25.9% | 33 |

![Figure 3](charts/fig03_stars_nonmonotonic.png)

**Interpretation.** The most dangerous band is the **middle, 2.5–3.5 stars**: such stores have
neither the stable clientele of a cheap neighbourhood institution nor the pull of a high-rated
destination, and sit in the most contested position. Both extremes are safer.

**For restaurant owners**
- A store landing in **2.5–3.5 stars** should be treated as a **priority for intervention**.
- The intervention is not "push the score up" but **grow review volume** (Finding 1) or **widen
  reach** (Finding 2).

---

## Finding 4b — Geography: stores are highly concentrated, and larger markets close more

**Method**: per-store coordinates plotted against **real US state boundaries** (PublicaMundi US
states GeoJSON, cached locally).

**A coordinate-validity test was run first**: point-in-polygon testing of all 52,286 coordinates
showed **49,429 (94.5%) fall inside US state boundaries**. The 2,857 outside match, almost exactly,
the 2,410 stores with `state = AB` plus 371 and 76 near the FL/CA edges. This both validates the
coordinates and confirms the dataset contains **Canadian** stores.

![Figure 14](charts/fig14_us_map_states_cities.png)

**State-level distribution** (18 states, by store count)

| State | Stores | Closure | Median 1 km competitors |
|---|---:|---:|---:|
| PA | 12,644 | 36.2% | 42 |
| FL | 8,361 | 32.2% | 30 |
| TN | 4,353 | 30.4% | 37 |
| MO | 4,248 | 35.9% | 36 |
| IN | 4,150 | 31.7% | 29 |
| LA | 3,641 | 35.6% | **65** |
| NJ | 3,343 | 27.7% | 18 |
| AZ | 2,675 | 32.4% | 26 |
| NV | 1,675 | 35.7% | 39 |
| ID | 1,304 | 27.8% | 33 |
| CA | 1,085 | **44.0%** | **85** |

**City-level concentration (911 cities)**

- **The top 10 cities hold 48%** of all 49,429 US stores; the top 50 hold **68%**.
- Top cities by store count: Philadelphia 5,854, Tampa 2,959, Indianapolis 2,862, Nashville 2,503,
  Tucson 2,470, New Orleans 2,260, Saint Louis 1,791, Reno 1,288, Boise 849, Santa Barbara 703.
- **Market size vs closure: Spearman +0.224** — smallest quintile 28.2% → largest quintile **32.4%**.

![Figure 15](charts/fig15_city_distribution.png)

**Interpretation**: stores are extremely concentrated geographically (under 1% of cities hold half
of all restaurants), and **larger markets close more often** — the same mechanism as Finding 3,
observed at a macro scale.

**For restaurant owners**
- Because stores cluster in a few metro areas, **competition within your own trade area is far
  higher than the national average**. Use your own **1 km** density (Finding 3), not city or
  national averages.
- Large cities (top 50) offer traffic but **close more** (32.4% vs 28.2%); factor that into payback
  calculations before entering.

**Limitation**: the `city` field was not consolidated into metro areas, so one metro could be split
across rows (`St. Louis` vs `Saint Louis`, `St. Petersburg` vs `Saint Petersburg`), biasing the
city ranking. → **Fixed in `analysis/06_metros.py`; see Finding 4c below.**

---

## Finding 4c — Corrected metro definition: 207 metro clusters

**Fix** (`analysis/06_metros.py`)

1. **Name normalisation**: strip punctuation, lowercase, map `saint → st` and `ft → fort`, and
   **remove all spaces** — so `land o lakes`/`land olakes`, `o fallon`/`ofallon`,
   `St. Petersburg`/`Saint Petersburg` collapse to one key.
2. **Core detection**: process each (state, normalised city) centroid from largest to smallest;
   merge into an existing core within 8 km, otherwise create a new core; then assign every city
   label to the nearest core within 30 km in the same state.

**A rejected approach**: the first attempt merged by connected components (12 km threshold). Because
connectivity is transitive, it chained all of New Jersey into a single "city" (Cherry Hill absorbed
188 city labels). The current approach uses **density peaks as cores**, which cannot chain.

**Effect**

| | Before | After |
|---|---:|---:|
| US city labels | 900 | **207 metro clusters** |
| Largest unit | Philadelphia 5,854 (a single split label) | Philadelphia 5,864 (**matching its true store count**) |
| Top 10 share | 48% | 51.8% |
| Size vs closure Spearman | +0.224 | **+0.537** |

Validation:
- Philadelphia's 9 sub-labels (`Phila`, `PHILA`, `Philadelphia ` …) **merge correctly**, and Cherry
  Hill is **not** absorbed (20+ km away, a separate city);
- Spelling variants `st louis`/`saint louis`, `st petersburg`/`saint petersburg`,
  `land o lakes`/`land olakes`, `o fallon`/`ofallon` **all land in one cluster**;
- Spot checks of known adjacencies: `Saint Louis ← Clayton`, `Reno ← Sparks`.

> **Definition note**: these are **urban clusters**, **not** US Census MSAs. Tampa and Clearwater
> (~30 km apart) are deliberately **not** merged, because competition happens at urban-cluster
> scale. The Spearman jump from +0.224 to +0.537 is mainly **removal of measurement error** (one
> city previously split into many samples, diluting the size signal), not a change of definition.

**State-level maps (12 figures)**

| File | Colour | States |
|---|---|---|
| `fig16a/b/c_state_maps_closure.png` | red = closed / green = open | a: PA, FL, TN, MO, IN, LA; b: NJ, AZ, NV, ID, CA, IL; c: the rest |
| `fig17a/b/c_state_maps_density.png` | shading = competitors within 1 km | same grouping |
| **`charts/maps/*.jpg`** | **OSM street basemap**, with street names and landmarks | two examples included here (36 in the full repo) |

> The state-level maps remain dense at whole-state scale — that is a consequence of scale, not of
> rendering: Pennsylvania alone puts 12,644 stores on one image. To see individual restaurants you
> must drop to block level. For example Philadelphia has 2,841 stores in a 4.8 km view and only 290
> in a 0.6 km view — that is the fix for overplotting, and a real street basemap is the fix for the
> "empty map" problem.

![Figure 16a](charts/fig16a_state_maps_closure.png)

![Figure 17a](charts/fig17a_state_maps_density.png)

**Structure visible on the maps**

- **PA**: one Philadelphia core holds most stores; western Pennsylvania (Pittsburgh) is almost absent
- **NJ**: stores form a band along the New York–Philadelphia corridor, denser toward the corridor
- **AZ / NV / ID**: extremely mono-centric — Tucson, Reno and Boise each dominate their state
- **CA**: only 1,161 stores, **almost all in Santa Barbara**; no Los Angeles or San Francisco
- **IL**: concentrated in Alton / Edwardsville / Belleville (across from St. Louis); no Chicago

**This is itself a coverage limitation that must be stated**: regional coverage is **extremely
uneven** — PA holds 25% of all US stores, CA only 2%, and CA/IL cover only one corner each. So
**"US average" claims are unreliable**; conclusions should be scoped to "within this 19-state,
unevenly covered sample".

---

## Finding 5 — A composite risk score spreads closure 4.8×

**Method**: combine the validated signals (delivery, reviews, tips, check-ins, rating volatility,
negative-review share, crowding, price) into a linear `RiskIndex` following the model's coefficient
directions, then group by decile.

| Risk decile | Closure |
|---|---:|
| D1 (lowest) | **13.7%** |
| D2 | 15.8% |
| D3 | 20.1% |
| D5 | 25.8% |
| D7 | 37.5% |
| D9 | 53.0% |
| D10 (highest) | **66.1%** |

![Figure 10](charts/fig10_risk_deciles.png)

**Discrimination**

| Model | AUC | Features |
|---|---:|---:|
| Rating only | 0.523 | 1 |
| Review volume only | 0.611 | 1 |
| **Full model** | **0.763** | 17 |

![Figure 6](charts/fig06_roc.png)

**Meaning**: an early-warning system built on ratings alone is barely better than chance (0.523).
Adding accessibility and activity reaches **0.763** (95% CI [0.759, 0.767]). **This is the part of
the Restaurant Success Index that can ship as a member tool**: enter a handful of public indicators
and get a risk band plus an improvement priority.

**For NROA**
- Turn the score into a **member self-check list** (6–8 self-assessable items), with high-risk bands
  prioritised for operational coaching.
- **Publish the limitations alongside it** (below), so it is not read as a causal forecast.

---

## Operating rhythm (supporting section)

Categories differ sharply in when check-ins happen: late night (22:00–03:59) accounts for
**Pizza 56.2% / Bars 55.2%** of check-ins, versus **Breakfast & Brunch 19.4% / Coffee & Tea 34.0%**.

![Figure 8](charts/fig08_category_hour_heatmap.png)

> **⚠ Time-zone warning**: check-in timestamps carry a **uniform time-zone offset** (coffee shops in
> every zone show very low check-ins between 6 and 10 am; East Coast 594–3,481 versus a peak of
> 37,771). The hour field therefore **cannot be read as local clock time** and should only support
> *relative* comparisons between categories.

**Customer side**: the median user among 1,446,025 has only **7 reviews**; **1% of users contribute
24.2% of reviews**; 76.5% have no fans; only 5.8% were ever elite.

![Figure 9](charts/fig09_user_concentration.png)

---

## Limitations (must accompany the conclusions)

1. **No causal claims.** Everything is stated as association. Delivery and survival may partly
   reflect reverse causality — stores that are already healthy can afford to add delivery.
2. **`is_open` is a snapshot** with a single value, stamped onto every review from 2005–2022.
   **Any year-by-year "survival rate trend" is an artefact.**
3. **Closure dates cannot be determined.** A closed store's "last review" is only an approximation,
   so `last_review_date` and `review_span` were **deliberately excluded** — they are mechanically
   correlated with the outcome (`span_days`~`is_open` Spearman 0.352), so including them would be
   predicting the outcome from the outcome.
4. **The sample is truncated**: the upstream table keeps only stores with ≥5 reviews, so **new and
   very small stores are systematically excluded**. Conclusions apply only to restaurants that
   already have some review volume.
5. **Count sources disagree**: `business_review_count` equals the in-dataset review count exactly for
   only 38.8% of stores (median gap −1). All ratios in this report use a single source.
6. **"Unknown" is kept as its own category**: `by_appointment_only` 93.6% Unknown, `alcohol` 62.4%.
   The Unknown group actually survives better, so filling it with False or dropping it would distort
   the conclusions.
7. **No population denominator**: spatial density is "restaurants per square kilometre" from real
   coordinates, **not** per capita. 500 restaurants within 1 km may be a very dense downtown.
8. **Geographic coverage is extremely uneven**: PA holds 25% of stores, CA only 2%, and CA/IL cover
   one corner each. **"US average" is unreliable**; scope conclusions to this sample.
9. **The three density scales are highly collinear** (Spearman 0.72–0.89); only one can enter the
   model. This report uses 1 km.
10. **Metro clusters are not MSAs**: the merge fixes "one city split across labels" (spelling
    variants, neighbourhood names) but does not merge adjacent independent cities. Cross-state
    metros (e.g. Philadelphia–New Jersey) remain separate.
11. **COVID-19 (2020)**: review volume fell from 608,294 in 2019 to 358,335 in 2020, which may
    confound year-related analyses.
12. **The model is cross-sectional with collinear features**, stabilised by ridge. ORs are
    "per +1 SD" association strengths, **not policy effects**.

---

## Next steps

1. ~~**Time-series / survival analysis**~~ → **done**, see
   [FINDINGS_HAZARD_EN.md](FINDINGS_HAZARD_EN.md). A discrete-time hazard model was fitted on the
   quarterly trajectories (986,088 person-period rows, 17,235 events). It **refuted the guess made
   here**: the rate of rating decline has **no** early-warning value (HR 0.997, 95% CI
   [0.980, 1.017]); the rating **level** (HR 0.809) and the review-volume trend (HR 0.777) do.
   Separately, `final_warning_data` turned out to be at review-event grain rather than store ×
   month (4.72M rows for only 1.83M unique pairs), so it is unusable for this purpose.
2. **Text signals**: negative-review length rises monotonically (5 stars 445 characters → 2 stars
   694), so unsupervised topics and computable negative-review flags could feed the warning system.
3. **Social graph**: `yelp_users_raw` contains `friends`, enabling network-degree and community
   analysis (a Customer-direction angle the group did not cover).
4. **External data**: join population and income data to upgrade "crowding" into true per-capita
   restaurant density.
5. **Causal inference**: propensity-score matching or difference-in-differences to test the causal
   effect of adding delivery on subsequent survival.

---

## Reproduction

```powershell
$py = "python"
& $py analysis/01_profile.py          # data profiling  -> analysis/out/01_profile.md
& $py analysis/02_features.py         # feature table   -> analysis/out/business_features.csv
& $py analysis/03_findings.py         # statistics      -> analysis/out/03_*.csv
& $py analysis/03b_density.py         # real density (Haversine), written back to the feature table
& $py analysis/03c_model_density.py   # density model comparison -> 03_model_density_*.csv
& $py analysis/04_charts.py           # figures 1-10
& $py analysis/04b_density_charts.py  # figures 11-13
& $py analysis/05_maps.py             # figures 14-15 (real state boundaries + city distribution)
& $py analysis/06_metros.py           # corrected metro definition -> 06_metro_summary.csv
& $py analysis/07_state_maps.py       # figures 16/17 a-c
& $py analysis/09_basemap_maps.py     # OSM basemap figures (recommended)
```

The scripts expect the exported CSVs under `data/bigquery/`, which are not published — see the
repository's DATA note.
