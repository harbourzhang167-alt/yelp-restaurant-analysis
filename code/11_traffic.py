"""Step 11 - flip the dependent variable: analyse CUSTOMER TRAFFIC instead of survival.

Design note (this is the whole point of the exercise):

    DV  = check-in intensity (check-ins per active month)  -- a footfall proxy
    IVs = STRUCTURAL features only: attributes, price, opening hours, local
          competition density, city size, rating composition, category, geography.

Volume-derived variables (n_reviews, n_tips, total_checkins, mean_useful,
engagement_index, tip_per_review, checkin_per_review) are deliberately EXCLUDED
from the right-hand side. Using them would predict traffic from traffic.

Two traps handled explicitly:
  * Check-ins peak platform-wide in 2015 and fall ~70% by 2021 — that is Yelp's
    check-in feature losing usage, NOT stores losing customers. So no "recent
    traffic trend" is used as a DV; instead the store's active-period midpoint
    year enters as a temporal control.
  * That midpoint year is mechanically tied to survival (a store whose check-ins
    end in 2011 is almost certainly closed) — exactly the same trap as
    last_review_date. It is therefore only ever a CONTROL, never a predictor.

Outputs (analysis/out/):
    07_traffic_features.csv      one row per business: traffic measures + IVs
    07_traffic_coefs.csv         OLS coefficients with bootstrap CIs
    07_traffic_poisson.csv       Poisson offset model (robustness)
    07_traffic_density.csv       the agglomeration vs competition tension
    07_traffic_category.csv      intensity by cuisine category
"""

from __future__ import annotations

import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(WS, "data", "bigquery")
OUT = os.path.join(WS, "analysis", "out")
RNG = np.random.default_rng(20261024)
NBOOT = 600

# ---------------------------------------------------------------- traffic ----
print("building traffic measures from 8.5M check-in events ...")
ev = pd.read_csv(os.path.join(D, "bdao_group4.cleaned_checkin_expanded.csv"))
ev["ts"] = pd.to_numeric(ev.checkin_timestamp, errors="coerce")
ev["dt"] = pd.to_datetime(ev.ts, unit="s")

g = ev.groupby("business_id").agg(
    ck_events=("ts", "size"),
    ck_first=("dt", "min"),
    ck_last=("dt", "max"),
)
g["ck_months"] = ((g.ck_last - g.ck_first).dt.days / 30.4375).clip(lower=1.0)
g["ck_per_month"] = g.ck_events / g.ck_months
g["mid_year"] = (g.ck_first + (g.ck_last - g.ck_first) / 2).dt.year.astype(float)
print(f"  businesses with check-ins: {len(g):,}")
print(f"  span months  median {g.ck_months.median():.1f}, p10 {g.ck_months.quantile(.1):.1f}")

# ---------------------------------------------------------------- design ----
feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
df = feat.merge(g, left_on="business_id", right_index=True, how="inner")

# 100 check-ins a month on average is implausible for a check-in feature; cap
# extreme outliers rather than letting a handful of venues drive the fit
cap = df.ck_per_month.quantile(0.995)
df["ck_per_month_capped"] = df.ck_per_month.clip(upper=cap)
df["log_rate"] = np.log(df.ck_per_month_capped)
print(f"\nmerged rows: {len(df):,}   rate cap p99.5 = {cap:.2f}/month")
print(df.ck_per_month.describe(percentiles=[.5, .9, .99]).round(3).to_string())

ATTRS = ["is_delivery", "is_takeout", "is_has_tv", "is_reservations",
         "is_outdoor_seating", "is_caters", "is_good_for_kids",
         "is_accepts_credit_cards", "is_bike_parking"]
# is_alcohol / is_wifi are NOT boolean in the data: their levels are
# full_bar / beer_and_wine and free / no / paid, so the is_* flag is 100% NaN and
# mean-filling it silently leaves NaN (which killed the SVD). Encode the real
# levels as dummies instead.
for level, name in [("full_bar", "alcohol_full_bar"),
                    ("beer_and_wine", "alcohol_beer_wine"),
                    ("free", "wifi_free"),
                    ("paid", "wifi_paid"),
                    ("no", "wifi_none")]:
    src = "attr_alcohol" if name.startswith("alcohol") else "attr_wifi"
    df[name] = (df[src].astype(str) == level).astype(float) if src in df.columns else 0.0
ATTRS += ["alcohol_full_bar", "alcohol_beer_wine", "wifi_free", "wifi_paid", "wifi_none"]
# by_appointment_only excluded: 94% Unknown, no usable contrast
# pct_neg is EXCLUDED from the main model: it correlates with business_stars at
# -0.937 (near-perfect collinearity), and including both flipped the star
# coefficient to -5.7% purely as a suppression artefact. Stars is kept because it
# is interpretable and comparable with the survival analysis; the pct_neg variant
# is reported separately as a sensitivity check.
NUM = ["price_num", "has_hours", "hours_open_late", "hours_days_listed",
       "n_within_1_0km", "city_restaurant_count",
       "business_stars", "std_stars", "mid_year"]

CATS = ["cat_Pizza", "cat_Mexican", "cat_Chinese", "cat_Japanese", "cat_Italian",
        "cat_American_Traditional", "cat_American_New", "cat_Burgers",
        "cat_Sandwiches", "cat_Seafood", "cat_Fast_Food", "cat_Breakfast_and_Brunch",
        "cat_Coffee_and_Tea", "cat_Bars", "cat_Nightlife", "cat_Desserts",
        "cat_Thai", "cat_Indian", "cat_Korean", "cat_Vietnamese"]

base = df.dropna(subset=["ck_per_month_capped"]).copy()
cols = [c for c in ATTRS + NUM + CATS if c in base.columns]
X = base[cols].copy()
# Unknown attributes are NaN -> mean-fill (they are a level, not zero)
for c in cols:
    if X[c].isna().any():
        if X[c].notna().any():
            X[c] = X[c].fillna(X[c].mean())
        else:                       # entirely missing -> drop rather than poison the fit
            cols.remove(c)
            X = X.drop(columns=[c])

Xs = (X - X.mean()) / X.std().replace(0, 1.0)
# hard guard: a single non-finite cell makes lstsq fail with an opaque SVD error
if not np.isfinite(Xs.values).all():
    bad = [c for c in cols if not np.isfinite(Xs[c].values).all()]
    raise SystemExit(f"non-finite design matrix in: {bad}")

y = base["log_rate"].values


def ols_fit(Xm, yv):
    A = np.hstack([np.ones((len(yv), 1)), Xm])
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    return beta, resid


beta, resid = ols_fit(Xs.values, y)
pred = np.hstack([np.ones((len(y), 1)), Xs.values]) @ beta
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"\nOLS on log(traffic/month): R^2 = {r2:.4f}, n = {len(y):,}, features = {len(cols)}")

# bootstrap by resampling businesses
print(f"bootstrapping {NBOOT} replicates ...")
boot = np.empty((NBOOT, len(cols) + 1))
n = len(y)
Xv = Xs.values
for b in range(NBOOT):
    idx = RNG.integers(0, n, n)
    bb, _ = ols_fit(Xv[idx], y[idx])
    boot[b] = bb

lo = np.percentile(boot, 2.5, axis=0)
hi = np.percentile(boot, 97.5, axis=0)

coefs = pd.DataFrame({
    "feature": cols,
    "coef_std": beta[1:],
    "pct_effect_per_SD": (np.exp(beta[1:]) - 1) * 100,
    "ci_lo_pct": (np.exp(lo[1:]) - 1) * 100,
    "ci_hi_pct": (np.exp(hi[1:]) - 1) * 100,
})
coefs["significant"] = ~((lo[1:] < 0) & (hi[1:] > 0))
coefs = coefs.reindex(coefs.coef_std.abs().sort_values(ascending=False).index)
coefs.to_csv(os.path.join(OUT, "07_traffic_coefs.csv"), index=False, encoding="utf-8-sig")
print("\n=== strongest traffic associations (percent effect per +1 SD) ===")
for _, r in coefs.head(14).iterrows():
    flag = "*" if r.significant else " "
    print(f" {flag} {r.feature:26} {r.pct_effect_per_SD:+7.1f}%  "
          f"[{r.ci_lo_pct:+7.1f}, {r.ci_hi_pct:+7.1f}]")

# ------------------------------------------------- Poisson (offset) check ----
print("\nPoisson with log(active months) offset (robustness) ...")


def poisson_irls(Xm, yv, offset, ridge=1.0, iters=40):
    A = np.hstack([np.ones((len(yv), 1)), Xm])
    beta = np.zeros(A.shape[1])
    pen = np.eye(A.shape[1]) * ridge
    pen[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(A @ beta + offset, -30, 30)
        mu = np.exp(eta)
        w = np.clip(mu, 1e-6, None)
        z = eta - offset + (yv - mu) / w
        beta = np.linalg.solve(A.T @ (A * w[:, None]) + pen, A.T @ (w * z))
    return beta


ycount = base.ck_events.values.astype(float)
offset = np.log(base.ck_months.values)
pb = poisson_irls(Xs.values, ycount, offset)
pois = pd.DataFrame({
    "feature": cols,
    "irr": np.exp(pb[1:]),
    "note": "IRR per +1 SD; DV = check-in count, offset = log(active months)",
})
pois = pois.reindex(pois.irr.sub(1).abs().sort_values(ascending=False).index)
pois.to_csv(os.path.join(OUT, "07_traffic_poisson.csv"), index=False, encoding="utf-8-sig")
print("  top: " + ", ".join(f"{r.feature}={r.irr:.2f}"
                            for _, r in pois.head(6).iterrows()))

# ------------------------------------------- collinearity + pct_neg variant ----
print("\ncollinearity check among numeric IVs:")
num_ok = [c for c in NUM if c in base.columns]
cm = base[num_ok].corr(method="spearman")
worst = (cm.where(~np.eye(len(cm), dtype=bool)).abs().stack()
         .sort_values(ascending=False).head(4))
for (a, b), v in worst.items():
    if a < b:
        print(f"  {a} ~ {b}: {v:+.3f}")

print("\nsensitivity: swap business_stars for pct_neg (the two are near-duplicates)")
alt_cols = [("pct_neg" if c == "business_stars" else c) for c in cols]
Xa = base[[c for c in alt_cols if c in base.columns]].copy()
for c in Xa.columns:
    if Xa[c].isna().any():
        Xa[c] = Xa[c].fillna(Xa[c].mean())
Xas = (Xa - Xa.mean()) / Xa.std().replace(0, 1.0)
ba, _ = ols_fit(Xas.values, y)
alt = pd.DataFrame({"feature": list(Xa.columns),
                    "pct_effect_per_SD": (np.exp(ba[1:]) - 1) * 100})
alt.to_csv(os.path.join(OUT, "07_traffic_pctneg_variant.csv"), index=False,
           encoding="utf-8-sig")
for f in ("pct_neg", "business_stars"):
    if f in set(alt.feature):
        v = alt.loc[alt.feature == f, "pct_effect_per_SD"].iloc[0]
        print(f"  {f:16} {v:+7.1f}%")

# ------------------------------------------------ density tension table ----
print("\ndensity: traffic vs survival tension ...")
t = base.dropna(subset=["n_within_1_0km"]).copy()
t["dens_dec"] = pd.qcut(t.n_within_1_0km, 10, labels=False, duplicates="drop")
rows = []
for dec, grp in t.groupby("dens_dec"):
    rows.append({
        "decile": int(dec) + 1,
        "n": len(grp),
        "avg_neighbours_1km": grp.n_within_1_0km.mean(),
        "median_traffic_per_month": grp.ck_per_month.median(),
        "mean_log_rate": grp.log_rate.mean(),
        "closure_rate": 1 - grp.is_open.mean(),
        "avg_stars": grp.business_stars.mean(),
    })
dens = pd.DataFrame(rows)
dens.to_csv(os.path.join(OUT, "07_traffic_density.csv"), index=False, encoding="utf-8-sig")
print(dens.round(3).to_string(index=False))

# ------------------------------------------------------ category table -----
cat_rows = []
for c in CATS:
    if c not in base.columns:
        continue
    sub = base[base[c] == 1]
    rest = base[base[c] == 0]
    if len(sub) < 300:
        continue
    cat_rows.append({
        "category": c.replace("cat_", ""),
        "n": len(sub),
        "median_traffic_per_month": sub.ck_per_month.median(),
        "mean_log_rate": sub.log_rate.mean(),
        "closure_rate": 1 - sub.is_open.mean(),
        "avg_stars": sub.business_stars.mean(),
        "vs_others_pct": (np.exp(sub.log_rate.mean() - rest.log_rate.mean()) - 1) * 100,
    })
cats = pd.DataFrame(cat_rows).sort_values("vs_others_pct", ascending=False)
cats.to_csv(os.path.join(OUT, "07_traffic_category.csv"), index=False, encoding="utf-8-sig")
print("\n=== traffic by category (vs all others) ===")
print(cats.round(3).to_string(index=False))

# ------------------------------------------------------------- save ---------
keep = ["business_id", "city", "state", "is_open", "ck_events", "ck_months",
        "ck_per_month", "ck_per_month_capped", "log_rate", "mid_year",
        "n_within_1_0km", "city_restaurant_count", "price_num", "business_stars",
        "pct_neg", "std_stars"] + ATTRS
keep = [c for c in keep if c in base.columns]
base[keep].to_csv(os.path.join(OUT, "07_traffic_features.csv"),
                  index=False, encoding="utf-8-sig")
print(f"\nwrote 07_traffic_features.csv ({len(base):,} rows)")
