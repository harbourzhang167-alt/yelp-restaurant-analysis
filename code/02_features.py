"""Step 2 - build the business-level feature table that both the analysis and
Tableau consume.

One row per restaurant (52,286), one column per usable signal. Deliberately
EXCLUDES last-review date and review span: those are mechanically tied to the
outcome (a closed restaurant's last review is roughly its closing date), so
including them would be predicting the outcome from itself.

Output: analysis/out/business_features.csv
"""

from __future__ import annotations

import io
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(WS, "data", "bigquery")
OUT = os.path.join(WS, "analysis", "out")
os.makedirs(OUT, exist_ok=True)


def P(n):
    return os.path.join(D, f"bdao_group4.{n}.csv")


print("loading restaurants ...")
cr = pd.read_csv(P("cleaned_restaurants"), low_memory=False)
yr = pd.read_csv(P("yelp_restaurants"), low_memory=False,
                 usecols=["business_id", "postal_code", "stars", "review_count", "hours"])
f = cr.merge(yr, on="business_id", how="left", suffixes=("", "_yelp"))

# ---------------------------------------------------------------- base ----
out = pd.DataFrame({
    "business_id": f.business_id,
    "name": f.name,
    "city": f.city,
    "state": f.state,
    "postal_code": f.postal_code,
    "latitude": pd.to_numeric(f.latitude, errors="coerce"),
    "longitude": pd.to_numeric(f.longitude, errors="coerce"),
    "is_open": f.is_open.astype(int),
    "business_stars": pd.to_numeric(f.business_stars, errors="coerce"),
    "business_review_count": pd.to_numeric(f.business_review_count, errors="coerce"),
})

# price tier: keep Unknown as its own level, plus a numeric version for modelling
out["price_tier"] = f.price_range.astype(str)
out["price_num"] = pd.to_numeric(f.price_range, errors="coerce")

# ------------------------------------------------------- attributes -------
# Unknown is preserved as a level because it carries signal (Unknown venues
# survive more often, i.e. missingness correlates with listing freshness).
ATTRS = ["alcohol", "noise_level", "wifi", "good_for_kids", "takeout", "delivery",
         "outdoor_seating", "has_tv", "reservations", "accepts_credit_cards",
         "bike_parking", "caters", "by_appointment_only"]
for a in ATTRS:
    out[f"attr_{a}"] = f[a].astype(str)

# convenience binary flags (True / False / Unknown)
for a in ATTRS:
    s = f[a].astype(str).str.lower()
    out[f"is_{a}"] = np.where(s == "true", 1, np.where(s == "false", 0, np.nan)).astype("float")

# --------------------------------------------------------- hours ----------
print("parsing hours ...")
hrs = f["hours"].fillna("")
out["has_hours"] = (hrs != "").astype(int)
# count how many weekdays are listed with a non-null value
out["hours_days_listed"] = hrs.str.count(r'"\w+":\s*"')  # crude but effective
out["hours_mentions_24"] = hrs.str.contains("0:0-0:0", regex=False).astype(int)


def hours_open_late(text: str) -> int:
    """Loose 'open past midnight' flag: any closing hour in 0-5 range."""
    if not text:
        return 0
    import re
    for m in re.finditer(r"(\d{1,2}):(\d{1,2})-(\d{1,2}):(\d{1,2})", text):
        close_h = int(m.group(3))
        if close_h <= 5:
            return 1
    return 0


out["hours_open_late"] = hrs.map(hours_open_late)

# ------------------------------------------------------ categories --------
print("parsing categories ...")
cats = f.categories.fillna("").astype(str)
MAJOR = ["Pizza", "Mexican", "Chinese", "Japanese", "Italian", "American (Traditional)",
         "American (New)", "Burgers", "Sandwiches", "Seafood", "Fast Food",
         "Breakfast & Brunch", "Coffee & Tea", "Bars", "Nightlife", "Desserts",
         "Thai", "Indian", "Korean", "Vietnamese", "Mediterranean", "Greek",
         "French", "Vegetarian", "Vegan", "Sushi Bars", "Chicken Wings",
         "Barbeque", "Salad", "Soup", "Cafes", "Bakeries", "Steakhouses"]
for c in MAJOR:
    out[f"cat_{c.replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')}"] = \
        cats.str.contains(re.escape(c), regex=True).astype(int)
out["n_categories"] = cats.str.count(",") + 1
out["category_first"] = cats.str.split(",").str[0].str.strip()

# ------------------------------------------------------- reviews ---------
print("aggregating reviews (4.7M rows) ...")
rv = pd.read_csv(P("cleaned_reviews"),
                 usecols=["business_id", "review_stars", "useful", "funny", "cool", "review_date"],
                 parse_dates=["review_date"])
agg = rv.groupby("business_id").agg(
    n_reviews=("review_stars", "size"),
    mean_stars=("review_stars", "mean"),
    std_stars=("review_stars", "std"),
    min_stars=("review_stars", "min"),
    max_stars=("review_stars", "max"),
    pct_1star=("review_stars", lambda s: (s == 1).mean()),
    pct_2star=("review_stars", lambda s: (s == 2).mean()),
    pct_3star=("review_stars", lambda s: (s == 3).mean()),
    pct_4star=("review_stars", lambda s: (s == 4).mean()),
    pct_5star=("review_stars", lambda s: (s == 5).mean()),
    pct_neg=("review_stars", lambda s: (s <= 2).mean()),
    pct_pos=("review_stars", lambda s: (s >= 4).mean()),
    mean_useful=("useful", "mean"),
    median_useful=("useful", "median"),
    pct_useful0=("useful", lambda s: (s == 0).mean()),
    sum_cool=("cool", "sum"),
    sum_funny=("funny", "sum"),
    first_review=("review_date", "min"),
)
out = out.merge(agg, on="business_id", how="left")

# early-window features: first 365 days from each business's first review
print("building first-year window features ...")
first = rv.groupby("business_id")["review_date"].min().rename("first_review")
rv2 = rv.merge(first, on="business_id", how="left")
rv2["in_year1"] = rv2.review_date <= (rv2.first_review + pd.Timedelta(days=365))
y1 = rv2[rv2.in_year1].groupby("business_id").agg(
    y1_n_reviews=("review_stars", "size"),
    y1_mean_stars=("review_stars", "mean"),
    y1_std_stars=("review_stars", "std"),
    y1_pct_neg=("review_stars", lambda s: (s <= 2).mean()),
)
out = out.merge(y1, on="business_id", how="left")

# ---------------------------------------------------------- tips ---------
print("aggregating tips ...")
tp = pd.read_csv(P("yelp_tips"), usecols=["business_id", "date", "compliment_count", "text"],
                 parse_dates=["date"])
tags = tp.groupby("business_id").agg(
    n_tips=("compliment_count", "size"),
    mean_tip_compliments=("compliment_count", "mean"),
    mean_tip_len=("text", lambda s: s.str.len().mean()),
    first_tip=("date", "min"),
)
out = out.merge(tags, on="business_id", how="left")
out["n_tips"] = out.n_tips.fillna(0)
out["has_tips"] = (out.n_tips > 0).astype(int)

# ------------------------------------------------------- checkins --------
print("aggregating check-ins ...")
ck = pd.read_csv(P("business_checkin_summary"), low_memory=False,
                 usecols=["business_id", "total_checkins", "category_major"])
ck["total_checkins"] = pd.to_numeric(ck.total_checkins, errors="coerce")
out = out.merge(ck[["business_id", "total_checkins", "category_major"]],
                on="business_id", how="left")
out["total_checkins"] = out.total_checkins.fillna(0)
out["has_checkins"] = (out.total_checkins > 0).astype(int)

ch = pd.read_csv(P("checkin_restaurant_hourly"),
                 usecols=["business_id", "hour", "checkin_count", "weekday"])
ch["late"] = ch.hour.isin([22, 23, 0, 1, 2, 3])
ch["late_cnt"] = np.where(ch.late, ch.checkin_count, 0)
cagg = ch.groupby("business_id").agg(ck_total=("checkin_count", "sum"), ck_late=("late_cnt", "sum"))
cagg["checkin_late_share"] = cagg.ck_late / cagg.ck_total.replace(0, np.nan)
out = out.merge(cagg[["checkin_late_share"]], on="business_id", how="left")
# peak-hour concentration: share of check-ins in the single busiest hour
peak = ch.groupby("business_id").apply(
    lambda g: g.checkin_count.max() / g.checkin_count.sum() if g.checkin_count.sum() else np.nan,
    include_groups=False).rename("checkin_peak_share")
out = out.merge(peak, on="business_id", how="left")

# ------------------------------------------------------- derived ---------
out["reviews_per_category"] = out.n_reviews / out.n_categories.replace(0, np.nan)
out["log_reviews"] = np.log1p(out.n_reviews)
out["useful_per_review"] = out.mean_useful
out["tip_per_review"] = out.n_tips / out.n_reviews.replace(0, np.nan)
out["checkin_per_review"] = out.total_checkins / out.n_reviews.replace(0, np.nan)
out["engagement_index"] = (
    (out.n_reviews.rank(pct=True) + out.n_tips.rank(pct=True) + out.total_checkins.rank(pct=True)) / 3
)

# market density: how many restaurants share this city and this postal code
city_n = out.groupby(["city", "state"]).business_id.transform("size")
out["city_restaurant_count"] = city_n
pc_n = out.groupby(["postal_code"]).business_id.transform("size")
out["zip_restaurant_count"] = pc_n
# same-major-category crowding within the city
out["city_cat_count"] = out.groupby(["city", "state", "category_first"]).business_id.transform("size")

# ------------------------------------------------------- save ------------
front = ["business_id", "name", "city", "state", "postal_code", "latitude", "longitude",
         "is_open", "business_stars", "business_review_count", "price_tier", "price_num",
         "category_first", "category_major", "n_categories",
         "n_reviews", "mean_stars", "std_stars", "pct_neg", "pct_pos", "pct_1star",
         "pct_5star", "mean_useful", "pct_useful0",
         "y1_n_reviews", "y1_mean_stars", "y1_std_stars", "y1_pct_neg",
         "n_tips", "mean_tip_compliments", "mean_tip_len",
         "total_checkins", "checkin_late_share", "checkin_peak_share",
         "has_hours", "hours_days_listed", "hours_open_late",
         "city_restaurant_count", "zip_restaurant_count", "city_cat_count",
         "engagement_index", "log_reviews", "tip_per_review", "checkin_per_review"]
front = [c for c in front if c in out.columns]
rest = [c for c in out.columns if c not in front]
out = out[front + rest]

path = os.path.join(OUT, "business_features.csv")
out.to_csv(path, index=False, encoding="utf-8-sig")
print(f"\nwrote {path}")
print(f"  shape: {out.shape[0]:,} rows x {out.shape[1]} cols")
print(f"  is_open: {out.is_open.value_counts().to_dict()}")
print(f"  missing by key column:")
for c in ["n_reviews", "n_tips", "total_checkins", "checkin_late_share", "y1_n_reviews",
          "std_stars", "hours_days_listed", "postal_code"]:
    print(f"    {c:24} null {out[c].isna().mean()*100:5.1f}%")
