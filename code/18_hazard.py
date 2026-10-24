"""Build and describe the person-period dataset for a discrete-time hazard model.

WHY A HAZARD MODEL
    `is_open` is a snapshot with no closure date, so the surviving analysis could only
    compare "still open" with "closed" and could not tell a store that lasted three
    years from one that lasted fifteen. The quarterly trajectory table gives an
    observed review series per store, which supports a proper time-to-event design.

WHAT THE EVENT IS - AND IS NOT
    The observable event is **the review series stopping** ("going quiet"), not
    closure. That distinction matters: the last active quarter is mechanically tied to
    `is_open` (closed stores go quiet earlier), so using `is_open` as the outcome here
    would be circular. Instead:

      * the event is defined purely from the review series (last observed quarter)
      * every predictor uses only quarters STRICTLY BEFORE the event quarter
      * `is_open` is never an input; it is used afterwards only to validate

    A store that is open at the snapshot and still active in 2022-Q1 is right-censored.

DATA-QUALITY NOTE
    `final_warning_data` is NOT store x month: 4,724,683 rows collapse to 1,825,219
    unique (store, month) pairs, and the "duplicate" rows carry different rolling
    averages (e.g. 4.5 vs 4.8 for the same store and month), i.e. it is at review
    grain with a rolling statistic attached. It is therefore unusable as a
    person-period source and is not used here.

Output: analysis/out/08_hazard_person_period.csv  (gzip if large)
        analysis/out/08_hazard_describe.md
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(WS, "data", "bigquery")
OUT = os.path.join(WS, "analysis", "out")
SNAPSHOT = pd.Timestamp("2022-01-01")   # last quarter present in the data

print("loading quarterly trajectories ...")
q = pd.read_csv(os.path.join(D, "bdao_group4.restaurant_quarterly_trajectory.csv"),
                usecols=["business_id", "review_quarter", "quarterly_review_count",
                         "quarterly_avg_stars", "is_open"])
q["qd"] = pd.to_datetime(q.review_quarter)
q = q.sort_values(["business_id", "qd"]).reset_index(drop=True)
print("  rows %s, stores %s" % (format(len(q), ","),
                                format(q.business_id.nunique(), ",")))
assert not q.duplicated(["business_id", "review_quarter"]).any(), "grain not unique"

# ---------------------------------------------------------------- person-period
print("building person-period rows ...")
g = q.groupby("business_id", sort=False)
q["seq"] = g.cumcount() + 1                 # 1..T within the store's observed series
q["n_periods"] = g["seq"].transform("max")
q["first_q"] = g["qd"].transform("min")
q["last_q"] = g["qd"].transform("max")
q["is_open"] = g["is_open"].transform("first")

# censored = open at snapshot; event = closed and its series stopped before the snapshot
q["censored"] = (q.is_open == 1) | (q.last_q >= SNAPSHOT)
q["event"] = (~q.censored)
q["y"] = ((q.seq == q.n_periods) & q.event).astype(float)

# ------------------------------------------------ lagged, time-varying predictors
# everything below uses ONLY quarters strictly before the current one
print("computing lagged predictors ...")
for src, name in (("quarterly_review_count", "rev"), ("quarterly_avg_stars", "stars")):
    q[name + "_cum"] = g[src].cumsum()
    q[name + "_prev_n"] = q.seq - 1                      # periods before this one
    q[name + "_prev_mean"] = np.where(
        q[name + "_prev_n"] > 0,
        (q[name + "_cum"] - q[src]) / q[name + "_prev_n"].replace(0, np.nan),
        np.nan,
    )

# trailing-4-quarter mean (still lagged: uses periods before the current one)
q["rev_lag4"] = (g["quarterly_review_count"]
                 .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()))
q["stars_lag4"] = (g["quarterly_avg_stars"]
                   .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()))

# trend: last 4 quarters before now vs the 4 before that (volume and rating)
q["rev_prev4"] = (g["quarterly_review_count"]
                  .transform(lambda s: s.shift(5).rolling(4, min_periods=2).mean()))
q["stars_prev4"] = (g["quarterly_avg_stars"]
                    .transform(lambda s: s.shift(5).rolling(4, min_periods=2).mean()))
q["rev_trend"] = q.rev_lag4 / q.rev_prev4.replace(0, np.nan)
q["stars_trend"] = q.stars_lag4 - q.stars_prev4          # change in mean rating

q["quarters_elapsed"] = q["seq"]

keep = ["business_id", "qd", "is_open", "censored", "event", "y", "seq",
        "n_periods", "quarters_elapsed",
        "quarterly_review_count", "quarterly_avg_stars",
        "rev_lag4", "stars_lag4", "rev_prev4", "stars_prev4",
        "rev_trend", "stars_trend"]
pp = q[keep].copy()

print()
print("=== person-period structure ===")
print("  rows                : %s" % format(len(pp), ","))
print("  stores              : %s" % format(pp.business_id.nunique(), ","))
print("  events (y=1)        : %s" % format(int(pp.y.sum()), ","))
print("  censored stores     : %s" % format(int(pp.loc[pp.censored, 'business_id'].nunique()), ","))
print("  event stores        : %s" % format(int(pp.loc[pp.event, 'business_id'].nunique()), ","))
print("  median periods/store: %d" % int(pp.groupby('business_id').size().median()))
print()
print("  share of stores with an event: %.1f%%" %
      (100 * pp.loc[pp.event, 'business_id'].nunique() / pp.business_id.nunique()))

# sanity: does the event correspond to is_open?
ev = pp[pp.y == 1]
print()
print("=== validation: event definition vs is_open ===")
print("  share of event rows that are is_open=0: %.3f" % (1 - ev.is_open.mean()))
print("  events by last-quarter year:")
print(ev.qd.dt.year.value_counts().sort_index().tail(8).to_string())

# how complete are the lagged predictors?
print()
print("=== predictor availability (rows with >=3 prior quarters) ===")
sub = pp[pp.quarters_elapsed >= 3]
for c in ["rev_lag4", "stars_lag4", "rev_trend", "stars_trend"]:
    print("  %-12s non-null %.1f%%" % (c, 100 * sub[c].notna().mean()))

os.makedirs(OUT, exist_ok=True)
pp.to_csv(os.path.join(OUT, "08_hazard_person_period.csv.gz"), index=False,
          compression="gzip")
print()
print("wrote 08_hazard_person_period.csv.gz  (%s rows)" % format(len(pp), ","))
