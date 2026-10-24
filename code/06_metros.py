"""Step 6 - consolidate cities into metropolitan areas, fixing the limitation that
Yelp's `city` field splits one metro across several labels.

Two problems are corrected:
  1. spelling/abbreviation variants   (St. Louis vs Saint Louis, St. Petersburg vs Saint Petersburg)
  2. separate political cities inside one urban area
     (Philadelphia / Cherry Hill / Wilmington / Camden, Tampa / Clearwater / St. Petersburg)

Method: normalise the name, then greedily cluster restaurants that are within
METRO_KM of an already-seeded urban core, keeping clusters inside one state so
cross-border metros are not fused.

Outputs: analysis/out/06_metro_summary.csv, 06_city_to_metro.csv
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
OUT = os.path.join(WS, "analysis", "out")
CORE_KM = 8.0            # city labels this close on a density peak share one core
ASSIGN_KM = 30.0         # a label further than this from every core stays its own
R = 6371.0088

US_STATES = {"AZ", "CA", "CO", "DE", "FL", "ID", "IL", "IN", "LA", "MO",
             "MT", "NC", "NJ", "NV", "PA", "TN"}
# excluded: AK/HI have no coordinates in this dataset, AB is Canada, XMS is a
# placeholder
#
# SCOPE NOTE: this merges labels that describe the SAME place -- spelling
# variants ('St. Louis'/'Saint Louis') and neighbourhood-level labels inside one
# city ('Phila'/'Philadelphia'). It deliberately does NOT fuse distinct
# neighbouring cities into a Census-style MSA: Tampa and Clearwater stay apart
# because they are genuinely different cities ~30 km apart, and competition
# operates at urban-area scale, not MSA scale. Treat the output as an
# 'urban-cluster label', not an official metropolitan statistical area.

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)

# ------------------------------------------------------------------ normalise
def norm_city(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[.,]", " ", s)
    s = s.replace("saint ", "st ")
    s = s.replace("ft ", "fort ")
    s = re.sub(r"\s+", " ", s).strip()
    # collapse spacing entirely: the dataset has 'land o lakes'/'land olakes',
    # 'o fallon'/'ofallon', 'st petersburg'/'stpetersburg' etc.
    return s.replace(" ", "")


feat["city_norm"] = feat.city.map(norm_city)

# ------------------------------------------------------------------ clustering
def haversine(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def build_metros(df: pd.DataFrame):
    """Detect urban cores as local density peaks, then attach each city to its
    nearest core.

    Connected-components clustering was tried first and rejected: it is
    transitive, so a chain of nearby suburbs fuses an entire state into one
    'metro' (Cherry Hill, NJ absorbed 188 city labels that way). Peak detection
    cannot chain, because only genuine density maxima become cores.
    """
    g = df.groupby(["state", "city_norm"]).agg(
        n=("business_id", "size"), lat=("latitude", "mean"), lon=("longitude", "mean")
    ).reset_index()

    cores = []           # (state, lat, lon, weight)
    for st, grp in g.sort_values("n", ascending=False).groupby("state"):
        for _, r in grp.iterrows():
            hit = -1
            for ci, c in enumerate(cores):
                if c[0] != st:
                    continue
                if haversine(r.lat, r.lon, c[1], c[2]) <= CORE_KM:
                    hit = ci
                    break
            if hit == -1:
                cores.append((st, r.lat, r.lon, r.n))
            else:
                st0, la0, lo0, w0 = cores[hit]
                tot = w0 + r.n
                cores[hit] = (st0, (la0 * w0 + r.lat * r.n) / tot,
                              (lo0 * w0 + r.lon * r.n) / tot, tot)

    # assign every city label to the nearest core in its state
    assign = {}
    for _, r in g.iterrows():
        best, best_d = None, ASSIGN_KM
        for ci, c in enumerate(cores):
            if c[0] != r.state:
                continue
            d = haversine(r.lat, r.lon, c[1], c[2])
            if d < best_d:
                best, best_d = ci, d
        assign[(r.state, r.city_norm)] = f"core-{best}" if best is not None \
            else f"solo-{r.state}-{r.city_norm}"

    print(f"  detected {len(cores)} urban cores; "
          f"{sum(1 for v in assign.values() if v.startswith('solo'))} labels kept solo")
    # order cores by size for readable ids
    size = {}
    for (st, cn), key in assign.items():
        size[key] = size.get(key, 0) + int(g[(g.state == st) & (g.city_norm == cn)].n.iloc[0])
    rank = {k: i for i, (k, _) in enumerate(sorted(size.items(), key=lambda kv: -kv[1]))}
    assign = {k: f"metro-{rank[v]:03d}" for k, v in assign.items()}
    return df.set_index(["state", "city_norm"]).index.map(assign)


print(f"restaurants: {len(feat):,}   distinct raw city labels: {feat.city.nunique():,}")
us = feat[feat.state.isin(US_STATES)].copy().reset_index(drop=True)
print(f"US rows used for clustering: {len(us):,} (states: {sorted(us.state.unique())})")

us["metro_key"] = build_metros(us)
us["metro_id"] = us.groupby("metro_key").ngroup()
print(f"metros after clustering: {us.metro_id.nunique():,}")

# display name = largest member city; attach member cities for transparency
mem = us.groupby(["metro_id", "city"]).size().rename("n").reset_index()
primary = mem.sort_values("n", ascending=False).drop_duplicates("metro_id").set_index("metro_id")["city"]
members = mem.groupby("metro_id")["city"].apply(
    lambda s: " / ".join(sorted(set(s))[:6])).rename("member_cities")

us["metro_name"] = us.metro_id.map(primary)
us["metro_label"] = us.metro_name + ", " + us.state

metro = us.groupby(["metro_id", "metro_label", "state"]).agg(
    restaurants=("business_id", "size"),
    n_city_labels=("city", "nunique"),
    closure_rate=("is_open", lambda s: 1 - s.mean()),
    avg_stars=("business_stars", "mean"),
    median_density=("n_within_1_0km", "median"),
    mean_density=("n_within_1_0km", "mean"),
    lat=("latitude", "mean"),
    lon=("longitude", "mean"),
).reset_index()
metro = metro.merge(members, on="metro_id")
metro = metro.sort_values("restaurants", ascending=False).reset_index(drop=True)
metro.to_csv(os.path.join(OUT, "06_metro_summary.csv"), index=False, encoding="utf-8-sig")
us[["business_id", "city", "state", "metro_id", "metro_name", "metro_label"]].to_csv(
    os.path.join(OUT, "06_city_to_metro.csv"), index=False, encoding="utf-8-sig")

print(f"\nwrote 06_metro_summary.csv ({len(metro):,} metros)")
print(metro.head(14)[["metro_label", "restaurants", "n_city_labels", "closure_rate",
                      "median_density"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ reporting
raw_city = feat[feat.state.isin(US_STATES)].city.nunique()
print(f"\n=== consolidation effect ===")
print(f"  raw city labels (US)      : {raw_city:,}")
print(f"  metros after merging      : {len(metro):,}")
print(f"  labels absorbed           : {raw_city - len(metro):,}")
tot = metro.restaurants.sum()
for k in (10, 25, 50, 100):
    share = metro.head(k).restaurants.sum() / tot
    print(f"  top {k:>3} metros share      : {share:.1%}")

variants = [("st louis", "saint louis"), ("st petersburg", "saint petersburg"),
            ("land o lakes", "land olakes"), ("o fallon", "ofallon")]
print("\n  spelling variants now merged (normalised-name lookup, not regex):")
c2m = dict(zip(zip(us.state, us.city.map(norm_city)), us.metro_id))
for a, b in variants:
    ka = [(st, norm_city(x)) for st, x in zip(us.state, us.city)]
    fa = {v for k, v in c2m.items() if k[1] == norm_city(a)}
    fb = {v for k, v in c2m.items() if k[1] == norm_city(b)}
    print(f"    {a!r} vs {b!r}: same metro -> {bool(fa & fb)}")

print("\n  known metro parentage check (suburb should share the core's metro_id):")
for core, suburb in [("Philadelphia", "Cherry Hill"), ("Tampa", "Clearwater"),
                     ("Saint Louis", "Clayton"), ("Reno", "Sparks"),
                     ("Boise", "Meridian")]:
    cid = set(us.loc[us.city == core, "metro_id"])
    sid = set(us.loc[us.city == suburb, "metro_id"])
    tag = "merged" if (cid & sid) else "SEPARATE"
    print(f"    {core:14} <- {suburb:14} {tag}")

print("\n  large multi-city metros (absorbed separate city labels):")
multi = metro[metro.n_city_labels >= 3].head(12)
for _, r in multi.iterrows():
    print(f"    {r.metro_label:28} {r.restaurants:>5} 家, {r.n_city_labels:>2} 个 city 标签  ({r.member_cities[:70]})")

# closure trend by merged metro size (the fixed version of the earlier claim)
m15 = metro[metro.restaurants >= 15].copy()
rho = m15[["restaurants", "closure_rate"]].corr(method="spearman").iloc[0, 1]
m15["_q"] = pd.qcut(m15.restaurants, 5, labels=False, duplicates="drop")
qg = m15.groupby("_q").agg(closure=("closure_rate", "mean"), avg=("restaurants", "mean"))
pd.DataFrame([{"metric": "metros_ge15", "value": len(m15)},
              {"metric": "spearman_size_vs_closure", "value": rho},
              {"metric": "closure_q1", "value": qg.closure.iloc[0]},
              {"metric": "closure_q5", "value": qg.closure.iloc[-1]},
              {"metric": "top10_share", "value": metro.head(10).restaurants.sum() / tot},
              {"metric": "top50_share", "value": metro.head(50).restaurants.sum() / tot}]
             ).to_csv(os.path.join(OUT, "06_metro_stats.csv"), index=False, encoding="utf-8-sig")
print(f"\n  merged-metro size vs closure: Spearman {rho:+.3f} "
      f"(Q1 {qg.closure.iloc[0]:.1%} -> Q5 {qg.closure.iloc[-1]:.1%})")
print("done.")
