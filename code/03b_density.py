"""Step 3b - true spatial business density from coordinates.

The dataset has no population or area field, but each restaurant carries its own
GPS coordinate (verified: 49,476 unique coordinate pairs for 52,286 restaurants,
so these are real per-business locations, not city centroids). That makes a
genuine spatial density computable: how many restaurants sit within a radius of
each restaurant.

This replaces the earlier crude proxy (city restaurant count) and re-tests the
crowding finding, including whether it survives WITHIN a single city.
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
OUT = os.path.join(WS, "analysis", "out")

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
R_EARTH = 6371.0088  # km


def neighbour_counts(lat_deg, lon_deg, radius_km):
    """Count other points within radius_km, pruning by a coarse lat/lon grid."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    step = max(radius_km / 111.0, 1e-6)
    cell_lat = np.floor(lat_deg / step).astype(np.int64)
    cell_lon = np.floor(lon_deg / step).astype(np.int64)

    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(len(lat_deg)):
        buckets.setdefault((int(cell_lat[i]), int(cell_lon[i])), []).append(i)

    out = np.zeros(len(lat_deg), dtype=np.int32)
    for (cy, cx), idxs in buckets.items():
        cand: list[int] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand.extend(buckets.get((cy + dy, cx + dx), []))
        if not cand:
            continue
        a_idx = np.array(idxs)
        b_idx = np.array(cand)
        la1, lo1 = lat[a_idx][:, None], lon[a_idx][:, None]
        la2, lo2 = lat[b_idx][None, :], lon[b_idx][None, :]
        dla, dlo = la2 - la1, lo2 - lo1
        h = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
        d = 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        out[a_idx] = (d <= radius_km).sum(axis=1) - 1
    return out


print("computing true spatial density (haversine neighbour counts) ...")
feat = feat.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
for r in (0.5, 1.0, 2.0):
    col = f"n_within_{str(r).replace('.', '_')}km"
    feat[col] = neighbour_counts(feat.latitude.values, feat.longitude.values, r)
    print(f"  {col:22} mean {feat[col].mean():7.1f}  median {feat[col].median():6.0f}  max {feat[col].max():5d}")

# density normalised by city size: is a restaurant in a dense spot WITHIN its city?
city_med = feat.groupby("city")["n_within_1_0km"].transform("median")
feat["density_vs_city"] = feat["n_within_1_0km"] / city_med.replace(0, np.nan)

# same-category neighbours within 1km (true local category competition)
print("computing same-category 1km density (by first category) ...")
feat["cat_first"] = feat.category_first.fillna("Other").astype(str)
same = np.zeros(len(feat), dtype=np.int32)
for cat, grp in feat.groupby("cat_first"):
    if len(grp) < 2:
        continue
    idx = grp.index.to_numpy()
    cnt = neighbour_counts(grp.latitude.values, grp.longitude.values, 1.0)
    same[idx] = cnt
feat["n_same_cat_1km"] = same
print(f"  n_same_cat_1km         mean {feat.n_same_cat_1km.mean():7.1f}  median {feat.n_same_cat_1km.median():6.0f}")

feat.to_csv(os.path.join(OUT, "business_features.csv"), index=False, encoding="utf-8-sig")
print("\nupdated business_features.csv")

# ---------------------------------------------------------------- report ---
closed = 1 - feat.is_open.values
print("\n=== correlation with survival (Spearman) ===")
rows = []
for c in ["n_within_0_5km", "n_within_1_0km", "n_within_2_0km", "n_same_cat_1km",
          "density_vs_city", "city_restaurant_count", "city_cat_count"]:
    rho = feat[[c, "is_open"]].dropna().corr(method="spearman").iloc[0, 1]
    rows.append({"metric": c, "spearman_vs_survival": rho})
    print(f"  {c:24} {rho:+.4f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "03_findingC_density_corr.csv"),
                          index=False, encoding="utf-8-sig")

print("\n=== closure rate by true-density decile (1km) ===")
feat["_d"] = pd.qcut(feat["n_within_1_0km"], 10, labels=False, duplicates="drop")
g = feat.groupby("_d").agg(n=("is_open", "size"), avg_neighbours=("n_within_1_0km", "mean"),
                           closure=("is_open", lambda s: 1 - s.mean()))
print(g.round(4).to_string())
g.reset_index().to_csv(os.path.join(OUT, "03_findingC_density_deciles.csv"),
                       index=False, encoding="utf-8-sig")

print("\n=== within-city robustness (quartiles of 1km density) ===")
city_rows = []
for city, s in feat.groupby("city"):
    if len(s) < 400:
        continue
    s = s.copy()
    s["q"] = pd.qcut(s["n_within_1_0km"], 4, labels=False, duplicates="drop")
    gg = s.groupby("q").agg(n=("is_open", "size"), closure=("is_open", lambda x: 1 - x.mean()),
                            avg=("n_within_1_0km", "mean"))
    for q, row in gg.iterrows():
        city_rows.append({"city": city, "quartile": int(q) + 1, "n": int(row.n),
                          "avg_neighbours": row.avg, "closure_rate": row.closure})
    line = " | ".join(f"Q{int(q)+1} {r.closure:.1%}" for q, r in gg.iterrows())
    print(f"  {city:16} {line}")
pd.DataFrame(city_rows).to_csv(os.path.join(OUT, "03_findingC_density_within_city.csv"),
                               index=False, encoding="utf-8-sig")
print("\ndone.")
