"""Cache street networks for the top metros (resumable; run repeatedly).

Centres each window on the DENSE part of the metro, not on its metric centroid.
The two differ a lot - in Tampa the densest decile sits 5.5 km from the centroid -
and the maps are drawn around the dense part, so centring the tiles on the
centroid left every rendered panel without streets.

Each metro needs roughly 25-40 small tiles at ~5 s each. Overpass occasionally
504s, so this reports progress and can simply be re-run to fill the gaps.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streetcache import CACHE, build_city, cells_for  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
TOP_N = int(os.environ.get("METRO_TOP_N", "12"))
HALF_KM = 2.6      # covers the 4.8 km cluster view with margin


def slug_for(label: str) -> str:
    return "m" + "".join(ch if ch.isalnum() else "_" for ch in str(label).lower())


def hash_dense_center(feat, metro):
    """Centre = median coordinate of the metro's densest eighth of restaurants."""
    c2m = pd.read_csv(os.path.join(OUT, "06_city_to_metro.csv"),
                      usecols=["business_id", "metro_id"])
    key = f"{metro.metro_id}"
    sub = c2m[c2m.metro_id.astype(str) == key]
    sub = feat[feat.business_id.isin(sub.business_id)]
    hi = sub.nlargest(max(40, len(sub) // 8), "n_within_1_0km")
    return float(hi.latitude.median()), float(hi.longitude.median()), len(sub)


def main():
    feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
    c2m = pd.read_csv(os.path.join(OUT, "06_city_to_metro.csv"),
                      usecols=["business_id", "metro_id"])
    feat = feat.merge(c2m, on="business_id", how="left")
    metro = pd.read_csv(os.path.join(OUT, "06_metro_summary.csv"))
    sel = metro.dropna(subset=["lat", "lon"]).sort_values(
        "restaurants", ascending=False).head(TOP_N)

    print(f"metros: {len(sel)}   window half-width {HALF_KM} km (around the dense core)")
    done = skipped = 0
    for k, (_, m) in enumerate(sel.iterrows(), 1):
        slug = slug_for(m.metro_label)
        path = os.path.join(CACHE, f"{slug}.json")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped += 1
            continue
        sub = feat[feat.metro_id == m.metro_id]
        if len(sub) < 200:
            continue
        hi = sub.nlargest(max(40, len(sub) // 8), "n_within_1_0km")
        clat, clon = float(hi.latitude.median()), float(hi.longitude.median())
        cells = cells_for(clat, clon, HALF_KM)
        print(f"[{k}/{len(sel)}] {m.metro_label}  centre=({clat:.4f},{clon:.4f})  "
              f"{len(cells)} tiles", flush=True)
        n = build_city(slug, clat, clon, HALF_KM, budget_seconds=900)
        if n:
            done += 1
            print(f"          ok {n} segments", flush=True)
        else:
            print("          incomplete (rerun to resume)", flush=True)
    print(f"\ncached {done}, already had {skipped}, of {len(sel)}")


if __name__ == "__main__":
    main()
