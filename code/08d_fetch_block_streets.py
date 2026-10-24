"""Cache a small (street-level) street window for each top metro - fast and robust.

The 2.6 km windows need ~25 tiles each and often lose one to an Overpass 504, so
nothing gets written for a long time. A 0.45 km window needs only ~1-4 tiles, so
this completes quickly and guarantees every metro gets a genuine street-level map.

Writes analysis/data/streets/<slug>_block.json
"""

from __future__ import annotations

import os
import sys
import time

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
# 1.0 km half-width (2.0 km across) is a deliberate compromise: large enough to
# cover the district view (1.5 km) from the same tiles, small enough that a metro
# needs only 4-9 tiles and therefore finishes despite Overpass 504s. The 2.6 km
# wide window needs ~25 tiles and rarely completes.
HALF_KM = float(os.environ.get("BLOCK_HALF_KM", "1.0"))


def slug_for(label):
    return "m" + "".join(ch if ch.isalnum() else "_" for ch in str(label).lower())


def main():
    feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
    c2m = pd.read_csv(os.path.join(OUT, "06_city_to_metro.csv"),
                      usecols=["business_id", "metro_id"])
    feat = feat.merge(c2m, on="business_id", how="left")
    metro = pd.read_csv(os.path.join(OUT, "06_metro_summary.csv"))
    sel = metro.dropna(subset=["lat", "lon"]).sort_values(
        "restaurants", ascending=False).head(TOP_N)

    done = skipped = 0
    for k, (_, m) in enumerate(sel.iterrows(), 1):
        sub = feat[feat.metro_id == m.metro_id]
        if len(sub) < 200:
            continue
        hi = sub.nlargest(max(40, len(sub) // 10), "n_within_1_0km")
        clat, clon = float(hi.latitude.median()), float(hi.longitude.median())
        slug = slug_for(m.metro_label) + "_w"
        path = os.path.join(CACHE, f"{slug}.json")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped += 1
            continue
        cells = cells_for(clat, clon, HALF_KM)
        t0 = time.time()
        print(f"[{k}/{len(sel)}] {m.metro_label}  {len(cells)} tiles", flush=True)
        n = build_city(slug, clat, clon, HALF_KM, budget_seconds=240)
        if n:
            done += 1
            print(f"          ok {n} segments in {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"          incomplete in {time.time()-t0:.0f}s", flush=True)
    print(f"\nblock windows: cached {done}, already had {skipped}")


if __name__ == "__main__":
    main()
