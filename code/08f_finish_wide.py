"""Finish the wide (2.6 km) street windows, prioritising near-complete cities.

Why a separate script: the generic fetcher processes cities in rank order and, on
hitting a stubborn tile, sits in a long retry loop while later cities never get
started. Most cities here were only 1-3 tiles short, so finishing them first
unlocks the cluster-scale maps immediately.

Each tile gets one attempt per pass with a bounded timeout; failures are simply
retried on the next pass.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streetcache import (CACHE, ENDPOINTS, ROAD_CLASSES, STEP_LAT, TILE_DIR,  # noqa: E402
                         UA, cells_for)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
WIDE_KM = 2.6
PASSES = int(os.environ.get("PASSES", "8"))
TILE_TIMEOUT = float(os.environ.get("TILE_TIMEOUT", "30"))


def slug_for(label):
    return "m" + "".join(c if c.isalnum() else "_" for c in str(label).lower())


def fetch_one(i, j, empty_ok=None):
    """One attempt at one tile. Returns 'ok' | 'empty' | 'fail'."""
    path = os.path.join(TILE_DIR, f"t_{i}_{j}.json")
    if os.path.exists(path):
        return "ok"
    import math
    s = i * STEP_LAT
    n = s + STEP_LAT
    step_lon = STEP_LAT / max(0.2, math.cos(math.radians((s + n) / 2)))
    w = j * step_lon
    e = w + step_lon
    query = (f"[out:json][timeout:25];\n"
             f'way["highway"~"^({ROAD_CLASSES})$"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});\n'
             f"out geom;")
    body = urllib.parse.urlencode({"data": query}).encode()
    try:
        req = urllib.request.Request(ENDPOINTS[0], data=body, headers=UA)
        with urllib.request.urlopen(req, timeout=TILE_TIMEOUT) as r:
            data = json.loads(r.read().decode())
        segs = []
        for el in data.get("elements", []):
            geom = el.get("geometry") or []
            if len(geom) < 2:
                continue
            tags = el.get("tags", {})
            segs.append({"highway": tags.get("highway", "unclassified"),
                         "name": tags.get("name", ""),
                         "coords": [[p["lon"], p["lat"]] for p in geom]})
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(segs, fh)
        return "ok"
    except Exception:
        return "fail"


def main():
    feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
    c2m = pd.read_csv(os.path.join(OUT, "06_city_to_metro.csv"),
                      usecols=["business_id", "metro_id"])
    feat = feat.merge(c2m, on="business_id", how="left")
    metro = pd.read_csv(os.path.join(OUT, "06_metro_summary.csv"))
    sel = metro.dropna(subset=["lat", "lon"]).sort_values(
        "restaurants", ascending=False).head(12)

    # build the work list, sorted by tiles-needed ascending
    jobs = []
    for _, m in sel.iterrows():
        slug = slug_for(m.metro_label)
        out_path = os.path.join(CACHE, f"{slug}.json")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue
        sub = feat[feat.metro_id == m.metro_id]
        hi = sub.nlargest(max(40, len(sub) // 8), "n_within_1_0km")
        clat, clon = float(hi.latitude.median()), float(hi.longitude.median())
        cells = cells_for(clat, clon, WIDE_KM)
        miss = [c for c in cells if not os.path.exists(
            os.path.join(TILE_DIR, f"t_{c[0]}_{c[1]}.json"))]
        if miss:
            jobs.append((len(miss), m.metro_label, slug, cells, clat, clon))
    jobs.sort(key=lambda t: t[0])
    print("cities short of a wide window (fewest tiles first):")
    for nmiss, label, *_ in jobs:
        print(f"   {nmiss:>3} tiles  {label}")
    print()

    for p in range(1, PASSES + 1):
        remaining = []
        for nmiss, label, slug, cells, clat, clon in jobs:
            miss = [c for c in cells if not os.path.exists(
                os.path.join(TILE_DIR, f"t_{c[0]}_{c[1]}.json"))]
            if not miss:
                continue
            got = fail = 0
            t0 = time.time()
            for c in miss:
                r = fetch_one(*c)
                if r == "ok":
                    got += 1
                else:
                    fail += 1
            still = [c for c in cells if not os.path.exists(
                os.path.join(TILE_DIR, f"t_{c[0]}_{c[1]}.json"))]
            print(f"pass {p}: {label:22} +{got} ok, {fail} fail, {len(still)} left  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if not still:
                segs = []
                for c in cells:
                    with open(os.path.join(TILE_DIR, f"t_{c[0]}_{c[1]}.json"),
                              encoding="utf-8") as fh:
                        segs.extend(json.load(fh))
                with open(os.path.join(CACHE, f"{slug}.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump(segs, fh)
                print(f"          wrote {slug}: {len(segs)} segments", flush=True)
            else:
                remaining.append((len(still), label, slug, cells, clat, clon))
        jobs = sorted(remaining, key=lambda t: t[0])
        if not jobs:
            print("all wide windows complete")
            break
        print(f"--- end of pass {p}: {len(jobs)} cities remaining ---\n", flush=True)

    print("done")


if __name__ == "__main__":
    main()
