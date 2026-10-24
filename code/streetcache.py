"""Cache street networks in small tiles (Overpass only tolerates small bboxes).

A ~0.012 deg tile is about 1.3 km x 1.1 km and comes back in a few seconds with
all road classes. Larger requests reliably fail with 504/429, so we tile, cache
each tile globally (tiles are shared between nearby cities) and merge per city.

Resumable: cached tiles are skipped, and each metro is written to
analysis/data/streets/<slug>.json once its tiles are all present.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(WS, "analysis", "data", "streets")
TILE_DIR = os.path.join(CACHE, "tiles")
os.makedirs(TILE_DIR, exist_ok=True)

# Only endpoints that actually answer are kept: overpass.kumi.systems and
# overpass.private.coffee both return HTTP 500 on this host, and rotating onto a
# dead endpoint just burns the retry budget before a good one is tried.
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
]
STEP_LAT = 0.012
UA = {"User-Agent": "bdao-coursework-restaurant-analysis/1.0 (academic project)"}

# Drivable roads only. Asking for `way["highway"]` also returns footways, steps
# and paths, which dominate the result (e.g. 1,135 of 1,627 ways in a Center City
# tile) and add nothing to a restaurant map.
ROAD_CLASSES = ("motorway|trunk|primary|secondary|tertiary|"
                "unclassified|residential|living_street")


def tile_key(lat_i, lon_i):
    return f"{lat_i}_{lon_i}"


def fetch_tile(lat_i, lon_i):
    """Fetch one small tile; cached on disk. Returns list of segments or []."""
    path = os.path.join(TILE_DIR, f"t_{lat_i}_{lon_i}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    s = lat_i * STEP_LAT
    n = s + STEP_LAT
    # keep tiles roughly square in metres
    mid = (s + n) / 2
    step_lon = STEP_LAT / max(0.2, math.cos(math.radians(mid)))
    w = lon_i * step_lon
    e = w + step_lon

    query = (f"[out:json][timeout:60];\n"
             f'way["highway"~"^({ROAD_CLASSES})$"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});\n'
             f"out geom;")
    body = urllib.parse.urlencode({"data": query}).encode()

    for attempt in range(6):
        ep = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            req = urllib.request.Request(ep, data=body, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode())
            segs = []
            for el in data.get("elements", []):
                geom = el.get("geometry") or []
                if len(geom) < 2:
                    continue
                tags = el.get("tags", {})
                segs.append({
                    "highway": tags.get("highway", "unclassified"),
                    "name": tags.get("name", ""),
                    "coords": [[p["lon"], p["lat"]] for p in geom],
                })
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(segs, fh)
            return segs
        except Exception:
            time.sleep(2 + attempt * 2 + random.random())
    return None


def cells_for(lat, lon, half_km):
    """Tile indices covering a square window."""
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
    i0 = int(math.floor((lat - dlat) / STEP_LAT))
    i1 = int(math.floor((lat + dlat) / STEP_LAT))
    step_lon = STEP_LAT / max(0.2, math.cos(math.radians(lat)))
    j0 = int(math.floor((lon - dlon) / step_lon))
    j1 = int(math.floor((lon + dlon) / step_lon))
    return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def build_city(slug, lat, lon, half_km, budget_seconds=150.0):
    """Merge tiles for one city. Returns number of segments, or None if incomplete."""
    out_path = os.path.join(CACHE, f"{slug}.json")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", encoding="utf-8") as fh:
            return len(json.load(fh))

    cells = cells_for(lat, lon, half_km)
    segs = []
    t0 = time.time()
    missing = 0
    for n_done, (i, j) in enumerate(cells):
        if time.time() - t0 > budget_seconds:
            print(f"    budget hit after {n_done}/{len(cells)} tiles "
                  f"({time.time()-t0:.0f}s); will resume", flush=True)
            return None
        got = fetch_tile(i, j)
        if got is None:
            missing += 1
            print(f"    tile {i},{j} unavailable "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
            if time.time() - t0 > budget_seconds:
                print(f"    budget hit; {missing} unavailable so far", flush=True)
                return None
            continue
        segs.extend(got)
    if missing:
        print(f"    {missing}/{len(cells)} tiles unavailable; not writing {slug}")
        return None
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(segs, fh)
    print(f"    wrote {slug}: {len(segs)} segments from {len(cells)} tiles")
    return len(segs)
