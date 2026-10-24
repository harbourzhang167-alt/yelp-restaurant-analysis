"""Fetch and cache real street geometry from OpenStreetMap's Overpass API.

Used to give the small-scale maps authentic street networks. Overpass is the
correct endpoint for programmatic queries (unlike the public raster tile server,
whose usage policy discourages bulk downloading).

Cache lives in analysis/data/streets/<slug>.json so re-rendering needs no network.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(WS, "analysis", "data", "streets")
os.makedirs(CACHE, exist_ok=True)

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Road classes worth drawing, split so a dense downtown can be requested in
# progressively lighter passes. `service`/`residential` are the bulk of a dense
# city and are what make a single query time out, so they are optional.
CORE_CLASSES = "motorway|trunk|primary|secondary|tertiary"
MINOR_CLASSES = "unclassified|residential|living_street|pedestrian|service"


HIGHWAY_FILTER = f"{CORE_CLASSES}|{MINOR_CLASSES}"


def _run_query(query: str, tries: int = 3) -> dict:
    body = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for attempt in range(tries):
        for ep in ENDPOINTS:
            try:
                req = urllib.request.Request(
                    ep, data=body,
                    headers={"User-Agent": "bdao-coursework-restaurant-analysis/1.0"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    return json.loads(r.read().decode())
            except Exception as exc:
                last = f"{ep}: {type(exc).__name__}"
                time.sleep(4 + 6 * attempt)
    raise RuntimeError(f"all Overpass endpoints failed ({last})")


def fetch_streets(slug: str, south: float, west: float, north: float, east: float,
                  force: bool = False, include_minor: bool = True) -> list:
    """Return [{highway, name, coords:[[lon,lat],...]}] for the bounding box."""
    path = os.path.join(CACHE, f"{slug}.json")
    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    classes = HIGHWAY_FILTER if include_minor else CORE_CLASSES
    query = (f"[out:json][timeout:180];\n"
             f'way["highway"~"^({classes})$"]({south},{west},{north},{east});\n'
             f"out geom;")
    data = _run_query(query)

    out = []
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        out.append({
            "highway": tags.get("highway", "unclassified"),
            "name": tags.get("name", ""),
            "coords": [[p["lon"], p["lat"]] for p in geom],
        })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"    cached {len(out)} street segments -> {os.path.relpath(path, WS)}")
    return out


def tile(slug, lat, lon, half_km, force=False):
    """Convenience: fetch a square window of half_km around a point."""
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.2, abs(__import__("math").cos(__import__("math").radians(lat)))))
    return fetch_streets(slug, lat - dlat, lon - dlon, lat + dlat, lon + dlon, force=force)


if __name__ == "__main__":
    # smoke test: Center City Philadelphia
    segs = tile("test_philadelphia_3km", 39.9526, -75.1652, 3.0)
    kinds = {}
    for s in segs:
        kinds[s["highway"]] = kinds.get(s["highway"], 0) + 1
    print("street classes:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    named = sum(1 for s in segs if s["name"])
    print(f"segments: {len(segs)}, with a street name: {named}")
    print("sample names:", sorted({s["name"] for s in segs if s["name"]})[:12])
