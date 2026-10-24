"""Cache OpenStreetMap raster basemap tiles and stitch a real street basemap.

Why this exists: the Overpass API (vector streets) became unreachable - SSL
handshake failures on the main endpoint, HTTP 500 on the mirrors - while
tile.openstreetmap.org stayed healthy. Raster tiles give a proper street basemap
with labels, and need only ~6-30 requests per map instead of hundreds.

Tiles are cached under analysis/data/tiles/<z>/<x>/<y>.png, so re-rendering is
offline. Usage stays light and cached, in line with the OSM tile policy.
Attribution (© OpenStreetMap contributors) is drawn on every map.

Web Mercator. Zoom 16 ~ 2.4 m/px at this latitude, so a 5 km map is a small grid.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
import urllib.request

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_ROOT = os.path.join(WS, "analysis", "data", "tiles")
os.makedirs(TILE_ROOT, exist_ok=True)

SERVERS = ["https://tile.openstreetmap.org"]     # a./b. mirrors also work
UA = {"User-Agent": "bdao-coursework-restaurant-analysis/1.0 (academic project)"}
TILE = 256


# ------------------------------------------------------------------ mercator
def lonlat_to_px(lon, lat, z):
    """Global pixel coordinates in Web Mercator at zoom z (scalar or array)."""
    n = TILE * (2 ** z)
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    s = np.sin(np.radians(np.asarray(lat, float)))
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    return x, y


def px_to_lonlat(x, y, z):
    n = TILE * (2 ** z)
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def fetch_tile(z, x, y, tries=4):
    path = os.path.join(TILE_ROOT, str(z), str(x), f"{y}.png")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last = None
    for a in range(tries):
        url = f"{SERVERS[a % len(SERVERS)]}/{z}/{x}/{y}.png"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            with open(path, "wb") as fh:
                fh.write(data)
            time.sleep(0.15)          # be polite to the tile server
            return path
        except Exception as exc:
            last = f"{type(exc).__name__}"
            time.sleep(1.0 + a)
    print(f"    tile {z}/{x}/{y} failed ({last})")
    return None


def basemap(clat, clon, half_km, zoom=16, out_w=1400, out_h=1400):
    """Return (RGB array, project(lon,lat)->(px,py)) for a square window.

    The image is downloaded as a tile grid then cropped to the exact window, so
    overlay coordinates are exact rather than approximated.
    """
    from PIL import Image

    # window half-size in global pixels
    px_per_km = (TILE * (2 ** zoom)) / 40075.016686 / math.cos(math.radians(clat))
    half_px = half_km * px_per_km
    cx, cy = lonlat_to_px(clon, clat, zoom)

    x0, x1 = cx - half_px, cx + half_px
    y0, y1 = cy - half_px, cy + half_px
    tx0, tx1 = int(x0 // TILE), int(x1 // TILE)
    ty0, ty1 = int(y0 // TILE), int(y1 // TILE)

    W = (tx1 - tx0 + 1) * TILE
    H = (ty1 - ty0 + 1) * TILE
    canvas = Image.new("RGB", (W, H), "#f2f2f0")
    got = miss = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            p = fetch_tile(zoom, tx, ty)
            if not p:
                miss += 1
                continue
            canvas.paste(Image.open(p).convert("RGB"),
                         ((tx - tx0) * TILE, (ty - ty0) * TILE))
            got += 1

    # crop the exact window, then scale to the output size
    ox, oy = x0 - tx0 * TILE, y0 - ty0 * TILE
    crop = canvas.crop((int(round(ox)), int(round(oy)),
                        int(round(ox + 2 * half_px)), int(round(oy + 2 * half_px))))
    crop = crop.resize((out_w, out_h), Image.LANCZOS)
    scale = out_w / (2 * half_px)          # output px per global px

    def project(lon, lat):
        gx, gy = lonlat_to_px(lon, lat, zoom)
        return (gx - x0) * scale, (gy - y0) * scale

    return crop, project, got, miss
