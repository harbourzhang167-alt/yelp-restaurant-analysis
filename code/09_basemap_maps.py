"""Step 9 - final metro maps on a REAL OpenStreetMap basemap.

Overlays the restaurant points on stitched OSM raster tiles, so each map shows
the actual streets, street names, parks, water and landmarks underneath.

Scales (per metro):
    cluster  4.8 km  zoom 14
    district 1.5 km  zoom 16
    block    0.6 km  zoom 17

Files: analysis/out/charts/basemap/<rank>_<slug>_<scale>.jpg
       (JPEG q90 - the stitched basemap is photographic, so PNG only bloats it)
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import INK, MUTED, font  # noqa: E402
from tiles import basemap  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts", "basemap")
os.makedirs(IMG, exist_ok=True)

RED, GREEN = (198, 62, 48), (28, 122, 62)

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
c2m = pd.read_csv(os.path.join(OUT, "06_city_to_metro.csv"),
                  usecols=["business_id", "metro_id"])
feat = feat.merge(c2m, on="business_id", how="left")
metro = pd.read_csv(os.path.join(OUT, "06_metro_summary.csv"))
us = feat[feat.metro_id.notna() & feat.latitude.notna()]
GLOBAL_MX = float(us["n_within_1_0km"].quantile(0.99))

SCALES = [("cluster", 2.4, 14, "城市群尺度 4.8 km"),
          ("district", 0.75, 16, "街区尺度 1.5 km"),
          ("block", 0.30, 17, "街口尺度 0.6 km")]

CANVAS = 1180          # basemap square size in px
PAD_T, PAD_B = 104, 122


def slug_for(label):
    return "m" + "".join(c if c.isalnum() else "_" for ch in str(label).lower() for c in [ch])


def render(m, rank, scale_name, half_km, zoom, label, clat, clon, segs_unused):
    w = CANVAS
    h = CANVAS + PAD_T + PAD_B
    base, proj, got, miss = basemap(clat, clon, half_km, zoom=zoom,
                                    out_w=CANVAS, out_h=CANVAS)

    img = Image.new("RGB", (w, h), "#ffffff")
    # dim the basemap slightly so the points read clearly on top of it
    base = Image.blend(base, Image.new("RGB", base.size, "#ffffff"), 0.18)
    img.paste(base, (0, PAD_T))
    d = ImageDraw.Draw(img)

    # restaurants of THIS metro inside the window
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.2, math.cos(math.radians(clat))))
    sel = us[(us.metro_id == m.metro_id)
             & us.longitude.between(clon - dlon, clon + dlon)
             & us.latitude.between(clat - dlat, clat + dlat)]
    if len(sel):
        dens = sel["n_within_1_0km"].to_numpy(float)
        px, py = proj(sel.longitude.to_numpy(float), sel.latitude.to_numpy(float))
        isopen = sel.is_open.to_numpy()
        rmax = 7.5 if zoom >= 17 else (6.0 if zoom >= 16 else 4.4)
        for i in np.argsort(isopen):            # closed drawn last
            r = 2.2 + (rmax - 2.2) * (dens[i] / max(GLOBAL_MX, 1)) ** 0.4
            x, y = px[i], py[i] + PAD_T
            d.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill="#ffffff")
            d.ellipse([x - r, y - r, x + r, y + r],
                      fill=RED if isopen[i] == 0 else GREEN)

    # header / footer
    d.rectangle([0, 0, w, PAD_T], fill="#ffffff")
    d.rectangle([0, h - PAD_B, w, h], fill="#ffffff")
    d.text((22, 16), f"NROA 餐厅留存 · {m.metro_label} · {label}", font=font(25, True), fill=INK)
    d.text((22, 54), f"{m.restaurants:,} 家门店 · 关闭率 {m['closure_rate']:.1%} · "
                     f"1km 内中位竞争对手 {m['median_density']:.0f} 家",
           font=font(15), fill=MUTED)
    d.text((22, 76), f"底图 © OpenStreetMap contributors ｜ 门店数据 business_features.csv（52,286 家）",
           font=font(12), fill=MUTED)
    if len(sel):
        d.text((22, h - PAD_B + 14),
               f"视野内 {len(sel):,} 家门店，关闭 {1 - sel.is_open.mean():.1%}",
               font=font(17, True), fill=INK)
    d.text((22, h - PAD_B + 44), "红 = 已关闭 ｜ 绿 = 营业中 ｜ 点大小 = 1km 内竞争对手数",
           font=font(14), fill=INK)
    d.text((22, h - PAD_B + 70),
           "注：同一点位可能多店重叠，密集处红色优先显示，故图上红色占比高于真实关闭率",
           font=font(12), fill=MUTED)
    d.rectangle([0, PAD_T, w - 1, PAD_T + CANVAS], outline="#9fb0bd", width=2)

    # JPEG rather than PNG: the stitched OSM basemap is photographic, so PNG
    # compression is nearly useless here (measured 81-94% of original at every
    # downscale), while JPEG q90 cuts a panel from ~2.3 MB to ~0.4 MB with map
    # labels still legible. Keeps the published repo ~8 MB instead of ~59 MB.
    name = f"{rank:02d}_{slug_for(m.metro_label)}_{scale_name}.jpg"
    img.save(os.path.join(IMG, name), "JPEG", quality=90, optimize=True)
    return name, len(sel), got, miss


TOP_N = int(os.environ.get("METRO_TOP_N", "12"))
SCALE_FILTER = os.environ.get("SCALES", "cluster,district,block").split(",")
sel_metros = metro.dropna(subset=["lat", "lon"]).sort_values(
    "restaurants", ascending=False).head(TOP_N)

print(f"rendering {len(sel_metros)} metros on OSM basemap; scales={SCALE_FILTER}")
total = 0
for rank, (_, m) in enumerate(sel_metros.iterrows(), start=1):
    sub = us[us.metro_id == m.metro_id]
    if len(sub) < 200:
        continue
    hi = sub.nlargest(max(40, len(sub) // 8), "n_within_1_0km")
    clat, clon = float(hi.latitude.median()), float(hi.longitude.median())
    print(f"[{rank}] {m.metro_label}  n={len(sub)}")
    for scale_name, half_km, zoom, label in SCALES:
        if scale_name not in SCALE_FILTER:
            continue
        name, seen, got, miss = render(m, rank, scale_name, half_km, zoom, label,
                                       clat, clon, None)
        total += 1
        print(f"     {name}  ({seen} in view, tiles {got}ok/{miss}miss)")

print(f"\nrendered {total} maps into {os.path.relpath(IMG, WS)}")
