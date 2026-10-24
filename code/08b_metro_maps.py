"""Step 8b - one image per large metro at three scales, with real streets.

Density is the core problem: at a 5 km view a central metro packs ~2,700 points
into ~1,300 px, so nothing is separable. This therefore renders three zooms per
metro:

    cluster  4.8 km  - the shape of the whole urban cluster
    district 1.5 km  - neighbourhood level, individual blocks visible
    block    0.6 km  - street level, individual restaurants separable

Streets are cached by analysis/08c_fetch_streets.py (OpenStreetMap, ODbL).
Files: analysis/out/charts/metro/metro<rank>_<scale>.png
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import GRID, INK, MUTED, font  # noqa: E402
from mapkit import Projector  # noqa: E402
from streetcache import CACHE  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts", "metro")
os.makedirs(IMG, exist_ok=True)

RED, GREEN = (193, 85, 59), (91, 140, 90)

# colour, width(px), label?
# Streets must stay clearly visible under the points without competing with them,
# so they sit in a mid-tone warm grey rather than near-white.
STYLE = {
    "motorway": ((247, 178, 96), 5, False),
    "trunk": ((247, 178, 96), 5, True),
    "primary": ((248, 202, 145), 4, True),
    "secondary": ((243, 218, 178), 3, True),
    "tertiary": ((222, 222, 219), 2, False),
    "unclassified": ((214, 214, 211), 2, False),
    "residential": ((210, 210, 207), 1, False),
    "living_street": ((210, 210, 207), 1, False),
}

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
c2m = pd.read_csv(os.path.join(OUT, "06_city_to_metro.csv"),
                  usecols=["business_id", "metro_id"])
feat = feat.merge(c2m, on="business_id", how="left")
metro = pd.read_csv(os.path.join(OUT, "06_metro_summary.csv"))
us = feat[feat.metro_id.notna() & feat.latitude.notna()]
GLOBAL_MX = float(us["n_within_1_0km"].quantile(0.99))

SCALES = [("cluster", 2.4, "城市群尺度 4.8 km"),
          ("district", 0.75, "街区尺度 1.5 km"),
          ("block", 0.30, "街口尺度 0.6 km")]


def slug_for(label):
    return "m" + "".join(ch if ch.isalnum() else "_" for ch in str(label).lower())


def load_streets(slug):
    p = os.path.join(CACHE, f"{slug}.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_any(slug):
    """Prefer the wide 2.6 km window, then the 2.0 km window, then the 0.9 km one."""
    for suffix, kind in (("", "full"), ("_w", "block"), ("_block", "block")):
        segs = load_streets(slug + suffix)
        if segs:
            return segs, kind
    return [], "none"


def street_center(segs, fallback):
    """Centre on the cached street extent, so the view always contains streets.

    The fetcher centres tiles on the metro's dense core; deriving the render
    centre independently (e.g. from a different quantile) can put the view
    kilometers outside the cached area, which renders streets-less panels.
    """
    if not segs:
        return fallback
    lons = [c[0] for s in segs for c in s["coords"]]
    lats = [c[1] for s in segs for c in s["coords"]]
    return float(np.median(lats)), float(np.median(lons))


def render(m, rank, scale_name, half_km, scale_label, clat, clon, segs, outdir):
    W, H = 1360, 1140
    img = Image.new("RGB", (W, H), "#fcfdfe")
    d = ImageDraw.Draw(img)

    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.2, math.cos(math.radians(clat))))
    vl = (clon - dlon, clon + dlon)
    vt = (clat - dlat, clat + dlat)
    box = (44, 104, W - 44, H - 128)
    proj = Projector(box, list(vl), list(vt), pad=0.0)

    # ---- streets (all classes present on this page)
    keep = []
    for s in segs:
        a = s["coords"]
        if (max(c[0] for c in a) < vl[0] or min(c[0] for c in a) > vl[1]
                or max(c[1] for c in a) < vt[0] or min(c[1] for c in a) > vt[1]):
            continue
        keep.append(s)
    keep.sort(key=lambda s: STYLE.get(s["highway"], ((0, 0, 0), 1, False))[1])
    for s in keep:
        col, w, _ = STYLE.get(s["highway"], ((238, 238, 236), 1, False))
        arr = np.asarray(s["coords"], float)
        px, py = proj(arr[:, 0], arr[:, 1])
        d.line(list(zip(px.tolist(), py.tolist())), fill=col, width=max(1, int(w * (0.8 if half_km > 1.5 else 1.0))))

    # ---- street name labels: ONE label per street name, on its longest in-view
    # segment. A long avenue is split into many way segments, so labelling every
    # segment repeats the same name down the whole map.
    best_len = {}
    for s in keep:
        _, _, want = STYLE.get(s["highway"], ((0, 0, 0), 1, False))
        name = s.get("name") or ""
        if not want or not name or len(name) < 4:
            continue
        a = s["coords"]
        span = (max(c[0] for c in a) - min(c[0] for c in a)) + \
               (max(c[1] for c in a) - min(c[1] for c in a))
        if span > best_len.get(name, (0, None))[0]:
            best_len[name] = (span, s)
    used = []
    for name, (_, s) in sorted(best_len.items(),
                              key=lambda kv: -kv[1][0])[:40]:
        arr = np.asarray(s["coords"], float)
        # place the label near the VIEW centre, not at the segment midpoint: a long
        # avenue's middle can fall outside the frame, so the name would never show
        dist = (arr[:, 0] - clon) ** 2 + (arr[:, 1] - clat) ** 2
        j = int(np.argmin(dist))
        mid = arr[j]
        if not (vl[0] < mid[0] < vl[1] and vt[0] < mid[1] < vt[1]):
            continue
        x, y = proj(mid[0], mid[1])
        f = font(11)
        tw = d.textbbox((0, 0), name, f)[2]
        bb = (x - tw / 2 - 3, y - 7, x + tw / 2 + 3, y + 9)
        if any(not (bb[2] < u[0] or bb[0] > u[2] or bb[3] < u[1] or bb[1] > u[3]) for u in used):
            continue
        used.append(bb)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((x - tw / 2 + dx, y - 7 + dy), name, font=f, fill="#ffffff")
        d.text((x - tw / 2, y - 7), name, font=f, fill=(88, 102, 114))

    # ---- restaurants for THIS metro (not the whole dataset)
    sel = us[(us.metro_id == m.metro_id)
             & us.longitude.between(*vl) & us.latitude.between(*vt)]
    if len(sel):
        dens = sel["n_within_1_0km"].to_numpy(float)
        px, py = proj(sel.longitude.to_numpy(float), sel.latitude.to_numpy(float))
        isopen = sel.is_open.to_numpy()
        rmax = 6.0 if half_km <= 0.4 else (4.6 if half_km <= 1.0 else 3.4)
        for i in np.argsort(isopen):
            r = 1.7 + (rmax - 1.7) * (dens[i] / max(GLOBAL_MX, 1)) ** 0.4
            d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r],
                      fill=RED if isopen[i] == 0 else GREEN)

    d.rectangle(box, outline="#93a5b3", width=2)
    d.text((44, 22), f"NROA 餐厅留存 · {m.metro_label} · {scale_label}",
           font=font(24, True), fill=INK)
    d.text((44, 58),
           f"{m.restaurants:,} 家门店 · 关闭率 {m['closure_rate']:.1%} · "
           f"1km 内中位竞争对手 {m['median_density']:.0f} 家 · "
           f"街道 © OpenStreetMap contributors",
           font=font(15), fill=MUTED)
    d.text((44, H - 100), f"视野内 {len(sel):,} 家门店，关闭 {1 - sel.is_open.mean():.1%}"
           if len(sel) else "视野内无门店", font=font(16, True), fill=INK)
    d.text((44, H - 72), "红 = 已关闭 ｜ 绿 = 营业中 ｜ 点大小 = 1km 内竞争对手数",
           font=font(14), fill=INK)
    d.text((44, H - 46), "数据：business_features.csv（52,286 家）｜ 街道：OpenStreetMap via Overpass（ODbL）",
           font=font(12), fill=MUTED)
    d.text((44, H - 24), "注：同一点位可能多店重叠；密集处红色优先显示，故红色占比偏高于真实关闭率",
           font=font(12), fill=MUTED)

    name = f"metro{rank:02d}_{slug_for(m.metro_label)[:26]}_{scale_name}.png"
    img.save(os.path.join(outdir, name))
    return name, len(sel)


TOP_N = int(os.environ.get("METRO_TOP_N", "12"))
sel = metro.dropna(subset=["lat", "lon"]).sort_values("restaurants", ascending=False).head(TOP_N)

made = 0
no_streets = []
for rank, (_, m) in enumerate(sel.iterrows(), start=1):
    sub = us[us.metro_id == m.metro_id]
    if len(sub) < 200:
        continue
    segs, kind = load_any(slug_for(m.metro_label))
    hi = sub.nlargest(max(40, len(sub) // 8), "n_within_1_0km")
    dense = (float(hi.latitude.median()), float(hi.longitude.median()))
    clat, clon = street_center(segs, dense)
    if kind == "none":
        no_streets.append(m.metro_label)
    print(f"[{rank}] {m.metro_label}  n={len(sub)}  streets={len(segs)} ({kind})")
    for scale_name, half_km, label in SCALES:
        # The street-level scale is always produced. The district scale is
        # produced when the cached window covers it. The 4.8 km cluster scale is
        # only drawn when the wide (2.6 km) cache exists - otherwise the panel
        # would show barely any streets.
        if scale_name == "cluster" and kind != "full":
            continue
        if scale_name == "district" and kind == "none":
            continue
        use_lat, use_lon = (clat, clon) if segs else dense
        n, seen = render(m, rank, scale_name, half_km, label, use_lat, use_lon, segs, IMG)
        made += 1
        print(f"     {n}  ({seen} in view)")

print(f"\nrendered {made} images into {os.path.relpath(IMG, WS)}")
if no_streets:
    print(f"metros still without a street cache ({len(no_streets)}): {', '.join(no_streets)}")
    print("  -> rerun analysis/08d_fetch_block_streets.py, then rerun this script")
