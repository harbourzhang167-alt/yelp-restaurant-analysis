"""Step 8 - small-scale city maps with REAL street networks (fig13 style).

Zoom goes from whole-state (fig16/17, ~300 km) down to ~4 km, where individual
restaurants become distinguishable. Streets come from OpenStreetMap via Overpass
and are cached in analysis/data/streets/.

Outputs (each 2x2 cities):
    fig18_a/b/c/d_city_maps.png   ~4 km across, whole urban cluster
    fig19_a/b_city_zoom.png       ~1.4 km across, densest neighbourhood only
"""

from __future__ import annotations

import io
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chartkit import GRID, INK, MUTED, font  # noqa: E402
from mapkit import Projector  # noqa: E402
from streets import tile  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# Re-wrap stdout for console-safe Chinese output, but never let that break the run:
# a truncated or already-rewrapped stream must not raise.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WS, "analysis", "out")
IMG = os.path.join(OUT, "charts")
os.makedirs(IMG, exist_ok=True)

feat = pd.read_csv(os.path.join(OUT, "business_features.csv"), low_memory=False)
metro = pd.read_csv(os.path.join(OUT, "06_metro_summary.csv"))
# business_features.csv has no metro_id, so join the city->metro mapping produced
# by 06_metros.py. Without this, city-name matching silently matched unrelated
# same-named cities in other states.
c2m = pd.read_csv(os.path.join(OUT, "06_city_to_metro.csv"),
                  usecols=["business_id", "metro_id", "metro_name", "metro_label"])
feat = feat.merge(c2m, on="business_id", how="left")
us = feat[feat.state.isin({"AZ", "CA", "CO", "DE", "FL", "ID", "IL", "IN", "LA",
                           "MO", "MT", "NC", "NJ", "NV", "PA", "TN"})]

RED, GREEN = (193, 85, 59), (91, 140, 90)

# street styling by class
STYLE = {
    "motorway": (255, 208, 140, 2.6), "trunk": (255, 208, 140, 2.4),
    "primary": (255, 224, 178, 2.0), "secondary": (255, 236, 205, 1.5),
    "tertiary": (245, 243, 238, 1.1), "unclassified": (240, 240, 238, 0.9),
    "residential": (238, 238, 236, 0.8), "living_street": (238, 238, 236, 0.8),
    "pedestrian": (232, 232, 230, 0.7), "service": (246, 246, 245, 0.5),
}


def pick_cities(k=14):
    """One panel per large metro, using the consolidated metro_id grouping."""
    m = metro.dropna(subset=["lat", "lon"]).sort_values("restaurants", ascending=False)
    rows = []
    for _, r in m.iterrows():
        sub = us[us.metro_id == r.metro_id]
        if len(sub) < 250:
            continue
        rows.append({
            "label": r.metro_label,
            "name": str(r.member_cities).split(" / ")[0],
            "lat": float(sub.latitude.median()),
            "lon": float(sub.longitude.median()),
            "n": int(len(sub)),
            "closure": float(1 - sub.is_open.mean()),
        })
        if len(rows) >= k:
            break
    return rows


def draw_streets(d, proj, segs, scale_hint):
    """Draw streets minor-first so arterials stay on top."""
    order = sorted(range(len(segs)),
                   key=lambda i: STYLE.get(segs[i]["highway"], (0, 0, 0, 0.7))[3])
    for i in order:
        s = segs[i]
        col_rgb_w = STYLE.get(s["highway"], (240, 240, 238, 0.8))
        arr = np.asarray(s["coords"], float)
        px, py = proj(arr[:, 0], arr[:, 1])
        # width scaled to the view so it looks right at both zoom levels
        w = max(1, int(round(col_rgb_w[3] * scale_hint)))
        d.line(list(zip(px.tolist(), py.tolist())), fill=col_rgb_w[:3], width=w)


def panel(rows, ci, w, h, half_km, colour_by, global_mx, streets_cache):
    """Render one city panel onto its own layer."""
    spec = rows[ci]
    layer = Image.new("RGB", (w, h), "#fbfcfd")
    d = ImageDraw.Draw(layer)

    # square window in degrees
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.2, math.cos(math.radians(spec["lat"]))))
    view_lon = (spec["lon"] - dlon, spec["lon"] + dlon)
    view_lat = (spec["lat"] - dlat, spec["lat"] + dlat)
    box = (0, 0, w, h)
    proj = Projector(box, list(view_lon), list(view_lat), pad=0.0)

    # streets
    slug = spec["slug"]
    segs = streets_cache.get(slug, [])
    if segs:
        # only street segments intersecting the view
        keep = []
        for s in segs:
            arr = np.asarray(s["coords"], float)
            if (arr[:, 0].max() < view_lon[0] or arr[:, 0].min() > view_lon[1]
                    or arr[:, 1].max() < view_lat[0] or arr[:, 1].min() > view_lat[1]):
                continue
            keep.append(s)
        scale_hint = 1.0 if half_km <= 1.0 else 1.4
        draw_streets(d, proj, keep, scale_hint)

    # restaurants in view
    sel = us[(us.longitude.between(*view_lon)) & (us.latitude.between(*view_lat))]
    dens = sel["n_within_1_0km"].to_numpy(float)
    px, py = proj(sel.longitude.to_numpy(float), sel.latitude.to_numpy(float))
    isopen = sel.is_open.to_numpy()
    rmax = 3.6 if half_km <= 1.0 else 3.0
    for i in np.argsort(isopen):        # closed last, stays visible
        r = 1.6 + (rmax - 1.6) * (dens[i] / max(global_mx, 1)) ** 0.4
        col = RED if isopen[i] == 0 else GREEN
        if colour_by == "density":
            f = min((dens[i] / max(global_mx, 1)) ** 0.32, 1)
            col = tuple(int(242 - (242 - c) * f) for c in (47, 111, 159))
        d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=col)

    # header + stats bands
    d.rectangle([0, 0, w, 32], fill="#ffffff")
    d.rectangle([0, h - 40, w, h], fill="#ffffff")
    d.text((10, 5), f"{spec['label']}", font=font(16, True), fill=INK)
    if len(sel):
        stats = (f"视野内 {len(sel):,} 家 · 关闭 {1 - sel.is_open.mean():.1%} · "
                 f"1km 内中位 {sel['n_within_1_0km'].median():.0f} 家")
    else:
        stats = "视野内无门店"
    d.text((10, h - 34), stats, font=font(12), fill=MUTED)
    d.text((10, h - 19), f"范围 {half_km*2:.1f} km × {half_km*2:.1f} km", font=font(11), fill=MUTED)
    d.rectangle([0, 0, w - 1, h - 1], outline=GRID)
    return layer


def render_grid(specs, colour_by, prefix, half_km, per_img=4):
    global_mx = float(us["n_within_1_0km"].quantile(0.99))
    groups = [specs[i:i + per_img] for i in range(0, len(specs), per_img)]
    for gi, group in enumerate(groups):
        W, H = 1520, 1240
        img = Image.new("RGB", (W, H), "#ffffff")
        d = ImageDraw.Draw(img)
        suffix = chr(ord('a') + gi)
        zoom = "城市群尺度" if half_km > 2 else "街区尺度"
        colour_desc = "红=已关闭 / 绿=营业中" if colour_by == "closure" else "颜色越深 = 竞争越密集"
        d.text((30, 22), f"图 {prefix}{suffix}｜{zoom}（{half_km*2:.0f} km 视野，含真实街道）：{colour_desc}",
               font=font(23, True), fill=INK)
        d.text((30, 56), f"街道来自 OpenStreetMap（Overpass API，本地缓存）；"
                         f"点 = 一家餐厅的真实坐标，点大小 = 1km 内竞争对手数",
               font=font(14), fill=MUTED)
        pw, ph = (W - 3 * 22) / 2, (H - 200) / 2
        for k, _ in enumerate(group):
            cx = 22 + (k % 2) * (pw + 22)
            cy = 96 + (k // 2) * (ph + 16)
            lay = panel(group, k, int(pw), int(ph), half_km, colour_by, global_mx, STREETS)
            img.paste(lay, (int(cx), int(cy)))
        # legend
        lx, ly = W - 470, H - 86
        if colour_by == "closure":
            d.ellipse([lx, ly, lx + 14, ly + 14], fill=RED)
            d.text((lx + 22, ly - 1), "已关闭", font=font(14), fill=INK)
            d.ellipse([lx + 120, ly, lx + 134, ly + 14], fill=GREEN)
            d.text((lx + 142, ly - 1), "营业中", font=font(14), fill=INK)
            d.text((lx, ly + 24), "点大小 = 1km 内竞争对手数（越大越密集）", font=font(12), fill=MUTED)
        else:
            for i, nb in enumerate([0, 40, 150, 400, 900]):
                col = tuple(int(242 - (242 - c) * min((nb / global_mx) ** 0.32, 1))
                            for c in (47, 111, 159))
                d.rectangle([lx + i * 52, ly, lx + i * 52 + 40, ly + 16], fill=col, outline="#ffffff")
            d.text((lx, ly + 24), "浅 → 深：1km 内竞争对手数（0 / 40 / 150 / 400 / 900）",
                   font=font(12), fill=MUTED)
        d.text((30, H - 40), "地图数据 © OpenStreetMap contributors（ODbL）｜ 门店与密度数据：business_features.csv",
               font=font(12), fill=MUTED)
        d.text((30, H - 22), "注：同一点位可能有多个门店重叠，密集处颜色以已关闭（红）优先显示",
               font=font(12), fill=MUTED)
        name = f"fig{prefix}{suffix}_city_maps_{'closure' if colour_by=='closure' else 'density'}.png"
        img.save(os.path.join(IMG, name))
        print("  rendered", name)


# --------------------------------------------------------------- fetch streets
specs = pick_cities(14)
print(f"cities to map: {len(specs)}")
for s in specs:
    s["slug"] = "city_" + s["name"].lower().replace(" ", "_").replace(".", "").replace("'", "") \
        + "_" + s["label"].split(",")[-1].strip().lower()
    s["slug"] = s["slug"].replace(",", "")
print("  " + ", ".join(f"{s['name']}({s['n']})" for s in specs))

STREETS = {}
for s in specs:
    try:
        STREETS[s["slug"]] = tile(s["slug"], s["lat"], s["lon"], 3.2)
    except Exception as exc:
        print(f"    {s['name']}: street fetch failed ({type(exc).__name__}) - panel will render without streets")
        STREETS[s["slug"]] = []

# --------------------------------------------------------------- densest zoom
zoom_specs = []
for s in specs[:8]:
    dlat = 0.9 / 111.0
    dlon = 0.9 / (111.0 * max(0.2, math.cos(math.radians(s["lat"]))))
    sel = us[(us.longitude.between(s["lon"] - dlon, s["lon"] + dlon))
             & (us.latitude.between(s["lat"] - dlat, s["lat"] + dlat))]
    if len(sel) < 30:
        continue
    z = dict(s)
    z["lat"] = float(sel.latitude.median())
    z["lon"] = float(sel.longitude.median())
    zoom_specs.append(z)

print(f"\nrendering {len(specs)} city panels and {len(zoom_specs)} zoom panels")
render_grid(specs, "closure", "18_", 2.0, per_img=4)
render_grid([s for s in specs], "density", "18d_", 2.0, per_img=4)
if zoom_specs:
    render_grid(zoom_specs, "closure", "19_", 0.7, per_img=4)
print("done.")
