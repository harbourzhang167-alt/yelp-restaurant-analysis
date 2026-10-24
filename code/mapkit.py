"""Reusable spatial map toolkit on Pillow (no basemap libraries available).

Provides the pieces fig13 used: an Albers projection fitted to a set of points,
crisp state outlines from the cached GeoJSON, restaurant points coloured either by
closure status or by density, and a legend/footnote block.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image, ImageDraw

from chartkit import ACCENT2, ACCENT3, GRID, INK, MUTED, WARN, font

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = os.path.join(WS, "analysis", "data", "us-states.geojson")

ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
NAME_OF_ABBR = {v: k for k, v in ABBR.items()}

GREEN = (91, 140, 90)
RED = (193, 85, 59)


def load_states():
    geo = json.load(open(GEO, encoding="utf-8"))
    out = {}
    for f in geo["features"]:
        out[f["properties"]["name"]] = f["geometry"]
    return out


def rings_of(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    return []


class Projector:
    """Albers equal-area conic, fitted so a point set fills the plotting box."""

    def __init__(self, box, lons, lats, pad=0.02):
        self.box = box
        x, y = self._raw(np.asarray(lons, float), np.asarray(lats, float))
        self.x0, self.x1 = x.min(), x.max()
        self.y0, self.y1 = y.min(), y.max()
        bx0, by0, bx1, by1 = box
        bw, bh = bx1 - bx0, by1 - by0
        dw, dh = max(self.x1 - self.x0, 1e-9), max(self.y1 - self.y0, 1e-9)
        self.s = min(bw / dw, bh / dh) * (1 - pad)
        self.ox = bx0 + (bw - dw * self.s) / 2
        self.oy = by0 + (bh - dh * self.s) / 2

    @staticmethod
    def _raw(lon_deg, lat_deg):
        lon = np.radians(lon_deg)
        lat = np.radians(lat_deg)
        lat0, lat1, lat2 = np.radians(29.5), np.radians(45.5), np.radians(37.5)
        lon0 = np.radians(-96.0)
        n = 0.5 * (np.sin(lat0) + np.sin(lat1))
        c = np.cos(lat0) ** 2 + 2 * n * np.sin(lat0)
        rho = np.sqrt(np.maximum(c - 2 * n * np.sin(lat), 0)) / n
        rho0 = np.sqrt(c - 2 * n * np.sin(lat2)) / n
        th = n * (lon - lon0)
        return rho * np.sin(th), rho0 - rho * np.cos(th)

    def __call__(self, lon, lat):
        x, y = self._raw(np.atleast_1d(lon), np.atleast_1d(lat))
        px = self.ox + (x - self.x0) * self.s
        py = self.oy + (self.y1 - y) * self.s
        if np.isscalar(lon) and np.isscalar(lat):
            return float(px[0]), float(py[0])
        return px, py


def density_colour(n, mx):
    """Light -> dark blue by relative density.

    A power scale (not linear) is required: the median restaurant has ~30
    competitors within 1km while the max is >1,300, so a linear ramp leaves almost
    every point almost white and only the very densest cores visible.
    """
    f = 0.0 if mx <= 0 else min((max(n, 0) / mx) ** 0.32, 1.0)
    return tuple(int(242 - (242 - cc) * f) for cc in (47, 111, 159))


def closure_colour(cr, lo=0.20, hi=0.60):
    """Green (low closure) -> amber -> red (high closure)."""
    t = min(max((cr - lo) / (hi - lo), 0), 1)
    if t < 0.5:
        u = t / 0.5
        return (int(91 + (183 - 91) * u), int(140 + (121 - 140) * u), int(90 + (31 - 90) * u))
    u = (t - 0.5) / 0.5
    return (int(183 + (193 - 183) * u), int(121 + (85 - 121) * u), int(31 + (59 - 31) * u))


def draw_points(d, proj, df, colour_by="closure", max_radius=4.6, min_radius=1.5,
                mx=None, halo=False):
    """Draw restaurant points. colour_by: 'closure' or 'density'."""
    dens = df["n_within_1_0km"].to_numpy(float) if "n_within_1_0km" in df else np.zeros(len(df))
    mxd = mx if mx is not None else max(dens.max(), 1)
    px, py = proj(df.longitude.to_numpy(float), df.latitude.to_numpy(float))
    order = np.argsort(df.is_open.to_numpy()) if colour_by == "closure" else np.arange(len(df))
    for i in order:
        r = min_radius + (max_radius - min_radius) * (dens[i] / mxd) ** 0.45
        if colour_by == "closure":
            col = RED if df.is_open.to_numpy()[i] == 0 else GREEN
            d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r],
                      fill=col, outline=(255, 255, 255) if halo else None)
        else:
            col = density_colour(dens[i], mxd)
            d.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=col)


def outline_states(d, proj, states_geo, only=None, width=1, colour="#b8c4cf"):
    for name, geom in states_geo.items():
        if only is not None and name not in only:
            continue
        for ring in rings_of(geom):
            arr = np.asarray(ring, float)
            px, py = proj(arr[:, 0], arr[:, 1])
            pts = list(zip(px.tolist(), py.tolist()))
            if len(pts) > 2:
                d.line(pts + [pts[0]], fill=colour, width=width)


def fill_states(d, proj, states_geo, values_by_abbr, mx, only=None):
    """Shade states by a value keyed on the two-letter abbreviation."""
    for name, geom in states_geo.items():
        if only is not None and name not in only:
            continue
        ab = ABBR.get(name, "")
        v = values_by_abbr.get(ab, 0)
        col = density_colour(v, mx) if mx else "#ffffff"
        for ring in rings_of(geom):
            arr = np.asarray(ring, float)
            px, py = proj(arr[:, 0], arr[:, 1])
            pts = list(zip(px.tolist(), py.tolist()))
            if len(pts) > 2:
                d.polygon(pts, fill=col, outline="#ffffff")


def label(d, x, y, text, size=13, bold=False, colour=INK, halo="#ffffff", center=True):
    f = font(size, bold)
    tw = d.textbbox((0, 0), text, f)[2]
    tx = x - tw / 2 if center else x
    if halo:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((tx + dx, y + dy), text, font=f, fill=halo)
    d.text((tx, y), text, font=f, fill=colour)


def density_legend(d, x, y, mx, title="点大小 = 1km 内竞争对手数"):
    d.text((x, y), title, font=font(13, True), fill=INK)
    for k, nb in enumerate([10, 100, 400, 900]):
        if nb > mx * 1.4:
            continue
        r = 1.5 + 3.1 * (nb / mx) ** 0.45
        cy = y + 26 + k * 30
        d.ellipse([x + 8 - r, cy + 8 - r, x + 8 + r, cy + 8 + r], fill="#8fa8bb")
        d.text((x + 30, cy - 1), f"{nb:,} 家", font=font(12), fill=MUTED)
