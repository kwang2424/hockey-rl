#!/usr/bin/env python3
"""Render training curves from progress.csv to a PNG (no matplotlib needed).

    python plot_progress.py runs/v0/progress.csv
"""

import csv
import sys

from PIL import Image, ImageDraw

W, H, PAD = 900, 220, 46
SERIES = [
    ("vs_chase_goal_diff_per_min", (214, 62, 62), "goal diff/min vs ChaseBot"),
    ("vs_random_goal_diff_per_min", (46, 110, 214), "goal diff/min vs Random"),
    ("entropy", (120, 120, 130), "policy entropy"),
]


def _panel(rows, key, colour, title):
    pts = [(float(r["step"]), float(r[key])) for r in rows if r.get(key) not in (None, "")]
    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)
    d.text((PAD, 6), title, fill=(40, 40, 48))
    if len(pts) < 2:
        d.text((PAD, H // 2), "not enough data", fill=(150, 150, 150))
        return img
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    sx = lambda v: PAD + (v - x0) / (x1 - x0) * (W - 2 * PAD)
    sy = lambda v: H - PAD + (v - y0) / (y1 - y0) * (2 * PAD - H)
    if y0 <= 0 <= y1:
        d.line([(PAD, sy(0)), (W - PAD, sy(0))], fill=(200, 200, 208))
    d.line([(PAD, sy(y0)), (PAD, sy(y1))], fill=(200, 200, 208))
    d.line([(sx(x), sy(y)) for x, y in pts], fill=colour, width=2)
    d.text((4, sy(y1) - 6), f"{y1:.2f}", fill=(90, 90, 100))
    d.text((4, sy(y0) - 6), f"{y0:.2f}", fill=(90, 90, 100))
    d.text((W - PAD - 60, H - 18), f"{x1/1e6:.1f}M steps", fill=(90, 90, 100))
    return img


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/v0/progress.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".csv", ".png")
    rows = list(csv.DictReader(open(path)))
    panels = [_panel(rows, k, c, t) for k, c, t in SERIES]
    sheet = Image.new("RGB", (W, H * len(panels)), (250, 250, 252))
    for i, p in enumerate(panels):
        sheet.paste(p, (0, i * H))
    sheet.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
