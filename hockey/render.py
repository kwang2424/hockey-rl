"""Headless renderer: turns a list of state snapshots into a GIF or PNGs.

Deliberately built on Pillow rather than a game engine. Training never touches
this code -- the sim emits state, the renderer replays it. Keeping those two
apart is what lets the sim run at tens of thousands of steps per second while
still being watchable.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from .config import Config, DEFAULT
from . import rink

ICE = (238, 244, 250)
LINE_RED = (196, 40, 48)
LINE_BLUE = (40, 88, 190)
BOARD = (28, 34, 44)
TEAM = [(214, 62, 62), (46, 110, 214)]
PUCK = (18, 18, 20)
CREASE = (150, 200, 240)


class Renderer:
    def __init__(self, cfg: Config = DEFAULT, scale: float = 14.0, pad: float = 1.2):
        self.cfg = cfg
        self.scale = scale
        self.pad = pad
        self.w = int((cfg.rink_length + 2 * pad) * scale)
        self.h = int((cfg.rink_width + 2 * pad) * scale)
        self._bg = self._draw_rink()

    # -- world -> pixel ------------------------------------------------
    def px(self, p):
        p = np.asarray(p, dtype=np.float64)
        x = (p[..., 0] + self.cfg.half_length + self.pad) * self.scale
        y = (self.cfg.half_width + self.pad - p[..., 1]) * self.scale   # y up
        return np.stack([x, y], axis=-1)

    def _circle(self, draw, center, radius_m, **kw):
        c = self.px(center)
        r = radius_m * self.scale
        draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], **kw)

    def _draw_rink(self):
        cfg = self.cfg
        img = Image.new("RGB", (self.w, self.h), (16, 20, 28))
        d = ImageDraw.Draw(img)

        # Ice surface: trace the rounded-rect boundary from the SDF's own
        # definition so the drawing and the collision geometry cannot drift.
        pts = []
        for i in range(240):
            a = 2 * math.pi * i / 240
            v = np.array([math.cos(a), math.sin(a)])
            lo, hi = 0.0, cfg.rink_length
            for _ in range(40):                     # bisect onto the boundary
                mid = 0.5 * (lo + hi)
                if rink.sdf(cfg, v * mid) < 0:
                    lo = mid
                else:
                    hi = mid
            pts.append(tuple(self.px(v * lo)))
        d.polygon(pts, fill=ICE, outline=BOARD)
        for k in range(3):
            d.line(pts + [pts[0]], fill=BOARD, width=3 - k)

        s, hw = self.scale, cfg.half_width
        d.line([tuple(self.px([0, -hw])), tuple(self.px([0, hw]))], fill=LINE_RED, width=max(2, int(0.3 * s)))
        for bx in (-cfg.rink_length / 6.0, cfg.rink_length / 6.0):
            d.line([tuple(self.px([bx, -hw])), tuple(self.px([bx, hw]))], fill=LINE_BLUE, width=max(2, int(0.3 * s)))
        self._circle(d, [0, 0], 4.5, outline=LINE_BLUE, width=2)

        for gx in (cfg.goal_line_x, -cfg.goal_line_x):
            d.line([tuple(self.px([gx, -hw])), tuple(self.px([gx, hw]))], fill=LINE_RED, width=2)
            sgn = 1.0 if gx > 0 else -1.0
            self._circle(d, [gx - sgn * 0.05, 0], 1.8, outline=CREASE, width=2)
            gh, gd = cfg.goal_half_width, cfg.goal_depth
            box = [self.px([gx, gh]), self.px([gx + sgn * gd, gh]),
                   self.px([gx + sgn * gd, -gh]), self.px([gx, -gh])]
            d.polygon([tuple(p) for p in box], fill=(226, 226, 232), outline=LINE_RED)
        for post in rink.post_positions(cfg):
            self._circle(d, post, max(cfg.post_radius, 0.11), fill=LINE_RED)
        return img

    # -- frames --------------------------------------------------------
    def frame(self, snap, env_idx=0, score=None, label=None, trails=None):
        """Draw one frame. ``trails`` is an optional list of (points, colour)
        paths in world coordinates, drawn under the skaters -- useful for
        seeing the shape of a turn rather than guessing at it."""
        cfg = self.cfg
        img = self._bg.copy()
        d = ImageDraw.Draw(img)

        for pts, colour in (trails or []):
            if len(pts) > 1:
                d.line([tuple(self.px(p)) for p in pts], fill=colour, width=2)

        pos = snap["skater_pos"][env_idx]
        th = snap["theta"][env_idx]
        blade = snap["blade_pos"][env_idx]
        owner = int(snap["possessor"][env_idx])

        for a in range(pos.shape[0]):
            col = TEAM[a]
            if owner == a:                       # halo the puck carrier
                self._circle(d, pos[a], cfg.skater_radius + 0.22, outline=(255, 214, 80), width=3)
            self._circle(d, pos[a], cfg.skater_radius, fill=col, outline=(20, 20, 24))
            # stick: from the body out to the blade point, plus the blade
            d.line([tuple(self.px(pos[a])), tuple(self.px(blade[a]))],
                   fill=(60, 44, 32), width=max(2, int(0.13 * self.scale)))
            nose = pos[a] + np.array([math.cos(th[a]), math.sin(th[a])]) * cfg.skater_radius * 0.55
            self._circle(d, nose, cfg.skater_radius * 0.3, fill=(250, 250, 252))
            self._circle(d, blade[a], cfg.capture_radius * 0.36, fill=(60, 44, 32))

        self._circle(d, snap["puck_pos"][env_idx], max(cfg.puck_radius * 3.2, 0.16), fill=PUCK)

        if score is not None:
            d.text((10, 8), f"{score[0]} - {score[1]}", fill=(240, 240, 245))
        if label:
            d.text((10, self.h - 16), label, fill=(200, 205, 215))
        return img


def save_gif(frames, path, fps=30, loop=0):
    if not frames:
        raise ValueError("no frames to write")
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=max(1, int(1000 / fps)), loop=loop, optimize=True,
    )
    return path
