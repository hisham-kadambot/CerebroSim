# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "matplotlib", "pillow", "pyyaml", "imageio-ffmpeg", "warp-lang>=1.18"]
# ///
"""Step 6: the demonstration - a scripted sequence like the stroke video, rendered offline and encoded with NVENC.

  brain + arteries -> zoom to the middle cerebral artery -> healthy flow -> inside the artery -> the clot (70 %) ->
  the jet and the pressure drop -> full blockage -> the three cases side by side -> pressure change -> end card.

Every frame comes from visualise.py's Warp ray tracer (the artery) and the stroke project's renderer (the brain).
The particles are simulated in order (their positions depend on the past), so the sequence is rendered offline and
the viewer plays the MP4 (play / seek / chapters) instead of rendering live.

  uv run demo.py --stills          # one still per shot -> reports/step6_strip.png (quick check, ~1 min)
  uv run demo.py                   # out/artery_demo.mp4 (1080p30, h264_nvenc) + out/chapters.json
  uv run demo.py --insert          # Step 7: out/story_insert/*.jpg, the scene played inside the stroke story
  uv run demo.py --serve           # web viewer on http://<this PC>:8002 (same network; port 8002 must be allowed)
"""
import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

import visualise as VZ

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FPS = 30
SIZE = (1920, 1080)
PORT = 8002
GOLD = (255, 214, 90)

# (start s, chapter title, caption) - the captions appear at the bottom; numbers come from the simulation
SHOTS = [
    (0.0, "Title", None),
    (5.0, "The artery in the brain", "The middle cerebral artery: where the story's clot lodges"),
    (13.0, "Healthy flow", "Healthy: blood from the carotid splits into two branches"),
    (27.0, "Inside the artery", "Inside the artery: fastest in the centre, slow at the wall"),
    (39.0, "A 70 % narrowing", "A clot narrows the artery by 70 %"),
    (52.0, "The pressure drop", None),
    (60.0, "Fully blocked", "Fully blocked: no flow beyond the clot"),
    (71.0, "Side by side", "Same heart, same brain - only the clot differs"),
    (83.0, "Pressure change", "Pressure change against the healthy artery"),
    (91.0, "End", None),
]
TOTAL = 95.0
# Step 7: the scene for the stroke story (frames in out/story_insert/, played by ..\stroke_in_3d after Scene 6)
INSERT_SECONDS = 26.0
INSERT_CLOSE = 18.5     # whole artery before, the cut-open close-up after
INSERT_DIR = os.path.join(OUT, "story_insert")


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    return smooth((t - a) / (b - a))


def lerp(a, b, u):
    return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u


# ---------------------------------------------------------------------------------------------------------------
# text layers
# ---------------------------------------------------------------------------------------------------------------


def layer(img, draw_fn, alpha):
    """Draw onto a transparent layer and composite it at alpha (0..1)."""
    if alpha <= 0.003:
        return img
    base = img.convert("RGBA")
    lay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fn(lay)
    if alpha < 1:
        a = np.asarray(lay).copy()
        a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
        lay = Image.fromarray(a)
    return Image.alpha_composite(base, lay).convert("RGB")


def bottom_caption(img, text, alpha, size=40):
    def draw(lay):
        d = ImageDraw.Draw(lay)
        f = VZ.font(size)
        w = d.textlength(text, font=f)
        x, y = img.width / 2, img.height - 92
        d.rounded_rectangle([x - w / 2 - 28, y - size / 2 - 18, x + w / 2 + 28, y + size / 2 + 18], radius=12,
                            fill=(0, 0, 0, 170))
        d.text((x, y), text, fill=(255, 255, 255, 255), font=f, anchor="mm")
    return layer(img, draw, alpha)


def title_block(img, title, sub, alpha, xy=(64, 52)):
    def draw(lay):
        d = ImageDraw.Draw(lay)
        d.text(xy, title, fill=(*GOLD, 255), font=VZ.font(46), anchor="la")
        if sub:
            d.text((xy[0], xy[1] + 62), sub, fill=(215, 215, 215, 255), font=VZ.font(26, False), anchor="la")
    return layer(img, draw, alpha)


def rgba_layer(img, fn, alpha):
    """Like layer(), for helpers that draw on an RGB image (visualise's colour bars, numbers panel)."""
    if alpha <= 0.003:
        return img
    over = img.copy()
    fn(over)
    return Image.fromarray((np.asarray(img) * (1 - alpha) + np.asarray(over) * alpha).astype(np.uint8))


def speed_bar(im):
    VZ.colour_bar(im, (64, im.height - 200), (340, 16), "turbo", 0, VZ.SPEED_MAX, "blood speed (m/s)",
                  [0, 1, 2, 3, 4], 1.25)


def pressure_bar(im, x=460):
    VZ.colour_bar(im, (x, im.height - 200), (340, 16), "RdYlBu_r", *VZ.P_RANGE, "blood pressure (mmHg)",
                  [0, 50, 100, 150], 1.25)


# ---------------------------------------------------------------------------------------------------------------
# the sequence
# ---------------------------------------------------------------------------------------------------------------


class Demo:
    def __init__(self, size=SIZE, ss=2):
        import warp as wp
        wp.init()
        self.wp = wp
        self.dev = wp.get_preferred_device()
        self.W, self.H = size
        self.ss = ss
        t0 = time.perf_counter()
        cases = {"healthy": VZ.Case("healthy", self.dev)}
        for c in VZ.CASES[1:]:
            cases[c] = VZ.Case(c, self.dev)
        self.sc = {c: VZ.Scene(cases[c], cases["healthy"], device=self.dev) for c in VZ.CASES}
        for s in self.sc.values():
            s.warm(beats=1.0)          # flow-weighted particles; all scenes share the same clock from here
        self.st = {c: VZ.stats(c) for c in VZ.CASES}
        g = cases["healthy"].geom
        self.g = g
        # cameras as (position, target, up)
        o = VZ.overview_cam(g, fov=40.0)
        self.cam_over = (o[0], o[0] + o[1] * 50.0, o[3])
        cc = VZ.clot_cam(g)
        self.cam_clot = (cc[0], cc[0] + cc[1] * 17.0, cc[3])
        self.plane = VZ.clot_plane(g, cc)
        self.keep = (*VZ.clot_point(g)[0], VZ.KEEP_R)
        self._ctx = None
        print(f"demo ready in {time.perf_counter() - t0:.1f} s", flush=True)

    # --- clocks
    @staticmethod
    def slowmo(t):
        """Simulated seconds per screen second: 1/10 for the whole artery, 1/20 for the close-ups."""
        return 0.1 if (t < 27.0 or t >= 71.0) else 0.05

    def advance(self, dt_screen, t):
        ms = dt_screen * self.slowmo(t) * 1e3
        for s in self.sc.values():
            s.advance(ms)

    # --- renders
    def cam(self, c, orbit=0.0):
        pos, tgt, up = c
        if orbit:
            a = math.radians(orbit)
            v = pos - tgt
            k = up / np.linalg.norm(up)
            v = v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * (k @ v) * (1 - math.cos(a))
            pos = tgt + v
        return VZ.look_at(pos, tgt, up)

    def cam_mix(self, a, b, u):
        return (lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerp(a[2], b[2], u))

    def artery(self, case, cam, w=None, h=None, **kw):
        w, h = w or self.W, h or self.H
        return Image.fromarray(self.sc[case].render(cam, w, h, ss=self.ss, fov=40.0, **kw))

    def overview(self, case, t, w=None, h=None, orbit=0.0):
        return self.artery(case, self.cam(self.cam_over, orbit), w, h, wall_mode=0, radius=0.11)

    def close(self, case, mode, cam=None, w=None, h=None):
        cam = cam or self.cam(self.cam_clot)
        if mode == "particles":
            return self.artery(case, cam, w, h, glass=1.0, plane=self.plane, plane_mode=1, keep=self.keep)
        if mode == "speed":
            return self.artery(case, cam, w, h, glass=1.0, plane=self.plane, plane_mode=2, section=0,
                               particles=False, keep=self.keep)
        return self.artery(case, cam, w, h, glass=1.0, plane=self.plane, plane_mode=2, section=1, particles=False,
                           keep=self.keep)

    def context(self, u, box):
        """The story's brain (stroke renderer), dollying in towards the segment; box = marker alpha."""
        if self._ctx is None:
            sys.path.insert(0, VZ.STROKE)
            import renderer as SR
            self._SR = SR
            self._ctx = SR.Renderer()
            self._seg = json.load(open(os.path.join(HERE, "geometry", "segment.json")))
        SR, r, seg = self._SR, self._ctx, self._seg
        J = np.array(seg["junction_point"])
        target = lerp(0.5 * (J + r.s.pivot), J, 0.75 * u)
        d = np.array([-1.0, 0.35, -0.55])
        d /= np.linalg.norm(d)
        pos = target + d * float(lerp(1.35, 0.42, u))
        view = SR.View(cam_pos=tuple(pos), cam_fwd=tuple(target - pos), shell=float(lerp(0.70, 0.86, u)),
                       regions=1.0, vessels=0.0, lesion=1.0, fov=45.0, clot=1.0)
        look = dataclasses.replace(SR.LOOKS["unity"], transparency="alpha", ss=self.ss)
        img = Image.fromarray(r.render(view, look, self.W, self.H))
        if box > 0:
            cp, cf, cr, cu, _ = r.camera(view)
            pts = np.vstack([np.array(b["points"]) for b in seg["branches"].values()])
            xy, _ = VZ.project((cp, cf, cr, cu), pts, self.W, self.H, 45.0)
            clot, _ = VZ.project((cp, cf, cr, cu), np.array(seg["clot"]["face_point"]), self.W, self.H, 45.0)
            lo, hi = xy.min(0) - 30, xy.max(0) + 30

            def draw(lay):
                d = ImageDraw.Draw(lay)
                d.rounded_rectangle([*lo, *hi], radius=18, outline=(*GOLD, 255), width=4)
                cx, cy = clot[0]
                d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], outline=(255, 255, 255, 255), width=4)
            img = layer(img, draw, box)
        return img

    def numbers(self, img, case, alpha, x=None, y=None, scale=1.25):
        x = self.W - int(560 * scale) if x is None else x
        y = self.H - int(300 * scale) if y is None else y   # lower right: the artery fills the upper right
        return rgba_layer(img, lambda im: VZ.numbers_panel(im, case, self.st[case], x, y, scale), alpha)

    # --- one frame
    def frame(self, t):
        W, H = self.W, self.H
        black = Image.new("RGB", (W, H), (0, 0, 0))
        st = self.st
        if t < 5.0:                                                     # title
            a = ramp(t, 0.3, 1.5) * (1 - ramp(t, 4.0, 5.0))

            def draw(lay):
                d = ImageDraw.Draw(lay)
                d.text((W / 2, H / 2 - 40), "Blood flow through a brain artery", fill=(*GOLD, 255),
                       font=VZ.font(72), anchor="mm")
                d.text((W / 2, H / 2 + 40), "healthy, 70 % narrowed and fully blocked", fill=(230, 230, 230, 255),
                       font=VZ.font(38, False), anchor="mm")
                d.text((W / 2, H / 2 + 100), "simulated and rendered on an NVIDIA GPU (Warp)",
                       fill=(170, 170, 170, 255), font=VZ.font(26, False), anchor="mm")
            return layer(black, draw, a)
        if t < 15.0:                                                    # brain -> the artery
            img = self.context(ramp(t, 5.0, 13.5), ramp(t, 8.0, 9.0))
            img = bottom_caption(img, SHOTS[1][2], ramp(t, 6.0, 7.0) * (1 - ramp(t, 12.5, 13.0)))
            if t > 13.0:
                u = ramp(t, 13.0, 15.0)
                ov = self.overview("healthy", t)
                img = Image.blend(img, ov, u)
            return img
        if t < 27.0:                                                    # healthy, whole artery
            img = self.overview("healthy", t, orbit=-8.0 * ramp(t, 15.0, 27.0))
            a = ramp(t, 15.5, 16.5)
            img = title_block(img, "Healthy", "particles: blood speed   wall: blood pressure", a)
            img = rgba_layer(img, lambda im: (speed_bar(im), pressure_bar(im)), a)
            img = self.numbers(img, "healthy", ramp(t, 18.0, 19.0))
            img = bottom_caption(img, SHOTS[2][2], ramp(t, 16.0, 17.0) * (1 - ramp(t, 25.5, 26.0)))
            return img
        if t < 39.0:                                                    # move in, cut open
            u = ramp(t, 27.0, 31.0)
            cam = self._mix_cam(u)
            if t < 31.0:
                img = self.artery("healthy", cam, wall_mode=0, radius=float(lerp(0.11, 0.07, u)))
            elif t < 32.0:
                a = self.artery("healthy", cam, wall_mode=0, radius=0.07)
                b = self.close("healthy", "particles", cam)
                img = Image.blend(a, b, ramp(t, 31.0, 32.0))
            else:
                img = self.close("healthy", "particles")
            a = ramp(t, 32.0, 33.0)
            img = title_block(img, "Healthy", "cut open: blood particles by speed", a)
            img = rgba_layer(img, lambda im: (speed_bar(im), pressure_bar(im)), a)
            img = bottom_caption(img, SHOTS[3][2], ramp(t, 32.5, 33.5) * (1 - ramp(t, 38.0, 38.6)))
            return img
        if t < 52.0:                                                    # 70 %: the clot, the jet
            if t < 41.0:
                img = Image.blend(self.close("healthy", "particles"), self.close("narrow70", "particles"),
                                  ramp(t, 39.0, 41.0))
            elif t < 46.0:
                img = self.close("narrow70", "particles")
            elif t < 47.0:
                img = Image.blend(self.close("narrow70", "particles"), self.close("narrow70", "speed"),
                                  ramp(t, 46.0, 47.0))
            else:
                img = self.close("narrow70", "speed")
            sub = "cut open: blood particles by speed" if t < 46.5 else "cut plane: blood speed"
            img = title_block(img, "70 % narrowed", sub, ramp(t, 40.0, 41.0))
            img = rgba_layer(img, lambda im: (speed_bar(im), pressure_bar(im)), 1.0)
            img = bottom_caption(img, SHOTS[4][2], ramp(t, 39.5, 40.5) * (1 - ramp(t, 45.0, 45.5)))
            img = bottom_caption(img, f"Blood squeezes through the gap at up to {st['narrow70']['umax']:.1f} m/s",
                                 ramp(t, 46.0, 47.0) * (1 - ramp(t, 51.4, 52.0)))
            img = self.numbers(img, "narrow70", ramp(t, 42.0, 43.0))
            return img
        if t < 60.0:                                                    # 70 %: pressure
            if t < 53.0:
                img = Image.blend(self.close("narrow70", "speed"), self.close("narrow70", "pressure"),
                                  ramp(t, 52.0, 53.0))
            else:
                img = self.close("narrow70", "pressure")
            img = title_block(img, "70 % narrowed", "cut plane: blood pressure", 1.0)
            img = rgba_layer(img, lambda im: (speed_bar(im), pressure_bar(im)), 1.0)
            img = self.numbers(img, "narrow70", 1.0)
            pa = st["narrow70"]["p_probe"].mean(0)
            names = st["narrow70"]["names"]
            img = bottom_caption(img, f"Pressure falls by {st['narrow70']['drop']:.0f} mmHg across the clot: "
                                      f"{pa[names.index('MCA before clot')]:.0f} before it, "
                                      f"{pa[names.index('MCA after clot')]:.0f} after (mean)",
                                 ramp(t, 53.0, 54.0) * (1 - ramp(t, 59.4, 60.0)), size=36)
            return img
        if t < 71.0:                                                    # fully blocked
            if t < 62.0:
                img = Image.blend(self.close("narrow70", "pressure"), self.close("occluded", "particles"),
                                  ramp(t, 60.0, 62.0))
            elif t < 66.0:
                img = self.close("occluded", "particles")
            elif t < 67.0:
                img = Image.blend(self.close("occluded", "particles"), self.close("occluded", "pressure"),
                                  ramp(t, 66.0, 67.0))
            else:
                img = self.close("occluded", "pressure")
            sub = "cut open: blood particles by speed" if t < 66.5 else "cut plane: blood pressure"
            img = title_block(img, "Fully blocked", sub, ramp(t, 61.0, 62.0))
            img = rgba_layer(img, lambda im: (speed_bar(im), pressure_bar(im)), 1.0)
            img = self.numbers(img, "occluded", ramp(t, 62.0, 63.0))
            img = bottom_caption(img, SHOTS[6][2] + " - the blood stands still",
                                 ramp(t, 61.0, 62.0) * (1 - ramp(t, 65.6, 66.2)))
            img = bottom_caption(img, "Pressure beyond the clot collapses (no side supply in this model)",
                                 ramp(t, 66.5, 67.5) * (1 - ramp(t, 70.4, 71.0)), size=36)
            return img
        if t < 83.0:                                                    # side by side
            tw, th = W // 3, int(W // 3 * 9 / 16)
            canvas = Image.new("RGB", (W, H), (0, 0, 0))
            y0 = 170
            for i, c in enumerate(VZ.CASES):
                canvas.paste(self.overview(c, t, tw, th), (tw * i, y0))
            canvas = rgba_layer(canvas, lambda im: [VZ.numbers_panel(im, c, st[c], tw * i + 40, y0 + th + 30, 1.15)
                                                    for i, c in enumerate(VZ.CASES)], 1.0)
            canvas = title_block(canvas, "Side by side", "particles: blood speed   wall: blood pressure", 1.0)
            canvas = rgba_layer(canvas, lambda im: (speed_bar(im), pressure_bar(im)), 1.0)
            img = Image.blend(Image.new("RGB", (W, H)), canvas, ramp(t, 71.0, 72.0))
            img = bottom_caption(img, SHOTS[7][2], ramp(t, 72.5, 73.5) * (1 - ramp(t, 82.4, 83.0)))
            return img
        if t < 91.0:                                                    # pressure change
            tw, th = W // 2, H // 2
            canvas = Image.new("RGB", (W, H), (0, 0, 0))
            for i, c in enumerate(VZ.CASES[1:]):
                im = self.artery(c, self.cam(self.cam_over), tw, th, wall_mode=1, glass=1.0, particles=False)
                VZ.caption(im, f"{VZ.LABEL[c]} minus healthy", (30, 20), 34, GOLD)
                canvas.paste(im, (tw * i, 200))
            VZ.colour_bar(canvas, (W // 2 - 400, 200 + th + 90), (800, 18), "RdBu_r", -VZ.DP_RANGE, VZ.DP_RANGE,
                          "pressure change vs healthy (mmHg): blue = lower, red = higher",
                          [-90, -60, -30, 0, 30, 60, 90], 1.4)
            canvas = title_block(canvas, "Pressure change", "wall pressure, peak of the heartbeat", 1.0)
            img = Image.blend(Image.new("RGB", (W, H)), canvas, ramp(t, 83.0, 84.0))
            img = bottom_caption(img, "Beyond the clot the brain loses pressure; in front of it, pressure builds",
                                 ramp(t, 84.5, 85.5) * (1 - ramp(t, 90.4, 91.0)), size=36)
            return img
        a = ramp(t, 91.0, 92.0) * (1 - ramp(t, 94.0, 95.0))            # end card

        def draw(lay):
            d = ImageDraw.Draw(lay)
            rows = [("", "Healthy", "70 % narrowed", "Fully blocked"),
                    ("Middle cerebral artery flow (mL/min)",
                     *[f"{max(st[c]['Q_mca'], 0.0) + 0.0:.0f}" for c in VZ.CASES]),
                    ("Pressure drop across the clot (mmHg)", *[f"{st[c]['drop']:.0f}" for c in VZ.CASES]),
                    ("Top speed (m/s)", *[f"{st[c]['umax']:.1f}" for c in VZ.CASES])]
            xs = (W / 2 - 560, W / 2 + 60, W / 2 + 330, W / 2 + 600)
            for r, row in enumerate(rows):
                y = H / 2 - 120 + r * 70
                for k, cell in enumerate(row):
                    col = (*GOLD, 255) if r == 0 else (235, 235, 235, 255)
                    d.text((xs[k], y), cell, fill=col, font=VZ.font(34, r == 0 or k > 0),
                           anchor="lm" if k == 0 else "mm")
            d.text((W / 2, H / 2 + 220), "One artery from the story, simulated over full heartbeats (lattice "
                   "Boltzmann, NVIDIA Warp)", fill=(170, 170, 170, 255), font=VZ.font(24, False), anchor="mm")
        return layer(black, draw, a)

    # --- Step 7: the scene inserted into the stroke story after Scene 6 (the story's clot is a full blockage)
    @staticmethod
    def insert_slowmo(t):
        return 0.1 if t < INSERT_CLOSE else 0.05

    def insert_frame(self, t):
        st = self.st
        bars = lambda im: (speed_bar(im), pressure_bar(im))
        if t < 7.0:                                                     # into the brain, the artery boxed
            img = self.context(ramp(t, 0.0, 6.0), ramp(t, 1.5, 2.5))
            if t >= 6.0:
                img = Image.blend(img, self.overview("healthy", t), ramp(t, 6.0, 7.0))
            return bottom_caption(img, "Inside the blocked artery: a blood-flow simulation",
                                  ramp(t, 1.0, 2.0) * (1 - ramp(t, 5.4, 6.0)))
        if t < 13.0:                                                    # healthy
            img = self.overview("healthy", t)
            if t >= 12.0:
                img = Image.blend(img, self.overview("occluded", t), ramp(t, 12.0, 13.0))
            img = title_block(img, "Before the clot", "particles: blood speed   wall: blood pressure",
                              ramp(t, 7.0, 8.0) * (1 - ramp(t, 12.0, 12.5)))
            img = rgba_layer(img, bars, ramp(t, 7.0, 8.0))
            img = self.numbers(img, "healthy", ramp(t, 8.0, 9.0) * (1 - ramp(t, 12.0, 12.5)))
            return bottom_caption(img, "Before the clot: blood splits into both branches",
                                  ramp(t, 7.5, 8.5) * (1 - ramp(t, 11.6, 12.2)))
        if t < INSERT_CLOSE + 1.0:                                      # blocked, whole artery
            img = self.overview("occluded", t)
            if t >= INSERT_CLOSE:
                img = Image.blend(img, self.close("occluded", "pressure"), ramp(t, INSERT_CLOSE, INSERT_CLOSE + 1))
            img = title_block(img, "With the clot", "particles: blood speed   wall: blood pressure",
                              ramp(t, 12.5, 13.5) * (1 - ramp(t, INSERT_CLOSE, INSERT_CLOSE + 0.5)))
            img = rgba_layer(img, bars, 1.0)
            img = self.numbers(img, "occluded", ramp(t, 13.5, 14.5) * (1 - ramp(t, INSERT_CLOSE, INSERT_CLOSE + 0.5)))
            return bottom_caption(img, "With the clot: no flow beyond it - the blood goes to the other branch",
                                  ramp(t, 13.5, 14.5) * (1 - ramp(t, INSERT_CLOSE - 0.6, INSERT_CLOSE)), size=36)
        img = self.close("occluded", "pressure")                        # blocked, cut open: pressure
        img = title_block(img, "With the clot", "cut plane: blood pressure", ramp(t, INSERT_CLOSE + 0.5, INSERT_CLOSE + 1.5))
        img = rgba_layer(img, bars, 1.0)
        img = self.numbers(img, "occluded", ramp(t, INSERT_CLOSE + 1.0, INSERT_CLOSE + 2.0))
        names = st["healthy"]["names"]
        before = st["healthy"]["p_probe"].mean(0)[names.index("MCA after clot")]
        after = st["occluded"]["p_probe"].mean(0)[names.index("MCA after clot")]
        return bottom_caption(img, f"Beyond the clot, blood pressure falls from {before:.0f} to {max(after, 0.0):.0f} mmHg",
                              ramp(t, INSERT_CLOSE + 1.0, INSERT_CLOSE + 2.0), size=36)

    def _mix_cam(self, u):
        o = self.cam(self.cam_over, -8.0)
        start = (o[0], o[0] + o[1] * np.linalg.norm(self.cam_over[1] - self.cam_over[0]), self.cam_over[2])
        return self.cam(self.cam_mix(start, self.cam_clot, u))


# ---------------------------------------------------------------------------------------------------------------
# export, stills, viewer
# ---------------------------------------------------------------------------------------------------------------


def run_sequence(demo, times, sink, render=None):
    """Render the frames at `times` (ascending; only those in `render` if given), advancing the particles."""
    prev = 0.0
    for t in times:
        demo.advance(t - prev, t)
        prev = t
        if render is None or t in render:
            sink(t, np.asarray(demo.frame(t)))


def export(a):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "artery_demo.mp4")
    n = int(round(TOTAL * FPS))
    demo = Demo(SIZE, a.ss)
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}",
           "-r", str(FPS), "-i", "-", "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
           "-cq", "19", "-b:v", "0", "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.perf_counter()

    def sink(t, img):
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
        i = int(round(t * FPS))
        if i % (FPS * 5) == 0:
            el = time.perf_counter() - t0
            print(f"  t {t:5.1f} s  frame {i + 1}/{n}  {(i + 1) / el:4.1f} frames/s  left "
                  f"{(n - i - 1) / max((i + 1) / el, 1e-6) / 60:4.1f} min", flush=True)
    run_sequence(demo, [i / FPS for i in range(n)], sink)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    chapters = [{"t": s, "title": name} for s, name, _ in SHOTS if name not in ("End",)]
    json.dump({"total": TOTAL, "chapters": chapters}, open(os.path.join(OUT, "chapters.json"), "w"), indent=1)
    print(f"done: {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {(time.perf_counter() - t0) / 60:.1f} min")


def stills(a):
    """One frame per shot (the particles still advance through the whole sequence, at 6 frames per second)."""
    demo = Demo(SIZE, 1)
    want = [2.5, 10.0, 22.0, 35.0, 44.0, 50.0, 56.0, 64.0, 69.0, 78.0, 88.0, 93.0]
    got = {}
    times = sorted(set([i / 6 for i in range(int(TOTAL * 6))] + want))
    run_sequence(demo, times, got.__setitem__, set(want))
    cols = 3
    sheet = Image.new("RGB", (cols * 960, math.ceil(len(want) / cols) * 540), (20, 20, 20))
    for k, t in enumerate(want):
        im = Image.fromarray(got[t]).resize((960, 540), Image.LANCZOS)
        VZ.caption(im, f"{t:.0f} s", (900, 20), 22, (120, 255, 120), anchor="ra")
        sheet.paste(im, ((k % cols) * 960, (k // cols) * 540))
    sheet.save(os.path.join(VZ.REPORTS, "step6_strip.png"))
    print("reports/step6_strip.png")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Artery flow demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:0;background:#000;color:#ddd;font-family:Arial,sans-serif}
main{max-width:1280px;margin:0 auto;padding:16px}
h1{font-size:20px;color:#ffd65a;margin:4px 0 12px}
video{width:100%;background:#000;display:block}
#ch{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
#ch button{background:#222;color:#ddd;border:1px solid #444;border-radius:6px;padding:6px 10px;cursor:pointer}
#ch button.on{background:#ffd65a;color:#000;border-color:#ffd65a}
</style></head><body><main>
<h1>Blood flow through a brain artery: healthy, 70 % narrowed, fully blocked</h1>
<video id="v" src="/video.mp4" controls preload="auto"></video>
<div id="ch"></div>
</main><script>
const v=document.getElementById('v'), ch=document.getElementById('ch');
fetch('/chapters.json').then(r=>r.json()).then(d=>{
  const bs=d.chapters.map((c,i)=>{const b=document.createElement('button');
    b.textContent=(i+1)+'. '+c.title; b.onclick=()=>{v.currentTime=c.t; v.play();}; ch.appendChild(b); return b;});
  v.addEventListener('timeupdate',()=>{let k=0; d.chapters.forEach((c,i)=>{if(v.currentTime>=c.t-0.05)k=i;});
    bs.forEach((b,i)=>b.classList.toggle('on',i===k));});
});
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault(); v.paused?v.play():v.pause();}});
</script></body></html>"""


def serve(a):
    import http.server
    import socket
    video = os.path.join(OUT, "artery_demo.mp4")
    if not os.path.exists(video):
        raise SystemExit("out/artery_demo.mp4 is missing: run `uv run demo.py` first")

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, ctype, body, extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            if self.path == "/chapters.json":
                return self._send(200, "application/json", open(os.path.join(OUT, "chapters.json"), "rb").read())
            if self.path.startswith("/video.mp4"):
                size = os.path.getsize(video)
                rng = self.headers.get("Range")
                lo, hi = 0, size - 1
                if rng and rng.startswith("bytes="):
                    a_, _, b_ = rng[6:].partition("-")
                    lo = int(a_) if a_ else max(0, size - int(b_))
                    hi = int(b_) if (a_ and b_) else min(size - 1, lo + (4 << 20) - 1)  # 4 MB per reply
                with open(video, "rb") as f:
                    f.seek(lo)
                    body = f.read(hi - lo + 1)
                extra = [("Accept-Ranges", "bytes")]
                if rng:
                    extra.append(("Content-Range", f"bytes {lo}-{hi}/{size}"))
                try:
                    return self._send(206 if rng else 200, "video/mp4", body, extra)
                except (BrokenPipeError, ConnectionResetError):
                    return
            self._send(404, "text/plain", b"not found")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        ip = "127.0.0.1"
    print(f"artery demo viewer: http://127.0.0.1:{PORT}  (other computers: http://{ip}:{PORT})", flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


def insert_export(a):
    """Step 7: the stroke story's inserted scene as numbered JPEG frames (+ insert.json). The particles must be
    simulated in order, so the stroke viewer plays these frames instead of rendering the scene live."""
    os.makedirs(INSERT_DIR, exist_ok=True)
    for f in os.listdir(INSERT_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(INSERT_DIR, f))
    demo = Demo(SIZE, a.ss)
    n = int(round(INSERT_SECONDS * FPS))
    t0, prev = time.perf_counter(), 0.0
    stride = max(1, a.insert_stride)
    for i in range(n):
        t = i / FPS
        ms = (t - prev) * demo.insert_slowmo(t) * 1e3
        for s in demo.sc.values():
            s.advance(ms)
        prev = t
        if i % stride:
            continue
        demo.insert_frame(t).save(os.path.join(INSERT_DIR, f"{i:04d}.jpg"), quality=92)
        if i % (FPS * 5) == 0:
            el = time.perf_counter() - t0
            print(f"  t {t:5.1f} s  frame {i + 1}/{n}  {(i + 1) / el:4.1f} frames/s", flush=True)
    json.dump({"fps": FPS, "seconds": INSERT_SECONDS, "frames": n, "stride": stride, "size": list(SIZE),
               "title": "Inside the artery", "after_scene": 6}, open(os.path.join(INSERT_DIR, "insert.json"), "w"),
              indent=1)
    print(f"done: {INSERT_DIR} ({n // stride} frames) in {(time.perf_counter() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--insert", action="store_true", help="Step 7: frames for the stroke story's inserted scene")
    ap.add_argument("--insert-stride", type=int, default=1, help="keep every n-th insert frame (quick checks)")
    a = ap.parse_args()
    if a.insert:
        insert_export(a)
    elif a.serve:
        serve(a)
    elif a.stills:
        stills(a)
    else:
        export(a)


if __name__ == "__main__":
    main()
