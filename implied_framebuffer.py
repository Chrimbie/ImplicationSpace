"""implied_framebuffer.py - graphics memory inside implication space.

Christopher's idea (2026-10-22): "create some spare bits out of
a partition to use for graphics in a CPU." A framebuffer whose
address space lives in an EXP-VRAM BlockStore:

  * the canvas is LOGICAL: W x H RGBA pixels at implied
    addresses; untouched pixels are DERIVED (the implication
    pattern), never stored;
  * the CPU rasterizes into a row cache; each drawn row is
    stored exactly once (block size == one canvas row);
  * PRESENT = read the visible window back out of the store
    (deriving whatever was never drawn) and show it.

Checks: every presented/verified window is compared pixel-exact
against an INDEPENDENT analytic render of the same shapes
(vectorized masks, no store involved), before and after a
close/reopen of the store.
"""
import json
import math
import struct
import sys
import time
import zlib

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_blockstore import BlockStore, derive_default

# ---------------------------------------------------------------- shapes
# A shape is a dict; two independent renderers consume this list:
# the CPU rasterizer (row spans into the store) and the analytic
# verifier (vectorized masks over a window).
SHAPES = [
    # diagonal band (center cluster)
    {"kind": "band", "x": 28400, "y": 29200, "rows": 2600,
     "w": 950, "slope": 0.5, "rgba": (64, 156, 255, 255)},
    # big rect (center cluster)
    {"kind": "rect", "x": 29600, "y": 29800, "w": 2600, "h": 1500,
     "rgba": (255, 176, 32, 255)},
    # overlapping circles (center cluster)
    {"kind": "circle", "cx": 31200, "cy": 30900, "r": 1400,
     "rgba": (240, 84, 84, 255)},
    {"kind": "circle", "cx": 32600, "cy": 30200, "r": 900,
     "rgba": (96, 220, 140, 255)},
    # checkerboard patch (center cluster)
    {"kind": "checker", "x": 29100, "y": 31800, "w": 1024, "h": 1024,
     "cell": 64, "rgba": (238, 238, 238, 255),
     "rgba2": (24, 24, 32, 255)},
    # far-corner cluster: proves addressing at the deep end
    {"kind": "circle", "cx": 60500, "cy": 61200, "r": 1200,
     "rgba": (176, 96, 255, 255)},
    {"kind": "rect", "x": 58800, "y": 59800, "w": 1500, "h": 900,
     "rgba": (64, 220, 220, 255)},
    # near-origin mark
    {"kind": "rect", "x": 100, "y": 100, "w": 700, "h": 420,
     "rgba": (255, 255, 255, 255)},
]
# deterministic scatter of small circles in the center cluster
_rng = np.random.default_rng(42)
for _ in range(26):
    SHAPES.append({
        "kind": "circle",
        "cx": int(_rng.integers(28800, 33800)),
        "cy": int(_rng.integers(29400, 33000)),
        "r": int(_rng.integers(40, 130)),
        "rgba": (int(_rng.integers(120, 256)),
                 int(_rng.integers(120, 256)),
                 int(_rng.integers(120, 256)), 255)})


# ---------------------------------------------------------------- fb
class ImpliedFramebuffer:
    def __init__(self, store, width, height):
        self.store = store
        self.w = width
        self.h = height
        self.row_bytes = width * 4
        self._rows = {}   # y -> bytearray (cached, mutable)
        self._dirty = set()

    def _row(self, y):
        r = self._rows.get(y)
        if r is None:
            r = bytearray(self.store.read(y * self.row_bytes,
                                          self.row_bytes))
            self._rows[y] = r
        return r

    def span(self, y, x0, x1, rgba):
        """Fill pixels [x0, x1) on row y with rgba."""
        if y < 0 or y >= self.h:
            return
        x0 = max(0, x0)
        x1 = min(self.w, x1)
        if x1 <= x0:
            return
        row = self._row(y)
        row[x0 * 4:x1 * 4] = bytes(rgba) * (x1 - x0)
        self._dirty.add(y)

    def flush(self):
        for y in sorted(self._dirty):
            self.store.write(y * self.row_bytes,
                             bytes(self._rows[y]))
        n = len(self._dirty)
        self._dirty.clear()
        self._rows.clear()
        return n

    def read_window(self, x, y, w, h):
        out = np.empty((h, w, 4), dtype=np.uint8)
        for j in range(h):
            off = (y + j) * self.row_bytes + x * 4
            out[j] = np.frombuffer(self.store.read(off, w * 4),
                                   dtype=np.uint8).reshape(w, 4)
        return out


def rasterize(fb, shapes):
    """Renderer 1: per-shape row spans into the framebuffer."""
    for s in shapes:
        k = s["kind"]
        if k == "rect":
            for y in range(s["y"], s["y"] + s["h"]):
                fb.span(y, s["x"], s["x"] + s["w"], s["rgba"])
        elif k == "circle":
            cx, cy, r = s["cx"], s["cy"], s["r"]
            for dy in range(-r, r + 1):
                half = int(math.sqrt(r * r - dy * dy))
                fb.span(cy + dy, cx - half, cx + half + 1,
                        s["rgba"])
        elif k == "band":
            for i in range(s["rows"]):
                x0 = s["x"] + int(i * s["slope"])
                fb.span(s["y"] + i, x0, x0 + s["w"], s["rgba"])
        elif k == "checker":
            for y in range(s["y"], s["y"] + s["h"]):
                cy = (y - s["y"]) // s["cell"]
                for cx0 in range(0, s["w"], s["cell"]):
                    cx = cx0 // s["cell"]
                    col = s["rgba"] if (cx + cy) % 2 == 0 \
                        else s["rgba2"]
                    fb.span(y, s["x"] + cx0,
                            s["x"] + cx0 + s["cell"], col)
        else:
            raise ValueError(k)


def analytic_window(shapes, fb_size, x, y, w, h):
    """Renderer 2 (the judge): vectorized masks over the window,
    background = derived bytes, shapes painted in order."""
    W, _H = fb_size
    row_bytes = W * 4
    out = np.empty((h, w, 4), dtype=np.uint8)
    for j in range(h):
        off = (y + j) * row_bytes + x * 4
        out[j] = derive_default(off, w * 4).reshape(w, 4)
    xs = np.arange(x, x + w)[None, :]
    ys = np.arange(y, y + h)[:, None]
    for s in shapes:
        k = s["kind"]
        if k == "rect":
            m = (xs >= s["x"]) & (xs < s["x"] + s["w"]) & \
                (ys >= s["y"]) & (ys < s["y"] + s["h"])
            out[m] = s["rgba"]
        elif k == "circle":
            m = (xs - s["cx"]) ** 2 + (ys - s["cy"]) ** 2 \
                <= s["r"] ** 2  # == rasterizer's floor(sqrt) spans
            out[m] = s["rgba"]
        elif k == "band":
            rel = ys - s["y"]
            x0 = s["x"] + (rel * s["slope"]).astype(np.int64)
            m = (rel >= 0) & (rel < s["rows"]) & \
                (xs >= x0) & (xs < x0 + s["w"])
            out[m] = s["rgba"]
        elif k == "checker":
            inb = (xs >= s["x"]) & (xs < s["x"] + s["w"]) & \
                  (ys >= s["y"]) & (ys < s["y"] + s["h"])
            cx = (xs - s["x"]) // s["cell"]
            cy = (ys - s["y"]) // s["cell"]
            m1 = inb & ((cx + cy) % 2 == 0)
            m2 = inb & ((cx + cy) % 2 == 1)
            out[m1] = s["rgba"]
            out[m2] = s["rgba2"]
    return out


def write_png(path, rgba):
    """Minimal PNG writer (RGB, alpha dropped for display)."""
    h, w, _ = rgba.shape
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    raw = b"".join(b"\x00" + rgb[j].tobytes() for j in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data)
                               & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2,
                                      0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
    return len(png)


def phone_shapes():
    """Compact composition for a phone: same huge canvas class,
    small drawn footprint (~0.5 GB stored)."""
    shapes = [
        {"kind": "band", "x": 2600, "y": 2700, "rows": 1400,
         "w": 500, "slope": 0.5, "rgba": (64, 156, 255, 255)},
        {"kind": "rect", "x": 3400, "y": 3300, "w": 1800, "h": 1100,
         "rgba": (255, 176, 32, 255)},
        {"kind": "circle", "cx": 4600, "cy": 4200, "r": 950,
         "rgba": (240, 84, 84, 255)},
        {"kind": "circle", "cx": 5600, "cy": 3800, "r": 600,
         "rgba": (96, 220, 140, 255)},
        {"kind": "checker", "x": 3000, "y": 5300, "w": 768,
         "h": 768, "cell": 48, "rgba": (238, 238, 238, 255),
         "rgba2": (24, 24, 32, 255)},
        {"kind": "circle", "cx": 30200, "cy": 30400, "r": 700,
         "rgba": (176, 96, 255, 255)},
        {"kind": "rect", "x": 29400, "y": 29500, "w": 900, "h": 500,
         "rgba": (64, 220, 220, 255)},
        {"kind": "rect", "x": 60, "y": 60, "w": 400, "h": 240,
         "rgba": (255, 255, 255, 255)},
    ]
    rng = np.random.default_rng(42)
    for _ in range(14):
        shapes.append({
            "kind": "circle",
            "cx": int(rng.integers(2800, 6400)),
            "cy": int(rng.integers(2900, 5900)),
            "r": int(rng.integers(30, 100)),
            "rgba": (int(rng.integers(120, 256)),
                     int(rng.integers(120, 256)),
                     int(rng.integers(120, 256)), 255)})
    return shapes


def main():
    small = "--small" in sys.argv
    phone = "--phone" in sys.argv
    W = H = 4096 if small else (32768 if phone else 65536)
    path = ("framebuffer_small.evb2" if small else
            "framebuffer_phone.evb2" if phone else
            "framebuffer.evb2")
    logical = W * H * 4
    shapes = SHAPES
    if phone:
        shapes = phone_shapes()
    if small:
        shapes = [dict(s) for s in SHAPES[:5]]
        shapes[0].update(x=300, y=300, rows=400)
        shapes[1].update(x=700, y=600, w=800, h=500)
        shapes[2].update(cx=1100, cy=1000, r=350)
        shapes[3].update(cx=1500, cy=800, r=220)
        shapes[4].update(x=500, y=1500, w=512, h=512, cell=64)
    print(f"canvas {W}x{H} RGBA = {logical:,} logical bytes "
          f"({logical / 2**30:.1f} GiB) in {path}")
    t0 = time.time()
    store = BlockStore.create(path, logical, block_size=W * 4)
    fb = ImpliedFramebuffer(store, W, H)
    print(f"empty store on disk: {store.stored_bytes():,} B")

    rows_written = 0
    for s in shapes:  # per-shape flush bounds RAM (phone-safe)
        rasterize(fb, [s])
        rows_written += fb.flush()
    t_draw = time.time() - t0
    print(f"CPU draw: {rows_written:,} rows stored in "
          f"{t_draw:.1f}s | store now {store.stored_bytes():,} B "
          f"({store.blocks_stored():,} blocks) for "
          f"{logical:,} logical B")

    # present + verify three windows
    wins = [(100, 100, 700, 420)] if small else [
        (2900, 3100, 1280, 720),      # phone composition
        (29300, 29700, 1280, 720),    # phone far cluster
        (20000, 20000, 640, 480)] if phone else [
        (29000, 29500, 1920, 1080),   # center composition
        (59600, 60200, 1600, 1200),   # far cluster
        (40000, 40000, 800, 600)]     # untouched: derived only
    present = None
    for (x, y, w, h) in wins:
        t1 = time.time()
        got = fb.read_window(x, y, w, h)
        dt = time.time() - t1
        want = analytic_window(shapes, (W, H), x, y, w, h)
        ok = bool(np.array_equal(got, want))
        print(f"window @({x},{y}) {w}x{h}: present {dt:.2f}s | "
              f"pixel-exact vs analytic render: "
              f"{'PASS' if ok else 'FAIL'}")
        assert ok, "window mismatch"
        if present is None:
            present = got
    png_path = ("framebuffer_phone_present.png" if phone
                else "framebuffer_present.png")
    n = write_png(png_path, present)
    print(f"presented frame written: {png_path} ({n:,} B)")

    store.close()
    store2 = BlockStore.open(path)
    fb2 = ImpliedFramebuffer(store2, W, H)
    x, y, w, h = wins[0]
    got = fb2.read_window(x, y, w, h)
    want = analytic_window(shapes, (W, H), x, y, w, h)
    ok = bool(np.array_equal(got, want))
    final_stored = store2.stored_bytes()
    print(f"after reopen: window pixel-exact: "
          f"{'PASS' if ok else 'FAIL'} | stored "
          f"{final_stored:,} B vs logical "
          f"{logical:,} B "
          f"({logical / final_stored:.1f}x)")
    assert ok
    store2.close()
    result = {"canvas": [W, H], "logical_bytes": logical,
              "stored_bytes": final_stored,
              "draw_seconds": round(t_draw, 2),
              "verified": True}
    print(json.dumps(result))
    print("IMPLIED FRAMEBUFFER: ALL PASS")


if __name__ == "__main__":
    main()
