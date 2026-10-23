"""frames_store.py - store a movie in implication space.

120 frames of an analytic animation (deterministic in t) at
2560x1440 RGBA, written frame-by-frame into an EVB2 BlockStore
on this machine, deep in a 2^41 envelope. A JSON manifest keeps
each frame's SHA-256 from ingest; playback reads every frame
back out of the store and re-hashes it - the containment test,
per frame. Finally a proof sheet (montage of 8 frames) is
assembled from the READ-BACK bytes only and saved as PNG.

The store (frames_qhd.evb2) persists - this is a library,
not a benchmark.
"""
import hashlib
import json
import os
import struct
import sys
import time
import zlib

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_blockstore import BlockStore

W, H = 2560, 1440
FRAMES = 120
STRIDE = 16 << 20
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "frames_qhd.evb2")
MANIFEST = os.path.join(HERE, "frames_qhd_manifest.json")
SHEET = os.path.join(HERE, "frames_proof_sheet.png")


def render_frame(t):
    """Analytic scene at time t - deterministic, no assets."""
    img = np.zeros((H, W, 4), dtype=np.uint8)
    sky = int(H * 0.40)
    grad = np.linspace(30, 200, sky, dtype=np.uint8)
    img[:sky, :, 0] = grad[:, None] // 3
    img[:sky, :, 1] = grad[:, None] // 2
    img[:sky, :, 2] = grad[:, None]
    mid_end = int(H * 0.75)
    img[sky:mid_end, :] = (24, 24, 32, 255)
    yy, xx = np.mgrid[sky:mid_end, 0:W]
    ph = 2 * np.pi * t / FRAMES
    balls = [
        (500 + 300 * np.sin(ph), 700 + 60 * np.sin(2 * ph), 160,
         (220, 80, 60)),
        (1300 + 500 * np.sin(ph + 2.1), 660 + 90 * np.cos(ph), 220,
         (240, 200, 90)),
        (2050 + 260 * np.sin(ph + 4.2), 720 + 70 * np.sin(3 * ph),
         140, (90, 200, 140)),
    ]
    band = img[sky:mid_end]
    for cx, cy, r, col in balls:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        band[m] = (*col, 255)
    img[sky:mid_end] = band
    rng = np.random.default_rng(11)
    tex_h = H - mid_end
    noise = rng.normal(128, 40, (tex_h // 8 + 2, W // 8 + 2, 3))
    big = np.kron(noise, np.ones((8, 8, 1)))[:tex_h, :W]
    shift = (t * 6) % W
    img[mid_end:, :, :3] = np.roll(
        np.clip(big, 0, 255).astype(np.uint8), shift, axis=1)
    img[:, :, 3] = 255
    return img


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2,
                                        0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    open(path, "wb").write(png)


def main():
    if os.path.exists(STORE):
        os.remove(STORE)
    store = BlockStore.create(STORE, 1 << 41)
    hashes = []
    t0 = time.perf_counter()
    for t in range(FRAMES):
        data = render_frame(t).tobytes()
        store.write(t * STRIDE, data)
        hashes.append(hashlib.sha256(data).hexdigest())
    store.save()
    dt = time.perf_counter() - t0
    print(f"ingest: {FRAMES} frames in {dt:.1f}s "
          f"({FRAMES / dt:.1f} img/s), store "
          f"{store.stored_bytes():,} B")
    json.dump({"frames": FRAMES, "width": W, "height": H,
               "stride": STRIDE, "sha256": hashes},
              open(MANIFEST, "w"), indent=1)

    t0 = time.perf_counter()
    ok = 0
    sheet_frames = []
    picks = [0, 17, 34, 51, 68, 85, 102, 119]
    for t in range(FRAMES):
        data = store.read(t * STRIDE, W * H * 4)
        if hashlib.sha256(data).hexdigest() == hashes[t]:
            ok += 1
        if t in picks:
            sheet_frames.append(np.frombuffer(
                data, dtype=np.uint8).reshape(H, W, 4))
    dt = time.perf_counter() - t0
    print(f"playback: {FRAMES} frames in {dt:.1f}s "
          f"({FRAMES / dt:.1f} img/s), hash-verified {ok}/{FRAMES}")
    store.close()

    tiles = [f[::4, ::4, :3] for f in sheet_frames]  # 640x360
    th, tw = tiles[0].shape[:2]
    sheet = np.zeros((th * 2, tw * 4, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 4)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    write_png(SHEET, sheet)
    print(f"proof sheet: {SHEET} ({os.path.getsize(SHEET):,} B) "
          f"from read-back frames {picks}")


if __name__ == "__main__":
    main()
