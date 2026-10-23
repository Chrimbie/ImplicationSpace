"""qhd_bench.py - benchmark: a 2560x1440 image, four ways.

Plain file vs EVB2 implication store vs IMPZ1 vs zip:
write/compress time, read/decompress time, size on disk.
Same synthetic image as qhd_test.py (seed 11) for continuity.
Median of 3 runs for the timed sections.
"""
import hashlib
import io
import os
import sys
import time
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/home/hatch/workspace/implication_space")
from exp_blockstore import BlockStore
from implication_zip import compress, decompress

W, H = 2560, 1440


def build_image():
    rng = np.random.default_rng(11)
    img = np.zeros((H, W, 4), dtype=np.uint8)
    sky = int(H * 0.40)
    grad = np.linspace(30, 200, sky, dtype=np.uint8)
    img[:sky, :, 0] = grad[:, None] // 3
    img[:sky, :, 1] = grad[:, None] // 2
    img[:sky, :, 2] = grad[:, None]
    mid_end = int(H * 0.75)
    img[sky:mid_end, :] = (24, 24, 32, 255)
    yy, xx = np.mgrid[sky:mid_end, 0:W]
    for cx, cy, r, col in [(500, 700, 160, (220, 80, 60)),
                           (1300, 660, 220, (240, 200, 90)),
                           (2050, 720, 140, (90, 200, 140))]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[sky:mid_end][m] = (*col, 255)
    tex_h = H - mid_end
    noise = rng.normal(128, 40, (tex_h // 8 + 2, W // 8 + 2, 3))
    big = np.kron(noise, np.ones((8, 8, 1)))[:tex_h, :W]
    img[mid_end:, :, :3] = np.clip(big, 0, 255).astype(np.uint8)
    img[H - 40:, :, :3] = rng.integers(0, 256, (40, W, 3),
                                       dtype=np.uint8)
    img[:, :, 3] = 255
    return img.tobytes()


def med(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[n // 2], r


def mbps(nbytes, secs):
    return nbytes / secs / 1e6


def main():
    data = build_image()
    n = len(data)
    print(f"image {W}x{H} raw {n:,} B  "
          f"sha {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"{'method':14} {'size':>12} {'write':>8} {'MB/s':>8} "
          f"{'read':>8} {'MB/s':>8}  check")
    # --- plain file ---------------------------------------
    def plain_w():
        with open("/tmp/qhd.raw", "wb") as f:
            f.write(data)
    t, _ = med(plain_w)
    sz = os.path.getsize("/tmp/qhd.raw")
    tw = t
    def plain_r():
        with open("/tmp/qhd.raw", "rb") as f:
            return f.read()
    t, out = med(plain_r)
    print(f"{'plain file':14} {sz:>12,} {tw:>7.3f}s "
          f"{mbps(n, tw):>8.0f} {t:>7.3f}s {mbps(n, t):>8.0f}  "
          f"{'OK' if out == data else 'FAIL'}")
    # --- implication store --------------------------------
    ADDR = 1 << 32
    def store_w():
        if os.path.exists("/tmp/qhd_b.evb2"):
            os.remove("/tmp/qhd_b.evb2")
        s = BlockStore.create("/tmp/qhd_b.evb2", 1 << 40)
        s.write(ADDR, data)
        s.save()
        s.close()
    t, _ = med(store_w, n=1)
    sz = os.path.getsize("/tmp/qhd_b.evb2")
    tw = t
    s = BlockStore.open("/tmp/qhd_b.evb2")
    t, out = med(lambda: s.read(ADDR, n))
    print(f"{'EVB2 store':14} {sz:>12,} {tw:>7.3f}s "
          f"{mbps(n, tw):>8.0f} {t:>7.3f}s {mbps(n, t):>8.0f}  "
          f"{'OK' if out == data else 'FAIL'}")
    # random row windows out of the space
    rng = np.random.default_rng(3)
    rows = rng.integers(0, H, 300)
    t0 = time.perf_counter()
    for y in rows:
        s.read(ADDR + int(y) * W * 4, W * 4)
    dt = time.perf_counter() - t0
    print(f"{'  row window':14} {'':>12} {'':>8} {'':>8} "
          f"{dt / 300 * 1000:>7.2f}ms {'per row':>8}")
    s.close()
    # --- IMPZ1 ----------------------------------------------
    t, blob = med(lambda: compress(data))
    tw = t
    t, out = med(lambda: decompress(blob))
    print(f"{'IMPZ1':14} {len(blob):>12,} {tw:>7.2f}s "
          f"{mbps(n, tw):>8.1f} {t:>7.2f}s {mbps(n, t):>8.1f}  "
          f"{'OK' if out == data else 'FAIL'}")
    # --- zip --------------------------------------------------
    def zip_w():
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("x", data)
        return b.getvalue()
    t, zblob = med(zip_w)
    tw = t
    def zip_r():
        with zipfile.ZipFile(io.BytesIO(zblob)) as z:
            return z.read("x")
    t, out = med(zip_r)
    print(f"{'zip deflate':14} {len(zblob):>12,} {tw:>7.2f}s "
          f"{mbps(n, tw):>8.1f} {t:>7.2f}s {mbps(n, t):>8.1f}  "
          f"{'OK' if out == data else 'FAIL'}")


if __name__ == "__main__":
    main()
