"""qhd_test.py - implication space vs a 2560x1440 image.

Christopher's question (2026-10-22): "could we try implication
space on a 2560x1440 image?" Two readings, both tested on the
same raw RGBA bytes:
  A) the pixels live in an EVB2 BlockStore, deep in a big
     envelope; windows read back out and checked pixel-exact.
  B) the same bytes go through IMPZ1 (his equation as a
     compressor) with zip alongside for reference.
The test image is synthetic and labeled as such: sky gradient
(smooth), flat analytic shapes (rule-like), blurred-noise
texture + a raw noise strip (photo-like entropy).
"""
import hashlib
import io
import sys
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/home/hatch/workspace/implication_space")

from exp_blockstore import BlockStore
from implication_zip import compress, decompress

W, H = 2560, 1440
rng = np.random.default_rng(11)
img = np.zeros((H, W, 4), dtype=np.uint8)

# top 40%: smooth vertical gradient (sky)
sky = int(H * 0.40)
grad = np.linspace(30, 200, sky, dtype=np.uint8)
img[:sky, :, 0] = grad[:, None] // 3
img[:sky, :, 1] = grad[:, None] // 2
img[:sky, :, 2] = grad[:, None]

# middle 35%: flat analytic shapes on a flat field
mid_end = int(H * 0.75)
img[sky:mid_end, :] = (24, 24, 32, 255)
yy, xx = np.mgrid[sky:mid_end, 0:W]
for cx, cy, r, col in [(500, 700, 160, (220, 80, 60)),
                       (1300, 660, 220, (240, 200, 90)),
                       (2050, 720, 140, (90, 200, 140))]:
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[sky:mid_end][m] = (*col, 255)
img[sky:mid_end, 900:1700] = img[sky:mid_end, 900:1700]

# bottom 25%: blurred noise (photo-like texture) + noise strip
tex_h = H - mid_end
noise = rng.normal(128, 40, (tex_h // 8 + 2, W // 8 + 2, 3))
big = np.kron(noise, np.ones((8, 8, 1)))[:tex_h, :W]
img[mid_end:, :, :3] = np.clip(big, 0, 255).astype(np.uint8)
img[H - 40:, :, :3] = rng.integers(0, 256, (40, W, 3),
                                   dtype=np.uint8)
img[:, :, 3] = 255

data = img.tobytes()
print(f"image: {W}x{H} RGBA, raw {len(data):,} bytes")
print(f"sha256(raw) = {hashlib.sha256(data).hexdigest()[:24]}...")

# ---- A) live in the space --------------------------------
store = BlockStore.create("/tmp/qhd.evb2", 1 << 40)
ADDR = 1 << 32                       # 4 GiB deep
store.write(ADDR, data)
store.save()
print(f"A) in-space: stored {store.stored_bytes():,} B in "
      f"{store.blocks_stored()} blocks at address {ADDR:,}")
back = store.read(ADDR, len(data))
print(f"A) full read-back sha match: "
      f"{hashlib.sha256(back).digest() == hashlib.sha256(data).digest()}")
for name, y in [("sky", 100), ("shapes", 700), ("texture", 1300)]:
    w0 = store.read(ADDR + (y * W) * 4, W * 4)
    w1 = data[(y * W) * 4:(y * W + W) * 4]
    print(f"A) window row {y:>5} ({name:7}): "
          f"{'PIXEL-EXACT' if w0 == w1 else 'MISMATCH'}")
store.close()

# ---- B) the compressor ------------------------------------
blob = compress(data)
ok = decompress(blob) == data
zb = io.BytesIO()
with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("x", data)
print(f"B) IMPZ1: {len(blob):,} B (ratio {len(blob)/len(data):.3f}) "
      f"round-trip {'PASS' if ok else 'FAIL'}")
print(f"B) zip:   {len(zb.getvalue()):,} B "
      f"(ratio {len(zb.getvalue())/len(data):.3f})")
