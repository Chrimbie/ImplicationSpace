"""fps_bench.py - how many 2560x1440 images per second can
implication space actually move?

Steady-state EVB2 store, frames at sequential addresses:
  ingest (bulk)      write N frames, save once
  ingest (per-frame) write + save every frame (cloud-style)
  playback           read the frames back sequentially
  random access      read them in random order
  derived playback   read frames from never-written addresses
                     (pure derivation - nothing stored at all)
Frame = the seed-11 synthetic QHD image (14,745,600 B).
"""
import hashlib
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_blockstore import BlockStore
from qhd_bench import build_image

N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
STRIDE = 16 << 20                    # 16 MiB per frame slot
PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fps.evb2"


def fps(label, secs, n=N):
    print(f"{label:20} {n / secs:8.1f} img/s   "
          f"({secs:.2f}s for {n})")
    return n / secs


def main():
    data = build_image()
    want = hashlib.sha256(data).digest()
    if os.path.exists(PATH):
        os.remove(PATH)
    s = BlockStore.create(PATH, 1 << 41)

    t = time.perf_counter()
    for i in range(N):
        s.write(i * STRIDE, data)
    s.save()
    fps("ingest (bulk)", time.perf_counter() - t)

    t = time.perf_counter()
    for i in range(N):
        s.write((N + i) * STRIDE, data)
        s.save()
    fps("ingest (per-frame)", time.perf_counter() - t)

    t = time.perf_counter()
    ok = 0
    for i in range(N):
        if hashlib.sha256(s.read(i * STRIDE, len(data))).digest() \
                == want:
            ok += 1
    dt = time.perf_counter() - t
    fps("playback", dt)
    print(f"{'  hash-verified':20} {ok}/{N} frames")

    rng = np.random.default_rng(5)
    order = rng.permutation(N)
    t = time.perf_counter()
    for i in order:
        s.read(int(i) * STRIDE, len(data))
    fps("random access", time.perf_counter() - t)

    t = time.perf_counter()
    for i in range(N):
        s.read((1000 + i) * STRIDE, len(data))   # never written
    fps("derived playback", time.perf_counter() - t)

    print(f"store on disk: {s.stored_bytes():,} B "
          f"for {2 * N} written frames "
          f"({2 * N * len(data):,} B of pixels)")
    s.close()
    os.remove(PATH)


if __name__ == "__main__":
    main()
