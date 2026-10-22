"""gpu_partition.py - Christopher's architecture, 2026-10-22:
partition the CPU, use ~a quarter of it to act as a GPU.

The partitioned cores form a tile-render pool (the way a GPU
works: independent tiles, rendered concurrently, painter order
preserved inside each tile) drawing into the implied
framebuffer. The rest of the CPU is untouched by the render.

Test, against the same canvas and shapes as
implied_framebuffer.py:
  * serial baseline - the existing single-core rasterizer;
  * partition       - cores//4 worker processes, each pinned
    to its partition core where the OS allows, rendering
    1024x256 tiles through a process pool;
  * proof           - the partition's store must equal the
    serial store pixel-for-pixel on every verified window,
    and the presented frames must be byte-identical;
  * measure         - draw times and the speedup of the
    quarter-CPU "GPU" over one core.
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_blockstore import BlockStore
from implied_framebuffer import (ImpliedFramebuffer, SHAPES,
                                 analytic_window, phone_shapes,
                                 rasterize, write_png)

TILE_W, TILE_H = 1024, 256

_SHAPES = None
_W = None
_PIN_OK = None


def _pin(core):
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})
            return True
        import ctypes
        k32 = ctypes.windll.kernel32
        return bool(k32.SetProcessAffinityMask(
            k32.GetCurrentProcess(), 1 << core))
    except Exception:
        return False


def _init(shapes, width, cores):
    global _SHAPES, _W, _PIN_OK
    _SHAPES = shapes
    _W = width
    idx = mp.current_process()._identity
    if idx:
        _PIN_OK = _pin(cores[(idx[0] - 1) % len(cores)])


def _render_tile(task):
    x, y, w, h = task
    arr = analytic_window(_SHAPES, (_W, _W), x, y, w, h)
    aff = None
    if hasattr(os, "sched_getaffinity"):
        try:
            aff = sorted(os.sched_getaffinity(0))
        except Exception:
            aff = None
    return (x, y, w, h, arr.tobytes(), os.getpid(), aff, _PIN_OK)


def shape_bbox(s):
    k = s["kind"]
    if k == "rect" or k == "checker":
        return (s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"])
    if k == "circle":
        return (s["cx"] - s["r"], s["cy"] - s["r"],
                s["cx"] + s["r"] + 1, s["cy"] + s["r"] + 1)
    if k == "band":
        return (s["x"], s["y"],
                s["x"] + int(s["rows"] * s["slope"]) + s["w"],
                s["y"] + s["rows"])
    raise ValueError(k)


def merge_regions(boxes):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        for b in boxes:
            for o in out:
                if not (b[2] < o[0] or o[2] < b[0] or
                        b[3] < o[1] or o[3] < b[1]):
                    o[0] = min(o[0], b[0])
                    o[1] = min(o[1], b[1])
                    o[2] = max(o[2], b[2])
                    o[3] = max(o[3], b[3])
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return [tuple(b) for b in boxes]


def tiles_for(regions, W, H):
    tiles = []
    for (x0, y0, x1, y1) in regions:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        y = y0
        while y < y1:
            x = x0
            while x < x1:
                tiles.append((x, y, min(TILE_W, x1 - x),
                              min(TILE_H, y1 - y)))
                x += TILE_W
            y += TILE_H
    return tiles


def write_tile(fb, x, y, w, h, data):
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
    for j in range(h):
        row = fb._row(y + j)
        row[x * 4:(x + w) * 4] = arr[j].tobytes()
        fb._dirty.add(y + j)


def build_serial(path, W, shapes):
    if os.path.exists(path):
        os.remove(path)
    store = BlockStore.create(path, W * W * 4, block_size=W * 4)
    fb = ImpliedFramebuffer(store, W, W)
    t0 = time.time()
    rows = 0
    for s in shapes:
        rasterize(fb, [s])
        rows += fb.flush()
    dt = time.time() - t0
    store.close()
    return dt, rows


def build_partition(path, W, shapes, cores):
    if os.path.exists(path):
        os.remove(path)
    store = BlockStore.create(path, W * W * 4, block_size=W * 4)
    fb = ImpliedFramebuffer(store, W, W)
    regions = merge_regions([shape_bbox(s) for s in shapes])
    tiles = tiles_for(regions, W, W)
    workers = {}
    t0 = time.time()
    with mp.Pool(len(cores), initializer=_init,
                 initargs=(shapes, W, cores)) as pool:
        for (x, y, w, h, data, pid, aff, pin) in \
                pool.imap_unordered(_render_tile, tiles):
            write_tile(fb, x, y, w, h, data)
            wrec = workers.setdefault(
                pid, {"affinity": aff, "pin_ok": pin, "tiles": 0})
            wrec["tiles"] += 1
    rows = fb.flush()
    dt = time.time() - t0
    store.close()
    return dt, rows, len(tiles), regions, workers


def verify(path, W, shapes, wins):
    store = BlockStore.open(path)
    fb = ImpliedFramebuffer(store, W, W)
    out = {}
    for (x, y, w, h) in wins:
        got = fb.read_window(x, y, w, h)
        want = analytic_window(shapes, (W, W), x, y, w, h)
        out[(x, y)] = (got, bool(np.array_equal(got, want)))
    store.close()
    return out


def main():
    phone = "--phone" in sys.argv
    W = 32768 if phone else 65536
    shapes = phone_shapes() if phone else SHAPES
    tag = "phone" if phone else "full"
    total = os.cpu_count() or 1
    n_part = max(1, total // 4)
    # high indices: on ARM (Pixel) Linux enumerates the
    # performance cluster last; on x86 any cores serve.
    cores = list(range(total - n_part, total))
    print(f"CPU: {total} logical cores | GPU partition: "
          f"{n_part} core(s) {cores} ({100 * n_part / total:.0f}%)")
    print(f"canvas {W}x{W} = {W * W * 4:,} logical bytes")

    wins = ([(2900, 3100, 1280, 720), (29300, 29700, 1280, 720),
             (20000, 20000, 640, 480)] if phone else
            [(29000, 29500, 1920, 1080),
             (59600, 60200, 1600, 1200),
             (40000, 40000, 800, 600)])

    p_ser = f"framebuffer_serial_{tag}.evb2"
    p_par = f"framebuffer_gpu_{tag}.evb2"
    t_ser, rows_ser = build_serial(p_ser, W, shapes)
    print(f"serial (1 core): {t_ser:.1f}s, {rows_ser:,} rows")
    vs = verify(p_ser, W, shapes, wins)
    if phone:
        # lean storage: serial windows are captured in RAM;
        # the serial store comes off the phone before the
        # partition store is built (peak ~1 store, not 2)
        os.remove(p_ser)
        print("(serial store removed - phone-lean mode)")
    t_par, rows_par, ntiles, regions, workers = build_partition(
        p_par, W, shapes, cores)
    print(f"partition ({n_part} cores): {t_par:.1f}s, "
          f"{rows_par:,} rows, {ntiles} tiles over "
          f"{len(regions)} region(s)")
    for pid, wrec in sorted(workers.items()):
        print(f"  worker pid {pid}: {wrec['tiles']} tiles, "
              f"affinity {wrec['affinity']}, "
              f"pin_ok {wrec['pin_ok']}")
    print(f"speedup: {t_ser / t_par:.2f}x on "
          f"{100 * n_part / total:.0f}% of the CPU")

    vp = verify(p_par, W, shapes, wins)
    allok = True
    for k in vs:
        same = bool(np.array_equal(vs[k][0], vp[k][0]))
        ok = vs[k][1] and vp[k][1] and same
        allok &= ok
        print(f"window @{k}: serial==analytic "
              f"{vs[k][1]}, partition==analytic {vp[k][1]}, "
              f"partition==serial {same} -> "
              f"{'PASS' if ok else 'FAIL'}")
    # presented frames byte-identical?
    write_png(f"framebuffer_serial_{tag}.png", vs[wins[0][:2]][0])
    write_png(f"framebuffer_gpu_{tag}.png", vp[wins[0][:2]][0])
    hs = hashlib.sha256(open(
        f"framebuffer_serial_{tag}.png", "rb").read()).hexdigest()
    hp = hashlib.sha256(open(
        f"framebuffer_gpu_{tag}.png", "rb").read()).hexdigest()
    print(f"presented frames identical: {hs == hp} "
          f"(sha256 {hp[:16]}...)")
    allok &= hs == hp
    print(json.dumps({"total_cores": total, "partition_cores":
                      n_part, "serial_s": round(t_ser, 2),
                      "partition_s": round(t_par, 2),
                      "speedup": round(t_ser / t_par, 2),
                      "identical": bool(allok)}))
    print("GPU PARTITION TEST:",
          "ALL PASS" if allok else "FAIL")


if __name__ == "__main__":
    main()
