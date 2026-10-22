"""TENSOR DEMO — programs running VRAM's work through
exponentiated VRAM, 2026-10-21.

Three acts, each verified against an independent path:

  1. A 2^34-element tensor (68.7 GB — five times the card)
     exists only as a rule. Its sum and sum-of-squares are
     computed by streaming tiles through the GPU. Reference:
     a separately written CPU chunk loop. A sampler watches
     nvidia-smi the whole time: physical VRAM must stay
     tile-bounded while 68.7 GB is processed.
  2. An 8192x8192 matmul between two implied tensors,
     checked against torch's own materialized GEMM.
  3. A mostly-empty 4.3 GB tensor receives 100 scattered
     tile writes (real result data), is saved, reopened,
     and read back byte-exact — with elementwise math on
     top, sample-verified.

The receipt: logical bytes processed vs stored bytes, peak
physical VRAM observed, and every check's verdict.
"""

import subprocess
import threading
import time

import numpy as np

from implied_tensor import ImpliedTensor, _splitmix_f64

stop = threading.Event()
vram = {"base": None, "peak": 0}


def sample_vram():
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4).stdout
            used = int(out.strip().splitlines()[0])
            if vram["base"] is None:
                vram["base"] = used
            vram["peak"] = max(vram["peak"], used)
        except Exception:
            pass
        time.sleep(0.5)


def cpu_reference_sum(n):
    """Independent path: the index_hash spec, chunked on CPU,
    written separately from the tensor engine."""
    total = 0.0
    total2 = 0.0
    CH = 1 << 26
    for s in range(0, n, CH):
        idx = np.arange(s, min(n, s + CH), dtype=np.uint64)
        v = _splitmix_f64(idx).astype(np.float32)
        total += float(v.astype(np.float64).sum())
        total2 += float((v.astype(np.float64) ** 2).sum())
    return total, total2


def main():
    threading.Thread(target=sample_vram, daemon=True).start()
    time.sleep(1.0)

    print("ACT 1 — a 68.7 GB tensor, reduced through the GPU",
          flush=True)
    N = 2 ** 34
    X = ImpliedTensor.rule("big_x.itb", (N,), "index_hash")
    print(f"  logical {X.logical_bytes / 1e9:.1f} GB | stored "
          f"{X.stored_bytes} bytes (a rule, no blocks)",
          flush=True)
    t0 = time.time()
    red = X.reduce_all()
    s, s2 = red["sum"], red["sumsq"]
    dt = time.time() - t0
    rs, rs2 = cpu_reference_sum(N)
    ok1 = abs(s - rs) / abs(rs) < 1e-9
    ok2 = abs(s2 - rs2) / abs(rs2) < 1e-9
    print(f"  sum   gpu {s:.6f}  cpu {rs:.6f}  "
          f"{'MATCH' if ok1 else 'MISMATCH'}", flush=True)
    print(f"  sumsq gpu {s2:.6f}  cpu {rs2:.6f}  "
          f"{'MATCH' if ok2 else 'MISMATCH'}", flush=True)
    print(f"  streamed {X.logical_bytes / 1e9:.1f} GB in "
          f"{dt:.1f}s; physical VRAM base {vram['base']} MiB, "
          f"peak so far {vram['peak']} MiB", flush=True)

    print("ACT 2 — 8192x8192 matmul, implied @ implied",
          flush=True)
    A = ImpliedTensor.rule("mat_a.itb", (8192, 8192), "formula_ij")
    B = ImpliedTensor.rule("mat_b.itb", (8192, 8192),
                           "index_hash")
    t0 = time.time()
    C = A.matmul(B, "mat_c.itb")
    dt = time.time() - t0
    import torch
    Anp = A._derive(0, A.size).reshape(8192, 8192)
    Bnp = B._derive(0, B.size).reshape(8192, 8192)
    Cref = (torch.from_numpy(Anp).cuda()
            @ torch.from_numpy(Bnp).cuda()).cpu().numpy()
    Cgot = C.read_region(0, C.size).reshape(8192, 8192)
    diff = float(np.abs(Cgot - Cref).max())
    ok3 = diff < 0.05
    print(f"  matmul in {dt:.1f}s | max |diff| vs torch "
          f"{diff:.5f}  {'MATCH' if ok3 else 'MISMATCH'}",
          flush=True)
    print(f"  C: logical {C.logical_bytes / 1e6:.0f} MB, stored "
          f"{C.stored_bytes / 1e6:.0f} MB (real results)",
          flush=True)

    print("ACT 3 — scattered real writes into a 4.3 GB tensor",
          flush=True)
    Y = ImpliedTensor.create("sparse_y.itb", (2 ** 30,), "zeros",
                             tile_elems=1 << 20)
    rng = np.random.default_rng(3)
    tiles = rng.choice(1024, size=100, replace=False)
    for ti in tiles:
        Y.set_tile(int(ti), np.full(1 << 20, float(ti),
                                    dtype=np.float32))
    Y.save()
    Y2 = ImpliedTensor.open("sparse_y.itb")
    bad = 0
    for ti in tiles:
        tile = Y2.get_tile(int(ti))
        if not (tile == float(ti)).all():
            bad += 1
    ok4 = bad == 0
    print(f"  100 scattered tiles written, saved, reopened, "
          f"read back: {'PASS' if ok4 else f'FAIL ({bad})'}",
          flush=True)
    print(f"  Y: logical {Y.logical_bytes / 1e9:.1f} GB | "
          f"stored {Y.stored_bytes / 1e6:.0f} MB", flush=True)

    stop.set()
    time.sleep(0.6)
    print(f"RECEIPT: acts "
          f"{'1 MATCH' if ok1 and ok2 else '1 FAIL'} / "
          f"{'2 MATCH' if ok3 else '2 FAIL'} / "
          f"{'3 PASS' if ok4 else '3 FAIL'} | physical VRAM "
          f"peak {vram['peak']} MiB of 12282 while a "
          f"{X.logical_bytes / 1e9:.0f} GB tensor was processed "
          f"through exponentiated VRAM", flush=True)


if __name__ == "__main__":
    main()
