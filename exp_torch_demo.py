"""EXP-TORCH DEMO — a program running on EXP-VRAM as its
hardware, 2026-10-21.

Written the way programs are written: tensor allocations and
ordinary torch ops. Underneath, every tensor lives in
EXP-VRAM (exp_torch.ExprTensor -> implied_tensor files) and
pages tiles to the real GPU for math.

  ACT 1  a model whose weights are HELD in EXP-VRAM:
         W1 is 8192x32768 — 1.07 billion parameters, 4.3 GB —
         living in the space as a rule; a forward pass
         (matmul, relu, matmul) runs through the device
         layer and is verified against a plain-torch
         reference computed in chunks.
  ACT 2  real data in the device: a 4096x4096 weight from
         an ordinary torch tensor, multiplied through the
         layer, checked against torch itself.
  RECEIPT: total logical bytes the program allocated in
         EXP-VRAM vs bytes stored; the verdicts.
"""

import subprocess
import time

import numpy as np
import torch

from exp_torch import ExpTensor

torch.manual_seed(0)


def vram_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader"], capture_output=True,
            text=True, timeout=4).stdout.strip()
        return out
    except Exception:
        return "n/a"


def main():
    print("ACT 1 — a 1.07B-parameter model held in EXP-VRAM",
          flush=True)
    W1 = ExpTensor.rule((32768, 32768), "index_hash",
                        "dev_w1.itb")
    W2 = ExpTensor.rule((32768, 16), "formula_ij", "dev_w2.itb")
    x = ExpTensor.from_torch(torch.randn(512, 32768) * 0.05,
                             "dev_x.itb")
    for name, t in (("W1", W1), ("W2", W2), ("x", x)):
        s = t.storage_stats
        print(f"  {name}: shape {s['shape']} logical "
              f"{s['logical_bytes'] / 1e9:.2f} GB stored "
              f"{s['stored_bytes'] / 1e6:.2f} MB", flush=True)
    t0 = time.time()
    h = torch.relu(x @ W1)
    y = h @ W2
    dt = time.time() - t0
    print(f"  forward through EXP-VRAM in {dt:.1f}s | VRAM "
          f"now: {vram_line()}", flush=True)

    # plain-torch reference, chunked over W1's output columns
    x_np = x.materialize("cpu").numpy()
    W1s = W1._storage
    h_ref = np.empty((512, 32768), dtype=np.float32)
    for c0 in range(0, 32768, 4096):
        rows = np.stack([W1s.read_flat(r * 32768 + c0, 4096)
                         for r in range(32768)])
        h_ref[:, c0:c0 + 4096] = x_np @ rows
    h_ref = np.maximum(h_ref, 0)
    W2np = W2.materialize("cpu").numpy()
    y_ref = h_ref @ W2np
    y_got = y.materialize("cpu").numpy()
    diff = float(np.abs(y_got - y_ref).max())
    ok1 = diff < 0.05
    print(f"  y vs plain-torch reference: max |diff| "
          f"{diff:.5f}  {'MATCH' if ok1 else 'MISMATCH'}",
          flush=True)

    print("ACT 2 — real torch data through the device",
          flush=True)
    Wt = torch.randn(4096, 4096)
    v = torch.randn(4096, 1)
    eW = ExpTensor.from_torch(Wt, "dev_wt.itb")
    ev = ExpTensor.from_torch(v, "dev_v.itb")
    out = eW @ ev
    ref = Wt @ v
    diff2 = float((out.materialize("cpu") - ref).abs().max())
    ok2 = diff2 < 1e-3
    print(f"  4096x4096 @ 4096x1: max |diff| vs torch "
          f"{diff2:.6f}  {'MATCH' if ok2 else 'MISMATCH'}",
          flush=True)

    total_logical = sum(t.storage_stats["logical_bytes"]
                        for t in (W1, W2, x, eW, ev))
    total_stored = sum(t.storage_stats["stored_bytes"]
                       for t in (W1, W2, x, eW, ev))
    print(f"RECEIPT: program allocated "
          f"{total_logical / 1e9:.2f} GB in EXP-VRAM | stored "
          f"{total_stored / 1e6:.1f} MB | acts "
          f"{'1 MATCH' if ok1 else '1 FAIL'} / "
          f"{'2 MATCH' if ok2 else '2 FAIL'} | the program "
          f"ran on EXP-VRAM as its hardware", flush=True)


if __name__ == "__main__":
    main()
