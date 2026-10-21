"""Sets of four bits as implication-computation units, on GPU.
Christopher's construction, confirmed 2026-10-21.

SCOPE TRIPLE (confirmed with Christopher before building):
  Claim:  a set of four bits on the GPU can run as an implication-
          computation unit -- and millions of them can run at once,
          each deriving its answers from its 4 bits, with the implied
          space never stored anywhere.
  Construction (his words): "transforming every one of those sets of
          four bits into implication computation"; the GPU "divvy up
          the work to everything using this method ... either all
          together in parallel or for other tasks."
          One GPU element per set of four. Each unit holds ONLY its
          nibble (controller value v in 0..15) and its recursive ID
          (its index). Its implied pattern at the canon width 32,768
          is DERIVED from the 4 bits by the multiplier rule (each real
          bit fans out to a block of 8,192; the combination IS the
          pattern) -- the pattern is never materialized on the GPU.
          Divided mode: each unit outputs its own answers.
          All-together mode: every unit's pattern contributes to one
          shared aggregate (XOR-folded joint pattern + totals).
  Measurement: (1) exactness -- sampled units' GPU answers vs naive
          CPU answers computed from the fully materialized pattern;
          (2) throughput -- units/sec and implied-bits-derived/sec at
          pool sizes 2^12 .. 2^24, both modes; (3) linear extrapolation
          to the full pool (25,769,803,776 sets = every nibble of the
          4070 Ti's 12 GB), labelled as an extrapolation.

PER-UNIT ANSWERS (defined over the DERIVED pattern, computed in closed
form from its 4-block structure -- the pattern has exactly 4 runs):
  popcount      = number of 1-bits in the 32,768-bit pattern
  weighted_sum  = sum over positions j of pattern_j * (j + 1)
  pattern_hash  = polynomial hash (base 257, mod 2^64) over the
                  pattern's 4,096 bytes, byte order left to right
All three are computed on GPU from (v, ID) alone; the CPU check
materializes the pattern and computes them naively.

Implementation note: PyTorch tensor ops (one element per unit, spread
across the GPU's cores by the framework). A hand-written CUDA kernel
is a possible later step, not part of this slice.
"""

import time

import numpy as np
import torch

WIDTH = 32768                    # canon width (4 -> 8 -> 128 -> 32768)
BLOCKS = 4
BLOCK = WIDTH // BLOCKS          # 8192 bits per real bit's block
BLOCK_BYTES = BLOCK // 8         # 1024
HASH_BASE = 257
HASH_SEED = 1469598103934665603  # definition constant for the hash
FULL_POOL = 25_769_803_776       # every nibble of 12 GiB (his number)
SIZES = [2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24]
VALUE_SEED = 4242                # controller values drawn (given, not
                                 # learned -- the units' 4 bits are
                                 # simply what they are for this slice)


def block_sums():
    """S_b = sum of (j+1) over positions j in block b (exact ints)."""
    out = []
    for b in range(BLOCKS):
        start = b * BLOCK
        out.append(BLOCK * (2 * start + BLOCK + 1) // 2)
    return out


def hash_run_constants():
    """powB = HASH_BASE^BLOCK_BYTES mod 2^64; G = geometric series
    sum_{i< BLOCK_BYTES} HASH_BASE^i mod 2^64. Python ints, exact."""
    mod = 2 ** 64
    powB = pow(HASH_BASE, BLOCK_BYTES, mod)
    G = 0
    p = 1
    for _ in range(BLOCK_BYTES):
        G = (G + p) % mod
        p = p * HASH_BASE % mod
    return powB, G


# ------------------------------------------------------------- CPU ------
def cpu_answers_naive(v):
    """Materialize the unit's pattern and compute answers naively."""
    bits4 = [(v >> (3 - b)) & 1 for b in range(BLOCKS)]
    pattern = np.repeat(np.array(bits4, dtype=np.uint8), BLOCK)
    popcount = int(pattern.sum())
    weighted = int((pattern.astype(np.int64)
                    * (np.arange(WIDTH, dtype=np.int64) + 1)).sum())
    byts = np.packbits(pattern)          # left-to-right byte order
    h = HASH_SEED
    mod = 2 ** 64
    for byte in byts:
        h = (h * HASH_BASE + int(byte)) % mod
    return popcount, weighted, h


# ------------------------------------------------------------- GPU ------
def gpu_answers(v_t, S_blocks, powB, G):
    """Per-unit answers from (v, ID) alone. v_t: (P,) int64 on GPU."""
    P = v_t.shape[0]
    dev = v_t.device
    # block bits: block b (left to right) = bit (3-b) of v
    bits = torch.stack([(v_t >> (3 - b)) & 1
                        for b in range(BLOCKS)], dim=1)      # (P, 4)
    popcount = bits.sum(dim=1) * BLOCK                       # (P,)
    S_t = torch.tensor(S_blocks, dtype=torch.int64, device=dev)
    weighted = (bits * S_t[None, :]).sum(dim=1)              # (P,)
    # polynomial hash over runs: h = h*powB + byte*G per block,
    # int64 arithmetic wraps mod 2^64 on CUDA
    h = torch.full((P,), HASH_SEED, dtype=torch.int64, device=dev)
    powB_t = torch.tensor(powB - 2 ** 64 if powB >= 2 ** 63 else powB,
                          dtype=torch.int64, device=dev)
    G_t = torch.tensor(G - 2 ** 64 if G >= 2 ** 63 else G,
                       dtype=torch.int64, device=dev)
    for b in range(BLOCKS):
        byte = bits[:, b] * 255
        h = h * powB_t + byte * G_t
    return popcount, weighted, h, bits


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log("=" * 72)
    log("SETS OF FOUR AS IMPLICATION-COMPUTATION UNITS -- GPU SLICE")
    log(f"device: {dev}"
        + (f" ({torch.cuda.get_device_name(0)})" if dev == 'cuda' else ""))
    log("Construction: Christopher's (see header). Derivation: his")
    log("multiplier rule (combination IS the pattern at width 32,768).")
    log("=" * 72)

    S_blocks = block_sums()
    powB, G = hash_run_constants()
    rng = np.random.default_rng(VALUE_SEED)

    # ---- exactness on a small pool first ------------------------------
    P0 = 4096
    v_small = rng.integers(0, 16, size=P0).astype(np.int64)
    v_t = torch.tensor(v_small, device=dev)
    pop, wei, hsh, _ = gpu_answers(v_t, S_blocks, powB, G)
    pop_c, wei_c, hsh_c = (pop.cpu().numpy(), wei.cpu().numpy(),
                           hsh.cpu().numpy().astype(np.uint64))
    sample = [0, 1, 2, 17, 255, 1000, 4095]
    ok = True
    for i in sample:
        cp, cw, ch = cpu_answers_naive(int(v_small[i]))
        match = (cp == int(pop_c[i]) and cw == int(wei_c[i])
                 and ch == int(hsh_c[i]))
        ok &= match
        log(f"  unit {i:5d} (v={int(v_small[i]):2d}): CPU naive "
            f"pop={cp} wsum={cw} hash={ch:#018x} | GPU match: {match}")
    log(f"EXACTNESS: {'PASS' if ok else 'FAIL'} "
        f"(GPU answers derived from 4 bits == naive answers from the "
        f"materialized 32,768-bit pattern)")

    # ---- throughput, both modes ---------------------------------------
    log("")
    log("THROUGHPUT (best of 3, synchronized):")
    results = {}
    for P in SIZES:
        v = rng.integers(0, 16, size=P).astype(np.int64)
        v_t = torch.tensor(v, device=dev)
        # divided mode: per-unit answers
        for mode in ('divided', 'together'):
            best = None
            for _ in range(3):
                if dev == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                pop, wei, hsh, bits = gpu_answers(
                    v_t, S_blocks, powB, G)
                if mode == 'together':
                    joint_bits = (bits.sum(dim=0) % 2)       # XOR fold
                    total_pop = pop.sum()
                    total_wei = wei.sum()
                    _ = (joint_bits, total_pop, total_wei)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                dt = time.time() - t0
                best = dt if best is None else min(best, dt)
            results[(P, mode)] = best
            log(f"  P={P:>10,}  {mode:8s}: {best * 1000:9.2f} ms   "
                f"{P / best:,.0f} units/s   "
                f"{P * WIDTH / best:,.0f} implied bits derived/s")
        del v_t
        if dev == 'cuda':
            torch.cuda.empty_cache()

    t24 = results[(SIZES[-1], 'divided')]
    rate = SIZES[-1] / t24
    full_s = FULL_POOL / rate
    log("")
    log(f"EXTRAPOLATION (linear, labelled as such): the full pool of "
        f"{FULL_POOL:,} sets at the measured 2^24 rate would take "
        f"~{full_s:.1f} s of GPU time in divided mode.")
    log("Stored per unit on the GPU: its 4 bits. The 32,768-bit pattern")
    log("exists only as derivation, in neither mode.")

    with open('implication_units_gpu_log.txt', 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
