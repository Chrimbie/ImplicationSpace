#!/usr/bin/env python3
"""Benchmark: Implication Space AS SPECIFIED — a supply of bits.

Implication Space multiplies the amount of bits a computation has to
work with. It does not compute. This benchmark therefore measures the
supply: (1) state held, (2) cost of instantiating the space,
(3) read cost inside an ordinary computation (XOR-fold over random
positions, verified identical on both sides), and (4) spaces the
normal side cannot instantiate at all.

Normal side's pattern build uses byte-level repetition (the period is
byte-aligned for L >= 3: each real bit's block is whole 0x00/0xFF
bytes) — the fastest honest materialization available to it.
"""

import sys
import time
import random

sys.path.insert(0, "/home/hatch/workspace/implication_space")
from layered_space_v0 import LayeredSpace

BITS = tuple((i * 7 + 1) % 2 for i in range(64))  # 64 real bits


def bench(fn, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def normal_pattern(L):
    """The same bit pattern, materialized (byte-aligned repetition)."""
    block = bytes([0xFF]) if L >= 3 else None
    parts = []
    per_bit_bytes = (1 << L) // 8
    for r in range(64):
        parts.append(bytes([0xFF if BITS[r] else 0x00]) * per_bit_bytes)
    return bytearray(b"".join(parts))


def main():
    rng = random.Random(42)
    print("=== state + instantiation ===")
    print(f"{'L':>3} {'width (bits)':>18} {'normal state':>14} "
          f"{'implied state':>13} {'build normal':>13} {'build implied':>14}")
    for L in (8, 13, 18, 23):
        sp = LayeredSpace(BITS, layers=L)
        t_norm = bench(lambda: normal_pattern(L), reps=3)
        t_imp = bench(lambda: LayeredSpace(BITS, layers=L), reps=50)
        print(f"{L:>3} {sp.width:>18,} {sp.width // 8:>11,} B "
              f"{72:>10} bits {t_norm * 1000:>10.2f} ms {t_imp * 1000:>11.3f} ms")
        arr = normal_pattern(L)
        def normal_bit(p, arr=arr):
            return (arr[p >> 3] >> (7 - (p & 7))) & 1
        pos = [rng.randrange(sp.width) for _ in range(20000)]
        assert all(normal_bit(p) == sp.bit(p) for p in pos[:2000])
        t_rn = bench(lambda: sum(normal_bit(p) for p in pos), reps=3)
        t_ri = bench(lambda: sum(sp.bit(p) for p in pos), reps=3)
        print(f"      reads verified identical (2,000 sampled); "
              f"XOR-fold 20k reads: normal {t_rn * 1000:.1f} ms "
              f"vs implied {t_ri * 1000:.1f} ms")
    print("\n=== spaces the normal side cannot instantiate ===")
    for L in (40, 60):
        sp = LayeredSpace(BITS, layers=L)
        pos = [rng.randrange(sp.width) for _ in range(10000)]
        t0 = time.perf_counter()
        _ = [sp.bit(p) for p in pos]
        dt = time.perf_counter() - t0
        print(f"L={L}: width {sp.width:,} bits "
              f"({sp.width / 8e12:,.1f} TB as material); 10k reads in "
              f"{dt * 1000:.1f} ms from 64 stored bits")


if __name__ == "__main__":
    main()
