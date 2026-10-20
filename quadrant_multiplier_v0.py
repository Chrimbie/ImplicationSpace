#!/usr/bin/env python3
"""Implication Space — quadrant multiplier (v0, 2026-10-20).

Christopher's design: separate the full amount of compute into
quadrants of four bits; each of those bits becomes an implication
space of its own. The whole register is multiplied, not just one
4-bit unit:

    per real bit:  1 -> 2 -> 32 -> 8192 implied bits
                   (children 2i/2i+1, then blocks of 16, then of 256,
                   applied per bit — the same growth as the primitive
                   bit multiplier, whose stage layout is exactly the
                   one-quadrant case of this space)

A register of R real bits (R a multiple of 4) therefore works with
R x 8192 implied bits at stage 3, while storing only the R real bits.
Quadrants are independent — no cross-quadrant composition is assumed
(that rule is Christopher's to state). Any implied position is
answered on demand by provenance: p // descendants_per_bit.
"""

import sys

sys.path.insert(0, "/home/hatch/workspace/implication_space")
from bit_multiplier_v0 import BitMultiplier

# descendants per real bit at each stage
DESC = (1, 2, 32, 8192)


class QuadrantSpace:
    """A full register, every bit the root of its own implied space."""

    def __init__(self, bits):
        assert len(bits) % 4 == 0 and all(b in (0, 1) for b in bits)
        self.bits = tuple(bits)
        self.n_real = len(bits)
        self.n_quadrants = self.n_real // 4

    def width(self, stage):
        return self.n_real * DESC[stage]

    def quadrant_of(self, stage, p):
        return self.ancestor(stage, p) // 4

    def ancestor(self, stage, p):
        assert 0 <= p < self.width(stage)
        return p // DESC[stage]

    def bit(self, stage, p):
        """On-demand value — provenance lookup only, nothing stored."""
        return self.bits[self.ancestor(stage, p)]

    def materialize(self, stage):
        """Build the multiplied space (verification only)."""
        return [self.bits[p // DESC[stage]] for p in range(self.width(stage))]


def _check(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    return cond


def run_tests():
    ok = True
    bits = tuple((i * 7 + 1) % 2 for i in range(64))  # arbitrary pattern
    qs = QuadrantSpace(bits)
    ok &= _check(qs.n_quadrants == 16 and qs.n_real == 64,
                 "64 real bits = 16 quadrants of four")
    ok &= _check([qs.width(s) for s in range(4)] ==
                 [64, 128, 2048, 524288],
                 "widths scale per bit: 64 -> 128 -> 2048 -> 524288")
    for s in range(4):
        mat = qs.materialize(s)
        ok &= _check(all(qs.bit(s, p) == mat[p]
                         for p in range(0, qs.width(s), 97)),
                     f"stage {s}: on-demand == materialized (sampled)")
    counts = [sum(1 for p in range(qs.width(3)) if qs.ancestor(3, p) == r)
              for r in range(64)]
    ok &= _check(counts == [8192] * 64,
                 "stage 3: every real bit has exactly 8192 descendants")
    # consistency: one quadrant == the primitive bit multiplier, all stages
    b4 = (1, 0, 1, 1)
    q1, bm = QuadrantSpace(b4), BitMultiplier(b4)
    ok &= _check(all(q1.materialize(s) == bm.materialize(s)
                     for s in range(4)),
                 "one quadrant reproduces bit_multiplier_v0 exactly")
    ok &= _check(all(qs.quadrant_of(3, p) == qs.ancestor(3, p) // 4
                     for p in (0, 8192, 200000, 524287)),
                 "quadrant addressing consistent")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    run_tests()
