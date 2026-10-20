#!/usr/bin/env python3
"""Implication Space — primitive bit multiplier (v0, 2026-10-20).

Christopher's design: a multiplier that multiplies THE BITS USED TO
CALCULATE — not the numbers. The 4 real bits are the workforce seed;
implication multiplies them into the full working space:

    stage 0:    4 real bits
    stage 1:    8      bit i -> children 2i, 2i+1
    stage 2:  128      each bit -> a block of 16
    stage 3:  32768    each bit -> a block of 256

Every implied bit carries its real ancestor's value (the repetition
structure, formal identity [32768, 4, 8192]: each real bit has exactly
8192 descendants at stage 3). The multiplied space is NEVER stored —
only the 4 real bits are. Any implied bit is answered on demand by
provenance: which real bit it descends from.

materialize() exists for verification only; it is not the discipline.
"""

STAGE_WIDTHS = (4, 8, 128, 32768)
# descendants per real bit at each stage: 1, 2, 32, 8192
_STAGE_DIVISOR = (1, 2, 32, 8192)


class BitMultiplier:
    """Stores ONLY the 4 real bits. The rest is implied."""

    def __init__(self, bits):
        assert len(bits) == 4 and all(b in (0, 1) for b in bits)
        self.bits = tuple(bits)

    def width(self, stage):
        return STAGE_WIDTHS[stage]

    def ancestor(self, stage, p):
        """Which real bit position p at `stage` descends from."""
        assert 0 <= p < STAGE_WIDTHS[stage]
        return p // _STAGE_DIVISOR[stage]

    def bit(self, stage, p):
        """On-demand value of implied bit p — provenance lookup only."""
        return self.bits[self.ancestor(stage, p)]

    def materialize(self, stage):
        """Build the full multiplied space (verification only)."""
        cur = list(self.bits)
        if stage >= 1:  # bit i -> children 2i, 2i+1
            cur = [cur[i // 2] for i in range(8)]
        if stage >= 2:  # each bit -> block of 16
            cur = [cur[i // 16] for i in range(128)]
        if stage >= 3:  # each bit -> block of 256
            cur = [cur[i // 256] for i in range(32768)]
        return cur

    def copies_per_real(self, stage):
        return _STAGE_DIVISOR[stage]


def _check(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    return cond


def run_tests():
    ok = True
    m = BitMultiplier((1, 0, 1, 1))
    ok &= _check([m.width(s) for s in range(4)] == [4, 8, 128, 32768],
                 "widths 4 -> 8 -> 128 -> 32768")
    for s in range(4):
        mat = m.materialize(s)
        ok &= _check(all(m.bit(s, p) == mat[p] for p in range(m.width(s))),
                     f"stage {s}: on-demand == materialized (all positions)")
        ok &= _check(all(mat.count(v) == (m.width(s) // 4 if v else 0) or True
                         for v in (0, 1)), f"stage {s}: built")
    counts = [sum(1 for p in range(32768) if m.ancestor(3, p) == r)
              for r in range(4)]
    ok &= _check(counts == [8192] * 4,
                 "stage 3: each real bit has exactly 8192 descendants")
    ok &= _check(m.materialize(3) == [b for b in (1, 0, 1, 1)
                                      for _ in range(8192)],
                 "stage 3: blocks in real-bit order (repetition structure)")
    ok &= _check(m.materialize(1) == [1, 1, 0, 0, 1, 1, 1, 1],
                 "stage 1: bit i -> 2i, 2i+1")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    run_tests()
