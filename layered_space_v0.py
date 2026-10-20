#!/usr/bin/env python3
"""Implication Space — layers as a VAR (v0, 2026-10-20).

Christopher's design step: the number of implication layers is a
variable, not a fixed ladder. One doubling per layer — a real bit's
descendants after L layers number 2^L, whatever grouping the old
stages used (stage 1 = 1 layer, stage 2 = 4 more, stage 3 = 8 more;
the canon endpoint 4 -> 32,768 is simply L = 13).

Consequences of the knob:
  - width of the working space = R x 2^L for R real bits
  - provenance of any implied position p: ancestor = p >> L
  - stored state never changes: the R real bits, full stop
  - L can be re-set at any time; the space re-implies instantly
    because nothing was ever materialized
"""

import sys

sys.path.insert(0, "/home/hatch/workspace/implication_space")
from bit_multiplier_v0 import BitMultiplier
from quadrant_multiplier_v0 import QuadrantSpace


class LayeredSpace:
    """A register of real bits + a layers VAR. The space is implied."""

    def __init__(self, bits, layers=13):
        assert len(bits) % 4 == 0 and all(b in (0, 1) for b in bits)
        self.bits = tuple(bits)
        self.n_real = len(bits)
        self.set_layers(layers)

    def set_layers(self, layers):
        assert isinstance(layers, int) and layers >= 0
        self.layers = layers  # <-- the VAR

    @property
    def width(self):
        return self.n_real << self.layers

    def ancestor(self, p):
        assert 0 <= p < self.width
        return p >> self.layers

    def bit(self, p):
        """On-demand value of implied bit p at the current layer count."""
        return self.bits[self.ancestor(p)]

    def materialize(self):
        """Build the space at the current L (verification only)."""
        d = 1 << self.layers
        return [self.bits[p // d] for p in range(self.width)]


def _check(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    return cond


def run_tests():
    ok = True
    bits = tuple((i * 7 + 1) % 2 for i in range(64))
    sp = LayeredSpace(bits, layers=0)
    ok &= _check(sp.width == 64 and sp.materialize() == list(bits),
                 "L=0: the space is the real bits themselves")
    for L in (1, 2, 5, 9):
        sp.set_layers(L)
        mat = sp.materialize()
        ok &= _check(sp.width == 64 << L and
                     all(sp.bit(p) == mat[p] for p in range(sp.width)),
                     f"L={L}: width {sp.width}, on-demand == materialized")
    # the canon is one setting of the VAR
    b4 = (1, 0, 1, 1)
    sp4 = LayeredSpace(b4, layers=13)
    ok &= _check(sp4.width == 32768 and
                 sp4.materialize() == BitMultiplier(b4).materialize(3),
                 "L=13 on 4 bits == the canon 32,768 (bit multiplier stage 3)")
    sp64 = LayeredSpace(bits, layers=13)
    ok &= _check(sp64.materialize() == QuadrantSpace(bits).materialize(3),
                 "L=13 on 64 bits == quadrant space stage 3")
    # the VAR moves and the space follows, state untouched
    sp.set_layers(3)
    before = [sp.bit(p) for p in range(sp.width)]
    sp.set_layers(13)
    ok &= _check(sp.bits == bits and sp.width == 524288 and
                 all(sp.bit(p) == before[p >> 10] for p in
                     range(0, sp.width, 4999)),
                 "re-setting L re-implies instantly; real bits unchanged")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    run_tests()
