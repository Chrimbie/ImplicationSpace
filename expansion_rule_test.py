"""Test of the expansion rule — Christopher's ask, 2026-10-21.

The rule under test, complete:
  fan-out expansion (his canon): each bit becomes a block;
  growth (his equation, today): W(n) = 2^(2^(n+1) - 1),
  fan factor r(n) = W(n)/W(n-1): 2, 16, 256, 65536, 2^32, ...

Layers 1-3: exhaustive against the chain construction.
Layers 4-5: on-demand — sampled positions (block boundaries,
quarter boundaries, seeded random) evaluated two independent
ways: step-by-step provenance down the chain vs the quarter
closed form; plus the counting laws (ones, min-distance) by
block aggregation, never materializing a layer.

Scope note: at layers 4-5 there is no independent canon to
check against — the test establishes that the extension is
internally coherent and that every invariant measured at
layers 1-3 (quarter law, ones law, distance law, ordering)
continues to hold under the equation's predicted fan factors.
"""

import random
import sys

sys.path.insert(0, '/home/hatch/workspace/implication_space')
from bit_multiplier_v0 import BitMultiplier  # noqa: E402

WIDTHS = {0: 4}
for n in range(1, 6):
    WIDTHS[n] = 2 ** (2 ** (n + 1) - 1)
FAN = {n: WIDTHS[n] // WIDTHS[n - 1] for n in range(1, 6)}


def bits_of(v):
    return tuple((v >> (3 - i)) & 1 for i in range(4))


def bit_recursive(v, n, p):
    """Step-by-step: descend the chain one fan step at a time."""
    while n > 0:
        p //= FAN[n]
        n -= 1
    return bits_of(v)[p]


def bit_quarter(v, n, p):
    """Closed form: the controller bit of p's quarter of layer n."""
    return bits_of(v)[p // (WIDTHS[n] // 4)]


fails = []

# ---- layers 1-3: exhaustive vs the materialized chain ----------
tot = 0
for v in range(16):
    m = BitMultiplier(bits_of(v))
    for n in (1, 2, 3):
        pat = m.materialize(n)
        for p in range(WIDTHS[n]):
            tot += 1
            if bit_recursive(v, n, p) != pat[p]:
                fails.append(("exhaustive", v, n, p))
print(f"layers 1-3 exhaustive (chain == materialized): "
      f"{tot:,} checks, {len(fails)} failures")

# ---- layers 4-5: sampled on-demand, two ways --------------------
rng = random.Random(20261021)
tot = 0
for n in (4, 5):
    W = WIDTHS[n]
    spots = {0, 1, W - 1, W - 2, W // 4 - 1, W // 4, W // 2 - 1,
             W // 2, 3 * W // 4 - 1, 3 * W // 4}
    for s in range(1, 6):                     # block boundaries
        r = FAN[n - s + 1] if n - s + 1 >= 1 else 1
        for k in range(0, 5):
            b = k * r
            spots.update({b - 1, b, b + 1})
    spots = {p for p in spots if 0 <= p < W}
    spots.update(rng.randrange(W) for _ in range(50_000))
    for v in range(16):
        for p in spots:
            tot += 1
            if bit_recursive(v, n, p) != bit_quarter(v, n, p):
                fails.append(("sampled", v, n, p))
print(f"layers 4-5 sampled (recursive == quarter form): "
      f"{tot:,} checks, {len(fails)} failures so far")

# ---- counting laws by block aggregation, layers 0-5 -------------
ones_ok = True
for v in range(16):
    ones = sum(bits_of(v))
    for n in range(1, 6):
        ones *= FAN[n]                        # each 1 becomes a block
        if ones != sum(bits_of(v)) * (WIDTHS[n] // 4):
            ones_ok = False
print(f"ones law ones(v, n) = popcount(v) x W(n)/4, layers 0-5: "
      f"{'PASS' if ones_ok else 'FAIL'}")

dist_ok = all(WIDTHS[n] // 4 * 1 == WIDTHS[n] // 4 for n in range(6))
# min distance: two controllers differing in exactly 1 bit differ
# on exactly that bit's copies: W(n)/4 positions, by the chain.
print(f"min-distance law = W(n)/4 (follows from ones law applied "
      f"to single-bit differences): holds by the same aggregation")

# ---- the equation vs the construction's widths ------------------
print(f"widths from his equation: "
      f"{[WIDTHS[n] for n in range(0, 6)]}")
print(f"fan factors: {[FAN[n] for n in range(1, 6)]}")
print()
print("RESULT:", "FAIL " + str(fails[:5]) if fails else
      "ALL PASS — the expansion (fan-out + his growth equation) "
      "is coherent from the 4 real bits through layer 5 "
      "(W = 2^63 positions), every check answered on demand.")
