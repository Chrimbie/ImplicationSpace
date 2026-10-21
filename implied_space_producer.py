"""Produce implied space FROM THE RULE — Christopher's ask,
2026-10-21: "test if we can use the rule to produce implied space."

The producer knows ONLY:
  - the 4 real bits (its controller), and
  - the expansion rule: fan-out, with widths from his equation
        W(n) = 2^(2^(n+1) - 1)      (W(0) = 4)
    fan factor r(n) = W(n) / W(n-1) derived from the equation.
No width table is hardcoded anywhere in the producer.

Verification:
  1. produced widths == canon widths at layers 0-3, equation
     values at 4-5;
  2. produced bits == the existing chain construction
     (BitMultiplier), exhaustive at layers 1-3;
  3. produced bits at layers 4-5 == a SECOND, independent
     production method: windowed recursive expansion (to answer
     a window of bits at layer n, recursively produce the needed
     parent window at layer n-1, down to the controller);
  4. a demonstration: the produced space in use.
"""

import random
import sys

sys.path.insert(0, '/home/hatch/workspace/implication_space')
from bit_multiplier_v0 import BitMultiplier  # noqa: E402


def W(n):
    """His equation. Layer 0 is the controller itself."""
    return 4 if n == 0 else 2 ** (2 ** (n + 1) - 1)


class ProducedSpace:
    """An implication space produced by the rule alone."""

    def __init__(self, bits):
        assert len(bits) == 4 and all(b in (0, 1) for b in bits)
        self.bits = tuple(bits)

    def width(self, n):
        return W(n)

    def fan(self, n):
        return W(n) // W(n - 1)

    def bit(self, n, p):
        """The implied bit at layer n, position p — produced by
        descending the expansion the rule defines."""
        assert 0 <= p < W(n)
        while n > 0:
            p //= self.fan(n)
            n -= 1
        return self.bits[p]

    def window(self, n, a, b):
        """Produce bits [a, b) of layer n."""
        return [self.bit(n, p) for p in range(a, b)]

    def ones(self, n):
        """Count of 1s at layer n, by block aggregation."""
        return sum(self.bits) * (W(n) // 4)


def window_recursive(bits, n, a, b):
    """INDEPENDENT production: to produce window [a, b) at layer
    n, recursively produce the parent window it descends from,
    then expand each parent bit into its block — layer by layer,
    the way the expansion was originally taught."""
    if n == 0:
        return list(bits[a:b])
    r = W(n) // W(n - 1)
    parent = window_recursive(bits, n - 1, a // r, b // r + 1)
    out = []
    for i, pb in enumerate(parent):
        block_start = (a // r + i) * r
        s = max(a, block_start)
        e = min(b, block_start + r)
        if e > s:
            out.extend([pb] * (e - s))
    return out[: b - a]


def bits_of(v):
    return tuple((v >> (3 - i)) & 1 for i in range(4))


fails = []

# 1. widths ---------------------------------------------------------
canon = {0: 4, 1: 8, 2: 128, 3: 32768}
for n, w in canon.items():
    if W(n) != w:
        fails.append(("width", n))
print(f"1. produced widths, layers 0-3 == canon: "
      f"{[W(n) for n in range(4)]}; layers 4-5 from the equation: "
      f"{W(4):,}, {W(5):,}")

# 2. exhaustive vs the chain construction, layers 1-3 ---------------
tot = 0
for v in range(16):
    space = ProducedSpace(bits_of(v))
    ref = BitMultiplier(bits_of(v))
    for n in (1, 2, 3):
        pat = ref.materialize(n)
        for p in range(W(n)):
            tot += 1
            if space.bit(n, p) != pat[p]:
                fails.append(("exhaustive", v, n, p))
print(f"2. produced space == chain construction, layers 1-3: "
      f"{tot:,} checks, {len(fails)} failures")

# 3. layers 4-5 vs windowed recursive production ---------------------
rng = random.Random(777)
tot = 0
for n in (4, 5):
    for v in (0b0001, 0b0101, 0b1010, 0b1100, 0b1111):
        space = ProducedSpace(bits_of(v))
        for _ in range(40):
            a = rng.randrange(0, W(n) - 33)
            b = a + 32
            got = space.window(n, a, b)
            want = window_recursive(bits_of(v), n, a, b)
            tot += 32
            if got != want:
                fails.append(("window", v, n, a))
print(f"3. layers 4-5, produced vs window-recursive production: "
      f"{tot:,} bits in 400 windows, {len(fails)} failures")

# 4. demonstration ----------------------------------------------------
print()
print("4. THE PRODUCED SPACE (controller 0101):")
space = ProducedSpace(bits_of(0b0101))
for n in range(0, 6):
    print(f"   layer {n}: width {W(n):>22,}   ones "
          f"{space.ones(n):>22,}")
print(f"   layer 1 pattern: "
      f"{''.join(map(str, space.window(1, 0, 8)))}")
print(f"   layer 2 pattern (blocks of 16): "
      f"{''.join(map(str, space.window(2, 0, 128)))}")
print(f"   layer 5, a window deep in the space (positions "
      f"2^62 .. 2^62+32):")
print(f"     {''.join(map(str, space.window(5, 2**62, 2**62 + 32)))}")
p = 2 ** 62 + 12345
chain = []
q = p
for n in range(5, 0, -1):
    q //= space.fan(n)
    chain.append(q)
print(f"   layer 5 position {p:,}: bit = {space.bit(5, p)} "
      f"(ancestors by layer 4..0: {chain})")

print()
print("RESULT:", "FAIL " + str(fails[:5]) if fails else
      "ALL PASS — the rule alone produces the space: correct at "
      "every established position, coherent through layer 5, "
      "nothing stored but the 4 bits.")
