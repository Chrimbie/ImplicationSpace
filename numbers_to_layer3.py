"""Every established number up to layer 3 — Christopher's working
sheet for the equation (2026-10-21). All values computed from the
solidified canon (bit_multiplier_v0.BitMultiplier); nothing added.
"""

import sys
from math import log2

sys.path.insert(0, '/home/hatch/workspace/implication_space')
from bit_multiplier_v0 import BitMultiplier  # noqa: E402

WIDTHS = (4, 8, 128, 32768)


def bits_of(v):
    return tuple((v >> (3 - i)) & 1 for i in range(4))


print("PER-LAYER NUMBERS")
print(f"{'layer':<6}{'width W':<10}{'W=2^e':<7}{'fan r':<8}"
      f"{'copies/real':<13}{'blocks':<8}{'block size':<11}"
      f"{'possible patterns':<18}{'realized'}")
prev = None
for s, W in enumerate(WIDTHS):
    fan = W // prev if prev else None
    copies = W // 4
    blocks = prev if prev else 4
    print(f"{s:<6}{W:<10}{int(log2(W)):<7}"
          f"{(str(fan) if fan else '-'):<8}{copies:<13}{blocks:<8}"
          f"{(str(fan) if fan else '-'):<11}{'2^' + str(W):<18}16")
    prev = W

print()
print("SEQUENCES (layer 0 -> 3)")
print("  widths:            4, 8, 128, 32768")
print("  exponents e (2^e): 2, 3, 7, 15")
print("  exponent jumps:    -, 1, 4, 8")
print("  fan factors r:     -, 2, 16, 256   (= 2^1, 2^4, 2^8)")
print("  copies per real:   1, 2, 32, 8192  (= 2^0, 2^1, 2^5, 2^13)")
print("  block sizes:       -, 2, 16, 256   (one block per bit of "
      "the layer above)")

print()
print("LAYER 1 — all 16 patterns (width 8)")
for v in range(16):
    pat = BitMultiplier(bits_of(v)).materialize(1)
    print(f"  {v:04b} (K={v + 1:>2}) -> {''.join(map(str, pat))}"
          f"   ones={sum(pat)}")

print()
print("ONES COUNT per controller, by layer (law: popcount(v) x W/4)")
print(f"  {'v':<6}{'pop':<5}{'L0':<5}{'L1':<5}{'L2':<6}{'L3'}")
for v in range(16):
    m = BitMultiplier(bits_of(v))
    ones = [sum(m.materialize(s)) for s in range(4)]
    print(f"  {v:04b}  {sum(bits_of(v)):<5}{ones[0]:<5}{ones[1]:<5}"
          f"{ones[2]:<6}{ones[3]}")

print()
print("DISTANCES between the 16 patterns, by layer")
for s in (1, 2, 3):
    pats = [BitMultiplier(bits_of(v)).materialize(s) for v in range(16)]
    ds = [sum(a != b for a, b in zip(pats[i], pats[j]))
          for i in range(16) for j in range(i + 1, 16)]
    print(f"  layer {s}: min={min(ds)}  max={max(ds)}  "
          f"(min = W/4 = {WIDTHS[s] // 4})")

print()
print("LAYER 2 structure (width 128 = 8 blocks of 16, block = the "
      "layer-1 bit repeated)")
for v in (0b0001, 0b0101, 0b1010, 0b1111):
    l1 = BitMultiplier(bits_of(v)).materialize(1)
    print(f"  {v:04b}: blocks = {''.join(map(str, l1))} "
          f"(each shown bit = a block of 16 identical bits)")

print()
print("LAYER 3 structure (width 32768 = 128 blocks of 256)")
print("  block pattern = the controller's layer-2 pattern; each of "
      "its 128 bits is a block of 256 identical bits.")
print("  total bits across all 16 layer-3 patterns: "
      f"{16 * 32768:,}")
