"""single_bit_exponentiation.py - Christopher's ask, 2026-10-22:
"let's try single bit exponentiation" (he had thought 4 was the
floor for the seed).

Construction under test: HIS, unchanged - the fan-out expansion
from expansion_rule_test.py. A seed of s real bits; at any wider
layer, each seed bit is a block and position p takes the seed
bit of its block:  bit(v, W, p) = seed_bit[p // (W / s)].
Widths from HIS growth equation generalized by its own terms:
g (generating bits) starts at s and doubles each layer;
W = 2^(g - 1) = combinations / 2. For s = 4 this must reproduce
canon exactly: widths 4, 8, 128, 32768 and patterns identical
to the BitMultiplier chain (regression gate - if s=4 fails,
the generalization is wrong, not the canon).

Checks per seed width s in {1, 2, 4}:
  widths        - layer widths follow 2^(g-1), g doubling
  chain==closed - materialized chain (each bit becomes a block,
                  step by step) == closed block form, exhaustive
                  while W <= 32768
  ones law      - ones(v, layer) = popcount(v) * W / s
  distinct      - number of distinct patterns over all 2^s seeds
  degeneracy    - does a layer's pattern vary across positions?
"""
import sys

sys.path.insert(0, '/home/hatch/workspace/implication_space')
from bit_multiplier_v0 import BitMultiplier  # noqa: E402


def widths_for(s, layers):
    """Seed layer W=s. Each next width = 2^(g-1) where g is the
    generating count of the layer below (g starts at s and
    doubles per layer) - canon check s=4: 4, 8, 128, 32768."""
    ws = [s]
    g = s
    for _ in range(layers):
        ws.append(2 ** (g - 1))
        g *= 2
    return ws


def seed_bits(v, s):
    return tuple((v >> (s - 1 - i)) & 1 for i in range(s))


def materialize_chain(v, s, ws):
    """Step by step: each bit becomes a block of the fan factor."""
    pat = list(seed_bits(v, s))
    for k in range(1, len(ws)):
        fan = ws[k] // ws[k - 1]
        pat = [b for b in pat for _ in range(fan)]
    return pat


def closed_bit(v, s, W, p):
    return seed_bits(v, s)[p // (W // s)]


def run(s):
    layers = 0
    while widths_for(s, layers + 1)[-1] <= 32768:
        layers += 1
    ws = widths_for(s, layers)
    print(f"--- seed width s = {s}: widths {ws}")
    seeds = range(2 ** s)
    ok = True

    # chain == closed form, exhaustive over materialized layers
    tot = 0
    for v in seeds:
        for li, W in enumerate(ws):
            sub = materialize_chain(v, s, ws[:li + 1])
            for p in range(W):
                tot += 1
                if sub[p] != closed_bit(v, s, W, p):
                    ok = False
                    print(f"  FAIL chain!=closed v={v} W={W} p={p}")
    print(f"  chain == closed form: {tot:,} checks "
          f"{'PASS' if ok else 'FAIL'}")

    # ones law on the widest materialized layer
    W = ws[-1]
    ones_ok = True
    for v in seeds:
        pat = materialize_chain(v, s, ws)
        if sum(pat) != sum(seed_bits(v, s)) * (W // s):
            ones_ok = False
    print(f"  ones law popcount x W/s at W={W:,}: "
          f"{'PASS' if ones_ok else 'FAIL'}")

    # distinct patterns + within-pattern variation, widest layer
    pats = {tuple(materialize_chain(v, s, ws)) for v in seeds}
    varies = [len(set(p)) > 1 for p in pats]
    print(f"  distinct patterns at W={W:,}: {len(pats)} "
          f"of {2 ** s} seeds")
    print(f"  patterns that vary across positions: "
          f"{sum(varies)} of {len(pats)}")
    return ws


def main():
    # regression gate: s=4 must equal the canon chain exactly
    print("--- regression: s=4 vs BitMultiplier (canon)")
    ok = True
    ws4 = widths_for(4, 3)
    assert ws4 == [4, 8, 128, 32768], ws4
    for v in range(16):
        m = BitMultiplier(seed_bits(v, 4))
        for li, n in enumerate((1, 2, 3), start=1):
            mine = materialize_chain(v, 4, ws4[:li + 1])
            if list(m.materialize(n)) != mine:
                ok = False
                print(f"  FAIL v={v} layer={n}")
    print(f"  s=4 widths + patterns == canon: "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok, "generalization breaks canon - stop"

    run(4)
    run(2)
    run(1)
    print()
    print("Widths from a single generating bit (s=1): "
          "1, 1, 2, 8, 128, 32768, 2^31, ... - the equation's "
          "ladder, reached from g=1 (first fan factor is 1: "
          "the seed implies a layer of its own width).")


if __name__ == "__main__":
    main()
