"""
explore_addressing.py — decoder/addressing view of the fan-out rule (DERIVED).

Canon: 4-bit controller K (0..15); implied bit p at the 32768 layer =
controller bit (p // 8192). Nothing but the four bits is ever calculated.
"""
import time

W = 32768
Q = 8192  # quarter size: copies per controller bit


def bit(chosen, p):
    return (chosen >> (p // Q)) & 1


def pattern(chosen):
    return [bit(chosen, p) for p in range(W)]


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


pats = [pattern(k) for k in range(16)]

# 1. Linearity over GF(2): pattern(a^b) == pattern(a) XOR pattern(b)?
lin_ok = all(
    all((pats[a][p] ^ pats[b][p]) == pats[a ^ b][p] for p in range(W))
    for a in range(16) for b in range(16)
)
print(f"[DERIVED] mapping K->pattern is linear over GF(2): {lin_ok}")

# 2. Weight / distance structure.
weights = sorted(set(sum(p) for p in pats))
print(f"[DERIVED] distinct pattern weights: {weights}")
dists = {}
for a in range(16):
    for b in range(a + 1, 16):
        d = hamming(pats[a], pats[b])
        dists[d] = dists.get(d, 0) + 1
print(f"[DERIVED] pairwise Hamming distance spectrum "
      f"(distance: pair-count): {dict(sorted(dists.items()))}")
dmin = min(dists)
print(f"[DERIVED] minimum distance d_min = {dmin} "
      f"(code parameters: [n,k,d] = [{W},4,{dmin}])")

# 3. Adjacent addresses: K vs K+1 distance.
adj = [hamming(pats[k], pats[(k + 1) % 16]) for k in range(16)]
print(f"[DERIVED] adjacent-address distances (K vs K+1): {adj}")

# 4. Per-bit query cost: time single-bit queries (the only operation).
for n in (10**3, 10**5):
    t0 = time.perf_counter()
    for i in range(n):
        bit(0b0101, (i * 2654435761) % W)
    dt = (time.perf_counter() - t0) / n * 1e9
    print(f"[DERIVED] per-bit query cost: ~{dt:.0f} ns "
          f"(width-independent: one shift+mask, no layer built)")

# 5. Nesting: each quarter is constant; pattern determined by 4 quarter values.
nested = all(all(pats[k][p] == pats[k][q * Q] for p in range(q * Q, (q + 1) * Q))
             for k in range(16) for q in range(4))
print(f"[DERIVED] every 8192-bit quarter is constant: {nested}")
print(f"[DERIVED] virtualization ratio: {W} answerable bits from 4 stored "
      f"bits = {W // 4}:1")
