"""
Implication Space as a memory addressing layer — honest systems test.
Setup: 16 entries addressed by 4-bit keys. Implication addressing stores
nothing but the key: the entry's "address" is its implied 32768-bit pattern
(bit p = key bit (p // 8192)). Retrieval under corruption: flip each address
bit independently with probability p, recover the key by majority vote per
8192-block, look up the entry.
Baseline: plain 4-bit keys, same corruption, no redundancy (any flip breaks it).
Metric: fraction of correct retrievals over trials.
"""
import random

WIDTH = 32768
BLOCK = 8192


def implied_bit(key, p):
    return (key >> (p // BLOCK)) & 1


def retrieve_implication(corrupted):
    """Majority vote per block -> key."""
    key = 0
    for b in range(4):
        ones = sum(corrupted[b * BLOCK + i] for i in range(BLOCK))
        if ones > BLOCK // 2:
            key |= (1 << b)
    return key


def trial(p, n=400):
    impl_ok = base_ok = 0
    for _ in range(n):
        key = random.randrange(16)
        # implication addressing: corrupt the full implied pattern
        corrupted = [implied_bit(key, i) ^ (random.random() < p)
                     for i in range(WIDTH)]
        if retrieve_implication(corrupted) == key:
            impl_ok += 1
        # baseline: corrupt the 4 key bits directly
        base_key = 0
        for b in range(4):
            bit = ((key >> b) & 1) ^ (random.random() < p)
            base_key |= bit << b
        if base_key == key:
            base_ok += 1
    return impl_ok / n, base_ok / n


random.seed(7)
print(f"{'corrupt':>7} | {'impl-space':>10} | {'plain 4-bit':>11}")
for p in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45):
    a, b = trial(p)
    print(f"{p:>7.2f} | {a:>10.3f} | {b:>11.3f}")
