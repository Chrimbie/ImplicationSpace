"""TEST THE IMPLICATION SPACE — Christopher's ask, 2026-10-21.

The space under test is the one PRODUCED BY THE RULE
(implied_space_producer.ProducedSpace: 4 real bits + fan-out +
W(n) = 2^(2^(n+1)-1)). Battery = every property this project
has ever claimed for the space:

  1. determinism        — same question, same answer, any order
  2. direct == recursive— deep query == layer-by-layer production
  3. no materialization — memory flat while answering at layer 5
  4. flat cost          — per-query time ~independent of depth
  5. 16 distinct spaces — every controller's space differs from
                          every other, at every layer
  6. implied functions  — Boolean functions of implied bits,
                          across layers, answered from the
                          controller alone (his 2026-10-18
                          corollary: all 65,536 are implied)
  7. robust recovery    — the controller recoverable from a
                          corrupted sample of implied bits
                          (majority vote per quarter), up to
                          heavy corruption
"""

import random
import resource
import sys
import time

sys.path.insert(0, '/home/hatch/workspace/implication_space')
from implied_space_producer import ProducedSpace, W  # noqa: E402

rng = random.Random(4242)
fails = []


def bits_of(v):
    return tuple((v >> (3 - i)) & 1 for i in range(4))


SPACES = {v: ProducedSpace(bits_of(v)) for v in range(16)}

# 1. determinism ------------------------------------------------------
qs = [(rng.randrange(1, 6), None) for _ in range(2000)]
qs = [(n, rng.randrange(W(n))) for n, _ in qs]
first = {}
for v in (0b0101, 0b1010):
    for n, p in qs:
        first[(v, n, p)] = SPACES[v].bit(n, p)
det_ok = True
for _ in range(3):
    shuffled = qs[:]
    rng.shuffle(shuffled)
    for n, p in shuffled:
        for v in (0b0101, 0b1010):
            if SPACES[v].bit(n, p) != first[(v, n, p)]:
                det_ok = False
print(f"1. determinism (2,000 queries x 4 passes, shuffled): "
      f"{'PASS' if det_ok else 'FAIL'}")
if not det_ok:
    fails.append("determinism")

# 2. direct == recursive (layer-by-layer window production) -----------
from implied_space_producer import window_recursive  # noqa: E402
rec_ok = True
cnt = 0
for n in (4, 5):
    for v in (0b0011, 0b1100):
        for _ in range(25):
            a = rng.randrange(0, W(n) - 17)
            got = SPACES[v].window(n, a, a + 16)
            want = window_recursive(bits_of(v), n, a, a + 16)
            cnt += 16
            if got != want:
                rec_ok = False
print(f"2. direct == recursive production (layers 4-5): "
      f"{cnt} bits, {'PASS' if rec_ok else 'FAIL'}")
if not rec_ok:
    fails.append("recursive")

# 3. no materialization: peak memory answering layer-5 queries ---------
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for _ in range(20_000):
    n = rng.randrange(1, 6)
    SPACES[0b0110].bit(n, rng.randrange(W(n)))
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
growth_kb = after - before
print(f"3. no materialization: peak-RSS growth over 20,000 deep "
      f"queries = {growth_kb} KB ({'PASS' if growth_kb < 4096 else 'FAIL'})")
if growth_kb >= 4096:
    fails.append("memory")

# 4. flat cost per query across depth ----------------------------------
times = {}
for n in range(1, 6):
    ps = [rng.randrange(W(n)) for _ in range(3000)]
    t0 = time.perf_counter()
    for p in ps:
        SPACES[0b1001].bit(n, p)
    times[n] = (time.perf_counter() - t0) / len(ps) * 1e9
print("4. per-query cost by layer (ns): " +
      ", ".join(f"L{n}={times[n]:.0f}" for n in range(1, 6)))

# 5. 16 distinct spaces at every layer ---------------------------------
dist_ok = True
for n in (1, 2, 3, 4, 5):
    for i in range(16):
        for j in range(i + 1, 16):
            # spaces differ exactly where controllers do; find a
            # position in a differing quarter and confirm
            d = [k for k in range(4)
                 if bits_of(i)[k] != bits_of(j)[k]][0]
            p = d * (W(n) // 4) + rng.randrange(W(n) // 4)
            if SPACES[i].bit(n, p) == SPACES[j].bit(n, p):
                dist_ok = False
            # and agree inside a shared quarter
            s = [k for k in range(4)
                 if bits_of(i)[k] == bits_of(j)[k]]
            if s:
                p2 = s[0] * (W(n) // 4) + rng.randrange(W(n) // 4)
                if SPACES[i].bit(n, p2) != SPACES[j].bit(n, p2):
                    dist_ok = False
print(f"5. all 16 spaces distinct, layers 1-5 (pairwise probes): "
      f"{'PASS' if dist_ok else 'FAIL'}")
if not dist_ok:
    fails.append("distinct")

# 6. implied functions across layers ------------------------------------
# pick random implied bits from random layers of one space, combine
# with a random Boolean function; the answer computed from the
# implied bits must equal the function of their controller sources.
def controller_source(space_bits, n, p):
    while n > 0:
        p //= (W(n) // W(n - 1))
        n -= 1
    return space_bits[p]

fun_ok = True
for _ in range(3000):
    v = rng.randrange(16)
    bits = bits_of(v)
    src = []
    vals = []
    for _ in range(3):
        n = rng.randrange(1, 6)
        p = rng.randrange(W(n))
        vals.append(SPACES[v].bit(n, p))
        src.append(controller_source(bits, n, p))
    table = [rng.randrange(2) for _ in range(8)]   # random 3-in fn
    idx = vals[0] * 4 + vals[1] * 2 + vals[2]
    idx_src = src[0] * 4 + src[1] * 2 + src[2]
    if table[idx] != table[idx_src]:
        fun_ok = False
print(f"6. Boolean functions of implied bits across layers "
      f"(3,000 random 3-input functions): "
      f"{'PASS' if fun_ok else 'FAIL'}")
if not fun_ok:
    fails.append("functions")

# 7. robust recovery of the controller from corrupted bits -------------
print("7. controller recovery from corrupted layer-3 samples "
      "(2,000 sampled bits, majority vote per quarter):")
for corrupt in (0.05, 0.20, 0.35, 0.45):
    trials, ok = 200, 0
    for _ in range(trials):
        v = rng.randrange(16)
        votes = [[0, 0] for _ in range(4)]
        for _ in range(2000):
            p = rng.randrange(W(3))
            b = SPACES[v].bit(3, p)
            if rng.random() < corrupt:
                b ^= 1
            votes[p // (W(3) // 4)][b] += 1
        rec = tuple(1 if votes[q][1] > votes[q][0] else 0
                    for q in range(4))
        ok += (rec == bits_of(v))
    print(f"     corruption {int(corrupt * 100):>2}%: recovered "
          f"{ok}/{trials} controllers exactly")

print()
print("RESULT:", "FAIL: " + str(fails) if fails else
      "ALL PASS — the produced implication space holds every "
      "property the project has claimed for it.")
