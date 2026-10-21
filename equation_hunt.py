"""Hunting the equation — Christopher's ask, 2026-10-21:
"Let's figure out that equation. Do you think UCE could figure it
out for us?"

Framing on record: UCE is a mirror, not an oracle. It cannot
originate the equation. What it CAN do: (1) hold the canon as a
structure of constraints; (2) expose, role by role, where each
candidate equation agrees with the canon and where it commits beyond
it. The candidate below was derived by collapsing the solidified
fan-out chain (bit_multiplier_v0.py): at every established width,
each real bit's descendants form exactly one quarter of the layer,
so the whole chain collapses into a single closed form. It is a
RECONSTRUCTION — verified equivalent to the canon on every
established case — not a recovery of Christopher's original
formulation. Whether it is his equation is his to say.

CANDIDATE (the quarter equation):
  For controller value v (bits b3 b2 b1 b0, left to right), layer
  width W, position j in [0, W):
      q = floor(4j / W)            (which quarter of the layer)
      B(v, W, j) = floor(v / 2^(3-q)) mod 2     (the q-th bit of v)
  In recursive-ID terms: a position's bit is the controller bit
  chosen at the FIRST step of its path; deeper steps refine the
  address within the quarter and never change the bit.
"""

import sys

sys.path.insert(0, '/home/hatch/workspace/implication_space')
sys.path.insert(0, '/home/hatch/workspace/custom-skills/local-uce/bin')

from bit_multiplier_v0 import BitMultiplier  # noqa: E402
from uce import ConceptEngine  # noqa: E402

LINES = []


def log(s=""):
    print(s, flush=True)
    LINES.append(s)


def B(v, W, j):
    q = (4 * j) // W
    return (v >> (3 - q)) & 1


def bits_of(v):
    return tuple((v >> (3 - i)) & 1 for i in range(4))


# ------------------------------------------------------- verify -------
log("=" * 72)
log("PART 1 — the quarter equation vs the solidified canon")
log("=" * 72)

# his four examples, 4 -> 8, verbatim from the record
examples = {0b0001: "00000011", 0b0101: "00110011",
            0b1010: "11001100", 0b0010: "00001100"}
ok = True
for v, expected in examples.items():
    got = "".join(str(B(v, 8, j)) for j in range(8))
    ok &= (got == expected)
    log(f"  {v:04b} -> {got}  (record: {expected})  "
        f"{'OK' if got == expected else 'MISMATCH'}")
log(f"his four examples: {'PASS' if ok else 'FAIL'}")

# exhaustive: all controllers, all positions, widths 8 / 128 / 32768,
# equation vs the chain construction (materialize)
total = bad = 0
for v in range(16):
    m = BitMultiplier(bits_of(v))
    for stage, W in ((1, 8), (2, 128), (3, 32768)):
        pat = m.materialize(stage)
        for j in range(W):
            total += 1
            if B(v, W, j) != pat[j]:
                bad += 1
log(f"exhaustive vs chain construction: {total:,} checks, "
    f"{bad} mismatches -> {'PASS' if bad == 0 else 'FAIL'}")

# ordering: patterns at each width are monotonic in v (K-th order
# preserved as a correspondence, per the record)
mono = True
for W in (8, 128, 32768):
    pats = [int("".join(str(B(v, W, j)) for j in range(W)), 2)
            for v in range(16)]
    mono &= all(pats[i] < pats[i + 1] for i in range(15))
log(f"K-th ordering (monotonic correspondence) at all widths: "
    f"{'PASS' if mono else 'FAIL'}")

# ladder extension (layers VAR widths W = 4 * 2^L): equation vs
# recursive provenance (halve the position per doubling layer)
def provenance_bit(v, L, j):
    # position j at width 4*2^L descends through L doublings
    a = j >> L          # ancestor index among the 4 real bits... in
    # units of real bits: j // 2^L gives the real-bit index directly
    return bits_of(v)[a]

lad_ok = True
for L in (1, 5, 13, 20, 40):
    W = 4 * 2 ** L
    for j in (0, 1, W // 4 - 1, W // 4, W // 2, W - 1, W // 3):
        if B(0b1010, W, j) != provenance_bit(0b1010, L, j):
            lad_ok = False
log(f"ladder widths (incl. W = 4*2^40) spot checks vs recursive "
    f"provenance: {'PASS' if lad_ok else 'FAIL'}")
log("")
log("NOTE: the equation COMMITS where the canon leaves open — it gives")
log("the same quarter law at every width, including unestablished")
log("ones. Equivalence is proven only on the established widths;")
log("the extension is the equation's own claim, not the record's.")

# ---------------------------------------------------------- UCE -------
log("")
log("=" * 72)
log("PART 2 — UCE: candidates against the canon, role by role")
log("=" * 72)

e = ConceptEngine()
e.create_structure("implication_equation")
for role in ("marking", "ordering", "locus", "recursion",
             "beyond_established"):
    e.add_role("implication_equation", role)
e.add_relation("implication_equation", "fixes", "marking", "ordering")
e.add_relation("implication_equation", "reads_only", "locus", "marking")

for d in ("record", "candidates"):
    e.create_domain(d)

e.instantiate("implication_equation", "record", "canon")
e.bind("canon", "marking",
       "the chosen combination fans out: every bit becomes a block")
e.bind("canon", "ordering",
       "combination order corresponds to pattern order")
e.bind("canon", "locus", "only the four bits are ever calculated")
e.bind("canon", "recursion", "same rule at every layer, turtles all the way up")
e.bind("canon", "beyond_established", "implied, not specified")

e.instantiate("implication_equation", "candidates", "quarter_equation")
e.bind("quarter_equation", "marking",
       "the chosen combination fans out: every bit becomes a block")
e.bind("quarter_equation", "ordering",
       "combination order corresponds to pattern order")
e.bind("quarter_equation", "locus",
       "only the four bits are ever calculated")
e.bind("quarter_equation", "recursion",
       "one closed form for all layers; the recursion is collapsed into it")
e.bind("quarter_equation", "beyond_established",
       "commits: the same quarter law at every width, established or not")

e.instantiate("implication_equation", "candidates", "identity_embedding")
e.bind("identity_embedding", "marking",
       "the value itself, zero-extended: 0001 -> 00000001")
e.bind("identity_embedding", "ordering",
       "counting-order identity: the k-th combination IS the k-th pattern")
e.bind("identity_embedding", "locus",
       "only the four bits are ever calculated")
e.bind("identity_embedding", "recursion",
       "same rule at every layer, turtles all the way up")
e.bind("identity_embedding", "beyond_established", "implied, not specified")

e.instantiate("implication_equation", "candidates", "tile_alternate")
e.bind("tile_alternate", "marking",
       "previous layer tiled across the next, alternate tiles flipped")
e.bind("tile_alternate", "ordering", "not established")
e.bind("tile_alternate", "locus",
       "only the four bits are ever calculated")
e.bind("tile_alternate", "recursion",
       "same rule at every layer, turtles all the way up")
e.bind("tile_alternate", "beyond_established", "not established")

e.instantiate("implication_equation", "candidates", "chatgpt_formula")
e.bind("chatgpt_formula", "marking", "not the fan rule: a formula-first account")
e.bind("chatgpt_formula", "ordering", "dropped: no K-th ordering in the account")
e.bind("chatgpt_formula", "locus", "unknown")
e.bind("chatgpt_formula", "recursion", "next width = 2^current / 2")
e.bind("chatgpt_formula", "beyond_established",
       "commits: the fourth width is 2^127 (rejected on record: 'it's supposed to be the 32,768')")

for cand in ("quarter_equation", "identity_embedding",
             "tile_alternate", "chatgpt_formula"):
    log("")
    log(e.compare_instances("canon", cand))

with open('equation_hunt_log.txt', 'w') as f:
    f.write("\n".join(LINES) + "\n")
