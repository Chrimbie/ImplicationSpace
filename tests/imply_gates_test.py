"""
Implication Space -- gate emergence test.
Question (Christopher's): do logic gates appear in the method with no
added rules? For each bit position in the implied 32,768-bit layer, take
its on/off across all 16 controller settings as a 16-bit truth table.
Collect the distinct tables. Then check whether AND, OR, XOR, NOT of
controller bits appear among them. Inputs: x = controller bit 0,
y = controller bit 1 (any pair behaves the same by symmetry).
Nothing but the four bits is ever calculated.
"""

WIDTH = 32768


def implied_bit(chosen, p):
    assert 0 <= chosen <= 0xF and 0 <= p < WIDTH
    return (chosen >> (p // 8192)) & 1


def truth_table(fn):
    # fn: (b0,b1,b2,b3) -> 0/1 ; table over K = 0..15, K's bit = table bit K
    t = 0
    for k in range(16):
        b = [(k >> i) & 1 for i in range(4)]
        t |= (fn(*b) & 1) << k
    return t


# Distinct truth tables realized by the method across the whole layer.
seen = set()
for p in range(WIDTH):
    t = 0
    for k in range(16):
        t |= implied_bit(k, p) << k
    seen.add(t)

gates = {
    "AND": lambda b0, b1, b2, b3: b0 & b1,
    "OR": lambda b0, b1, b2, b3: b0 | b1,
    "XOR": lambda b0, b1, b2, b3: b0 ^ b1,
    "NOT": lambda b0, b1, b2, b3: 1 - b0,
}
print(f"distinct truth tables realized by the method: {len(seen)}")
for name, fn in gates.items():
    gt = truth_table(fn)
    print(f"{name}: table={gt:016b} realized={'YES' if gt in seen else 'NO'}")

# What ARE the realized tables, in words?
proj = {truth_table(lambda b0, b1, b2, b3, i=i: (b0, b1, b2, b3)[i]): f"b{i}"
        for i in range(4)}
unexplained = [t for t in seen if t not in proj]
print(f"realized tables that are not plain controller-bit copies: {len(unexplained)}")
print("OVERALL:", "GATES EMERGE" if any(
    truth_table(fn) in seen for fn in gates.values()) else "NO GATES -- projections only")
