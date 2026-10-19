"""
Implication Space -- 32,768-bit layer test.
Christopher's rules, same fan-out shape:
  - 4 -> 8:   each bit -> pair (bit i -> bits 2i, 2i+1)
  - 8 -> 128: each bit -> block of 16 (bit j -> bits 16j..16j+15)
  - 128 -> 32768: each bit -> block of 256 (bit k -> bits 256k..256k+255)
Nothing but the four bits is ever calculated; wider layers are implied.
"""


def mark(chosen, p):
    assert 0 <= chosen <= 0xF, "four bits, one of 0000-1111"
    assert 0 <= p <= 32767, "all 32,768 bits of the next layer"
    return (chosen >> (p // 8192)) & 1


failed = 0
total = 0
for chosen in range(16):
    for p in range(32768):
        total += 1
        if mark(chosen, p) != (chosen >> (p // 8192)) & 1:
            failed += 1
            print(f"[FAIL] chosen={chosen:04b} pos={p}")

# Spelled-out example: 0101 -> quarters on, off, on, off (8192 bits each).
for p, want in [(0, 1), (8191, 1), (8192, 0), (16383, 0),
                (16384, 1), (24575, 1), (24576, 0), (32767, 0)]:
    total += 1
    if mark(0b0101, p) != want:
        failed += 1
        print(f"[FAIL] example 0101 pos={p}: want {want}")

print(f"{total - failed}/{total} markings correct; "
      f"nothing but the four bits was ever calculated.")
print("OVERALL:", "PASS" if failed == 0 else "FAIL")
