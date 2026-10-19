"""
Implication Space -- full chain test.
Christopher's rules:
  - 4 bits, any one of 0000-1111; whichever is chosen is the combination.
  - 4 -> 8: each bit fans out to a pair (bit i -> bits 2i, 2i+1).
  - 8 -> 128: each of the 8 bits fans out to a block of 16
    (bit j -> bits 16j .. 16j+15).
Nothing but the four bits is ever calculated; wider layers are implied.
"""


def mark_8(chosen, p):
    assert 0 <= chosen <= 0xF and 0 <= p <= 7
    return (chosen >> (p // 2)) & 1


def mark_128(chosen, p):
    assert 0 <= chosen <= 0xF and 0 <= p <= 127
    return mark_8(chosen, p // 16)


failed = 0
total = 0
for chosen in range(16):
    for p in range(128):
        total += 1
        want = (chosen >> (p // 32)) & 1
        if mark_128(chosen, p) != want:
            failed += 1
            print(f"[FAIL] chosen={chosen:04b} pos={p}: want {want}")

# Spelled-out example: 0101 -> 8-bit 00110011 -> 128-bit blocks
# 0,1 on; 2,3 off; 4,5 on; 6,7 off (16 bits per block).
for p, want in [(0, 1), (31, 1), (32, 0), (63, 0),
                (64, 1), (95, 1), (96, 0), (127, 0)]:
    total += 1
    if mark_128(0b0101, p) != want:
        failed += 1
        print(f"[FAIL] example 0101 pos={p}: want {want}")

print(f"{total - failed}/{total} markings correct; "
      f"nothing but the four bits was ever calculated.")
print("OVERALL:", "PASS" if failed == 0 else "FAIL")
