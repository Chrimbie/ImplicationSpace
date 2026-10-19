"""
Implication Space — bit-doubling test.
Christopher's rule: the chosen 4-bit combination marks the 8 bits of the
next layer by fanning out -- every bit becomes a pair. Input bit i marks
output bits 2i and 2i+1.
  0001 -> 00000011
  0101 -> 00110011
  1010 -> 11001100
  0010 -> 00001100
Nothing but the four bits is ever calculated.
"""


def mark(chosen, position):
    assert 0 <= chosen <= 0xF, "four bits, one of 0000-1111"
    assert 0 <= position <= 7, "all eight bits of the next layer"
    return (chosen >> (position // 2)) & 1


failed = 0
total = 0
for chosen in range(16):
    for p in range(8):
        total += 1
        want = (chosen >> (p // 2)) & 1
        if mark(chosen, p) != want:
            failed += 1
            print(f"[FAIL] chosen={chosen:04b} bit={p}: want {want}")

# Christopher's four examples, spelled out.
examples = {
    0b0001: 0b00000011,
    0b0101: 0b00110011,
    0b1010: 0b11001100,
    0b0010: 0b00001100,
}
for chosen, want_bits in examples.items():
    got = sum(mark(chosen, p) << p for p in range(8))
    total += 1
    if got != want_bits:
        failed += 1
        print(f"[FAIL] example {chosen:04b}: got {got:08b}, want {want_bits:08b}")

print(f"{total - failed}/{total} markings correct; "
      f"nothing but the four bits was ever calculated.")
print("OVERALL:", "PASS" if failed == 0 else "FAIL")
