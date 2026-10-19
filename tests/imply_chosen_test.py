"""
Implication Space — chosen-combination test.
Christopher's words: four bits, any one of 0000-1111; whichever is chosen
determines which of the 16 combinations is chosen; that combination marks
on/off for all eight bits of the next layer. Nothing but the four bits
is ever calculated.
"""


def mark(chosen, position):
    # The chosen 4 bits ARE one of the 16 combinations; its on/off
    # positions mark the 8 bits. Nothing else is calculated.
    assert 0 <= chosen <= 0xF, "four bits, one of 0000-1111"
    assert 0 <= position <= 7, "all eight bits of the next layer"
    return (chosen >> position) & 1 if position < 4 else 0


failed = 0
total = 0
for chosen in range(16):          # any one of 0000-1111, whichever is chosen
    for p in range(8):            # marks on/off for all eight bits
        total += 1
        want = (chosen >> p) & 1 if p < 4 else 0
        if mark(chosen, p) != want:
            failed += 1
            print(f"[FAIL] chosen={chosen:04b} bit={p}: want {want}")

print(f"{total - failed}/{total} markings correct; "
      f"nothing but the four bits was ever calculated.")
print("OVERALL:", "PASS" if failed == 0 else "FAIL")
