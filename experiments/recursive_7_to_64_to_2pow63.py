"""Recursive implication experiment: 7 physical bits -> 64 -> 2**63 implied bits.

Uses the same parity rule at each level:
    bit(controller, position) = parity(controller & (position + 1))
The 64-bit middle layer is never stored in direct_bit().
This is a deterministic mapping, not independent 2**63-bit memory.
"""
import random

WIDTH = 1 << 63

def first_bit(controller: int, position: int) -> int:
    assert 0 <= controller < 128 and 0 <= position < 64
    return (controller & (position + 1)).bit_count() & 1

def direct_bit(controller: int, position: int) -> int:
    """Query one bit of the huge layer without constructing the 64-bit layer."""
    assert 0 <= controller < 128 and 0 <= position < WIDTH
    mask = position + 1
    parity = 0
    for i in range(64):
        if (mask >> i) & 1:
            parity ^= first_bit(controller, i)
    return parity

def layer_by_layer_bit(controller: int, position: int) -> int:
    """Independent reference: materialize only the 64-bit intermediate."""
    assert 0 <= controller < 128 and 0 <= position < WIDTH
    middle = sum(first_bit(controller, i) << i for i in range(64))
    return (middle & (position + 1)).bit_count() & 1

def test_recursive_mapping() -> None:
    # 1,000 low positions and 12 high positions per controller: 1,012 each.
    positions = list(range(1000)) + [WIDTH - 1 - i for i in range(12)]
    mismatches = [
        (controller, position)
        for controller in range(128)
        for position in positions
        if direct_bit(controller, position) != layer_by_layer_bit(controller, position)
    ]
    assert not mismatches, mismatches[:10]
    assert [direct_bit(127, p) for p in (0, 1, 2, 63, 64, WIDTH - 1)] == [1, 1, 0, 1, 0, 1]
    print(f"PASS: {128 * len(positions):,} comparisons, 0 mismatches")
    print(f"Implied layer width: {WIDTH:,} bits")

if __name__ == "__main__":
    test_recursive_mapping()
