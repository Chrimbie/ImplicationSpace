"""Experimental 7-bit controller -> 64-bit implied arithmetic, queried one bit at a time.

Separate from the repository's established 4-bit fan-out model.
No 64-bit operand or result is materialized in query_result_bit().
The host still stores controller, bit index, carry, and the fixed algorithm.
"""
MASK = (1 << 64) - 1

def implied_bit(controller: int, index: int) -> int:
    if not (0 <= controller < 128 and 0 <= index < 64):
        raise ValueError("controller must be 0..127 and index 0..63")
    return (controller & (index + 1)).bit_count() & 1

def query_result_bit(controller: int, index: int) -> int:
    """Bit index of implied(controller) * (controller & 15), mod 2**64."""
    if not (0 <= controller < 128 and 0 <= index < 64):
        raise ValueError("controller must be 0..127 and index 0..63")
    multiplier = controller & 15
    carry = 0
    for position in range(index + 1):
        total = multiplier * implied_bit(controller, position) + carry
        carry = total // 2
    return total & 1

def reference(controller: int) -> int:
    operand = sum(implied_bit(controller, i) << i for i in range(64))
    return (operand * (controller & 15)) & MASK

def test_exhaustive() -> None:
    for controller in range(128):
        expected = reference(controller)
        for index in range(64):
            assert query_result_bit(controller, index) == ((expected >> index) & 1), (
                controller, index
            )

if __name__ == "__main__":
    test_exhaustive()
    print("PASS: 8,192/8,192 on-demand bit queries (128 states x 64 bits)")
