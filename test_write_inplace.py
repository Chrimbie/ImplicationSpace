"""test_write_inplace.py - the hot-region write path.

A block rewritten in place must (a) read back as the latest
content, (b) not grow the file, (c) coexist with normal
append writes."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_blockstore import BlockStore

P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "test_inplace.evb2")
if os.path.exists(P):
    os.remove(P)
s = BlockStore.create(P, 1 << 32)
data_a = bytes(range(256)) * 4096        # exactly 1 MiB
s.write(0, data_a)
size1 = s.stored_bytes()
data_b = b"\xAB" * len(data_a)
for _ in range(50):
    s.write_inplace(0, data_b)
size2 = s.stored_bytes()
assert s.read(0, len(data_b)) == data_b, "content after rewrites"
assert size2 == size1, f"file grew: {size1} -> {size2}"
# partial in-place: patch 100 bytes inside the block
s.write_inplace(500, b"\x00" * 100)
want = bytearray(data_b)
want[500:600] = b"\x00" * 100
assert s.read(0, len(data_b)) == bytes(want), "partial patch"
# append path still works elsewhere
s.write(1 << 21, data_a)
assert s.read(1 << 21, len(data_a)) == data_a, "append elsewhere"
s.close()
s2 = BlockStore.open(P)
assert s2.read(0, len(data_b)) == bytes(want), "reopen"
s2.close()
os.remove(P)
print("write_inplace: ALL PASS "
      f"(file {size1:,} B before and after 50 rewrites)")
