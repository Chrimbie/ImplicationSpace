"""USAGE TEST — is implied space usable?

The same jobs, done the way ordinary programs do them, through
three ordinary interfaces — none of which knows what a
controller is:

  1. FILE style    — open/seek/read/write on a 2 GB implied file
  2. ARRAY style   — numpy-style slicing on a 5 GB implied array,
                     plus a 2-D matrix view
  3. RAW style     — the buffer API directly (the stranger way)
  4. ANOTHER PROGRAM — a child process opens the file-style
                     buffer and reads the data back itself

Every check is byte-verified. The last lines give the total
accounting: logical bytes in use vs bytes actually stored.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_vram import ImpliedBuffer, ImpliedFile, ImpliedArray

D = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, ok):
    results.append(ok)
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")


print("1. FILE style — a 2 GB implied file")
p1 = os.path.join(D, "usage_file.evb")
if os.path.exists(p1):
    os.remove(p1)
REC = b"a record stored at the one-gigabyte mark"
with ImpliedFile.create(p1, 2 * 10 ** 9) as f:
    f.write(b"IMPLIED-FILE-HEADER v1")
    f.seek(10 ** 9)
    f.write(REC)
    f.seek(0)
    check("header reads back", f.read(22) == b"IMPLIED-FILE-HEADER v1")
    f.seek(10 ** 9)
    check("record at 1 GB reads back", f.read(len(REC)) == REC)
    check("size reports 2 GB", f.size == 2 * 10 ** 9)

print("2. ARRAY style — a 5 GB implied array + matrix view")
p2 = os.path.join(D, "usage_array.evb")
if os.path.exists(p2):
    os.remove(p2)
buf2 = ImpliedBuffer.create(p2, 5 * 10 ** 9)
arr = ImpliedArray(buf2)
check("len(arr) is 5 billion", len(arr) == 5 * 10 ** 9)
arr[10 ** 9:10 ** 9 + 5] = b"hello"
check("slice assignment reads back",
      bytes(arr[10 ** 9:10 ** 9 + 5]) == b"hello")
arr[4_999_999_999] = 0xAB
check("last element assignment", arr[4_999_999_999] == 0xAB)
m = arr.matrix(1000, 10 ** 6)          # 1 GB matrix view
m[999, 999_999] = 7
m[0, 0] = 3
check("matrix corner writes", m[999, 999_999] == 7 and m[0, 0] == 3)
check("matrix row is an array window", len(m[5]) == 10 ** 6)
buf2.save()

print("3. RAW style — buffer API")
p3 = os.path.join(D, "usage_raw.evb")
if os.path.exists(p3):
    os.remove(p3)
buf3 = ImpliedBuffer.create(p3, 10 ** 9)
buf3.write(123_456, b"raw style works")
buf3.save()
check("raw write/read",
      ImpliedBuffer.open(p3).read(123_456, 15) == b"raw style works")

print("4. ANOTHER PROGRAM — child process, file interface")
child = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, r'%s');"
     "from exp_vram import ImpliedFile;"
     "f = ImpliedFile.open(r'%s');"
     "f.seek(10**9);"
     "print(f.read(40).decode())" % (D, p1)],
    capture_output=True, text=True)
check("child process read the 1 GB record",
      child.stdout.strip()
      == "a record stored at the one-gigabyte mark")

logical = 2 * 10 ** 9 + 5 * 10 ** 9 + 10 ** 9
stored = sum(os.path.getsize(p) for p in (p1, p2, p3))
print(f"\nACCOUNTING: {logical:,} logical bytes in use across "
      f"3 buffers | {stored:,} bytes actually stored | "
      f"ratio {logical / stored:,.0f}x")
print(f"USAGE TEST: {'ALL PASS' if all(results) else 'FAILURES'} "
      f"({sum(results)}/{len(results)})")
