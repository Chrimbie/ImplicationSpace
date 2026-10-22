"""THE STRANGER TEST — can programs use exponentiated VRAM?

This program knows NOTHING about implication space. It imports
only the public exp_vram API (plus the standard library) and
treats an ImpliedBuffer like memory. Three phases, each a
SEPARATE process, passing data only through the buffer file:

  write    create a 1 TB buffer; write a 64 KB payload blob
           at offset 500 GB, 512 scattered bytes across the
           whole terabyte, and 3 controller values; save.
  verify   (new process) open the file; check the size, read
           the blob and scattered bytes back and compare
           byte-for-byte; hash a never-written region; then
           write its own message near the end and save.
  verify2  (third process) confirm both strangers' data.

PASS lines are the receipt. The payload is regenerated from a
fixed seed inside every phase, so no phase trusts another's
memory — only the buffer file.
"""

import hashlib
import random
import sys

from exp_vram import ImpliedBuffer

PATH = "stranger_pool.evb"
SIZE = 10 ** 12                     # 1 TB
BLOB_OFF = 500 * 10 ** 9
BLOB_LEN = 65_536
DERIVED_OFF = 700 * 10 ** 9
DERIVED_LEN = 1 << 20
MSG = b"second stranger was here"
MSG_OFF = 999 * 10 ** 9


def payload(n=BLOB_LEN):
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(
            b"stranger-payload" + ctr.to_bytes(4, "little")).digest()
        ctr += 1
    return bytes(out[:n])


def scattered():
    rng = random.Random(42)
    return [(rng.randrange(0, SIZE - 1), (i * 37 + 11) % 256)
            for i in range(512)]


def phase_write():
    buf = ImpliedBuffer.create(PATH, SIZE, layer=3)
    buf.write(BLOB_OFF, payload())
    for off, val in scattered():
        buf.write(off, bytes([val]))
    buf.set_unit(12345, 9)
    buf.set_unit(10 ** 8, 3)
    buf.set_unit(SIZE // 4096 - 1, 15)
    buf.save()
    import os
    print(f"WRITE  size={buf.size:,} bytes (1 TB) "
          f"stored_accounted={buf.stored_bytes():,} "
          f"file_on_disk={os.path.getsize(PATH):,}")
    print(f"WRITE  derived region sha256: "
          f"{hashlib.sha256(buf.read(DERIVED_OFF, DERIVED_LEN)).hexdigest()}")


def phase_verify():
    buf = ImpliedBuffer.open(PATH)
    ok = buf.size == SIZE
    print(f"VERIFY size is 1 TB: {'PASS' if ok else 'FAIL'}")
    got = buf.read(BLOB_OFF, BLOB_LEN)
    ok = got == payload()
    print(f"VERIFY 64 KB blob @ 500 GB byte-exact: "
          f"{'PASS' if ok else 'FAIL'}")
    bad = sum(1 for off, val in scattered()
              if buf.read(off, 1)[0] != val)
    print(f"VERIFY 512 scattered bytes: "
          f"{'PASS' if bad == 0 else f'FAIL ({bad} wrong)'}")
    units_ok = (buf.get_unit(12345) == 9
                and buf.get_unit(10 ** 8) == 3
                and buf.get_unit(SIZE // 4096 - 1) == 15)
    print(f"VERIFY controller values: "
          f"{'PASS' if units_ok else 'FAIL'}")
    print(f"VERIFY derived region sha256: "
          f"{hashlib.sha256(buf.read(DERIVED_OFF, DERIVED_LEN)).hexdigest()}")
    buf.write(MSG_OFF, MSG)
    buf.save()
    print("VERIFY wrote its own message @ 999 GB")


def phase_verify2():
    buf = ImpliedBuffer.open(PATH)
    ok = buf.read(MSG_OFF, len(MSG)) == MSG
    print(f"VERIFY2 second stranger's message byte-exact: "
          f"{'PASS' if ok else 'FAIL'}")
    ok = buf.read(BLOB_OFF, BLOB_LEN) == payload()
    print(f"VERIFY2 original blob still byte-exact: "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"VERIFY2 derived region sha256: "
          f"{hashlib.sha256(buf.read(DERIVED_OFF, DERIVED_LEN)).hexdigest()}")


if __name__ == "__main__":
    {"write": phase_write, "verify": phase_verify,
     "verify2": phase_verify2}[sys.argv[1]]()
