"""exp_blockstore — the dense half of EXP-VRAM storage.

Why this exists (2026-10-21, Christopher's win condition:
"hold the whole model inside EXP-VRAM storage"): the EVB1
patch table stores one dict entry per written BYTE, which is
right for sparse writes and impossible for dense gigabytes.
This store keeps the same law at block granularity:

  * a region never written is DERIVED — byte-identical to
    exp_vram.ImpliedBuffer's derivation (layer 3, default
    controller 0), so untouched space costs nothing;
  * a written block is stored whole on disk, append-only;
  * the index is sparse — only stored blocks appear — so a
    2^60-byte logical envelope has an empty index until
    something real is written into it.

stored_bytes() is the true file size: header, payloads,
index, orphans and all.
"""

import json
import os
import struct

import numpy as np

MAGIC = b"EVB2"
ENTRY = struct.Struct("<QQI")  # block_index, payload_off, len
UNIT_BYTES = 4096  # layer 3, as in exp_vram.WIDTH_BITS


def derive_default(offset, length):
    """exp_vram.ImpliedBuffer._derive with default_unit=0 and
    no set controllers — the content of unwritten space."""
    pos = np.arange(offset, offset + length, dtype=np.uint64)
    unit = pos // np.uint64(UNIT_BYTES)
    j = (pos % np.uint64(UNIT_BYTES)).astype(np.uint64)
    x = unit * np.uint64(0x9E3779B97F4A7C15) ^ j
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x & np.uint64(0xFF)).astype(np.uint8)


class BlockStore:
    def __init__(self, path, size, block_size, fh):
        self.path = path
        self.size = int(size)
        self.block_size = int(block_size)
        self.blocks = {}  # block_index -> (payload_off, len)
        self._fh = fh

    # ---------------- construction / persistence ----------
    @classmethod
    def create(cls, path, size_bytes, block_size=1 << 20):
        header = {"size": int(size_bytes),
                  "block_size": int(block_size), "layer": 3}
        hb = json.dumps(header).encode()
        fh = open(path, "w+b")
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(hb)))
        fh.write(hb)
        st = cls(path, size_bytes, block_size, fh)
        st.save()
        return st

    @classmethod
    def open(cls, path):
        fh = open(path, "r+b")
        fh.seek(0)
        assert fh.read(4) == MAGIC, "not an EVB2 block store"
        (hlen,) = struct.unpack("<I", fh.read(4))
        header = json.loads(fh.read(hlen))
        st = cls(path, header["size"], header["block_size"], fh)
        fh.seek(0, 2)
        end = fh.tell()
        if end >= 16:
            fh.seek(end - 16)
            table_off, table_len = struct.unpack("<QQ", fh.read(16))
            if 0 < table_off < end and table_len % ENTRY.size == 0:
                fh.seek(table_off)
                raw = fh.read(table_len)
                for i in range(0, table_len, ENTRY.size):
                    bi, off, ln = ENTRY.unpack_from(raw, i)
                    st.blocks[int(bi)] = (int(off), int(ln))
        return st

    def save(self):
        """Append the current index at EOF + a 16-byte footer
        pointing at it. Append-only: older indexes remain as
        orphans; the newest footer wins."""
        fh = self._fh
        fh.seek(0, 2)
        table_off = fh.tell()
        for bi in sorted(self.blocks):
            off, ln = self.blocks[bi]
            fh.write(ENTRY.pack(bi, off, ln))
        table_len = fh.tell() - table_off
        fh.write(struct.pack("<QQ", table_off, table_len))
        fh.flush()
        os.fsync(fh.fileno())

    def close(self):
        self.save()
        self._fh.close()

    # ---------------- accounting --------------------------
    def stored_bytes(self):
        self._fh.flush()
        return os.fstat(self._fh.fileno()).st_size

    def blocks_stored(self):
        return len(self.blocks)

    # ---------------- block plumbing ----------------------
    def _block_len(self, bi):
        start = bi * self.block_size
        return min(self.block_size, self.size - start)

    def _read_block(self, bi):
        start = bi * self.block_size
        blen = self._block_len(bi)
        ent = self.blocks.get(bi)
        if ent is None:
            return derive_default(start, blen).tobytes()
        off, ln = ent
        assert ln == blen, "stored payload length mismatch"
        self._fh.seek(off)
        return self._fh.read(ln)

    # ---------------- memory-shaped API -------------------
    def read(self, offset, length):
        if offset < 0 or offset + length > self.size:
            raise IndexError("read outside store")
        out = bytearray()
        pos = offset
        end = offset + length
        while pos < end:
            bi = pos // self.block_size
            bstart = bi * self.block_size
            blk = self._read_block(bi)
            lo = pos - bstart
            take = min(end - pos, len(blk) - lo)
            out += blk[lo:lo + take]
            pos += take
        return bytes(out)

    def write(self, offset, data):
        if offset < 0 or offset + len(data) > self.size:
            raise IndexError("write outside store")
        view = memoryview(data)
        pos = offset
        done = 0
        fh = self._fh
        while done < len(data):
            bi = pos // self.block_size
            bstart = bi * self.block_size
            blen = self._block_len(bi)
            lo = pos - bstart
            take = min(len(data) - done, blen - lo)
            if take == blen and lo == 0:
                payload = bytes(view[done:done + take])
            else:
                cur = bytearray(self._read_block(bi))
                cur[lo:lo + take] = view[done:done + take]
                payload = bytes(cur)
            fh.seek(0, 2)
            poff = fh.tell()
            fh.write(payload)
            self.blocks[bi] = (poff, len(payload))
            pos += take
            done += take
        fh.flush()


def _selftest():
    import random
    import tempfile
    from exp_vram import ImpliedBuffer

    rng = random.Random(7)
    tmp = tempfile.mkdtemp()
    size = 3 * 65536 + 123
    p = os.path.join(tmp, "t.evb2")
    st = BlockStore.create(p, size, block_size=65536)
    ref = ImpliedBuffer("unused", size, 3)
    model = bytearray(ref.read(0, size))

    # untouched == EVB1 derivation, incl. across unit edges
    for _ in range(50):
        o = rng.randrange(0, size - 1000)
        n = rng.randrange(1, 1000)
        assert st.read(o, n) == ref.read(o, n), "derive mismatch"
    print("derive == ImpliedBuffer: PASS")

    # writes: full blocks, partials, boundary crossings
    for trial in range(60):
        o = rng.randrange(0, size - 1)
        n = rng.randrange(1, min(200000, size - o))
        data = bytes(rng.randrange(256) for _ in range(min(n, 4096)))
        if n > len(data):
            data = (data * (n // len(data) + 1))[:n]
        st.write(o, data)
        model[o:o + n] = data
    for _ in range(80):
        o = rng.randrange(0, size - 500)
        n = rng.randrange(1, 500)
        assert st.read(o, n) == bytes(model[o:o + n]), \
            f"readback mismatch trial"
    print("writes vs model (full/partial/crossing): PASS")

    st.close()
    st2 = BlockStore.open(p)
    for _ in range(40):
        o = rng.randrange(0, size - 500)
        n = rng.randrange(1, 500)
        assert st2.read(o, n) == bytes(model[o:o + n])
    print(f"reopen persistence: PASS | stored "
          f"{st2.stored_bytes():,} B for {size:,} logical")
    st2.close()

    # the exbibyte envelope: index stays empty, reads derive
    p2 = os.path.join(tmp, "big.evb2")
    big = BlockStore.create(p2, 1 << 60)
    far = 500_000_000_000
    assert big.read(far, 8192) == ref.__class__("u", 1 << 60, 3) \
        .read(far, 8192)
    print(f"2^60 envelope: stored {big.stored_bytes()} B, "
          f"derived read at 500 GB == EVB1: PASS")
    big.close()
    print("exp_blockstore selftest: ALL PASS")


if __name__ == "__main__":
    _selftest()
