"""exp_vram — exponentiated VRAM as a library programs can use.

Christopher's question (2026-10-21): "how can i be sure that
programs can use it?" Answer: a plain memory-shaped API.

    from exp_vram import ImpliedBuffer
    buf = ImpliedBuffer.create("pool.evb", size_bytes=10**12)
    buf.write(123_456_789, b"hello from a terabyte")
    data = buf.read(123_456_789, 21)          # byte-exact
    buf.save()

THE MODEL
  A buffer of `size_bytes` logical bytes, divided into units.
  Each unit's content is DERIVED from one controller nibble by
  this library's derivation rule: the unit's implication
  pattern bit at that position (Christopher's fan-out: at
  layer 3 each controller bit governs a quarter of the unit)
  combined with a splitmix64 address hash for byte grain.
  Reads anywhere are served by derivation — that is where the
  exponentiation lives. Writes are stored honestly: a sparse
  patch table (every written byte costs a stored byte) plus a
  sparse controller map (units whose value differs from the
  buffer default). stored_bytes() always reports the truth.

  Layer widths (implied bytes per unit) follow Christopher's
  growth equation W(n) = 2^(2^(n+1) - 1) bits:
  L1 = 1 B, L2 = 16 B, L3 = 4096 B, L4 = 2^28 B, L5 = 2^60 B.
"""

import json
import os
import struct

import numpy as np

WIDTH_BITS = {1: 8, 2: 128, 3: 32_768, 4: 2 ** 31, 5: 2 ** 63}
MAGIC = b"EVB1"


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class ImpliedBuffer:
    def __init__(self, path, size_bytes, layer, default_unit=0):
        self.path = path
        self.size = int(size_bytes)
        self.layer = int(layer)
        self.unit_bytes = WIDTH_BITS[self.layer] // 8
        self.default_unit = int(default_unit) & 15
        self.units = {}          # unit_index -> nibble (sparse)
        self.patches = {}       # byte_offset -> byte (sparse)

    # ---------------- construction / persistence --------------
    @classmethod
    def create(cls, path, size_bytes, layer=3, default_unit=0):
        buf = cls(path, size_bytes, layer, default_unit)
        buf.save()
        return buf

    @classmethod
    def open(cls, path):
        with open(path, "rb") as f:
            raw = f.read()
        assert raw[:4] == MAGIC, "not an exp_vram buffer file"
        (hlen,) = struct.unpack("<I", raw[4:8])
        header = json.loads(raw[8:8 + hlen])
        buf = cls(path, header["size"], header["layer"],
                  header["default_unit"])
        body = memoryview(raw)[8 + hlen:]
        (nu,) = struct.unpack("<Q", body[:8])
        off = 8
        if nu:
            arr = np.frombuffer(body[off:off + nu * 9],
                                dtype=np.uint8).reshape(nu, 9)
            idx = arr[:, :8].copy().view(np.uint64).ravel()
            buf.units = {int(i): int(v)
                         for i, v in zip(idx, arr[:, 8])}
            off += nu * 9
        (np_,) = struct.unpack("<Q", body[off:off + 8])
        off += 8
        if np_:
            arr = np.frombuffer(body[off:off + np_ * 9],
                                dtype=np.uint8).reshape(np_, 9)
            idx = arr[:, :8].copy().view(np.uint64).ravel()
            buf.patches = {int(i): int(v)
                           for i, v in zip(idx, arr[:, 8])}
        return buf

    def save(self):
        header = {"size": self.size, "layer": self.layer,
                  "default_unit": self.default_unit}
        hb = json.dumps(header).encode()
        with open(self.path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(hb)))
            f.write(hb)
            for table in (self.units, self.patches):
                items = sorted(table.items())
                f.write(struct.pack("<Q", len(items)))
                if items:
                    arr = np.empty((len(items), 9), dtype=np.uint8)
                    idx = np.array([k for k, _ in items],
                                   dtype=np.uint64)
                    arr[:, :8] = idx.view(np.uint8).reshape(-1, 8)
                    arr[:, 8] = [v for _, v in items]
                    f.write(arr.tobytes())

    # ---------------- accounting ------------------------------
    def stored_bytes(self):
        """Honest stored cost: header estimate + 9 bytes per
        sparse unit + 9 bytes per patched byte."""
        return 64 + 9 * (len(self.units) + len(self.patches))

    # ---------------- derivation ------------------------------
    def _derive(self, offset, length):
        """Derived bytes for [offset, offset+length), before
        patch overlay. Vectorized splitmix over addresses;
        the controller bit for each byte's quarter (fan-out
        canon) flips the hash byte when set."""
        pos = np.arange(offset, offset + length, dtype=np.uint64)
        unit = pos // np.uint64(self.unit_bytes)
        j = (pos % np.uint64(self.unit_bytes)).astype(np.uint64)
        # controller value per position (sparse lookup, chunked)
        v = np.full(len(pos), self.default_unit, dtype=np.uint64)
        if self.units:
            keys = np.array(sorted(self.units), dtype=np.uint64)
            vals = np.array([self.units[int(k)] for k in keys],
                            dtype=np.uint64)
            hit = np.searchsorted(keys, unit)
            hit_c = np.minimum(hit, len(keys) - 1)
            match = keys[hit_c] == unit
            v = np.where(match, vals[hit_c], v)
        quarter = (j * np.uint64(4)) // np.uint64(self.unit_bytes)
        bit = (v >> quarter) & np.uint64(1)
        x = unit * np.uint64(0x9E3779B97F4A7C15) ^ j
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
        b = (x & np.uint64(0xFF)).astype(np.uint8)
        b = np.where(bit == 1, b ^ np.uint8(0xFF), b)
        return b

    # ---------------- memory-shaped API -----------------------
    def read(self, offset, length):
        if offset < 0 or offset + length > self.size:
            raise IndexError("read outside buffer")
        out = bytearray(self._derive(offset, length).tobytes())
        if self.patches:
            lo, hi = offset, offset + length
            for p, val in self.patches.items():
                if lo <= p < hi:
                    out[p - lo] = val
        return bytes(out)

    def write(self, offset, data):
        if offset < 0 or offset + len(data) > self.size:
            raise IndexError("write outside buffer")
        for i, byte in enumerate(data):
            self.patches[offset + i] = byte

    def set_unit(self, unit_index, value):
        if not 0 <= unit_index < self.size // self.unit_bytes + 1:
            raise IndexError("unit outside buffer")
        self.units[int(unit_index)] = int(value) & 15

    def get_unit(self, unit_index):
        return self.units.get(int(unit_index), self.default_unit)


# ==================================================================
# THE USABLE LAYER — implied space behind ordinary interfaces.
# A program should not need to know what a controller is.
# ==================================================================

class ImpliedFile:
    """File-object semantics over an ImpliedBuffer: open, seek,
    read, write — over a logical file as large as you declare.
    Size is fixed at creation (a virtual disk, not a growing
    file); every write is stored honestly as patches."""

    def __init__(self, buffer):
        self._b = buffer
        self._pos = 0
        self.closed = False

    @classmethod
    def create(cls, path, size_bytes, layer=3):
        return cls(ImpliedBuffer.create(path, size_bytes, layer))

    @classmethod
    def open(cls, path):
        return cls(ImpliedBuffer.open(path))

    @property
    def size(self):
        return self._b.size

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._b.size + offset
        self._pos = max(0, min(self._pos, self._b.size))
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._b.size - self._pos
        n = max(0, min(n, self._b.size - self._pos))
        data = self._b.read(self._pos, n)
        self._pos += n
        return data

    def write(self, data):
        self._b.write(self._pos, data)
        self._pos += len(data)
        return len(data)

    def save(self):
        self._b.save()

    def close(self):
        if not self.closed:
            self._b.save()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class ImpliedArray:
    """A numpy-flavored 1-D uint8 array over an ImpliedBuffer
    (or a window of one). Slices read derived+patched bytes;
    assignments write patches. len() can be a trillion."""

    def __init__(self, buffer, offset=0, length=None):
        self._b = buffer
        self._off = int(offset)
        self._len = (buffer.size - self._off) if length is None \
            else int(length)
        self.shape = (self._len,)
        self.dtype = np.uint8

    def __len__(self):
        return self._len

    def _norm(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._len)
            return start, stop, step
        i = int(key)
        if i < 0:
            i += self._len
        if not 0 <= i < self._len:
            raise IndexError("index out of range")
        return i, i + 1, 1

    def __getitem__(self, key):
        start, stop, step = self._norm(key)
        if stop <= start:
            return np.empty(0, dtype=np.uint8)
        raw = self._b.read(self._off + start, stop - start)
        arr = np.frombuffer(raw, dtype=np.uint8)
        if step != 1:
            arr = arr[::step]
        if isinstance(key, slice):
            return arr.copy()
        return int(arr[0])

    def __setitem__(self, key, value):
        start, stop, step = self._norm(key)
        if step != 1:
            idxs = range(start, stop, step)
            if np.isscalar(value):
                for i in idxs:
                    self._b.write(self._off + i, bytes([value]))
            else:
                vals = np.asarray(value, dtype=np.uint8).ravel()
                for i, v in zip(idxs, vals):
                    self._b.write(self._off + i, bytes([int(v)]))
            return
        if isinstance(value, (bytes, bytearray)):
            data = bytes(value)
        elif np.isscalar(value):
            data = bytes([int(value)]) * (stop - start)
        else:
            data = np.asarray(value, dtype=np.uint8).tobytes()
        if len(data) != stop - start:
            raise ValueError("assignment size mismatch")
        self._b.write(self._off + start, data)

    def materialize(self, offset=0, length=None):
        """Explicitly pull a region into a real numpy array."""
        if length is None:
            length = self._len - offset
        return np.frombuffer(
            self._b.read(self._off + offset, length),
            dtype=np.uint8).copy()

    def matrix(self, rows, cols):
        return ImpliedMatrix(self._b, self._off, rows, cols)


class ImpliedMatrix:
    """2-D view: m[i, j] byte access, m[i] row as ImpliedArray."""

    def __init__(self, buffer, offset, rows, cols):
        if offset + rows * cols > buffer.size:
            raise ValueError("matrix exceeds buffer")
        self._b = buffer
        self._off = offset
        self.shape = (rows, cols)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            r, c = self.shape
            if i < 0:
                i += r
            if j < 0:
                j += c
            return int(self._b.read(self._off + i * c + j, 1)[0])
        i = int(key)
        r, c = self.shape
        if i < 0:
            i += r
        return ImpliedArray(self._b, self._off + i * c, c)

    def __setitem__(self, key, value):
        i, j = key
        r, c = self.shape
        if i < 0:
            i += r
        if j < 0:
            j += c
        self._b.write(self._off + i * c + j, bytes([int(value)]))


# ------------------------- CLI ---------------------------------
def _cli(argv):
    import sys
    cmd = argv[0]
    if cmd == "create":
        size = int(float(argv[2]) * 1e9)
        layer = int(argv[3]) if len(argv) > 3 else 3
        buf = ImpliedBuffer.create(argv[1], size, layer)
        print(f"created {argv[1]}: {size:,} logical bytes, "
              f"layer {layer}, stored "
              f"{buf.stored_bytes()} accounted bytes")
    elif cmd == "info":
        buf = ImpliedBuffer.open(argv[1])
        print(f"{argv[1]}: {buf.size:,} logical bytes | "
              f"layer {buf.layer} | units set: {len(buf.units)} "
              f"| patched bytes: {len(buf.patches)} | stored "
              f"accounted: {buf.stored_bytes():,} | file: "
              f"{os.path.getsize(argv[1]):,} bytes")
    elif cmd == "write":
        buf = ImpliedBuffer.open(argv[1])
        buf.write(int(argv[2]), argv[3].encode())
        buf.save()
        print(f"wrote {len(argv[3])} bytes at {argv[2]}")
    elif cmd == "read":
        buf = ImpliedBuffer.open(argv[1])
        print(repr(buf.read(int(argv[2]), int(argv[3]))))
    else:
        print("usage: exp_vram.py create|info|write|read ...")
        return 2
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
