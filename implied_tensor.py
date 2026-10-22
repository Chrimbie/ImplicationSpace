"""implied_tensor — tensors that live in exponentiated VRAM.

Christopher's requirement (2026-10-21): "programs to run what
vram would normally run through the exponentiated vram."

So this is a tensor runtime whose storage is implication
space. A program allocates a tensor of any logical size;
tiles are served on demand — DERIVED when the tile follows
the tensor's rule (never stored), PAGED from block storage
when real data was written there. Compute happens on the
GPU one tile at a time: physical VRAM only ever holds a
tile, while the tensor being processed can be larger than
the card — larger than the machine's RAM, if it's derived.

    from implied_tensor import ImpliedTensor
    X = ImpliedTensor.rule("big.itb", (2**34,), "index_hash")
    total = X.sum()          # streams 68.7 GB through the GPU
    C = A.matmul(B)          # tiled GEMM, both sides implied

STORAGE (.itb): header (shape, dtype, tile size, rule) + a
directory of materialized blocks + the blocks themselves.
Unwritten tiles are NOT in the file. stats() always reports
logical bytes vs stored bytes separately.

RULES for derived tiles (deterministic, position-only):
  zeros       every element 0
  index_hash  splitmix64(element index) mapped to [0, 1)
  formula_ij  2-D: ((i*31 + j*17) % 97)/97 - 0.5
"""

import json
import os
import struct

import numpy as np

MAGIC = b"ITB1"
TILE_ELEMS = 1 << 24                  # 16M floats = 64 MB tiles


def _splitmix_f64(idx):
    x = (idx + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return (x >> 11) * (1.0 / (1 << 53))


class ImpliedTensor:
    def __init__(self, path, shape, rule="zeros", tile_elems=None):
        self.path = path
        self.shape = tuple(int(s) for s in shape)
        self.size = int(np.prod(self.shape))
        self.rule = rule
        self.tile_elems = tile_elems or TILE_ELEMS
        self.blocks = {}              # tile_index -> bytes
        self._cache = {}
        if os.path.exists(path):
            self._load()

    # ---------------- constructors ----------------------------
    @classmethod
    def create(cls, path, shape, rule="zeros", tile_elems=None):
        if os.path.exists(path):
            os.remove(path)
        t = cls(path, shape, rule, tile_elems)
        t.save()
        return t

    @classmethod
    def rule(cls, path, shape, rule):
        return cls.create(path, shape, rule)

    @classmethod
    def open(cls, path):
        return cls(path, (1,))        # header load fixes shape

    # ---------------- persistence -----------------------------
    def _load(self):
        with open(self.path, "rb") as f:
            raw = f.read()
        assert raw[:4] == MAGIC, "not an implied tensor file"
        (hlen,) = struct.unpack("<I", raw[4:8])
        hdr = json.loads(raw[8:8 + hlen])
        self.shape = tuple(hdr["shape"])
        self.size = int(np.prod(self.shape))
        self.rule = hdr["rule"]
        self.tile_elems = hdr["tile_elems"]
        body = raw[8 + hlen:]
        (nb,) = struct.unpack("<Q", body[:8])
        off = 8
        self.blocks = {}
        for _ in range(nb):
            (ti, blen) = struct.unpack("<QQ", body[off:off + 16])
            off += 16
            self.blocks[int(ti)] = body[off:off + blen]
            off += blen

    def save(self):
        hdr = {"shape": list(self.shape), "rule": self.rule,
               "tile_elems": self.tile_elems}
        hb = json.dumps(hdr).encode()
        with open(self.path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(hb)))
            f.write(hb)
            f.write(struct.pack("<Q", len(self.blocks)))
            for ti in sorted(self.blocks):
                data = self.blocks[ti]
                f.write(struct.pack("<QQ", ti, len(data)))
                f.write(data)

    # ---------------- accounting ------------------------------
    @property
    def logical_bytes(self):
        return self.size * 4

    @property
    def stored_bytes(self):
        return sum(len(b) for b in self.blocks.values()) + 128

    def stats(self):
        return {"shape": self.shape, "logical_bytes":
                self.logical_bytes, "stored_bytes":
                self.stored_bytes, "blocks": len(self.blocks),
                "rule": self.rule}

    # ---------------- tile serving ----------------------------
    def _tile_range(self, ti):
        start = ti * self.tile_elems
        return start, min(self.size, start + self.tile_elems)

    def get_tile(self, ti):
        """numpy float32 tile: paged if written, else derived."""
        if ti in self._cache:
            return self._cache[ti]
        if ti in self.blocks:
            arr = np.frombuffer(self.blocks[ti], "<f4").copy()
        else:
            start, end = self._tile_range(ti)
            arr = self._derive(start, end)
        if len(self._cache) > 4:
            self._cache.clear()
        self._cache[ti] = arr
        return arr

    def _derive(self, start, end):
        n = end - start
        if self.rule == "zeros":
            return np.zeros(n, dtype=np.float32)
        if self.rule == "index_hash":
            idx = np.arange(start, end, dtype=np.uint64)
            return _splitmix_f64(idx).astype(np.float32)
        if self.rule == "formula_ij":
            rows, cols = self.shape
            idx = np.arange(start, end, dtype=np.uint64)
            i = (idx // cols).astype(np.float64)
            j = (idx % cols).astype(np.float64)
            return (((i * 31 + j * 17) % 97) / 97.0 - 0.5) \
                .astype(np.float32)
        raise ValueError(f"unknown rule {self.rule}")

    def set_tile(self, ti, arr):
        arr = np.asarray(arr, dtype=np.float32)
        self.blocks[ti] = arr.astype("<f4").tobytes()
        self._cache[ti] = arr

    def read_flat(self, start, length):
        """Exact-range read: for block-free (pure rule) tensors,
        derive precisely the requested range; otherwise fall
        back to the tiled region reader."""
        if not self.blocks:
            return self._derive(start, start + length)
        return self.read_region(start, length)

    def read_region(self, start, length):
        out = np.empty(length, dtype=np.float32)
        pos = 0
        while pos < length:
            ti = (start + pos) // self.tile_elems
            tstart, _ = self._tile_range(ti)
            tile = self.get_tile(ti)
            off = start + pos - tstart
            take = min(length - pos, len(tile) - off)
            out[pos:pos + take] = tile[off:off + take]
            pos += take
        return out

    # ---------------- GPU tile engine -------------------------
    def _gpu(self):
        import torch
        return torch

    def _derive_gpu(self, start, end):
        """Derive a tile ON the GPU (splitmix / formula in
        torch int64) — the EXP-VRAM pattern: derivation at the
        card. Falls back to CPU derivation without CUDA."""
        try:
            torch = self._gpu()
            if not torch.cuda.is_available():
                return None
            idx = torch.arange(start, end, device="cuda",
                               dtype=torch.int64)
            if self.rule == "zeros":
                return torch.zeros(end - start, device="cuda")
            if self.rule == "index_hash":
                def lsr(x, k):      # logical shift on int64 bits
                    return (x >> k) & ((1 << (64 - k)) - 1)
                x = idx + 0x9E3779B97F4A7C15
                x = (x ^ lsr(x, 30)) * 0xBF58476D1CE4E5B9
                x = (x ^ lsr(x, 27)) * 0x94D049BB133111EB
                x = x ^ lsr(x, 31)
                return (lsr(x, 11).double()
                        * (1.0 / (1 << 53))).float()
            if self.rule == "formula_ij":
                cols = self.shape[1]
                i = (idx // cols).double()
                j = (idx % cols).double()
                return (((i * 31 + j * 17) % 97) / 97.0 - 0.5) \
                    .float()
        except Exception:
            return None
        return None

    def _tile_gpu(self, ti):
        """Tile as a CUDA tensor: paged blocks go through numpy;
        unwritten tiles derive on the GPU when possible."""
        torch = self._gpu()
        if ti in self.blocks:
            return torch.from_numpy(self.get_tile(ti)).cuda()
        start, end = self._tile_range(ti)
        t = self._derive_gpu(start, end)
        if t is not None:
            return t
        return torch.from_numpy(self.get_tile(ti)).cuda()

    def reduce_all(self):
        """ONE streaming pass computing sum, sumsq and max —
        tiles are derived or paged exactly once."""
        n_tiles = (self.size + self.tile_elems - 1) \
            // self.tile_elems
        s = 0.0
        s2 = 0.0
        peak = -1e30
        for ti in range(n_tiles):
            t = self._tile_gpu(ti)
            d = t.double()
            s += float(d.sum())
            s2 += float((d * d).sum())
            peak = max(peak, float(t.max()))
            del t, d
        return {"sum": s, "sumsq": s2, "max": peak}

    def sum(self):
        return self.reduce_all()["sum"]

    def sumsq(self):
        return self.reduce_all()["sumsq"]

    def max(self):
        return self.reduce_all()["max"]

    def elementwise(self, other, op, out_path):
        """out = self <op> other (tensor or scalar), streamed."""
        out = ImpliedTensor.create(out_path, self.shape, "zeros")
        n_tiles = (self.size + self.tile_elems - 1) \
            // self.tile_elems
        torch = self._gpu()
        for ti in range(n_tiles):
            a = torch.from_numpy(self.get_tile(ti)).cuda()
            if isinstance(other, ImpliedTensor):
                b = torch.from_numpy(other.get_tile(ti)).cuda()
            else:
                b = float(other)
            r = a * b if op == "mul" else a + b
            out.set_tile(ti, r.cpu().numpy())
            del a, r
        out.save()
        return out

    def matmul(self, other, out_path):
        """2-D tiled GEMM: self (m,k) @ other (k,n) -> (m,n).
        Tiles of the OUTPUT are computed and stored; inputs are
        served tile by tile (derived or paged)."""
        assert len(self.shape) == 2 and len(other.shape) == 2
        m, k = self.shape
        k2, n = other.shape
        assert k == k2
        out = ImpliedTensor.create(out_path, (m, n), "zeros")
        torch = self._gpu()
        TS = 1024
        def block(t, r0, r1, c0, c1):
            rows = []
            for r in range(r0, r1):
                rows.append(t.read_flat(r * t.shape[1] + c0,
                                        c1 - c0))
            return np.stack(rows)
        for i0 in range(0, m, TS):
            i1 = min(m, i0 + TS)
            for j0 in range(0, n, TS):
                j1 = min(n, j0 + TS)
                acc = None
                for k0 in range(0, k, TS):
                    k1 = min(k, k0 + TS)
                    A = torch.from_numpy(
                        block(self, i0, i1, k0, k1)).cuda()
                    B = torch.from_numpy(
                        block(other, k0, k1, j0, j1)).cuda()
                    prod = A @ B
                    acc = prod if acc is None else acc + prod
                    del A, B, prod
                cblk = acc.cpu().numpy()
                for r in range(i1 - i0):
                    start = (i0 + r) * n + j0
                    ti = start // out.tile_elems
                    # C tiles here are row blocks smaller than a
                    # storage tile: write via region writer
                    out._write_region(start, j1 - j0,
                                      cblk[r])
                del acc, cblk
        out.save()
        return out

    def _write_region(self, start, length, data):
        pos = 0
        while pos < length:
            ti = (start + pos) // self.tile_elems
            tstart, _ = self._tile_range(ti)
            tile = self.get_tile(ti).copy()
            off = start + pos - tstart
            take = min(length - pos, len(tile) - off)
            tile[off:off + take] = data[pos:pos + take]
            self.set_tile(ti, tile)
            pos += take
