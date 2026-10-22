"""exp_torch — EXP-VRAM as a device for PyTorch programs.

Christopher's end-state (2026-10-21): "make implicated vram
into actual vram... run as my hardware." This module is that
layer for his stack: a torch Tensor subclass whose storage
lives in EXP-VRAM (implied_tensor block files). Programs
write ordinary tensor code —

    W = ExpTensor.rule((8192, 32768), "index_hash", "w1.itb")
    h = torch.relu(x @ W)
    y = h @ W2

— and every allocation is an address in implication space.
Tiles page into the physical GPU only for the moment of
math; results write back as blocks. dtype is float32 in v1.

Escape hatch, kept honest: .cuda()/.cpu() materialize a real
tensor (that data then lives in real memory, by definition).
"""

import os

import numpy as np
import torch

from implied_tensor import ImpliedTensor

_counter = [0]


def _fresh_path(tag):
    _counter[0] += 1
    return f"exp_{tag}_{_counter[0]}.itb"


class ExpTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, storage, shape):
        inst = torch.Tensor._make_subclass(
            cls, torch.empty(shape, device="meta"), False)
        return inst

    def __init__(self, storage, shape):
        self._storage = storage
        self._shape = tuple(shape)

    # ---------------- factories (the "allocation") ------------
    @classmethod
    def _wrap(cls, storage):
        return cls(storage, storage.shape)

    @classmethod
    def zeros(cls, shape, path=None):
        return cls._wrap(ImpliedTensor.create(
            path or _fresh_path("zeros"), shape, "zeros"))

    @classmethod
    def rule(cls, shape, rule, path=None):
        return cls._wrap(ImpliedTensor.create(
            path or _fresh_path(rule), shape, rule))

    @classmethod
    def from_torch(cls, t, path=None):
        t = t.detach().to("cpu", torch.float32).contiguous()
        st = ImpliedTensor.create(
            path or _fresh_path("data"), tuple(t.shape), "zeros")
        flat = t.numpy().ravel()
        pos = 0
        while pos < len(flat):
            ti = (pos // st.tile_elems)
            st.set_tile(ti, st.get_tile(ti))  # ensure block exists
            chunk = flat[pos:pos + st.tile_elems]
            tile = st.get_tile(ti).copy()
            off = pos - ti * st.tile_elems
            tile[off:off + len(chunk)] = chunk
            st.set_tile(ti, tile)
            pos += len(chunk)
        st.save()
        return cls._wrap(st)

    # ---------------- basics ----------------------------------
    @property
    def storage_stats(self):
        return self._storage.stats()

    def materialize(self, device="cuda"):
        """Pull the whole tensor into real memory (small ones,
        verification, or an explicit escape hatch)."""
        flat = self._storage.read_region(0, self._storage.size)
        return torch.from_numpy(flat.reshape(self._shape)).to(
            device)

    def exp_sum(self):
        return self._storage.reduce_all()["sum"]

    # ---------------- dispatch --------------------------------
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        try:
            base = func.overloadpacket._qualified_op_name \
                .replace("::", ".")
            ov = getattr(func, "_overload_name", "")
            name = f"{base}.{ov}" if ov else base
        except Exception:
            name = str(func)

        def is_exp(x):
            return isinstance(x, ExpTensor)

        if name == "aten.detach":
            return args[0]
        if name == "aten.clone":
            src = args[0]
            out = ImpliedTensor.create(_fresh_path("clone"),
                                       src._shape, "zeros")
            for ti in range((src._storage.size
                             + src._storage.tile_elems - 1)
                            // src._storage.tile_elems):
                out.set_tile(ti, src._storage.get_tile(ti))
            out.save()
            return cls._wrap(out)
        if name in ("aten.mm", "aten.matmul"):
            a, b = args[0], args[1]
            if is_exp(a) and is_exp(b):
                out = a._storage.matmul(b._storage,
                                        _fresh_path("mm"))
                return cls._wrap(out)
        if name in ("aten.add.Tensor", "aten.mul.Tensor"):
            a, b = args[0], args[1]
            op = "add" if name.startswith("aten.add") else "mul"
            if is_exp(a) and is_exp(b):
                out = a._storage.elementwise(
                    b._storage, op, _fresh_path(op))
                return cls._wrap(out)
            if is_exp(a) and np.isscalar(b):
                out = a._storage.elementwise(
                    b, op, _fresh_path(op))
                return cls._wrap(out)
        if name in ("aten.add.Scalar", "aten.mul.Scalar"):
            a, s = args[0], args[1]
            op = "add" if name.startswith("aten.add") else "mul"
            out = a._storage.elementwise(float(s), op,
                                         _fresh_path(op))
            return cls._wrap(out)
        if name == "aten.relu":
            a = args[0]
            out = ImpliedTensor.create(_fresh_path("relu"),
                                       a._shape, "zeros")
            n = (a._storage.size + a._storage.tile_elems - 1) \
                // a._storage.tile_elems
            for ti in range(n):
                tile = a._storage.get_tile(ti)
                out.set_tile(ti, np.maximum(tile, 0))
            out.save()
            return cls._wrap(out)
        if name == "aten.sum":
            a = args[0]
            return torch.tensor(a._storage.reduce_all()["sum"])
        return NotImplemented
