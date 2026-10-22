"""Block 0 of CodeLlama 70B, executed with every weight byte
served from inside EXP-VRAM storage (the BlockStore), never
from the GGUF file. The math is Christopher's harness
(codellama70b_expvram_block.py) imported unchanged; only the
backing source changes, and the address map comes from the
ingest manifest.

Success criterion: the output sha256 equals the file-backed
run's — ee96f50f337bc978a11c2021c2034d815ee10ca4d43ed79c9658f2d72f55d878.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from gguf import GGUFReader

import codellama70b_expvram_block as base
from exp_blockstore import BlockStore

EXPECTED = ("ee96f50f337bc978a11c2021c2034d815ee10ca4d43ed"
            "79c9658f2d72f55d878")


class SpaceGGUF:
    """Same read contract as the harness's LogicalGGUF,
    backed by the BlockStore. Tensor metadata (names, types,
    shapes) still comes from the reader; weight BYTES come
    only from the space."""

    def __init__(self, reader, store, manifest):
        self.tensors = {t.name: t for t in reader.tensors}
        self.address = {t["name"]: (t["offset"], t["length"])
                        for t in manifest["tensors"]}
        self.store = store
        self.logical_size = manifest["envelope_bytes"]
        self.calls = 0
        self.served = 0

    def read(self, name, start, length):
        base_off, total = self.address[name]
        if start < 0 or start + length > total:
            raise IndexError(name)
        self.calls += 1
        self.served += length
        return np.frombuffer(
            self.store.read(base_off + start, length),
            dtype=np.uint8)


def vector_from_space(s, name):
    """base.vector(), with the length taken from the
    manifest instead of the harness's in-RAM address map."""
    t = s.tensors[name]
    typ = base.gguf_type(t)
    _, total = s.address[name]
    raw = s.read(name, 0, total)
    if typ == "F32":
        return torch.from_numpy(
            raw.view("<f4").copy()).to(base.DEVICE)
    if typ == "F16":
        return torch.from_numpy(
            raw.view("<f2").astype(np.float32)).to(base.DEVICE)
    raise NotImplementedError(f"{name}: norm dtype {typ}")


def main():
    manifest = json.loads(
        Path("codellama70b_space_manifest.json").read_text())
    path = base.find_blob()
    reader = GGUFReader(str(path), "r")
    store = BlockStore.open(manifest["store"])
    s = SpaceGGUF(reader, store, manifest)
    names = ["blk.0.attn_norm.weight", "blk.0.attn_q.weight",
             "blk.0.attn_k.weight", "blk.0.attn_v.weight",
             "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
             "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
             "blk.0.ffn_down.weight"]
    dim = base.tensor_dims(s.tensors["blk.0.attn_q.weight"])[0]
    g = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(dim, generator=g).to(base.DEVICE).float() * 0.1
    if base.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    norm = vector_from_space(s, names[0])
    xn = base.rmsnorm(x, norm)
    q = base.linear(xn, s, names[1])
    k = base.linear(xn, s, names[2])
    v = base.linear(xn, s, names[3])
    n_heads = q.numel() // 128
    n_kv_heads = k.numel() // 128
    attended = v.view(n_kv_heads, 128).repeat_interleave(
        n_heads // n_kv_heads, dim=0).reshape(-1)
    attn = base.linear(attended, s, names[4])
    h = x + attn
    fn = vector_from_space(s, names[5])
    hf = base.rmsnorm(h, fn)
    gate = base.linear(hf, s, names[6])
    up = base.linear(hf, s, names[7])
    ff = torch.nn.functional.silu(gate) * up
    down = base.linear(ff, s, names[8])
    y = h + down
    if base.DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    sha = hashlib.sha256(
        y.cpu().numpy().astype("<f4").tobytes()).hexdigest()
    result = {
        "backing": "EXP-VRAM BlockStore only (no GGUF reads)",
        "elapsed_seconds": elapsed,
        "weight_reads": s.calls,
        "weight_bytes_served": s.served,
        "output_sha256_float32": sha,
        "expected_sha256": EXPECTED,
        "output_finite": bool(torch.isfinite(y).all().item()),
        "cuda_peak_allocated_mib":
            torch.cuda.max_memory_allocated() / 2**20
            if base.DEVICE == "cuda" else None,
        "match": sha == EXPECTED,
    }
    print(json.dumps(result, indent=2), flush=True)
    Path("codellama70b_block0_from_space_result.json").write_text(
        json.dumps(result, indent=2))
    print("BLOCK 0 FROM INSIDE THE SPACE:",
          "MATCH — identical output, weights served from "
          "EXP-VRAM storage" if result["match"] else
          "MISMATCH", flush=True)
    store.close()


if __name__ == "__main__":
    main()
