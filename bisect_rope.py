"""Bisect: is the RoPE convention the bug?

Position 0 is convention-free (RoPE = identity). Process
[BOS] once, snapshot the KV caches, then process 'hello'
at position 1 under BOTH conventions and compare top-5
predictions. The real model's next token after 'hello'
starts with ',' (Ollama reference)."""
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from exp_blockstore import BlockStore
from run_block0_from_space import SpaceGGUF, vector_from_space
from run_inference_from_space import linear, embed_row

DEVICE = base.DEVICE


def rope_interleaved(x, pos, freq_base=10000.0):
    hd = x.shape[-1]
    i = torch.arange(hd // 2, device=x.device,
                     dtype=torch.float32)
    freqs = freq_base ** (-2.0 * i / hd)
    ang = pos * freqs
    cos, sin = torch.cos(ang), torch.sin(ang)
    pairs = x.float().reshape(*x.shape[:-1], hd // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    out = torch.stack([a * cos - b * sin,
                       a * sin + b * cos], dim=-1)
    return out.reshape(x.shape).to(x.dtype)


def rope_half(x, pos, freq_base=10000.0):
    hd = x.shape[-1]
    half = hd // 2
    i = torch.arange(half, device=x.device, dtype=torch.float32)
    freqs = freq_base ** (-2.0 * i / hd)
    ang = pos * freqs
    cos, sin = torch.cos(ang), torch.sin(ang)
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    out = torch.cat([x1 * cos - x2 * sin,
                     x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)


def forward(s, token_id, pos, caches, rope_fn):
    x = embed_row(s, "token_embd.weight", token_id)
    for li in range(80):
        p = f"blk.{li}."
        w = vector_from_space(s, p + "attn_norm.weight")
        xn = base.rmsnorm(x, w)
        q = linear(xn, s, p + "attn_q.weight").view(64, 128)
        k = linear(xn, s, p + "attn_k.weight").view(8, 128)
        v = linear(xn, s, p + "attn_v.weight").view(8, 128)
        q = rope_fn(q, pos)
        k = rope_fn(k, pos)
        ck, cv = caches[li]
        ck = k.unsqueeze(0) if ck is None else torch.cat(
            [ck, k.unsqueeze(0)], dim=0)
        cv = v.unsqueeze(0) if cv is None else torch.cat(
            [cv, v.unsqueeze(0)], dim=0)
        caches[li] = (ck, cv)
        qh = q.view(8, 8, 128)
        scores = torch.einsum("ghd,phd->ghp", qh, ck) \
            / (128 ** 0.5)
        attn = torch.softmax(scores.float(), dim=-1)
        o = torch.einsum("ghp,phd->ghd", attn, cv)
        x = x + linear(o.reshape(8192), s,
                       p + "attn_output.weight")
        w = vector_from_space(s, p + "ffn_norm.weight")
        hf = base.rmsnorm(x, w)
        gate = linear(hf, s, p + "ffn_gate.weight")
        up = linear(hf, s, p + "ffn_up.weight")
        ff = torch.nn.functional.silu(gate) * up
        x = x + linear(ff, s, p + "ffn_down.weight")
    w = vector_from_space(s, "output_norm.weight")
    xn = base.rmsnorm(x, w)
    return linear(xn, s, "output.weight")


def top5(logits, toks):
    v, idx = torch.topk(logits.float(), 5)
    return [(int(i), toks[int(i)], round(float(x), 3))
            for i, x in zip(idx, v)]


def main():
    manifest = json.loads(
        Path("codellama70b_space_manifest.json").read_text())
    path = base.find_blob()
    reader = GGUFReader(str(path), "r")
    store = BlockStore.open(manifest["store"])
    s = SpaceGGUF(reader, store, manifest)
    toks = list(reader.get_field(
        "tokenizer.ggml.tokens").contents())
    hello = toks.index("hello")

    caches = [(None, None)] * 80
    caches = [(None, None) for _ in range(80)]
    t0 = time.time()
    logits = forward(s, 1, 0, caches, rope_interleaved)
    r0 = top5(logits, toks)
    Path("bisect_result.json").write_text(json.dumps(
        {"pos0": r0}, indent=1))
    print(f"pos0 (BOS) done {time.time() - t0:.0f}s | top5: "
          f"{r0}", flush=True)
    snap = [(ck.clone() if ck is not None else None,
             cv.clone() if cv is not None else None)
            for ck, cv in caches]

    logits_a = forward(s, hello, 1, caches, rope_interleaved)
    ra = top5(logits_a, toks)
    d = json.loads(Path("bisect_result.json").read_text())
    d["pos1_interleaved"] = ra
    Path("bisect_result.json").write_text(json.dumps(d, indent=1))
    print(f"pos1 INTERLEAVED done {time.time() - t0:.0f}s | "
          f"top5: {ra}", flush=True)

    caches = [(ck.clone() if ck is not None else None,
               cv.clone() if cv is not None else None)
              for ck, cv in snap]
    logits_b = forward(s, hello, 1, caches, rope_half)
    rb = top5(logits_b, toks)
    d = json.loads(Path("bisect_result.json").read_text())
    d["pos1_rotate_half"] = rb
    Path("bisect_result.json").write_text(json.dumps(d, indent=1))
    print(f"pos1 ROTATE_HALF done {time.time() - t0:.0f}s | "
          f"top5: {rb}", flush=True)
    store.close()


if __name__ == "__main__":
    main()
