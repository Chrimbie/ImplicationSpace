"""Full inference on CodeLlama 70B, held in EXP-VRAM.

Every weight byte is served from the BlockStore
(codellama70b.evb2) — the GGUF file is opened only for
tensor metadata and the vocabulary. The forward pass is
Christopher's harness math extended to all 80 blocks with
RoPE and a KV cache; generation is greedy.

    prompt: [BOS] hello        ->  up to N new tokens

The output text is the receipt: a code model continuing
"hello" coherently means the weights in the space are
really the model.
"""
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

base.TILE_ROWS = 512
DEVICE = base.DEVICE
MAX_NEW = 6


# ---------------- Q6_K (one tensor in this model) ---------
def dequant_q6_k(data, nrows, cols):
    """GGUF Q6_K: per 256 elements — ql 128 B (nibbles),
    qh 64 B (2-bit highs), 16 int8 scales, fp16 super-scale.
    Layout per ggml's dequantize_row_q6_K."""
    assert cols % 256 == 0
    nb = cols // 256
    a = np.asarray(data, dtype=np.uint8).reshape(nrows, nb, 210)
    ql = a[:, :, 0:128]
    qh = a[:, :, 128:192]
    sc = a[:, :, 192:208].copy().view(np.int8).astype(np.float32)
    d = a[:, :, 208:210].copy().view("<f2").astype(np.float32)
    d = d.reshape(nrows, nb, 1)
    out = np.empty((nrows, nb, 256), dtype=np.float32)
    is_idx = np.arange(32) // 16
    for h in range(2):
        ql_h = ql[:, :, h * 64:(h + 1) * 64].reshape(
            nrows, nb, 2, 32).astype(np.int32)
        qh_h = qh[:, :, h * 32:(h + 1) * 32].astype(np.int32)
        sc_h = sc[:, :, h * 8:(h + 1) * 8]
        lo = ql_h & 0xF
        hi = ql_h >> 4
        f0 = qh_h & 3
        f1 = (qh_h >> 2) & 3
        f2 = (qh_h >> 4) & 3
        f3 = (qh_h >> 6) & 3
        s0 = sc_h[:, :, is_idx]
        s1 = sc_h[:, :, 2 + is_idx]
        s2 = sc_h[:, :, 4 + is_idx]
        s3 = sc_h[:, :, 6 + is_idx]
        base_o = h * 128
        out[:, :, base_o + 0:base_o + 32] = \
            d * s0 * ((lo[:, :, 0] | (f0 << 4)) - 32)
        out[:, :, base_o + 32:base_o + 64] = \
            d * s1 * ((lo[:, :, 1] | (f1 << 4)) - 32)
        out[:, :, base_o + 64:base_o + 96] = \
            d * s2 * ((hi[:, :, 0] | (f2 << 4)) - 32)
        out[:, :, base_o + 96:base_o + 128] = \
            d * s3 * ((hi[:, :, 1] | (f3 << 4)) - 32)
    return out.reshape(nrows, cols)


def linear(x, store, name):
    t = store.tensors[name]
    dims = base.tensor_dims(t)
    typ = base.gguf_type(t)
    cols, rows = dims
    if typ == "Q4_0":
        return base.linear(x, store, name)
    if typ != "Q6_K":
        raise NotImplementedError(f"{name}: {typ}")
    packed_row = cols * 210 // 256
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    TR = base.TILE_ROWS
    for r in range(0, rows, TR):
        n = min(TR, rows - r)
        raw = store.read(name, r * packed_row, n * packed_row)
        w = torch.from_numpy(
            dequant_q6_k(raw, n, cols)).to(DEVICE)
        out[r:r + n] = torch.mv(w, x)
        del w
    return out


def embed_row(store, name, row):
    t = store.tensors[name]
    dims = base.tensor_dims(t)
    typ = base.gguf_type(t)
    cols = dims[0]
    if typ == "Q4_0":
        packed_row = cols * 18 // 32
        raw = store.read(name, row * packed_row, packed_row)
        return torch.from_numpy(
            base.dequant_q4_0(raw, 1, cols)).to(DEVICE)[0]
    if typ == "Q6_K":
        packed_row = cols * 210 // 256
        raw = store.read(name, row * packed_row, packed_row)
        return torch.from_numpy(
            dequant_q6_k(raw, 1, cols)).to(DEVICE)[0]
    if typ == "F32":
        raw = store.read(name, row * cols * 4, cols * 4)
        return torch.from_numpy(
            raw.view("<f4").copy()).to(DEVICE)
    raise NotImplementedError(f"embed {typ}")


# ---------------- RoPE (llama.cpp interleaved pairs) -----
def rope(x, pos, freq_base=10000.0):
    """x: (n_heads, head_dim) on DEVICE."""
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


def main():
    manifest = json.loads(
        Path("codellama70b_space_manifest.json").read_text())
    path = base.find_blob()
    reader = GGUFReader(str(path), "r")
    store = BlockStore.open(manifest["store"])
    s = SpaceGGUF(reader, store, manifest)

    q6 = [t.name for t in reader.tensors
          if base.gguf_type(t) == "Q6_K"]
    print("Q6_K tensor:", q6, flush=True)

    toks = list(reader.get_field(
        "tokenizer.ggml.tokens").contents())
    bos = int(reader.get_field(
        "tokenizer.ggml.bos_token_id").contents())
    eos = int(reader.get_field(
        "tokenizer.ggml.eos_token_id").contents())
    hello = toks.index("hello")

    n_layer = 80
    cache_k = [None] * n_layer
    cache_v = [None] * n_layer

    def forward(token_id, pos):
        x = embed_row(s, "token_embd.weight", token_id)
        for li in range(n_layer):
            p = f"blk.{li}."
            w = vector_from_space(s, p + "attn_norm.weight")
            xn = base.rmsnorm(x, w)
            q = linear(xn, s, p + "attn_q.weight") \
                .view(64, 128)
            k = linear(xn, s, p + "attn_k.weight") \
                .view(8, 128)
            v = linear(xn, s, p + "attn_v.weight") \
                .view(8, 128)
            q = rope(q, pos)
            k = rope(k, pos)
            if cache_k[li] is None:
                cache_k[li] = k.unsqueeze(0)
                cache_v[li] = v.unsqueeze(0)
            else:
                cache_k[li] = torch.cat(
                    [cache_k[li], k.unsqueeze(0)], dim=0)
                cache_v[li] = torch.cat(
                    [cache_v[li], v.unsqueeze(0)], dim=0)
            K = cache_k[li]  # (pos+1, 8, 128)
            V = cache_v[li]
            qh = q.view(8, 8, 128)
            scores = torch.einsum("ghd,pgd->ghp", qh, K) \
                / (128 ** 0.5)
            attn = torch.softmax(scores.float(), dim=-1)
            o = torch.einsum("ghp,pgd->ghd", attn, V)
            o = o.reshape(64 * 128)
            x = x + linear(o, s, p + "attn_output.weight")
            w = vector_from_space(s, p + "ffn_norm.weight")
            hf = base.rmsnorm(x, w)
            gate = linear(hf, s, p + "ffn_gate.weight")
            up = linear(hf, s, p + "ffn_up.weight")
            ff = torch.nn.functional.silu(gate) * up
            x = x + linear(ff, s, p + "ffn_down.weight")
        w = vector_from_space(s, "output_norm.weight")
        xn = base.rmsnorm(x, w)
        logits = linear(xn, s, "output.weight")
        return logits

    def detok(ids):
        buf = bytearray()
        for tid in ids:
            t = toks[tid]
            if t.startswith("<0x") and t.endswith(">"):
                buf.append(int(t[3:5], 16))
            elif tid in (0, 1, 2):
                continue
            else:
                buf += t.replace("▁", " ").encode("utf-8")
        return buf.decode("utf-8", "replace")

    prompt = [bos, hello]
    print(f"prompt tokens: {prompt} ('hello' = {hello})",
          flush=True)
    t0 = time.time()
    nxt = None
    for pos, tid in enumerate(prompt):
        logits = forward(tid, pos)
        nxt = int(torch.argmax(logits).item())
        print(f"  prefill pos {pos} in "
              f"{time.time() - t0:.1f}s -> next {nxt} "
              f"{toks[nxt]!r}", flush=True)
    gen = []
    pos = len(prompt)
    while len(gen) < MAX_NEW and nxt != eos:
        gen.append(nxt)
        print(f"  token {len(gen)}: {nxt} {toks[nxt]!r} | "
              f"cum {time.time() - t0:.1f}s", flush=True)
        t_step = time.time()
        logits = forward(nxt, pos)
        pos += 1
        nxt = int(torch.argmax(logits).item())
        print(f"    (step took {time.time() - t_step:.1f}s)",
              flush=True)
    text = detok(prompt[1:] + gen)
    result = {
        "prompt": "hello",
        "generated_token_ids": gen,
        "generated_text_after_hello": detok(gen),
        "full_text": text,
        "seconds_total": time.time() - t0,
        "weight_bytes_served": s.served,
        "backing": "EXP-VRAM BlockStore only",
    }
    print(json.dumps(result, indent=2), flush=True)
    Path("codellama70b_inference_result.json").write_text(
        json.dumps(result, indent=2))
    print("GENERATED:", repr(text), flush=True)
    store.close()


if __name__ == "__main__":
    main()
