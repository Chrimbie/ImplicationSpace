"""mini_forward_test.py - miniature of run_inference_from_space.forward,
two independent implementations, 3-token sequence, compared exactly.

A) STRUCTURAL COPY of the production forward (einsum attention,
   KV cache, interleaved rope) at small dims with random weights.
B) INDEPENDENT REFERENCE: per-head loops, no cache (full recompute
   per position), rope via complex multiplication.

If A != B, the diff localizes the wiring error. If A == B, the
production attention mechanics are right and the bug lives elsewhere.
"""
import math
import numpy as np

rng = np.random.default_rng(7)

HID = 32          # hidden size
HEADS = 4         # query heads
KV = 2            # kv heads
HD = HID // HEADS  # 8 head dim
LAYERS = 3
VOCAB = 50
FF = 64
REP = HEADS // KV
THETA = 10000.0

# random "weights"
W = {}
for l in range(LAYERS):
    p = f"l{l}."
    for name, shape in [("an", (HID,)), ("aq", (HID, HID)),
                        ("ak", (KV * HD, HID)), ("av", (KV * HD, HID)),
                        ("ao", (HID, HID)), ("fn", (HID,)),
                        ("fg", (FF, HID)), ("fd", (HID, FF)),
                        ("fu", (FF, HID))]:
        if len(shape) == 1:
            W[p + name] = 1.0 + 0.05 * rng.standard_normal(shape)
        else:
            W[p + name] = rng.standard_normal(shape) / math.sqrt(shape[1])
W["emb"] = rng.standard_normal((VOCAB, HID)) * 0.3
W["norm"] = 1.0 + 0.05 * rng.standard_normal(HID)
W["out"] = rng.standard_normal((VOCAB, HID)) / math.sqrt(HID)

TOKS = [5, 11, 23]


def rms(v, w):
    return v * (1.0 / math.sqrt(float(np.mean(v * v)) + 1e-5)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- A
def rope_interleaved_np(x, pos):
    """Copy of production rope: x (..., HD); pairs (2i, 2i+1)."""
    out = x.copy()
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2) / HD))
    ang = pos * inv
    c, s = np.cos(ang), np.sin(ang)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def forward_A(tok, pos, caches):
    x = W["emb"][tok].copy()
    for l in range(LAYERS):
        p = f"l{l}."
        h = rms(x, W[p + "an"])
        q = (W[p + "aq"] @ h).reshape(HEADS, HD)
        k = (W[p + "ak"] @ h).reshape(KV, HD)
        v = (W[p + "av"] @ h).reshape(KV, HD)
        q = rope_interleaved_np(q, pos)
        k = rope_interleaved_np(k, pos)
        ck, cv = caches[l]
        ck = k[None] if ck is None else np.concatenate([ck, k[None]], 0)
        cv = v[None] if cv is None else np.concatenate([cv, v[None]], 0)
        caches[l] = (ck, cv)
        qh = q.reshape(KV, REP, HD)
        scores = np.einsum("ghd,pgd->ghp", qh, ck) / math.sqrt(HD)
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = e / e.sum(axis=-1, keepdims=True)
        o = np.einsum("ghp,pgd->ghd", attn, cv).reshape(HID)
        x = x + W[p + "ao"] @ o
        h2 = rms(x, W[p + "fn"])
        x = x + W[p + "fd"] @ (silu(W[p + "fg"] @ h2)
                                * (W[p + "fu"] @ h2))
    return W["out"] @ rms(x, W["norm"]), x


# ---------------------------------------------------------------- B
def rope_ref(vec, pos):
    """Per-head-dim complex rotation, written independently."""
    out = vec.copy()
    for i in range(HD // 2):
        ang = pos * (THETA ** (-2.0 * i / HD))
        c, s = math.cos(ang), math.sin(ang)
        a, b = vec[2 * i], vec[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


def forward_B(seq, upto):
    """Full recompute: hidden states for all positions 0..upto,
    attention by explicit loops. Returns logits for position upto."""
    xs = [W["emb"][t].copy() for t in seq[: upto + 1]]
    for l in range(LAYERS):
        p = f"l{l}."
        qs, ks, vs = [], [], []
        for pos in range(upto + 1):
            h = rms(xs[pos], W[p + "an"])
            qs.append((W[p + "aq"] @ h).reshape(HEADS, HD))
            ks.append((W[p + "ak"] @ h).reshape(KV, HD))
            vs.append((W[p + "av"] @ h).reshape(KV, HD))
        new_xs = []
        for pos in range(upto + 1):
            o = np.zeros(HID)
            for head in range(HEADS):
                kvh = head // REP
                qv = rope_ref(qs[pos][head], pos)
                scores = []
                for pp in range(pos + 1):
                    kv_ = rope_ref(ks[pp][kvh], pp)
                    scores.append(float(qv @ kv_) / math.sqrt(HD))
                mx = max(scores)
                ex = [math.exp(s - mx) for s in scores]
                tot = sum(ex)
                acc = np.zeros(HD)
                for pp in range(pos + 1):
                    acc += (ex[pp] / tot) * vs[pp][kvh]
                o[head * HD:(head + 1) * HD] = acc
            x = xs[pos] + W[p + "ao"] @ o
            h2 = rms(x, W[p + "fn"])
            x = x + W[p + "fd"] @ (silu(W[p + "fg"] @ h2)
                                    * (W[p + "fu"] @ h2))
            new_xs.append(x)
        xs = new_xs
    return W["out"] @ rms(xs[upto], W["norm"]), xs[upto]


def main():
    caches = [(None, None) for _ in range(LAYERS)]
    worst = 0.0
    for pos, tok in enumerate(TOKS):
        la, xa = forward_A(tok, pos, caches)
        lb, xb = forward_B(TOKS, pos)
        dh = float(np.max(np.abs(xa - xb)))
        dl = float(np.max(np.abs(la - lb)))
        worst = max(worst, dh, dl)
        print(f"pos {pos} tok {tok}: hidden maxdiff {dh:.3e} | "
              f"logits maxdiff {dl:.3e} | argmax A "
              f"{int(np.argmax(la))} B {int(np.argmax(lb))}")
    print("MATCH - attention mechanics + cache are correct"
          if worst < 1e-9 else
          f"MISMATCH worst {worst:.3e} - bug localized above")


if __name__ == "__main__":
    main()
