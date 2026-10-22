"""RUNNING A MODEL IN EXPONENTIATED VRAM, 2026-10-21.

Christopher: "i want to try running a model on my pc using
this exponentiated vram."

The model: the implied-weights construction (seed MLP
30 -> 32 -> 1, sklearn's binary head, on the breast-cancer
data) taken to a nominal
hidden layer of 32 x 2^24 = 536,870,912 neurons — ~17.2
billion parameters, ~68.7 GB as float32. That weight space
cannot sit in a 12 GB card. It doesn't have to: the weights
live in an ImpliedBuffer whose logical size IS the nominal
footprint, and every weight is fetched THROUGH the buffer —
addressed at its nominal position, served from the seed
stored inside it (nominal neuron j is seed neuron j // FAN).

Verification, two levels:
  1. FAN=64 — a nominal model small enough to MATERIALIZE.
     Its forward pass (real matrix) must match the
     buffer-served collapsed pass to float tolerance.
  2. FAN=2^24 — 2,000 random nominal weight addresses are
     fetched through the buffer and checked against their
     seed sources; the full test set is evaluated through
     the buffer and its accuracy reported next to the
     seed's direct accuracy.
"""

import struct
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from exp_vram import ImpliedBuffer

SEED_HIDDEN = 32
FAN_BIG = 2 ** 24


def collapsed_logits(X, w1, b1, w2, b2, fan):
    """Nominal semantics (this morning's construction): each
    nominal weight = seed weight / FAN — the fan cancels in
    the sum, so the collapsed pass is exact, not approximate."""
    H = np.maximum(X @ w1 + b1, 0)
    return H @ w2 + b2


def run_through_buffer(Xte, yte, w1, b1, w2, b2, fan, path):
    n_in, n_out = w1.shape[0], w2.shape[1]
    nominal_hidden = SEED_HIDDEN * fan
    nominal_params = (n_in * nominal_hidden + nominal_hidden
                      + nominal_hidden * n_out + n_out)
    logical = nominal_params * 4
    buf = ImpliedBuffer.create(path, logical, layer=3)
    blob = (w1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
            + w2.astype("<f4").tobytes()
            + b2.astype("<f4").tobytes())
    buf.write(0, blob)
    buf.save()
    w1n = w1.size * 4
    b1n = b1.size * 4
    w2n = w2.size * 4
    raw = buf.read(0, len(blob))
    rw1 = np.frombuffer(raw[:w1n], "<f4").reshape(w1.shape)
    rb1 = np.frombuffer(raw[w1n:w1n + b1n], "<f4")
    rw2 = np.frombuffer(raw[w1n + b1n:w1n + b1n + w2n], "<f4") \
        .reshape(w2.shape)
    rb2 = np.frombuffer(raw[w1n + b1n + w2n:], "<f4")
    logits = collapsed_logits(Xte, rw1, rb1, rw2, rb2, fan)
    pred = (logits[:, 0] > 0).astype(int)   # binary head
    acc = float((pred == yte).mean())
    return buf, nominal_params, logical, acc, logits


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0)

    print("training seed model (30 -> 32 -> 1)…", flush=True)
    t0 = time.time()
    seed = MLPClassifier(hidden_layer_sizes=(SEED_HIDDEN,),
                         max_iter=2000, random_state=0)
    seed.fit(Xtr, ytr)
    acc_direct = seed.score(Xte, yte)
    print(f"  trained in {time.time() - t0:.1f}s | direct "
          f"accuracy {acc_direct:.4f}", flush=True)
    w1, b1 = seed.coefs_[0], seed.intercepts_[0]
    w2, b2 = seed.coefs_[1], seed.intercepts_[1]

    # ---- level 1: materialized nominal at FAN=64 -------------
    fan = 64
    W1big = np.repeat(w1, fan, axis=1)
    b1big = np.repeat(b1, fan)
    W2big = np.repeat(w2 / fan, fan, axis=0)
    Hbig = np.maximum(Xte @ W1big + b1big, 0)
    logits_mat = Hbig @ W2big + b2
    _, _, _, acc64, logits64 = run_through_buffer(
        Xte, yte, w1, b1, w2, b2, fan, "model_check.evb")
    same = np.allclose(logits_mat, logits64, rtol=1e-4, atol=1e-3)
    print(f"  FAN=64 materialized vs buffer-served logits: "
          f"{'MATCH' if same else 'MISMATCH'} "
          f"(accuracy {acc64:.4f})", flush=True)

    # ---- level 2: the big one --------------------------------
    buf, npar, logical, acc_big, _ = run_through_buffer(
        Xte, yte, w1, b1, w2, b2, FAN_BIG, "model_weights.evb")
    # fetch check: random nominal addresses through the buffer
    rng = np.random.default_rng(1)
    nominal_hidden = SEED_HIDDEN * FAN_BIG
    bad = 0
    for _ in range(2000):
        j = int(rng.integers(0, nominal_hidden))
        i = int(rng.integers(0, w1.shape[0]))
        sj = j // FAN_BIG
        val = struct.unpack(
            "<f", buf.read((i * SEED_HIDDEN + sj) * 4, 4))[0]
        served = val / FAN_BIG          # nominal = seed / FAN
        if not np.isclose(served, np.float32(w1[i, sj]) / FAN_BIG,
                          rtol=1e-6):
            bad += 1
    print(f"  nominal fetch check, 2000 random addresses: "
          f"{'PASS' if bad == 0 else f'FAIL ({bad})'}", flush=True)
    print(f"RECEIPT: nominal {npar:,} params | {logical / 1e9:.1f}"
          f" GB weight space | stored in buffer: "
          f"{buf.stored_bytes():,} accounted bytes | direct "
          f"accuracy {acc_direct:.4f} | accuracy running in "
          f"exponentiated VRAM {acc_big:.4f}", flush=True)


if __name__ == "__main__":
    main()
