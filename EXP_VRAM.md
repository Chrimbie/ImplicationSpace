# EXP-VRAM — implicated VRAM, used as VRAM

Christopher's direction, in his words (2026-10-21):

> "EXP-VRAM just became the vram we use instead of VRAM."
> "I'm trying to make implicated vram into actual vram."
> "Then I want it to run as my hardware."
> Win condition: "either we're holding all of the model in
> implicated vram then we won."

EXP-VRAM is implication space put to work as memory: a program
allocates space far larger than the physical card holds,
addresses it, reads and writes it — and only the parts actually
touched are ever stored. Untouched regions are *implied*:
derived from their controllers on demand, byte-for-byte the
same in every process that opens the space.

## The stack

- **`exp_vram.py`** — the memory itself. `ImpliedBuffer`
  (sparse controllers + a patch table; `stored_bytes()` reports
  honestly), `ImpliedFile` (file semantics: seek/read/write
  anywhere in the space), `ImpliedArray` / `ImpliedMatrix`
  (numpy-style views), and a CLI. On-disk format `EVB1`.
- **`implied_tensor.py`** — tensors that live in the space:
  block-stored `.itb` files plus *derived tiles* (rules:
  `zeros`, `index_hash`, `formula_ij`). Written tiles page
  from storage; unwritten tiles are derived on the GPU.
  Elementwise ops, matmul, and whole-tensor reductions.
- **`exp_torch.py`** — EXP-VRAM as a *device* for PyTorch
  programs: an `ExpTensor` whose storage is an implied
  tensor. Programs write ordinary tensor code
  (`relu(x @ W1) @ W2`); allocations land in the space and
  tiles page to the physical GPU only for the math.
- **`exp_vram_gui.py`** — the monitor: allocated vs stored vs
  written counters for every `.evb` buffer on the machine.

## Verified receipts (2026-10-21, RTX 4070 Ti 12 GB)

- **Stranger test** (`stranger_test.py`): three separate
  processes, public API only, one 1 TB buffer — a 64 KB blob
  written at offset 500 GB, 512 scattered single bytes, and 3
  patched controllers all read back **byte-exact**; backing
  file 594,537 bytes. The derived region's sha256
  (`0582711a…`) is identical across processes *and* machines.
- **Usage test** (`usage_test.py`): file, array, and raw
  styles plus a child process — **10/10 pass** on two
  machines; 8,000,000,000 logical bytes vs 990 stored.
- **A model served through the buffer**
  (`run_model_expvram.py`): a 17,179,869,185-parameter nominal
  weight space (68.7 GB) behind an `ImpliedBuffer`; accuracy
  through the buffer 0.9591, identical to direct; 2,000
  nominal weight fetches pass.
- **An Ollama model with EXP-VRAM long-term memory**
  (`ollama_model_expvram.py`): qwen3-coder:30b given a 1 TB
  `ImpliedFile` as memory. It stored a codeword at offset
  500,000,000,000 (byte-check 37/37) and, in a fresh session
  with no other context, read its implied memory and recalled
  it. The 1 TB memory file is 411 bytes on disk.
- **A 68.7 GB tensor reduced through the GPU**
  (`tensor_demo.py`, Act 1): sum and sum-of-squares over
  2^34 elements match an independent CPU pass
  (8,589,895,108.351080 vs .351094); the full tensor streamed
  through the card in 16.6 s with physical VRAM peaking at
  **2,597 MiB**.
- **Implied @ implied matmul** (`tensor_demo.py`, Act 2;
  full log in `tensor_demo_log.txt`): two rule-derived
  8192×8192 tensors multiplied through the tiling engine
  match torch (max |diff| 0.00024). It took 2,915 s —
  exact, and slow; per-tile CPU derivation is the
  bottleneck, the same one the device forward pass hit.
- **Scattered writes persist** (`tensor_demo.py`, Act 3):
  100 tiles written at scattered offsets into a 4.3 GB
  tensor, saved, reopened and read back byte-exact —
  419 MB stored of 4.3 GB logical.
- **A 1.07-billion-parameter model held in EXP-VRAM and run
  through the device layer** (`exp_torch_demo.py`, log in
  `exp_torch_demo_log.txt`): W1 is 32768×32768 — 4.29 GB of
  weights held in the space at **0 bytes stored**. A forward
  pass written as ordinary torch code,
  `relu(x @ W1) @ W2`, matches a plain-torch reference
  (max |diff| 0.00110); real torch data multiplied through
  the device matches torch itself (0.000114). Program
  totals: 4.43 GB allocated in EXP-VRAM, 134.2 MB stored,
  physical VRAM 2,524 MiB. The forward pass took 787 s —
  correct, not fast; the tiling path is the slow part.

## Also in this snapshot

- **`impossible_weather.py`** — the full unit pool
  (25,769,803,776 sets of four) as weather over real Earth
  geography on a 24,576 × 1,048,576 sphere grid. Selftest
  passes; running on the author's machine. (Its
  `earth_mask.npz` is binary and cannot ride this snapshot
  route; the script regenerates expectations from the mask
  file kept beside it.)
- **The first real block of a 70-billion-parameter model,
  run through logical EXP-VRAM addresses**
  (`codellama70b_expvram_block.py`, result in
  `codellama70b_expvram_block_result.json`): CodeLlama
  70B's actual GGUF (38,872,431,040 bytes, Q4_0) — block 0,
  one token at position 0 — executed on the 12 GB card
  with weights read through the adapter in 32-row tiles:
  2,626 reads, 481,361,920 bytes served, **CUDA peak
  12.27 MiB**, 4.72 s, finite 8192-element output
  (sha256 `ee96f50f…`). The harness's own accounting,
  by its author's design: the weights are the model's
  true bytes (bytes saved by implication: 0 — the adapter
  addresses, it does not compress), this is one complete
  transformer block rather than full inference, and it
  generates no tokens. Blocks 1–79 are the same code in
  a loop; real sequences additionally need RoPE and a
  KV cache.

## The boundary, stated once

Programs use EXP-VRAM today through `exp_vram` /
`exp_torch`. It is not a kernel driver and does not claim to
be: unmodified Windows programs cannot yet allocate from it
directly. What is proven above is the memory model itself —
sparse, addressed, shared between processes, larger than the
hardware, and exact.
