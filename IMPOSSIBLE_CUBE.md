# The GPU form and the Impossible Cube — results, 2026-10-21

Two results from 2026-10-21, built on the implication space in
this repository (fan-out expansion; growth equation
W(n) = 2^(2^(n+1) − 1); see EVIDENCE_2026-10-21.md).

## 1. Sets of four as implication-computation units (GPU)

`implication_units_gpu.py`. Christopher's construction: every set
of four bits is transformed into implication computation — one
GPU element per set of four; each unit holds only its nibble and
derives answers (popcount, weighted sum, polynomial hash) about
its full 32,768-bit implied pattern from the 4 bits, in closed
form. The pattern is never materialized on the GPU. Two modes:
divided (per-unit answers) and all-together (shared aggregate).

**Exactness: PASS** — GPU answers match naive CPU answers
computed from fully materialized patterns, on sampled units,
in both the builder's run and Christopher's independent run
on his own machine (RTX 4070 Ti), 2026-10-21.

**Throughput (Christopher's run, best of 3, synchronized)** —
these figures supersede the builder's earlier ones:

| pool | divided | all-together |
|---|---|---|
| 4,096 | 3,994,389 units/s | 8,208,251 units/s |
| 65,536 | 160,185,260 units/s | 53,865,943 units/s |
| 1,048,576 | 651,852,158 units/s | 728,032,861 units/s |
| 16,777,216 | 490,630,951 units/s | 469,381,556 units/s |

Peak implied-bits-derived rate: ~2.39 × 10^13 bits/s.
Linear extrapolation (labelled as such): the full pool of
25,769,803,776 sets — every set of four bits in the card's
12 GB, by Christopher's divide-by-4 construction — at the
measured 2^24 rate would take ~52.5 s of GPU time in divided
mode. Stored per unit on the GPU: its 4 bits. The 32,768-bit
pattern exists only as derivation, in neither mode.

## 2. The Impossible Cube

`impossible_cube.py` (+ `impossible_cube_selftest_log.txt`).
Christopher's test shape: a multicolored cube spinning in
PyOpenGL — the simplest display he knows — driven by the full
pool, aggregated.

- **Pool:** all 25,769,803,776 units, evolving on the GPU in
  tiles. Evolution rule (v'(i) = (v(i−1) + 3·v(i) + v(i+1) +
  step) mod 16, ring) is marked EMPIRICAL in the code — a
  simple mixing rule so the surface lives; nothing depends on it.
- **Impossibility accounting:** implied state = 25,769,803,776
  × 32,768 bits ≈ **105.5 TB**. Stored state: the controller
  bytes, **25.8 GB** in RAM. The full state cannot be stored on
  the machine it runs on; the simulation runs because the
  state is implied, not instantiated.
- **Display:** 6 faces × 128 × 128 cells = 98,304 cells; the
  pool divides exactly, each cell aggregating 2^18 units
  (pattern density, warm fraction, hue mix, all in closed
  form). Cube spins at 60 fps; faces refresh per sweep; the
  window title carries the live accounting (sweep, units,
  TB implied, GB stored).
- **Verification:** self-test passes (evolution rule and
  aggregates vs independent references; popcount closed form
  vs materialized patterns). On Christopher's PC the GPU sweep
  is bit-identical to the CPU sweep. **Running on his machine
  2026-10-21** — launched by him; his screenshot shows the
  title bar: "25,769,803,776 units | 106 TB implied | 26 GB
  stored," cube spinning with the aggregated surface.

## What this is, in the project's own terms

Implication space is an **infinite addressing space**: it
multiplies addressable positions and derives their contents on
demand; it does not create information. The cube is that fact
made visible — a simulation that is impossible to *store* and
entirely possible to *run*.
