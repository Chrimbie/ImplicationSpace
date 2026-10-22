# Graphics in implication space (Oct 22)

Christopher's idea, in his words: "create some spare bits out
of a partition to use for graphics in a CPU" — then refined:
"we'll partition the CPU and use maybe a quarter of it to act
as a GPU." The test machine, by his direction, is his phone —
a Pixel 6a, CPU only.

## The implied framebuffer — VERIFIED (PC + phone)

`implied_framebuffer.py`: a framebuffer whose address space
lives in an EVB2 BlockStore. Block size = one canvas row; the
CPU rasterizes into a row cache; each drawn row is stored
exactly once; untouched pixels are DERIVED (the implication
pattern), never stored. Present = read a window back out of
the store and materialize it as a PNG.

Canvas 65,536 x 65,536 RGBA = 17,179,869,184 logical bytes
(16 GiB — larger than the test PC's 12 GiB of VRAM):

- PC: empty store 79 B; draw 53.8 s; 1.74 GB stored; every
  presented window pixel-exact against an independent
  analytic render, including a window of pure derived
  canvas, and again after close/reopen. ALL PASS.
- Phone (Pixel 6a, Termux, Christopher's own run): the same
  full canvas; draw 107.7 s; store 4.68 GB (per-shape flush
  rewrites overlapping rows — the block count, 6,645, is
  identical); all windows pixel-exact; reopen exact. The
  presented frame came out at 974,100 bytes, exactly the
  PC frame's size. ALL PASS.

## The GPU partition — built, VM-verified, phone run pending

`gpu_partition.py`: a quarter of the CPU's cores (cores//4,
pinned to the performance cluster on ARM) form a tile-render
pool — GPU-style: 1024x256 tiles rendered concurrently,
painter order preserved inside each tile, single writer into
the framebuffer. The serial one-core draw is the baseline;
the partition must match it pixel-for-pixel.

VM smoke test (2 vCPUs, partition = 1 core): ALL PASS,
partition 1.6x the serial time even single-staffed (the tile
renderer is vectorized and never draws a row twice: 5,209
row writes vs the serial path's 10,235 for the same picture).
The phone run (`python gpu_partition.py --phone`) is the
measurement that counts and is Christopher's to make.

Honest scope, same as EXP_VRAM.md: this is a renderer and a
memory model that programs written against it can use. It is
not an OS graphics driver; unmodified apps cannot see it.

## Also today: single-bit exponentiation

`single_bit_exponentiation.py` (+ log): Christopher asked
whether the growth equation W(n) = 2^(2^(n+1) - 1) can start
from one generating bit — he had thought 4 was the floor.
The fan-out construction generalizes (s=4 reproduces canon
exactly, regression-gated). From g=1 the widths follow the
equation — 1, 1, 2, 8, 128, 32768 — and the counting laws
hold, but no pattern varies across positions: one seed bit is
one block, so every layer is that bit repeated. Growth works
from 1; differentiation needs at least 2 seed bits. The
floor he sensed is structural, not arithmetic.
