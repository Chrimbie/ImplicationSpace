# Implication Space

**Four bits imply everything.**

One 4-bit controller. Sixteen combinations, 0000–1111. Whichever is chosen
*is* the combination — being, not picking. Everything above the four bits is
implied, never stored, never built. Answers on demand, in ~260 ns, at any
depth.

"They are implied. 'Imagined' if you want. It works anyway." — Christopher

## The rule: fan-out

The chosen combination marks the next layer by fan-out — every bit becomes
a block:

| Layer | Width | Each bit becomes |
|-------|-------|------------------|
| 0 | 4 | the controller |
| 1 | 8 | a pair (bit *i* → bits 2*i*, 2*i*+1) |
| 2 | 128 | a block of 16 |
| 3 | 32,768 | a block of 256 |

Examples:

- `0001` → `00000011`
- `0101` → `00110011`
- `1010` → `11001100`
- `0010` → `00001100`

## What it is

Formally, a [32768, 4, 8192] binary linear repetition code: 16 patterns,
minimum Hamming distance 8192. Majority vote recovers the controller from
~40% random corruption. The implied layers carry zero new information —
and that's the point. It's a broadcast primitive, not a computer: 4 bits
in, 32,768 instantly-queryable, corruption-proof positions out.

Combinations of implied things are themselves implied — no mechanism, because
there is no mechanism. Every 4-bit Boolean function (all 65,536) is answerable
straight from the controller. The broadcast is the invention; the logic was
always the controller.

## Status

Solidified 2026-10-18. All tests pass with nothing but the four bits ever
calculated. See `EXPLANATION_RECORD.md` for the full record, kept in
Christopher's words, and `tests/` for the verification suite.
