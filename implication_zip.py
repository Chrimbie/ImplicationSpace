"""implication_zip.py - a zip-like compressor whose engine is
Christopher's fan-out equation, run in reverse.

The equation (his, derived 2026-10-21): a pattern of width
W(n) = 2^(2^(n+1) - 1) is exactly the expansion of a controller
of width W(n-1): out[p] = controller[p // (W(n)/W(n-1))].
Widths: seed 4 -> 8 -> 128 -> 32768 bits.

Synthesis (his direction): controller -> pattern, free.
Analysis (this compressor): pattern -> controller + deviation.
For each 32768-bit chunk:
  controller_j = majority bit of group j        (128 bits)
  deviation    = chunk XOR expand(controller)   (32768 bits)
and the controller is itself analyzed the same way, down to
the 4-bit seed. Decompression re-expands the seed and XORs
each layer's deviation back in - byte-identical by
construction, the same containment shape as the cloud store.

Each deviation is stored in the cheapest of three modes:
  ZERO   - the layer IS the derivation; nothing stored
  SPARSE - positions of the deviation's 1-bits, gap-varinted
  RAW    - the deviation bits themselves
Mode choice is per layer per chunk, measured, not assumed
(EMPIRICAL mechanism, like zip's stored fallback: data that
shares nothing with the rule pays full price, data that is
the rule pays ~nothing). Controller tie-break (majority tie
-> group's first bit) is EMPIRICAL.

Format IMPZ1: magic(5) + orig_len u64le, then per chunk:
seed byte, then for W in (8, 128, 32768): mode byte, and for
SPARSE a u32le payload length + varint stream, for RAW the
W/8 bytes, for ZERO nothing.
"""
import struct
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIDTHS = [4, 8, 128, 32768]      # seed, then the layers
CHUNK_BITS = WIDTHS[-1]
CHUNK_BYTES = CHUNK_BITS // 8    # 4096
MAGIC = b"IMPZ1"
ZERO, SPARSE, RAW = 0, 1, 2


def expand(ctrl_bits, w_from, w_to):
    """His rule: out[p] = ctrl[p // (w_to / w_from)]."""
    return np.repeat(ctrl_bits, w_to // w_from)


def analyze(bits, w_idx):
    """Pattern -> (seed, [deviation per layer ascending]).
    bits: uint8 0/1 array of width WIDTHS[w_idx]."""
    if w_idx == 0:
        return bits.copy(), []
    w = WIDTHS[w_idx]
    s = WIDTHS[w_idx - 1]
    g = w // s
    groups = bits.reshape(s, g)
    sums = groups.sum(axis=1)
    ctrl = np.where(sums * 2 > g, 1,
                    np.where(sums * 2 < g, 0,
                             groups[:, 0])).astype(np.uint8)
    dev = bits ^ expand(ctrl, s, w)
    seed, devs = analyze(ctrl, w_idx - 1)
    return seed, devs + [dev]


def synthesize(seed, devs):
    """(seed, deviations) -> pattern bits, inverse of analyze."""
    bits = seed
    for i, dev in enumerate(devs):
        w = WIDTHS[i + 1]
        bits = expand(bits, WIDTHS[i], w) ^ dev
    return bits


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf, pos):
    n = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not (b & 0x80):
            return n, pos
        shift += 7


def encode_dev(dev):
    """Cheapest of ZERO / SPARSE / RAW for one deviation."""
    nz = np.nonzero(dev)[0]
    if len(nz) == 0:
        return bytes([ZERO])
    raw = np.packbits(dev).tobytes()
    gaps = np.diff(np.concatenate(([-1], nz)))
    sparse = b"".join(_varint(int(gp - 1)) for gp in gaps)
    if len(sparse) + 4 < len(raw):
        return (bytes([SPARSE]) + struct.pack("<I", len(sparse))
                + sparse)
    return bytes([RAW]) + raw


def decode_dev(buf, pos, width):
    mode = buf[pos]
    pos += 1
    if mode == ZERO:
        return np.zeros(width, dtype=np.uint8), pos
    if mode == RAW:
        n = width // 8
        dev = np.unpackbits(np.frombuffer(
            buf[pos:pos + n], dtype=np.uint8)).astype(np.uint8)
        return dev, pos + n
    (ln,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    end = pos + ln
    dev = np.zeros(width, dtype=np.uint8)
    p = -1
    while pos < end:
        gap_m1, pos = _read_varint(buf, pos)
        p += gap_m1 + 1
        dev[p] = 1
    return dev, pos


def compress(data: bytes) -> bytes:
    out = bytearray(MAGIC)
    out += struct.pack("<Q", len(data))
    for off in range(0, len(data), CHUNK_BYTES):
        chunk = data[off:off + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            chunk = chunk + bytes(CHUNK_BYTES - len(chunk))
        bits = np.unpackbits(np.frombuffer(
            chunk, dtype=np.uint8)).astype(np.uint8)
        seed, devs = analyze(bits, len(WIDTHS) - 1)
        seed_byte = 0
        for b in seed:
            seed_byte = (seed_byte << 1) | int(b)
        out.append(seed_byte << 4)          # seed in high nibble
        for dev in devs:                    # layers 8, 128, 32768
            out += encode_dev(dev)
    return bytes(out)


def decompress(blob: bytes) -> bytes:
    assert blob[:5] == MAGIC, "not an IMPZ1 stream"
    (orig_len,) = struct.unpack_from("<Q", blob, 5)
    pos = 13
    out = bytearray()
    n_chunks = (orig_len + CHUNK_BYTES - 1) // CHUNK_BYTES
    for _ in range(n_chunks):
        seed_byte = blob[pos]
        pos += 1
        seed = np.array([(seed_byte >> (7 - i)) & 1
                         for i in range(4)], dtype=np.uint8)
        devs = []
        for w in WIDTHS[1:]:
            dev, pos = decode_dev(blob, pos, w)
            devs.append(dev)
        bits = synthesize(seed, devs)
        out += np.packbits(bits).tobytes()
    return bytes(out[:orig_len])


def _bench():
    import io
    import os
    import zipfile
    from exp_blockstore import derive_default

    cases = {}
    here = os.path.dirname(os.path.abspath(__file__))
    cases["source code (text)"] = open(
        os.path.join(here, "cloud_server.py"), "rb").read()
    rng = np.random.default_rng(7)
    cases["random 256 KiB"] = rng.integers(
        0, 256, 256 * 1024, dtype=np.uint8).tobytes()
    cases["zeros 256 KiB"] = bytes(256 * 1024)
    buf = bytearray()
    off = 0
    while len(buf) < 256 * 1024:
        buf += bytes(derive_default(off, 65536))
        off += 65536
    cases["EVB hash background 256 KiB"] = bytes(buf[:256 * 1024])
    # fan-out data: each 4096B chunk expanded from a random
    # 128-bit controller by HIS rule (out[p] = c[p // 256])
    chunks = []
    for _ in range(64):
        ctrl = rng.integers(0, 2, 128, dtype=np.uint8)
        chunks.append(np.packbits(np.repeat(ctrl, 256)).tobytes())
    cases["fan-out derived 256 KiB"] = b"".join(chunks)
    near = bytearray(cases["fan-out derived 256 KiB"])
    flip = rng.choice(len(near), len(near) // 100, replace=False)
    for i in flip:
        near[i] ^= 0xFF
    cases["fan-out + 1% noise"] = bytes(near)
    png = os.path.join(here, "framebuffer_present_pc.png")
    if os.path.exists(png):
        cases["presented frame (PNG)"] = open(png, "rb").read()

    print(f"{'case':28} {'raw':>10} {'IMPZ1':>10} {'ratio':>7} "
          f"{'zip':>10} {'roundtrip':>9}")
    ok_all = True
    for name, data in cases.items():
        blob = compress(data)
        back = decompress(blob)
        ok = back == data
        ok_all &= ok
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("x", data)
        ratio = len(blob) / len(data) if data else 1.0
        print(f"{name:28} {len(data):>10} {len(blob):>10} "
              f"{ratio:>7.3f} {len(zb.getvalue()):>10} "
              f"{'PASS' if ok else 'FAIL':>9}")
    edge = [b"", b"x", b"hello world" * 3, bytes(range(256))]
    for e in edge:
        ok_all &= decompress(compress(e)) == e
    print("edge cases (empty/1B/33B/256B):",
          "PASS" if all(decompress(compress(e)) == e for e in edge)
          else "FAIL")
    print("ALL ROUND-TRIPS:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "bench":
        _bench()
    elif len(sys.argv) == 4 and sys.argv[1] == "c":
        open(sys.argv[3], "wb").write(
            compress(open(sys.argv[2], "rb").read()))
    elif len(sys.argv) == 4 and sys.argv[1] == "d":
        open(sys.argv[3], "wb").write(
            decompress(open(sys.argv[2], "rb").read()))
    else:
        print("usage: implication_zip.py bench | "
              "c <in> <out.impz> | d <in.impz> <out>")
