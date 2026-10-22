"""Ingest the whole CodeLlama 70B into EXP-VRAM storage.

The model's tensor payloads — every learned byte — are
streamed into a BlockStore at the same logical addresses
Christopher's block-0 harness assigns them (reader order,
concatenated), inside a 2^60-byte envelope. The GGUF header
and metadata stay file-side as an index (names, shapes,
types); the weights themselves live in the space.

Ground truth recorded in the manifest: the sha256 of the
payload stream in address order, computed from the source
as it is written. The containment proof must reproduce it
reading only from the store.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gguf import GGUFReader

from exp_blockstore import BlockStore

STORE = "codellama70b.evb2"
MANIFEST = "codellama70b_space_manifest.json"
CHUNK = 64 << 20
ENVELOPE = 1 << 60


def find_blob():
    root = Path.home() / ".ollama" / "models"
    manifests = list((root / "manifests").rglob("70b"))
    for m in manifests:
        if "codellama" not in str(m).lower():
            continue
        obj = json.loads(m.read_text(encoding="utf-8"))
        for layer in obj.get("layers", []):
            if "model" in layer.get("mediaType", ""):
                p = root / "blobs" / layer["digest"].replace(":", "-")
                if p.exists():
                    return p
    raise FileNotFoundError("codellama:70b blob not found")


def main():
    path = find_blob()
    print(f"source GGUF: {path} ({path.stat().st_size:,} B)",
          flush=True)
    reader = GGUFReader(str(path), "r")

    tensors = []
    offset = 0
    for t in reader.tensors:
        raw = np.asarray(t.data).view(np.uint8).reshape(-1)
        tensors.append((t.name, offset, len(raw), raw))
        offset += len(raw)
    payload_total = offset
    print(f"{len(tensors)} tensors, payload "
          f"{payload_total:,} B", flush=True)

    if os.path.exists(STORE):
        os.remove(STORE)
    st = BlockStore.create(STORE, ENVELOPE)

    h = hashlib.sha256()
    written = 0
    last_ckpt = 0
    t0 = time.time()
    for name, off, ln, raw in tensors:
        pos = 0
        while pos < ln:
            piece = raw[pos:pos + CHUNK]
            data = piece.tobytes()
            h.update(data)
            st.write(off + pos, data)
            pos += len(data)
            written += len(data)
            if written - last_ckpt >= (4 << 30):
                st.save()
                last_ckpt = written
                el = time.time() - t0
                print(f"  {written / 2**30:.1f} GiB in "
                      f"{el:.0f}s ({written / el / 1e6:.0f} MB/s)",
                      flush=True)
    st.close()
    stored = os.path.getsize(STORE)
    manifest = {
        "model": "codellama:70b",
        "source_file": str(path),
        "source_file_bytes": path.stat().st_size,
        "payload_bytes": payload_total,
        "payload_stream_sha256": h.hexdigest(),
        "store": STORE,
        "envelope_bytes": ENVELOPE,
        "store_file_bytes": stored,
        "tensors": [{"name": n, "offset": o, "length": l}
                    for n, o, l, _ in tensors],
    }
    Path(MANIFEST).write_text(json.dumps(manifest, indent=1))
    print(f"INGEST DONE: {payload_total:,} payload bytes into "
          f"the space | store file {stored:,} B | payload "
          f"sha256 {h.hexdigest()[:16]}... | "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
