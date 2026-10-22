"""Containment proof: read the model's payload back out of
EXP-VRAM storage — only from the store — and reproduce the
ingest ground-truth hash."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_blockstore import BlockStore

CHUNK = 64 << 20

m = json.loads(Path("codellama70b_space_manifest.json")
               .read_text())
st = BlockStore.open(m["store"])
print(f"store: {m['store']} | envelope "
      f"{st.size:,} B | blocks stored "
      f"{st.blocks_stored():,} | file "
      f"{st.stored_bytes():,} B", flush=True)

h = hashlib.sha256()
pos = 0
total = m["payload_bytes"]
t0 = time.time()
while pos < total:
    data = st.read(pos, min(CHUNK, total - pos))
    h.update(data)
    pos += len(data)
got = h.hexdigest()
want = m["payload_stream_sha256"]
ok = got == want
print(f"read back {total:,} B in {time.time() - t0:.0f}s")
print(f"payload sha256 from space: {got}")
print(f"ingest ground truth      : {want}")
print("CONTAINMENT:", "PROVEN — every payload byte reads "
      "back from inside EXP-VRAM storage, hash-identical"
      if ok else "FAILED — hash mismatch")
st.close()
