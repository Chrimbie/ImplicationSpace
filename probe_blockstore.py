import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_blockstore import BlockStore

P = "probe_store.evb2"
CHUNK = 64 << 20
TOTAL = 1 << 30

try:
    os.remove(P)
except FileNotFoundError:
    pass

buf = bytearray(CHUNK)
for i in range(0, CHUNK, 4096):
    buf[i:i + 32] = hashlib.sha256(i.to_bytes(8, "little")).digest()
buf = bytes(buf)

st = BlockStore.create(P, 1 << 60)
t0 = time.time()
pos = 0
while pos < TOTAL:
    st.write(pos, buf)
    pos += CHUNK
st.save()
wt = time.time() - t0
print(f"WRITE {TOTAL/2**30:.1f} GiB in {wt:.1f}s = "
      f"{TOTAL/wt/1e6:.0f} MB/s | stored {st.stored_bytes():,} B",
      flush=True)

t0 = time.time()
pos = 0
ok = True
while pos < TOTAL:
    data = st.read(pos, CHUNK)
    if data != buf:
        ok = False
        break
    pos += CHUNK
rt = time.time() - t0
print(f"READ {TOTAL/2**30:.1f} GiB in {rt:.1f}s = "
      f"{TOTAL/rt/1e6:.0f} MB/s | byte-exact {ok}", flush=True)
st.close()
os.remove(P)
print("probe done")
