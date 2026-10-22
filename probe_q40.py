import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gguf import GGUFReader
import gguf.quants as gq

import codellama70b_expvram_block as base

path = base.find_blob()
r = GGUFReader(str(path), "r")
t = [x for x in r.tensors if x.name == "blk.0.attn_q.weight"][0]
cols, rows = base.tensor_dims(t)
raw = np.asarray(t.data).view(np.uint8).reshape(-1)
packed_row = cols * 18 // 32
mine = base.dequant_q4_0(raw[:8 * packed_row], 8, cols)
ref = np.asarray(gq.dequantize(
    raw[:8 * packed_row].reshape(8 * (cols // 32), 18),
    t.tensor_type), dtype=np.float32).reshape(8, cols)
print("Q4_0 attn_q: max |mine - ref|:",
      float(np.abs(mine - ref).max()))

t2 = [x for x in r.tensors if x.name == "token_embd.weight"][0]
print("token_embd type:", base.gguf_type(t2),
      "dims:", base.tensor_dims(t2))
