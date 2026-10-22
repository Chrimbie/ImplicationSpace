import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gguf import GGUFReader

import codellama70b_expvram_block as base
from run_inference_from_space import dequant_q6_k

try:
    import gguf.quants as gq
    print("gguf.quants available:",
          [n for n in dir(gq) if "6" in n or "dequant" in n.lower()])
except Exception as e:
    print("gguf.quants import failed:", e)
    gq = None

path = base.find_blob()
r = GGUFReader(str(path), "r")
t = [x for x in r.tensors
     if base.gguf_type(x) == "Q6_K"][0]
print("tensor:", t.name, "dims:", base.tensor_dims(t))
cols, rows = base.tensor_dims(t)
raw = np.asarray(t.data).view(np.uint8).reshape(-1)
packed_row = cols * 210 // 256

mine = dequant_q6_k(raw[:8 * packed_row], 8, cols)
print("mine: mean %.5f std %.5f min %.3f max %.3f"
      % (mine.mean(), mine.std(), mine.min(), mine.max()))
print("row stds:", [round(float(x), 4) for x in mine.std(axis=1)])

if gq is not None and hasattr(gq, "dequantize"):
    try:
        ref = gq.dequantize(
            raw[:8 * packed_row].reshape(8 * (cols // 256), 210),
            t.tensor_type)
        ref = np.asarray(ref, dtype=np.float32).reshape(8, cols)
        print("ref : mean %.5f std %.5f min %.3f max %.3f"
              % (ref.mean(), ref.std(), ref.min(), ref.max()))
        print("max |mine - ref|:", float(np.abs(mine - ref).max()))
    except Exception as e:
        print("gguf.quants.dequantize failed:", type(e).__name__, e)
