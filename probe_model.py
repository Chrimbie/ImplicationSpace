import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader

import codellama70b_expvram_block as base

path = base.find_blob()
r = GGUFReader(str(path), "r")

types = Counter()
names = []
for t in r.tensors:
    types[base.gguf_type(t)] += 1
    names.append(t.name)
print("tensor types:", dict(types))
print("has token_embd:", "token_embd.weight" in names)
print("has output:", "output.weight" in names)
print("has output_norm:", "output_norm.weight" in names)
print("blocks present:", sum(1 for n in names
                             if n.startswith("blk.79.")))

for key in ["llama.block_count", "llama.embedding_length",
            "llama.feed_forward_length",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.rope.freq_base",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.model"]:
    f = r.get_field(key)
    print(key, "=", f.contents() if f else None)

f = r.get_field("tokenizer.ggml.tokens")
toks = f.contents()
print("vocab size:", len(toks))
for cand in ["hello", "▁hello", "Hello", "▁Hello"]:
    if cand in toks:
        print(f"token {cand!r}: id {toks.index(cand)}")
# byte-level fallback check
print("id 1:", toks[1], "| id 2:", toks[2])
