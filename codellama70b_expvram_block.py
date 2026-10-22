"""No-flags experiment: one real CodeLlama 70B GGUF block, token position zero.
Not complete inference; weights come from existing GGUF disk mmap and are NOT
compressed by implication. EXP-VRAM is a logical-address adapter, not a CUDA driver.
Requires: pip install gguf numpy torch
"""
import json, os, time, hashlib
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader

MODEL = 'codellama:70b'
TILE_ROWS = 32
LOGICAL_ENVELOPE = 1 << 60  # 1 EiB, logical addresses only
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def find_blob():
    root = Path.home() / '.ollama' / 'models'
    manifests = list((root / 'manifests').rglob('70b')) if (root / 'manifests').exists() else []
    for m in manifests:
        if 'codellama' not in str(m).lower():
            continue
        obj = json.loads(m.read_text(encoding='utf-8'))
        for layer in obj.get('layers', []):
            if 'model' in layer.get('mediaType', ''):
                p = root / 'blobs' / layer['digest'].replace(':', '-')
                if p.exists(): return p
    prefix = 'sha256-1436d66b69757a245f02d000874c670507949d11ad5c188a623652052c6aa508'
    p = root/'blobs'/prefix
    if p.exists(): return p
    raise FileNotFoundError('Cannot locate codellama:70b GGUF in Ollama models')


class LogicalGGUF:
    """Only the model's true on-disk bytes are accessible, mapped at logical offsets.
    No fabricated model values, and no claim of replacing physical weight storage.
    """
    def __init__(self, reader):
        self.tensors = {t.name: t for t in reader.tensors}
        self.address = {}
        offset = 0
        for name,t in self.tensors.items():
            raw = np.asarray(t.data).view(np.uint8).reshape(-1)
            self.address[name] = (offset, raw)
            offset += len(raw)
        if offset > LOGICAL_ENVELOPE: raise ValueError('Logical envelope too small')
        self.logical_size = LOGICAL_ENVELOPE
        self.stored_bytes = offset
        self.calls = 0
        self.served = 0
    def read(self, name, start, length):
        base, raw = self.address[name]
        if start < 0 or start + length > len(raw): raise IndexError(name)
        self.calls += 1
        self.served += length
        # Position in the 1 EiB logical envelope is base+start.
        virtual_address = base + start
        assert virtual_address + length <= self.logical_size
        return raw[start:start+length]


_GGML = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0',
         7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K',
         12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K'}


def gguf_type(t):
    tt = t.tensor_type
    name = getattr(tt, 'name', None)
    if name:
        return name
    try:
        return _GGML[int(tt)]
    except Exception:
        return str(tt).split('.')[-1]


def dequant_q4_0(data, nrows, cols):
    """Exact GGUF Q4_0 block layout: 2 bytes fp16 scale + 16 packed nibbles.
    Each block's low nibbles represent first 16 values, high next 16.
    """
    blocks = cols // 32
    a = np.asarray(data, dtype=np.uint8).reshape(nrows, blocks, 18)
    d = np.ascontiguousarray(a[:,:,:2]).view('<f2').astype(np.float32).reshape(nrows, blocks, 1)
    q = a[:,:,2:]
    low = (q & 15).astype(np.int16) - 8
    high = (q >> 4).astype(np.int16) - 8
    return (np.concatenate((low, high), axis=2).astype(np.float32) * d).reshape(nrows,cols)


def tensor_dims(t):
    # GGUFReader shape is file-order: [input_cols, output_rows] for matrices.
    return [int(x) for x in t.shape]


def linear(x, store, name):
    t = store.tensors[name]
    dims = tensor_dims(t)
    typ = gguf_type(t)
    if len(dims) != 2: raise ValueError(f'{name}: not 2D, {dims}')
    cols, rows = dims
    if x.numel() != cols: raise ValueError(f'{name}: input {x.numel()} != {cols}')
    if typ != 'Q4_0': raise NotImplementedError(f'{name}: quantization {typ}; this first test supports Q4_0 only')
    if cols % 32: raise ValueError('Q4_0 cols not divisible by 32')
    packed_row = cols * 18 // 32
    out = torch.empty(rows, device=DEVICE, dtype=torch.float32)
    for r in range(0,rows,TILE_ROWS):
        n = min(TILE_ROWS,rows-r)
        raw = store.read(name,r*packed_row,n*packed_row)
        weights = torch.from_numpy(dequant_q4_0(raw,n,cols)).to(DEVICE)
        out[r:r+n] = torch.mv(weights,x)
        del weights
    if DEVICE=='cuda': torch.cuda.synchronize()
    return out


def vector(store, name):
    t = store.tensors[name]
    typ = gguf_type(t)
    raw = store.read(name,0,len(store.address[name][1]))
    if typ == 'F32': return torch.from_numpy(raw.view('<f4').copy()).to(DEVICE)
    if typ == 'F16': return torch.from_numpy(raw.view('<f2').astype(np.float32)).to(DEVICE)
    raise NotImplementedError(f'{name}: norm dtype {typ}')


def rmsnorm(x,w,eps=1e-5):
    return (x * torch.rsqrt(torch.mean(x*x)+eps)) * w


def main():
    print('CodeLlama 70B: FIRST REAL TRANSFORMER BLOCK through logical EXP-VRAM addresses',flush=True)
    print('One-token test, position=0; NOT full inference and generates NO tokens.',flush=True)
    path=find_blob()
    print('GGUF:',path,'bytes:',path.stat().st_size,flush=True)
    reader=GGUFReader(str(path),'r')
    s=LogicalGGUF(reader)
    names=['blk.0.attn_norm.weight','blk.0.attn_q.weight','blk.0.attn_k.weight',
           'blk.0.attn_v.weight','blk.0.attn_output.weight','blk.0.ffn_norm.weight',
           'blk.0.ffn_gate.weight','blk.0.ffn_up.weight','blk.0.ffn_down.weight']
    for name in names:
        if name not in s.tensors: raise KeyError(f'Missing GGUF tensor: {name}')
        print(name,gguf_type(s.tensors[name]),tensor_dims(s.tensors[name]),flush=True)
    dim=tensor_dims(s.tensors['blk.0.attn_q.weight'])[0]
    g=torch.Generator(device='cpu').manual_seed(42)
    x=torch.randn(dim,generator=g).to(DEVICE).float()*0.1
    if DEVICE=='cuda': torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    norm=vector(s,names[0]); xn=rmsnorm(x,norm)
    q=linear(xn,s,names[1]); k=linear(xn,s,names[2]); v=linear(xn,s,names[3])
    # For one input token with empty cache, each attention head has exactly one
    # key: softmax([score]) = 1. RoPE at position 0 is identity, so q/k don't
    # affect this single-token attention output. Concatenated V is exact.
    n_heads=q.numel()//128
    n_kv_heads=k.numel()//128
    if n_heads % n_kv_heads != 0 or v.numel()!=k.numel():
        raise ValueError('Unexpected head structure; requires inspect metadata')
    attended=v.view(n_kv_heads,128).repeat_interleave(n_heads//n_kv_heads,dim=0).reshape(-1)
    attn=linear(attended,s,names[4]); h=x+attn
    fn=vector(s,names[5]); hf=rmsnorm(h,fn)
    gate=linear(hf,s,names[6]); up=linear(hf,s,names[7]); ff=torch.nn.functional.silu(gate)*up
    down=linear(ff,s,names[8]); y=h+down
    if DEVICE=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    result={
        'model':MODEL,'backend':'real GGUF Q4_0 reads via logical EXP-VRAM adapter; CUDA tiled matvec',
        'complete_transformer_block':True,'full_model_inference':False,'generated_tokens':0,
        'test_sequence_length':1,'test_position':0,'device':DEVICE,
        'elapsed_seconds':elapsed,'logical_envelope_bytes':s.logical_size,
        'actual_model_file_bytes':path.stat().st_size,'model_weight_bytes_saved':0,
        'weight_reads':s.calls,'weight_bytes_served':s.served,
        'output_elements':y.numel(),'output_sha256_float32':hashlib.sha256(y.cpu().numpy().astype('<f4').tobytes()).hexdigest(),
        'output_finite':bool(torch.isfinite(y).all().item()),
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if DEVICE=='cuda' else None,
    }
    print(json.dumps(result,indent=2),flush=True)
    Path('codellama70b_expvram_block_result.json').write_text(json.dumps(result,indent=2))
    print('Saved codellama70b_expvram_block_result.json',flush=True)

if __name__=='__main__':
    main()
