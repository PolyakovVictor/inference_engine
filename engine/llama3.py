import subprocess

from pathlib import Path
from typing import List, Sequence, Union
import json, argparse, random, time, os
# from extra.models.llama import Transformer, convert_from_huggingface, convert_from_gguf, fix_bf16
from tinygrad.llm.gguf import gguf_load
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters, Variable
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm
# from extra.bench_log import BenchEvent, WallTimeEvent


MODEL_PARAMS = {
  "1B": {
    "args": {"dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192},
    "files": 1
  },
  "8B": {
    "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336},
    "files": 1
  },
  "70B": {
    "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 28672},
    "files": 8
  },
  "405B": {
    "args": {"dim": 16384, "n_heads": 128, "n_kv_heads": 8, "n_layers": 126, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 53248},
    "files": 191
  },
}

def precompute_freqs_cis(dim:int, end:int, theta:float=10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)

def complex_mult(A, c, d):
    a,b = A[..., 0:1], A[..., 1:2]
    ro = a*c - b*d
    co = a*d + b*c
    return ro.cat(co, dim=-1)

def apply_rotate_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
    assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)
    assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult(xq,c,d)
    xk_out = complex_mult(xk,c,d)
    return xq_out.flatten(3), xk_out.flatten(3)


def repeat_kv(x: Tensor, n_rep:int) -> Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1: return x
    return x.repeat((1,1,1,n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Tokenizer:
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    def __init__(self, model_path: str) -> None:
        print(f"{model_path=}")
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        mergeable_ranks = load_tiktoken_bpe(model_path)
        self.num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, 256-5)
        ]
        self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}
        self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)
    
    @property
    def bos_id(self): return self.special_tokens["<|begin_of_text|>"]
    @property
    def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id]>"]}
    def encode(self, text: str): return self.model.encode(text=text)
    def decode(self, tokens: Sequence): return self.model.decode(tokens)


def load(fn: str):
    if fn.endswith('.index.json'):
        with open(fn) as fp: weight_map = json.load(fp)['weight_map']
        parts = {n:load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
        return {k: parts[n][k] for k,n in weight_map.items()}
    elif fn.endswith(".gguf"):
        gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
        return gguf_load(gguf_tensor)[1]
    elif fn.endswith(".safetensors"):
        return safe_load(fn)
    else:
        torch_load(fn)


class Attention:
    def __init__(self, dim: int, n_heads: int, max_content=0, linear=nn.Linear, qk_norm: float | None = None, n_kv_heads: int = 8) -> None:
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads # count params and count experts
        self.n_rep = self.n_heads // self.n_kv_heads # probably n_kv_heads
        self.max_content = max_content

        if os.getenv("WQKV", None): 
            self.wqkv = linear(dim, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim * 2, bias=False)
        else:
            self.wq = linear(dim, self.n_heads*self.head_dim, bias=False)
            self.wk = linear(dim, self.n_kv_heads*self.head_dim, bias=False)
            self.wv = linear(dim, self.n_kv_heads*self.head_dim, bias=False)
        
        self.wo = linear(self.n_heads*self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(dim, qk_norm) if qk_norm is not None else None 
        self.k_norm = nn.RMSNorm(dim, qk_norm) if qk_norm is not None else None 
    
    def __call__(self, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Tensor|None=None) -> Tensor:
        if os.getenv("WQKV"):
            xqkv = self.wqkv(x)
            xqkv = xqkv.reshape(xqkv.shape[0], xqkv.shape[1], self.n_kv_heads, self.n_rep + 2, self.head_dim)
            xq = xqkv[:,:,:,:self.n_rep].reshape(xqkv.shape[0], xqkv.shape[1], -1)
            xk = xqkv[:,:,:,self.n_rep:self.n_rep+1].reshape(xqkv.shape[0], xqkv.shape[1], -1)
            xv = xqkv[:,:,:,self.n_rep+1:self.n_rep+2].reshape(xqkv.shape[0], xqkv.shape[1], -1)
        else:
            xq,xk,xv = self.wq(x), self.wk(x.contiguous_backward()), self.wv(x)
        
        if self.q_norm is not None and self.k_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        
        if x.dtype == dtypes.bfloat16: xq, xk = xq.contiguous_backward(), xk.contiguous_backward()

        xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
        xk = xk.reshape(xq.shape[0], xq.shape[1], self.n_kv_heads, self.head_dim)
        xv = xv.reshape(xq.shape[0], xq.shape[1], self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotate_emb(xq, xk, freqs_cis)
        bsz, seqlen, _, _ = xq.shape
        
        if self.max_content:
            if not hasattr(self, "cache_kv"):
                self.cache_kv = Tensor.zeros(2, bsz, self.max_content, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
                if isinstance(x.device, tuple): self.cache_kv.shard_((x.device), axis=3 if os.getenv("SHARD_KVCACHE") else None).realize()
            assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"{xk.dtype=}, {xv.dtype=}, {self.cache_kv.dtype=}"
            self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :].assign(Tensor.stack(xk, xv)).realize()
            
            keys = self.cache_kv[0, :, 0:start_pos+seqlen, :, :]
            values = self.cache_kv[1, :, 0:start_pos+seqlen, :, :]
        else:
            assert start_pos == 0
            keys, values = xk, xv
        
        if self.max_content:
            keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
            xq, keys, values = xq.transpose(1, 2), keys.transpose(1,2), values.transpose(1,2)
            attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1,2)
        else:
            xq,keys,values = xq.transpose(1,2), keys.transpose(1,2), values.transpose(1,2)
            attn = xq.scaled_dot_product_attention(keys, values, is_causal=True, enable_gqa=True).transpose(1,2)
        attn = attn.reshape(bsz, seqlen, -1)
        return self.wo(attn)


class FeedForward:
    def __init__(self, dim:int, hidden_dim:int, linear=nn.Linear) -> None:
        self.w1 = linear(dim, hidden_dim, bias=False)
        self.w2 = linear(hidden_dim, dim, bias=False)
        self.w3 = linear(dim, hidden_dim, bias=False)
    
    def __call__(self, x:Tensor) -> Tensor:
        w1 = self.w1(x).silu()
        w3 = self.w3(x.contiguous_backward())
        return self.w2(w1*w3)


class TransformerBlock:
    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, norm_eps, max_context: int, linear: nn.Linear,
                 feed_forward=FeedForward) -> None:
        self.attention = Attention(dim, n_heads, max_context, n_kv_heads=n_kv_heads)
        self.feed_forward = feed_forward(dim, hidden_dim, linear)
        self.attention_norm = nn.RMSNorm(dim, norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    
    def __call__(self, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Tensor|None):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return (h+self.feed_forward(self.ffn_norm(h))).contiguous().contiguous_backward()

# TODO add a output 
class Transformer:
    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_layers: int, n_kv_heads: int, norm_eps: float, rope_theta: int,
                 vocab_size: int, max_context: int = 8192) -> None:
        self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context, linear=nn.Linear) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim, norm_eps)
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.max_context = max_context
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context+2, rope_theta).contiguous().contiguous_backward()

    def forward(self, tokens: Tensor, start_pos: Union[Variable, int], temperature: float = 0.2):
        # 1 - tokens to vectors
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens).contiguous()
        freqs_cis = self.freqs_cis.cast(h.dtype)[:, start_pos:start_pos+seqlen, :, :, :]
        for l in self.layers: h = l(h, start_pos, freqs_cis, None)
        logits = self.output(self.norm(h).contiguous().contiguous_backward()).contiguous_backward()
        import math
        if math.isnan(temperature): return logits
        return logits[:, -1, :].flatten().argmax() # TODO add sampling


    def __call__(self, tokens:Tensor, start_pos:int, ):
        return self.forward(tokens, start_pos)


def build_transformer(model_path: Path, model_size: str = "8B", load_weights: bool = True):
    model = Transformer(**MODEL_PARAMS[model_size]["args"])
    if not load_weights: return model
    if model_path.is_dir():
        if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
    else:
        weights = load(str(model_path))
    # weights = fix_bf16(weights) # TODO
    with Context(BEAM=0):
        load_state_dict(model, weights, strict=False, consume=False)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="Path to model")
    parser.add_argument("--download", help="Need to download model?", default=False, action="store_true")
    parser.add_argument("--temperature", help="Temperature", default=0.7, type=float)
    parser.add_argument("--size", help="Model size", choices=["1B", "8B", "70B"], default="8B")


    args = parser.parse_args()
    if args.download: subprocess.run("curl -O -L https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model")
    assert args.model, "Please provide model via --model"
    tokenizer = Tokenizer(model_path=f"./{args.model}/tokenizer.model")
    tokens = tokenizer.encode('tell me some joke')
    
    TEMPERATURE = args.temperature
    print(f"seed = {Tensor._seed}\nTemperature = {TEMPERATURE}")

    model = build_transformer(model_path=args.model, model_size=args.size)
    logits = model(Tensor([tokens]), 0)
    print(f"{logits.numpy()=}")
    print(f"{tokenizer.decode([logits.item()])=}")
    print(f"{logits=}")
    print(f"test decode {tokenizer.decode([91729])}")
