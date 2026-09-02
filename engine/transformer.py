import math

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding
from tinygrad.dtype import dtypes

from .config import ModelConfig


class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor) -> Tensor:
        variance = (x * x).mean(axis=-1, keepdim=True)
        rsqrt = (variance + self.eps).rsqrt()
        return x * rsqrt * self.weight


def rotate_half(x: Tensor) -> Tensor:
    d = x.shape[-1] // 2 # shape = 1, 50, 32, 64 # reshape change the shape but not total count of elemetns in Tensor
    x1 = x[..., :d]
    x2 = x[..., d:]
    return (-x2).cat(x1, dim=-1)

def apply_rotary_emb(x: Tensor, start_pos: int, theta: float = 10000.0) -> Tensor:
    batch, seq_len, n_heads, head_dim = x.shape
    assert head_dim % 2 == 0
    head_dim = int(head_dim)
    seq_len = int(seq_len)

    positions = Tensor.arange(
        start_pos,
        start_pos + seq_len,
        dtype=dtypes.float32
    ).reshape(seq_len, 1)

    dim_indices = Tensor.arange(
        0,
        head_dim,
        2,
        dtype=dtypes.float32
    ).reshape(1, head_dim // 2)

    freqs = positions * (theta ** (-dim_indices / head_dim)) 
    emb = freqs.cat(freqs, dim=-1)
    sin = emb.sin().reshape(1, seq_len, 1, head_dim).cast(x.dtype)
    cos = emb.cos().reshape(1, seq_len, 1, head_dim).cast(x.dtype)

    return (x * cos) + (rotate_half(x) * sin)


class SwiGLU:
    def __init__(self, dim: int, hidden_dim: int) -> None:
        self.gate_proj = Linear(dim, hidden_dim, bias=False)
        self.up_proj = Linear(dim, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class Attention:
    def __init__(self, cfg: ModelConfig) -> None:
        self.hidden_size = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // self.n_heads
        self.num_kv_groups = self.n_heads // self.n_kv_heads
        self.rope_theta = cfg.rope_theta
        self.max_context = cfg.max_position_embeddings

        self.q_proj = Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.cache_k: Tensor = Tensor.zeros(1, self.max_context, self.n_kv_heads, self.head_dim, dtype=dtypes.float16).contiguous().realize()
        self.cache_v: Tensor = Tensor.zeros(1, self.max_context, self.n_kv_heads, self.head_dim, dtype=dtypes.float16).contiguous().realize()

    def reset_cache(self) -> None:
        self.cache_k.assign(Tensor.zeros_like(self.cache_k).contiguous())
        self.cache_v.assign(Tensor.zeros_like(self.cache_v).contiguous())

    def __call__(self, x: Tensor, start_pos: int = 0, mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, start_pos=start_pos, theta=self.rope_theta).contiguous()
        k = apply_rotary_emb(k, start_pos=start_pos, theta=self.rope_theta).contiguous()
        v = v.contiguous()
        print("=== DEBUG KV ===")
        print(f"start_pos          = {start_pos}")
        print(f"seq_len            = {seq_len}")
        print(f"batch              = {batch}")
        print(f"n_kv_heads         = {self.n_kv_heads}")
        print(f"head_dim           = {self.head_dim}")
        print(f"k.shape            = {k.shape}")
        print(f"v.shape            = {v.shape}")
        print(f"cache_k.shape      = {self.cache_k.shape}")
        print(f"cache_v.shape      = {self.cache_v.shape}")
        print("================")

        # Рахуємо новий повний кеш
        new_k_cache = self.cache_k.shrink((None, (0, start_pos), None, None)).cat(k, dim=1).pad((None, (0, self.max_context - start_pos - seq_len), None, None))
        new_v_cache = self.cache_v.shrink((None, (0, start_pos), None, None)).cat(v, dim=1).pad((None, (0, self.max_context - start_pos - seq_len), None, None))

        # Фізично записуємо його в пам'ять (обов'язково realize!)
        self.cache_k.assign(new_k_cache).realize()
        self.cache_v.assign(new_v_cache).realize()

        # Тепер дістаємо потрібний шматок для множення
        keys = self.cache_k.shrink((None, (0, start_pos + seq_len), None, None))
        values = self.cache_v.shrink((None, (0, start_pos + seq_len), None, None))

        if self.num_kv_groups > 1:
            keys = keys.repeat_interleave(self.num_kv_groups, dim=2)
            values = values.repeat_interleave(self.num_kv_groups, dim=2)

        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = q.matmul(keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        probs = scores.softmax(axis=-1)
        output = probs.matmul(values)
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output)


class TransformerBlock:
    def __init__(self, cfg: ModelConfig) -> None:
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def reset_cache(self) -> None:
        self.self_attn.reset_cache()

    def __call__(self, x: Tensor, start_pos: int = 0, mask: Tensor | None = None) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), start_pos=start_pos, mask=mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Transformer:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def reset_cache(self) -> None:
        for layer in self.layers:
            layer.reset_cache()

    def __call__(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        batch, seq_len = tokens.shape
        x = self.embed_tokens(tokens)

        mask = None

        if seq_len > 1:
            past_mask = Tensor.zeros(seq_len, start_pos, dtype=dtypes.float16)
            causal_mask = Tensor.full((seq_len, seq_len), float("-inf"), dtype=dtypes.float16).triu(1)
            mask = past_mask.cat(causal_mask, dim=-1).reshape(1, 1, seq_len, start_pos + seq_len)

        for n,layer in enumerate(self.layers):
            x = layer(x, start_pos=start_pos, mask=mask)

        x = self.norm(x)
        return self.lm_head(x)


class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(self.scale.dtype).T*self.scale)

  @staticmethod
  def quantize(tensors, device, scale_dtype=dtypes.float16, quantize_embeds=False):
    new_tensors = {}
    for name,v in tensors.items():
        is_linear = any(x in name for x in ["self_attn", "mlp", "lm_head"])
        is_embed = quantize_embeds and "embed_tokens" in name
        if (is_linear or is_embed) and name.endswith(".widht"):
            v = v.cast(scale_dtype)
            scale = v.abs().max(axis=1) / 127.0
            int8_weight = (v.T/scale).T.round().cast(dtype=dtypes.int8) # without round(), cast truncates -34.9 to -34
            new_tensors[name] = int8_weight
            new_tensors[name.replace('weight', 'scale')] = scale
            if isinstance(device, tuple):
                new_tensors[name].shard_(device, axis=-1)
                new_tensors[name.replace('weight', 'scale')].shard_(device, axis=None)
        else:
            new_tensors[name] = v
    return new_tensors


class Int8Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight, self.scale = Tensor.ones(vocab_size, embed_size, dtype=dtypes.int8), Tensor.ones(vocab_size, dtype=dtypes.half)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz).unsqueeze(-1)
    big_shp = idx.shape+(self.vocab_sz, self.embed_sz)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), (self.weight.cast(self.scale.dtype).T*self.scale).T
    return (arange == idx).mul(vals).sum(-2, dtype=vals.dtype)

