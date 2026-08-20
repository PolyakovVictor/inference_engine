import math
from typing import Any
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding
from tinygrad.nn.state import get_state_dict, safe_save, load_state_dict, safe_load

from .config import ModelConfig


class SelfAttention:
    """
    Механізм Multi-Head Attention з каузальною (causal) маскою для авторегресійних моделей.
    """
    def __init__(self, dim: int, n_heads: int) -> None:
        # Розмірність моделі має націло ділитися на кількість голів
        assert dim % n_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Лінійні проєкції для векторів Query, Key, Value та фінального виходу
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.o_proj = Linear(dim, dim)
        
    def __call__(self, x: Tensor) -> Any:
        # x shape: [batch, sequence_length, dimension]
        batch, seq_len, _ = x.shape

        # 1. Проєктуємо вхідний тензор у Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Розбиваємо простір ознак на незалежні голови уваги (Multi-Head)
        # [batch, seq, dim] -> [batch, seq, n_heads, head_dim]
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # 3. Змінюємо осі для паралельного матричного множення по кожній голові
        # [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Обчислюємо ваги уваги (скалярний добуток Q і транспонованого K)
        # scores shape: [batch, n_heads, seq, seq]
        scores: Tensor = q.matmul(k.transpose(2, 3))
        
        # Масштабування (Scaled Dot-Product) для запобігання надто малим градієнтам при Softmax
        scores = scores / (self.head_dim ** 0.5)

        # 5. Каузальна маска (трикутна матриця): забороняє токенам дивитися в майбутнє
        mask = Tensor.ones(seq_len, seq_len).tril()
        mask = mask.reshape(1, 1, seq_len, seq_len)

        # Заповнюємо майбутні позиції -inf, щоб softmax перетворив їх на 0
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = scores.softmax(axis=-1)

        # 6. Зважене підсумовування значень (Values)
        # output shape: [batch, n_heads, seq, head_dim]
        output = attention.matmul(v)

        # 7. Збираємо всі голови назад в єдиний вихідний вектор розмірності dim
        # [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads, head_dim] -> [batch, seq, dim]
        output = output.transpose(1, 2)
        output = output.reshape(batch, seq_len, self.dim)
        
        # Фінальна вихідна лінійна трансформація
        return self.o_proj(output)


class MLP:
    """
    Повнозв'язна мережа (Feed-Forward Network), що обробляє кожен токен незалежно.
    """
    def __init__(self, dim: int, hidden_dim: int) -> None:
        # Збільшуємо розмірність до hidden_dim для нелінійної сепарації
        self.fc1 = Linear(dim, hidden_dim)
        # Повертаємо назад до базової розмірності dim
        self.fc2 = Linear(hidden_dim, dim)

    def __call__(self, x: Tensor) -> Any:
        x = self.fc1(x)
        x = x.gelu()  # Нелінійна функція активації GELU
        x = self.fc2(x)
        return x


class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.weight = Tensor.ones(dim)
    
    def __call__(self, x: Tensor) -> Tensor:
        variance = (x * x).mean(axis=-1, keepdim=True)
        rsqrt = (variance + self.eps).rsqrt()
        return x * rsqrt * self.weight

def rotate_half(x: Tensor) -> Tensor:
    d = x.shape[-1] // 2
    x1 = x[:, :, :, :d]
    x2 = x[:, :, :, d:]
    return (-x2).cat(x1, dim=-1)

def apply_rotary_emb(x: Tensor, start_pos: int, theta: float = 10000.0) -> Tensor:
    batch, seq_len, n_heads, head_dim = x.shape
    assert head_dim % 2 == 0
    positions = Tensor.arange(start_pos, start_pos+seq_len).reshape(seq_len, 1)
    dim_indices = Tensor.arange(0, head_dim, 2).reshape(1, head_dim // 2)
    freqs = positions * (theta ** (-dim_indices / head_dim)) # type: ignore

    emb = freqs.cat(freqs, dim=-1)
    
    sin = emb.sin().reshape(1, seq_len, 1, head_dim)
    cos = emb.cos().reshape(1, seq_len, 1, head_dim)
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

        self.q_proj = Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def __call__(self, x: Tensor, start_pos: int = 0, mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, start_pos=start_pos, theta=self.rope_theta)
        k = apply_rotary_emb(k, start_pos=start_pos, theta=self.rope_theta)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)
        
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scores = q.matmul(k.transpose(2,3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask
        
        probs = scores.softmax(axis=-1)
        output = probs.matmul(v)

        output = output.transpose(1,2).reshape(batch, seq_len, -1)
        return self.o_proj(output)


class TransformerBlock:
    def __init__(self, cfg: ModelConfig) -> None:
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
    
    def __call__(self, x: Tensor, start_pos: int = 0, mask: Tensor | None = None) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), start_pos=start_pos, mask=mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Transformer:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.embed_tokens =  Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    
    def __call__(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
        batch, seq_len = tokens.shape
        x = self.embed_tokens(tokens)

        mask = None
        if seq_len > 1:
            mask = Tensor.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)
        
        for layer in self.layers:
            x = layer(x, start_pos=start_pos, mask=mask)
        
        x = self.norm(x)
        return self.lm_head(x)
        