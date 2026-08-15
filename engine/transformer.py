from typing import Any
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear


class SelfAttention:
    def __init__(self, dim, n_heads) -> None:
        assert dim % n_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.o_proj = Linear(dim, dim)
        
    def __call__(self, x: Tensor) -> Any:
        # x = [batch, sequence, dimension]
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [batch, seq, heads, head_dim]
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores = Q @ K^T
        scores: Tensor = q.matmul(k.transpose(2,3))
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = Tensor.ones(seq_len, seq_len).tril()

        mask = mask.reshape(1,1,seq_len,seq_len)

        scores = scores.masked_fill(mask==0, float("-inf"))
        attention = scores.softmax(axis=-1)

        # Attention @ V
        output = attention.matmul(v)

        # [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        output = output.transpose(1,2)
        output = output.reshape(batch, seq_len, self.dim)
        return self.o_proj(output)


class MLP:
    def __init__(self, dim, hidden_dim) -> None:
        self.fc1 = Linear(dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, dim)

    def __call__(self, x) -> Any:
        x = self.fc1(x)
        x = x.gelu() # TODO test with quick_gelu
        x = self.fc2(x)
        return x


class TransformerBlock:
    def __init__(self, dim, n_heads, hidden_dim) -> None:
        self.attention = SelfAttention(dim, n_heads)
        self.mlp = MLP(dim, hidden_dim)
    
    def __call__(self, x: Tensor) -> Any:
        # Attention + residual
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

#              x
#              │
#       ┌──────┴──────┐
#       │             │
#       ▼             │
#   Attention         │
#       │             │
#       └──────┐      │
#              ▼      │
#              + ◄────┘
#              │
#       ┌──────┴──────┐
#       │             │
#       ▼             │
#          MLP        │
#       │             │
#       └──────┐      │
#              ▼      │
#              + ◄────┘
#              │
#              ▼
#              x'


class TinyLLM:
    def __init__(self, vocab_size, dim, n_heads, hidden_dim, n_layear) -> None:
        self.vocab_size = vocab_size
        self.dim = dim

        self.token_embedding = Tensor.uniform(vocab_size, dim, low=-0.1, high=0.1)
        self.position_embedding = Tensor.uniform(1064, dim, low=-0.1, high=0.1)
        self.blocks = [TransformerBlock(dim,n_heads,hidden_dim) for _ in range(n_layear)]
        self.lm_head = Linear(dim, vocab_size)
    
    def __call__(self, tokens: Tensor) -> Any:
        print(tokens.shape)
        seq_len = tokens.shape[-1]
        x = self.token_embedding[tokens]
        positions = Tensor.arange(seq_len)
        x = x + self.position_embedding[positions]
        x = x.unsqueeze(1)
        for block in self.blocks: x = block(x)
        logits = self.lm_head(x)
        return logits
