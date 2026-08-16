from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    n_heads: int
    hidden_dim: int
    n_layers: int
    max_seq_len: int = 1024
