import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=cfg.get("rope_theta", 10000.0),
            max_position_embeddings=cfg.get("max_position_embeddings", 2048),
        )
