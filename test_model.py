from tinygrad.tensor import Tensor

from engine.transformer import TinyLLM


VOCAB_SIZE = 100
DIM = 64
HEADS = 4
HIDDEN_DIM = 256
LAYERS = 2

model = TinyLLM(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    n_heads=HEADS,
    hidden_dim=HIDDEN_DIM,
    n_layear=LAYERS
)

tokens = Tensor([10,20,30,40])
logits = model(tokens)

print(f"token shape: {tokens.shape}")
print(f"logits shape: {logits.shape}")
