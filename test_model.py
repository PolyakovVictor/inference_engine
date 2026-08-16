from tinygrad.tensor import Tensor

from engine.transformer import TinyLLM
from engine.tokenizer import SimpleTokenizer
from engine.config import ModelConfig


VOCAB_SIZE = 100
DIM = 64
HEADS = 4
HIDDEN_DIM = 256
LAYERS = 2

config = ModelConfig(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    n_heads=HEADS,
    hidden_dim=HIDDEN_DIM,
    n_layers=LAYERS
)

model = TinyLLM(config)

tokens = Tensor([[11,20,30,40]])
result = model.generate(tokens, 10)
tokenizer = SimpleTokenizer(vocab={'<unk>': 0})
encoded = tokenizer.encode("hello world")
print(encoded)
print(tokenizer.decode(encoded))