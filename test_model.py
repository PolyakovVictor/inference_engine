from tinygrad.tensor import Tensor

from engine.transformer import TinyLLM, SimpleTokenizer


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

tokens = Tensor([[11,20,30,40]])
result = model.generate(tokens, 10)
tokenizer = SimpleTokenizer()
encoded = tokenizer.encode("hello world")
print(encoded)
print(tokenizer.decode(encoded))