from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load

from engine.transformer import TinyLLM
from engine.tokenizer import SimpleTokenizer
from engine.config import ModelConfig


VOCAB_SIZE = 100
DIM = 64
HEADS = 4
HIDDEN_DIM = 256
LAYERS = 2
PATH_TO_STATE = 'models/tinyllm.safetensor'

config = ModelConfig(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    n_heads=HEADS,
    hidden_dim=HIDDEN_DIM,
    n_layers=LAYERS
)

model = TinyLLM(config)

model.load(PATH_TO_STATE)
tokens = Tensor([[11,20,30,40]])
result = model.generate(tokens, 10)
tokenizer = SimpleTokenizer(vocab={'<unk>': 0})
encoded = tokenizer.encode("hello world")
model.save(PATH_TO_STATE)