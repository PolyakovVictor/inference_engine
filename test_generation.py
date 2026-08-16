from tinygrad.tensor import Tensor

from engine.transformer import TinyLLM
from engine.tokenizer import SimpleTokenizer


vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "hello": 2,
    "world": 3,
    "I": 4,
    "am": 5,
    "a": 6,
    "robot": 7,
    "my": 8,
    "name": 9,
    "is": 10,
}

tokenizer = SimpleTokenizer(vocab)

model = TinyLLM(vocab_size=len(vocab), dim=64, n_heads=4, hidden_dim=256, n_layear=2)

text = "Hello world"
token_ids = tokenizer.encode(text)
tokens = Tensor([token_ids])
result = model.generate(tokens, max_new_tokens=5)
result_ids = result.numpy()[0].tolist()


print("generated token ids: ", result_ids)
print("generated text: ", tokenizer.decode(result_ids))