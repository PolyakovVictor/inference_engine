from typing import Any
from tinygrad import Variable
from tinygrad.engine.jit import TinyJit
from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict

from engine.helper import DEBUG
from engine.transformer import Transformer
from engine.tokenizer import Tokenizer
from engine.config import ModelConfig


class Runtime:
    def __init__(self, model_path: str | Path) -> None:
        self.model_dir = Path(model_path)
        self.config = ModelConfig.from_json(self.model_dir / "config.json")
        self.tokenizer = Tokenizer(self.model_dir / "tokenizer.json")
        self.model = Transformer(self.config)
        self.load_weights()
        self.max_context = self.config.max_position_embeddings
        self._jit_decode = TinyJit(self._decode_step)

    def _decode_step(self, tokens: Tensor, start_pos: Any):
        return self.model(tokens=tokens, start_pos=start_pos).realize()

    def load_weights(self) -> None:
        state_dict = safe_load(str(self.model_dir / "model.safetensors"))
        cleaned_state = {k.removeprefix("model."): v for k,v in state_dict.items()}
        if "lm_head.weight" not in cleaned_state and "embed_tokens.weight" in cleaned_state:
            cleaned_state["lm_head.weight"] = cleaned_state["embed_tokens.weight"]
        load_state_dict(self.model, cleaned_state, strict=False)
    
    def sample(self, logits: Tensor, temperature: float = 0.7) -> int:
        if temperature == 0: return int(logits.argmax().item())
        probs = (logits / temperature).softmax(axis=-1)
        return int(probs.multinomial().item())

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        self.model.reset_cache()
        tokens = self.tokenizer.encode(prompt)
        input_tensor = Tensor([tokens])
        logits = self.model(input_tensor, start_pos=0)
        next_tokens = int(logits[0,-1].argmax().item())

        generated = [next_tokens]
        start_pos = len(tokens)

        if DEBUG > 0: print(f"[DEBUG] Initial tokens: {tokens}")
        for _ in range(max_new_tokens):
            if next_tokens in self.tokenizer.eos_ids:
                break
            input_tensor = Tensor([[next_tokens]])
            var = Variable("start_pos", 1, self.max_context).bind(start_pos)
            logits = self._jit_decode(input_tensor, var)
            next_tokens = self.sample(logits[0,-1])
 
            generated.append(next_tokens)
            start_pos += 1
        return self.tokenizer.decode(tokens + generated)
