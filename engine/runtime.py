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

    def load_weights(self) -> None:
        state_dict = safe_load(str(self.model_dir / "model.safetensors"))
        cleaned_state = {k.removeprefix("model."): v for k,v in state_dict.items()}
        if "lm_head.weight" not in cleaned_state and "embed_tokens.weight" in cleaned_state:
            cleaned_state["lm_head.weight"] = cleaned_state["embed_tokens.weight"]
        load_state_dict(self.model, cleaned_state)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        tokens = self.tokenizer.encode(prompt)
        input_tensor = Tensor([tokens])

        if DEBUG > 0: print(f"[DEBUG] Initial tokens: {tokens}")
        for _ in range(max_new_tokens):
            logits = self.model(input_tensor)
            next_tokens = logits[:, -1, :].argmax(axis=-1).item()
            tokens.append(next_tokens)
            if next_tokens == self.tokenizer.eos_ids: break
            input_tensor = Tensor([tokens])
        return self.tokenizer.decode(tokens)