from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict

from engine.config import ModelConfig
from engine.transformer import Transformer


MODEL_DIR = Path('models/qwen')

config = ModelConfig.from_json(MODEL_DIR / "config.json")
model = Transformer(config)

state_dict = safe_load(str(MODEL_DIR / "model.safetensors"))

cleaned_state = {}
for k,v in state_dict.items():
    new_k = k.removeprefix("model.")
    cleaned_state[new_k] = v

if "lm_head.weight" not in cleaned_state and "embed_tokens.weight" in cleaned_state:
    cleaned_state["lm_head.weight"] = cleaned_state["embed_tokens.weight"]

# cleaned_state["embed_tokens"] = cleaned_state["embed_tokens.weight"]

model_state_keys = set(get_state_dict(model).keys())
loaded_keys = set(cleaned_state.keys())

missing = model_state_keys - loaded_keys
unexpected = loaded_keys - model_state_keys

if missing:
    print(f"Missing keys ({len(missing)}): {missing}")
if unexpected:
    print(f"Unexpected keys ({len(unexpected)}): {unexpected}")

if not missing:
    load_state_dict(model, cleaned_state)
    print("The model is successfully uploaded")
