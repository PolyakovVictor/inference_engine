from tinygrad.nn.state import safe_load


weights = safe_load("models/qwen/model.safetensors")
print(f"Total keys: {len(weights)}")
for k,v in list(weights.items())[:15]:
    print(f"{k:<50} -> {v.shape} ({v.dtype})")
