from engine.runtime import Runtime


MODEL_PATH = "models/qwen"

runtime = Runtime(MODEL_PATH)
prompt = 'write a simple minecraft clone'
output = runtime.generate(prompt=prompt, max_new_tokens=40)

print("\n--- Output ---")
print(output)