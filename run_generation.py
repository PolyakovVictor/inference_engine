import time
import sys

from engine.runtime import Runtime

START_TIME = time.time()
MODEL_PATH = "models/qwen"

runtime = Runtime(MODEL_PATH)
prompt = input("Enter some prompt: ")
prompt = prompt if prompt else 'write a simple minecraft clone'
output = runtime.generate(prompt=prompt, max_new_tokens=40)

print("\n--- Output ---")
print(output)

print(f"\n--- Time: {time.time() - START_TIME} ---")