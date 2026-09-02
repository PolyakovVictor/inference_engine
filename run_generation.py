import os
import time
import sys

from engine.runtime import Runtime

START_TIME = time.time()
MODEL_PATH = "models/Llama-3-8B"
TOKENS = int(os.getenv("TOKENS", 100))

runtime = Runtime(MODEL_PATH)
runtime.start_conversation()
system_prompt = "You are a helpful assistant."
first_turn = True
while True:
    user_prompt = input("Enter a prompt:") 
    if first_turn:
        delta = runtime.format_turn("system", system_prompt) + runtime.format_turn("user", user_prompt, add_generation_prompt=True)
        first_turn = False
    else:
        delta = runtime.format_turn("user", user_prompt, add_generation_prompt=True)
    print(f"Delta: {delta}")
    output = runtime.generate(user_text=delta, max_new_tokens=TOKENS)

    print("\n--- Output ---")
    print(output)
    runtime.generate(user_text=runtime.format_turn("assistant", output).removesuffix("<|assistant|>\n"), max_new_tokens=0)

    print(f"\n--- Time: {time.time() - START_TIME} ---")
