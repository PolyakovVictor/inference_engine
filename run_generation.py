import time
import sys

from engine.runtime import Runtime

START_TIME = time.time()
MODEL_PATH = "models/tinyllama"

runtime = Runtime(MODEL_PATH)
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]
while 1:
    print(runtime.tokenizer.tokenizer.encode("hello").ids[:3])
    user_prompt = input("Enter a prompt:") 
    messages.append({"role": "user", "content": user_prompt})
    prompt = runtime.format_chat(messages=messages)
    output = runtime.generate(prompt=prompt, max_new_tokens=40)

    print("\n--- Output ---")
    print(output)
    messages.append({"role": "assistant", "content": output})

    print(f"\n--- Time: {time.time() - START_TIME} ---")
