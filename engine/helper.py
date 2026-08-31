import os
from pathlib import Path


DEBUG = int(os.getenv('DEBUG', 0))

def resolve_chat_template(model_dir: Path | str) -> str:
    if isinstance(model_dir, str): model_dir = Path(model_dir)
    name = model_dir.name.lower()
    templates = {
        'chatml': ['chatml', 'qwen'],
        'llama3': ['llama3', 'llama-3'],
    }
    for k,v in templates.items():
        if name in v: return k
    return "chatml"
