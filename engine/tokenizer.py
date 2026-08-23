from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, tokenizer_path: Path | str) -> None: 
        self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        self.eos_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
        self.eos_ids = set()
        for t in self.eos_tokens:
            tid = self.tokenizer.token_to_id(t)
            if tid is not None: self.eos_ids.add(tid)
    def encode(self, text: str) -> list[int]: return self.tokenizer.encode(text).ids
    def decode(self, tokens: list[int]) -> str: return self.tokenizer.decode(tokens)
 