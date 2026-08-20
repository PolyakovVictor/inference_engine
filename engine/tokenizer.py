from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, tokenizer_path: Path | str) -> None: self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))
    def encode(self, text: str) -> list[int]: return self.tokenizer.encode(text).ids
    def decode(self, tokens: list[int]) -> str: return self.tokenizer.decode(tokens)
    @property
    def eos_ids(self,) -> int | None: return self.tokenizer.token_to_id("<|im_end|>") or self.tokenizer.token_to_id("<|endoftext|>")
 