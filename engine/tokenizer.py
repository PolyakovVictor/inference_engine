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

    def encode(self, text: str, add_bos: bool = True) -> list[int]: 
        ids = self.tokenizer.encode(text).ids
        if not add_bos:
            bos_id = self.tokenizer.token_to_id("<s>")
            if bos_id is not None and ids and ids[0] == bos_id:
                ids = ids[1:]
        return ids

    def decode(self, tokens: list[int]) -> str: return self.tokenizer.decode(tokens)
 