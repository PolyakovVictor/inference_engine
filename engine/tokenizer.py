from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, tokenizer_path: Path | str) -> None: 
        self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        self.eos_tokens = ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id>", "<|end|>"]
        self.eos_ids: set[int] = set()
        for t in self.eos_tokens:
            tid = self.tokenizer.token_to_id(t)
            if tid is not None: self.eos_ids.add(tid)
        self.bos_id = self.tokenizer.token_to_id("<s>")
        if self.bos_id is None: self.bos_id = self.tokenizer.token_to_id("<|begin_of_text|>")
        try:
            for tok in getattr(self.tokenizer, "get_vocab", lambda: {})().keys():
                if tok in ("<|im_end|>", "<|endoftext|>", "</s>"):
                    tid = self.tokenizer.token_to_id(tok)
                    if tid is not None: self.eos_ids.add(tid)
        except Exception: pass

    def encode(self, text: str, add_bos: bool = True) -> list[int]: 
        ids = self.tokenizer.encode(text).ids
        if not add_bos and self.bos_id is not None and ids and ids[0] == self.bos_id:
            ids = ids[1:]
        return ids

    def decode(self, tokens: list[int]) -> str: return self.tokenizer.decode(tokens)
 