class SimpleTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "hello": 2,
            "world": 3,
            "I": 4,
            "am": 5,
            "a": 6,
            "robot": 7,
        }
        self.inverse_vocab = {
            v: k
            for k,v in self.vocab.items()
        }
    
    def encode(self, text: str):
        words = text.split()
        return [
            self.vocab.get(word, self.vocab["<unk>"])
            for word in words
        ]
    
    def decode(self, tokens: list):
        words = [self.inverse_vocab.get(token, "<unk>") for token in tokens]
        return " ".join(words)
