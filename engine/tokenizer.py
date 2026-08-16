class SimpleTokenizer:
    def __init__(self, vocab: dict) -> None:
        self.vocab = vocab
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
