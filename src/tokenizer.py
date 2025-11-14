#python src/tokenizer.py --input data/wiki_clean.txt --output_dir outputs/tokenizer --vocab_size 30000 --min_freq 2
#Üstteki kodla çalıştır
import json
from collections import Counter
from typing import List, Iterable


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


class WordTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = SPECIAL_TOKENS.copy()
        self.token_to_id = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def fit_on_texts(
        self,
        texts: Iterable[str],
        vocab_size: int = 30000,
        min_freq: int = 2,
    ):
        counter = Counter()
        for line in texts:
            tokens = self._basic_tokenize(line)
            counter.update(tokens)

        vocab = SPECIAL_TOKENS.copy()
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token in vocab:
                continue
            vocab[token] = len(vocab)
            if len(vocab) >= vocab_size:
                break

        self.token_to_id = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}

    def _basic_tokenize(self, text: str) -> List[str]:
        # super basit: boşluk bazlı + biraz temizlik
        text = text.strip()
        return text.split()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._basic_tokenize(text)
        ids = []
        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<bos>"])
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, SPECIAL_TOKENS["<unk>"]))
        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(int(i), "<unk>")
            if tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)


def build_tokenizer_from_file(
    input_path: str,
    output_path: str,
    vocab_size: int = 30000,
    min_freq: int = 2,
):
    def text_iter():
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.strip()

    tokenizer = WordTokenizer()
    tokenizer.fit_on_texts(text_iter(), vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path} (vocab_size={tokenizer.vocab_size})")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Plain text corpus path")
    parser.add_argument(
        "--output_dir",
        default="outputs/tokenizer",
        help="Directory to save tokenizer.json",
    )
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "tokenizer.json")
    build_tokenizer_from_file(
        args.input,
        out_path,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )