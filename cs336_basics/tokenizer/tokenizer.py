import pickle
import regex as re
from collections.abc import Iterable, Iterator
from functools import lru_cache

from .pretokenization import PRETOKENIZE_PAT

class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.bytes_id_dict = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
        
        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.special_regex = None
        if special_tokens:
            sorted_specials = sorted(special_tokens, key=len, reverse=True)
            special_pat = "|".join(re.escape(t) for t in sorted_specials)
            self.special_regex = re.compile(f"({special_pat})")
            
        self.pretokenize_regex = re.compile(PRETOKENIZE_PAT)

    @classmethod
    def from_file(cls, filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls(data["vocab"], data["merges"], special_tokens)

    @lru_cache(maxsize=30000)
    def _bpe_merge(self, piece_bytes: bytes) -> list[int]:
        parts = [piece_bytes[i:i+1] for i in range(len(piece_bytes))]
        while len(parts) > 1:
            min_rank = float("inf")
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self.bpe_ranks.get((parts[i], parts[i + 1]), float("inf"))
                if rank < min_rank:
                    min_rank, best_idx = rank, i
            if best_idx == -1:
                break
            parts[best_idx] = parts[best_idx] + parts[best_idx + 1]
            parts.pop(best_idx + 1)
        return [self.bytes_id_dict[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        ids = []
        if self.special_regex:
            parts = self.special_regex.split(text)
            for i, part in enumerate(parts):
                if not part:
                    continue
                if i % 2 == 1:
                    ids.append(self.bytes_id_dict[part.encode()])
                else:
                    for match in self.pretokenize_regex.finditer(part):
                        ids.extend(self._bpe_merge(match.group().encode()))
        else:
            for match in self.pretokenize_regex.finditer(text):
                ids.extend(self._bpe_merge(match.group().encode()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buf = ""
        for text in iterable:
            buf += text
            
            special_matches = list(self.special_regex.finditer(buf)) if self.special_regex else []
            pre_matches = list(self.pretokenize_regex.finditer(buf))
            
            if not pre_matches:
                continue
            
            if special_matches:
                process_until = min(special_matches[-1].start(), pre_matches[-1].start())
            else:
                process_until = pre_matches[-1].start()
                
            if process_until > 0:
                yield from self.encode(buf[:process_until])
                buf = buf[process_until:]
        
        if buf:
            yield from self.encode(buf)

    def decode(self, ids: list[int]) -> str:
        res = bytearray()
        for i in ids:
            res.extend(self.vocab[i])
        return res.decode("utf-8", errors="replace")