from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import regex as re
from concurrent.futures import ProcessPoolExecutor

from .pretokenization import PRETOKENIZE_PAT


@dataclass
class SpecialTokenChunk:
    token_id: int


@dataclass
class StringTextChunk:
    text: str


@dataclass
class PretokenizedTextChunk:
    pretokens: list[list[bytes]]


@dataclass
class TokenizedTextChunk:
    token_ids: list[int]


Chunk = SpecialTokenChunk | StringTextChunk | PretokenizedTextChunk | TokenizedTextChunk


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        根据给定的词表、合并规则和可选的特殊标记构建分词器。
        """
        self.vocab = vocab
        self.bytes_id_dict = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = None
        self.bpe_ranks = {merge: i for i, merge in enumerate(self.merges)}
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.pretokenize_regex = re.compile(PRETOKENIZE_PAT)
        self.special_tokens_regex = None
        if self.special_tokens:
            special_tokens_pat = "|".join([re.escape(t) for t in self.special_tokens])
            self.special_tokens_regex = re.compile(special_tokens_pat)

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """
        从序列化的词表文件和合并规则文件中构造并返回 Tokenizer 实例。
        """
        vocab = {}
        merges = []
        return cls(vocab, merges, special_tokens)

    def _process_chunk(self, chunk):
        ids = []
        merged_pretokens = self._merge(chunk.pretokens)
        for pretoken in merged_pretokens:
            ids.extend(self._pretoken_to_ids(pretoken))
        return ids

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为 token ID 序列。
        """
        chunks = self._pretokenize(text)

        res: list[int] = []
        for chunk in chunks:
            if isinstance(chunk, PretokenizedTextChunk):
                res.extend(self._process_chunk(chunk))
            elif isinstance(chunk, SpecialTokenChunk):
                res.append(chunk.token_id)

        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buf: str = ""

        for text in iterable:
            buf += text

            special_matches = list(re.finditer(self.special_tokens_regex, buf)) if self.special_tokens_regex else None
            pretokenize_matches = list(re.finditer(self.pretokenize_regex, buf))

            process_until = (
                min(special_matches[-1].end(), pretokenize_matches[-2].end())
                if special_matches
                else pretokenize_matches[-1].start()
            )

            token_text = buf[: process_until + 1]

            yield from self.encode(token_text)

            buf = buf[process_until + 1 :]

        if buf:
            yield from self.encode(buf)

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 序列解码回文本字符串。
        """
        res = bytes([])
        for id in ids:
            res += self.vocab[id]
        return res.decode(errors="replace")

    def _pretokenize(self, text: str):

        chunks: list[Chunk] = []
        last_match = 0
        if self.special_tokens_regex:
            for match in self.special_tokens_regex.finditer(text):
                if last_match < match.start() - 1:
                    chunks.append(StringTextChunk(text[last_match : match.start()]))
                chunks.append(SpecialTokenChunk(self.bytes_id_dict[match.group().encode()]))
                last_match = match.end()

        chunks.append(StringTextChunk(text[last_match : len(text)]))

        for i, chunk in [(i, c) for i, c in enumerate(chunks) if isinstance(c, StringTextChunk)]:
            chunk_pretokens: list[list[bytes]] = []
            for match in self.pretokenize_regex.finditer(chunk.text):
                b = match.group().encode()
                chunk_pretokens.append([b[i : i + 1] for i in range(len(b))])
            chunks[i] = PretokenizedTextChunk(chunk_pretokens)

        return chunks

    def _merge(self, pretokens: list[list[bytes]]) -> list[list[bytes]]:
        for pretoken in pretokens:
            while len(pretoken) > 1:
                min_rank = float("inf")
                best_idx = -1

                for i in range(len(pretoken) - 1):
                    pair = (pretoken[i], pretoken[i + 1])
                    rank = self.bpe_ranks.get(pair, float("inf"))
                    if rank < min_rank:
                        min_rank = rank
                        best_idx = i

                if best_idx == -1:
                    break

                pretoken[best_idx] = pretoken[best_idx] + pretoken[best_idx + 1]
                pretoken.pop(best_idx + 1)

        return pretokens

    def _pretoken_to_ids(self, pretoken: list[bytes]):
        return [self.bytes_id_dict[k] for k in pretoken]
