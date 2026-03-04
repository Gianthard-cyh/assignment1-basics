from dataclasses import dataclass
from functools import partial
import os
import typing
import regex as re
from collections import Counter
from rich.progress import track
from multiprocessing import Process, Lock, Pool
from multiprocessing.synchronize import Lock as LockType

from cs336_basics.pretokenization_example import find_chunk_boundaries

"""
The Byte Pair Encoding (BPE) algorithm.
"""

PRETOKENIZE_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZE_BATCHSIZE = 1000
pretokenize_lock = Lock()


@dataclass
class Pretoken:
    tokens: tuple[bytes, ...]
    count: int


@dataclass
class AdjacentPair:
    left: bytes
    right: bytes
    count: int
    occurrences: list[Pretoken]

    def __lt__(self, other):
        # Note: Maximum heap
        if self.count != other.count:
            return self.count < other.count
        else:
            if self.left != other.left:
                return self.left < other.left
            else:
                return self.right < other.right

    def get_tuple(self):
        return (self.left, self.right)


class PairHeap:
    def __init__(self) -> None:
        self.hp: list[AdjacentPair] = []
        self.index_map: dict[tuple[bytes, bytes], int] = {}

    def push(self, x: AdjacentPair) -> None:
        idx = len(self.hp)
        self.hp.append(x)
        self.index_map[x.get_tuple()] = idx
        self._sift_up(idx)

    def pop(self) -> AdjacentPair:
        if not self.hp:
            raise IndexError("pop from empty heap")
        top = self.hp[0]
        # print(f"[pop] {top.get_tuple()}")
        last = self.hp.pop()
        del self.index_map[top.get_tuple()]
        if self.hp:
            self.hp[0] = last
            self.index_map[last.get_tuple()] = 0
            self._sift_down(0)
        return top

    def top(self) -> AdjacentPair:
        return self.hp[0]

    def remove(self, x: AdjacentPair) -> None:
        # print(f"[remove] {x.get_tuple()}")
        if x.get_tuple() not in self.index_map:
            return
        idx = self.index_map[x.get_tuple()]
        last_idx = len(self.hp) - 1
        self._swap(idx, last_idx)

        removed = self.hp.pop()
        del self.index_map[removed.get_tuple()]

        if idx < len(self.hp):
            idx = self._sift_up(idx)
            self._sift_down(idx)

    def update_count(self, x: AdjacentPair, new_count: int) -> None:
        if x.get_tuple() not in self.index_map:
            return
        idx = self.index_map[x.get_tuple()]
        target = self.hp[idx]
        target.count = new_count

        idx = self._sift_up(idx)
        self._sift_down(idx)

    def _sift_up(self, idx: int) -> int:
        while idx > 0:
            parent = (idx - 1) // 2
            if self.hp[idx] > self.hp[parent]:
                self._swap(idx, parent)
                idx = parent
            else:
                break
        return idx

    def _sift_down(self, idx: int) -> int:
        n = len(self.hp)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            largest = idx
            if left < n and self.hp[left] > self.hp[largest]:
                largest = left
            if right < n and self.hp[right] > self.hp[largest]:
                largest = right
            if largest == idx:
                break
            self._swap(idx, largest)
            idx = largest
        return idx

    def _swap(self, i: int, j: int) -> None:
        self.hp[i], self.hp[j] = self.hp[j], self.hp[i]
        self.index_map[self.hp[i].get_tuple()] = i
        self.index_map[self.hp[j].get_tuple()] = j

    def __len__(self) -> int:
        return len(self.hp)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = _init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    with _load_file(input_path) as file:
        adj, heap = _pretokenize(file, special_tokens)

    while len(vocab) < vocab_size:
        _merge(vocab, adj, heap, merges)

    print("====================== Vocab ======================")
    print(vocab)

    return (vocab, merges)


def _load_file(path: str | os.PathLike):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    return open(path, "rb")


def _process_mini_chunk(mini_chunks: list[str]):
    counter = Counter()
    for mini_chunk in mini_chunks:
        chunk_pretokens = re.findall(PRETOKENIZE_PAT, mini_chunk)
        chunk_pretoken_counts = Counter(chunk_pretokens)
        counter += chunk_pretoken_counts
    return counter


def _pretokenize(
    file: typing.BinaryIO, special_tokens: list[str]
) -> tuple[dict[tuple[bytes, bytes], AdjacentPair], PairHeap]:
    num_processes = 16
    print("begin find_chunk_boundaries")
    boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
    print(f"{len(boundaries)} boundaries found")

    mini_chunks: list[str] = []
    for start, end in track(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries) - 1):
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        special_tokens_pat = "|".join([re.escape(i) for i in special_tokens])
        mini_chunks.extend(re.split(special_tokens_pat, chunk))

    batch_list = [mini_chunks[i : i + PRETOKENIZE_BATCHSIZE] for i in range(0, len(mini_chunks), PRETOKENIZE_BATCHSIZE)]
    counter = Counter[str]()
    print(f"split {len(mini_chunks)} mini chunks")
    with Pool(processes=17) as pool:
        for result in track(
            pool.imap(_process_mini_chunk, batch_list, chunksize=5),
            description="Processing mini chunks...",
            total=len(batch_list),
        ):
            counter += result

    pretokens: list[Pretoken] = []
    for k, v in counter.items():
        b = k.encode()
        pretoken_tuple = tuple(b[i : i + 1] for i in range(len(b)))
        pretokens.append(Pretoken(pretoken_tuple, v))

    pairs: dict[tuple[bytes, bytes], AdjacentPair] = {}
    for k in pretokens:
        for i in range(0, len(k.tokens) - 1):
            adj_tuple = (k.tokens[i], k.tokens[i + 1])
            if adj_tuple in pairs:
                pair = pairs[adj_tuple]
            else:
                pair = pairs[adj_tuple] = AdjacentPair(k.tokens[i], k.tokens[i + 1], 0, [])
            pair.count += k.count
            pair.occurrences.append(k)

    heap = PairHeap()
    for k, v in pairs.items():
        heap.push(v)

    return (pairs, heap)


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}

    for i in range(0, 256):
        vocab[len(vocab)] = i.to_bytes()

    for token in special_tokens:
        vocab[len(vocab)] = token.encode()

    return vocab


# merge
# 2. 在pretokens里合并（用得到吗？）-> 用得到，因为如果合并的token后面的token合并，他找到的前序token需要是合并完这个
# 3. 修改adj数组
#    改哪些：改和合并的token相邻的，无论是在哪个pretoken里
#    怎么改：把前面adj的后面那个改成合并后的，后面adj的前面那个改成合并后的
#    可能重复吗：low和lower，合并ow的时候前面是l。可以先统计ow的出现，前面后面分别是什么，压进一个数组里，然后来统计，一次更新防止重复
#    更新频率：首先要在堆里找到原来那两个删除，然后把新的插进去
def _merge(
    vocab: dict[int, bytes],
    pairs: dict[tuple[bytes, bytes], AdjacentPair],
    heap: PairHeap,
    merge: list[tuple[bytes, bytes]],
):
    """
    从堆中取出频率最高的一对token，并进行merge。更新这对token相邻出现的所有pretoken。
    """
    pair = heap.pop()
    merged_token = pair.left + pair.right

    # print(f"[merge] {pair.left} + {pair.right} -> {merged_token}")
    merge.append((pair.left, pair.right))

    for occ_pretok in pair.occurrences:
        i = 0
        while i + 1 < len(occ_pretok.tokens):
            if occ_pretok.tokens[i] == pair.left and occ_pretok.tokens[i + 1] == pair.right:
                _merge_pretoken(occ_pretok, i, pairs, heap)
            i += 1

    vocab[len(vocab)] = merged_token


def _merge_pretoken(
    pretok: Pretoken,
    pos: int,
    pairs: dict[tuple[bytes, bytes], AdjacentPair],
    heap: PairHeap,
):
    """
    Pretoken 级别的 merge。
    @param pretok: 要合并的对所在的pretoken
    @param pos: 这一对当中的第一个元素在pretok内的索引
    """
    # print(f"_merge_pretoken({pretok.tokens})")
    old_tokens = pretok.tokens
    merged_token = old_tokens[pos] + old_tokens[pos + 1]
    new_tokens = old_tokens[:pos] + (old_tokens[pos] + old_tokens[pos + 1],) + old_tokens[pos + 2 :]

    pair = pairs[(old_tokens[pos], old_tokens[pos + 1])]
    pair.count -= pretok.count

    if pair.count == 0:
        del pairs[pair.get_tuple()]

    if pos - 1 >= 0:
        _remove_pair((old_tokens[pos - 1], old_tokens[pos]), pretok.count, pairs, heap)
        _push_pair((old_tokens[pos - 1], merged_token), pairs, heap, pretok)

    if pos + 2 < len(pretok.tokens):
        _remove_pair((old_tokens[pos + 1], old_tokens[pos + 2]), pretok.count, pairs, heap)
        _push_pair((merged_token, old_tokens[pos + 2]), pairs, heap, pretok)

    pretok.tokens = new_tokens


def _remove_pair(
    pair_tuple: tuple[bytes, bytes],
    count: int,
    adj: dict[tuple[bytes, bytes], AdjacentPair],
    heap: PairHeap,
):
    pair = adj[pair_tuple]
    newcount = pair.count - count
    if newcount == 0:
        heap.remove(pair)
        del adj[pair_tuple]
    else:
        heap.update_count(pair, newcount)


def _push_pair(
    pair_tuple: tuple[bytes, bytes], adj: dict[tuple[bytes, bytes], AdjacentPair], heap: PairHeap, pretoken: Pretoken
):
    if pair_tuple in adj:
        pair = adj[pair_tuple]
        newcount = pair.count + pretoken.count
        if pretoken not in pair.occurrences:
            pair.occurrences.append(pretoken)
        heap.update_count(pair, newcount)
    else:
        pair = adj[pair_tuple] = AdjacentPair(pair_tuple[0], pair_tuple[1], pretoken.count, [pretoken])
        adj[pair_tuple] = pair
        heap.push(pair)
