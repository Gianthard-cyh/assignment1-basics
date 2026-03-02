import os

"""
The Byte Pair Encoding (BPE) algorithm.
"""


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
):
    with _load_file(input_path) as file:
        pretokens = _pretokenize(file)

    vocab = _init_vocab(pretokens)
    while len(vocab) <= vocab_size:
        _merge()

    return vocab


def _load_file(path: str):
    if not os.path.exists(str):
        raise FileNotFoundError(f"File {path} does not exist.")

    return open(path, "r")
