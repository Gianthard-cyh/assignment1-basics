import pathlib
import json
from cs336_basics.bpe import train_bpe
import time
from datetime import timedelta

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent
input_path = DATA_PATH / "data" / "owt_train.txt"
start_time = time.perf_counter()

print("Training BPE with vocab_size=32000...")
vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=32000,
    special_tokens=["<|endoftext|>"],
)

end_time = time.perf_counter()
duration = end_time - start_time

print(f"Training complete. Elapsed time: {duration:.2f} seconds")
print(f"Time formatted: {str(timedelta(seconds=int(duration)))}")

save_path = DATA_PATH / "data"
vocab_path = save_path / "owt_vocab.json"
merges_path = save_path / "owt_merges.txt"

with open(vocab_path, "w") as f:
    json.dump(vocab, f)

with open(merges_path, "w") as f:
    for merge in merges:
        f.write(f"{merge[0]} {merge[1]}\n")

print(f"BPE saved to {save_path}")
