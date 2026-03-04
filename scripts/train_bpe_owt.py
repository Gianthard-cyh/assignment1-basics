import pathlib
import pickle
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

save_path = DATA_PATH / "data" / "bpe_model_owt.pt"
checkpoint = {
    "vocab": vocab,
    "merges": merges,
}

with open(save_path, "wb") as f:
    pickle.dump(checkpoint, f)

print(f"BPE saved to {save_path}")
