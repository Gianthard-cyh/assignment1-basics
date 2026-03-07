import pickle
import sys
import pathlib
import statistics
from collections import Counter

def analyze_bpe(model_path):
    print(f"Loading BPE model from {model_path}...")
    try:
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    vocab = checkpoint.get("vocab")
    merges = checkpoint.get("merges")

    if not vocab or not merges:
        print("Error: Invalid model format. Missing 'vocab' or 'merges'.")
        return

    print("\n" + "="*50)
    print("BPE Model Analysis")
    print("="*50)

    # 1. Basic Stats
    vocab_size = len(vocab)
    num_merges = len(merges)
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Number of Merges: {num_merges}")

    # 2. Merge Analysis
    print("\n--- Merges ---")
    print("First 5 merges:")
    for i, merge in enumerate(merges[:5]):
        print(f"  {i+1}. {merge}")
    print("...")
    print("Last 5 merges:")
    for i, merge in enumerate(merges[-5:]):
        print(f"  {num_merges - 4 + i}. {merge}")

    # 3. Vocabulary Analysis
    print("\n--- Vocabulary ---")
    
    # Token lengths
    token_lengths = [len(token) for token in vocab.values()]
    avg_len = statistics.mean(token_lengths)
    max_len = max(token_lengths)
    min_len = min(token_lengths)
    
    print(f"Average Token Length (bytes): {avg_len:.2f}")
    print(f"Max Token Length (bytes): {max_len}")
    print(f"Min Token Length (bytes): {min_len}")

    # Longest tokens
    sorted_vocab = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)
    print("\nTop 20 Longest Tokens:")
    for i, (idx, token) in enumerate(sorted_vocab[:20]):
        try:
            token_str = token.decode('utf-8', errors='replace')
        except:
            token_str = str(token)
        print(f"  {i+1}. ID: {idx}, Length: {len(token)}, Bytes: {token}, String: {token_str!r}")

    # Shortest tokens (sample)
    print("\nSample Shortest Tokens (length 1):")
    shortest_tokens = [v for k, v in vocab.items() if len(v) == 1]
    for i, token in enumerate(shortest_tokens[:5]):
         print(f"  {i+1}. {token}")

    # 4. Special Tokens Check (simple heuristic)
    print("\n--- Special Tokens Check ---")
    special_candidates = [v for k, v in vocab.items() if b"<|" in v and b"|>" in v]
    if special_candidates:
        print(f"Found {len(special_candidates)} potential special tokens:")
        for t in special_candidates:
            print(f"  {t}")
    else:
        print("No obvious special tokens found (containing '<|' and '|>').")

if __name__ == "__main__":
    # Default path
    default_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "bpe_model_owt.pt"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = default_path
        print(f"No model path provided. Using default: {model_path}")
    
    analyze_bpe(model_path)
