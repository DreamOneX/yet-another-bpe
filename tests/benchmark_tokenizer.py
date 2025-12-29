"""Benchmark script for BBPETokenizer performance.

Usage:
    uv run python tests/benchmark_tokenizer.py
"""

import statistics
import time
from pathlib import Path

from yet_another_bpe.tokenizer import BBPETokenizer


def benchmark_encode(tokenizer: BBPETokenizer, texts: list[str], iterations: int = 3) -> dict[str, float]:
    """Benchmark encoding performance.
    
    Returns:
        Dictionary with timing statistics.
    """
    times: list[float] = []
    total_tokens = 0
    
    for _ in range(iterations):
        start = time.perf_counter()
        for text in texts:
            ids = tokenizer.encode(text)
            total_tokens += len(ids)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_tokens = total_tokens // iterations
    return {
        "mean_time_ms": statistics.mean(times) * 1000,
        "std_time_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_time_ms": min(times) * 1000,
        "total_tokens": avg_tokens,
        "tokens_per_second": avg_tokens / statistics.mean(times),
    }


def benchmark_single_word(tokenizer: BBPETokenizer, word: str, iterations: int = 1000) -> dict[str, float]:
    """Benchmark single word encoding."""
    times: list[float] = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = tokenizer.encode(word)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "mean_time_us": statistics.mean(times) * 1_000_000,
        "std_time_us": statistics.stdev(times) * 1_000_000 if len(times) > 1 else 0,
        "min_time_us": min(times) * 1_000_000,
        "max_time_us": max(times) * 1_000_000,
    }


def main() -> None:
    """Run benchmark suite."""
    # Load model
    model_dir = Path(__file__).parent.parent / "models" / "tinystories_bpe"
    if not model_dir.exists():
        print(f"Model not found at {model_dir}")
        print("Please train a model first with: uv run train-tiny-stories")
        return
    
    print(f"Loading model from {model_dir}...")
    tokenizer = BBPETokenizer.from_file(model_dir)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    
    # Test data
    test_data_dir = Path(__file__).parent / "data"
    
    # Benchmark 1: Single word encoding
    print("=" * 60)
    print("Benchmark 1: Single Word Encoding (1000 iterations)")
    print("=" * 60)
    
    test_words = ["hello", "world", "tokenization", "supercalifragilisticexpialidocious"]
    for word in test_words:
        stats = benchmark_single_word(tokenizer, word)
        print(f"  '{word}' ({len(word)} chars):")
        print(f"    Mean: {stats['mean_time_us']:.2f} μs")
        print(f"    Min:  {stats['min_time_us']:.2f} μs")
        print(f"    Max:  {stats['max_time_us']:.2f} μs")
    print()
    
    # Benchmark 2: Sentence encoding
    print("=" * 60)
    print("Benchmark 2: Sentence Encoding")
    print("=" * 60)
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! This is a test of the BPE tokenizer.",
        "Once upon a time, in a land far far away, there lived a princess.",
    ]
    
    stats = benchmark_encode(tokenizer, sentences * 100, iterations=5)
    print(f"  {len(sentences) * 100} sentences:")
    print(f"    Mean time: {stats['mean_time_ms']:.2f} ms")
    print(f"    Tokens/second: {stats['tokens_per_second']:.0f}")
    print()
    
    # Benchmark 3: Large text file
    print("=" * 60)
    print("Benchmark 3: Large Text File")
    print("=" * 60)
    
    large_file = test_data_dir / "TinyStoriesV2-GPT4-valid.txt"
    if large_file.exists():
        # Read first 100KB
        with open(large_file, encoding="utf-8") as f:
            large_text = f.read(100_000)
        
        # Split into lines for batch processing
        lines = [line for line in large_text.split("\n") if line.strip()][:500]
        
        print(f"  Processing {len(lines)} lines ({len(large_text)} chars)...")
        stats = benchmark_encode(tokenizer, lines, iterations=3)
        print(f"    Mean time: {stats['mean_time_ms']:.2f} ms")
        print(f"    Total tokens: {stats['total_tokens']}")
        print(f"    Tokens/second: {stats['tokens_per_second']:.0f}")
    else:
        print(f"  Skipped: {large_file} not found")
    print()
    
    # Benchmark 4: Repeated word (cache effectiveness)
    print("=" * 60)
    print("Benchmark 4: Repeated Words (Cache Effectiveness)")
    print("=" * 60)
    
    repeated_text = ["hello world"] * 1000
    stats = benchmark_encode(tokenizer, repeated_text, iterations=5)
    print(f"  1000x 'hello world':")
    print(f"    Mean time: {stats['mean_time_ms']:.2f} ms")
    print(f"    Tokens/second: {stats['tokens_per_second']:.0f}")
    print()


if __name__ == "__main__":
    main()
