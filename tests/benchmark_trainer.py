"""Benchmark script for BBPETrainer performance.

Usage:
    uv run python tests/benchmark_trainer.py
"""

import time
from pathlib import Path

from yet_another_bpe.trainer import BBPETrainer, BBPETrainerConfig


def benchmark_train(corpus_path: Path, vocab_size: int, iterations: int = 3) -> dict[str, float]:
    """Benchmark training performance.
    
    Returns:
        Dictionary with timing statistics.
    """
    times: list[float] = []
    
    model = None
    for _ in range(iterations):
        config = BBPETrainerConfig(
            vocab_size=vocab_size,
            min_frequency=1,
            max_workers=1,  # Single-threaded for consistent benchmarking
            special_tokens=["[EOS]"],
        )
        trainer = BBPETrainer(config)
        
        start = time.perf_counter()
        model = trainer.train([corpus_path])
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    assert model is not None
    import statistics
    return {
        "mean_time_s": statistics.mean(times),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "vocab_size": len(model.vocab),
        "merges_count": len(model.merges),
    }


def main() -> None:
    """Run benchmark suite."""
    fixtures_dir = Path(__file__).parent / "fixtures_gpt2"
    
    print("=" * 60)
    print("BBPETrainer Benchmark")
    print("=" * 60)
    print()
    
    # Benchmark 1: Small corpus (corpus.en - 134KB)
    small_corpus = fixtures_dir / "corpus.en"
    if small_corpus.exists():
        print(f"Benchmark 1: Small corpus ({small_corpus.name})")
        print("-" * 40)
        stats = benchmark_train(small_corpus, vocab_size=500, iterations=3)
        print(f"  Vocab size target: 500")
        print(f"  Final vocab size: {stats['vocab_size']}")
        print(f"  Merges learned: {stats['merges_count']}")
        print(f"  Mean time: {stats['mean_time_s']:.3f}s")
        print(f"  Min time:  {stats['min_time_s']:.3f}s")
        print()
    
    # Benchmark 2: Medium corpus with larger vocab
    if small_corpus.exists():
        print("Benchmark 2: Small corpus, larger vocab")
        print("-" * 40)
        stats = benchmark_train(small_corpus, vocab_size=1000, iterations=3)
        print(f"  Vocab size target: 1000")
        print(f"  Final vocab size: {stats['vocab_size']}")
        print(f"  Merges learned: {stats['merges_count']}")
        print(f"  Mean time: {stats['mean_time_s']:.3f}s")
        print(f"  Min time:  {stats['min_time_s']:.3f}s")
        print()
    
    # Benchmark 3: Large corpus (5MB TinyStories sample)
    large_corpus = fixtures_dir / "tinystories_sample_5M.txt"
    if large_corpus.exists():
        print(f"Benchmark 3: Large corpus ({large_corpus.name})")
        print("-" * 40)
        stats = benchmark_train(large_corpus, vocab_size=1000, iterations=1)
        print(f"  Vocab size target: 1000")
        print(f"  Final vocab size: {stats['vocab_size']}")
        print(f"  Merges learned: {stats['merges_count']}")
        print(f"  Time: {stats['mean_time_s']:.3f}s")
        print()
    else:
        print(f"Skipping large corpus benchmark: {large_corpus} not found")
        print()


if __name__ == "__main__":
    main()
