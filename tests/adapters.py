"""
Adapter functions for BPE tokenizer tests.

This module provides adapter functions that bridge test code with our BPE implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from yet_another_bpe.tokenizer import BBPETokenizer
from yet_another_bpe.trainer import BBPETrainer, BBPETrainerConfig


class TokenizerAdapter:
    """Adapter to wrap BBPETokenizer with the expected test interface."""
    
    def __init__(self, tokenizer: BBPETokenizer) -> None:
        self._tokenizer: BBPETokenizer = tokenizer
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(ids)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Memory-efficient encoding for large files (line by line)."""
        for line in iterable:
            for token_id in self._tokenizer.encode(line):
                yield token_id


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> TokenizerAdapter:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID)
            to bytes (token bytes). NOTE: This is inverted from our internal format.
        merges (list[tuple[bytes, bytes]]): BPE merges.
        special_tokens (list[str] | None): A list of string special tokens.

    Returns:
        A BPE tokenizer adapter with encode/decode/encode_iterable methods.
    """
    # Convert vocab from int->bytes to bytes->int (our internal format)
    vocab_internal: dict[bytes, int] = {v: k for k, v in vocab.items()}
    
    tokenizer = BBPETokenizer(
        vocab=vocab_internal,
        merges=merges,
        special_tokens=special_tokens or [],
    )
    
    return TokenizerAdapter(tokenizer)


def run_train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer and output its vocabulary and merges.

    Args:
        input_path (str | Path): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary.
        special_tokens (list[str]): A list of string special tokens.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: Mapping from int (token ID) to bytes (token bytes)
            merges: BPE merges ordered by creation
    """
    config = BBPETrainerConfig(
        vocab_size=vocab_size,
        min_frequency=1,
        max_workers=1,
        chunk_size_bytes=1024 * 1024 * 1024,  # 1GB chunks
        seed=42,
        special_tokens=special_tokens,
    )
    trainer = BBPETrainer(config)
    
    input_file = Path(input_path) if not isinstance(input_path, Path) else input_path
    model = trainer.train([input_file])
    
    # model.vocab is bytes -> int, we need int -> bytes
    vocab_inv: dict[int, bytes] = {v: k for k, v in model.vocab.items()}
    
    return vocab_inv, model.merges
