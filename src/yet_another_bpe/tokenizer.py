"""Byte-level BPE (BBPE) tokenizer implementation.

This module provides a tokenizer that uses trained BBPE models for encoding
and decoding text. Optimized for performance with heap-based merging and caching.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final, NamedTuple, Protocol, cast

import regex


class _CacheInfo(NamedTuple):
    """Type for lru_cache.cache_info() return value."""
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class _CachedWordEncoder(Protocol):
    """Protocol for lru_cache wrapped word encoder function."""
    
    def __call__(self, word: str) -> tuple[int, ...]: ...
    def cache_clear(self) -> None: ...
    def cache_info(self) -> _CacheInfo: ...


class BBPETokenizer:
    """Byte-level BPE tokenizer with optimized encoding.
    
    Performance optimizations:
    - Heap-based merge selection: O(n log n) instead of O(n²)
    - LRU cache for repeated word encodings
    - Linked-list style traversal to avoid list slicing
    """

    # GPT-2 style pre-tokenization pattern
    _GPT2_PAT: Final[str] = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Cache size for word encodings (tune based on memory constraints)
    _CACHE_SIZE: Final[int] = 8192

    def __init__(
        self,
        vocab: dict[bytes, int] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Initialize tokenizer with optional pre-loaded model.

        Args:
            vocab: Mapping from token bytes to token IDs.
            merges: List of merge operations in order.
            special_tokens: List of special tokens.
        """
        self._vocab: dict[bytes, int] = vocab or {}
        self._vocab_inv: dict[int, bytes] = {v: k for k, v in self._vocab.items()}
        self._merges: list[tuple[bytes, bytes]] = merges or []
        self._special_tokens: list[str] = special_tokens or []
        self._special_tokens_set: frozenset[str] = frozenset(self._special_tokens)
        
        # Type annotation for pattern (assigned in _compile_pattern)
        self._pattern: regex.Pattern[str]

        # Build merge rank lookup for fast merging
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self._merges)
        }

        # Compile pre-tokenization pattern
        self._compile_pattern()
        
        # Create cached version of _encode_word_impl
        # Cast to Protocol type for proper type checking of cache methods
        self._encode_word_cached: _CachedWordEncoder = cast(
            _CachedWordEncoder,
            cast(object, lru_cache(maxsize=self._CACHE_SIZE)(self._encode_word_impl))
        )

    def _compile_pattern(self) -> None:
        """Compile the pre-tokenization regex pattern."""
        pattern = self._GPT2_PAT
        if self._special_tokens:
            escaped = [regex.escape(t) for t in self._special_tokens]
            pattern = f"{'|'.join(escaped)}|{pattern}"
        self._pattern = regex.compile(pattern)

    @classmethod
    def from_file(cls, model_dir: str | Path) -> BBPETokenizer:
        """Load a tokenizer from a saved model directory.

        Args:
            model_dir: Path to directory containing vocab.json and merges.txt.

        Returns:
            A BBPETokenizer instance.
        """
        model_path = Path(model_dir)

        # Load vocabulary
        vocab_file = model_path / "vocab.json"
        with open(vocab_file, encoding="utf-8") as f:
            vocab_str = cast(dict[str, int], json.load(f))

        # Convert string keys back to bytes using latin-1
        vocab: dict[bytes, int] = {
            k.encode("latin-1"): v for k, v in vocab_str.items()
        }

        # Load merges
        merges_file = model_path / "merges.txt"
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                # Split on first space only (tokens may contain spaces)
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    token1 = parts[0].encode("latin-1")
                    token2 = parts[1].encode("latin-1")
                    merges.append((token1, token2))

        # Load special tokens if available
        special_tokens_file = model_path / "special_tokens.json"
        special_tokens: list[str] = []
        if special_tokens_file.exists():
            with open(special_tokens_file, encoding="utf-8") as f:
                special_tokens = cast(list[str], json.load(f))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: Input text to encode.

        Returns:
            List of token IDs.
        """
        if not text:
            return []

        # Pre-tokenize
        pretokens: list[str] = self._pattern.findall(text)

        token_ids: list[int] = []
        special_tokens_set = self._special_tokens_set
        vocab = self._vocab
        
        for pretoken in pretokens:
            # Check if it's a special token (use set for O(1) lookup)
            if pretoken in special_tokens_set:
                token_bytes = pretoken.encode("utf-8")
                if token_bytes in vocab:
                    token_ids.append(vocab[token_bytes])
                continue

            # Convert to bytes and apply BPE (cached)
            ids = self._encode_word_cached(pretoken)
            token_ids.extend(ids)

        return token_ids

    def _encode_word_impl(self, word: str) -> tuple[int, ...]:
        """Encode a single pre-tokenized word using optimized BPE.
        
        Uses a heap-based algorithm for O(n log n) complexity instead of O(n²).
        Returns tuple for hashability (caching).

        Args:
            word: A single pre-tokenized word.

        Returns:
            Tuple of token IDs for this word.
        """
        word_bytes = word.encode("utf-8")
        n = len(word_bytes)
        
        if n == 0:
            return ()
        
        if n == 1:
            # Single byte - direct lookup
            token = bytes([word_bytes[0]])
            return (self._vocab.get(token, self._vocab.get(b"[UNK]", 0)),)
        
        # Initialize tokens and linked-list structure
        # tokens[i] = current token at position i (None if merged into previous)
        tokens: list[bytes | None] = [bytes([b]) for b in word_bytes]
        
        # next_pos[i] = next active position after i (-1 if end)
        # prev_pos[i] = previous active position before i (-1 if start)
        next_pos: list[int] = [i + 1 for i in range(n)]
        next_pos[n - 1] = -1
        prev_pos: list[int] = [i - 1 for i in range(n)]
        
        merge_ranks = self._merge_ranks
        
        # Build initial heap: (rank, position, left_token, right_token)
        # We include tokens in heap entries to detect stale entries
        heap: list[tuple[int, int, bytes, bytes]] = []
        
        pos = 0
        while pos != -1:
            next_p = next_pos[pos]
            if next_p != -1:
                left = tokens[pos]
                right = tokens[next_p]
                if left is not None and right is not None:
                    pair = (left, right)
                    if pair in merge_ranks:
                        heapq.heappush(heap, (merge_ranks[pair], pos, left, right))
            pos = next_p
        
        # Process merges
        while heap:
            _rank, pos, left_expected, right_expected = heapq.heappop(heap)
            
            # Check if this entry is stale (tokens have changed)
            if tokens[pos] is None:
                continue
            next_p = next_pos[pos]
            if next_p == -1:
                continue
            if tokens[next_p] is None:
                continue
            
            left = tokens[pos]
            right = tokens[next_p]
            
            # Verify tokens haven't changed since heap entry was created
            if left != left_expected or right != right_expected:
                continue
            
            # Perform merge (left and right are verified non-None above)
            assert left is not None and right is not None
            merged = left + right
            tokens[pos] = merged
            tokens[next_p] = None
            
            # Update linked list: remove next_p from chain
            next_next = next_pos[next_p]
            next_pos[pos] = next_next
            if next_next != -1:
                prev_pos[next_next] = pos
            
            # Add new pairs to heap
            # New pair with previous token
            prev_p = prev_pos[pos]
            if prev_p != -1:
                prev_token = tokens[prev_p]
                if prev_token is not None:
                    pair = (prev_token, merged)
                    if pair in merge_ranks:
                        heapq.heappush(heap, (merge_ranks[pair], prev_p, prev_token, merged))
            
            # New pair with next token
            if next_next != -1:
                next_token = tokens[next_next]
                if next_token is not None:
                    pair = (merged, next_token)
                    if pair in merge_ranks:
                        heapq.heappush(heap, (merge_ranks[pair], pos, merged, next_token))
        
        # Collect final tokens
        vocab = self._vocab
        unk_id = vocab.get(b"[UNK]", 0)
        
        result: list[int] = []
        pos = 0
        while pos != -1:
            token = tokens[pos]
            if token is not None:
                result.append(vocab.get(token, unk_id))
            pos = next_pos[pos]
        
        return tuple(result)

    def _encode_word(self, word: str) -> list[int]:
        """Encode a single pre-tokenized word using BPE.

        This is a wrapper that returns a list (for API compatibility).

        Args:
            word: A single pre-tokenized word.

        Returns:
            List of token IDs for this word.
        """
        return list(self._encode_word_cached(word))

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: Sequence of token IDs.

        Returns:
            Decoded text string.
        """
        if not ids:
            return ""

        # Convert IDs to bytes
        vocab_inv = self._vocab_inv
        byte_chunks: list[bytes] = [
            vocab_inv[token_id]
            for token_id in ids
            if token_id in vocab_inv
        ]

        # Concatenate and decode
        all_bytes = b"".join(byte_chunks)
        try:
            return all_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback: decode with error replacement
            return all_bytes.decode("utf-8", errors="replace")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode multiple texts into token IDs.

        Args:
            texts: Sequence of input texts.

        Returns:
            List of token ID lists.
        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, ids_batch: Sequence[Sequence[int]]) -> list[str]:
        """Decode multiple token ID sequences back to text.

        Args:
            ids_batch: Sequence of token ID sequences.

        Returns:
            List of decoded text strings.
        """
        return [self.decode(ids) for ids in ids_batch]

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self._vocab)

    @property
    def special_tokens(self) -> list[str]:
        """Return the list of special tokens."""
        return self._special_tokens.copy()

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary as a string-to-ID mapping.

        Returns:
            Dictionary mapping token strings to IDs.
        """
        return {k.decode("latin-1"): v for k, v in self._vocab.items()}
    
    def clear_cache(self) -> None:
        """Clear the word encoding cache."""
        self._encode_word_cached.cache_clear()
    
    def cache_info(self) -> str:
        """Return cache statistics."""
        info = self._encode_word_cached.cache_info()
        return f"hits={info.hits}, misses={info.misses}, size={info.currsize}/{info.maxsize}"
