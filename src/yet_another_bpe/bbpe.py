"""Byte-level BPE (BBPE) trainer implementation.

This module provides a scalable implementation of Byte-level Byte Pair Encoding (BBPE)
for training tokenizers on large text corpora.
"""

import json
import mmap
import regex as re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BBPETrainerConfig:
    """Configuration of a BBPE trainer.

    Attributes:
        vocab_size: Target vocabulary size, including special tokens.
        min_frequency: Minimum pair frequency for a merge to be considered.
        max_workers: Maximum number of worker threads for parallel I/O.
        chunk_size_bytes: Logical chunk size when splitting large corpora.
        seed: Random seed (unused, kept for compatibility).
        special_tokens: List of special tokens that must appear in the
            vocabulary and never be merged away.
    """

    vocab_size: int = 32000
    min_frequency: int = 2
    max_workers: int = 8
    chunk_size_bytes: int = 8 * 1024 * 1024
    seed: int = 42
    special_tokens: Sequence[str] = field(
        default_factory=lambda: ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )


class BBPEModel:
    """Container for a trained BBPE model."""

    def __init__(
        self,
        vocab: Mapping[bytes, int],
        merges: Sequence[tuple[bytes, bytes]],
        special_tokens: Sequence[str],
    ) -> None:
        self.vocab: dict[bytes, int] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens)


class BBPETrainer:
    """Byte-level BPE (BBPE) trainer."""

    def __init__(self, config: BBPETrainerConfig | None = None) -> None:
        self.config: BBPETrainerConfig = config or BBPETrainerConfig()
        self._vocab: dict[bytes, int] = {}
        self._merges: list[tuple[bytes, bytes]] = []

    def train(self, files: Sequence[str | Path]) -> BBPEModel:
        """Train a BBPE model from one or more text files.

        Args:
            files: Paths to input text files (UTF-8 encoded).

        Returns:
            A BBPEModel instance containing the learned vocabulary and merges.
        """
        if not files:
            raise ValueError("At least one file must be provided")

        path_files = [Path(f) if isinstance(f, str) else f for f in files]

        # Stage 1: Preprocess corpus into sequences
        sequences = list(self._preprocess_corpus(path_files))

        # Handle empty corpus
        if not sequences:
            vocab = self._init_base_vocab()
            self._vocab = vocab
            self._merges = []
            return BBPEModel(vocab=vocab, merges=[], special_tokens=list(self.config.special_tokens))

        # Stage 2: Run merge loop
        vocab, merges = self._merge_loop(sequences)
        self._vocab = vocab
        self._merges = merges

        return BBPEModel(vocab=vocab, merges=merges, special_tokens=list(self.config.special_tokens))

    def save(self, output_dir: str | Path) -> None:
        """Persist the trained model to disk."""
        if not self._vocab:
            raise ValueError("Model has not been trained yet. Call train() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary as JSON
        vocab_file = output_path / "vocab.json"
        vocab_str = {token_bytes.decode('latin-1'): idx for token_bytes, idx in self._vocab.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)

        # Save merges as text file
        merges_file = output_path / "merges.txt"
        with open(merges_file, 'w', encoding='utf-8') as f:
            for token1, token2 in self._merges:
                _ = f.write(f"{token1.decode('latin-1')} {token2.decode('latin-1')}\n")

    def _init_base_vocab(self) -> dict[bytes, int]:
        """Initialize vocabulary with 256 bytes + special tokens."""
        vocab: dict[bytes, int] = {}
        next_id = 0

        for byte_val in range(256):
            vocab[bytes([byte_val])] = next_id
            next_id += 1

        for special_token in self.config.special_tokens:
            token_bytes = special_token.encode('utf-8')
            if token_bytes not in vocab:
                vocab[token_bytes] = next_id
                next_id += 1

        return vocab

    def _preprocess_corpus(self, files: Sequence[Path]) -> list[list[int]]:
        """Preprocess corpus files into byte sequences."""

        def find_utf8_boundary(data: bytes, pos: int) -> int:
            if pos >= len(data):
                return len(data)
            while pos > 0 and (data[pos] & 0b11000000) == 0b10000000:
                pos -= 1
            return pos

        def process_chunk(file_path: Path, start: int, end: int) -> list[list[int]]:
            with open(file_path, 'rb') as f:
                if end - start < 1024:
                    _ = f.seek(start)
                    chunk_data = f.read(end - start)
                else:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        chunk_data = mm[start:end]

            try:
                text = chunk_data.decode('utf-8')
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"File {file_path} contains invalid UTF-8 at position {start + e.start}."
                ) from e

            # GPT-2 style pre-tokenization
            GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pattern = GPT2_PAT
            if self.config.special_tokens:
                escaped = [re.escape(t) for t in self.config.special_tokens]
                pattern = f"{'|'.join(escaped)}|{pattern}"

            pretokens: list[str] = re.findall(pattern, text)
            return [list(t.encode('utf-8')) for t in pretokens if t]

        def get_chunks(file_path: Path) -> list[tuple[int, int]]:
            file_size = file_path.stat().st_size
            if file_size == 0:
                return []
            if file_size <= self.config.chunk_size_bytes:
                return [(0, file_size)]

            chunks = []
            start = 0
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    while start < file_size:
                        tentative_end = min(start + self.config.chunk_size_bytes, file_size)
                        if tentative_end < file_size:
                            boundary_start = max(0, tentative_end - 4)
                            boundary_data = mm[boundary_start:tentative_end + 1]
                            local_pos = tentative_end - boundary_start
                            adjusted = find_utf8_boundary(boundary_data, local_pos)
                            actual_end = boundary_start + adjusted
                        else:
                            actual_end = file_size
                        if actual_end > start:
                            chunks.append((start, actual_end))
                            start = actual_end
                        else:
                            start += 1
            return chunks

        all_sequences: list[list[int]] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures: list[Future[list[list[int]]]] = []
            for file_path in files:
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                for start, end in get_chunks(file_path):
                    futures.append(executor.submit(process_chunk, file_path, start, end))

            for future in futures:
                for seq in future.result():
                    if seq:
                        all_sequences.append(seq)

        return all_sequences

    def _merge_loop(self, sequences: list[list[int]]) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
        """Run the BPE merge loop with fully incremental updates."""
        vocab = self._init_base_vocab()
        next_id = len(vocab)

        # Build word frequency dict
        word_freq: dict[tuple[bytes, ...], int] = defaultdict(int)
        for seq in sequences:
            word_tuple = tuple(bytes([b]) for b in seq)
            word_freq[word_tuple] += 1

        # Compute initial pair counts and pair-to-words index
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

        for word_tuple, freq in word_freq.items():
            for j in range(len(word_tuple) - 1):
                pair = (word_tuple[j], word_tuple[j + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word_tuple)

        # Calculate number of merges
        num_merges = max(0, self.config.vocab_size - len(vocab))
        merges: list[tuple[bytes, bytes]] = []

        for _ in range(num_merges):
            if not pair_counts:
                break

            # Find best pair (highest freq, lexicographically largest for ties)
            best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            if pair_counts[best_pair] < self.config.min_frequency:
                break

            p0, p1 = best_pair
            merged = p0 + p1

            # Only process words that contain the best pair
            affected_words = list(pair_to_words.get(best_pair, set()))

            for word_tuple in affected_words:
                freq = word_freq.get(word_tuple, 0)
                if freq == 0:
                    continue

                # Remove word from word_freq
                del word_freq[word_tuple]

                # Decrement old pair counts and remove from index
                for j in range(len(word_tuple) - 1):
                    old_pair = (word_tuple[j], word_tuple[j + 1])
                    pair_counts[old_pair] -= freq
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                        if old_pair in pair_to_words:
                            del pair_to_words[old_pair]
                    else:
                        pair_to_words[old_pair].discard(word_tuple)

                # Apply merge
                new_word: list[bytes] = []
                j = 0
                while j < len(word_tuple):
                    if j < len(word_tuple) - 1 and word_tuple[j] == p0 and word_tuple[j + 1] == p1:
                        new_word.append(merged)
                        j += 2
                    else:
                        new_word.append(word_tuple[j])
                        j += 1
                new_tuple = tuple(new_word)

                # Add new word to word_freq
                word_freq[new_tuple] = word_freq.get(new_tuple, 0) + freq

                # Increment new pair counts and add to index
                for j in range(len(new_tuple) - 1):
                    new_pair = (new_tuple[j], new_tuple[j + 1])
                    pair_counts[new_pair] += freq
                    pair_to_words[new_pair].add(new_tuple)

            merges.append(best_pair)

            if merged not in vocab:
                vocab[merged] = next_id
                next_id += 1

        return vocab, merges
