from __future__ import annotations

import heapq
import json
import mmap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import regex


@dataclass
class BBPETrainerConfig:
    """Configuration of a BBPE trainer.

    Attributes:
        vocab_size: Target vocabulary size, including special tokens.
        min_frequency: Minimum pair frequency for a merge to be considered.
        max_workers: Maximum number of worker threads/processes used for
            parallel stages (I/O, counting, merging).
        chunk_size_bytes: Logical chunk size when splitting large corpora.
            Used together with memory mapping to create virtual windows.
        seed: Random seed for any stochastic behavior (e.g., down-sampling).
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
    """Light-weight placeholder for a trained BBPE model.

    This is only an interface stub so that `BBPETrainer.train` can return
    something meaningful. A real implementation would store merges, vocab,
    and any auxiliary tables required for fast tokenization.
    """

    def __init__(
        self,
        vocab: Mapping[int, bytes],
        merges: Sequence[Tuple[bytes, bytes]],
        special_tokens: Sequence[str],
    ) -> None:
        """Initialize a BBPEModel.

        Args:
            vocab: Final token-to-id mapping (id -> bytes).
            merges: Sequence of merge operations, each represented as a pair
                of byte sequences.
            special_tokens: Special tokens that should be handled separately
                by the tokenizer.
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens)

    def get_vocab(self) -> Dict[int, bytes]:
        """Return the vocabulary as dict[int, bytes]."""
        return self.vocab

    def get_merges(self) -> List[Tuple[bytes, bytes]]:
        """Return the merge operations as list[tuple[bytes, bytes]]."""
        return self.merges


class BBPETrainer:
    """Byte-level BPE (BBPE) trainer.

    The trainer is organized into three explicit stages that correspond to
    typical bottlenecks in large-scale corpus processing:

    1. Stage I – Preprocessing (disk I/O bound):
       Use memory-mapped files to avoid copying, and split the corpus into
       logical chunks processed in parallel.

    2. Stage II – Initial statistics (CPU / hash + memory bound):
       Use a MapReduce-style pattern to parallelize pair counting across
       chunks, combined with a fast non-cryptographic hash for token pairs.

    3. Stage III – Merge loop (algorithmic complexity bound):
       Use a max-heap priority queue keyed by pair frequency, and perform
       frequency-based incremental updates similar in spirit to Hugging Face
       `tokenizers`' inner loop.
    """

    def __init__(self, config: Optional[BBPETrainerConfig] = None) -> None:
        """Initialize a BBPETrainer.

        Args:
            config: Optional configuration object. If omitted, a default
                configuration is created.
        """
        self.config = config or BBPETrainerConfig()
        self._vocab: Dict[int, bytes] = {}
        self._merges: List[Tuple[bytes, bytes]] = []

        # GPT-2 style pretokenization pattern
        self._pretokenize_pattern = r"(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

        # Build combined pattern with special tokens
        if self.config.special_tokens:
            escaped_tokens = [regex.escape(token) for token in self.config.special_tokens]
            special_pattern = '|'.join(escaped_tokens)
            self._pretokenize_pattern = f"{special_pattern}|{self._pretokenize_pattern}"

        self._compiled_pattern = regex.compile(self._pretokenize_pattern)

    # === Public API ===

    def train(self, files: Sequence[str | Path]) -> BBPEModel:
        """Train a BBPE model from one or more text files.

        Args:
            files: Paths to input text files that constitute the training
                corpus. Files are assumed to be UTF-8 encoded.

        Returns:
            A `BBPEModel` instance containing the learned vocabulary and
            merge operations.
        """
        # Convert all paths to Path objects
        file_paths = [Path(f) if isinstance(f, str) else f for f in files]

        # Stage I: Preprocess corpus into byte sequences (word tokens)
        word_tokens = self._stage1_preprocess_corpus(file_paths)

        # Initialize vocabulary with all single bytes
        for i in range(256):
            self._vocab[i] = bytes([i])

        # Add special tokens to vocabulary
        next_id = 256
        for token in self.config.special_tokens:
            token_bytes = token.encode('utf-8')
            self._vocab[next_id] = token_bytes
            next_id += 1

        # Stage II: Count initial pair frequencies
        pair_counts = self._stage2_initial_pair_counts(word_tokens)

        # Stage III: Execute merge loop
        num_merges = self.config.vocab_size - len(self._vocab)
        self._stage3_merge_loop(word_tokens, pair_counts, num_merges, next_id)

        return BBPEModel(self._vocab, self._merges, self.config.special_tokens)

    def train_from_text(self, text: str, num_merges: int) -> BBPEModel:
        """Train a BBPE model directly from text.

        Args:
            text: Training text string.
            num_merges: Number of merge operations to perform.

        Returns:
            A `BBPEModel` instance containing the learned vocabulary and
            merge operations.
        """
        # Pretokenize text
        pretokens = self._compiled_pattern.findall(text)

        # Convert pretokens to byte sequences
        word_tokens: List[List[bytes]] = []
        for token in pretokens:
            if token in self.config.special_tokens:
                # Special tokens are treated as single indivisible units
                word_tokens.append([token.encode('utf-8')])
            else:
                # Regular tokens are split into single bytes
                token_bytes = token.encode('utf-8')
                word_tokens.append([bytes([b]) for b in token_bytes])

        # Initialize vocabulary with all single bytes
        for i in range(256):
            self._vocab[i] = bytes([i])

        # Add special tokens to vocabulary
        next_id = 256
        for token in self.config.special_tokens:
            token_bytes = token.encode('utf-8')
            self._vocab[next_id] = token_bytes
            next_id += 1

        # Count initial pair frequencies
        pair_counts = self._stage2_initial_pair_counts(word_tokens)

        # Execute merge loop
        self._stage3_merge_loop(word_tokens, pair_counts, num_merges, next_id)

        return BBPEModel(self._vocab, self._merges, self.config.special_tokens)

    def save(self, output_dir: str | Path) -> None:
        """Persist the trained model to disk.

        Args:
            output_dir: Directory where vocabulary and merges should be
                written.

        Raises:
            RuntimeError: If called before `train`.
        """
        if not self._vocab:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary as JSON
        import base64
        vocab_json = {
            str(k): base64.b64encode(v).decode('ascii')
            for k, v in self._vocab.items()
        }

        with open(output_path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_json, f, indent=2)

        # Save merges as text file
        with open(output_path / "merges.txt", "w", encoding="utf-8") as f:
            for a, b in self._merges:
                # Use repr for bytes that might not be valid UTF-8
                try:
                    a_str = a.decode('utf-8', errors='backslashreplace')
                    b_str = b.decode('utf-8', errors='backslashreplace')
                    f.write(f"{repr(a_str)} {repr(b_str)}\n")
                except Exception:
                    f.write(f"{a.hex()} {b.hex()}\n")

    def get_vocab(self) -> Dict[int, bytes]:
        """Return the vocabulary as dict[int, bytes]."""
        return self._vocab

    def get_merges(self) -> List[Tuple[bytes, bytes]]:
        """Return the merge operations as list[tuple[bytes, bytes]]."""
        return self._merges

    # === Stage I: Preprocessing (I/O bound) ===

    def _stage1_preprocess_corpus(
        self, files: Sequence[Path]
    ) -> List[List[bytes]]:
        """Preprocess corpus and return byte-level token sequences.

        Args:
            files: Paths to the corpus files.

        Returns:
            A list of word tokens, where each word is a list of bytes.
        """
        word_tokens: List[List[bytes]] = []

        for file_path in files:
            if not file_path.exists():
                continue

            file_size = file_path.stat().st_size
            if file_size == 0:
                continue

            # Read file content
            with open(file_path, 'rb') as f:
                try:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        text = mmapped_file[:].decode('utf-8', errors='ignore')
                except ValueError:
                    # mmap fails on empty files on some systems
                    text = f.read().decode('utf-8', errors='ignore')

            # Pretokenize
            pretokens = self._compiled_pattern.findall(text)

            # Convert to byte sequences
            for token in pretokens:
                if token in self.config.special_tokens:
                    word_tokens.append([token.encode('utf-8')])
                else:
                    token_bytes = token.encode('utf-8')
                    word_tokens.append([bytes([b]) for b in token_bytes])

        return word_tokens

    # === Stage II: Initial counting (CPU / hash bound) ===

    def _stage2_initial_pair_counts(
        self, word_tokens: List[List[bytes]]
    ) -> Dict[Tuple[bytes, bytes], int]:
        """Compute global pair frequencies.

        Args:
            word_tokens: List of word tokens (each word is a list of bytes).

        Returns:
            A dictionary mapping byte pairs to their frequencies.
        """
        pairs: Dict[Tuple[bytes, bytes], int] = defaultdict(int)

        for word in word_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1

        return dict(pairs)

    # === Stage III: Merge loop (algorithmic complexity bound) ===

    def _stage3_merge_loop(
        self,
        word_tokens: List[List[bytes]],
        pair_counts: Dict[Tuple[bytes, bytes], int],
        num_merges: int,
        next_id: int,
    ) -> None:
        """Run the BBPE merge loop.

        Args:
            word_tokens: List of word tokens to process (modified in place).
            pair_counts: Initial pair frequencies.
            num_merges: Number of merges to perform.
            next_id: Next vocabulary ID to use.
        """
        for _ in range(num_merges):
            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            best_count = pair_counts[best_pair]

            if best_count < self.config.min_frequency:
                break

            # Merge this pair in all words
            for i, word in enumerate(word_tokens):
                word_tokens[i] = self._merge_pair(word, best_pair)

            # Record the merge
            self._merges.append(best_pair)

            # Add new token to vocabulary
            new_token = best_pair[0] + best_pair[1]
            self._vocab[next_id] = new_token
            next_id += 1

            # Recount all pairs (brute force for simplicity)
            pair_counts = self._stage2_initial_pair_counts(word_tokens)

    def _merge_pair(
        self, tokens: List[bytes], pair: Tuple[bytes, bytes]
    ) -> List[bytes]:
        """Merge the specified pair in a token list.

        Args:
            tokens: List of byte tokens.
            pair: The pair to merge.

        Returns:
            New list with the pair merged.
        """
        new_tokens: List[bytes] = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens
