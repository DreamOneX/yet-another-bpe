"""Byte-level BPE (BBPE) trainer implementation.

This module provides a scalable implementation of Byte-level Byte Pair Encoding (BBPE)
for training tokenizers on large text corpora. The trainer uses memory-mapped files,
parallel processing, and incremental updates to efficiently handle large datasets.
"""

import heapq
import json
import mmap
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


# Module-level helper function for ProcessPoolExecutor
# (needs to be picklable, so can't be a method)
def _process_batch_for_pairs_with_tracking(
    sequences: list[list[int]], 
    min_frequency: int = 1,
    seq_offset: int = 0
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
    """Process a batch of sequences and return pair counts with sequence tracking.

    This is a module-level function to allow pickling for ProcessPoolExecutor.
    It returns both pair counts and which sequences contain which pairs.

    Args:
        sequences: Batch of byte sequences
        min_frequency: Minimum frequency threshold
        seq_offset: Offset to add to sequence indices (for batching)

    Returns:
        Tuple of (pair_counts, pair_to_sequences)
    """
    local_counts: Counter[tuple[int, int]] = Counter()
    pair_to_sequences: dict[tuple[int, int], set[int]] = defaultdict(set)

    for seq_idx, seq in enumerate(sequences):
        # Skip sequences that are too short to have pairs
        if len(seq) < 2:
            continue

        # Count all adjacent pairs in this sequence
        global_seq_idx = seq_offset + seq_idx
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            local_counts[pair] += 1
            pair_to_sequences[pair].add(global_seq_idx)

    # Filter by minimum frequency
    if min_frequency > 1:
        filtered_counts = {
            pair: count
            for pair, count in local_counts.items()
            if count >= min_frequency
        }
        # Also filter pair_to_sequences to match
        filtered_mapping = {
            pair: seqs
            for pair, seqs in pair_to_sequences.items()
            if pair in filtered_counts
        }
        return filtered_counts, filtered_mapping

    return dict(local_counts), dict(pair_to_sequences)


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
        vocab: Mapping[bytes, int],
        merges: Sequence[tuple[bytes, bytes]],
        special_tokens: Sequence[str],
    ) -> None:
        """Initialize a BBPEModel.

        Args:
            vocab: Final token-to-id mapping.
            merges: Sequence of merge operations, each represented as a pair
                of tokens.
            special_tokens: Special tokens that should be handled separately
                by the tokenizer.
        """
        self.vocab: dict[bytes, int] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens)


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

    Only interfaces and high-level algorithm descriptions are provided here.
    All function bodies are left as `pass` on purpose.
    """

    def __init__(self, config: BBPETrainerConfig | None = None) -> None:
        """Initialize a BBPETrainer.

        Args:
            config: Optional configuration object. If omitted, a default
                configuration is created.
        """
        self.config: BBPETrainerConfig = config or BBPETrainerConfig()
        self._vocab: dict[bytes, int] = {}
        self._merges: list[tuple[bytes, bytes]] = []
        self._sequences: list[list[int]] = []  # Store corpus sequences for merge loop

    # === Public API ===

    def train(self, files: Sequence[str | Path]) -> BBPEModel:
        """Train a BBPE model from one or more text files.

        This is the high-level orchestration method. It calls the three
        private stages in order:

        1. `_stage1_preprocess_corpus` – maps files into memory and yields
           tokenized byte sequences in parallel.
        2. `_stage2_initial_pair_counts` – runs a MapReduce-style pass to
           compute global pair frequencies.
        3. `_stage3_merge_loop` – repeatedly picks the most frequent pair
           using a max-heap and applies merges with incremental updates.

        Args:
            files: Paths to input text files that constitute the training
                corpus. Files are assumed to be UTF-8 encoded.

        Returns:
            A `BBPEModel` instance containing the learned vocabulary and
            merge operations.

        Notes:
            - This interface is designed for large corpora that do not fit
              into RAM. The internal use of memory mapping and chunked
              processing makes the trainer scalable.
            - The actual implementation must be careful to keep memory usage
              nearly linear in the number of unique token pairs, not the
              raw corpus size.
        """
        # Validate input
        if not files:
            raise ValueError("At least one file must be provided")

        # Convert all files to Path objects for type consistency
        path_files = [Path(f) if isinstance(f, str) else f for f in files]

        # Stage 1: Preprocess corpus
        sequences = list(self._stage1_preprocess_corpus(path_files))

        # Handle empty corpus
        if not sequences:
            # Return model with only base vocabulary
            vocab: dict[bytes, int] = {}
            next_id = 0

            # Add all single bytes (0-255)
            for byte_val in range(256):
                vocab[bytes([byte_val])] = next_id
                next_id += 1

            # Add special tokens
            for special_token in self.config.special_tokens:
                vocab[special_token.encode('utf-8')] = next_id
                next_id += 1

            self._vocab = vocab
            self._merges = []

            return BBPEModel(
                vocab=vocab,
                merges=[],
                special_tokens=list(self.config.special_tokens)
            )

        # Save sequences for Stage 3
        self._sequences = sequences

        # Stage 2: Count pairs and build pair-to-sequences mapping
        pair_counts, pair_to_sequences = self._stage2_initial_pair_counts(sequences)

        # Stage 3: Merge loop
        vocab, merges = self._stage3_merge_loop(pair_counts, pair_to_sequences)

        # Store results in instance
        self._vocab = vocab
        self._merges = merges

        # Return model
        return BBPEModel(
            vocab=vocab,
            merges=merges,
            special_tokens=list(self.config.special_tokens)
        )

    def save(self, output_dir: str | Path) -> None:
        """Persist the trained model to disk.

        Args:
            output_dir: Directory where vocabulary and merges should be
                written. A typical layout would include:

                * ``vocab.json`` – token to index mapping.
                * ``merges.txt`` – merge rules in application order.

        Raises:
            RuntimeError: If called before `train`, i.e., when no model has
                been learned yet.
        """


        # Validate that model has been trained
        if not self._vocab:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary as JSON
        vocab_file = output_path / "vocab.json"
        # Convert bytes keys to strings using latin-1 (preserves all byte values 0-255)
        vocab_str: dict[str, int] = {}
        for token_bytes, idx in self._vocab.items():
            # Use latin-1 to preserve all byte values 0-255
            token_str = token_bytes.decode('latin-1')
            vocab_str[token_str] = idx

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)

        # Save merges as text file
        merges_file = output_path / "merges.txt"
        with open(merges_file, 'w', encoding='utf-8') as f:
            for token1, token2 in self._merges:
                # Decode bytes to strings using latin-1
                t1 = token1.decode('latin-1')
                t2 = token2.decode('latin-1')
                _ = f.write(f"{t1} {t2}\n")

    # === Stage I: Preprocessing (I/O bound) ===

    def _stage1_preprocess_corpus(
        self, files: Sequence[Path]
    ) -> Iterable[list[int]]:
        """Preprocess corpus and yield byte-level token sequences.

        This stage addresses the disk I/O bottleneck.

        Algorithm (recommended implementation):
            1. For each file, create an `mmap` view instead of reading the
                file into memory. This allows the OS page cache to handle
                actual disk I/O.
            2. Logically split each memory-mapped file into chunks of roughly
                `config.chunk_size_bytes`. Overlap a few bytes between chunks
                to avoid splitting multi-byte UTF-8 sequences.
            3. Use a thread pool (or process pool, depending on the Python
                runtime) with up to `config.max_workers` workers. Each worker:
                * Decodes its chunk into Unicode text.
                * Applies any normalization rules (e.g., NFKC).
                * Converts text into a byte-level representation.
            4. Yield the resulting byte sequences as lists of integers
                (0–255) to subsequent stages.

        Args:
            files: Paths to the corpus files, already normalized to `Path`
                objects by the caller.

        Returns:
            An iterable of byte-id sequences. In practice, this may be a
            generator that streams data to Stage II.

        Notes:
            - Python's `mmap` module is a natural fit here and mirrors the
              use of `mmap` in lower-level languages.
            - Chunk boundaries must be handled carefully to avoid corrupting
              Unicode characters and to keep tokenization deterministic.
        """

        def find_utf8_boundary(data: bytes, pos: int) -> int:
            """Find a valid UTF-8 character boundary at or before pos.

            UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF).
            We scan backwards to find a byte that is NOT a continuation byte.
            """
            if pos >= len(data):
                return len(data)

            # Scan backwards to find a non-continuation byte
            while pos > 0 and (data[pos] & 0b11000000) == 0b10000000:
                pos -= 1

            return pos

        def process_chunk(file_path: Path, start: int, end: int) -> list[int]:
            """Process a single chunk of a file.

            Args:
                file_path: Path to the file
                start: Start byte offset
                end: End byte offset

            Returns:
                List of byte values (0-255) after normalization
            """
            try:
                with open(file_path, 'rb') as f:
                    # For small chunks or files, just read directly
                    if end - start < 1024:  # Less than 1KB
                        _ = f.seek(start)
                        chunk_data = f.read(end - start)
                    else:
                        # Use mmap for larger chunks
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            chunk_data = mm[start:end]

                # Decode UTF-8 with error handling
                try:
                    text = chunk_data.decode('utf-8')
                except UnicodeDecodeError as e:
                    # Raise a more informative error for non-UTF-8 files
                    raise ValueError(
                        f"File {file_path} contains invalid UTF-8 at position {start + e.start}. "  # pyright: ignore[reportImplicitStringConcatenation]
                        "All input files must be valid UTF-8 encoded text."
                    ) from e

                # Apply NFKC normalization
                normalized = unicodedata.normalize('NFKC', text)

                # Normalize line endings to Unix-style (\n) for consistency
                normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')

                # Convert to bytes and return as list of integers
                byte_data = normalized.encode('utf-8')
                return list(byte_data)

            except FileNotFoundError as exc:
                raise FileNotFoundError(f"File not found: {file_path}") from exc

        def get_chunks(file_path: Path) -> list[tuple[int, int]]:
            """Split a file into chunks, respecting UTF-8 boundaries.

            Returns:
                List of (start_offset, end_offset) tuples
            """
            file_size = file_path.stat().st_size

            # Empty file
            if file_size == 0:
                return []

            # Small file - process as single chunk
            if file_size <= self.config.chunk_size_bytes:
                return [(0, file_size)]

            chunks: list[tuple[int, int]] = []
            start = 0

            # Use mmap to read only boundary bytes, avoiding loading entire file
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    while start < file_size:
                        # Calculate tentative end
                        tentative_end = min(start + self.config.chunk_size_bytes, file_size)

                        # Adjust to UTF-8 boundary if not at end of file
                        if tentative_end < file_size:
                            # Only read a small window around the boundary (max 4 bytes back for UTF-8)
                            boundary_start = max(0, tentative_end - 4)
                            boundary_data = mm[boundary_start:tentative_end + 1]
                            # Adjust position within the small buffer
                            local_pos = tentative_end - boundary_start
                            adjusted_local = find_utf8_boundary(boundary_data, local_pos)
                            actual_end = boundary_start + adjusted_local
                        else:
                            actual_end = file_size

                        # Avoid empty chunks
                        if actual_end > start:
                            chunks.append((start, actual_end))
                            start = actual_end
                        else:
                            # Edge case: move forward at least one byte
                            start += 1

            return chunks

        # Process all files
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunk processing tasks with order tracking
            futures: list[Future[list[int]]] = []

            for file_path in files:

                # Check file exists
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Get chunks for this file
                chunks = get_chunks(file_path)

                # Submit processing tasks for each chunk
                for start, end in chunks:
                    future = executor.submit(process_chunk, file_path, start, end)
                    futures.append(future)

            # Yield results in order (important for deterministic output)
            for future in futures:
                result = future.result()
                # Only yield non-empty sequences
                if result:
                    yield result

    # === Stage II: Initial counting (CPU / hash bound) ===

    def _stage2_initial_pair_counts(
        self, sequences: Iterable[list[int]]
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
        """Compute global pair frequencies via a MapReduce-style pass.

        This stage is CPU and memory bound and benefits from parallel
        hashing.

        Algorithm (recommended implementation):
            1. Partition the incoming sequences into work units (e.g., in
                batches of sentences).
            2. Use a worker pool with up to `config.max_workers` to process
                these units in parallel (Map step):
                * For each sequence, iterate over adjacent pairs
                  :math:`(b_i, b_{i+1})`.
                * Update a local hash map from pair to count. The keys may be
                  tuples of integers or a compact integer encoding of a pair.
                * Use a fast, non-cryptographic hash (e.g., xxhash or a
                  Rust-style FxHash equivalent) to minimize CPU overhead.
            3. After mapping, merge all local hash maps into a single global
                map (Reduce step), summing counts for identical pairs.
            4. Optionally discard pairs whose frequency is below
                `config.min_frequency` to save memory.

        Args:
            sequences: Iterable of byte-id sequences produced by
                `_stage1_preprocess_corpus`.

        Returns:
            A tuple of (pair_counts, pair_to_sequences) where:
            
            * ``pair_counts`` is a dictionary mapping byte pairs to their 
              global frequencies.
            * ``pair_to_sequences`` is a dictionary mapping pairs to the set
              of sequence indices that contain them.

        Notes:
            - The MapReduce pattern matches well with Python's
              `concurrent.futures` API.
            - To keep memory usage manageable, partial reductions can be
              performed periodically (tree-style reduce).
        """
        # Convert to list for batching (needed for ProcessPoolExecutor)
        seq_list = list(sequences)

        # Handle empty input
        if not seq_list:
            return {}, {}

        # Batch size for parallel processing
        batch_size = 1000

        # For small datasets, use single-threaded processing
        if len(seq_list) < batch_size:
            counts, mapping = _process_batch_for_pairs_with_tracking(
                seq_list, self.config.min_frequency, 0
            )
            return counts, mapping

        # Split into batches
        batches = [
            seq_list[i:i + batch_size]
            for i in range(0, len(seq_list), batch_size)
        ]

        # Parallel processing with ProcessPoolExecutor (Map step)
        global_counts: Counter[tuple[int, int]] = Counter()
        global_pair_to_sequences: dict[tuple[int, int], set[int]] = defaultdict(set)

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches with proper sequence offsets
            futures: list[Future[tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]]] = []
            for batch_idx, batch in enumerate(batches):
                seq_offset = batch_idx * batch_size
                future: Future[tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]] = executor.submit(
                    _process_batch_for_pairs_with_tracking, 
                    batch, 
                    1,  # min_freq=1 for local counts
                    seq_offset
                )
                futures.append(future)

            # Collect and merge results (Reduce step)
            for future in futures:
                result: tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]] = future.result()
                local_counts: dict[tuple[int, int], int] = result[0]
                local_mapping: dict[tuple[int, int], set[int]] = result[1]
                global_counts.update(local_counts)
                # Merge the pair_to_sequences mappings
                for pair, seq_indices in local_mapping.items():
                    global_pair_to_sequences[pair] |= seq_indices

        # Filter by min_frequency
        filtered_counts = {
            pair: count
            for pair, count in global_counts.items()
            if count >= self.config.min_frequency
        }
        
        # Filter pair_to_sequences to only include pairs that meet min_frequency
        filtered_mapping = {
            pair: seq_indices
            for pair, seq_indices in global_pair_to_sequences.items()
            if pair in filtered_counts
        }

        return filtered_counts, filtered_mapping

    # === Stage III: Merge loop (algorithmic complexity bound) ===

    def _stage3_merge_loop(
        self, 
        pair_counts: dict[tuple[int, int], int],
        pair_to_sequences: dict[tuple[int, int], set[int]]
    ) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
        """Run the BBPE merge loop using a max-heap priority queue.

        This stage is dominated by algorithmic complexity: selecting the
        best pair and updating statistics after each merge. A priority
        queue (max-heap) gives an efficient implementation.

        Algorithm (recommended implementation):
            1. Initialize the base vocabulary with all single bytes
               (0–255) plus `config.special_tokens`.
            2. Build a max-heap (priority queue) from `pair_counts`, where
                each node stores (negative_frequency, pair_id) so that the
                top of the heap is always the most frequent pair.
            3. Repeatedly:
               * Pop the most frequent pair :math:`p^\\*` from the heap.
               * If its current frequency is below `config.min_frequency`,
                 or the vocabulary has reached `config.vocab_size`,
                 terminate.
               * Add a new token corresponding to :math:`p^\\*` to the
                 vocabulary and record this merge.
               * Apply the merge to the corpus representation (conceptually,
                 or via references to token graphs).
               * Call `_update_pair_counts_incremental` to adjust only the
                 counts of pairs affected by the merge, pushing updated
                 entries back into the heap.
               * Periodically rebuild the heap to clear stale entries and
                 prevent heap bloat.

        Args:
            pair_counts: Initial pair frequencies computed by
                `_stage2_initial_pair_counts`.
            pair_to_sequences: Mapping from pairs to sequence indices that
                contain them, also computed by `_stage2_initial_pair_counts`.

        Returns:
            A tuple ``(vocab, merges)`` where:

            * ``vocab`` is a mapping from bytes tokens to integer ids.
            * ``merges`` is the sequence of merge operations in order,
              each as a pair of bytes tokens.

        Notes:
            - The incremental update strategy mirrors the core logic of
              Hugging Face `tokenizers`: recompute only local neighborhoods
              instead of rescanning the full corpus.
            - To ensure correctness, lazy heap updates can be used
              (keeping a separate map of \"current\" frequencies and ignoring
              stale heap entries).
            - Heap is periodically rebuilt when it grows too large relative
              to the number of active pairs, preventing memory bloat.
        """


        # 1. Initialize base vocabulary
        vocab: dict[bytes, int] = {}
        id_to_token: dict[int, bytes] = {}  # Reverse mapping for looking up tokens by ID
        next_id = 0

        # Add all single bytes (0-255)
        for byte_val in range(256):
            token = bytes([byte_val])
            vocab[token] = next_id
            id_to_token[next_id] = token
            next_id += 1

        # Add special tokens
        for special_token in self.config.special_tokens:
            token_bytes = special_token.encode('utf-8')
            if token_bytes not in vocab:
                vocab[token_bytes] = next_id
                id_to_token[next_id] = token_bytes
                next_id += 1

        # 2. Build max-heap and maintain global pair counts
        heap: list[tuple[int, tuple[int, int]]] = []
        global_pair_counts = dict(pair_counts)  # Copy to maintain mutable state

        for pair, count in global_pair_counts.items():
            if count >= self.config.min_frequency:
                heapq.heappush(heap, (-count, pair))

        # 3. Track which sequences contain which pairs (for incremental updates)
        # Make a mutal copy since we'll modify it during merges
        pair_to_sequences = defaultdict(set, pair_to_sequences)

        # 4. Merge loop
        merges: list[tuple[bytes, bytes]] = []

        while heap and len(vocab) < self.config.vocab_size:
            # Pop the most frequent pair
            neg_count, (id1, id2) = heapq.heappop(heap)
            count = -neg_count

            # Check if this pair is still valid (lazy deletion from heap)
            if (id1, id2) not in global_pair_counts:
                continue
            if global_pair_counts[(id1, id2)] != count:
                # Stale entry, re-push with current count
                current_count = global_pair_counts[(id1, id2)]
                if current_count >= self.config.min_frequency:
                    heapq.heappush(heap, (-current_count, (id1, id2)))
                continue

            # Check frequency threshold
            if count < self.config.min_frequency:
                break

            # Look up the tokens by their IDs
            token1 = id_to_token[id1]
            token2 = id_to_token[id2]
            new_token = token1 + token2  # Concatenate bytes

            # Add to vocabulary
            if new_token not in vocab:
                vocab[new_token] = next_id
                id_to_token[next_id] = new_token
                merges.append((token1, token2))

                # Get affected sequences for this merge
                affected_seq_indices = pair_to_sequences.get((id1, id2), set())

                # Apply merge and update pair counts incrementally
                if self._sequences and affected_seq_indices:
                    # Track per-sequence pair changes for pair_to_sequences update
                    # Maps pair -> set of seq_idx where the pair was removed/added
                    pairs_removed_from: dict[tuple[int, int], set[int]] = defaultdict(set)
                    pairs_added_to: dict[tuple[int, int], set[int]] = defaultdict(set)

                    self._update_pair_counts_incremental(
                        merged_pair=(id1, id2),
                        affected_seq_indices=affected_seq_indices,
                        heap_view=heap,
                        global_pair_counts=global_pair_counts,
                        new_id=next_id,
                        pairs_removed_from=pairs_removed_from,
                        pairs_added_to=pairs_added_to,
                    )

                    # Update pair_to_sequences mapping incrementally
                    # Remove the merged pair entirely
                    if (id1, id2) in pair_to_sequences:
                        del pair_to_sequences[(id1, id2)]

                    # Remove old neighboring pairs from their respective sequences
                    for pair, seq_indices in pairs_removed_from.items():
                        if pair in pair_to_sequences:
                            pair_to_sequences[pair] -= seq_indices
                            if not pair_to_sequences[pair]:
                                del pair_to_sequences[pair]

                    # Add new pairs to their respective sequences
                    for pair, seq_indices in pairs_added_to.items():
                        pair_to_sequences[pair] |= seq_indices

                    # Periodically rebuild heap to prevent bloat
                    # When heap grows too large relative to active pairs, rebuild it
                    if len(heap) > 3 * len(global_pair_counts):
                        heap = [
                            (-count, pair)
                            for pair, count in global_pair_counts.items()
                            if count >= self.config.min_frequency
                        ]
                        heapq.heapify(heap)

                next_id += 1

        return vocab, merges



    def _update_pair_counts_incremental(
        self,
        merged_pair: tuple[int, int],
        affected_seq_indices: set[int],
        heap_view: list[tuple[int, tuple[int, int]]],
        global_pair_counts: dict[tuple[int, int], int],
        new_id: int,
        pairs_removed_from: dict[tuple[int, int], set[int]],
        pairs_added_to: dict[tuple[int, int], set[int]],
    ) -> None:
        """Incrementally update pair counts after applying a merge.

        This helper method is responsible for updating statistics for only
        those positions in the corpus that are affected by a newly merged
        pair. It is conceptually similar to the inner loop used by Hugging
        Face `tokenizers`.

        Algorithm (recommended implementation):
            1. For each affected sequence, identify spans where the merged
               pair appears and replace them with the new token id.
            2. For each position where the tokenization changes, decrement
               counts of old neighboring pairs and increment counts of new
               neighboring pairs in `global_pair_counts`.
            3. For every pair whose frequency changed, push a new entry into
               the heap (`heap_view`). The actual heap implementation can be
               `heapq` plus an indirection table to handle stale entries.

        Args:
            merged_pair: The pair of token ids that has just been merged into
                a single new token.
            affected_seq_indices: Set of sequence indices that contain
                occurrences of `merged_pair`.
            heap_view: A heap-like object that supports `heappush` and
                `heappop`. The exact type is left unspecified here so that
                different heap strategies can be experimented with.
            global_pair_counts: Global dictionary of pair frequencies that
                must remain consistent with the current corpus state.
            new_id: The token ID assigned to the newly merged pair.
            pairs_removed_from: Output dict mapping pairs to the set of
                sequence indices from which they were removed.
            pairs_added_to: Output dict mapping pairs to the set of
                sequence indices to which they were added.

        Returns:
            None. The method mutates `global_pair_counts` and updates the
            heap in-place.

        Notes:
            - The key challenge is to keep this update local and avoid
              touching unaffected parts of the corpus, which would be too
              slow for large datasets.
            - In a real implementation, `affected_sequences` would likely be
              represented using more compact structures than raw Python
              lists to reduce interpreter overhead.
        """
        id1, id2 = merged_pair

        # Track pairs that changed
        pairs_changed: Counter[tuple[int, int]] = Counter()

        for seq_idx in affected_seq_indices:
            seq = self._sequences[seq_idx]
            
            # Use collect-rebuild strategy to avoid O(n²) from repeated list slicing
            new_seq: list[int] = []
            i = 0
            
            while i < len(seq):
                # Check if we can merge at this position
                if i < len(seq) - 1 and seq[i] == id1 and seq[i + 1] == id2:
                    # Record old neighboring pairs before merge
                    if new_seq:  # Has left neighbor
                        old_left_pair = (new_seq[-1], seq[i])
                        pairs_changed[old_left_pair] -= 1
                        pairs_removed_from[old_left_pair].add(seq_idx)
                    if i + 2 < len(seq):  # Has right neighbor
                        old_right_pair = (seq[i + 1], seq[i + 2])
                        pairs_changed[old_right_pair] -= 1
                        pairs_removed_from[old_right_pair].add(seq_idx)

                    # Decrement the merged pair itself
                    pairs_changed[(id1, id2)] -= 1

                    # Apply the merge by adding the new token
                    new_seq.append(new_id)
                    
                    # Record new neighboring pairs after merge
                    if len(new_seq) > 1:  # Has left neighbor
                        new_left_pair = (new_seq[-2], new_id)
                        pairs_changed[new_left_pair] += 1
                        pairs_added_to[new_left_pair].add(seq_idx)
                    if i + 2 < len(seq):  # Has right neighbor
                        new_right_pair = (new_id, seq[i + 2])
                        pairs_changed[new_right_pair] += 1
                        pairs_added_to[new_right_pair].add(seq_idx)

                    # Skip both tokens that were merged
                    i += 2
                else:
                    # No merge, keep the token
                    new_seq.append(seq[i])
                    i += 1
            
            # Replace the sequence with the rebuilt one
            self._sequences[seq_idx] = new_seq

        # Update global pair counts and heap
        for pair, delta in pairs_changed.items():
            if delta == 0:
                continue

            # Update global counts
            if pair in global_pair_counts:
                global_pair_counts[pair] += delta
                if global_pair_counts[pair] <= 0:
                    del global_pair_counts[pair]
            elif delta > 0:
                global_pair_counts[pair] = delta

            # Push updated pair to heap if it meets min_frequency
            if pair in global_pair_counts:
                count = global_pair_counts[pair]
                if count >= self.config.min_frequency:
                    heapq.heappush(heap_view, (-count, pair))
