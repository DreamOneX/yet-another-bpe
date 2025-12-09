"""Tests for BPE implementation."""

import unicodedata
from pathlib import Path

import pytest

from yet_another_bpe.bbpe import BBPEModel, BBPETrainer, BBPETrainerConfig


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


def _build_pair_to_sequences_for_tests(sequences: list[list[int]]) -> dict[tuple[int, int], set[int]]:
    """Helper function for test_bpe.py to build pair_to_sequences mapping."""
    from collections import defaultdict
    pair_to_sequences: dict[tuple[int, int], set[int]] = defaultdict(set)
    for seq_idx, seq in enumerate(sequences):
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_to_sequences[pair].add(seq_idx)
    return dict(pair_to_sequences)


class TestStage1Preprocessing:
    """Tests for Stage 1: Preprocessing (I/O bound)"""

    def test_stage1_simple_ascii_text(self):
        """Test preprocessing of simple ASCII text."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "simple.txt"
        assert test_file.exists(), f"Test file {test_file} not found"
        
        # Call preprocessing
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        # Verify we got sequences
        assert len(sequences) > 0, "Should return at least one sequence"
        
        # Verify all sequences contain only bytes (0-255)
        for seq in sequences:
            assert isinstance(seq, list), "Each sequence should be a list"
            assert all(isinstance(b, int) and 0 <= b <= 255 for b in seq), \
                "All elements should be integers in range 0-255"
        
        # Verify the content matches expected bytes
        # "Hello world!" -> [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]
        expected_bytes = "Hello world!".encode('utf-8')
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        assert all_bytes == expected_bytes, f"Expected {expected_bytes}, got {all_bytes}"

    def test_stage1_unicode_text(self):
        """Test preprocessing of Unicode text with normalization."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "unicode.txt"
        assert test_file.exists(), f"Test file {test_file} not found"
        
        # Read original content and apply NFKC normalization
        original_text = test_file.read_text(encoding='utf-8')
        normalized_text = unicodedata.normalize('NFKC', original_text)
        expected_bytes = normalized_text.encode('utf-8')
        
        # Call preprocessing
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        # Verify sequences
        assert len(sequences) > 0
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        
        # Should match normalized version
        assert all_bytes == expected_bytes, \
            f"Unicode text should be NFKC normalized"

    def test_stage1_multiline_text(self):
        """Test preprocessing of multiline text."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "multiline.txt"
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        assert len(sequences) > 0
        # Verify newlines are preserved
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        assert b'\n' in all_bytes or b'\r\n' in all_bytes, \
            "Newlines should be preserved"

    def test_stage1_empty_file(self):
        """Test preprocessing of empty file (edge case)."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "empty.txt"
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        # Empty file should return empty list or list with empty sequence
        assert len(sequences) == 0 or (len(sequences) == 1 and len(sequences[0]) == 0), \
            "Empty file should produce no sequences or one empty sequence"

    def test_stage1_large_file_chunking(self):
        """Test that large files are properly chunked."""
        # Use small chunk size to force chunking
        config = BBPETrainerConfig(
            chunk_size_bytes=1024,  # 1KB chunks
            max_workers=2
        )
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "large.txt"
        assert test_file.exists(), f"Large test file {test_file} not found"
        
        # Verify file is larger than chunk size
        file_size = test_file.stat().st_size
        assert file_size > config.chunk_size_bytes, \
            f"Test file should be larger than chunk size for this test"
        
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        # Should get multiple sequences due to chunking
        assert len(sequences) > 0
        
        # Verify all bytes are valid
        for seq in sequences:
            assert all(0 <= b <= 255 for b in seq)
        
        # Verify total content matches (no data loss from chunking)
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        expected_text = test_file.read_text(encoding='utf-8')
        expected_normalized = unicodedata.normalize('NFKC', expected_text)
        expected_bytes = expected_normalized.encode('utf-8')
        
        assert all_bytes == expected_bytes, \
            "Chunking should not lose or corrupt data"

    def test_stage1_multiple_files(self):
        """Test preprocessing multiple files."""
        config = BBPETrainerConfig(max_workers=2)
        trainer = BBPETrainer(config)
        
        test_files = [
            TEST_DATA_DIR / "simple.txt",
            TEST_DATA_DIR / "unicode.txt",
            TEST_DATA_DIR / "multiline.txt"
        ]
        
        sequences = list(trainer._stage1_preprocess_corpus(test_files))
        
        # Should get sequences from all files
        assert len(sequences) >= len(test_files), \
            "Should process all files"
        
        # All sequences should be valid
        for seq in sequences:
            assert isinstance(seq, list)
            assert all(isinstance(b, int) and 0 <= b <= 255 for b in seq)

    def test_stage1_parallel_processing(self):
        """Test that parallel processing works correctly."""
        # Test with different worker counts
        for max_workers in [1, 2, 4]:
            config = BBPETrainerConfig(max_workers=max_workers)
            trainer = BBPETrainer(config)
            
            test_files = [
                TEST_DATA_DIR / "simple.txt",
                TEST_DATA_DIR / "unicode.txt"
            ]
            
            sequences = list(trainer._stage1_preprocess_corpus(test_files))
            
            # Results should be consistent regardless of worker count
            assert len(sequences) > 0
            for seq in sequences:
                assert all(0 <= b <= 255 for b in seq)

    def test_stage1_nonexistent_file(self):
        """Test that nonexistent files raise appropriate error."""
        config = BBPETrainerConfig()
        trainer = BBPETrainer(config)
        
        nonexistent = TEST_DATA_DIR / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            _ = list(trainer._stage1_preprocess_corpus([nonexistent]))

    def test_stage1_utf8_boundary_handling(self):
        """Test that UTF-8 character boundaries are respected in chunking."""
        # Create a file with multi-byte UTF-8 characters
        test_file = TEST_DATA_DIR / "utf8_boundary_test.txt"
        # Chinese characters are 3 bytes each in UTF-8
        content = "你好世界" * 1000  # Lots of multi-byte chars
        test_file.write_text(content, encoding='utf-8')
        
        # Use very small chunk size to test boundary handling
        config = BBPETrainerConfig(
            chunk_size_bytes=100,  # Small chunks
            max_workers=1
        )
        trainer = BBPETrainer(config)
        
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        # Reconstruct and verify no corruption
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        reconstructed = all_bytes.decode('utf-8')
        expected_normalized = unicodedata.normalize('NFKC', content)
        
        assert reconstructed == expected_normalized, \
            "UTF-8 boundaries should be respected, no character corruption"
        
        # Cleanup
        test_file.unlink()


class TestStage2PairCounting:
    """Tests for Stage 2: Pair Counting (CPU/hash bound)"""

    def test_stage2_simple_pairs(self):
        """Test simple pair counting."""
        config = BBPETrainerConfig(min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        # "Hello" -> [72, 101, 108, 108, 111]
        sequences = [[72, 101, 108, 108, 111]]
        
        pair_counts, _ = trainer._stage2_initial_pair_counts(sequences)
        
        # Expected pairs: (72,101), (101,108), (108,108), (108,111)
        assert (72, 101) in pair_counts  # 'H' + 'e'
        assert (101, 108) in pair_counts  # 'e' + 'l'
        assert (108, 108) in pair_counts  # 'l' + 'l'
        assert (108, 111) in pair_counts  # 'l' + 'o'
        
        # Each pair appears once
        assert pair_counts[(72, 101)] == 1
        assert pair_counts[(101, 108)] == 1
        assert pair_counts[(108, 108)] == 1
        assert pair_counts[(108, 111)] == 1

    def test_stage2_repeated_pairs(self):
        """Test counting of repeated pairs across multiple sequences."""
        config = BBPETrainerConfig(min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Two identical sequences
        sequences = [
            [72, 101, 108, 108, 111],  # "Hello"
            [72, 101, 108, 108, 111],  # "Hello"
        ]
        
        pair_counts, _ = trainer._stage2_initial_pair_counts(sequences)
        
        # Each pair should appear twice
        assert pair_counts[(72, 101)] == 2
        assert pair_counts[(101, 108)] == 2
        assert pair_counts[(108, 108)] == 2
        assert pair_counts[(108, 111)] == 2

    def test_stage2_min_frequency_filter(self):
        """Test that pairs below min_frequency are filtered out."""
        config = BBPETrainerConfig(min_frequency=3, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Create sequences where some pairs appear less than 3 times
        sequences = [
            [1, 2, 3],  # pairs: (1,2), (2,3) - each appears once
            [4, 5, 6],  # pairs: (4,5), (5,6) - each appears once
            [7, 8, 7, 8, 7, 8],  # pairs: (7,8), (8,7) - each appears 3 times
        ]
        
        pair_counts, _ = trainer._stage2_initial_pair_counts(sequences)
        
        # Only pairs with frequency >= 3 should be present
        assert (7, 8) in pair_counts
        assert pair_counts[(7, 8)] == 3
        
        # (8,7) appears only 2 times, should be filtered out
        assert (8, 7) not in pair_counts
        
        # Low frequency pairs should not be in result
        assert (1, 2) not in pair_counts
        assert (2, 3) not in pair_counts
        assert (4, 5) not in pair_counts
        assert (5, 6) not in pair_counts

    def test_stage2_empty_sequences(self):
        """Test handling of empty sequence list."""
        config = BBPETrainerConfig(min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        sequences = []
        pair_counts, _ = trainer._stage2_initial_pair_counts(sequences)
        
        # Should return empty dict
        assert pair_counts == {}

    def test_stage2_single_byte_sequences(self):
        """Test sequences with single bytes (no pairs)."""
        config = BBPETrainerConfig(min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        sequences = [[1], [2], [3]]
        pair_counts, _ = trainer._stage2_initial_pair_counts(sequences)
        
        # No pairs possible
        assert pair_counts == {}

    def test_stage2_parallel_counting(self):
        """Test that parallel counting produces consistent results."""
        # Create a larger dataset
        sequences = [[i, i+1, i+2] for i in range(100)]
        
        # Test with different worker counts
        results = []
        for max_workers in [1, 2, 4]:
            config = BBPETrainerConfig(min_frequency=1, max_workers=max_workers)
            trainer = BBPETrainer(config)
            pair_counts, _ = trainer._stage2_initial_pair_counts(sequences)
            results.append(pair_counts)
        
        # All results should be identical
        assert results[0] == results[1] == results[2], \
            "Parallel processing should produce consistent results"

    def test_stage2_integration_with_stage1(self):
        """Test Stage 2 with real output from Stage 1."""
        config = BBPETrainerConfig(min_frequency=1, max_workers=1)  # Use min_frequency=1
        trainer = BBPETrainer(config)
        
        # Use a real test file
        test_file = TEST_DATA_DIR / "simple.txt"
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        # Count pairs
        pair_counts, _ = trainer._stage2_initial_pair_counts(sequences)
        
        # Should have some pairs
        assert len(pair_counts) > 0
        
        # All counts should be >= min_frequency
        for count in pair_counts.values():
            assert count >= config.min_frequency


class TestStage3MergeLoop:
    """Tests for Stage 3: Merge Loop (algorithmic complexity bound)"""

    def test_stage3_vocab_initialization(self):
        """Test that vocabulary is properly initialized with bytes 0-255 and special tokens."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Empty pair counts for testing vocab initialization only
        pair_counts = {}
        pair_to_sequences = {}
        
        vocab, merges = trainer._stage3_merge_loop(pair_counts, pair_to_sequences)
        
        # Should have 256 bytes + 4 special tokens = 260
        assert len(vocab) == 260
        
        # Check all bytes 0-255 are present
        for byte_val in range(256):
            token = bytes([byte_val])
            assert token in vocab
            assert vocab[token] == byte_val
        
        # Check special tokens
        assert b'[PAD]' in vocab
        assert b'[UNK]' in vocab
        assert b'[BOS]' in vocab
        assert b'[EOS]' in vocab
        
        # No merges should have occurred
        assert len(merges) == 0

    def test_stage3_basic_merge(self):
        """Test basic merge operation."""
        config = BBPETrainerConfig(vocab_size=265, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Simulate some pair counts
        pair_counts = {
            (72, 101): 10,  # 'H' + 'e' - highest frequency
            (101, 108): 5,  # 'e' + 'l'
            (108, 108): 3,  # 'l' + 'l'
            (111, 32): 2,   # 'o' + ' '
        }
        
        # Need to set up sequences for full implementation
        trainer._sequences = [
            [72, 101, 108, 108, 111],  # "Hello"
            [72, 101, 108, 108, 111],  # "Hello"
        ]
        pair_to_sequences = _build_pair_to_sequences_for_tests(trainer._sequences)
        
        vocab, merges = trainer._stage3_merge_loop(pair_counts, pair_to_sequences)
        
        # Should have base vocab (260) + some merges
        assert len(vocab) >= 260
        
        # Should have performed some merges
        assert len(merges) > 0
        
        # First merge should be the most frequent pair
        assert merges[0] == (b'H', b'e') or merges[0] == (bytes([72]), bytes([101]))
        
        # All merges should be tuples of bytes
        for merge in merges:
            assert isinstance(merge, tuple)
            assert len(merge) == 2
            assert isinstance(merge[0], bytes)
            assert isinstance(merge[1], bytes)

    def test_stage3_merge_ordering(self):
        """Test that merges are performed in frequency order."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        pair_counts = {
            (65, 66): 100,  # 'A' + 'B' - highest
            (67, 68): 50,   # 'C' + 'D' - medium
            (69, 70): 10,   # 'E' + 'F' - lowest
        }
        
        trainer._sequences = [[65, 66], [67, 68], [69, 70]]
        pair_to_sequences = _build_pair_to_sequences_for_tests(trainer._sequences)
        
        vocab, merges = trainer._stage3_merge_loop(pair_counts, pair_to_sequences)
        
        # First merge should be highest frequency
        assert merges[0] == (b'A', b'B') or merges[0] == (bytes([65]), bytes([66]))

    def test_stage3_vocab_size_limit(self):
        """Test that merging stops when vocab_size is reached."""
        config = BBPETrainerConfig(vocab_size=262, min_frequency=1, max_workers=1)  # Only 2 merges allowed
        trainer = BBPETrainer(config)
        
        pair_counts = {
            (65, 66): 10,
            (67, 68): 9,
            (69, 70): 8,
            (71, 72): 7,
            (73, 74): 6,
        }
        
        trainer._sequences = [[65, 66], [67, 68], [69, 70], [71, 72], [73, 74]]
        pair_to_sequences = _build_pair_to_sequences_for_tests(trainer._sequences)
        
        vocab, merges = trainer._stage3_merge_loop(pair_counts, pair_to_sequences)
        
        # Should stop at vocab_size
        assert len(vocab) == 262
        
        # Should have exactly 2 merges (262 - 260 base vocab)
        assert len(merges) == 2

    def test_stage3_min_frequency_threshold(self):
        """Test that pairs below min_frequency are not merged."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=5, max_workers=1)
        trainer = BBPETrainer(config)
        
        pair_counts = {
            (65, 66): 10,  # Above threshold
            (67, 68): 5,   # At threshold
            (69, 70): 4,   # Below threshold
            (71, 72): 1,   # Below threshold
        }
        
        trainer._sequences = [[65, 66], [67, 68], [69, 70], [71, 72]]
        pair_to_sequences = _build_pair_to_sequences_for_tests(trainer._sequences)
        
        vocab, merges = trainer._stage3_merge_loop(pair_counts, pair_to_sequences)
        
        # Should only merge pairs with frequency >= 5
        assert len(merges) <= 2
        
        # Check that low-frequency pairs are not in merges
        low_freq_merges = [(b'E', b'F'), (b'G', b'H')]
        for merge in merges:
            assert merge not in low_freq_merges

    def test_stage3_special_tokens(self):
        """Test that special tokens are properly included in vocab."""
        config = BBPETrainerConfig(
            vocab_size=300,
            min_frequency=1,
            max_workers=1,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
        )
        trainer = BBPETrainer(config)
        
        pair_counts = {}
        pair_to_sequences = {}
        vocab, merges = trainer._stage3_merge_loop(pair_counts, pair_to_sequences)
        
        # Should have 256 bytes + 5 special tokens = 261
        assert len(vocab) == 261
        
        # All special tokens should be present
        assert b'[PAD]' in vocab
        assert b'[UNK]' in vocab
        assert b'[BOS]' in vocab
        assert b'[EOS]' in vocab
        assert b'[MASK]' in vocab
        
        # Special tokens should have IDs after the 256 bytes
        assert vocab[b'[PAD]'] >= 256
        assert vocab[b'[UNK]'] >= 256

    def test_stage3_integration_with_stage1_and_2(self):
        """Test Stage 3 with real output from Stage 1 and 2."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Use a real test file
        test_file = TEST_DATA_DIR / "simple.txt"
        sequences = list(trainer._stage1_preprocess_corpus([test_file]))
        
        # Save sequences for Stage 3
        trainer._sequences = sequences
        
        # Count pairs and pair_to_sequences
        pair_counts, pair_to_sequences = trainer._stage2_initial_pair_counts(sequences)
        
        # Run merge loop
        vocab, merges = trainer._stage3_merge_loop(pair_counts, pair_to_sequences)
        
        # Should have base vocab
        assert len(vocab) >= 260
        
        # All vocab keys should be bytes
        for token in vocab.keys():
            assert isinstance(token, bytes)
        
        # All merges should be tuples of bytes
        for merge in merges:
            assert isinstance(merge, tuple)
            assert len(merge) == 2
            assert isinstance(merge[0], bytes)
            assert isinstance(merge[1], bytes)


class TestTrainOrchestration:
    """Tests for train() orchestration method"""

    def test_train_simple_corpus(self):
        """Test training on a simple corpus."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "simple.txt"
        model = trainer.train([test_file])
        
        # Verify model is returned
        assert isinstance(model, BBPEModel)
        
        # Verify vocab contains base bytes + special tokens
        assert len(model.vocab) >= 260  # 256 bytes + 4 special tokens
        
        # Verify all vocab keys are bytes
        for token in model.vocab.keys():
            assert isinstance(token, bytes)
        
        # Verify merges is a list
        assert isinstance(model.merges, list)
        
        # Verify special tokens match config
        assert model.special_tokens == list(config.special_tokens)

    def test_train_empty_corpus(self):
        """Test training on empty file."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "empty.txt"
        model = trainer.train([test_file])
        
        # Should return model with only base vocab
        assert isinstance(model, BBPEModel)
        assert len(model.vocab) == 260  # 256 bytes + 4 special tokens
        assert len(model.merges) == 0  # No merges

    def test_train_multiple_files(self):
        """Test training on multiple files."""
        config = BBPETrainerConfig(vocab_size=280, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        
        files = [
            TEST_DATA_DIR / "simple.txt",
            TEST_DATA_DIR / "unicode.txt"
        ]
        model = trainer.train(files)
        
        # Verify model is returned
        assert isinstance(model, BBPEModel)
        assert len(model.vocab) >= 260

    def test_train_vocab_size_limit(self):
        """Test that vocab_size is respected."""
        config = BBPETrainerConfig(vocab_size=265, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "simple.txt"
        model = trainer.train([test_file])
        
        # Vocab size should not exceed limit
        assert len(model.vocab) <= 265

    def test_train_min_frequency(self):
        """Test that min_frequency is respected."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=10, max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "simple.txt"
        model = trainer.train([test_file])
        
        # With high min_frequency, should have few or no merges
        # (simple.txt is small, most pairs appear only once)
        assert len(model.merges) == 0 or len(model.merges) < 5

    def test_train_model_attributes(self):
        """Test that returned model has correct attributes."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "simple.txt"
        model = trainer.train([test_file])
        
        # Verify vocab type
        assert isinstance(model.vocab, dict)
        for key, value in model.vocab.items():
            assert isinstance(key, bytes)
            assert isinstance(value, int)
        
        # Verify merges type
        assert isinstance(model.merges, list)
        for merge in model.merges:
            assert isinstance(merge, tuple)
            assert len(merge) == 2
            assert isinstance(merge[0], bytes)
            assert isinstance(merge[1], bytes)
        
        # Verify special_tokens type
        assert isinstance(model.special_tokens, list)
        for token in model.special_tokens:
            assert isinstance(token, str)

    def test_train_integration(self):
        """Full integration test with all stages."""
        config = BBPETrainerConfig(vocab_size=280, min_frequency=1, max_workers=2)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "multiline.txt"
        model = trainer.train([test_file])
        
        # Verify all stages executed successfully
        assert isinstance(model, BBPEModel)
        assert len(model.vocab) >= 260
        
        # Verify trainer state was updated
        assert len(trainer._vocab) >= 260
        assert isinstance(trainer._merges, list)
        assert isinstance(trainer._sequences, list)
        
        # Verify base bytes are present
        for i in range(256):
            token = bytes([i])
            assert token in model.vocab
        
        # Verify special tokens are present
        for special_token in config.special_tokens:
            token_bytes = special_token.encode('utf-8')
            assert token_bytes in model.vocab


class TestModelPersistence:
    """Tests for save() method"""

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates output directory."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Train model
        test_file = TEST_DATA_DIR / "simple.txt"
        trainer.train([test_file])
        
        # Save to temp directory
        output_dir = tmp_path / "model"
        trainer.save(output_dir)
        
        # Verify directory exists
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_save_creates_vocab_file(self, tmp_path):
        """Test that vocab.json is created."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])
        
        output_dir = tmp_path / "model"
        trainer.save(output_dir)
        
        vocab_file = output_dir / "vocab.json"
        assert vocab_file.exists()
        
        # Verify it's valid JSON
        import json
        with open(vocab_file, encoding='utf-8') as f:
            vocab = json.load(f)
        assert isinstance(vocab, dict)
        assert len(vocab) >= 260

    def test_save_creates_merges_file(self, tmp_path):
        """Test that merges.txt is created."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])
        
        output_dir = tmp_path / "model"
        trainer.save(output_dir)
        
        merges_file = output_dir / "merges.txt"
        assert merges_file.exists()
        
        # Verify format
        with open(merges_file, encoding='utf-8') as f:
            lines = f.readlines()
        
        # Each line should have 2 tokens
        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.strip().split()
                assert len(parts) == 2

    def test_save_without_training(self, tmp_path):
        """Test that save raises error if not trained."""
        config = BBPETrainerConfig()
        trainer = BBPETrainer(config)
        
        with pytest.raises(ValueError, match="not been trained"):
            trainer.save(tmp_path / "model")

    def test_save_vocab_content(self, tmp_path):
        """Test vocabulary content is correct."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=2, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])
        
        output_dir = tmp_path / "model"
        trainer.save(output_dir)
        
        import json
        with open(output_dir / "vocab.json", encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Check base bytes are present (at least 256)
        assert len(vocab) >= 260
        
        # Check special tokens (they should be UTF-8 encoded)
        assert "[PAD]" in vocab
        assert "[UNK]" in vocab
        assert "[BOS]" in vocab
        assert "[EOS]" in vocab

    def test_save_merges_content(self, tmp_path):
        """Test merges content is correct."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])
        
        output_dir = tmp_path / "model"
        trainer.save(output_dir)
        
        with open(output_dir / "merges.txt", encoding='utf-8') as f:
            lines = f.readlines()
        
        # Should have some merges (simple.txt has repeated characters)
        non_empty_lines = [line for line in lines if line.strip()]
        assert len(non_empty_lines) > 0
        
        # Each line should have 1 or 2 tokens (2 for normal merges, 1 if one token is empty)
        # Use maxsplit=1 to handle tokens that may contain spaces
        valid_merges = 0
        for line in non_empty_lines:
            parts = line.strip().split(maxsplit=1)
            # Allow 1 or 2 parts (1 part means one token is empty/whitespace)
            assert len(parts) >= 1 and len(parts) <= 2, f"Expected 1-2 parts, got {len(parts)}: {parts}"
            if len(parts) == 2:
                valid_merges += 1
        
        # At least some merges should be valid (2 parts)
        assert valid_merges > 0, "No valid merges found"
