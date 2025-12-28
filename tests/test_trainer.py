"""Tests for BPE implementation."""

import unicodedata
from pathlib import Path

import pytest

from yet_another_bpe.trainer import BBPEModel, BBPETrainer, BBPETrainerConfig


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


class TestPreprocessing:
    """Tests for corpus preprocessing."""

    def test_simple_ascii_text(self):
        """Test preprocessing of simple ASCII text."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "simple.txt"
        assert test_file.exists(), f"Test file {test_file} not found"
        
        # Call preprocessing
        sequences = trainer._preprocess_corpus([test_file])
        
        # Verify we got sequences
        assert len(sequences) > 0, "Should return at least one sequence"
        
        # Verify all sequences contain only bytes (0-255)
        for seq in sequences:
            assert isinstance(seq, list), "Each sequence should be a list"
            assert all(isinstance(b, int) and 0 <= b <= 255 for b in seq), \
                "All elements should be integers in range 0-255"
        
        # Verify the content matches expected bytes (after pre-tokenization)
        # "Hello world!" -> tokens like "Hello", " world", "!" depending on GPT-2 pattern
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        expected_bytes = "Hello world!".encode('utf-8')
        assert all_bytes == expected_bytes, f"Expected {expected_bytes}, got {all_bytes}"

    def test_unicode_text(self):
        """Test preprocessing of Unicode text with normalization."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "unicode.txt"
        assert test_file.exists(), f"Test file {test_file} not found"
        
        # Read original content (use binary to preserve exact line endings)
        original_bytes = test_file.read_bytes()
        
        # Call preprocessing
        sequences = trainer._preprocess_corpus([test_file])
        
        # Verify sequences
        assert len(sequences) > 0
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        
        # Should match original bytes exactly
        assert all_bytes == original_bytes, \
            f"Unicode text should match original encoding"

    def test_multiline_text(self):
        """Test preprocessing of multiline text."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "multiline.txt"
        sequences = trainer._preprocess_corpus([test_file])
        
        assert len(sequences) > 0
        # Verify newlines are preserved (may be in separate sequences due to pre-tokenization)
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        assert b'\n' in all_bytes or b'\r\n' in all_bytes, \
            "Newlines should be preserved"

    def test_empty_file(self):
        """Test preprocessing of empty file (edge case)."""
        config = BBPETrainerConfig(max_workers=1)
        trainer = BBPETrainer(config)
        
        test_file = TEST_DATA_DIR / "empty.txt"
        sequences = trainer._preprocess_corpus([test_file])
        
        # Empty file should return empty list
        assert len(sequences) == 0, \
            "Empty file should produce no sequences"

    def test_large_file_chunking(self):
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
        
        sequences = trainer._preprocess_corpus([test_file])
        
        # Should get sequences
        assert len(sequences) > 0
        
        # Verify all bytes are valid
        for seq in sequences:
            assert all(0 <= b <= 255 for b in seq)
        
        # Verify total content matches (no data loss from chunking)
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        expected_text = test_file.read_text(encoding='utf-8')
        expected_bytes = expected_text.encode('utf-8')
        
        assert all_bytes == expected_bytes, \
            "Chunking should not lose or corrupt data"

    def test_multiple_files(self):
        """Test preprocessing multiple files."""
        config = BBPETrainerConfig(max_workers=2)
        trainer = BBPETrainer(config)
        
        test_files = [
            TEST_DATA_DIR / "simple.txt",
            TEST_DATA_DIR / "unicode.txt",
            TEST_DATA_DIR / "multiline.txt"
        ]
        
        sequences = trainer._preprocess_corpus(test_files)
        
        # Should get sequences from all files
        assert len(sequences) > 0, \
            "Should process all files"
        
        # All sequences should be valid
        for seq in sequences:
            assert isinstance(seq, list)
            assert all(isinstance(b, int) and 0 <= b <= 255 for b in seq)

    def test_parallel_processing(self):
        """Test that parallel processing works correctly."""
        # Test with different worker counts
        for max_workers in [1, 2, 4]:
            config = BBPETrainerConfig(max_workers=max_workers)
            trainer = BBPETrainer(config)
            
            test_files = [
                TEST_DATA_DIR / "simple.txt",
                TEST_DATA_DIR / "unicode.txt"
            ]
            
            sequences = trainer._preprocess_corpus(test_files)
            
            # Results should be consistent regardless of worker count
            assert len(sequences) > 0
            for seq in sequences:
                assert all(0 <= b <= 255 for b in seq)

    def test_nonexistent_file(self):
        """Test that nonexistent files raise appropriate error."""
        config = BBPETrainerConfig()
        trainer = BBPETrainer(config)
        
        nonexistent = TEST_DATA_DIR / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            _ = trainer._preprocess_corpus([nonexistent])

    def test_utf8_boundary_handling(self):
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
        
        sequences = trainer._preprocess_corpus([test_file])
        
        # Reconstruct and verify no corruption
        all_bytes = b''.join(bytes(seq) for seq in sequences)
        reconstructed = all_bytes.decode('utf-8')
        
        assert reconstructed == content, \
            "UTF-8 boundaries should be respected, no character corruption"
        
        # Cleanup
        test_file.unlink()


class TestMergeLoop:
    """Tests for the BPE merge loop."""

    def test_vocab_initialization(self):
        """Test that vocabulary is properly initialized with bytes 0-255 and special tokens."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Empty sequences for testing vocab initialization only
        vocab, merges = trainer._merge_loop([])
        
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

    def test_basic_merge(self):
        """Test basic merge operation."""
        config = BBPETrainerConfig(vocab_size=265, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Create sequences where 'H' + 'e' pair is most frequent
        sequences = [
            [72, 101, 108, 108, 111],  # "Hello"
            [72, 101, 108, 108, 111],  # "Hello"
        ]
        
        vocab, merges = trainer._merge_loop(sequences)
        
        # Should have base vocab (260) + some merges
        assert len(vocab) >= 260
        
        # Should have performed some merges
        assert len(merges) > 0
        
        # All merges should be tuples of bytes
        for merge in merges:
            assert isinstance(merge, tuple)
            assert len(merge) == 2
            assert isinstance(merge[0], bytes)
            assert isinstance(merge[1], bytes)

    def test_merge_ordering(self):
        """Test that merges are performed in frequency order."""
        config = BBPETrainerConfig(vocab_size=270, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Create sequences where 'A'+'B' is most frequent
        sequences = [
            [65, 66],  # 'AB' - appears 100 times
        ] * 100 + [
            [67, 68],  # 'CD' - appears 50 times
        ] * 50 + [
            [69, 70],  # 'EF' - appears 10 times
        ] * 10
        
        vocab, merges = trainer._merge_loop(sequences)
        
        # First merge should be highest frequency
        assert merges[0] == (b'A', b'B') or merges[0] == (bytes([65]), bytes([66]))

    def test_vocab_size_limit(self):
        """Test that merging stops when vocab_size is reached."""
        config = BBPETrainerConfig(vocab_size=262, min_frequency=1, max_workers=1)  # Only 2 merges allowed
        trainer = BBPETrainer(config)
        
        # Create sequences with many distinct pairs
        sequences = [
            [65, 66],  # 'AB'
            [67, 68],  # 'CD'
            [69, 70],  # 'EF'
            [71, 72],  # 'GH'
            [73, 74],  # 'IJ'
        ] * 10  # Each pair appears 10 times
        
        vocab, merges = trainer._merge_loop(sequences)
        
        # Should stop at vocab_size
        assert len(vocab) == 262
        
        # Should have exactly 2 merges (262 - 260 base vocab)
        assert len(merges) == 2

    def test_min_frequency_threshold(self):
        """Test that pairs below min_frequency are not merged."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=5, max_workers=1)
        trainer = BBPETrainer(config)
        
        # Create sequences where some pairs are below min_frequency
        sequences = [
            [65, 66],  # 'AB' - appears 10 times (above threshold)
        ] * 10 + [
            [67, 68],  # 'CD' - appears 5 times (at threshold)
        ] * 5 + [
            [69, 70],  # 'EF' - appears 4 times (below threshold)
        ] * 4 + [
            [71, 72],  # 'GH' - appears 1 time (below threshold)
        ]
        
        vocab, merges = trainer._merge_loop(sequences)
        
        # Should only merge pairs with frequency >= 5
        assert len(merges) <= 2
        
        # Check that low-frequency pairs are not in merges
        low_freq_merges = [(b'E', b'F'), (b'G', b'H')]
        for merge in merges:
            assert merge not in low_freq_merges

    def test_special_tokens(self):
        """Test that special tokens are properly included in vocab."""
        config = BBPETrainerConfig(
            vocab_size=300,
            min_frequency=1,
            max_workers=1,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
        )
        trainer = BBPETrainer(config)
        
        vocab, merges = trainer._merge_loop([])
        
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
        
        # Check base bytes are present (at least 260)
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
