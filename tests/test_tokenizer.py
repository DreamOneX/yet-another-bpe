"""Tests for BBPETokenizer implementation."""

import json
from pathlib import Path

import pytest

from yet_another_bpe.tokenizer import BBPETokenizer
from yet_another_bpe.trainer import BBPETrainer, BBPETrainerConfig


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


class TestTokenizerInit:
    """Tests for tokenizer initialization."""

    def test_empty_init(self):
        """Test initialization with no arguments."""
        tokenizer = BBPETokenizer()

        assert tokenizer.vocab_size == 0
        assert tokenizer.special_tokens == []

    def test_init_with_vocab(self):
        """Test initialization with vocabulary."""
        vocab = {b"a": 0, b"b": 1, b"ab": 2}
        tokenizer = BBPETokenizer(vocab=vocab)

        assert tokenizer.vocab_size == 3
        assert tokenizer.get_vocab() == {"a": 0, "b": 1, "ab": 2}

    def test_init_with_merges(self):
        """Test initialization with merges."""
        vocab = {b"a": 0, b"b": 1, b"ab": 2}
        merges = [(b"a", b"b")]
        tokenizer = BBPETokenizer(vocab=vocab, merges=merges)

        assert tokenizer.vocab_size == 3

    def test_init_with_special_tokens(self):
        """Test initialization with special tokens."""
        special_tokens = ["[PAD]", "[UNK]"]
        tokenizer = BBPETokenizer(special_tokens=special_tokens)

        assert tokenizer.special_tokens == special_tokens


class TestTokenizerFromFile:
    """Tests for loading tokenizer from files."""

    def test_load_from_trained_model(self, tmp_path):
        """Test loading tokenizer from a trained and saved model."""
        # Train a model first
        config = BBPETrainerConfig(vocab_size=270, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])

        # Save the model
        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        # Load tokenizer
        tokenizer = BBPETokenizer.from_file(model_dir)

        assert tokenizer.vocab_size >= 260
        assert isinstance(tokenizer.special_tokens, list)

    def test_load_vocab_content(self, tmp_path):
        """Test that loaded vocabulary matches saved vocabulary."""
        # Train and save
        config = BBPETrainerConfig(vocab_size=270, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        # Load tokenizer
        tokenizer = BBPETokenizer.from_file(model_dir)

        # Load vocab.json directly for comparison
        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            saved_vocab = json.load(f)

        assert tokenizer.get_vocab() == saved_vocab

    def test_load_special_tokens(self, tmp_path):
        """Test that special tokens are loaded correctly."""
        # Train with custom special tokens
        config = BBPETrainerConfig(
            vocab_size=270,
            min_frequency=1,
            max_workers=1,
            special_tokens=["<|endoftext|>", "<|pad|>"]
        )
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        # Load tokenizer
        tokenizer = BBPETokenizer.from_file(model_dir)

        assert "<|endoftext|>" in tokenizer.special_tokens
        assert "<|pad|>" in tokenizer.special_tokens

    def test_load_without_special_tokens_file(self, tmp_path):
        """Test loading when special_tokens.json doesn't exist."""
        # Create minimal model files without special_tokens.json
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create vocab.json
        vocab = {chr(i): i for i in range(256)}
        with open(model_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f)

        # Create empty merges.txt
        (model_dir / "merges.txt").write_text("")

        # Load tokenizer (should work with empty special tokens)
        tokenizer = BBPETokenizer.from_file(model_dir)

        assert tokenizer.special_tokens == []

    def test_load_nonexistent_directory(self):
        """Test that loading from nonexistent directory raises error."""
        with pytest.raises(FileNotFoundError):
            BBPETokenizer.from_file("/nonexistent/path")


class TestEncode:
    """Tests for encoding text to token IDs."""

    @pytest.fixture
    def trained_tokenizer(self, tmp_path):
        """Create a trained tokenizer for testing."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "multiline.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        return BBPETokenizer.from_file(model_dir)

    def test_encode_empty_string(self, trained_tokenizer):
        """Test encoding empty string."""
        ids = trained_tokenizer.encode("")
        assert ids == []

    def test_encode_simple_text(self, trained_tokenizer):
        """Test encoding simple ASCII text."""
        text = "hello"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_returns_valid_ids(self, trained_tokenizer):
        """Test that encoded IDs are within vocab range."""
        text = "Hello, world!"
        ids = trained_tokenizer.encode(text)

        vocab_size = trained_tokenizer.vocab_size
        for token_id in ids:
            assert 0 <= token_id < vocab_size

    def test_encode_unicode(self, trained_tokenizer):
        """Test encoding Unicode text."""
        text = "你好世界"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_with_spaces(self, trained_tokenizer):
        """Test encoding text with spaces."""
        text = "hello world"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_special_characters(self, trained_tokenizer):
        """Test encoding special characters."""
        text = "Hello! How are you? Fine, thanks."
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_multiline(self, trained_tokenizer):
        """Test encoding multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert len(ids) > 0


class TestDecode:
    """Tests for decoding token IDs to text."""

    @pytest.fixture
    def trained_tokenizer(self, tmp_path):
        """Create a trained tokenizer for testing."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "multiline.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        return BBPETokenizer.from_file(model_dir)

    def test_decode_empty_list(self, trained_tokenizer):
        """Test decoding empty list."""
        text = trained_tokenizer.decode([])
        assert text == ""

    def test_decode_single_token(self, trained_tokenizer):
        """Test decoding a single token."""
        # Encode and decode single character
        original = "a"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_decode_multiple_tokens(self, trained_tokenizer):
        """Test decoding multiple tokens."""
        original = "hello"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_decode_invalid_id(self, trained_tokenizer):
        """Test decoding with invalid token ID (should skip or handle gracefully)."""
        # Use very large ID that doesn't exist
        ids = [999999]
        # Should not raise, just skip unknown tokens
        decoded = trained_tokenizer.decode(ids)
        assert isinstance(decoded, str)


class TestRoundtrip:
    """Tests for encode-decode roundtrip consistency."""

    @pytest.fixture
    def trained_tokenizer(self, tmp_path):
        """Create a trained tokenizer for testing."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "multiline.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        return BBPETokenizer.from_file(model_dir)

    def test_roundtrip_simple_ascii(self, trained_tokenizer):
        """Test roundtrip for simple ASCII text."""
        original = "Hello, world!"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_roundtrip_unicode(self, trained_tokenizer):
        """Test roundtrip for Unicode text."""
        original = "你好世界 Hello 안녕하세요"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_roundtrip_numbers(self, trained_tokenizer):
        """Test roundtrip for text with numbers."""
        original = "The answer is 42."
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_roundtrip_punctuation(self, trained_tokenizer):
        """Test roundtrip for text with punctuation."""
        original = "Hello! How are you? I'm fine, thanks."
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_roundtrip_multiline(self, trained_tokenizer):
        """Test roundtrip for multiline text."""
        original = "Line 1\nLine 2\nLine 3"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_roundtrip_whitespace(self, trained_tokenizer):
        """Test roundtrip for various whitespace."""
        original = "Hello   world\t\ttabs"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original

    def test_roundtrip_empty_string(self, trained_tokenizer):
        """Test roundtrip for empty string."""
        original = ""
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == original


class TestBatchProcessing:
    """Tests for batch encode/decode methods."""

    @pytest.fixture
    def trained_tokenizer(self, tmp_path):
        """Create a trained tokenizer for testing."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "multiline.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        return BBPETokenizer.from_file(model_dir)

    def test_encode_batch_empty(self, trained_tokenizer):
        """Test batch encoding empty list."""
        result = trained_tokenizer.encode_batch([])
        assert result == []

    def test_encode_batch_single(self, trained_tokenizer):
        """Test batch encoding single text."""
        texts = ["hello"]
        result = trained_tokenizer.encode_batch(texts)

        assert len(result) == 1
        assert result[0] == trained_tokenizer.encode("hello")

    def test_encode_batch_multiple(self, trained_tokenizer):
        """Test batch encoding multiple texts."""
        texts = ["hello", "world", "test"]
        result = trained_tokenizer.encode_batch(texts)

        assert len(result) == 3
        for i, text in enumerate(texts):
            assert result[i] == trained_tokenizer.encode(text)

    def test_decode_batch_empty(self, trained_tokenizer):
        """Test batch decoding empty list."""
        result = trained_tokenizer.decode_batch([])
        assert result == []

    def test_decode_batch_single(self, trained_tokenizer):
        """Test batch decoding single sequence."""
        ids = trained_tokenizer.encode("hello")
        result = trained_tokenizer.decode_batch([ids])

        assert len(result) == 1
        assert result[0] == "hello"

    def test_decode_batch_multiple(self, trained_tokenizer):
        """Test batch decoding multiple sequences."""
        texts = ["hello", "world", "test"]
        ids_batch = trained_tokenizer.encode_batch(texts)
        result = trained_tokenizer.decode_batch(ids_batch)

        assert result == texts

    def test_batch_roundtrip(self, trained_tokenizer):
        """Test batch encode-decode roundtrip."""
        original = ["Hello!", "World!", "Test 123"]
        ids_batch = trained_tokenizer.encode_batch(original)
        decoded = trained_tokenizer.decode_batch(ids_batch)

        assert decoded == original


class TestSpecialTokens:
    """Tests for special token handling."""

    def test_special_token_in_vocab(self, tmp_path):
        """Test that special tokens are properly in vocabulary."""
        config = BBPETrainerConfig(
            vocab_size=270,
            min_frequency=1,
            max_workers=1,
            special_tokens=["<|endoftext|>"]
        )
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        tokenizer = BBPETokenizer.from_file(model_dir)
        vocab = tokenizer.get_vocab()

        assert "<|endoftext|>" in vocab

    def test_encode_special_token(self, tmp_path):
        """Test encoding text containing special token."""
        config = BBPETrainerConfig(
            vocab_size=270,
            min_frequency=1,
            max_workers=1,
            special_tokens=["<|endoftext|>"]
        )
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        tokenizer = BBPETokenizer.from_file(model_dir)

        # Encode text with special token
        text = "Hello<|endoftext|>World"
        ids = tokenizer.encode(text)

        # Should contain the special token ID
        vocab = tokenizer.get_vocab()
        special_token_id = vocab["<|endoftext|>"]
        assert special_token_id in ids

    def test_decode_special_token(self, tmp_path):
        """Test decoding special token."""
        config = BBPETrainerConfig(
            vocab_size=270,
            min_frequency=1,
            max_workers=1,
            special_tokens=["<|endoftext|>"]
        )
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "simple.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        tokenizer = BBPETokenizer.from_file(model_dir)

        # Roundtrip with special token
        original = "Hello<|endoftext|>World"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)

        assert decoded == original


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def trained_tokenizer(self, tmp_path):
        """Create a trained tokenizer for testing."""
        config = BBPETrainerConfig(vocab_size=300, min_frequency=1, max_workers=1)
        trainer = BBPETrainer(config)
        trainer.train([TEST_DATA_DIR / "multiline.txt"])

        model_dir = tmp_path / "model"
        trainer.save(model_dir)

        return BBPETokenizer.from_file(model_dir)

    def test_very_long_text(self, trained_tokenizer):
        """Test encoding very long text."""
        text = "Hello world! " * 1000
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == text

    def test_single_character(self, trained_tokenizer):
        """Test encoding single character."""
        for char in "abcXYZ123!@#":
            ids = trained_tokenizer.encode(char)
            decoded = trained_tokenizer.decode(ids)
            assert decoded == char

    def test_only_whitespace(self, trained_tokenizer):
        """Test encoding only whitespace."""
        text = "   \t\t\n\n   "
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == text

    def test_repeated_characters(self, trained_tokenizer):
        """Test encoding repeated characters."""
        text = "aaaaaaaaaa"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == text

    def test_mixed_scripts(self, trained_tokenizer):
        """Test encoding mixed scripts."""
        text = "Hello 你好 مرحبا Привет"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == text

    def test_emoji(self, trained_tokenizer):
        """Test encoding emoji."""
        text = "Hello 👋 World 🌍"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)

        assert decoded == text
