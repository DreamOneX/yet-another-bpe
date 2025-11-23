"""Tests for BPE implementation."""

from yet_another_bpe.bpe import BPE


def test_bpe_init():
    """Test BPE initialization."""
    bpe = BPE()
    assert bpe is not None
