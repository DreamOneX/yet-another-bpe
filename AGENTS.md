# AGENTS.md

## Project Overview

`yet-another-bpe` is a Byte Pair Encoding (BPE) tokenizer implementation for Python 3.12+.

## Build Toolchain

- **Package Manager**: `uv` (recommended) or `pip`
- **Build Backend**: `hatchling`
- **Python Version**: >= 3.12

### Setup

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

## Code Style

### Docstrings

Use **Google Style** docstrings:

```python
def function(arg1: str, arg2: int) -> bool:
    """Short description of function.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.
    """
```

### Type Checking

Use **Pyright** (strict mode recommended):

```bash
# Run type checking
uv run pyright src/
```

- Avoid `Any` types where possible
- Use `typing.cast()` for explicit type narrowing
- Use specific `# pyright: ignore[errorCode]` instead of broad ignores

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_tokenizer.py

# Run with coverage
uv run pytest --cov=yet_another_bpe
```

### Test Structure

- `tests/test_tokenizer.py` - Tokenizer encode/decode tests
- `tests/test_trainer.py` - Trainer unit tests
- `tests/test_tokenizer_gpt2.py` - GPT-2 compatibility tests (uses tiktoken)
- `tests/test_train_bpe_gpt2.py` - Training compatibility tests

### Benchmarks

```bash
# Tokenizer benchmark
uv run python tests/benchmark_tokenizer.py

# Trainer benchmark
uv run python tests/benchmark_trainer.py
```

## Project Structure

```
src/yet_another_bpe/
├── __init__.py
├── tokenizer.py    # BBPETokenizer class
├── trainer.py      # BBPETrainer class
└── scripts/
    └── train_bpe.py

tests/
├── test_tokenizer.py
├── test_trainer.py
├── test_tokenizer_gpt2.py
├── test_train_bpe_gpt2.py
├── adapters.py           # Test adapters for external tests
├── common.py             # GPT-2 byte encoding utilities
├── fixtures_gpt2/        # GPT-2 test fixtures
└── benchmark_*.py        # Benchmark scripts
```

## Key Dependencies

- `regex` - Advanced regex for pre-tokenization
- `rich` - Console output formatting
- `tiktoken` (dev) - GPT-2 reference tokenizer for testing
- `pytest` (dev) - Testing framework
