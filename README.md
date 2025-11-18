# Yet Another BPE

A byte-level BPE (Byte Pair Encoding) tokenizer trainer with special token support.

## Features

- 🔤 Byte-level tokenization for robust handling of any text
- 🎯 Special token support (e.g., `<|endoftext|>`, `<|pad|>`)
- 🎨 Beautiful CLI output with Rich library
- 📦 Modern Python packaging with PEP standards
- 🚀 Easy to use and extend

## Installation

Using uv (recommended):

```bash
uv pip install -e .
```

Using pip:

```bash
pip install -e .
```

## Quick Start

```python
from yet_another_bpe.trainer import adapters

# Train a BPE tokenizer
vocab, merges = adapters(
    input_path="your_training_data.txt",
    vocab_size=500,
    special_tokens=['<|endoftext|>', '<|pad|>']
)

print(f"Vocabulary size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")
```

## Development

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format and lint code:

```bash
ruff format .
ruff check .
```

## Project Structure

```
yet-another-bpe/
├── src/
│   └── yet_another_bpe/
│       ├── __init__.py
│       ├── bpe.py          # Byte-level BPE trainer
│       └── trainer.py      # Training utilities
├── examples/
│   └── example_usage.py    # Usage examples
├── tests/
│   └── test_byte_bpe.py    # Test cases
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```

## Requirements

- Python >= 3.12
- regex >= 2023.12.25
- rich >= 13.7.0

## License

MIT
