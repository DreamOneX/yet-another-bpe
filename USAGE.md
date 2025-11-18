# Usage Guide

## Using uv (Recommended)

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Install the package in editable mode
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

### 3. Run examples

```bash
python examples/example_usage.py
```

### 4. Run tests

```bash
python tests/test_byte_bpe.py
```

## Quick Example

```python
from yet_another_bpe.trainer import adapters, ByteLevelBPETrainer

# Example 1: Using the adapters function
vocab, merges = adapters(
    input_path="your_data.txt",
    vocab_size=500,
    special_tokens=['<|endoftext|>']
)

# Example 2: Direct training
trainer = ByteLevelBPETrainer(special_tokens=['<|endoftext|>'])
text = "Hello world! This is a test."
trainer.train(text, num_merges=50, verbose=True)

vocab = trainer.get_vocab()
merges = trainer.get_merges()
```

## Development Workflow

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Run tests with coverage
pytest --cov=src/yet_another_bpe --cov-report=html
```

## Building and Publishing

```bash
# Build the package
uv build

# The built files will be in dist/
```
