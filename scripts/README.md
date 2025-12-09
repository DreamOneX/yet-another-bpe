# Training Scripts

This directory contains scripts for training BPE models.

## train_bpe.py

Trains a BPE (Byte Pair Encoding) model using the TinyStoriesV2-GPT4-valid.txt dataset.

### Usage

```bash
# From the project root directory
python scripts/train_bpe.py
```

### Configuration

The script uses the following configuration:
- **Vocabulary size**: 8000 tokens
- **Minimum frequency**: 2 (pairs must appear at least twice to be merged)
- **Max workers**: 4 (parallel processing threads)
- **Chunk size**: 8MB
- **Special tokens**: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`

### Output

The trained model will be saved to `models/tinystories_bpe/` with the following files:
- `vocab.json` - Token to index mapping
- `merges.txt` - Merge rules in application order

### Customization

To train with different parameters, modify the `BBPETrainerConfig` in the script:

```python
config = BBPETrainerConfig(
    vocab_size=16000,      # Change vocabulary size
    min_frequency=5,       # Require higher frequency
    max_workers=8,         # Use more workers
    # ... other parameters
)
```

To train on a different dataset, change the `data_file` path:

```python
data_file = project_root / "path" / "to" / "your" / "data.txt"
```
