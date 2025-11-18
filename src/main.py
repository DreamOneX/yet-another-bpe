#!/usr/bin/env python3
"""
Main demo for the BBPE (Byte-level BPE) Trainer
Demonstrates the usage of BBPETrainer with rich terminal output
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.progress import track

from bbpe import BBPETrainer, BBPETrainerConfig

console = Console()


def print_header():
    """Print a nice header."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]Byte-Level BPE (BBPE) Trainer Demo[/bold blue]\n"
        "[dim]A demonstrative implementation with rich visualization[/dim]",
        border_style="blue",
        box=box.DOUBLE
    ))
    console.print()


def print_config(config: BBPETrainerConfig):
    """Display the training configuration."""
    console.print("[bold cyan]Training Configuration:[/bold cyan]")

    config_table = Table(box=box.ROUNDED, show_header=False)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow bold")

    config_table.add_row("Target vocabulary size", str(config.vocab_size))
    config_table.add_row("Min frequency", str(config.min_frequency))
    config_table.add_row("Max workers", str(config.max_workers))
    config_table.add_row("Chunk size", f"{config.chunk_size_bytes // (1024*1024)} MB")
    config_table.add_row("Random seed", str(config.seed))

    console.print(config_table)
    console.print()


def print_special_tokens(special_tokens):
    """Display special tokens."""
    if special_tokens:
        console.print("[bold cyan]Special Tokens:[/bold cyan]")
        for token in special_tokens:
            console.print(f"  [green]•[/green] {token}")
        console.print()


def demo_simple_text():
    """Demo 1: Train on a simple text string."""
    console.print(Panel.fit(
        "[bold magenta]Demo 1: Training on Simple Text[/bold magenta]",
        border_style="magenta"
    ))
    console.print()

    # Sample text
    sample_text = """
Hello world! This is a byte-level BPE trainer. <|endoftext|>
Natural language processing is fascinating. <|endoftext|>
BPE stands for Byte Pair Encoding, a subword tokenization algorithm. <|endoftext|>
This trainer implements the three-stage architecture for scalability. <|endoftext|>
It uses memory mapping for I/O efficiency, MapReduce for counting, and a max-heap for merges. <|endoftext|>
""" * 3  # Repeat for more training data

    console.print("[dim]Sample text (first 200 chars):[/dim]")
    console.print(f"[italic]{sample_text[:200]}...[/italic]\n")

    # Create trainer config
    config = BBPETrainerConfig(
        vocab_size=400,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"]
    )

    print_config(config)
    print_special_tokens(config.special_tokens)

    # Train
    console.print("[bold green]Starting training...[/bold green]\n")

    trainer = BBPETrainer(config)
    num_merges = config.vocab_size - 256 - len(config.special_tokens)

    model = trainer.train_from_text(sample_text, num_merges)

    console.print("[bold green] Training completed![/bold green]\n")

    # Display results
    display_results(model)


def demo_from_file():
    """Demo 2: Train from a file."""
    console.print(Panel.fit(
        "[bold magenta]Demo 2: Training from File[/bold magenta]",
        border_style="magenta"
    ))
    console.print()

    # Check for training corpus file
    corpus_file = Path(__file__).parent.parent / "training_corpus.txt"

    if not corpus_file.exists():
        console.print(f"[yellow]Training corpus not found at: {corpus_file}[/yellow]")
        console.print("[dim]Creating a sample corpus file...[/dim]\n")

        # Create a sample corpus
        sample_corpus = """
The quick brown fox jumps over the lazy dog. <|endoftext|>
Machine learning is a subset of artificial intelligence. <|endoftext|>
Python is a popular programming language for data science. <|endoftext|>
Tokenization is the process of splitting text into smaller units called tokens. <|endoftext|>
BPE (Byte Pair Encoding) is an efficient tokenization algorithm used in modern NLP. <|endoftext|>
Large language models like GPT use BPE for tokenization. <|endoftext|>
This implementation uses a three-stage architecture for better performance. <|endoftext|>
Memory mapping helps reduce memory usage when processing large files. <|endoftext|>
""" * 20  # Repeat to have enough data

        corpus_file.write_text(sample_corpus, encoding='utf-8')
        console.print(f"[green] Created sample corpus at: {corpus_file}[/green]\n")

    console.print(f"[cyan]Training corpus:[/cyan] {corpus_file}")
    console.print(f"[dim]File size: {corpus_file.stat().st_size:,} bytes[/dim]\n")

    # Create trainer config
    config = BBPETrainerConfig(
        vocab_size=500,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|startoftext|>", "<|pad|>"]
    )

    print_config(config)
    print_special_tokens(config.special_tokens)

    # Train
    console.print("[bold green]Starting training...[/bold green]\n")

    trainer = BBPETrainer(config)
    model = trainer.train([corpus_file])

    console.print("[bold green] Training completed![/bold green]\n")

    # Display results
    display_results(model)

    # Save model
    output_dir = Path(__file__).parent.parent / "output"
    console.print(f"[cyan]Saving model to:[/cyan] {output_dir}\n")

    trainer.save(output_dir)

    console.print("[green]Model saved successfully![/green]")
    console.print("  [dim]• vocab.json[/dim]")
    console.print("  [dim]• merges.txt[/dim]\n")


def display_results(model):
    """Display training results in a nice format."""
    vocab = model.get_vocab()
    merges = model.get_merges()

    # Summary statistics
    console.print("[bold cyan]Training Results:[/bold cyan]")

    results_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="yellow", justify="right")
    results_table.add_column("Description", style="dim")

    results_table.add_row("Total vocabulary size", str(len(vocab)), "All tokens")
    results_table.add_row("Base bytes", "256", "Single bytes (0-255)")
    results_table.add_row("Special tokens", str(len(model.special_tokens)), "Reserved tokens")
    results_table.add_row("Merged tokens", str(len(merges)), "Learned combinations")

    console.print(results_table)
    console.print()

    # Show sample merges
    if merges:
        console.print("[bold cyan]Sample Merge Operations:[/bold cyan]")
        console.print("[dim]Showing first 10 and last 5 merges[/dim]\n")

        merge_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        merge_table.add_column("#", style="cyan", justify="center", width=5)
        merge_table.add_column("Token 1", style="green", justify="center")
        merge_table.add_column("Token 2", style="green", justify="center")
        merge_table.add_column("Result", style="yellow bold", justify="center")
        merge_table.add_column("Length", style="dim", justify="center")

        # Show first 10 merges
        sample_indices = list(range(min(10, len(merges)))) + \
                        (list(range(max(10, len(merges) - 5), len(merges))) if len(merges) > 15 else [])

        shown = set()
        for i in sample_indices:
            if i in shown:
                continue
            shown.add(i)

            if i == 10 and len(merges) > 15:
                merge_table.add_row("...", "...", "...", "...", "...")

            token1, token2 = merges[i]
            merged = token1 + token2

            try:
                t1_display = token1.decode('utf-8', errors='backslashreplace')
                t2_display = token2.decode('utf-8', errors='backslashreplace')
                merged_display = merged.decode('utf-8', errors='backslashreplace')
            except Exception:
                t1_display = token1.hex()
                t2_display = token2.hex()
                merged_display = merged.hex()

            merge_table.add_row(
                str(i + 1),
                f"'{t1_display}'",
                f"'{t2_display}'",
                f"'{merged_display}'",
                f"{len(merged)} bytes"
            )

        console.print(merge_table)
        console.print()

    # Show special tokens in vocabulary
    console.print("[bold cyan]Special Tokens in Vocabulary:[/bold cyan]")

    for special_token in model.special_tokens:
        token_bytes = special_token.encode('utf-8')
        found_id = None
        for token_id, vocab_bytes in vocab.items():
            if vocab_bytes == token_bytes:
                found_id = token_id
                break

        if found_id is not None:
            console.print(f"  [green][/green] {special_token:<20} � Token ID {found_id}")
        else:
            console.print(f"  [red][/red] {special_token:<20} � [red]Not found[/red]")

    console.print()


def main():
    """Main entry point."""
    print_header()

    # Run Demo 1
    demo_simple_text()

    console.print("\n" + "="*80 + "\n")

    # Run Demo 2
    demo_from_file()

    # Final message
    console.print()
    console.print(Panel.fit(
        "[bold green]All demos completed successfully![/bold green]\n"
        "[dim]You can now use the trained model for tokenization[/dim]",
        border_style="green",
        box=box.DOUBLE
    ))
    console.print()


if __name__ == "__main__":
    main()
