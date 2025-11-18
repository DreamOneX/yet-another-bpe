"""
Example usage of the adapters function for training byte-level BPE
"""

from yet_another_bpe.trainer import adapters
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def main():
    """
    Demonstrate how to use the adapters function
    """
    console.print()
    console.print(Panel.fit(
        "[bold blue]Byte-Level BPE Training Example[/bold blue]\n"
        "[dim]Using the adapters() function[/dim]",
        border_style="blue",
        box=box.DOUBLE
    ))
    console.print()
    
    # Step 1: Prepare training data
    console.print("[bold cyan]Step 1:[/bold cyan] Create training data file\n")
    
    training_text = """The quick brown fox jumps over the lazy dog.<|endoftext|>
Machine learning is a subset of artificial intelligence.<|endoftext|>
Python is a popular programming language for data science.<|endoftext|>
Natural language processing enables computers to understand human language.<|endoftext|>
Deep learning uses neural networks with multiple layers.<|endoftext|>
""" * 10  # Repeat to have enough training data
    
    input_file = "training_corpus.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(training_text)
    
    console.print(f"[green]Created training file:[/green] {input_file}")
    console.print(f"[dim]File size: {len(training_text)} characters[/dim]\n")
    
    # Step 2: Define parameters
    console.print("[bold cyan]Step 2:[/bold cyan] Define training parameters\n")
    
    vocab_size = 500
    special_tokens = ['<|endoftext|>', '<|startoftext|>', '<|pad|>']
    
    params_table = Table(box=box.ROUNDED, show_header=False)
    params_table.add_column("Parameter", style="cyan bold")
    params_table.add_column("Value", style="yellow")
    params_table.add_row("Input file", input_file)
    params_table.add_row("Target vocabulary size", str(vocab_size))
    params_table.add_row("Special tokens", ", ".join(special_tokens))
    
    console.print(params_table)
    console.print()
    
    # Step 3: Train the tokenizer
    console.print("[bold cyan]Step 3:[/bold cyan] Train byte-level BPE tokenizer\n")
    
    vocab, merges = adapters(
        input_path=input_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    # Step 4: Analyze results
    console.print()
    console.print("[bold cyan]Step 4:[/bold cyan] Analyze the results\n")
    
    # Count different token types
    base_bytes = sum(1 for token_id, token_bytes in vocab.items() if len(token_bytes) == 1 and token_id < 256)
    special = sum(1 for token_bytes in vocab.values() if token_bytes.decode('utf-8', errors='ignore') in special_tokens)
    merged = len(vocab) - 256 - len(special_tokens)
    
    results_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    results_table.add_column("Category", style="cyan")
    results_table.add_column("Count", style="yellow", justify="right")
    results_table.add_column("Description", style="dim")
    
    results_table.add_row("Base bytes", "256", "All single bytes (0-255)")
    results_table.add_row("Special tokens", str(len(special_tokens)), "Reserved special tokens")
    results_table.add_row("Merged tokens", str(len(merges)), "Learned byte combinations")
    results_table.add_row("Total vocabulary", str(len(vocab)), "Final vocabulary size")
    
    console.print(results_table)
    console.print()
    
    # Show some interesting merged tokens
    console.print("[bold cyan]Sample merged tokens:[/bold cyan]\n")
    
    sample_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    sample_table.add_column("Merge #", style="cyan", justify="center", width=8)
    sample_table.add_column("Merged Token", style="green")
    sample_table.add_column("Bytes", style="dim")
    
    # Show first 5 and last 5 merges
    sample_indices = list(range(0, min(5, len(merges)))) + list(range(max(0, len(merges)-5), len(merges)))
    shown = set()
    
    for i in sample_indices:
        if i not in shown:
            shown.add(i)
            token1, token2 = merges[i]
            merged = token1 + token2
            try:
                display = merged.decode('utf-8', errors='backslashreplace')
            except:
                display = str(merged)
            
            sample_table.add_row(
                str(i + 1),
                display,
                f"{len(merged)} bytes"
            )
    
    console.print(sample_table)
    console.print()
    
    # Step 5: Show how to use the results
    console.print("[bold cyan]Step 5:[/bold cyan] Using the results\n")
    
    usage_text = """
[yellow]vocab[/yellow] - Dictionary mapping token IDs to bytes
  Example: vocab[0] = b'\\x00', vocab[256] = b'<|endoftext|>'

[yellow]merges[/yellow] - List of merge rules in order
  Example: [(b' ', b't'), (b'e', b'r'), ...]

These can be used to:
  - Build an encoder/decoder for tokenization
  - Save to disk for later use
  - Convert to other tokenizer formats
"""
    
    console.print(Panel(usage_text.strip(), border_style="green", title="[bold]Output Format[/bold]"))
    console.print()
    
    # Verify special tokens
    console.print("[bold green]Verification:[/bold green] Special tokens in vocabulary\n")
    
    for token_str in special_tokens:
        token_bytes = token_str.encode('utf-8')
        found = False
        for token_id, vocab_bytes in vocab.items():
            if vocab_bytes == token_bytes:
                console.print(f"  [green]OK[/green] {token_str} -> Token ID {token_id}")
                found = True
                break
        if not found:
            console.print(f"  [red]MISSING[/red] {token_str}")
    
    console.print()
    console.print(Panel.fit(
        "[bold green]Training Complete![/bold green]\n"
        f"[cyan]Vocabulary size:[/cyan] [yellow]{len(vocab)}[/yellow]\n"
        f"[cyan]Merge rules:[/cyan] [yellow]{len(merges)}[/yellow]",
        border_style="green"
    ))
    console.print()


if __name__ == "__main__":
    main()

