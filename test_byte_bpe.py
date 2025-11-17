"""
Test script for byte-level BPE with special tokens
"""

from trainer import adapters, ByteLevelBPETrainer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def test_byte_level_bpe():
    """Test byte-level BPE trainer with special tokens"""
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]Byte-Level BPE Test[/bold blue]\n[dim]with Special Token Support[/dim]",
        border_style="blue",
        box=box.DOUBLE
    ))
    console.print()
    
    # Sample text with special tokens
    text = """Hello world! This is a test of byte-level BPE.<|endoftext|>
Another document starts here. It's working great!<|endoftext|>
Third document with some repeated words: hello hello world world.<|endoftext|>
"""
    
    console.print(Panel(
        f"[italic]{text}[/italic]",
        title="[bold cyan]Training Text[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    
    # Define special tokens
    special_tokens = ['<|endoftext|>']
    
    # Create trainer with special tokens
    trainer = ByteLevelBPETrainer(special_tokens=special_tokens)
    
    # Train
    console.print(Panel.fit(
        f"[cyan]Training with special tokens:[/cyan] [green]{special_tokens}[/green]",
        border_style="cyan"
    ))
    console.print()
    
    trainer.train(text, num_merges=50, verbose=True)
    
    # Show results
    console.print()
    console.print(Panel.fit("[bold blue]Vocabulary Analysis[/bold blue]", border_style="blue"))
    console.print()
    
    vocab = trainer.get_vocab()
    merges = trainer.get_merges()
    
    # Show special tokens in vocabulary
    console.print("[bold cyan]Special tokens in vocabulary:[/bold cyan]")
    special_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    special_table.add_column("Token ID", style="cyan", justify="center")
    special_table.add_column("Token (bytes)", style="green")
    special_table.add_column("Token (decoded)", style="yellow")
    
    for token_id, token_bytes in vocab.items():
        try:
            decoded = token_bytes.decode('utf-8')
            if decoded in special_tokens:
                special_table.add_row(str(token_id), repr(token_bytes), decoded)
        except:
            pass
    
    console.print(special_table)
    console.print()
    
    # Show some merged tokens
    console.print("[bold cyan]Sample of merged tokens (last 10):[/bold cyan]")
    merge_sample_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    merge_sample_table.add_column("#", style="cyan", justify="center", width=4)
    merge_sample_table.add_column("Merge", style="green")
    merge_sample_table.add_column("Result", style="yellow")
    
    for i, (token1, token2) in enumerate(merges[-10:], start=len(merges)-9):
        try:
            t1 = token1.decode('utf-8', errors='backslashreplace')
            t2 = token2.decode('utf-8', errors='backslashreplace')
            result = (token1 + token2).decode('utf-8', errors='backslashreplace')
        except:
            t1 = repr(token1)
            t2 = repr(token2)
            result = repr(token1 + token2)
        
        merge_sample_table.add_row(str(i), f"{t1} + {t2}", result)
    
    console.print(merge_sample_table)
    console.print()


def test_adapters_function():
    """Test the adapters function"""
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]Testing adapters() Function[/bold blue]",
        border_style="blue",
        box=box.DOUBLE
    ))
    console.print()
    
    # Create a test file
    test_file_path = "test_training_data.txt"
    
    test_text = """Hello world! This is a BPE trainer.<|endoftext|>
It supports byte-level encoding and special tokens.<|endoftext|>
The quick brown fox jumps over the lazy dog.<|endoftext|>
Testing, testing, 1, 2, 3!<|endoftext|>
Hello again! Repeated words help BPE learn better patterns.<|endoftext|>
""" * 5  # Repeat to have more training data
    
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    console.print(f"[green]Created test file: {test_file_path}[/green]\n")
    
    # Call adapters function
    special_tokens = ['<|endoftext|>', '<|startoftext|>']
    vocab_size = 300
    
    vocab, merges = adapters(
        input_path=test_file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    
    # Display results
    console.print()
    console.print(Panel.fit("[bold green]Results[/bold green]", border_style="green"))
    console.print()
    
    result_table = Table(box=box.ROUNDED, show_header=False)
    result_table.add_column("Metric", style="cyan bold")
    result_table.add_column("Value", style="yellow")
    result_table.add_row("Vocabulary size", str(len(vocab)))
    result_table.add_row("Number of merges", str(len(merges)))
    result_table.add_row("Base bytes", "256")
    result_table.add_row("Special tokens", str(len(special_tokens)))
    
    console.print(result_table)
    console.print()
    
    # Verify special tokens
    console.print("[bold cyan]Verifying special tokens in vocabulary:[/bold cyan]")
    found_special = []
    for token_id, token_bytes in vocab.items():
        try:
            decoded = token_bytes.decode('utf-8')
            if decoded in special_tokens:
                found_special.append((token_id, decoded))
                console.print(f"  [green]OK[/green] Token ID {token_id}: [yellow]{decoded}[/yellow]")
        except:
            pass
    
    if len(found_special) == len(special_tokens):
        console.print(f"\n[bold green]Success! All {len(special_tokens)} special tokens found in vocabulary.[/bold green]")
    else:
        console.print(f"\n[bold red]Warning: Only {len(found_special)}/{len(special_tokens)} special tokens found.[/bold red]")
    
    console.print()


if __name__ == "__main__":
    # Test 1: Basic byte-level BPE
    test_byte_level_bpe()
    
    console.print("\n" + "="*80 + "\n")
    
    # Test 2: adapters function
    test_adapters_function()

