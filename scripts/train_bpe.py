"""Script to train and save a BPE model using TinyStoriesV2-GPT4-valid.txt dataset."""

from pathlib import Path

from yet_another_bpe.bbpe import BBPETrainer, BBPETrainerConfig

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

def main():
    """Train a BPE model on the TinyStories validation dataset."""
    # Define paths
    project_root = Path(__file__).parent.parent
    data_file = project_root / "tests" / "data" / "TinyStoriesV2-GPT4-valid.txt"
    # data_file = project_root / "tests" / "data" / "sample.txt"
    output_dir = project_root / "models" / "tinystories_bpe"

    # Verify data file exists
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Training BPE model on: {data_file}")
    print(f"Output directory: {output_dir}")

    # Create trainer configuration
    config = BBPETrainerConfig(
        vocab_size=5000,  # Smaller vocab for this dataset
        min_frequency=2,
        max_workers=8,
        chunk_size_bytes=20 * 1024 * 1024,  # 8MB chunks
        seed=42,
        special_tokens=[ "<|endoftext|>" ],
    )
    
    console = Console()
    trainer = BBPETrainer(config=config)

    # Train the model
    console.print("\n[bold cyan]Starting training...[/bold cyan]")
    model = trainer.train(files=[data_file])
    console.print("[bold green]✓ Training complete![/bold green]")

    # Save the model
    console.print(f"\n[bold yellow]Saving model to:[/bold yellow] [dim]{output_dir}[/dim]")
    trainer.save(output_dir=output_dir)
    console.print("[bold green]✓ Model saved successfully![/bold green]")

    # Print summary
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_row("[cyan]Vocabulary size:[/cyan]", f"[bold]{len(model.vocab)}[/bold]")
    summary_table.add_row("[cyan]Number of merges:[/cyan]", f"[bold]{len(model.merges)}[/bold]")
    summary_table.add_row("[cyan]Special tokens:[/cyan]", f"[bold]{model.special_tokens}[/bold]")
    
    console.print(
        Panel(
            summary_table,
            title="[bold green]Training Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    main()
