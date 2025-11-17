"""
Simple BPE (Byte Pair Encoding) Trainer
Implemented using a naive brute-force approach for easy understanding
"""

import regex
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box
from rich.text import Text

console = Console()


class ByteLevelBPETrainer:
    """
    Byte-level BPE trainer with special token support
    """
    def __init__(self, special_tokens=None, pretokenize_pattern=None):
        self.merges = []  # List of (bytes, bytes) merge operations
        self.vocab = {}  # Dict[int, bytes] - vocabulary mapping index to bytes
        self.special_tokens = special_tokens or []
        self.pretokenize_pattern = pretokenize_pattern or r"(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self.compiled_pattern = regex.compile(self.pretokenize_pattern)
        
        # Build special token pattern to protect them from being split
        if self.special_tokens:
            # Escape special regex characters in tokens
            escaped_tokens = [regex.escape(token) for token in self.special_tokens]
            special_pattern = '|'.join(escaped_tokens)
            # Combine special token pattern with regular pattern (no capturing group)
            self.pretokenize_pattern = f"{special_pattern}|{self.pretokenize_pattern}"
            self.compiled_pattern = regex.compile(self.pretokenize_pattern)
    
    def _pretokenize(self, text):
        """
        Pretokenize text using regex pattern, preserving special tokens
        Returns a list of pretokens (words/chunks)
        """
        return self.compiled_pattern.findall(text)
    
    def train(self, text, num_merges, verbose=True):
        """
        Train the BPE model on byte level
        
        Args:
            text: Training text (str or bytes)
            num_merges: Number of merges to perform
            verbose: Whether to print progress
        """
        if isinstance(text, str):
            text = text.encode('utf-8')
        
        # 1. Pretokenize text and convert to bytes
        if self.special_tokens:
            # Handle text as string to preserve special tokens
            text_str = text.decode('utf-8') if isinstance(text, bytes) else text
            pretokens = self._pretokenize(text_str)
            
            # Convert pretokens to bytes, but keep special tokens as-is
            word_tokens = []
            for token in pretokens:
                if token in self.special_tokens:
                    # Special tokens are treated as single indivisible units
                    word_tokens.append([token.encode('utf-8')])
                else:
                    # Regular tokens are split into bytes
                    token_bytes = token.encode('utf-8') if isinstance(token, str) else token
                    word_tokens.append([bytes([b]) for b in token_bytes])
        else:
            # Simple byte-level split without special tokens
            pretokens = self._pretokenize(text.decode('utf-8'))
            word_tokens = []
            for token in pretokens:
                token_bytes = token.encode('utf-8')
                word_tokens.append([bytes([b]) for b in token_bytes])
        
        if verbose:
            console.print(Panel.fit(
                f"[cyan]Pretokenized into[/cyan] [bold yellow]{len(pretokens)}[/bold yellow] [cyan]chunks[/cyan]",
                border_style="cyan"
            ))
        
        # Initialize vocabulary with all single bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
        
        # Add special tokens to vocabulary
        next_id = 256
        special_token_map = {}
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            self.vocab[next_id] = token_bytes
            special_token_map[token_bytes] = next_id
            next_id += 1
        
        # Count total tokens
        total_tokens = sum(len(word) for word in word_tokens)
        
        if verbose:
            stats_table = Table(box=box.ROUNDED, show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow bold")
            stats_table.add_row("Initial token count", str(total_tokens))
            stats_table.add_row("Initial vocabulary size", str(len(self.vocab)))
            stats_table.add_row("Special tokens", str(len(self.special_tokens)))
            console.print(stats_table)
            console.print()
            
            console.print(Panel(f"[bold green]Starting Byte-Level BPE Training[/bold green] - Performing {num_merges} merges", border_style="green"))
            console.print()
        
        # 3. Perform specified number of merge operations
        progress_iter = track(range(num_merges), description="[cyan]Training Progress") if verbose else range(num_merges)
        
        for i in progress_iter:
            # Count frequency of all adjacent token pairs across all words
            pair_counts = self._count_pairs_in_words(word_tokens)
            
            # Stop early if no more token pairs to merge
            if not pair_counts:
                if verbose:
                    console.print(f"[yellow]Warning: No more token pairs to merge, stopping after iteration {i}[/yellow]")
                break
            
            # Find the most frequent token pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            
            if verbose:
                # Create merge info table
                merge_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
                merge_table.add_column("Label", style="dim")
                merge_table.add_column("Value")
                merge_table.add_row("Merge #", f"[bold magenta]{i+1}[/bold magenta]")
                
                # Display bytes in a readable format
                try:
                    token1_display = best_pair[0].decode('utf-8', errors='backslashreplace')
                    token2_display = best_pair[1].decode('utf-8', errors='backslashreplace')
                    merged_display = (best_pair[0] + best_pair[1]).decode('utf-8', errors='backslashreplace')
                except:
                    token1_display = repr(best_pair[0])
                    token2_display = repr(best_pair[1])
                    merged_display = repr(best_pair[0] + best_pair[1])
                
                merge_table.add_row("Token pair", f"[green]{token1_display}[/green] + [green]{token2_display}[/green] -> [bold green]{merged_display}[/bold green]")
                merge_table.add_row("Frequency", f"[yellow]{best_count}[/yellow] times")
            
            # Merge this token pair in all words
            word_tokens = self._merge_pair_in_words(word_tokens, best_pair)
            
            # Record this merge operation
            self.merges.append(best_pair)
            
            # Add the newly merged token to vocabulary
            new_token = best_pair[0] + best_pair[1]
            self.vocab[next_id] = new_token
            next_id += 1
            
            if verbose:
                total_tokens = sum(len(word) for word in word_tokens)
                merge_table.add_row("Tokens after merge", f"[cyan]{total_tokens}[/cyan]")
                merge_table.add_row("Vocabulary size", f"[cyan]{len(self.vocab)}[/cyan]")
                
                console.print(merge_table)
                console.print()
        
        if verbose:
            # Final summary
            total_tokens = sum(len(word) for word in word_tokens)
            
            summary = Text()
            summary.append("*** Training Complete! ***\n\n", style="bold green")
            summary.append(f"Final token count: ", style="cyan")
            summary.append(f"{total_tokens}\n", style="yellow bold")
            summary.append(f"Final vocabulary size: ", style="cyan")
            summary.append(f"{len(self.vocab)}\n", style="yellow bold")
            summary.append(f"Number of merges performed: ", style="cyan")
            summary.append(f"{len(self.merges)}", style="yellow bold")
            
            console.print(Panel(summary, border_style="green", box=box.DOUBLE))
        
        return word_tokens
    
    def _count_pairs_in_words(self, word_tokens):
        """
        Count frequency of all adjacent token pairs across all words (brute force)
        Does not count pairs that cross word boundaries
        """
        pairs = {}
        for word in word_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def _merge_pair(self, tokens, pair):
        """
        Merge the specified token pair in a single token list (brute force replacement)
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            # Check if current position matches the token pair to merge
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                # Merge these two tokens
                new_tokens.append(pair[0] + pair[1])
                i += 2  # Skip the next token
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def _merge_pair_in_words(self, word_tokens, pair):
        """
        Merge the specified token pair in all words (brute force replacement)
        """
        return [self._merge_pair(word, pair) for word in word_tokens]
    
    def get_vocab(self):
        """
        Get the vocabulary as dict[int, bytes]
        """
        return self.vocab
    
    def get_merges(self):
        """
        Get the list of merge operations as list[tuple[bytes, bytes]]
        """
        return self.merges


def adapters(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer and return vocabulary and merges.
    
    Args:
        input_path: Path to the training text file
        vocab_size: Desired vocabulary size (must be >= 256 + len(special_tokens))
        special_tokens: List of special tokens to preserve (e.g., ['<|endoftext|>'])
    
    Returns:
        vocab: Dictionary mapping token IDs to bytes
        merges: List of merge operations as (bytes, bytes) tuples
    """
    # Validate vocab_size
    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(f"vocab_size must be at least {min_vocab_size} (256 base bytes + {len(special_tokens)} special tokens)")
    
    # Read training text
    console.print(Panel.fit(
        f"[bold cyan]Loading training data from:[/bold cyan]\n[yellow]{input_path}[/yellow]",
        border_style="cyan"
    ))
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    console.print(f"[green]Loaded {len(text):,} characters[/green]\n")
    
    # Calculate number of merges needed
    num_merges = vocab_size - min_vocab_size
    
    console.print(Panel.fit(
        f"[cyan]Target vocab size:[/cyan] [bold yellow]{vocab_size}[/bold yellow]\n"
        f"[cyan]Base bytes:[/cyan] [yellow]256[/yellow]\n"
        f"[cyan]Special tokens:[/cyan] [yellow]{len(special_tokens)}[/yellow]\n"
        f"[cyan]Merges to perform:[/cyan] [bold yellow]{num_merges}[/bold yellow]",
        border_style="blue"
    ))
    console.print()
    
    # Display special tokens
    if special_tokens:
        console.print("[bold cyan]Special tokens:[/bold cyan]")
        for token in special_tokens:
            console.print(f"  [green]{token}[/green]")
        console.print()
    
    # Train the BPE model
    trainer = ByteLevelBPETrainer(special_tokens=special_tokens)
    trainer.train(text, num_merges=num_merges, verbose=True)
    
    # Get vocabulary and merges
    vocab = trainer.get_vocab()
    merges = trainer.get_merges()
    
    return vocab, merges


class SimpleBPETrainer:
    def __init__(self, pretokenize_pattern=None):
        self.merges = []  # Record the order of merge operations
        self.vocab = set()  # Vocabulary
        self.pretokenize_pattern = pretokenize_pattern or r"(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self.compiled_pattern = regex.compile(self.pretokenize_pattern)
    
    def _pretokenize(self, text):
        """
        Pretokenize text using regex pattern
        Returns a list of pretokens (words/chunks)
        """
        return self.compiled_pattern.findall(text)
    
    def train(self, text, num_merges):
        """
        Train the BPE model
        
        Args:
            text: Training text
            num_merges: Number of merges to perform
        """
        # 1. Pretokenize text using regex pattern
        pretokens = self._pretokenize(text)
        
        console.print(Panel.fit(
            f"[cyan]Pretokenized into[/cyan] [bold yellow]{len(pretokens)}[/bold yellow] [cyan]chunks[/cyan]",
            border_style="cyan"
        ))
        console.print(f"[dim]First 10 pretokens:[/dim] {pretokens[:10]}\n")
        
        # 2. Convert each pretoken to character-level tokens
        # We'll work with a list of word-level token lists to avoid merging across word boundaries
        word_tokens = [[char for char in word] for word in pretokens]
        
        # Add initial characters to vocabulary
        for word in word_tokens:
            self.vocab.update(word)
        
        # Count total tokens
        total_tokens = sum(len(word) for word in word_tokens)
        
        # Create initial stats table
        stats_table = Table(box=box.ROUNDED, show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow bold")
        stats_table.add_row("Initial token count", str(total_tokens))
        stats_table.add_row("Initial vocabulary size", str(len(self.vocab)))
        console.print(stats_table)
        console.print()
        
        # 3. Perform specified number of merge operations
        console.print(Panel(f"[bold green]Starting BPE Training[/bold green] - Performing {num_merges} merges", border_style="green"))
        console.print()
        
        for i in track(range(num_merges), description="[cyan]Training Progress"):
            # Count frequency of all adjacent token pairs across all words
            pair_counts = self._count_pairs_in_words(word_tokens)
            
            # Stop early if no more token pairs to merge
            if not pair_counts:
                console.print(f"[yellow]Warning: No more token pairs to merge, stopping after iteration {i}[/yellow]")
                break
            
            # Find the most frequent token pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            
            # Create merge info table
            merge_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            merge_table.add_column("Label", style="dim")
            merge_table.add_column("Value")
            merge_table.add_row("Merge #", f"[bold magenta]{i+1}[/bold magenta]")
            merge_table.add_row("Token pair", f"[green]'{best_pair[0]}'[/green] + [green]'{best_pair[1]}'[/green] → [bold green]'{best_pair[0] + best_pair[1]}'[/bold green]")
            merge_table.add_row("Frequency", f"[yellow]{best_count}[/yellow] times")
            
            # Merge this token pair in all words
            word_tokens = self._merge_pair_in_words(word_tokens, best_pair)
            
            # Record this merge operation
            self.merges.append(best_pair)
            
            # Add the newly merged token to vocabulary
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add(new_token)
            
            total_tokens = sum(len(word) for word in word_tokens)
            merge_table.add_row("Tokens after merge", f"[cyan]{total_tokens}[/cyan]")
            merge_table.add_row("Vocabulary size", f"[cyan]{len(self.vocab)}[/cyan]")
            
            console.print(merge_table)
            console.print()
        
        # Final summary
        total_tokens = sum(len(word) for word in word_tokens)
        
        summary = Text()
        summary.append("*** Training Complete! ***\n\n", style="bold green")
        summary.append(f"Final token count: ", style="cyan")
        summary.append(f"{total_tokens}\n", style="yellow bold")
        summary.append(f"Final vocabulary size: ", style="cyan")
        summary.append(f"{len(self.vocab)}\n", style="yellow bold")
        summary.append(f"Number of merges performed: ", style="cyan")
        summary.append(f"{len(self.merges)}", style="yellow bold")
        
        console.print(Panel(summary, border_style="green", box=box.DOUBLE))
        
        return word_tokens
    
    def _count_pairs(self, tokens):
        """
        Count frequency of all adjacent token pairs in a single token list (brute force)
        """
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def _count_pairs_in_words(self, word_tokens):
        """
        Count frequency of all adjacent token pairs across all words (brute force)
        Does not count pairs that cross word boundaries
        """
        pairs = {}
        for word in word_tokens:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def _merge_pair(self, tokens, pair):
        """
        Merge the specified token pair in a single token list (brute force replacement)
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            # Check if current position matches the token pair to merge
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                # Merge these two tokens
                new_tokens.append(pair[0] + pair[1])
                i += 2  # Skip the next token
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def _merge_pair_in_words(self, word_tokens, pair):
        """
        Merge the specified token pair in all words (brute force replacement)
        """
        return [self._merge_pair(word, pair) for word in word_tokens]
    
    def encode(self, text):
        """
        Encode text using the learned merge rules
        """
        # Pretokenize the text first
        pretokens = self._pretokenize(text)
        
        # Convert each pretoken to character-level tokens
        word_tokens = [[char for char in word] for word in pretokens]
        
        # Apply merges in the order they were learned
        for pair in self.merges:
            word_tokens = self._merge_pair_in_words(word_tokens, pair)
        
        # Flatten the result
        result = []
        for word in word_tokens:
            result.extend(word)
        
        return result
    
    def get_vocab(self):
        """
        Get the vocabulary
        """
        return sorted(self.vocab)
    
    def get_merges(self):
        """
        Get the list of merge operations
        """
        return self.merges


def demo():
    """
    Demonstrate the usage of BPE trainer with pretokenization
    """
    # Sample text
    text = "Hello world! This is a BPE trainer. It's working correctly, isn't it? Yes, it's amazing! 123 numbers too."
    
    # Print header with style
    console.print()
    console.print(Panel.fit(
        "[bold blue]BPE Trainer Demo[/bold blue]\n[dim](with Pretokenization)[/dim]",
        border_style="blue",
        box=box.DOUBLE
    ))
    console.print()
    
    console.print(Panel(
        f"[italic]{text}[/italic]",
        title="[bold cyan]Training Text[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    
    # Create trainer with GPT-2 style pretokenization
    trainer = SimpleBPETrainer()
    
    # Train (perform 20 merges)
    result_tokens = trainer.train(text, num_merges=20)
    
    console.print()
    console.print(Panel.fit("[bold blue]Training Results[/bold blue]", border_style="blue"))
    console.print()
    
    # Sample of final word tokens
    console.print("[bold cyan]Sample of final word tokens (first 5 words):[/bold cyan]")
    token_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    token_table.add_column("Word #", style="cyan", justify="center")
    token_table.add_column("Tokens", style="green")
    
    for i, word in enumerate(result_tokens[:5]):
        token_table.add_row(str(i+1), str(word))
    
    console.print(token_table)
    console.print()
    
    # Learned merge rules
    console.print(f"[bold cyan]Learned merge rules ({len(trainer.get_merges())} total):[/bold cyan]")
    
    merge_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    merge_table.add_column("#", style="cyan", justify="center", width=4)
    merge_table.add_column("Token 1", style="green", justify="center")
    merge_table.add_column("Token 2", style="green", justify="center")
    merge_table.add_column("Result", style="yellow bold", justify="center")
    
    for i, merge in enumerate(trainer.get_merges(), 1):
        merge_table.add_row(
            str(i),
            f"'{merge[0]}'",
            f"'{merge[1]}'",
            f"'{merge[0] + merge[1]}'"
        )
    
    console.print(merge_table)
    console.print()
    
    # Test encoding new text
    test_text = "Hello! It's 456."
    encoded = trainer.encode(test_text)
    
    console.print(Panel.fit("[bold blue]Encoding Test[/bold blue]", border_style="blue"))
    console.print()
    
    encode_table = Table(box=box.ROUNDED, show_header=False)
    encode_table.add_column("Label", style="cyan bold")
    encode_table.add_column("Value", style="white")
    encode_table.add_row("Input text", f"[italic]{test_text}[/italic]")
    encode_table.add_row("Encoded tokens", f"[green]{encoded}[/green]")
    
    console.print(encode_table)
    console.print()


if __name__ == "__main__":
    demo()

