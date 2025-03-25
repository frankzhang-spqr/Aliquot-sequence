import matplotlib.pyplot as plt
import numpy as np
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint
from rich.table import Table
from rich.layout import Layout
import imageio.v2 as imageio
import os
import time

def proper_divisors(n):
    """Returns the proper divisors and their sum for n."""
    if n == 1:
        return [], 0
    divisors = [1]  # 1 is always a divisor
    total = 1
    sqrt_n = int(n**0.5)
    for i in range(2, sqrt_n + 1):
        if n % i == 0:
            divisors.append(i)
            total += i
            if i != n // i:  # Avoid adding square root twice
                divisors.append(n // i)
                total += n // i
    return sorted(divisors), total

def update_sequence_file(number, divisors, total):
    """Updates the sequence file with new calculations in sorted order."""
    filename = "aliquot_sequences.txt"
    entry = f"Number: {number}\nDivisors: {divisors}\nSum: {total}\n\n"
    
    # Read existing entries
    entries = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read().strip()
            if content:
                entries = content.split('\n\n')
    
    # Add new entry if it doesn't exist
    if not any(f"Number: {number}" in e for e in entries):
        entries.append(entry.strip())
        
        # Sort entries based on the number
        def get_number(entry):
            return int(entry.split('\n')[0].split(': ')[1])
        
        entries.sort(key=get_number)
        
        # Write back to file
        with open(filename, 'w') as f:
            f.write('\n\n'.join(entries) + '\n\n')

def update_chains_file(sequence):
    """Updates the chains file with a complete sequence for each unique number."""
    filename = "aliquot_chains.txt"
    
    # Add 0 to the sequence if it ends with 1
    if sequence[-1] == 1:
        sequence = sequence + [0]
    
    # Create entries for unique numbers in sequence with their full remaining sequence
    entries = {}
    for i, num in enumerate(sequence):
        if num not in entries:
            entry = f"Chain starting from {num}:\n"
            entry += " -> ".join(map(str, sequence[i:]))
            entries[num] = entry + "\n\n"
    
    # Read existing file entries
    existing_entries = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read().strip()
            if content:
                for entry in content.split('\n\n'):
                    if entry:
                        num = int(entry.split('\n')[0].split()[3].rstrip(':'))
                        existing_entries[num] = entry + "\n\n"
    
    # Merge entries, keeping only the longest sequence for each number
    all_entries = existing_entries | entries
    
    # Sort entries based on starting number
    sorted_entries = sorted(all_entries.values(), key=lambda e: int(e.split('\n')[0].split()[3].rstrip(':')))
    
    # Write back to file
    with open(filename, 'w') as f:
        f.write(''.join(sorted_entries))

def create_terminal_table(sequence, divisors_list):
    """Creates a rich table for terminal display."""
    table = Table(title="Aliquot Sequence Analysis")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Number", justify="right", style="green")
    table.add_column("Proper Divisors", style="magenta")
    table.add_column("Sum", justify="right", style="yellow")
    
    for idx, (num, divs) in enumerate(zip(sequence, divisors_list)):
        table.add_row(
            str(idx),
            str(num),
            str(divs),
            str(sum(divs))
        )
    return table

def plot_sequence(sequence, step, fig_path="temp_plots"):
    """Plots the sequence up to the current step and saves it."""
    plt.figure(figsize=(10, 6))
    plt.plot(sequence[:step+1], marker='o', linestyle='-', color='b')
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Aliquot Sequence (Step {step})")
    plt.grid(True)
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    filename = f"{fig_path}/step_{step:03d}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def create_layout(progress, table):
    """Creates a combined layout with progress and table."""
    layout = Layout()
    layout.split(
        Layout(Panel(progress), size=3),
        Layout(Panel(table))
    )
    return layout

def aliquot_sequence(n, max_iter=1000):
    """Generates the aliquot sequence with live visualization."""
    sequence = [n]
    divisors_list = []
    seen = set()
    frames = []
    
    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        expand=True
    )
    
    task = progress.add_task("[cyan]Calculating sequence...", total=max_iter)
    
    with Live(create_layout(progress, "Initializing..."), refresh_per_second=4) as live:
        for step in range(max_iter):
            # Calculate divisors and update sequence
            divs, total = proper_divisors(sequence[-1])
            divisors_list.append(divs)
            
            # Update terminal display
            table = create_terminal_table(sequence, divisors_list)
            live.update(create_layout(progress, table))
            
            # Save to files
            update_sequence_file(sequence[-1], divs, total)
            update_chains_file(sequence)
            
            # Create and save plot frame
            frame_path = plot_sequence(sequence, step)
            frames.append(frame_path)
            
            # Check for termination
            if total in seen or total == 0:
                progress.update(task, completed=max_iter)
                break
            
            seen.add(total)
            sequence.append(total)
            progress.update(task, advance=1)
            time.sleep(0.1)  # Slow down for visibility
    
    # Create animated GIF
    images = [imageio.imread(frame) for frame in frames]
    # Ensure aliquot_gifs directory exists
    if not os.path.exists('aliquot_gifs'):
        os.makedirs('aliquot_gifs')
    # Save gif with number as filename
    gif_path = f'aliquot_gifs/{n}.gif'
    imageio.mimsave(gif_path, images, duration=0.5)
    
    # Cleanup temporary files
    for frame in frames:
        os.remove(frame)
    os.rmdir('temp_plots')
    
    return sequence, divisors_list, gif_path

# Example usage
n = 138  # Example starting number
rprint("[bold green]Starting Aliquot Sequence Analysis[/bold green]")
sequence, divisors_list, gif_path = aliquot_sequence(n)
rprint("\n[bold blue]Analysis Complete![/bold blue]")
rprint(f"[yellow]Results saved to aliquot_sequences.txt, aliquot_chains.txt, and {gif_path}[/yellow]")
