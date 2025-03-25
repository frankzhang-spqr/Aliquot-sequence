import matplotlib.pyplot as plt
import numpy as np
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint
from rich.table import Table
from rich.console import Group
from rich.prompt import Prompt, IntPrompt
import imageio.v2 as imageio
import os
import time
from typing import Dict, List, Set, Tuple

# Global cache for sequences
sequence_cache: Dict[int, List[int]] = {}
divisors_cache: Dict[int, List[int]] = {}

def proper_divisors(n: int) -> Tuple[List[int], int]:
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

def update_sequence_file(number: int, divisors: List[int], total: int) -> None:
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

def update_chains_file(sequence: List[int]) -> None:
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

def create_terminal_table(sequence: List[int], divisors_list: List[List[int]]) -> Table:
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

def plot_sequence(sequence: List[int], step: int, fig_path: str = "temp_plots") -> str:
    """Plots the sequence up to the current step and saves it."""
    # Create figure with fixed size
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Plot points and lines
    x = list(range(step + 1))
    y = sequence[:step + 1]
    plt.plot(x, y, marker='o', linestyle='-', color='b', zorder=1)
    
    # Add labels for each point
    for i, value in enumerate(y):
        plt.annotate(str(value), (i, value), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=8)
    
    # Customize the plot
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Aliquot Sequence Starting from {sequence[0]} (Step {step})")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate sequence range for consistent scaling
    max_val = max(sequence)
    plt.ylim(-1, max_val * 1.2)  # Fixed y-axis range
    plt.xlim(-0.5, max(10, step + 1.5))  # Fixed x-axis range
    
    # Save the frame with fixed dimensions
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    filename = f"{fig_path}/step_{step:03d}.png"
    plt.savefig(filename, dpi=100, bbox_inches=None)  # Don't adjust bbox
    plt.close()
    return filename

def ensure_min_frames(sequence: List[int], frames: List[str]) -> List[str]:
    """Ensures at least one frame exists for the sequence."""
    if not frames:
        frame_path = plot_sequence(sequence, 0)
        frames.append(frame_path)
    return frames

def calculate_sequence(n: int, max_iter: int = 1000, use_cache: bool = True) -> Tuple[List[int], List[List[int]]]:
    """Core logic for calculating aliquot sequence, shared by both modes."""
    if use_cache and n in sequence_cache:
        return sequence_cache[n], divisors_cache[n]

    sequence = [n]
    divisors_list = []
    seen: Set[int] = set()

    if n == 1:
        divs, total = proper_divisors(1)
        divisors_list.append(divs)
    else:
        for step in range(max_iter):
            divs, total = proper_divisors(sequence[-1])
            divisors_list.append(divs)

            if use_cache and total in sequence_cache:
                sequence.append(total)
                remaining_sequence = sequence_cache[total]
                divisors_list.extend(divisors_cache[total])
                sequence.extend(remaining_sequence[1:])
                break

            update_sequence_file(sequence[-1], divs, total)
            update_chains_file(sequence)

            if total in seen or total == 0:
                break
            if step >= max_iter - 1:
                break

            seen.add(total)
            sequence.append(total)

    if use_cache:
        for i, num in enumerate(sequence):
            if num not in sequence_cache:
                sequence_cache[num] = sequence[i:]
                divisors_cache[num] = divisors_list[i:]

    return sequence, divisors_list

def create_sequence_gif(sequence: List[int], gif_path: str) -> str:
    """Create GIF visualization of the sequence, shared by both modes."""
    frames = []
    
    try:
        # Create a frame for each step
        for step in range(len(sequence)):
            frame_path = plot_sequence(sequence[:step + 1], step)
            frames.append(frame_path)

        # Create animated GIF
        if not os.path.exists('aliquot_gifs'):
            os.makedirs('aliquot_gifs')

        if frames:
            images = [imageio.imread(frame) for frame in frames]
            imageio.mimsave(gif_path, images, duration=0.5)
        else:
            frame_path = plot_sequence([sequence[0]], 0)
            imageio.mimsave(gif_path, [imageio.imread(frame_path)], duration=0.5)
            os.remove(frame_path)
    except Exception as e:
        print(f"Error creating GIF: {str(e)}")
        gif_path = ""
    finally:
        # Cleanup temporary files
        for frame in frames:
            try:
                os.remove(frame)
            except:
                pass
        if os.path.exists('temp_plots'):
            try:
                os.rmdir('temp_plots')
            except:
                pass

    return gif_path

def aliquot_sequence_no_live(n: int, max_iter: int = 1000, use_cache: bool = True) -> Tuple[List[int], List[List[int]], str]:
    """Generates the aliquot sequence without live visualization."""
    sequence, divisors_list = calculate_sequence(n, max_iter, use_cache)
    gif_path = f'aliquot_gifs/{n}.gif'
    gif_path = create_sequence_gif(sequence, gif_path)
    return sequence, divisors_list, gif_path

def aliquot_sequence(n: int, max_iter: int = 1000, use_cache: bool = True) -> Tuple[List[int], List[List[int]], str]:
    """Generates the aliquot sequence with live visualization."""
    sequence = [n]
    divisors_list = []
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    )
    task = progress.add_task("[cyan]Calculating sequence...", total=max_iter)
    table = create_terminal_table(sequence, divisors_list)
    
    with Live(Panel(Group(progress, table)), refresh_per_second=4) as live:
        # Calculate sequence with live updates
        sequence, divisors_list = calculate_sequence(n, max_iter, use_cache)
        for i in range(len(sequence)):
            # Update terminal display
            table = create_terminal_table(sequence[:i+1], divisors_list[:i+1])
            live.update(Panel(Group(progress, table)))
            progress.advance(task)
            time.sleep(0.1)
        progress.update(task, completed=max_iter)
    
    # Create visualization
    gif_path = f'aliquot_gifs/{n}.gif'
    gif_path = create_sequence_gif(sequence, gif_path)
    return sequence, divisors_list, gif_path

def process_range(start: int, end: int) -> None:
    """Process a range of numbers, utilizing cached results for efficiency."""
    total_numbers = end - start + 1
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    )
    
    overall_task = progress.add_task(f"[cyan]Processing numbers from {start} to {end}...", total=total_numbers)
    table = Table(title="Processing Status")
    table.add_column("Number", style="cyan")
    table.add_column("Status", style="green")
    
    try:
        with Live(Panel(Group(progress, table)), refresh_per_second=4) as live:
            for n in range(start, end + 1):
                progress.update(overall_task, description=f"[cyan]Processing number {n}...")
                
                # Calculate for current number
                try:
                    sequence, divisors_list, gif_path = aliquot_sequence_no_live(n)
                    status = "[green]Complete[/green]"
                    if not gif_path:
                        status += " (GIF failed)"
                except Exception as e:
                    print(f"Error processing {n}: {str(e)}")
                    status = "[red]Failed[/red]"
                
                # Update the table with the new row
                table.add_row(str(n), status)
                
                # Update progress
                progress.advance(overall_task)
            
            # Mark task as complete
            progress.update(overall_task, completed=total_numbers)
            
    except Exception as e:
        print(f"Error in process_range: {str(e)}")
    finally:
        rprint(f"\n[bold blue]Completed processing range {start} to {end}![/bold blue]")
        rprint("[yellow]All results have been saved to files and GIFs have been generated.[/yellow]")

def main() -> None:
    """Main function with mode selection and input handling."""
    rprint("[bold green]Aliquot Sequence Calculator[/bold green]")
    rprint("Choose mode:")
    rprint("1. Single number")
    rprint("2. Range of numbers")
    
    mode = Prompt.ask("Select mode", choices=["1", "2"])
    
    if mode == "1":
        n = IntPrompt.ask("Enter a number")
        rprint(f"\n[bold green]Starting Aliquot Sequence Analysis for {n}[/bold green]")
        try:
            sequence, divisors_list, gif_path = aliquot_sequence(n)
            rprint("\n[bold blue]Analysis Complete![/bold blue]")
            if gif_path:
                rprint(f"[yellow]Results saved to aliquot_sequences.txt, aliquot_chains.txt, and {gif_path}[/yellow]")
            else:
                rprint("[yellow]Results saved to aliquot_sequences.txt and aliquot_chains.txt (GIF generation failed)[/yellow]")
        except Exception as e:
            rprint(f"[red]Error analyzing number {n}: {str(e)}[/red]")
    else:
        start = IntPrompt.ask("Enter start number")
        end = IntPrompt.ask("Enter end number")
        if start > end:
            start, end = end, start
        process_range(start, end)

if __name__ == "__main__":
    main()
