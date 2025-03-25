# Aliquot Sequence Generator

This project generates and visualizes aliquot sequences, which are sequences of numbers where each number is the sum of the proper divisors of the previous number.

## What is an Aliquot Sequence?

An aliquot sequence is formed by:
1. Starting with a positive integer
2. Finding all its proper divisors (numbers that divide evenly into it, excluding the number itself)
3. Taking the sum of these proper divisors to get the next number
4. Repeating this process with each new number until:
   - The sequence reaches 0 (after hitting 1)
   - A number repeats (creating a cycle)
   - The sequence grows without bound

## Features

- Generates aliquot sequences for any starting number
- Creates animated GIFs visualizing the sequence progression
- Outputs three types of files:
  - `aliquot_sequences.txt`: Contains each number's proper divisors and their sum
  - `aliquot_chains.txt`: Shows the complete chain for each number in the sequence
  - `aliquot_gifs/`: Directory containing animated visualizations of sequences

## File Formats

### aliquot_sequences.txt
Lists each number with its proper divisors and their sum:
```
Number: n
Divisors: [list of proper divisors]
Sum: sum of proper divisors
```

### aliquot_chains.txt
Shows the complete sequence chain for each number:
```
Chain starting from n:
n -> a -> b -> ... -> 1 -> 0
```
- Each number appears only once with its complete sequence
- All sequences continue until reaching 0
- Chains are sorted by their starting number

## Usage

Run the script with Python 3:
```bash
python aliquot.py
```

The script will:
1. Generate the aliquot sequence starting from the specified number
2. Create an animated visualization
3. Save the results to the output files
4. Display a live progress table in the terminal
