#!/usr/bin/env python3
"""
Select 250 rows from eval_data to maximize total weight sum.

Uses weighted random sampling where weight = sum of miner emissions.
Validates duplicates with all miners' data (max 95 duplicates per miner).

Usage:
    python3 backtest/select_max_weight_data.py --weights backtest/row_weights.json --output backtest/data.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set
import random

# Selection strategy constants
MAX_DUPLICATES_PER_MINER = 100  # Skip if adding row would make duplicates >= this
MIN_WEIGHT_THRESHOLD = 0.0  # Minimum weight to consider (filters out low-quality rows)
WEIGHT_EXPONENT = 3.0  # Exponent for weight (weight^exp) to favor higher weights


def normalize_json_item(item: Dict) -> str:
    """Normalize JSON item for comparison (same as validator)."""
    return json.dumps(item, sort_keys=True)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    if not Path(file_path).exists():
        print(f"  Warning: File not found: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
    return data


def load_miners_data_by_miner(miners_dir: str) -> Dict[str, Set[str]]:
    """Load all miners' data and return dict: miner_file -> set of normalized items."""
    print(f"   Loading all miners' data for duplicate checking...")
    miners_data = {}
    miner_files = list(Path(miners_dir).glob("*.jsonl"))
    
    processed = 0
    for miner_file in miner_files:
        miner_data = load_jsonl(str(miner_file))
        miner_normalized = set()
        for item in miner_data:
            normalized = normalize_json_item(item)
            miner_normalized.add(normalized)
        miners_data[str(miner_file)] = miner_normalized
        processed += 1
        if processed % 50 == 0:
            print(f"     Processed {processed}/{len(miner_files)} files...")
    
    print(f"   ✓ Loaded data from {processed} miners")
    return miners_data


def select_max_weight_rows(
    eval_data: List[Dict],
    row_weights: Dict[str, float],
    num_rows: int,
    miners_dir: str,
    seed: int = None
) -> List[Dict]:
    """
    Select rows using weighted random sampling with weight exponentiation.
    
    Strategy:
    1. Filter rows by minimum weight threshold
    2. Apply weight exponentiation (weight^exp) to favor higher weights
    3. Use weighted random sampling from all available rows
    4. Check duplicates with previous selections and miners' data
    
    Args:
        eval_data: Full eval dataset
        row_weights: Dictionary mapping normalized row -> weight
        num_rows: Number of rows to select
        miners_dir: Directory with miners' data files
        seed: Random seed for reproducibility
    
    Returns:
        List of selected rows
    """
    print(f"\n2. Selecting {num_rows} rows using weighted random sampling...")
    print(f"   Strategy: Weight exponentiation (exp={WEIGHT_EXPONENT})")
    print(f"   Minimum weight threshold: {MIN_WEIGHT_THRESHOLD}")
    
    # Load miners' data
    miners_data = load_miners_data_by_miner(miners_dir)
    
    # Create list of (normalized, weight, original_item, index)
    all_rows = []
    for idx, item in enumerate(eval_data):
        normalized = normalize_json_item(item)
        weight = row_weights.get(normalized, 0.0)
        all_rows.append((normalized, weight, item, idx))
    
    # Filter by minimum weight threshold
    available_rows = [(n, w, i, idx) for n, w, i, idx in all_rows if w >= MIN_WEIGHT_THRESHOLD]
    filtered_count = len(all_rows) - len(available_rows)
    print(f"   ✓ Filtered {filtered_count} rows below weight threshold {MIN_WEIGHT_THRESHOLD}")
    print(f"   ✓ {len(available_rows)} rows available for selection")
    
    if len(available_rows) < num_rows:
        print(f"   ⚠️  WARNING: Only {len(available_rows)} rows available, less than {num_rows}!")
        print(f"   Lowering minimum weight threshold to 0.0 to get more rows...")
        available_rows = [(n, w, i, idx) for n, w, i, idx in all_rows if w > 0]
        if len(available_rows) < num_rows:
            print(f"   ⚠️  Still only {len(available_rows)} rows available!")
    
    # Initialize random number generator
    rng = random.Random(seed) if seed is not None else random.Random()
    
    # Track selected rows and duplicate counts per miner
    selected = []
    selected_normalized = set()
    miner_duplicate_counts = {miner_file: 0 for miner_file in miners_data.keys()}
    
    attempts = 0
    skipped_duplicate = 0
    skipped_miner_limit = 0
    total_weight = 0.0
    max_attempts = num_rows * 100  # Safety limit
    
    print(f"\n   Starting weighted random selection...")
    
    while len(selected) < num_rows and len(available_rows) > 0 and attempts < max_attempts:
        attempts += 1
        
        if len(available_rows) == 0:
            break
        
        # Apply weight exponentiation for sampling
        weights_for_sampling = [max(w ** WEIGHT_EXPONENT, 0.001) for _, w, _, _ in available_rows]
        
        # Weighted random selection from available rows
        selected_idx = rng.choices(range(len(available_rows)), weights=weights_for_sampling, k=1)[0]
        normalized, weight, item, original_idx = available_rows[selected_idx]
        
        # Check 1: Skip if duplicate with previous selections
        if normalized in selected_normalized:
            available_rows.pop(selected_idx)
            skipped_duplicate += 1
            continue
        
        # Check 2: Check if adding this row would cause any miner to have too many duplicates
        would_exceed = False
        for miner_file, miner_normalized in miners_data.items():
            current_dup_count = miner_duplicate_counts[miner_file]
            if normalized in miner_normalized:
                if current_dup_count >= MAX_DUPLICATES_PER_MINER:
                    would_exceed = True
                    break
        
        if would_exceed:
            available_rows.pop(selected_idx)
            skipped_miner_limit += 1
            continue
        
        # Row is valid - add to selection
        selected.append(item)
        selected_normalized.add(normalized)
        total_weight += weight
        
        # Update duplicate counts for miners
        for miner_file, miner_normalized in miners_data.items():
            if normalized in miner_normalized:
                miner_duplicate_counts[miner_file] += 1
        
        # Remove from available list
        available_rows.pop(selected_idx)
        
        # Progress update
        if len(selected) % 50 == 0:
            max_dup = max(miner_duplicate_counts.values()) if miner_duplicate_counts else 0
            print(f"     Selected {len(selected)}/{num_rows} rows... (weight: {total_weight:.2f}, max dup: {max_dup})")
    
    if len(selected) < num_rows:
        print(f"\n   ⚠️  WARNING: Only selected {len(selected)}/{num_rows} rows after {attempts} attempts!")
        print(f"     Skipped (duplicate with selection): {skipped_duplicate}")
        print(f"     Skipped (would exceed {MAX_DUPLICATES_PER_MINER} duplicates): {skipped_miner_limit}")
        print(f"     Remaining available rows: {len(available_rows)}")
    
    print(f"   ✓ Selection complete: {len(selected)} rows selected in {attempts} attempts")
    print(f"   Total weight: {total_weight:.2f}")
    print(f"   Average weight per row: {total_weight / len(selected):.2f}" if len(selected) > 0 else "")
    
    # Show duplicate statistics
    max_dup = max(miner_duplicate_counts.values()) if miner_duplicate_counts else 0
    miners_with_duplicates = sum(1 for count in miner_duplicate_counts.values() if count > 0)
    print(f"   Duplicate statistics:")
    print(f"     Max duplicates with any miner: {max_dup}")
    print(f"     Miners with duplicates: {miners_with_duplicates}/{len(miners_data)}")
    
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select rows from eval_data to maximize total weight sum"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="backtest/row_weights.json",
        help="Path to row weights JSON file (default: backtest/row_weights.json)"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="backtest/eval_data.jsonl",
        help="Path to eval dataset (default: backtest/eval_data.jsonl)"
    )
    parser.add_argument(
        "--miners-dir",
        type=str,
        default="backtest/hf_datajsonl",
        help="Directory with miners' data files (default: backtest/hf_datajsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest/data.jsonl",
        help="Output file path (default: backtest/data.jsonl)"
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=250,
        help="Number of rows to select (default: 250)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SELECT DATA TO MAXIMIZE TOTAL WEIGHT")
    print("=" * 70)
    
    # Load weights
    print(f"\n1. Loading row weights: {args.weights}")
    if not Path(args.weights).exists():
        print(f"ERROR: Weights file not found: {args.weights}")
        print(f"Please run calculate_weights.py first!")
        return 1
    
    with open(args.weights, 'r', encoding='utf-8') as f:
        row_weights = json.load(f)
    print(f"   ✓ Loaded weights for {len(row_weights)} rows")
    
    # Load eval data
    print(f"\n1.5. Loading eval dataset: {args.eval_data}")
    eval_data = load_jsonl(args.eval_data)
    if not eval_data:
        print("ERROR: Could not load eval dataset!")
        return 1
    print(f"   ✓ Loaded {len(eval_data):,} rows from eval dataset")
    
    # Select rows
    selected = select_max_weight_rows(
        eval_data,
        row_weights,
        args.num_rows,
        args.miners_dir,
        seed=args.seed
    )
    
    # Verify no duplicates within selected
    print(f"\n3. Verifying no duplicates within selected rows...")
    selected_normalized = set(normalize_json_item(item) for item in selected)
    if len(selected_normalized) != len(selected):
        print(f"   ⚠️  WARNING: Found duplicates! Removing...")
        seen = set()
        unique_selected = []
        for item in selected:
            normalized = normalize_json_item(item)
            if normalized not in seen:
                unique_selected.append(item)
                seen.add(normalized)
        selected = unique_selected
        print(f"   ✓ After deduplication: {len(selected)} unique rows")
    else:
        print(f"   ✓ No duplicates found")
    
    # Calculate total weight of selected rows
    selected_weight = sum(row_weights.get(normalize_json_item(item), 0.0) for item in selected)
    print(f"   Total weight of selected rows: {selected_weight:.2f}")
    
    # Save output
    print(f"\n4. Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Saved {len(selected)} rows")
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Eval dataset: {len(eval_data):,} rows")
    print(f"Rows selected: {len(selected)}")
    print(f"Total weight: {selected_weight:.2f}")
    print(f"Output file: {args.output}")
    print("=" * 70)
    
    if len(selected) < args.num_rows:
        print(f"\n⚠️  WARNING: Only {len(selected)} rows selected, less than {args.num_rows}!")
    
    return 0


if __name__ == "__main__":
    exit(main())

