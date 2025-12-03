#!/usr/bin/env python3
"""
Calculate total weight sum for a given data.jsonl file.

Uses the weights calculated by calculate_weights.py to determine
the total weight (sum of miner emissions) for the rows in the data file.

Usage:
    python3 backtest/calculate_data_weight.py --data backtest/data.jsonl --weights backtest/row_weights.json
    python3 backtest/calculate_data_weight.py --data backtest/other_data.jsonl --weights backtest/row_weights.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def normalize_json_item(item: Dict) -> str:
    """Normalize JSON item for comparison (same as validator)."""
    return json.dumps(item, sort_keys=True)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    if not Path(file_path).exists():
        print(f"ERROR: File not found: {file_path}")
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


def calculate_data_weight(data: List[Dict], row_weights: Dict[str, float]) -> Dict:
    """
    Calculate total weight and statistics for a dataset.
    
    Args:
        data: List of data items
        row_weights: Dictionary mapping normalized row -> weight
    
    Returns:
        Dictionary with weight statistics
    """
    total_weight = 0.0
    row_weights_list = []
    zero_weight_count = 0
    
    for item in data:
        normalized = normalize_json_item(item)
        weight = row_weights.get(normalized, 0.0)
        total_weight += weight
        row_weights_list.append(weight)
        if weight == 0.0:
            zero_weight_count += 1
    
    stats = {
        'total_rows': len(data),
        'total_weight': total_weight,
        'avg_weight_per_row': total_weight / len(data) if data else 0.0,
        'min_weight': min(row_weights_list) if row_weights_list else 0.0,
        'max_weight': max(row_weights_list) if row_weights_list else 0.0,
        'rows_with_zero_weight': zero_weight_count,
        'rows_with_weight': len(data) - zero_weight_count,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Calculate total weight sum for a data.jsonl file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.jsonl file to analyze"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="backtest/row_weights.json",
        help="Path to row weights JSON file (default: backtest/row_weights.json)"
    )
    parser.add_argument(
        "--miner-weights",
        type=str,
        default="backtest/miner_data_weights.json",
        help="Path to miner data weights JSON file (from calculate_weights.py) to get expected rank (default: backtest/miner_data_weights.json)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CALCULATE DATA WEIGHT")
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
    
    # Load data
    print(f"\n2. Loading data file: {args.data}")
    data = load_jsonl(args.data)
    if not data:
        print("ERROR: Could not load data file!")
        return 1
    print(f"   ✓ Loaded {len(data):,} rows from data file")
    
    # Calculate weight
    print(f"\n3. Calculating total weight...")
    stats = calculate_data_weight(data, row_weights)
    
    # Calculate expected rank if miner_weights file exists
    expected_rank = None
    total_miners = None
    if Path(args.miner_weights).exists():
        print(f"\n4. Calculating expected rank from {args.miner_weights}...")
        try:
            with open(args.miner_weights, 'r', encoding='utf-8') as f:
                miner_weights_data = json.load(f)
            
            ranked_miners = miner_weights_data.get('ranked_miners', [])
            if ranked_miners:
                # Find where this data weight would rank
                # Sort all data weights in descending order
                all_data_weights = [m['data_weight'] for m in ranked_miners]
                all_data_weights.append(stats['total_weight'])
                all_data_weights.sort(reverse=True)
                
                # Find the rank (1-indexed)
                expected_rank = all_data_weights.index(stats['total_weight']) + 1
                total_miners = len(ranked_miners)
                
                print(f"   ✓ Found rank information for {total_miners} miners")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load miner weights file: {e}")
    else:
        print(f"\n4. Miner weights file not found: {args.miner_weights}")
        print(f"   Skipping rank calculation (run calculate_weights.py first)")
    
    # Display results
    print(f"\n" + "=" * 70)
    print("WEIGHT STATISTICS")
    print("=" * 70)
    print(f"Total rows: {stats['total_rows']}")
    print(f"Total weight: {stats['total_weight']:.2f}")
    print(f"Average weight per row: {stats['avg_weight_per_row']:.6f}")
    print(f"Min weight: {stats['min_weight']:.6f}")
    print(f"Max weight: {stats['max_weight']:.6f}")
    print(f"Rows with weight > 0: {stats['rows_with_weight']}")
    print(f"Rows with weight = 0: {stats['rows_with_zero_weight']}")
    
    if expected_rank is not None:
        print(f"\nExpected Rank: {expected_rank} / {total_miners}")
        percentile = (1 - (expected_rank - 1) / total_miners) * 100
        print(f"Percentile: {percentile:.1f}% (better than {expected_rank - 1} out of {total_miners} miners)")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

