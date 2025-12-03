#!/usr/bin/env python3
"""
Calculate weights for all eval_data rows based on miner emissions.

For each row in eval_data:
- Iterate through top N miners' data (by emission)
- If a miner uses that row, add that miner's emission to the row's weight
- Save weights to a JSON file for use by selection script

After calculating row weights:
- Calculate each miner's data weight (sum of row weights in their dataset)
- Rank miners by data weight
- Compare with real rank by emission
- Calculate correlation between data weight rank and emission rank

Usage:
    python3 backtest/calculate_weights.py --output backtest/row_weights.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Constants
TOP_N_MINERS = 60  # Number of top emission miners to use for weight calculation


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


def calculate_row_weight(
    eval_data: List[Dict],
    miners_dir: str,
    miner_emissions: Dict[str, float],
    filename_to_hotkey: Dict[str, str],
    top_n: int = TOP_N_MINERS
) -> Dict[str, float]:
    """
    Calculate weight for each eval_data row based on top N miner emissions.
    
    For each row, sum up the emissions of all top N miners who use that row.
    
    Args:
        eval_data: Evaluation dataset
        miners_dir: Directory with miners' data files
        miner_emissions: Dictionary mapping hotkey -> emission
        filename_to_hotkey: Mapping from filename to hotkey
        top_n: Number of top emission miners to use
    
    Returns:
        Dictionary mapping normalized row -> weight
    """
    print(f"\n2. Calculating weights for eval_data rows...")
    print(f"   Using top {top_n} miners by emission")
    print(f"   Weight = sum of emissions from miners who use each row")
    
    # Get top N miners by emission
    sorted_miners = sorted(
        miner_emissions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    top_miners_set = {hotkey for hotkey, _ in sorted_miners}
    top_miner_emissions = {hotkey: emission for hotkey, emission in sorted_miners}
    
    print(f"   ✓ Selected top {len(top_miners_set)} miners")
    print(f"   Emission range: {min(top_miner_emissions.values()):.6f} - {max(top_miner_emissions.values()):.6f}")
    
    # Normalize all eval_data rows
    eval_normalized = {}
    for item in eval_data:
        normalized = normalize_json_item(item)
        eval_normalized[normalized] = item
    
    # Initialize weights (default to 0.0)
    row_weights = {normalized: 0.0 for normalized in eval_normalized.keys()}
    
    # Load all miner files
    miner_files = list(Path(miners_dir).glob("*.jsonl"))
    print(f"   Found {len(miner_files)} miner data files")
    print(f"   Processing top {top_n} miners' data...")
    
    processed = 0
    total_emission_added = 0.0
    
    for miner_file in miner_files:
        # Get hotkey for this miner file
        filename = Path(miner_file).name
        hotkey = filename_to_hotkey.get(filename)
        
        # Only process if this miner is in top N
        if not hotkey or hotkey not in top_miners_set:
            continue
        
        miner_data = load_jsonl(str(miner_file))
        if not miner_data:
            continue
        
        emission = top_miner_emissions[hotkey]
        
        # For each row in this miner's data, add emission to its weight
        for item in miner_data:
            normalized = normalize_json_item(item)
            if normalized in row_weights:
                row_weights[normalized] += emission
                total_emission_added += emission
        
        processed += 1
        if processed % 10 == 0:
            print(f"     Processed {processed}/{len(top_miners_set)} files...")
    
    print(f"   ✓ Processed {processed} miner files")
    print(f"   Total emission added to weights: {total_emission_added:.2f}")
    
    # Statistics
    weights_list = list(row_weights.values())
    non_zero = sum(1 for w in weights_list if w > 0)
    print(f"\n   Weight statistics:")
    print(f"     Rows with weight > 0: {non_zero}/{len(weights_list)}")
    if weights_list:
        print(f"     Min weight: {min(weights_list):.6f}")
        print(f"     Max weight: {max(weights_list):.6f}")
        print(f"     Avg weight: {sum(weights_list)/len(weights_list):.6f}")
        non_zero_weights = [w for w in weights_list if w > 0]
        if non_zero_weights:
            print(f"     Avg weight (non-zero): {sum(non_zero_weights)/len(non_zero_weights):.6f}")
    
    return row_weights


def calculate_miner_data_weights(
    miners_dir: str,
    row_weights: Dict[str, float],
    filename_to_hotkey: Dict[str, str],
    miner_emissions: Dict[str, float]
) -> List[Dict]:
    """
    Calculate data weight for each miner by summing row weights in their dataset.
    
    Args:
        miners_dir: Directory with miners' data files
        row_weights: Dictionary mapping normalized row -> weight
        filename_to_hotkey: Mapping from filename to hotkey
        miner_emissions: Dictionary mapping hotkey -> emission
    
    Returns:
        List of dictionaries with miner info and data weights
    """
    print(f"\n3. Calculating data weight for each miner...")
    
    miner_data_weights = []
    miner_files = list(Path(miners_dir).glob("*.jsonl"))
    
    processed = 0
    for miner_file in miner_files:
        filename = Path(miner_file).name
        hotkey = filename_to_hotkey.get(filename)
        
        if not hotkey or hotkey not in miner_emissions:
            continue
        
        miner_data = load_jsonl(str(miner_file))
        if not miner_data:
            continue
        
        # Calculate total weight for this miner's data
        total_weight = 0.0
        for item in miner_data:
            normalized = normalize_json_item(item)
            weight = row_weights.get(normalized, 0.0)
            total_weight += weight
        
        emission = miner_emissions[hotkey]
        
        miner_info = {
            "hotkey": hotkey,
            "filename": filename,
            "emission": emission,
            "data_weight": total_weight,
            "data_rows": len(miner_data)
        }
        miner_data_weights.append(miner_info)
        
        processed += 1
        if processed % 50 == 0:
            print(f"     Processed {processed} miners...")
    
    print(f"   ✓ Calculated data weights for {len(miner_data_weights)} miners")
    
    return miner_data_weights


def rank_and_compare(
    miner_data_weights: List[Dict]
) -> Tuple[List[Dict], Dict]:
    """
    Rank miners by data weight and compare with emission rank.
    
    Args:
        miner_data_weights: List of miner info with data weights
    
    Returns:
        Tuple of (ranked_miners, correlation_stats)
    """
    print(f"\n4. Ranking miners and comparing with emission rank...")
    
    # Sort by data weight (descending)
    ranked_by_weight = sorted(
        miner_data_weights,
        key=lambda x: x['data_weight'],
        reverse=True
    )
    
    # Add ranks
    for i, miner in enumerate(ranked_by_weight):
        miner['data_weight_rank'] = i + 1
    
    # Sort by emission (descending) and add emission ranks
    ranked_by_emission = sorted(
        miner_data_weights,
        key=lambda x: x['emission'],
        reverse=True
    )
    
    # Create emission rank map
    emission_rank_map = {}
    for i, miner in enumerate(ranked_by_emission):
        emission_rank_map[miner['hotkey']] = i + 1
    
    # Add emission ranks to weight-ranked list
    for miner in ranked_by_weight:
        miner['emission_rank'] = emission_rank_map[miner['hotkey']]
        miner['rank_difference'] = miner['data_weight_rank'] - miner['emission_rank']
    
    # Calculate correlation
    data_weight_ranks = [m['data_weight_rank'] for m in ranked_by_weight]
    emission_ranks = [m['emission_rank'] for m in ranked_by_weight]
    
    # Spearman correlation (rank correlation)
    spearman_corr, spearman_p = spearmanr(data_weight_ranks, emission_ranks)
    
    # Pearson correlation (on actual values)
    data_weights = [m['data_weight'] for m in ranked_by_weight]
    emissions = [m['emission'] for m in ranked_by_weight]
    pearson_corr, pearson_p = pearsonr(data_weights, emissions)
    
    correlation_stats = {
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p),
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'num_miners': len(ranked_by_weight)
    }
    
    print(f"   ✓ Ranked {len(ranked_by_weight)} miners")
    print(f"   Spearman correlation (rank): {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"   Pearson correlation (values): {pearson_corr:.4f} (p={pearson_p:.4f})")
    
    return ranked_by_weight, correlation_stats


def main():
    parser = argparse.ArgumentParser(
        description="Calculate weights for eval_data rows based on miner emissions"
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
        default="backtest/row_weights.json",
        help="Output file for row weights (default: backtest/row_weights.json)"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="backtest/miners_metadata.json",
        help="Path to miners metadata JSON file (default: backtest/miners_metadata.json)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_MINERS,
        help=f"Number of top emission miners to use (default: {TOP_N_MINERS})"
    )
    parser.add_argument(
        "--output-miner-weights",
        type=str,
        default="backtest/miner_data_weights.json",
        help="Output file for miner data weights and rankings (default: backtest/miner_data_weights.json)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CALCULATE ROW WEIGHTS BASED ON MINER EMISSIONS")
    print("=" * 70)
    
    # Load eval data
    print(f"\n1. Loading eval dataset: {args.eval_data}")
    eval_data = load_jsonl(args.eval_data)
    if not eval_data:
        print("ERROR: Could not load eval dataset!")
        return 1
    print(f"   ✓ Loaded {len(eval_data):,} rows from eval dataset")
    
    # Load metadata file
    metadata_file_path = Path(args.metadata_file)
    if not metadata_file_path.exists():
        print(f"ERROR: Metadata file not found: {args.metadata_file}")
        print(f"Please run download_miners_data.py first!")
        return 1
    
    print(f"\n1.5. Loading miner metadata from file: {args.metadata_file}")
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        miner_emissions = metadata.get('miner_emissions', {})
        filename_to_hotkey = metadata.get('filename_to_hotkey', {})
        
        print(f"   ✓ Loaded emissions for {len(miner_emissions)} miners")
        print(f"   ✓ Loaded filename mappings for {len(filename_to_hotkey)} files")
    except Exception as e:
        print(f"ERROR: Could not load metadata file: {e}")
        return 1
    
    if not miner_emissions:
        print("ERROR: No miner emissions found in metadata file!")
        return 1
    
    # Calculate row weights using top N miners
    row_weights = calculate_row_weight(
        eval_data,
        args.miners_dir,
        miner_emissions,
        filename_to_hotkey,
        top_n=args.top_n
    )
    
    # Save row weights
    print(f"\n2.5. Saving row weights to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(row_weights, f, indent=2)
    
    print(f"   ✓ Saved weights for {len(row_weights)} rows")
    
    # Calculate miner data weights
    miner_data_weights = calculate_miner_data_weights(
        args.miners_dir,
        row_weights,
        filename_to_hotkey,
        miner_emissions
    )
    
    # Rank and compare
    ranked_miners, correlation_stats = rank_and_compare(miner_data_weights)
    
    # Save miner data weights and rankings
    print(f"\n5. Saving miner data weights and rankings to {args.output_miner_weights}...")
    output_miner_path = Path(args.output_miner_weights)
    output_miner_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "top_n_miners_used": args.top_n,
        "correlation_stats": correlation_stats,
        "ranked_miners": ranked_miners
    }
    
    with output_miner_path.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✓ Saved data for {len(ranked_miners)} miners")
    
    # Display top miners comparison
    print(f"\n" + "=" * 70)
    print("TOP 20 MINERS COMPARISON")
    print("=" * 70)
    print(f"{'Rank':<6} {'Data Wt Rank':<14} {'Emission Rank':<14} {'Diff':<8} {'Data Weight':<12} {'Emission':<10}")
    print("-" * 70)
    for i, miner in enumerate(ranked_miners[:20]):
        print(f"{i+1:<6} {miner['data_weight_rank']:<14} {miner['emission_rank']:<14} "
              f"{miner['rank_difference']:<8} {miner['data_weight']:<12.2f} {miner['emission']:<10.6f}")
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Eval dataset: {len(eval_data):,} rows")
    print(f"Total miners: {len(miner_emissions)}")
    print(f"Top N miners used: {args.top_n}")
    print(f"Miners with data weights: {len(ranked_miners)}")
    print(f"Rows with weight > 0: {sum(1 for w in row_weights.values() if w > 0)}")
    print(f"\nCorrelation Analysis:")
    print(f"  Spearman (rank correlation): {correlation_stats['spearman_correlation']:.4f}")
    print(f"  Pearson (value correlation): {correlation_stats['pearson_correlation']:.4f}")
    print(f"\nOutput files:")
    print(f"  Row weights: {args.output}")
    print(f"  Miner data weights: {args.output_miner_weights}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
