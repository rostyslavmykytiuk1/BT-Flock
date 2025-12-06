#!/usr/bin/env python3
"""
Script to check if training data has duplicates with other miners' data.

This script:
1. Loads your training data (data.jsonl)
2. Loads all other miners' data from hf_datajsonl/ directory
3. Compares using the same normalization method as validator
4. Reports duplicate counts with each miner
5. Warns if > 100 duplicates (the threshold that causes penalty)

Usage:
    python3 backtest/validate_data.py
    python3 backtest/validate_data.py --your-data backtest/data.jsonl --miners-dir backtest/hf_datajsonl
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse


def normalize_json_item(item: Dict) -> str:
    """
    Normalize a JSON item to a string for comparison (same as validator).
    This matches the validator's comparison method exactly.
    """
    return json.dumps(item, sort_keys=True)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
        return data
    except FileNotFoundError:
        print(f"  Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return []


def count_duplicates(
    your_data: List[Dict],
    other_data: List[Dict]
) -> Tuple[int, List[Dict]]:
    """
    Count duplicate rows between two datasets.
    
    Returns:
        (duplicate_count, list_of_duplicate_items)
    """
    # Normalize both datasets (same as validator)
    your_set = set(normalize_json_item(item) for item in your_data)
    other_set = set(normalize_json_item(item) for item in other_data)
    
    # Find intersection (duplicates)
    duplicates_set = your_set & other_set
    
    # Convert back to original items for reporting
    duplicate_items = []
    for item in your_data:
        if normalize_json_item(item) in duplicates_set:
            duplicate_items.append(item)
    
    return len(duplicates_set), duplicate_items


def extract_miner_info(filename: str) -> Dict[str, str]:
    """
    Extract miner information from filename.
    Format: {username}_{repo}_{commit_hash}.jsonl
    """
    name = Path(filename).stem  # Remove .jsonl extension
    parts = name.rsplit('_', 2)  # Split from right, max 2 splits
    
    if len(parts) == 3:
        username, repo, commit = parts
        return {
            'username': username,
            'repo': repo,
            'commit': commit,
            'filename': filename
        }
    else:
        return {
            'username': 'unknown',
            'repo': 'unknown',
            'commit': 'unknown',
            'filename': filename
        }


def validate_against_eval_data(
    your_data: List[Dict],
    eval_data_path: str
) -> bool:
    """
    Validate that all training data rows exist in evaluation dataset.
    
    Args:
        your_data: Your training data
        eval_data_path: Path to evaluation dataset
    
    Returns:
        True if all rows are valid, False otherwise
    """
    print(f"\n1. Validating your data against eval dataset: {eval_data_path}")
    
    if not Path(eval_data_path).exists():
        print(f"  ERROR: Evaluation dataset not found: {eval_data_path}")
        return False
    
    eval_data = load_jsonl(eval_data_path)
    if not eval_data:
        print("  ERROR: Could not load evaluation dataset!")
        return False
    
    print(f"   ✓ Loaded {len(eval_data)} rows from eval dataset")
    
    # Normalize eval data for comparison (same as validator)
    eval_set = set(normalize_json_item(item) for item in eval_data)
    
    # Check each training row
    missing = []
    for i, item in enumerate(your_data):
        normalized = normalize_json_item(item)
        if normalized not in eval_set:
            missing.append((i, item))
    
    if missing:
        print(f"\n  ❌ VALIDATION FAILED: {len(missing)} rows in your data are NOT in eval dataset!")
        print(f"     First 5 invalid rows:")
        for idx, item in missing[:5]:
            item_preview = json.dumps(item)[:100]
            print(f"       Row {idx}: {item_preview}...")
        if len(missing) > 5:
            print(f"       ... and {len(missing) - 5} more")
        print("\n  ⚠️  WARNING: Invalid data will get score 0.0 from validator!")
        print("     Fix your data before checking duplicates.")
        return False
    
    print(f"   ✓ Validation passed: All {len(your_data)} rows are in eval dataset")
    return True


def check_all_miners(
    your_data_path: str,
    miners_dir: str,
    eval_data_path: str = None,
    threshold: int = 100
) -> Dict:
    """
    Check duplicates with all miners' data.
    
    Args:
        your_data_path: Path to your training data (data.jsonl)
        miners_dir: Directory containing other miners' JSONL files
        eval_data_path: Path to evaluation dataset (for validation)
        threshold: Warning threshold for duplicates (default: 100)
    
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print("Checking for duplicates with other miners' data")
    print("=" * 70)
    
    # Load your data
    print(f"\n0. Loading your training data: {your_data_path}")
    your_data = load_jsonl(your_data_path)
    if not your_data:
        print("  ERROR: Could not load your training data!")
        return {}
    
    print(f"   ✓ Loaded {len(your_data)} rows from your data")
    
    # Validate against eval data if provided
    if eval_data_path:
        if not validate_against_eval_data(your_data, eval_data_path):
            print("\n" + "!" * 70)
            print("VALIDATION FAILED - Cannot proceed with duplicate checking")
            print("Fix your data first!")
            print("!" * 70)
            return {}
    
    # Find all miner data files
    print(f"\n2. Scanning miners' data directory: {miners_dir}")
    miners_dir_path = Path(miners_dir)
    if not miners_dir_path.exists():
        print(f"  ERROR: Directory not found: {miners_dir}")
        return {}
    
    miner_files = list(miners_dir_path.glob("*.jsonl"))
    print(f"   ✓ Found {len(miner_files)} miner data files")
    
    if not miner_files:
        print("  WARNING: No miner data files found!")
        return {}
    
    # Check duplicates with each miner
    print(f"\n3. Comparing with {len(miner_files)} miners...")
    print("-" * 70)
    
    results = {
        'your_data_count': len(your_data),
        'miners_checked': len(miner_files),
        'miners_with_duplicates': 0,
        'miners_above_threshold': 0,
        'max_duplicates': 0,
        'details': []
    }
    
    for miner_file in sorted(miner_files):
        miner_info = extract_miner_info(miner_file.name)
        
        # Load miner data
        miner_data = load_jsonl(str(miner_file))
        if not miner_data:
            continue
        
        # Count duplicates
        duplicate_count, duplicate_items = count_duplicates(your_data, miner_data)
        
        # Store results
        is_above_threshold = duplicate_count > threshold
        result_entry = {
            'miner': miner_info['username'],
            'repo': miner_info['repo'],
            'commit': miner_info['commit'][:16] + '...',  # Short commit hash
            'filename': miner_file.name,
            'miner_data_count': len(miner_data),
            'duplicate_count': duplicate_count,
            'above_threshold': is_above_threshold,
            'duplicate_percentage': (duplicate_count / len(your_data) * 100) if your_data else 0
        }
        results['details'].append(result_entry)
        
        # Update statistics
        if duplicate_count > 0:
            results['miners_with_duplicates'] += 1
        if is_above_threshold:
            results['miners_above_threshold'] += 1
            results['max_duplicates'] = max(results['max_duplicates'], duplicate_count)
        
        # Print result with repository name
        status = "⚠️  WARNING" if is_above_threshold else "✓ OK"
        repo_full = f"{miner_info['username']}/{miner_info['repo']}"
        print(f"{status} | {miner_info['username']:20} | "
              f"Repo: {repo_full:40} | "
              f"Duplicates: {duplicate_count:3d} / {len(your_data)} "
              f"({result_entry['duplicate_percentage']:5.1f}%)")
    
    return results


def print_summary(results: Dict):
    """Print summary of duplicate checking results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if not results:
        print("No results to display.")
        return
    
    print(f"\nYour training data: {results['your_data_count']} rows")
    print(f"Miners checked: {results['miners_checked']}")
    print(f"Miners with duplicates: {results['miners_with_duplicates']}")
    print(f"Miners above threshold (>100): {results['miners_above_threshold']}")
    
    if results['max_duplicates'] > 0:
        print(f"Maximum duplicates found: {results['max_duplicates']}")
    
    # Show top duplicates
    if results['details']:
        print("\n" + "-" * 70)
        print("Top 10 miners with most duplicates:")
        print("-" * 70)
        
        sorted_details = sorted(
            results['details'],
            key=lambda x: x['duplicate_count'],
            reverse=True
        )[:10]
        
        for i, detail in enumerate(sorted_details, 1):
            threshold_marker = " ⚠️" if detail['above_threshold'] else ""
            repo_full = f"{detail['miner']}/{detail['repo']}"
            print(f"{i:2d}. {detail['miner']:20} | "
                  f"Repo: {repo_full:40} | "
                  f"{detail['duplicate_count']:3d} duplicates "
                  f"({detail['duplicate_percentage']:5.1f}%){threshold_marker}")
    
    # Final warning
    if results['miners_above_threshold'] > 0:
        print("\n" + "!" * 70)
        print("⚠️  WARNING: You have >100 duplicates with some miners!")
        print("   If you submit later than them, you will get score 0.0")
        print("   Consider regenerating your data with a different seed.")
        print("!" * 70)
    elif results['miners_with_duplicates'] > 0:
        print("\n✓ You have some duplicates but all are below the 100 threshold.")
        print("  This is acceptable, but consider reducing duplicates for better uniqueness.")
    else:
        print("\n✓ No duplicates found! Your data is unique.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check for duplicates with other miners' data"
    )
    parser.add_argument(
        "--your-data",
        type=str,
        default="backtest/data.jsonl",
        help="Path to your training data (default: backtest/data.jsonl)"
    )
    parser.add_argument(
        "--miners-dir",
        type=str,
        default="backtest/hf_datajsonl",
        help="Directory containing other miners' data (default: backtest/hf_datajsonl)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Warning threshold for duplicates (default: 100)"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="backtest/eval_data.jsonl",
        help="Path to evaluation dataset for validation (default: backtest/eval_data.jsonl)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation against eval dataset (not recommended)"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.your_data).exists():
        print(f"ERROR: Your training data not found: {args.your_data}")
        return 1
    
    if not Path(args.miners_dir).exists():
        print(f"ERROR: Miners directory not found: {args.miners_dir}")
        return 1
    
    # Run duplicate check
    eval_data_path = None if args.skip_validation else args.eval_data
    results = check_all_miners(
        args.your_data,
        args.miners_dir,
        eval_data_path=eval_data_path,
        threshold=args.threshold
    )
    
    # Print summary
    print_summary(results)
    
    # Return error code if above threshold
    if results.get('miners_above_threshold', 0) > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

