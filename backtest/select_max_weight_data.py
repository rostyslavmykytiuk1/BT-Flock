#!/usr/bin/env python3
"""
Generate 5 datasets, each containing 250 rows from eval_data to maximize total weight sum.

Uses two-phase approach:
- Phase 1: Adaptive weighted random sampling with capacity-aware probability adjustments
- Phase 2: Local search improvement (swap-based) to refine the solution

Validates duplicates with all miners' data (max 100 duplicates per miner).

Each dataset is saved to backtest/generated_data/data_{time_stamp}.jsonl

Usage:
    python3 backtest/select_max_weight_data.py --weights backtest/row_weights.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set
import random
import time
from datetime import datetime

# Selection strategy constants
MAX_DUPLICATES_PER_MINER = 100  # Skip if adding row would make duplicates >= this
MIN_WEIGHT_THRESHOLD = 0.0  # Minimum weight to consider (filters out low-quality rows)
WEIGHT_EXPONENT = 2.5  # Exponent for weight (weight^exp) to favor higher weights


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




def can_add_row(
    normalized: str,
    selected_normalized: Set[str],
    miner_duplicate_counts: Dict[str, int],
    row_to_miners: Dict[str, Set[str]]
) -> bool:
    """Check if a row can be added without violating constraints."""
    if normalized in selected_normalized:
        return False
    # Use pre-computed row_to_miners mapping for O(1) lookup instead of O(miners)
    affected_miners = row_to_miners.get(normalized, set())
    for miner_file in affected_miners:
        if miner_duplicate_counts[miner_file] >= MAX_DUPLICATES_PER_MINER:
            return False
    return True


def calculate_total_weight(selected: List[Dict], row_weights: Dict[str, float]) -> float:
    """Calculate total weight of selected rows."""
    return sum(row_weights.get(normalize_json_item(item), 0.0) for item in selected)


def local_search_improvement(
    selected: List[Dict],
    available_rows: List[tuple],
    row_weights: Dict[str, float],
    row_to_miners: Dict[str, Set[str]],
    max_iterations: int = 1000
) -> List[Dict]:
    """
    Improve solution using local search (1:1 swaps).
    Returns improved solution.
    """
    selected_normalized = {normalize_json_item(item) for item in selected}
    
    # Get all unique miner files from row_to_miners
    all_miner_files = set()
    for miners_set in row_to_miners.values():
        all_miner_files.update(miners_set)
    miner_duplicate_counts: Dict[str, int] = {miner_file: 0 for miner_file in all_miner_files}
    
    # Initialize duplicate counts using pre-computed mapping
    for item in selected:
        normalized = normalize_json_item(item)
        affected_miners = row_to_miners.get(normalized, set())
        for miner_file in affected_miners:
            miner_duplicate_counts[miner_file] += 1
    
    current_weight = calculate_total_weight(selected, row_weights)
    improved = True
    iterations = 0
    total_improvements = 0
    last_log_iteration = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Log progress every 50 iterations
        if iterations - last_log_iteration >= 50:
            print(f"     Local search iteration {iterations}... (weight: {current_weight:.2f}, improvements: {total_improvements})")
            last_log_iteration = iterations
        
        # Try swapping each selected row
        for i, selected_item in enumerate(selected):
            selected_norm = normalize_json_item(selected_item)
            selected_weight = row_weights.get(selected_norm, 0.0)
            
            # Find best swap candidate
            best_swap_idx = None
            best_improvement = 0.0
            
            for j, (avail_norm, avail_weight, avail_item, _) in enumerate(available_rows):
                if avail_norm == selected_norm:
                    continue
                
                # Check if swap would improve weight
                improvement = avail_weight - selected_weight
                if improvement <= 0:
                    continue
                
                # Temporarily remove selected row and check if we can add available row
                selected_affected_miners = row_to_miners.get(selected_norm, set())
                for miner_file in selected_affected_miners:
                    miner_duplicate_counts[miner_file] -= 1
                
                # Check if available row can be added
                temp_selected = selected_normalized - {selected_norm}
                can_add = can_add_row(avail_norm, temp_selected, miner_duplicate_counts, row_to_miners)
                
                # Restore counts
                for miner_file in selected_affected_miners:
                    miner_duplicate_counts[miner_file] += 1
                
                if can_add and improvement > best_improvement:
                    best_improvement = improvement
                    best_swap_idx = j
            
            # Perform swap if improvement found
            if best_swap_idx is not None:
                avail_norm, avail_weight, avail_item, _ = available_rows[best_swap_idx]
                
                # Remove old row from counts
                selected_affected_miners = row_to_miners.get(selected_norm, set())
                for miner_file in selected_affected_miners:
                    miner_duplicate_counts[miner_file] -= 1
                
                # Add new row to counts
                avail_affected_miners = row_to_miners.get(avail_norm, set())
                for miner_file in avail_affected_miners:
                    miner_duplicate_counts[miner_file] = miner_duplicate_counts.get(miner_file, 0) + 1
                
                # Update selected list
                selected[i] = avail_item
                selected_normalized.remove(selected_norm)
                selected_normalized.add(avail_norm)
                
                # Update available rows (swap the rows)
                available_rows[best_swap_idx] = (selected_norm, selected_weight, selected_item, 0)
                
                current_weight += best_improvement
                total_improvements += 1
                improved = True
                # Log each improvement
                if total_improvements % 10 == 0:
                    print(f"     Found {total_improvements} improvements so far... (current weight: {current_weight:.2f})")
                break  # Start over after each improvement
    
    if total_improvements > 0:
        print(f"     Made {total_improvements} improvements in {iterations} iterations")
    else:
        print(f"     No improvements found after {iterations} iterations")
    
    return selected


def iterated_local_search(
    selected: List[Dict],
    available_rows: List[tuple],
    row_weights: Dict[str, float],
    row_to_miners: Dict[str, Set[str]],
    all_rows: List[tuple],
    num_restarts: int = 5,
    kick_size: int = 5
) -> List[Dict]:
    """
    Iterated Local Search with kicks to escape local optima.
    
    Strategy:
    1. Do local search until convergence
    2. Apply a "kick" - randomly swap kick_size rows (even if worse)
    3. Do local search again from new state
    4. Repeat num_restarts times, keeping best solution
    
    This explores different regions of solution space and can find better solutions
    than single local search.
    """
    best_selected = selected.copy()
    best_weight = calculate_total_weight(selected, row_weights)
    
    current_selected = selected.copy()
    current_available = available_rows.copy()
    
    print(f"     Starting Iterated Local Search ({num_restarts} restarts, kick_size={kick_size})...")
    print(f"     Initial weight: {best_weight:.2f}")  
    
    for restart in range(num_restarts):
        # Step 1: Local search until convergence
        print(f"     Restart {restart + 1}/{num_restarts}: Local search...")
        current_selected = local_search_improvement(
            current_selected, current_available, row_weights, row_to_miners, max_iterations=500
        )
        current_weight = calculate_total_weight(current_selected, row_weights)
        
        # CRITICAL: Always check and update best solution after local search
        if current_weight > best_weight:
            best_selected = current_selected.copy()
            best_weight = current_weight
            print(f"       ✓ New best weight: {best_weight:.2f}")
        
        # Step 2: Apply "kick" - randomly swap some rows to escape local optimum
        if restart < num_restarts - 1:  # Don't kick after last restart
            print(f"       Applying kick (swapping {kick_size} random rows)...")
            
            # Rebuild available rows from all_rows (what's not currently selected)
            # This ensures we have the full set of available rows, not just what was
            # in the modified available_rows list
            current_normalized = {normalize_json_item(item) for item in current_selected}
            current_available = []
            for norm, weight, item, idx in all_rows:
                if norm not in current_normalized:
                    current_available.append((norm, weight, item, idx))
            
            # Get all rows (selected + available) for kick
            all_for_kick = []
            for item in current_selected:
                norm = normalize_json_item(item)
                weight = row_weights.get(norm, 0.0)
                all_for_kick.append((norm, weight, item, 'selected'))
            all_for_kick.extend(current_available)
            
            # Initialize duplicate counts
            all_miner_files = set()
            for miners_set in row_to_miners.values():
                all_miner_files.update(miners_set)
            miner_dup_counts = {miner_file: 0 for miner_file in all_miner_files}
            for item in current_selected:
                norm = normalize_json_item(item)
                affected = row_to_miners.get(norm, set())
                for miner_file in affected:
                    miner_dup_counts[miner_file] += 1
            
            # Try to swap kick_size random selected rows with random available rows
            import random
            selected_indices = list(range(len(current_selected)))
            random.shuffle(selected_indices)
            kicks_applied = 0
            
            for idx in selected_indices[:kick_size]:
                if len(current_available) == 0:
                    break
                
                selected_item = current_selected[idx]
                selected_norm = normalize_json_item(selected_item)
                
                # Try to find a random available row that can replace it
                random.shuffle(current_available)
                for avail_norm, avail_weight, avail_item, _ in current_available:
                    if avail_norm == selected_norm:
                        continue
                    
                    # Check if swap is valid
                    selected_affected = row_to_miners.get(selected_norm, set())
                    for miner_file in selected_affected:
                        miner_dup_counts[miner_file] -= 1
                    
                    temp_selected = current_normalized - {selected_norm}
                    can_add = can_add_row(avail_norm, temp_selected, miner_dup_counts, row_to_miners)
                    
                    for miner_file in selected_affected:
                        miner_dup_counts[miner_file] += 1
                    
                    if can_add:
                        # Apply kick swap (even if weight is worse)
                        selected_affected = row_to_miners.get(selected_norm, set())
                        for miner_file in selected_affected:
                            miner_dup_counts[miner_file] -= 1
                        
                        avail_affected = row_to_miners.get(avail_norm, set())
                        for miner_file in avail_affected:
                            miner_dup_counts[miner_file] = miner_dup_counts.get(miner_file, 0) + 1
                        
                        current_selected[idx] = avail_item
                        current_normalized.remove(selected_norm)
                        current_normalized.add(avail_norm)
                        
                        # Update available rows
                        current_available = [(n, w, i, idx) for n, w, i, idx in current_available if n != avail_norm]
                        current_available.append((selected_norm, row_weights.get(selected_norm, 0.0), selected_item, 0))
                        
                        kicks_applied += 1
                        break
            
            kick_weight = calculate_total_weight(current_selected, row_weights)
            print(f"       Kick applied: {kicks_applied} swaps, new weight: {kick_weight:.2f} (may be worse)")
            
            # Rebuild current_available after kick for next local search iteration
            # This ensures available_rows list is correct for the next iteration
            current_normalized = {normalize_json_item(item) for item in current_selected}
            current_available = []
            for norm, weight, item, idx in all_rows:
                if norm not in current_normalized:
                    current_available.append((norm, weight, item, idx))
    
    # Final check: ensure we return the absolute best solution found
    final_weight = calculate_total_weight(best_selected, row_weights)
    if abs(final_weight - best_weight) > 0.01:  # Small tolerance for floating point
        print(f"     ⚠️  WARNING: Best weight mismatch! best_weight={best_weight:.2f}, final_weight={final_weight:.2f}")
        # Recalculate to be sure
        best_weight = final_weight
    
    print(f"     Best weight found: {best_weight:.2f} (improvement: {best_weight - calculate_total_weight(selected, row_weights):.2f})")
    return best_selected


def select_max_weight_rows(
    eval_data: List[Dict],
    row_weights: Dict[str, float],
    num_rows: int,
    miners_dir: str,
    seed: int = None
) -> List[Dict]:
    """
    Select rows using greedy algorithm with local search improvement.
    
    Strategy (based on research: greedy + local search is effective for constrained optimization):
    1. Build initial solution using weighted random sampling (good exploration)
    2. Improve solution using local search: swap selected rows with better available rows
    3. This combines the exploration of random with the refinement of local search
    
    Args:
        eval_data: Full eval dataset
        row_weights: Dictionary mapping normalized row -> weight
        num_rows: Number of rows to select
        miners_dir: Directory with miners' data files
        seed: Random seed for reproducibility
    
    Returns:
        List of selected rows
    """
    print(f"\n2. Selecting {num_rows} rows using greedy + local search...")
    print(f"   Phase 1: Build initial solution with weighted random sampling")
    print(f"   Phase 2: Improve solution with local search (swap-based)")
    print(f"   Weight exponent: {WEIGHT_EXPONENT}")
    print(f"   Minimum weight threshold: {MIN_WEIGHT_THRESHOLD}")
    
    # Load miners' data
    miners_data = load_miners_data_by_miner(miners_dir)
    
    # Pre-compute reverse index: row -> set of miners that contain it (for O(1) lookup)
    print(f"   Building row-to-miners index...")
    row_to_miners: Dict[str, Set[str]] = {}
    for miner_file, miner_normalized in miners_data.items():
        for normalized in miner_normalized:
            if normalized not in row_to_miners:
                row_to_miners[normalized] = set()
            row_to_miners[normalized].add(miner_file)
    print(f"   ✓ Built index for {len(row_to_miners)} unique rows")
    
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
    rng = random.Random(seed or int(time.time()))
    
    # Track selected rows and duplicate counts per miner
    selected = []
    selected_normalized = set()
    miner_duplicate_counts = {miner_file: 0 for miner_file in miners_data.keys()}
    
    attempts = 0
    skipped_duplicate = 0
    skipped_miner_limit = 0
    total_weight = 0.0
    
    print(f"\n   Phase 1: Building initial solution with weighted random sampling...")
    
    # Phase 1: Build initial solution using weighted random sampling
    while len(selected) < num_rows and len(available_rows) > 0:
        attempts += 1
        
        if len(available_rows) == 0:
            break
        
        # Filter available rows: remove duplicates with selected and rows that would exceed limit
        # Use list comprehension with early filtering for better performance
        valid_candidates = []
        rows_to_remove = set()
        
        for row_tuple in available_rows:
            normalized, weight, item, original_idx = row_tuple
            
            # Check 1: Skip if duplicate with previous selections (permanently invalid)
            if normalized in selected_normalized:
                skipped_duplicate += 1
                rows_to_remove.add(normalized)
                continue
            
            # Check 2: Check if adding this row would cause any miner to have too many duplicates
            # Use pre-computed row_to_miners for O(1) lookup instead of O(miners) iteration
            affected_miners = row_to_miners.get(normalized, set())
            would_exceed = False
            max_dup_if_added = 0
            total_remaining_capacity = 0
            affected_count = 0
            
            for miner_file in affected_miners:
                current_dup_count = miner_duplicate_counts[miner_file]
                if current_dup_count >= MAX_DUPLICATES_PER_MINER:
                    would_exceed = True
                    break
                affected_count += 1
                remaining_capacity = MAX_DUPLICATES_PER_MINER - current_dup_count
                total_remaining_capacity += remaining_capacity
                max_dup_if_added = max(max_dup_if_added, current_dup_count + 1)
            
            if would_exceed:
                skipped_miner_limit += 1
                # Don't remove permanently - might become valid later
                continue
            
            # Calculate capacity factor while we have the data
            if affected_count > 0:
                avg_remaining_capacity = total_remaining_capacity / affected_count
                capacity_factor = min(avg_remaining_capacity / 20.0, 1.5)
                if max_dup_if_added >= MAX_DUPLICATES_PER_MINER * 0.9:
                    capacity_factor *= 0.5
            else:
                capacity_factor = 1.2
            
            # Calculate selection weight
            base_weight = max(weight ** WEIGHT_EXPONENT, 0.001)
            selection_weight = base_weight * capacity_factor
            
            valid_candidates.append((normalized, weight, item, original_idx, selection_weight))
        
        # Remove permanently invalid rows
        if rows_to_remove:
            available_rows = [(n, w, i, idx) for n, w, i, idx in available_rows if n not in rows_to_remove]
        
        if len(valid_candidates) == 0:
            print(f"   ⚠️  No valid candidates found, stopping...")
            break
        
        # Extract selection weights (already calculated above)
        selection_weights = [candidate[4] for candidate in valid_candidates]
        # Update valid_candidates to remove selection_weight (keep original format)
        valid_candidates = [(c[0], c[1], c[2], c[3]) for c in valid_candidates]
        
        # Weighted random selection
        if len(selection_weights) == 0:
            break
        
        selected_idx = rng.choices(range(len(valid_candidates)), weights=selection_weights, k=1)[0]
        best_normalized, best_weight, best_item, best_original_idx = valid_candidates[selected_idx]
        
        # Add selected candidate to selection
        selected.append(best_item)
        selected_normalized.add(best_normalized)
        total_weight += best_weight
        
        # Update duplicate counts for miners using pre-computed mapping
        affected_miners = row_to_miners.get(best_normalized, set())
        for miner_file in affected_miners:
            miner_duplicate_counts[miner_file] += 1
        
        # Remove selected candidate from available_rows
        available_rows = [(n, w, i, idx) for n, w, i, idx in available_rows if n != best_normalized]
        
        # Progress update
        if len(selected) % 50 == 0:
            max_dup = max(miner_duplicate_counts.values()) if miner_duplicate_counts else 0
            print(f"     Selected {len(selected)}/{num_rows} rows... (weight: {total_weight:.2f}, max dup: {max_dup})")
    
    if len(selected) < num_rows:
        print(f"\n   ⚠️  WARNING: Only selected {len(selected)}/{num_rows} rows after {attempts} iterations!")
        print(f"     Skipped (duplicate with selection): {skipped_duplicate}")
        print(f"     Skipped (would exceed {MAX_DUPLICATES_PER_MINER} duplicates): {skipped_miner_limit}")
        print(f"     Remaining available rows: {len(available_rows)}")
    
    initial_weight = total_weight
    print(f"   ✓ Phase 1 complete: {len(selected)} rows selected")
    print(f"   Initial total weight: {initial_weight:.2f}")
    print(f"   Average weight per row: {initial_weight / len(selected):.2f}" if len(selected) > 0 else "")
    
    # Phase 2: Iterated Local Search with kicks
    if len(selected) == num_rows:
        print(f"\n   Phase 2: Improving solution with Iterated Local Search...")
        # Build list of all unselected rows for swapping
        all_available = []
        selected_normalized_set = {normalize_json_item(item) for item in selected}
        for normalized, weight, item, idx in all_rows:
            norm = normalize_json_item(item)
            if norm not in selected_normalized_set:
                all_available.append((norm, weight, item, idx))
        
        selected = iterated_local_search(selected, all_available, row_weights, row_to_miners, all_rows)
        improved_weight = calculate_total_weight(selected, row_weights)
        improvement = improved_weight - initial_weight
        print(f"   ✓ Phase 2 complete")
        print(f"   Improved total weight: {improved_weight:.2f} (+{improvement:.2f})")
        total_weight = improved_weight
    else:
        print(f"   Skipping Phase 2: incomplete initial solution")
    
    # Show duplicate statistics using pre-computed mapping
    final_miner_duplicate_counts: Dict[str, int] = {}
    for item in selected:
        normalized = normalize_json_item(item)
        affected_miners = row_to_miners.get(normalized, set())
        for miner_file in affected_miners:
            final_miner_duplicate_counts[miner_file] = final_miner_duplicate_counts.get(miner_file, 0) + 1
    
    max_dup = max(final_miner_duplicate_counts.values()) if final_miner_duplicate_counts else 0
    miners_with_duplicates = len(final_miner_duplicate_counts)
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
        "--num-rows",
        type=int,
        default=250,
        help="Number of rows to select per dataset (default: 250)"
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=5,
        help="Number of datasets to generate (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
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
    
    # Create output directory
    output_dir = Path("backtest/generated_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate multiple datasets
    print(f"\n2. Generating {args.num_datasets} datasets...")
    all_output_files = []
    
    for dataset_num in range(1, args.num_datasets + 1):
        print(f"\n" + "=" * 70)
        print(f"GENERATING DATASET {dataset_num}/{args.num_datasets}")
        print("=" * 70)
        
        # Use different seed for each dataset to get variety
        # If user provided a seed, use it as base and add dataset_num
        dataset_seed = (args.seed + dataset_num) if args.seed is not None else None
        
        # Select rows
        selected = select_max_weight_rows(
            eval_data,
            row_weights,
            args.num_rows,
            args.miners_dir,
            seed=dataset_seed
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
        
        # Ensure we have exactly 250 rows
        if len(selected) != args.num_rows:
            print(f"   ⚠️  WARNING: Dataset {dataset_num} has {len(selected)} rows, expected {args.num_rows}!")
        
        # Calculate total weight of selected rows
        selected_weight = sum(row_weights.get(normalize_json_item(item), 0.0) for item in selected)
        print(f"   Total weight of selected rows: {selected_weight:.2f}")
        
        # Generate timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_filename = f"data_{timestamp}.jsonl"
        output_path = output_dir / output_filename
        
        # Save output
        print(f"\n4. Saving dataset {dataset_num} to {output_path}...")
        with output_path.open('w', encoding='utf-8') as f:
            for item in selected:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"   ✓ Saved {len(selected)} rows to {output_path}")
        all_output_files.append(str(output_path))
        
        # Small delay to ensure unique timestamps
        time.sleep(0.01)
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Eval dataset: {len(eval_data):,} rows")
    print(f"Datasets generated: {args.num_datasets}")
    print(f"Rows per dataset: {args.num_rows}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    for i, output_file in enumerate(all_output_files, 1):
        print(f"  {i}. {output_file}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

