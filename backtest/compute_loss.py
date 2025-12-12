# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import argparse
import torch
import random
import shutil
import tempfile
import json
import numpy as np
import bittensor as bt
from flockoff.constants import Competition
from flockoff import constants
from flockoff.validator.trainer import (
    train_lora,
    download_dataset,
)
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, List, Tuple

load_dotenv()

# Reference points for rank estimation (rank: loss)
# Lower loss = better rank (rank 1 is best)
RANK_REFERENCE_POINTS = [
    (1, 2.414466),
    (41, 2.416089),
    (81, 2.416649),
    (121, 2.416813),
    (161, 2.416864),
]

# Constants
MINERS_METADATA_FILE = "backtest/miners_metadata.json"
MINERS_DATA_DIR = "backtest/hf_datajsonl"
NUM_RUNS_PER_MINER = 3  # Number of times to compute loss with different seeds

# UIDs to process - modify this array directly
UIDS_TO_PROCESS = [
    # Add your UIDs here, e.g.:
    # 1, 2, 3, 4, 5,
]


def estimate_rank(loss: float) -> int:
    """
    Estimate rank based on loss value using linear interpolation between reference points.
    Lower loss = better rank (rank 1 is best).
    
    Args:
        loss: The loss value to estimate rank for
        
    Returns:
        Estimated rank (integer)
    """
    # If loss is better than or equal to rank 1, return rank 1
    if loss <= RANK_REFERENCE_POINTS[0][1]:
        return 1
    
    # If loss is worse than rank 161, estimate beyond 161
    if loss >= RANK_REFERENCE_POINTS[-1][1]:
        # Extrapolate beyond rank 161
        last_rank, last_loss = RANK_REFERENCE_POINTS[-1]
        second_last_rank, second_last_loss = RANK_REFERENCE_POINTS[-2]
        
        # Calculate slope
        rank_diff = last_rank - second_last_rank
        loss_diff = last_loss - second_last_loss
        
        if loss_diff > 0:
            loss_beyond = loss - last_loss
            rank_increase = int((loss_beyond / loss_diff) * rank_diff)
            return last_rank + rank_increase
        else:
            return last_rank + 1
    
    # Find the two reference points the loss falls between
    for i in range(len(RANK_REFERENCE_POINTS) - 1):
        rank1, loss1 = RANK_REFERENCE_POINTS[i]
        rank2, loss2 = RANK_REFERENCE_POINTS[i + 1]
        
        if loss1 <= loss <= loss2:
            # Linear interpolation
            if loss2 == loss1:
                return rank1
            
            # Interpolate rank
            loss_ratio = (loss - loss1) / (loss2 - loss1)
            estimated_rank = rank1 + (rank2 - rank1) * loss_ratio
            return int(round(estimated_rank))
    
    # Fallback (shouldn't reach here)
    return 161


def load_miners_metadata(metadata_file: str) -> Dict:
    """Load miners metadata from JSON file."""
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Miners metadata file not found: {metadata_file}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata


def get_miner_filename(uid: int, metadata: Dict) -> Optional[str]:
    """Get filename for a miner UID from metadata."""
    miners = metadata.get('miners', [])
    for miner in miners:
        if miner.get('uid') == uid:
            return miner.get('filename')
    return None


def get_dataset_path_for_uid(uid: int, metadata: Dict, miners_dir: str) -> Optional[str]:
    """Get full dataset path for a miner UID."""
    filename = get_miner_filename(uid, metadata)
    if not filename:
        return None
    
    dataset_path = Path(miners_dir) / filename
    if dataset_path.exists():
        return str(dataset_path)
    return None


def process_miner(
    uid: int,
    dataset_path: str,
    eval_data_dir: str,
    cache_dir: str,
    competition: Competition,
    base_data_dir: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Process a single miner's dataset and return the average loss from 3 runs.
    
    Returns:
        (average_loss, error_message)
    """
    dataset_path = os.path.expanduser(dataset_path)
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        return None, f"Dataset path not found: {dataset_path}"
    
    # Determine data directory for this miner
    if base_data_dir:
        data_dir = os.path.join(base_data_dir, f"miner_{uid}")
    else:
        # Use parent directory of dataset path
        if os.path.isfile(dataset_path):
            data_dir = os.path.dirname(dataset_path)
        else:
            data_dir = dataset_path
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy dataset to data_dir/data.jsonl
    if os.path.isfile(dataset_path):
        if dataset_path.endswith(".jsonl"):
            target_path = os.path.join(data_dir, "data.jsonl")
            if dataset_path != target_path:
                shutil.copy2(dataset_path, target_path)
        else:
            return None, f"Dataset file must be a .jsonl file: {dataset_path}"
    else:
        # It's a directory, check for data.jsonl
        data_jsonl_path = os.path.join(data_dir, "data.jsonl")
        if not os.path.exists(data_jsonl_path):
            return None, f"data.jsonl not found in directory: {data_dir}"
    
    # Run training 3 times with different random seeds
    losses = []
    errors = []
    
    for run_num in range(1, NUM_RUNS_PER_MINER + 1):
        # Generate a different random seed for each run
        lucky_num = int.from_bytes(os.urandom(4), "little")
        
        bt.logging.info(f"  Run {run_num}/{NUM_RUNS_PER_MINER} for UID {uid} (seed: {lucky_num})")
        
        try:
            eval_loss = train_lora(
                lucky_num,
                competition.bench,
                competition.rows,
                cache_dir=cache_dir,
                data_dir=data_dir,
                eval_data_dir=eval_data_dir,
            )
            losses.append(eval_loss)
            bt.logging.info(f"    ✓ Run {run_num} Loss: {eval_loss:.6f}")
        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)
            bt.logging.error(f"    ✗ Run {run_num} Error: {error_msg}")
            if "CUDA" in error_msg:
                bt.logging.error("    CUDA error detected")
    
    # Calculate average if we have at least one successful run
    if losses:
        avg_loss = np.mean(losses)
        bt.logging.info(f"  ✓ UID {uid}: Average loss from {len(losses)}/{NUM_RUNS_PER_MINER} runs: {avg_loss:.6f}")
        if errors:
            bt.logging.warning(f"  ⚠️  UID {uid}: {len(errors)} run(s) failed")
        return avg_loss, None
    else:
        # All runs failed
        error_summary = "; ".join(errors[:3])  # Show first 3 errors
        if len(errors) > 3:
            error_summary += f" ... and {len(errors) - 3} more"
        return None, f"All {NUM_RUNS_PER_MINER} runs failed: {error_summary}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute loss for miners by UID using the same logic, model, and resources as the validator"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=MINERS_METADATA_FILE,
        help=f"Path to miners metadata JSON file (default: {MINERS_METADATA_FILE})",
    )
    parser.add_argument(
        "--miners-dir",
        type=str,
        default=MINERS_DATA_DIR,
        help=f"Directory containing miner data files (default: {MINERS_DATA_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/data/hf_cache",
        help="Directory to store downloaded model files.",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default="~/data/eval_data",
        help="Directory to store evaluation datasets.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory to store training data for all miners (default: temporary directory). Each miner will use a subdirectory.",
    )

    args = parser.parse_args()

    # Initialize logging (without bittensor config requirements)
    try:
        # Try to initialize with a minimal config
        class MinimalConfig:
            pass
        config = MinimalConfig()
        bt.logging(config=config)
    except Exception:
        # If that fails, just initialize without config
        bt.logging()
    
    # Load miners metadata
    bt.logging.info(f"Loading miners metadata from: {args.metadata_file}")
    try:
        metadata = load_miners_metadata(args.metadata_file)
    except Exception as e:
        bt.logging.error(f"Failed to load miners metadata: {e}")
        return 1
    
    # Expand user paths
    if args.cache_dir and args.cache_dir.startswith("~"):
        args.cache_dir = os.path.expanduser(args.cache_dir)
    if args.eval_data_dir and args.eval_data_dir.startswith("~"):
        args.eval_data_dir = os.path.expanduser(args.eval_data_dir)
    if args.miners_dir and args.miners_dir.startswith("~"):
        args.miners_dir = os.path.expanduser(args.miners_dir)
    
    # Get competition parameters (same as validator)
    competition = Competition.from_defaults()
    eval_namespace = competition.repo
    
    bt.logging.info(f"Competition parameters:")
    bt.logging.info(f"  Benchmark loss: {competition.bench}")
    bt.logging.info(f"  Expected rows: {competition.rows}")
    bt.logging.info(f"  Eval namespace: {eval_namespace}")
    bt.logging.info(f"  Runs per miner: {NUM_RUNS_PER_MINER}")
    
    # Download eval dataset once (same as validator) - shared for all miners
    eval_data_dir = args.eval_data_dir
    bt.logging.info(
        f"Downloading eval dataset: {eval_namespace}/{constants.eval_commit}"
    )
    download_dataset(
        eval_namespace,
        constants.eval_commit,
        local_dir=eval_data_dir,
        cache_dir=args.cache_dir,
    )
    os.makedirs(eval_data_dir, exist_ok=True)
    for fname in os.listdir(eval_data_dir):
        if fname.endswith(".jsonl"):
            src = os.path.join(eval_data_dir, fname)
            dst = os.path.join(eval_data_dir, "data.jsonl")
            if src != dst:
                os.replace(src, dst)
                bt.logging.info(f"Renamed {fname} → data.jsonl")
    
    # Set up GPU (same as validator.py)
    torch.backends.cudnn.benchmark = True
    
    # Determine base data directory
    use_temp = args.data_dir is None
    if use_temp:
        temp_base = tempfile.mkdtemp(prefix="flock_loss_")
        base_data_dir = temp_base
        bt.logging.info(f"Using temporary directory: {temp_base}")
    else:
        base_data_dir = os.path.expanduser(args.data_dir)
        os.makedirs(base_data_dir, exist_ok=True)
    
    # Validate UIDs and find dataset paths
    miner_tasks = []
    for uid in UIDS_TO_PROCESS:
        dataset_path = get_dataset_path_for_uid(uid, metadata, args.miners_dir)
        if dataset_path:
            miner_tasks.append((uid, dataset_path))
            bt.logging.info(f"✓ UID {uid}: Found dataset at {dataset_path}")
        else:
            filename = get_miner_filename(uid, metadata)
            if filename:
                bt.logging.warning(f"✗ UID {uid}: Filename '{filename}' found in metadata but file not found in {args.miners_dir}")
            else:
                bt.logging.warning(f"✗ UID {uid}: Not found in miners metadata")
    
    if not miner_tasks:
        bt.logging.error("No valid miners found! Please check UIDs and ensure data files exist.")
        return 1
    
    bt.logging.info(f"Processing {len(miner_tasks)} miner(s)")
    
    # Display reference points for rank estimation
    print(f"\n{'=' * 60}")
    print("RANK ESTIMATION REFERENCE POINTS")
    print(f"{'=' * 60}")
    print("(Lower loss = better rank)")
    for rank, loss in RANK_REFERENCE_POINTS:
        print(f"  Rank {rank}: Loss {loss:.6f}")
    print(f"{'=' * 60}\n")
    
    # Process each miner with progress bar
    results = []
    output_file = "loss_res.txt"
    
    # Initialize progress bar
    pbar = tqdm(total=len(miner_tasks), desc="Processing miners", unit="miner")
    
    for i, (uid, dataset_path) in enumerate(miner_tasks, 1):
        dataset_filename = os.path.basename(dataset_path)
        pbar.set_description(f"Processing UID {uid}")
        
        bt.logging.info(f"\n{'=' * 60}")
        bt.logging.info(f"Processing miner {i}/{len(miner_tasks)}: UID {uid}")
        bt.logging.info(f"Dataset: {dataset_filename}")
        bt.logging.info(f"{'=' * 60}")
        
        loss, error = process_miner(
            uid=uid,
            dataset_path=dataset_path,
            eval_data_dir=eval_data_dir,
            cache_dir=args.cache_dir,
            competition=competition,
            base_data_dir=base_data_dir,
        )
        
        # Log results immediately after processing
        if error is None:
            estimated_rank = estimate_rank(loss)
            print(f"\n{'=' * 60}")
            print(f"✓ Completed: UID {uid} ({dataset_filename})")
            print(f"  Average Loss: {loss:.6f}")
            print(f"  Estimated Rank: {estimated_rank}")
            print(f"{'=' * 60}")
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss:.6f}",
                'rank': estimated_rank
            })
        else:
            print(f"\n{'=' * 60}")
            print(f"✗ Failed: UID {uid} ({dataset_filename})")
            print(f"  Error: {error}")
            print(f"{'=' * 60}")
        
        results.append({
            'uid': uid,
            'dataset_path': dataset_path,
            'dataset_filename': dataset_filename,
            'loss': loss,
            'error': error,
        })
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    
    # Output all results to file and console
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    
    success_count = 0
    output_lines = []
    
    for i, result in enumerate(results, 1):
        uid = result['uid']
        filename = result['dataset_filename']
        if result['error'] is None:
            loss_value = result['loss']
            estimated_rank = estimate_rank(loss_value)
            output_line = f"UID {uid} ({filename}): {loss_value:.6f}: {estimated_rank}"
            output_lines.append(output_line)
            print(f"\nMiner {i}: UID {uid} ({filename})")
            print(f"  Average Loss: {loss_value:.6f}")
            print(f"  Estimated Rank: {estimated_rank}")
            success_count += 1
        else:
            error_msg = result['error']
            output_line = f"UID {uid} ({filename}): ERROR - {error_msg}"
            output_lines.append(output_line)
            print(f"\nMiner {i}: UID {uid} ({filename})")
            print(f"  Error: {error_msg}")
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {success_count}/{len(results)} miners processed successfully")
    print(f"{'=' * 60}\n")
    
    # Write results to file (append mode)
    with open(output_file, 'a', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    bt.logging.info(f"Results appended to: {output_file}")
    print(f"Results appended to: {output_file}")
    
    # Cleanup temporary directory if used
    if use_temp:
        bt.logging.info(f"Cleaning up temporary directory: {temp_base}")
        shutil.rmtree(temp_base, ignore_errors=True)
    
    # Return 0 if all succeeded, 1 if any failed
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit(main())

