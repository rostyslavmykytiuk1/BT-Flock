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
import bittensor as bt
from flockoff.constants import Competition
from flockoff import constants
from flockoff.validator.trainer import (
    train_lora,
    download_dataset,
)
from dotenv import load_dotenv

load_dotenv()


def process_dataset(dataset_path, eval_data_dir, cache_dir, competition, lucky_num, base_data_dir=None):
    """Process a single dataset and return the loss."""
    dataset_path = os.path.expanduser(dataset_path)
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        bt.logging.error(f"Dataset path not found: {dataset_path}")
        return None, f"Path not found: {dataset_path}"
    
    # Determine data directory for this dataset
    if base_data_dir:
        # Use a subdirectory based on dataset filename
        dataset_name = os.path.basename(dataset_path)
        if dataset_name.endswith(".jsonl"):
            dataset_name = dataset_name[:-6]  # Remove .jsonl extension
        data_dir = os.path.join(base_data_dir, dataset_name)
    else:
        # Use parent directory of dataset path
        if os.path.isfile(dataset_path):
            data_dir = os.path.dirname(dataset_path)
        else:
            data_dir = dataset_path
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Handle dataset path - if it's a file, copy it to data_dir/data.jsonl
    # If it's a directory, check for data.jsonl inside
    if os.path.isfile(dataset_path):
        if dataset_path.endswith(".jsonl"):
            target_path = os.path.join(data_dir, "data.jsonl")
            if dataset_path != target_path:
                shutil.copy2(dataset_path, target_path)
                bt.logging.info(f"Copied dataset to: {target_path}")
            else:
                bt.logging.info(f"Using dataset at: {target_path}")
        else:
            bt.logging.error(f"Dataset file must be a .jsonl file: {dataset_path}")
            return None, f"Invalid file type: {dataset_path}"
    else:
        # It's a directory, check for data.jsonl
        data_jsonl_path = os.path.join(data_dir, "data.jsonl")
        if not os.path.exists(data_jsonl_path):
            bt.logging.error(f"data.jsonl not found in directory: {data_dir}")
            return None, f"data.jsonl not found in: {data_dir}"
        bt.logging.info(f"Using dataset at: {data_jsonl_path}")
    
    # Train and evaluate
    bt.logging.info(f"Starting LoRA training for: {dataset_path}")
    bt.logging.info(f"Using data directory: {data_dir}")
    
    try:
        eval_loss = train_lora(
            lucky_num,
            competition.bench,
            competition.rows,
            cache_dir=cache_dir,
            data_dir=data_dir,
            eval_data_dir=eval_data_dir,
        )
        bt.logging.info(f"Training complete with eval loss: {eval_loss}")
        return eval_loss, None
    except Exception as e:
        bt.logging.error(f"Training error for {dataset_path}: {e}")
        if "CUDA" in str(e):
            bt.logging.error("CUDA error detected")
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Compute loss for one or more datasets using the same logic, model, and resources as the validator"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to the dataset directory or data.jsonl file(s). Can specify multiple paths.",
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
        help="Base directory to store training data for all datasets (default: temporary directory). Each dataset will use a subdirectory.",
    )
    parser.add_argument(
        "--lucky-num",
        type=int,
        default=None,
        help="Random seed for training (default: random)",
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
    
    bt.logging.info("Starting loss computation")
    bt.logging.info(f"Processing {len(args.dataset_path)} dataset(s)")

    # Expand user paths
    if args.cache_dir and args.cache_dir.startswith("~"):
        args.cache_dir = os.path.expanduser(args.cache_dir)

    if args.eval_data_dir and args.eval_data_dir.startswith("~"):
        args.eval_data_dir = os.path.expanduser(args.eval_data_dir)

    # Get competition parameters (same as validator)
    competition = Competition.from_defaults()
    eval_namespace = competition.repo

    bt.logging.info(f"Competition parameters:")
    bt.logging.info(f"  Benchmark loss: {competition.bench}")
    bt.logging.info(f"  Expected rows: {competition.rows}")
    bt.logging.info(f"  Eval namespace: {eval_namespace}")

    # Download eval dataset once (same as validator) - shared for all datasets
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

    # Generate random seed if not provided (same as validator)
    if args.lucky_num is None:
        lucky_num = int.from_bytes(os.urandom(4), "little")
    else:
        lucky_num = args.lucky_num

    bt.logging.info(f"Using random seed: {lucky_num}")

    # Set up GPU
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

    # Process each dataset
    results = []
    for i, dataset_path in enumerate(args.dataset_path, 1):
        bt.logging.info(f"\n{'=' * 60}")
        bt.logging.info(f"Processing dataset {i}/{len(args.dataset_path)}: {dataset_path}")
        bt.logging.info(f"{'=' * 60}")
        
        loss, error = process_dataset(
            dataset_path=dataset_path,
            eval_data_dir=eval_data_dir,
            cache_dir=args.cache_dir,
            competition=competition,
            lucky_num=lucky_num,
            base_data_dir=base_data_dir,
        )
        
        results.append({
            'dataset_path': dataset_path,
            'loss': loss,
            'error': error,
        })

    # Output all results
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    
    success_count = 0
    for i, result in enumerate(results, 1):
        print(f"\nDataset {i}: {result['dataset_path']}")
        if result['error'] is None:
            print(f"  Loss: {result['loss']:.6f}")
            success_count += 1
        else:
            print(f"  Error: {result['error']}")
    
    print(f"\n{'=' * 60}")
    print(f"Summary: {success_count}/{len(results)} datasets processed successfully")
    print(f"{'=' * 60}\n")

    # Cleanup temporary directory if used
    if use_temp:
        bt.logging.info(f"Cleaning up temporary directory: {temp_base}")
        shutil.rmtree(temp_base, ignore_errors=True)

    # Return 0 if all succeeded, 1 if any failed
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit(main())

