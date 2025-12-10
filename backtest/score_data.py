#!/usr/bin/env python3
"""
Score your data locally by simulating the validator's training and evaluation process.

This script:
1. Takes your data.jsonl file
2. Uses the existing evaluation dataset (backtest/eval_data.jsonl)
3. Trains a LoRA model on your data
4. Evaluates it and returns the score (same as validators use)

Usage:
    python3 backtest/score_data.py
    python3 backtest/score_data.py --data backtest/data.jsonl
    python3 backtest/score_data.py --data backtest/data.jsonl --eval-data backtest/eval_data.jsonl
"""

import os
import sys
import argparse
import shutil
import tempfile
from pathlib import Path

# Add parent directory to path to import flockoff modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import bittensor as bt
from flockoff.constants import Competition
from flockoff.validator.trainer import train_lora
from flockoff.validator.validator_utils import compute_score
from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    """Configure logging for the script"""
    # Initialize bittensor logging - it will work with defaults
    try:
        # Try to initialize with a minimal config
        class MinimalConfig:
            pass
        config = MinimalConfig()
        bt.logging(config=config)
    except Exception:
        # If that fails, just initialize without config
        bt.logging()


def main():
    parser = argparse.ArgumentParser(
        description="Score your data locally using the validator's training process"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="backtest/data.jsonl",
        help="Path to your data.jsonl file (default: backtest/data.jsonl)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="backtest/eval_data.jsonl",
        help="Path to evaluation data.jsonl file (default: backtest/eval_data.jsonl)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to store training data (default: temporary directory)",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default=None,
        help="Directory to store evaluation data (default: temporary directory)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directories after completion (for debugging)",
    )
    parser.add_argument(
        "--lucky-num",
        type=int,
        default=None,
        help="Random seed for training (default: random)",
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Expand user paths
    data_path = os.path.expanduser(args.data)
    eval_data_path = os.path.expanduser(args.eval_data)
    
    # Check if data files exist
    if not os.path.exists(data_path):
        bt.logging.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    if not os.path.exists(eval_data_path):
        bt.logging.error(f"Evaluation data file not found: {eval_data_path}")
        sys.exit(1)
    
    # Get competition parameters
    competition = Competition.from_defaults()
    bt.logging.info(f"Competition parameters:")
    bt.logging.info(f"  Benchmark loss: {competition.bench}")
    bt.logging.info(f"  Min benchmark: {competition.minb}")
    bt.logging.info(f"  Max benchmark: {competition.maxb}")
    bt.logging.info(f"  Expected rows: {competition.rows}")
    
    # Create temporary directories if not specified
    use_temp = args.data_dir is None or args.eval_data_dir is None
    if use_temp:
        temp_base = tempfile.mkdtemp(prefix="flock_score_")
        bt.logging.info(f"Using temporary directory: {temp_base}")
        data_dir = args.data_dir or os.path.join(temp_base, "train_data")
        eval_data_dir = args.eval_data_dir or os.path.join(temp_base, "eval_data")
    else:
        data_dir = os.path.expanduser(args.data_dir)
        eval_data_dir = os.path.expanduser(args.eval_data_dir)
    
    try:
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(eval_data_dir, exist_ok=True)
        
        # Copy user's data to training directory
        train_data_path = os.path.join(data_dir, "data.jsonl")
        bt.logging.info(f"Copying your data to: {train_data_path}")
        shutil.copy2(data_path, train_data_path)
        
        # Count rows in user's data
        with open(train_data_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f)
        bt.logging.info(f"Your dataset has {row_count} rows")
        
        # Copy evaluation dataset to eval directory
        eval_data_dest = os.path.join(eval_data_dir, "data.jsonl")
        bt.logging.info(f"Using evaluation dataset: {eval_data_path}")
        shutil.copy2(eval_data_path, eval_data_dest)
        
        # Generate random seed if not provided
        if args.lucky_num is None:
            import random
            lucky_num = random.randint(0, 2**32 - 1)
        else:
            lucky_num = args.lucky_num
        
        bt.logging.info(f"Using random seed: {lucky_num}")
        
        # Train and evaluate
        bt.logging.info("=" * 60)
        bt.logging.info("Starting LoRA training and evaluation...")
        bt.logging.info("=" * 60)
        bt.logging.info("This may take several minutes depending on your GPU...")
        
        raw_score = train_lora(
            lucky_num=lucky_num,
            benchmark_loss=competition.bench,
            eval_size=competition.rows,
            cache_dir=None,
            data_dir=data_dir,
            eval_data_dir=eval_data_dir,
        )
        
        bt.logging.info("=" * 60)
        
        # Check if training actually succeeded
        # If raw_score equals benchmark_loss, it likely means training failed
        # (train_lora returns benchmark_loss as fallback on error)
        if abs(raw_score - competition.bench) < 1e-6:
            print("\n" + "=" * 60)
            print("⚠️  TRAINING FAILED")
            print("=" * 60)
            print("The training process encountered an error and returned the")
            print("benchmark loss as a fallback value. This is NOT a real score.")
            print()
            print("Common causes:")
            print("  - CUDA/GPU not available (required for training)")
            print("  - Out of memory")
            print("  - Model download failed")
            print("  - Other training errors")
            print()
            print("Please check the error messages above for details.")
            print("=" * 60)
            sys.exit(1)
        
        bt.logging.info("TRAINING COMPLETE")
        bt.logging.info("=" * 60)
        
        # Compute normalized score
        normalized_score = compute_score(
            loss=raw_score,
            benchmark_loss=competition.bench,
            min_bench=competition.minb,
            max_bench=competition.maxb,
            power=competition.pow,
            bench_height=competition.bheight,
            miner_comp_id=competition.id,
            real_comp_id=competition.id,
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Raw Score (Eval Loss):     {raw_score:.6f}")
        print(f"Normalized Score:           {normalized_score:.6f}")
        print(f"Normalized Score (%):       {normalized_score * 100:.2f}%")
        print()
        print("Score Interpretation:")
        print(f"  Benchmark loss:           {competition.bench:.6f}")
        print(f"  Min benchmark:            {competition.minb:.6f}")
        print(f"  Max benchmark:            {competition.maxb:.6f}")
        if raw_score < competition.minb:
            print("  ✓ Your score is EXCELLENT (below min benchmark)")
        elif raw_score <= competition.bench:
            print("  ✓ Your score is GOOD (below or equal to benchmark)")
        elif raw_score <= competition.maxb:
            print("  ⚠ Your score is ACCEPTABLE (above benchmark but below max)")
        else:
            print("  ✗ Your score is POOR (above max benchmark)")
        print("=" * 60)
        
        # Cleanup temporary directories
        if use_temp and not args.keep_temp:
            bt.logging.info(f"Cleaning up temporary directory: {temp_base}")
            shutil.rmtree(temp_base, ignore_errors=True)
        elif args.keep_temp:
            bt.logging.info(f"Keeping temporary directory: {temp_base}")
        
    except KeyboardInterrupt:
        bt.logging.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        bt.logging.error(f"Error during scoring: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

