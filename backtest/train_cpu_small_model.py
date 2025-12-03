#!/usr/bin/env python3
"""
Train a small model on CPU to directly measure dataset quality.

This script:
1. Uses a small CPU-friendly model (GPT-2 or TinyLlama)
2. For EACH miner separately:
   - Loads a FRESH untrained model from scratch
   - Trains LoRA on that miner's dataset only
   - Measures validation loss (lower = better dataset)
   - Discards the model completely
3. Ranks miners by loss and compares with emission rank

IMPORTANT: Each miner is evaluated independently with a fresh model.
No training state is carried over between miners.

Key concepts:
- Epoch: One full pass through the training dataset (e.g., 1 epoch = see all 250 examples once)
- Learning Rate: How big steps the model takes during training
  - Starts at initial value (e.g., 5e-5)
  - Decays linearly to 0 by end of training (this is normal!)
  - The decreasing LR you see in logs is expected behavior

This is the most direct approach - it measures what validators actually do,
just with a smaller model that can run on CPU.

Usage:
    python3 backtest/train_cpu_small_model.py
"""

import os
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
import gc

# Try to import training dependencies
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from flockoff.validator.dataset import SFTDataset, SFTDataCollator
    from flockoff.validator.constants import model2template
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False
    print("Warning: Training dependencies not available")

# Small CPU-friendly models
SMALL_MODELS = [
    "gpt2",  # ~124M parameters, very fast on CPU
    "gpt2-medium",  # ~355M parameters, still CPU-friendly
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ~1.1B, might be slow on CPU
]

# Default to GPT-2 (smallest, fastest)
DEFAULT_MODEL = "gpt2"


def normalize_json_item(item: Dict) -> str:
    """Normalize JSON item for comparison."""
    return json.dumps(item, sort_keys=True)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    if not Path(file_path).exists():
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data


def save_temp_jsonl(data: List[Dict], filepath: str):
    """Save data to temporary JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def train_small_model_cpu(
    miner_data: List[Dict],
    eval_data: List[Dict],
    model_name: str = DEFAULT_MODEL,
    eval_size: int = 250,  # Normalize dataset size (like validator does)
    num_epochs: int = 2,  # Number of epochs (like validator: 2 epochs)
    batch_size: int = 2,
    max_length: int = 512,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    learning_rate: float = 5e-5
) -> Optional[float]:
    """
    Train a small model on CPU and return validation loss.
    
    Matches validator's training approach:
    1. Load FRESH untrained model from scratch
    2. Normalize dataset size to eval_size (prune if larger)
    3. Train for num_epochs epochs (same for ALL miners)
    4. Evaluate on eval_data
    5. Return validation loss
    6. Model is discarded (not saved)
    
    Args:
        miner_data: Training data (miner's dataset)
        eval_data: Evaluation data (fixed eval set)
        model_name: Model to use (default: gpt2)
        eval_size: Maximum dataset size (prune to this size, like validator)
        num_epochs: Number of training epochs (same for ALL miners, like validator: 2)
        batch_size: Batch size
        max_length: Maximum sequence length
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        learning_rate: Initial learning rate (will decay linearly to 0 over epochs)
    
    Returns:
        Validation loss (lower = better), or None if training fails
    """
    if not HAS_TRAINING_DEPS:
        return None
    
    try:
        # Set device to CPU
        device = torch.device("cpu")
        torch.set_num_threads(4)  # Use multiple CPU threads
        
        # IMPORTANT: We load a FRESH model from scratch for each miner
        # This ensures each miner's dataset is evaluated independently
        # No previous training state is carried over
        
        # Get template for model (use GPT-2 simple template if not in model2template)
        if model_name in model2template:
            template = model2template[model_name]
        else:
            # Simple template for GPT-2 (no special formatting)
            template = {
                "system_format": None,
                "user_format": "{content}",
                "assistant_format": "{content}",
                "system": None
            }
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=None
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load FRESH model from scratch (untrained base model)
        print(f"      Loading {model_name} on CPU (fresh model, no previous training)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            cache_dir=None
        )
        
        # LoRA config for faster training
        # GPT-2 modules: c_attn, c_proj, c_fc
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-2 specific
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model = model.to(device)
        
        # Create temporary files for datasets
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = os.path.join(tmpdir, "train.jsonl")
            eval_file = os.path.join(tmpdir, "eval.jsonl")
            
            # Normalize dataset size (like validator does)
            # If miner has more than eval_size examples, prune to eval_size
            normalized_miner_data = miner_data[:eval_size] if len(miner_data) > eval_size else miner_data
            
            save_temp_jsonl(normalized_miner_data, train_file)
            save_temp_jsonl(eval_data[:100], eval_file)  # Use subset for speed
            
            # Prepare datasets using SFTDataset
            train_dataset = SFTDataset(
                file=train_file,
                tokenizer=tokenizer,
                max_seq_length=max_length,
                template=template
            )
            
            # Ensure dataset is exactly eval_size (like validator does)
            if len(train_dataset) > eval_size:
                print(f"      Pruning dataset from {len(train_dataset)} to {eval_size} examples (like validator)")
                train_dataset.data_list = train_dataset.data_list[:eval_size]
            
            eval_dataset = SFTDataset(
                file=eval_file,
                tokenizer=tokenizer,
                max_seq_length=max_length,
                template=template
            )
            
            data_collator = SFTDataCollator(tokenizer, max_length)
            
            # TrainingArguments matching validator's approach
            # Validator uses: num_train_epochs=2, normalizes dataset size to eval_size
            # This ensures:
            # 1. Same dataset size for all miners (normalized to eval_size)
            # 2. Same number of epochs for all miners (num_epochs)
            # 3. Same learning rate schedule (linear decay over epochs)
            # 4. Fair comparison between miners
            
            training_args = TrainingArguments(
                output_dir=os.path.join(tmpdir, "output"),
                num_train_epochs=num_epochs,  # Fixed epochs for ALL miners (like validator: 2 epochs)
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,  # Initial LR (same for all miners)
                lr_scheduler_type="linear",  # Linear decay: starts at learning_rate, ends at 0 after num_epochs
                warmup_steps=0,  # No warmup (start at full learning rate)
                logging_steps=10,
                eval_strategy="epoch",  # Evaluate at end of each epoch (like validator)
                save_strategy="no",  # Don't save checkpoints
                report_to="none",
                bf16=False,  # Use float32 for CPU
                fp16=False,  # Use float32 for CPU
                dataloader_num_workers=0,  # Avoid multiprocessing issues
                remove_unused_columns=False,  # Preserve target_mask
            )
            
            # Use regular Trainer with SFTDataCollator (preserves target_mask)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train
            print(f"      Training on {len(train_dataset)} examples for {num_epochs} epochs...")
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get('eval_loss', None)
        
        # IMPORTANT: Cleanup - discard model and trainer completely
        # This ensures no state is carried over to the next miner
        del model
        del trainer
        gc.collect()
        
        return float(eval_loss) if eval_loss is not None else None
        
    except Exception as e:
        print(f"      Error during training: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        gc.collect()
        return None


# ============================================================================
# CONFIGURABLE: List of emission ranks to train
# Edit this list to specify which ranked miners to train
# Example: [1, 20, 40, 60, 80, 100, 120, 140, 160, 180] trains top 10 miners
# ============================================================================
RANKS_TO_TRAIN = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180]


def main():
    parser = argparse.ArgumentParser(
        description="Train small model on CPU to measure dataset quality"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="backtest/eval_data.jsonl",
        help="Path to eval dataset (default: backtest/eval_data.jsonl)"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="backtest/miners_metadata.json",
        help="Path to miners metadata JSON file (default: backtest/miners_metadata.json)"
    )
    parser.add_argument(
        "--miners-dir",
        type=str,
        default="backtest/hf_datajsonl",
        help="Directory with miners' data files (default: backtest/hf_datajsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=SMALL_MODELS,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=250,
        help="Maximum dataset size (prune to this size, like validator). Default: 250"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs (same for ALL miners, like validator). Default: 2"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of miners to evaluate (for testing, default: all)"
    )
    parser.add_argument(
        "--train-ranks",
        type=str,
        default=None,
        help="Comma-separated list of emission ranks to train (e.g., '1,20,40,60'). If not set, uses RANKS_TO_TRAIN from code"
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train all miners (ignores --train-ranks and RANKS_TO_TRAIN)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest/cpu_training_results.json",
        help="Output file for results (default: backtest/cpu_training_results.json)"
    )
    
    args = parser.parse_args()
    
    if not HAS_TRAINING_DEPS:
        print("ERROR: Training dependencies not available!")
        print("Required: transformers, peft, trl")
        return 1
    
    print("=" * 70)
    print("TRAIN SMALL MODEL ON CPU TO MEASURE DATASET QUALITY")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset size: {args.eval_size} (normalized, like validator)")
    print(f"Training epochs: {args.num_epochs} (same for all miners, like validator)")
    print(f"  → All miners train on same dataset size ({args.eval_size} examples)")
    print(f"  → All miners train for same epochs ({args.num_epochs} epochs)")
    print(f"  → Ensures fair comparison (matches validator approach)")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    
    # Load eval data
    print(f"\n1. Loading eval dataset: {args.eval_data}")
    eval_data = load_jsonl(args.eval_data)
    if not eval_data:
        print("ERROR: Could not load eval dataset!")
        return 1
    print(f"   ✓ Loaded {len(eval_data):,} rows from eval dataset")
    
    # Load miner metadata
    print(f"\n2. Loading miner metadata: {args.metadata_file}")
    if not Path(args.metadata_file).exists():
        print(f"ERROR: Metadata file not found: {args.metadata_file}")
        return 1
    
    with open(args.metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    miner_emissions = metadata.get('miner_emissions', {})
    filename_to_hotkey = metadata.get('filename_to_hotkey', {})
    
    print(f"   ✓ Loaded metadata for {len(miner_emissions)} miners")
    
    # Rank miners by emission (higher emission = better rank)
    print(f"\n3. Ranking miners by emission...")
    ranked_miners = sorted(
        [(hotkey, emission) for hotkey, emission in miner_emissions.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create rank map: rank -> (hotkey, emission)
    rank_to_miner = {i + 1: (hotkey, emission) for i, (hotkey, emission) in enumerate(ranked_miners)}
    
    print(f"   ✓ Ranked {len(ranked_miners)} miners by emission")
    print(f"   Top 5: {[f'Rank {i+1}' for i in range(min(5, len(ranked_miners)))]}")
    
    # Determine which ranks to train
    if args.train_all:
        ranks_to_train = list(range(1, len(ranked_miners) + 1))
        print(f"\n4. Training ALL {len(ranks_to_train)} miners (--train-all specified)")
    elif args.train_ranks:
        ranks_to_train = [int(r.strip()) for r in args.train_ranks.split(',')]
        print(f"\n4. Training miners at ranks: {ranks_to_train} (from --train-ranks)")
    else:
        ranks_to_train = RANKS_TO_TRAIN
        print(f"\n4. Training miners at ranks: {ranks_to_train} (from RANKS_TO_TRAIN in code)")
        print(f"   💡 Edit RANKS_TO_TRAIN in the code to change which ranks to train")
    
    # Filter to only miners at specified ranks
    miners_to_train = {}
    for rank in ranks_to_train:
        if rank in rank_to_miner:
            hotkey, emission = rank_to_miner[rank]
            miners_to_train[hotkey] = {
                'rank': rank,
                'emission': emission,
                'hotkey': hotkey
            }
        else:
            print(f"   ⚠️  Rank {rank} not found (only {len(ranked_miners)} miners available)")
    
    print(f"   ✓ Will train {len(miners_to_train)} miners")
    
    # Score each miner's dataset
    print(f"\n5. Training model on selected miners' datasets...")
    print(f"   ⚠️  This will take a while (CPU training is slow)")
    print(f"   Each miner takes ~1-5 minutes depending on dataset size")
    
    miner_scores = []
    miner_files = list(Path(args.miners_dir).glob("*.jsonl"))
    
    if args.max_samples:
        miner_files = miner_files[:args.max_samples]
        print(f"   Limiting to {args.max_samples} miners for testing")
    
    processed = 0
    failed = 0
    
    for miner_file in miner_files:
        filename = miner_file.name
        hotkey = filename_to_hotkey.get(filename)
        
        if not hotkey or hotkey not in miner_emissions:
            continue
        
        # Only train miners at specified ranks
        if hotkey not in miners_to_train:
            continue
        
        miner_info = miners_to_train[hotkey]
        rank = miner_info['rank']
        emission = miner_info['emission']
        
        miner_data = load_jsonl(str(miner_file))
        if not miner_data or len(miner_data) < 10:
            print(f"\n   [SKIP] Rank {rank}: {filename[:30]}... (insufficient data)")
            continue
        
        print(f"\n   [{processed + 1}/{len(miners_to_train)}] Rank {rank}: {filename[:30]}...")
        print(f"      Emission: {emission:.6f}")
        
        # Train and evaluate
        # Process: Fresh model → Normalize dataset size → Train for fixed epochs → Get loss → Discard model
        # Matches validator's approach exactly
        eval_loss = train_small_model_cpu(
            miner_data,
            eval_data,
            model_name=args.model,
            eval_size=args.eval_size,  # Normalize dataset size (like validator)
            num_epochs=args.num_epochs,  # Fixed epochs for ALL miners (like validator: 2)
            batch_size=args.batch_size
        )
        
        if eval_loss is None:
            failed += 1
            print(f"      ✗ Training failed, skipping")
            continue
        
        miner_info_result = {
            "hotkey": hotkey,
            "filename": filename,
            "rank": rank,
            "emission": float(emission),
            "eval_loss": float(eval_loss),
            "data_rows": len(miner_data)
        }
        miner_scores.append(miner_info_result)
        
        processed += 1
        print(f"      ✓ Loss: {eval_loss:.4f}, Emission: {emission:.6f}")
        
        if processed % 10 == 0:
            print(f"\n   Progress: {processed} successful, {failed} failed")
    
    print(f"\n   ✓ Completed: {processed} miners trained, {failed} failed")
    if miner_scores:
        print(f"   Trained miners at ranks: {sorted([m['rank'] for m in miner_scores])}")
    
    if not miner_scores:
        print("ERROR: No miners successfully evaluated!")
        return 1
    
    # Rank and compare
    print(f"\n4. Ranking miners by validation loss (lower = better)...")
    
    # Sort by loss (lower is better)
    ranked_by_loss = sorted(miner_scores, key=lambda x: x['eval_loss'])
    for i, miner in enumerate(ranked_by_loss):
        miner['loss_rank'] = i + 1  # Rank 1 = best (lowest loss)
    
    # Sort by emission and add emission ranks
    ranked_by_emission = sorted(miner_scores, key=lambda x: x['emission'], reverse=True)
    emission_rank_map = {miner['hotkey']: i + 1 for i, miner in enumerate(ranked_by_emission)}
    
    for miner in miner_scores:
        miner['emission_rank'] = emission_rank_map[miner['hotkey']]
        miner['rank_difference'] = miner['loss_rank'] - miner['emission_rank']
    
    # Calculate correlations
    # Note: Lower loss = better, so we need to invert for correlation
    loss_ranks = [m['loss_rank'] for m in ranked_by_loss]
    emission_ranks = [m['emission_rank'] for m in ranked_by_loss]
    
    # For correlation: lower loss rank should correlate with higher emission rank
    # So we correlate loss_rank with emission_rank (both are ranks, 1 = best)
    spearman_corr, spearman_p = spearmanr(loss_ranks, emission_ranks)
    
    # For Pearson: invert loss (higher loss = worse, so we want negative correlation)
    losses = [m['eval_loss'] for m in ranked_by_loss]
    emissions = [m['emission'] for m in ranked_by_loss]
    pearson_corr, pearson_p = pearsonr(losses, emissions)
    # Invert: lower loss should correlate with higher emission
    # So we expect negative correlation
    pearson_corr = -pearson_corr  # Invert since lower loss = better
    
    # Save results
    results = {
        "model": args.model,
        "eval_size": args.eval_size,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "num_miners": len(miner_scores),
        "correlations": {
            "loss_rank_vs_emission_rank": {
                "spearman": float(spearman_corr),
                "spearman_p": float(spearman_p),
            },
            "loss_vs_emission": {
                "pearson": float(pearson_corr),
                "pearson_p": float(pearson_p),
                "note": "Inverted - lower loss should correlate with higher emission"
            }
        },
        "ranked_miners": ranked_by_loss
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Display results
    print(f"\n" + "=" * 70)
    print("CORRELATION WITH EMISSION RANK")
    print("=" * 70)
    
    print(f"\nLoss Rank vs Emission Rank:")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"  (Higher correlation = better prediction)")
    
    print(f"\nLoss vs Emission (inverted):")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
    print(f"  (Positive correlation = lower loss → higher emission)")
    
    # Show top 20
    print(f"\n" + "=" * 70)
    print("TOP 20 MINERS (Ranked by Validation Loss - Lower is Better)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Loss Rank':<12} {'Emission Rank':<14} {'Diff':<8} {'Loss':<12} {'Emission':<10}")
    print("-" * 70)
    for i, miner in enumerate(ranked_by_loss[:20]):
        print(f"{i+1:<6} {miner['loss_rank']:<12} {miner['emission_rank']:<14} {miner['rank_difference']:<8} "
              f"{miner['eval_loss']:<12.4f} {miner['emission']:<10.6f}")
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model used: {args.model}")
    print(f"Total miners evaluated: {len(miner_scores)}")
    print(f"Failed: {failed}")
    print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"Results saved to: {args.output}")
    print("=" * 70)
    
    if spearman_corr > 0.5:
        print(f"\n✓ Strong correlation! CPU training approach works well!")
    elif spearman_corr > 0.3:
        print(f"\n⚠️  Moderate correlation - may need more epochs or different model")
    else:
        print(f"\n⚠️  Low correlation - may need to adjust training parameters")
    
    return 0


if __name__ == "__main__":
    exit(main())

