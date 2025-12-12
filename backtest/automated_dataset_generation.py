#!/usr/bin/env python3
"""
Automated dataset generation workflow.

This script combines:
1. Download miners' data (from download_miners_data.py)
2. Calculate row weights (from calculate_row_weights.py)
3. Generate one top dataset with 250 rows (from select_max_weight_data.py)
4. Calculate eval loss (from compute_loss.py)
5. Send results to Discord webhook
6. Runs forever, continuously generating datasets and reporting results

Usage:
    python3 backtest/automated_dataset_generation.py --netuid 96
"""

import os
import sys
import json
import argparse
import shutil
import tempfile
import time
import requests
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
from dotenv import load_dotenv

import bittensor as bt
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError

from flockoff.constants import Competition
from flockoff import constants
from flockoff.validator.trainer import train_lora, download_dataset
from flockoff.miners.data import ModelId, ModelMetadata

# Import functions from existing scripts
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backtest.select_max_weight_data import (
        normalize_json_item,
        load_jsonl,
        select_max_weight_rows,
    )
    from backtest.calculate_row_weights import (
        normalize_emission,
        calculate_row_weight,
    )
except ImportError:
    # Fallback: define functions inline if imports fail
    def normalize_json_item(item: Dict) -> str:
        return json.dumps(item, sort_keys=True)
    
    def load_jsonl(file_path: str) -> List[Dict]:
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

load_dotenv()

# Constants
MAX_DUPLICATES_PER_MINER = 100
MIN_WEIGHT_THRESHOLD = 0.0
WEIGHT_EXPONENT = 2.5
TOP_N_MINERS = 100
MIN_EMISSION_THRESHOLD = 0.5
MAX_EMISSION_THRESHOLD = 0.7
MAX_EMISSION_THRESHOLD_FOR_VALIDATORS = 10.0

# Rank estimation reference points
RANK_REFERENCE_POINTS = [
    (1, 2.414466),
    (41, 2.416089),
    (81, 2.416649),
    (121, 2.416813),
    (161, 2.416864),
]

# Output directories
BASE_OUT_DIR = "backtest/hf_datajsonl"
OUTPUT_METADATA_FILE = "backtest/miners_metadata.json"
OUTPUT_WEIGHTS_FILE = "backtest/row_weights.json"
OUTPUT_DIR = Path("backtest/generated_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_rank(loss: float) -> int:
    """Estimate rank based on loss value using linear interpolation."""
    if loss <= RANK_REFERENCE_POINTS[0][1]:
        return 1
    
    if loss >= RANK_REFERENCE_POINTS[-1][1]:
        last_rank, last_loss = RANK_REFERENCE_POINTS[-1]
        second_last_rank, second_last_loss = RANK_REFERENCE_POINTS[-2]
        rank_diff = last_rank - second_last_rank
        loss_diff = last_loss - second_last_loss
        
        if loss_diff > 0:
            loss_beyond = loss - last_loss
            rank_increase = int((loss_beyond / loss_diff) * rank_diff)
            return last_rank + rank_increase
        else:
            return last_rank + 1
    
    for i in range(len(RANK_REFERENCE_POINTS) - 1):
        rank1, loss1 = RANK_REFERENCE_POINTS[i]
        rank2, loss2 = RANK_REFERENCE_POINTS[i + 1]
        
        if loss1 <= loss <= loss2:
            if loss2 == loss1:
                return rank1
            loss_ratio = (loss - loss1) / (loss2 - loss1)
            estimated_rank = rank1 + (rank2 - rank1) * loss_ratio
            return int(round(estimated_rank))
    
    return 161


def send_discord_message(webhook_url: str, message: str) -> bool:
    """Send message to Discord webhook."""
    try:
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"   ⚠️  Failed to send Discord message: {e}")
        return False


def download_miners_data(subtensor: bt.subtensor, netuid: int) -> Dict:
    """Download miners' data exactly like download_miners_data.py."""
    print(f"\n{'=' * 70}")
    print("STEP 1: DOWNLOADING MINERS DATA")
    print(f"{'=' * 70}")
    
    # Clear existing data
    if os.path.exists(BASE_OUT_DIR):
        print(f"   Clearing existing data in {BASE_OUT_DIR}...")
        for filename in os.listdir(BASE_OUT_DIR):
            file_path = os.path.join(BASE_OUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"     Error removing {filename}: {e}")
    
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    
    # Get metagraph
    metagraph = subtensor.metagraph(netuid)
    metagraph.sync(subtensor=subtensor)
    current_uids = metagraph.uids.tolist()
    print(f"   ✓ Connected. Found {len(current_uids)} UIDs")
    
    # Get miner emissions
    miner_emissions = {}
    emissions = metagraph.E.copy() if hasattr(metagraph.E, 'copy') else metagraph.E
    uids = metagraph.uids.tolist()
    hotkeys = metagraph.hotkeys
    
    for i, uid in enumerate(uids):
        if uid != 4294967295:
            emission = float(emissions[i])
            hotkey = hotkeys[i]
            miner_emissions[hotkey] = emission
    
    print(f"   ✓ Found emissions for {len(miner_emissions)} miners")
    
    # Download data files
    print(f"   Downloading miners' data files...")
    miners_metadata_list = []
    downloaded_count = 0
    filename_to_hotkey = {}
    
    for uid_i in current_uids:
        hotkey = metagraph.hotkeys[uid_i]
        emission = miner_emissions.get(hotkey, 0.0)
        
        # Skip validators
        if emission > MAX_EMISSION_THRESHOLD_FOR_VALIDATORS:
            continue
        
        # Get metadata
        try:
            metadata = bt.core.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)
            if not metadata:
                continue
            
            commitment = metadata["info"]["fields"][0]
            if isinstance(commitment, tuple) and len(commitment) > 0 and isinstance(commitment[0], dict):
                raw_keys = [key for key in commitment[0].keys() if key.startswith("Raw")]
                if raw_keys:
                    raw_key = raw_keys[0]
                    raw_data = commitment[0][raw_key][0]
                    chain_str = "".join(chr(j) for j in raw_data)
                    
                    if not chain_str.startswith("{"):
                        try:
                            model_id = ModelId.from_compressed_str(chain_str)
                            repoid = model_id.namespace
                            revision = model_id.commit
                            
                            safe_repoid = repoid.replace("/", "_").replace("\\", "_")
                            safe_revision = revision.replace("/", "_").replace("\\", "_")
                            filename = f"{safe_repoid}_{safe_revision}.jsonl"
                            dest_path = os.path.join(BASE_OUT_DIR, filename)
                            
                            # Download if not exists
                            if not os.path.exists(dest_path):
                                try:
                                    cached_path = hf_hub_download(
                                        repo_id=repoid,
                                        filename="data.jsonl",
                                        revision=revision,
                                        repo_type="dataset",
                                    )
                                    shutil.copy2(cached_path, dest_path)
                                    downloaded_count += 1
                                except Exception as e:
                                    pass
                            
                            filename_to_hotkey[filename] = hotkey
                            
                            miner_info = {
                                "uid": int(uid_i),
                                "hotkey": hotkey,
                                "namespace": repoid,
                                "commit": revision,
                                "filename": filename,
                                "emission": emission,
                            }
                            miners_metadata_list.append(miner_info)
                        except Exception:
                            pass
        except Exception:
            continue
    
    # Filter to exclude validators
    filtered_miner_emissions = {
        hotkey: emission 
        for hotkey, emission in miner_emissions.items() 
        if emission <= MAX_EMISSION_THRESHOLD_FOR_VALIDATORS
    }
    
    filtered_filename_to_hotkey = {
        filename: hotkey
        for filename, hotkey in filename_to_hotkey.items()
        if miner_emissions.get(hotkey, 0.0) <= MAX_EMISSION_THRESHOLD_FOR_VALIDATORS
    }
    
    metadata = {
        "netuid": netuid,
        "block": int(metagraph.block.item() if hasattr(metagraph.block, 'item') else metagraph.block),
        "miner_emissions": filtered_miner_emissions,
        "filename_to_hotkey": filtered_filename_to_hotkey,
        "miners": miners_metadata_list
    }
    
    # Save metadata
    output_path = Path(OUTPUT_METADATA_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Downloaded {downloaded_count} data files")
    print(f"   ✓ Saved metadata for {len(miners_metadata_list)} miners")
    
    return metadata


def calculate_weights(eval_data: List[Dict], metadata: Dict) -> Dict[str, float]:
    """Calculate row weights exactly like calculate_row_weights.py."""
    print(f"\n{'=' * 70}")
    print("STEP 2: CALCULATING ROW WEIGHTS")
    print(f"{'=' * 70}")
    
    miner_emissions = metadata.get('miner_emissions', {})
    filename_to_hotkey = metadata.get('filename_to_hotkey', {})
    
    row_weights = calculate_row_weight(
        eval_data,
        BASE_OUT_DIR,
        miner_emissions,
        filename_to_hotkey,
        top_n=TOP_N_MINERS,
        min_emission_threshold=MIN_EMISSION_THRESHOLD,
        max_emission_threshold=MAX_EMISSION_THRESHOLD
    )
    
    # Save weights
    output_path = Path(OUTPUT_WEIGHTS_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(row_weights, f, indent=2)
    
    print(f"   ✓ Saved weights for {len(row_weights)} rows")
    
    return row_weights


def generate_dataset(eval_data: List[Dict], row_weights: Dict[str, float], metadata: Dict) -> str:
    """Generate one top dataset with 250 rows."""
    print(f"\n{'=' * 70}")
    print("STEP 3: GENERATING DATASET")
    print(f"{'=' * 70}")
    
    selected = select_max_weight_rows(
        eval_data,
        row_weights,
        250,  # num_rows
        BASE_OUT_DIR,  # miners_dir
        seed=None
    )
    
    # Verify no duplicates
    selected_normalized = set(normalize_json_item(item) for item in selected)
    if len(selected_normalized) != len(selected):
        seen = set()
        unique_selected = []
        for item in selected:
            normalized = normalize_json_item(item)
            if normalized not in seen:
                unique_selected.append(item)
                seen.add(normalized)
        selected = unique_selected
    
    # Generate timestamp and save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_filename = f"data_{timestamp}.jsonl"
    output_path = OUTPUT_DIR / output_filename
    
    with output_path.open('w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Generated dataset with {len(selected)} rows")
    print(f"   ✓ Saved to: {output_path}")
    
    return str(output_path)


def compute_loss(dataset_path: str, eval_data_dir: str, cache_dir: str, competition: Competition, lucky_num: int) -> Optional[float]:
    """Compute eval loss exactly like compute_loss.py."""
    dataset_path = os.path.expanduser(dataset_path)
    
    if not os.path.exists(dataset_path):
        print(f"   ⚠️  Dataset path not found: {dataset_path}")
        return None
    
    # Create temp data directory
    dataset_name = os.path.basename(dataset_path)
    if dataset_name.endswith(".jsonl"):
        dataset_name = dataset_name[:-6]
    
    temp_data_dir = tempfile.mkdtemp(prefix="flock_loss_")
    data_dir = os.path.join(temp_data_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy dataset to data_dir/data.jsonl
    target_path = os.path.join(data_dir, "data.jsonl")
    shutil.copy2(dataset_path, target_path)
    
    try:
        eval_loss = train_lora(
            lucky_num,
            competition.bench,
            competition.rows,
            cache_dir=cache_dir,
            data_dir=data_dir,
            eval_data_dir=eval_data_dir,
        )
        
        # Cleanup
        shutil.rmtree(temp_data_dir, ignore_errors=True)
        
        return eval_loss
    except Exception as e:
        print(f"   ⚠️  Training error: {e}")
        shutil.rmtree(temp_data_dir, ignore_errors=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Automated dataset generation workflow"
    )
    parser.add_argument("--netuid", type=int, default=96, help="The subnet UID.")
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
        "--eval-data",
        type=str,
        default="backtest/eval_data.jsonl",
        help="Path to eval dataset (default: backtest/eval_data.jsonl)"
    )
    parser.add_argument(
        "--lucky-num",
        type=int,
        default=None,
        help="Random seed for training (default: random)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations (default: None, runs forever)",
    )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    
    args = parser.parse_args()
    config = bt.config(parser)
    
    # Initialize logging
    try:
        bt.logging(config=config)
    except Exception:
        bt.logging()
    
    # Get Discord webhook URL from .env
    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not discord_webhook_url:
        print("⚠️  WARNING: DISCORD_WEBHOOK_URL not found in .env file")
        print("   Results will not be sent to Discord")
    
    # Expand user paths
    if args.cache_dir and args.cache_dir.startswith("~"):
        args.cache_dir = os.path.expanduser(args.cache_dir)
    if args.eval_data_dir and args.eval_data_dir.startswith("~"):
        args.eval_data_dir = os.path.expanduser(args.eval_data_dir)
    
    # Initialize subtensor
    print("=" * 70)
    print("AUTOMATED DATASET GENERATION WORKFLOW")
    print("=" * 70)
    
    subtensor = bt.subtensor(config=config)
    
    # Get competition parameters
    competition = Competition.from_defaults()
    eval_namespace = competition.repo
    
    # Download eval dataset once
    print(f"\n0. Downloading eval dataset: {eval_namespace}/{constants.eval_commit}")
    download_dataset(
        eval_namespace,
        constants.eval_commit,
        local_dir=args.eval_data_dir,
        cache_dir=args.cache_dir,
    )
    os.makedirs(args.eval_data_dir, exist_ok=True)
    for fname in os.listdir(args.eval_data_dir):
        if fname.endswith(".jsonl"):
            src = os.path.join(args.eval_data_dir, fname)
            dst = os.path.join(args.eval_data_dir, "data.jsonl")
            if src != dst:
                os.replace(src, dst)
    
    # Load eval data
    eval_data = load_jsonl(args.eval_data)
    if not eval_data:
        print("ERROR: Could not load eval dataset!")
        return 1
    
    print(f"   ✓ Loaded {len(eval_data):,} rows from eval dataset")
    
    # Generate random seed if not provided
    if args.lucky_num is None:
        lucky_num = int.from_bytes(os.urandom(4), "little")
    else:
        lucky_num = args.lucky_num
    
    # Set up GPU
    torch.backends.cudnn.benchmark = True
    
    iteration = 0
    while True:
        if args.max_iterations and iteration >= args.max_iterations:
            print(f"\n   Reached max iterations ({args.max_iterations}). Stopping.")
            break
        
        iteration += 1
        max_iter_str = f"/{args.max_iterations}" if args.max_iterations else ""
        print(f"\n{'=' * 70}")
        print(f"ITERATION {iteration}{max_iter_str}")
        print(f"{'=' * 70}")
        
        try:
            # Step 1: Download miners data
            metadata = download_miners_data(subtensor, config.netuid)
            
            # Step 2: Calculate row weights
            row_weights = calculate_weights(eval_data, metadata)
            
            # Step 3: Generate dataset
            dataset_path = generate_dataset(eval_data, row_weights, metadata)
            dataset_filename = os.path.basename(dataset_path)
            
            # Step 4: Compute loss
            print(f"\n{'=' * 70}")
            print("STEP 4: COMPUTING EVAL LOSS")
            print(f"{'=' * 70}")
            
            eval_loss = compute_loss(
                dataset_path,
                args.eval_data_dir,
                args.cache_dir,
                competition,
                lucky_num
            )
            
            if eval_loss is None:
                print(f"   ⚠️  Failed to compute loss, continuing to next iteration...")
                continue
            
            # Step 5: Estimate rank
            estimated_rank = estimate_rank(eval_loss)
            
            print(f"\n{'=' * 70}")
            print("RESULTS")
            print(f"{'=' * 70}")
            print(f"Dataset: {dataset_filename}")
            print(f"Eval Loss: {eval_loss:.6f}")
            print(f"Estimated Rank: {estimated_rank}")
            print(f"{'=' * 70}")
            
            # Step 6: Send to Discord
            if discord_webhook_url:
                message = f"{dataset_filename}: {eval_loss:.6f}: {estimated_rank}"
                print(f"\nSending result to Discord...")
                send_discord_message(discord_webhook_url, message)
            
            # Step 7: Report rank and continue
            if estimated_rank <= 60:
                print(f"\n✓ EXCELLENT! Rank {estimated_rank} is <= 60. Continuing to find even better results...")
                if discord_webhook_url:
                    success_msg = f"🎉 EXCELLENT! Dataset {dataset_filename} achieved rank {estimated_rank} (loss: {eval_loss:.6f}). Continuing..."
                    send_discord_message(discord_webhook_url, success_msg)
            else:
                print(f"\n⚠️  Rank {estimated_rank} is > 60. Continuing to next iteration...")
                if discord_webhook_url:
                    continue_msg = f"⚠️  Iteration {iteration}: Rank {estimated_rank} (loss: {eval_loss:.6f}). Continuing..."
                    send_discord_message(discord_webhook_url, continue_msg)
        
        except Exception as e:
            print(f"\n⚠️  Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            if discord_webhook_url:
                error_msg = f"❌ Error in iteration {iteration}: {str(e)[:200]}"
                send_discord_message(discord_webhook_url, error_msg)
            continue
    
    subtensor.close()
    
    print(f"\n{'=' * 70}")
    print("WORKFLOW COMPLETE")
    print(f"{'=' * 70}")
    
    return 0


if __name__ == "__main__":
    exit(main())

