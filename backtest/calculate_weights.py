#!/usr/bin/env python3
"""
Calculate weights for all eval_data rows based on miner emissions.

For each row in eval_data:
- Iterate through all miners' data
- If a miner uses that row, add that miner's emission to the row's weight
- Save weights to a JSON file for use by selection script

Usage:
    python3 backtest/calculate_weights.py --output backtest/row_weights.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict
import bittensor as bt


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


def get_miner_emissions(subtensor: bt.subtensor, netuid: int) -> Dict[str, float]:
    """
    Get emission for each miner hotkey from metagraph.
    Includes all miners regardless of emission.
    
    Args:
        subtensor: Bittensor subtensor instance
        netuid: Subnet UID
    
    Returns:
        Dictionary mapping hotkey -> emission
    """
    print(f"   Getting miner emissions from metagraph...")
    miner_emissions = {}
    
    try:
        metagraph = subtensor.metagraph(netuid)
        metagraph.sync(subtensor=subtensor)
        print(f"   ✓ Metagraph synced (block: {metagraph.block.item() if hasattr(metagraph.block, 'item') else metagraph.block})")
        
        emissions = metagraph.E.copy() if hasattr(metagraph.E, 'copy') else metagraph.E
        uids = metagraph.uids.tolist()
        hotkeys = metagraph.hotkeys
        
        for i, uid in enumerate(uids):
            if uid != 4294967295:  # Skip invalid UIDs
                emission = float(emissions[i])
                hotkey = hotkeys[i]
                miner_emissions[hotkey] = emission
        
        print(f"   ✓ Found {len(miner_emissions)} miners")
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not get metagraph data: {e}")
        print(f"   Will use all miners for weight calculation")
    
    return miner_emissions


def build_filename_to_hotkey_mapping(
    subtensor: bt.subtensor, 
    netuid: int, 
    miners_dir: str
) -> Dict[str, str]:
    """
    Build mapping from filename (repoid_revision.jsonl) to hotkey.
    
    Args:
        subtensor: Bittensor subtensor instance
        netuid: Subnet UID
        miners_dir: Directory with miner data files
    
    Returns:
        Dictionary mapping filename -> hotkey
    """
    print(f"   Building filename to hotkey mapping...")
    mapping = {}
    
    try:
        metagraph = subtensor.metagraph(netuid)
        metagraph.sync(subtensor=subtensor)
        uids = metagraph.uids.tolist()
        hotkeys = metagraph.hotkeys
        
        miner_files = list(Path(miners_dir).glob("*.jsonl"))
        matched = 0
        max_to_query = 500
        queried = 0
        
        for i, uid in enumerate(uids):
            if uid == 4294967295:
                continue
            
            if queried >= max_to_query:
                print(f"   Reached query limit ({max_to_query}), stopping mapping")
                break
            
            hotkey = hotkeys[i]
            try:
                metadata = bt.core.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)
                queried += 1
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
                            from flockoff.miners.data import ModelId
                            try:
                                model_id = ModelId.from_compressed_str(chain_str)
                                repoid = model_id.namespace
                                revision = model_id.commit
                                
                                safe_repoid = repoid.replace("/", "_").replace("\\", "_")
                                safe_revision = revision.replace("/", "_").replace("\\", "_")
                                filename = f"{safe_repoid}_{safe_revision}.jsonl"
                                
                                mapping[filename] = hotkey
                                matched += 1
                            except Exception:
                                pass
            except Exception:
                continue
        
        print(f"   ✓ Matched {matched}/{len(miner_files)} files to hotkeys")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not build mapping: {e}")
    
    return mapping


def calculate_row_weights(
    eval_data: List[Dict],
    miners_dir: str,
    miner_emissions: Dict[str, float],
    filename_to_hotkey: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
    """
    Calculate weight for each eval_data row based on miner emissions.
    
    For each row, sum up the emissions of all miners who use that row.
    
    Args:
        eval_data: Evaluation dataset
        miners_dir: Directory with miners' data files
        miner_emissions: Dictionary mapping hotkey -> emission
        filename_to_hotkey: Optional mapping from filename to hotkey
    
    Returns:
        Dictionary mapping normalized row -> weight
    """
    print(f"\n2. Calculating weights for eval_data rows...")
    print(f"   Weight = sum of emissions from miners who use each row")
    
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
    print(f"   Processing miners' data...")
    
    processed = 0
    total_emission_added = 0.0
    
    for miner_file in miner_files:
        miner_data = load_jsonl(str(miner_file))
        if not miner_data:
            continue
        
        # Get hotkey for this miner file
        filename = Path(miner_file).name
        hotkey = None
        if filename_to_hotkey:
            hotkey = filename_to_hotkey.get(filename)
        
        # If we have hotkey and emission, use it; otherwise skip this miner
        if not hotkey or hotkey not in miner_emissions:
            continue
        
        emission = miner_emissions[hotkey]
        
        # For each row in this miner's data, add emission to its weight
        for item in miner_data:
            normalized = normalize_json_item(item)
            if normalized in row_weights:
                row_weights[normalized] += emission
                total_emission_added += emission
        
        processed += 1
        if processed % 50 == 0:
            print(f"     Processed {processed}/{len(miner_files)} files...")
    
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
        help="Output file for weights (default: backtest/row_weights.json)"
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=96,
        help="Subnet UID (default: 96)"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="backtest/miners_metadata.json",
        help="Path to miners metadata JSON file (from download_miners_data.py). If provided, will use this instead of querying chain. (default: backtest/miners_metadata.json)"
    )
    parser.add_argument(
        "--use-metadata-file",
        action="store_true",
        help="Use metadata file if it exists (otherwise will query chain)"
    )
    
    # Add bittensor args
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    
    args = bt.config(parser)
    
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
    
    # Get miner emissions and build filename mapping
    miner_emissions = {}
    filename_to_hotkey = {}
    
    # Try to load from metadata file first
    metadata_file_path = Path(args.metadata_file)
    use_file = args.use_metadata_file and metadata_file_path.exists()
    
    if use_file:
        print(f"\n1.5. Loading miner metadata from file: {args.metadata_file}")
        try:
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            miner_emissions = metadata.get('miner_emissions', {})
            filename_to_hotkey = metadata.get('filename_to_hotkey', {})
            
            print(f"   ✓ Loaded emissions for {len(miner_emissions)} miners")
            print(f"   ✓ Loaded filename mappings for {len(filename_to_hotkey)} files")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load metadata file: {e}")
            print(f"   Falling back to querying chain...")
            use_file = False
    
    if not use_file:
        try:
            print(f"\n1.5. Getting metagraph data from chain...")
            subtensor = bt.subtensor(config=args)
            
            # Get miner emissions (all miners, no exclusion)
            miner_emissions = get_miner_emissions(
                subtensor, args.netuid
            )
            
            # Build filename to hotkey mapping
            try:
                filename_to_hotkey = build_filename_to_hotkey_mapping(
                    subtensor, args.netuid, args.miners_dir
                )
            except Exception as e:
                print(f"   ⚠️  Warning: Could not build filename mapping: {e}")
                print(f"   Will skip miners without hotkey mapping")
            
            subtensor.close()
        except Exception as e:
            print(f"   ⚠️  Warning: Could not get metagraph data: {e}")
            print(f"   Will use all miners for weight calculation")
    
    # Calculate weights
    row_weights = calculate_row_weights(
        eval_data,
        args.miners_dir,
        miner_emissions,
        filename_to_hotkey if filename_to_hotkey else None
    )
    
    # Save weights
    print(f"\n3. Saving weights to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(row_weights, f, indent=2)
    
    print(f"   ✓ Saved weights for {len(row_weights)} rows")
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Eval dataset: {len(eval_data):,} rows")
    print(f"Total miners: {len(miner_emissions)}")
    print(f"Rows with weight > 0: {sum(1 for w in row_weights.values() if w > 0)}")
    print(f"Output file: {args.output}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

