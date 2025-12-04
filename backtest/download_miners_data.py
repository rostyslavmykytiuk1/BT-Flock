#!/usr/bin/env python3
"""
Download miners' data and metadata from Hugging Face and Bittensor chain.

This script:
1. Downloads commit info (metadata) for all miners from the chain
2. Downloads data.jsonl files from Hugging Face for each miner
3. Gets miner emissions from the metagraph
4. Builds filename to hotkey mapping
5. Stores all information in a JSON file for later use

Usage:
    python3 backtest/download_miners_data.py --netuid 96
"""

import os
import argparse
import bittensor as bt
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError
from flockoff.miners.data import ModelId, ModelMetadata

# Output files
OUTPUT_LOG_FILE = "backtest/download_log.txt"
OUTPUT_METADATA_FILE = "backtest/miners_metadata.json"
BASE_OUT_DIR = "backtest/hf_datajsonl"


class DualLogger:
    """Simple logger that writes to both console and file."""
    def __init__(self, filepath: str):
        dirpath = os.path.dirname(os.path.abspath(filepath))
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
    
    def log(self, msg: str):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()
    
    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def set_config():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download miners' data and metadata from Hugging Face and Bittensor chain"
    )
    parser.add_argument("--netuid", type=int, default=96, help="The subnet UID.")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing data in hf_datajsonl directory before downloading"
    )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.config(parser)
    return config


def retrieve_model_metadata(
    subtensor: bt.subtensor, subnet_uid: int, hotkey: str
) -> Optional[ModelMetadata]:
    """Retrieves model metadata on this subnet for specific hotkey."""
    metadata = bt.core.extrinsics.serving.get_metadata(subtensor, subnet_uid, hotkey)

    if not metadata:
        return None

    try:
        commitment = metadata["info"]["fields"][0]
        chain_str = None

        # Handle tuple of dictionary with various Raw types (Raw24, Raw61, Raw68, etc.)
        if (
            isinstance(commitment, tuple)
            and len(commitment) > 0
            and isinstance(commitment[0], dict)
        ):
            raw_keys = [key for key in commitment[0].keys() if key.startswith("Raw")]

            if raw_keys:
                raw_key = raw_keys[0]
                raw_data = commitment[0][raw_key][0]
                chain_str = "".join(chr(i) for i in raw_data)
            else:
                return None
        else:
            return None

        # Check if this is JSON data (special case) or a repository ID
        if chain_str.startswith("{"):
            return None

        # Parse the chain_str
        try:
            model_id = ModelId.from_compressed_str(chain_str)
            model_metadata = ModelMetadata(id=model_id, block=metadata["block"])
            return model_metadata
        except Exception as e:
            return None

    except Exception as e:
        return None


def download_datajsonl(repoid: str, revision: str, logger: DualLogger) -> Optional[str]:
    """
    Download 'data.jsonl' from a public Hugging Face dataset repo at a specific revision
    and save it to 'hf_datajsonl' as '<repoid_revision>.jsonl'.

    Args:
        repoid: Hugging Face dataset repo ID, e.g., "user/repo".
        revision: Branch, tag, or commit SHA.
        logger: Logger instance for output

    Returns:
        The local destination path if successful, None otherwise.
    """
    if not repoid or not revision:
        return None

    repo_type = "dataset"
    filename_in_repo = "data.jsonl"

    # Sanitize destination filename: repoid_revision.jsonl
    safe_repoid = repoid.replace("/", "_").replace("\\", "_")
    safe_revision = revision.replace("/", "_").replace("\\", "_")
    dest_filename = f"{safe_repoid}_{safe_revision}.jsonl"
    dest_path = os.path.join(BASE_OUT_DIR, dest_filename)

    # Skip if file already exists
    if os.path.exists(dest_path):
        logger.log(f"   → Skipping (already exists): {dest_filename}")
        return dest_path

    logger.log(f"   → Downloading '{filename_in_repo}' from {repoid} @ {revision}...")

    try:
        cached_path = hf_hub_download(
            repo_id=repoid,
            filename=filename_in_repo,
            revision=revision,
            repo_type=repo_type,
        )
        shutil.copy2(cached_path, dest_path)
        logger.log(f"   ✓ Saved to: {dest_path}")
        return dest_path
    except EntryNotFoundError as e:
        logger.log(f"   ✗ '{filename_in_repo}' not found in {repoid}@{revision}")
    except RepositoryNotFoundError as e:
        logger.log(f"   ✗ Repository not found for {repoid}")
    except HfHubHTTPError as e:
        logger.log(f"   ✗ HTTP error for {repoid}@{revision}")
    except Exception as e:
        logger.log(f"   ✗ Unexpected error for {repoid}@{revision}: {e}")
    return None


def get_miner_emissions(subtensor: bt.subtensor, netuid: int) -> Dict[str, float]:
    """
    Get emission for each miner hotkey from metagraph.
    
    Args:
        subtensor: Bittensor subtensor instance
        netuid: Subnet UID
    
    Returns:
        Dictionary mapping hotkey -> emission
    """
    miner_emissions = {}
    
    try:
        metagraph = subtensor.metagraph(netuid)
        metagraph.sync(subtensor=subtensor)
        
        emissions = metagraph.E.copy() if hasattr(metagraph.E, 'copy') else metagraph.E
        uids = metagraph.uids.tolist()
        hotkeys = metagraph.hotkeys
        
        for i, uid in enumerate(uids):
            if uid != 4294967295:  # Skip invalid UIDs
                emission = float(emissions[i])
                hotkey = hotkeys[i]
                miner_emissions[hotkey] = emission
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not get metagraph data: {e}")
    
    return miner_emissions


def build_filename_to_hotkey_mapping(
    subtensor: bt.subtensor, 
    netuid: int, 
    miners_dir: str,
    logger: DualLogger
) -> Dict[str, str]:
    """
    Build mapping from filename (repoid_revision.jsonl) to hotkey.
    
    Args:
        subtensor: Bittensor subtensor instance
        netuid: Subnet UID
        miners_dir: Directory with miner data files
        logger: Logger instance for output
    
    Returns:
        Dictionary mapping filename -> hotkey
    """
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
        
        logger.log(f"   Building filename to hotkey mapping for {len(miner_files)} files...")
        
        for i, uid in enumerate(uids):
            if uid == 4294967295:
                continue
            
            if queried >= max_to_query:
                logger.log(f"   Reached query limit ({max_to_query}), stopping mapping")
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
        
        logger.log(f"   ✓ Matched {matched}/{len(miner_files)} files to hotkeys")
    except Exception as e:
        logger.log(f"   ⚠️  Warning: Could not build mapping: {e}")
    
    return mapping


def main():
    logger = DualLogger(OUTPUT_LOG_FILE)
    MAX_EMISSION_THRESHOLD = 10.0  # Skip validators with emission > this value
    
    try:
        config = set_config()
        
        logger.log("=" * 70)
        logger.log("DOWNLOAD MINERS DATA")
        logger.log("=" * 70)
        
        # Initialize subtensor
        logger.log(f"\n1. Connecting to Bittensor network (netuid={config.netuid})...")
        subtensor = bt.subtensor(config=config)
        metagraph = subtensor.metagraph(config.netuid)
        metagraph.sync(subtensor=subtensor)
        current_uids = metagraph.uids.tolist()
        logger.log(f"   ✓ Connected. Found {len(current_uids)} UIDs")
        
        # Get miner emissions
        logger.log(f"\n2. Getting miner emissions from metagraph...")
        miner_emissions = get_miner_emissions(subtensor, config.netuid)
        logger.log(f"   ✓ Found emissions for {len(miner_emissions)} miners")
        
        # Prepare output directory - always clear existing files before downloading
        logger.log(f"\n3. Preparing output directory: {BASE_OUT_DIR}")
        if os.path.exists(BASE_OUT_DIR):
            logger.log(f"   Clearing existing data...")
            for filename in os.listdir(BASE_OUT_DIR):
                file_path = os.path.join(BASE_OUT_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.log(f"     Removed: {filename}")
                except Exception as e:
                    logger.log(f"     Error removing {filename}: {e}")
            logger.log(f"   ✓ Cleared {BASE_OUT_DIR} directory")
        else:
            os.makedirs(BASE_OUT_DIR, exist_ok=True)
            logger.log(f"   ✓ Directory ready")
        
        # Download metadata and data files
        logger.log(f"\n4. Downloading miners' metadata and data files...")
        logger.log(f"   Skipping validators with emission > {MAX_EMISSION_THRESHOLD}")
        miners_metadata_list = []
        downloaded_count = 0
        skipped_count = 0
        skipped_high_emission_count = 0
        error_count = 0
        
        for uid_i in current_uids:
            hotkey = metagraph.hotkeys[uid_i]
            emission = miner_emissions.get(hotkey, 0.0)
            
            # Skip validators with emission > 10
            if emission > MAX_EMISSION_THRESHOLD:
                skipped_high_emission_count += 1
                logger.log(f"\n   UID {uid_i} | Hotkey: {hotkey[:16]}... | Emission: {emission:.2f} (SKIPPED - emission > {MAX_EMISSION_THRESHOLD})")
                continue
            
            metadata_i = retrieve_model_metadata(subtensor, config.netuid, hotkey)
            
            if metadata_i is None:
                skipped_count += 1
                continue
            
            # Build filename
            safe_repoid = metadata_i.id.namespace.replace("/", "_").replace("\\", "_")
            safe_revision = metadata_i.id.commit.replace("/", "_").replace("\\", "_")
            filename = f"{safe_repoid}_{safe_revision}.jsonl"
            
            # Download data file
            logger.log(f"\n   UID {uid_i} | Hotkey: {hotkey[:16]}... | Emission: {emission:.2f}")
            logger.log(f"   Repo: {metadata_i.id.namespace} @ {metadata_i.id.commit[:16]}...")
            
            download_path = download_datajsonl(
                metadata_i.id.namespace,
                metadata_i.id.commit,
                logger
            )
            
            # Store metadata
            miner_info = {
                "uid": int(uid_i),
                "hotkey": hotkey,
                "namespace": metadata_i.id.namespace,
                "commit": metadata_i.id.commit,
                "competition_id": metadata_i.id.competition_id,
                "block": int(metadata_i.block),
                "filename": filename,
                "emission": emission,
                "downloaded": download_path is not None
            }
            miners_metadata_list.append(miner_info)
            
            if download_path:
                downloaded_count += 1
            else:
                error_count += 1
            
            logger.log("   " + "-" * 66)
        
        logger.log(f"\n   Summary:")
        logger.log(f"     Total miners: {len(current_uids)}")
        logger.log(f"     Skipped (emission > {MAX_EMISSION_THRESHOLD}): {skipped_high_emission_count}")
        logger.log(f"     With metadata: {len(miners_metadata_list)}")
        logger.log(f"     Downloaded: {downloaded_count}")
        logger.log(f"     Errors: {error_count}")
        logger.log(f"     Skipped (no metadata): {skipped_count}")
        
        # Build filename to hotkey mapping
        logger.log(f"\n5. Building filename to hotkey mapping...")
        filename_to_hotkey = build_filename_to_hotkey_mapping(
            subtensor, config.netuid, BASE_OUT_DIR, logger
        )
        
        # Filter miner_emissions to exclude validators (emission > 10)
        filtered_miner_emissions = {
            hotkey: emission 
            for hotkey, emission in miner_emissions.items() 
            if emission <= MAX_EMISSION_THRESHOLD
        }
        logger.log(f"   Filtered emissions: {len(filtered_miner_emissions)} miners (excluded {len(miner_emissions) - len(filtered_miner_emissions)} validators)")
        
        # Filter filename_to_hotkey to exclude validators
        filtered_filename_to_hotkey = {
            filename: hotkey
            for filename, hotkey in filename_to_hotkey.items()
            if miner_emissions.get(hotkey, 0.0) <= MAX_EMISSION_THRESHOLD
        }
        logger.log(f"   Filtered filename mappings: {len(filtered_filename_to_hotkey)} miners (excluded {len(filename_to_hotkey) - len(filtered_filename_to_hotkey)} validators)")
        
        # Prepare final metadata structure (only miners, no validators)
        final_metadata = {
            "netuid": config.netuid,
            "block": int(metagraph.block.item() if hasattr(metagraph.block, 'item') else metagraph.block),
            "total_miners": len(current_uids),
            "miners_with_metadata": len(miners_metadata_list),
            "miner_emissions": filtered_miner_emissions,  # Only miners, no validators
            "filename_to_hotkey": filtered_filename_to_hotkey,  # Only miners, no validators
            "miners": miners_metadata_list  # Only miners, no validators
        }
        
        # Save metadata to JSON file
        logger.log(f"\n6. Saving metadata to {OUTPUT_METADATA_FILE}...")
        output_path = Path(OUTPUT_METADATA_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(final_metadata, f, indent=2)
        
        logger.log(f"   ✓ Saved metadata for {len(miners_metadata_list)} miners (validators excluded)")
        
        # Final summary
        logger.log(f"\n" + "=" * 70)
        logger.log("SUMMARY")
        logger.log("=" * 70)
        logger.log(f"NetUID: {config.netuid}")
        logger.log(f"Block: {final_metadata['block']}")
        logger.log(f"Total UIDs: {len(current_uids)}")
        logger.log(f"Skipped validators (emission > {MAX_EMISSION_THRESHOLD}): {skipped_high_emission_count}")
        logger.log(f"Miners in metadata file: {len(miners_metadata_list)}")
        logger.log(f"Data files downloaded: {downloaded_count}")
        logger.log(f"Filename mappings (miners only): {len(filtered_filename_to_hotkey)}")
        logger.log(f"Miner emissions stored: {len(filtered_miner_emissions)}")
        logger.log(f"Metadata file: {OUTPUT_METADATA_FILE}")
        logger.log(f"Data directory: {BASE_OUT_DIR}")
        logger.log("=" * 70)
        
        subtensor.close()
        
    except Exception as e:
        logger.log(f"\nERROR: {e}")
        import traceback
        logger.log(traceback.format_exc())
        return 1
    finally:
        logger.close()
    
    return 0


if __name__ == "__main__":
    exit(main())

