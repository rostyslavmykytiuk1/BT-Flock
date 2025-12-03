import bittensor as bt
import functools
import multiprocessing
import time
from typing import Optional, Any


def run_in_subprocess(func: functools.partial, ttl: int) -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete."""

    def wrapped_func(
        func: functools.partial,
        queue: multiprocessing.Queue,
        log_queue: multiprocessing.Queue,
    ):
        try:
            log_queue.put(f"Starting {func.func.__name__} in subprocess")
            start_time = time.time()
            result = func()
            elapsed = time.time() - start_time
            log_queue.put(f"Completed {func.func.__name__} in {elapsed:.2f} seconds")
            queue.put(result)
        except (Exception, BaseException) as e:
            log_queue.put(f"Error in subprocess: {str(e)}")
            queue.put(e)

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    log_queue = ctx.Queue()
    process = ctx.Process(target=wrapped_func, args=[func, queue, log_queue])

    bt.logging.info(f"Starting subprocess for {func.func.__name__}")
    process.start()

    # Monitor the process and collect logs
    timeout = time.time() + ttl
    while process.is_alive() and time.time() < timeout:
        try:
            while not log_queue.empty():
                bt.logging.info(log_queue.get(block=False))
            time.sleep(0.5)
        except Exception:
            pass

    if process.is_alive():
        bt.logging.warning(
            f"Process for {func.func.__name__} timed out after {ttl} seconds. Terminating..."
        )
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Collect any remaining logs
    while not log_queue.empty():
        try:
            bt.logging.info(log_queue.get(block=False))
        except Exception:
            pass

    try:
        result = queue.get(block=False)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, BaseException):
            raise Exception(f"BaseException raised in subprocess: {str(result)}")
        return result
    except Exception as e:
        raise Exception(f"Failed to get result from subprocess: {str(e)}")


def debug_commit_process(func, wallet, subtensor, subnet_uid, data):
    """A simplified version that tries to debug the commit process without subprocesses"""
    bt.logging.info(
        f"Attempting direct commit call with subnet_uid: {subnet_uid}, data length: {len(data)}"
    )
    try:
        # First check if we can ping the endpoint
        bt.logging.info(f"Chain endpoint: {subtensor.chain_endpoint}")
        result = func(wallet, subnet_uid, data)
        bt.logging.info(f"Commit result: {result}")
        return result
    except Exception as e:
        bt.logging.error(f"Direct commit error: {str(e)}")
        bt.logging.debug(f"Exception type: {type(e)}")
        raise e


async def store_model_metadata(
    subtensor: bt.subtensor, wallet: Optional[bt.wallet], subnet_uid: str, data: str
):
    """Stores model metadata on this subnet for a specific wallet."""
    if wallet is None:
        raise ValueError("No wallet available to write to the chain.")

    bt.logging.info(f"Preparing to commit metadata to subnet {subnet_uid}")
    bt.logging.debug(
        f"Wallet hotkey: {wallet.hotkey.ss58_address if hasattr(wallet, 'hotkey') else 'Not available'}"
    )

    # Get network status before committing
    try:
        bt.logging.info("Checking network status...")
        # Use proper Bittensor API calls - adjust based on available methods
        bt.logging.debug(f"Network: {subtensor.network}")
        bt.logging.debug(f"Chain endpoint: {subtensor.chain_endpoint}")
    except Exception as e:
        bt.logging.error(f"Failed to get network status: {str(e)}")

    # Check if subnet exists and if wallet is registered
    try:
        bt.logging.info("Checking subnets...")
        subnets = subtensor.get_subnets()
        # Check if subnet_uid is in subnets directly
        subnet_exists = int(subnet_uid) in subnets if subnets else False
        bt.logging.info(f"Subnet {subnet_uid} exists: {subnet_exists}")
    except Exception as e:
        bt.logging.error(f"Failed to check subnet existence: {str(e)}")

    try:
        # Try a simpler approach without subprocess first
        bt.logging.info("Attempting direct commit first for debugging...")
        result = debug_commit_process(
            subtensor.commit, wallet, subtensor, subnet_uid, data
        )
        bt.logging.success(f"Direct commit succeeded with result: {result}")
        return result
    except Exception as e:
        bt.logging.error(f"Direct commit failed: {str(e)}")
        bt.logging.info("Falling back to subprocess approach...")

    # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
    partial = functools.partial(
        subtensor.commit,
        wallet,
        subnet_uid,
        data,
    )

    bt.logging.info(
        f"Committing metadata to subnet {subnet_uid} with timeout of 60 seconds..."
    )
    try:
        return run_in_subprocess(partial, 60)
    except Exception as e:
        bt.logging.error("Diagnostic information:")
        bt.logging.error(f"1. Error received: {str(e)}")
        bt.logging.error(
            "2. The 'no close frame received or sent' error suggests a WebSocket connection issue"
        )
        bt.logging.error(
            "3. This typically happens when the connection to the Bittensor network is interrupted"
        )

        # Suggest possible solutions
        bt.logging.info("Possible solutions:")
        bt.logging.info("1. Check your internet connection")
        bt.logging.info(
            "2. Try a different chain endpoint using: subtensor = bt.subtensor(chain_endpoint='wss://...')"
        )
        bt.logging.info("3. Ensure your subnet_uid is correct")
        bt.logging.info("4. Make sure your wallet has enough balance for network fees")
        raise e
