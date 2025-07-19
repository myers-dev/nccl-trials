import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    """Initializes the distributed environment for each process."""
    # These environment variables are needed for `init_process_group`
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group.
    # 'nccl' is the best backend for multi-GPU communication.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Initialized Rank {rank} of {world_size} on PID {os.getpid()}.")

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def arithmetic_worker(rank, world_size):
    """
    The main function for each GPU process.
    'rank' is the unique ID for this process, from 0 to world_size-1.
    """
    print(f"--> Starting worker on rank {rank}.")
    setup(rank, world_size)

    # Set the GPU for this specific process. This is crucial.
    torch.cuda.set_device(rank)

    # Create a tensor with a unique value for each rank.
    # The tensor must be on the correct GPU device.
    tensor = torch.ones(3, 3, device=rank) * (rank)
    print(f"Rank {rank} | Initial tensor:\n{tensor}\n")

    # --- The Core Operation ---
    # dist.all_reduce sums the tensor data from all processes and places
    # the final result in-place on each process's tensor.
    # The default operation is SUM.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} | Tensor after all_reduce:\n{tensor}\n")

    # The expected result is the sum of 0+1+2+3+4+5+6+7 = 28
    # So the final tensor on every GPU should be all 28s.
    expected_sum = sum(range(1, world_size ))
    assert torch.all(tensor.eq(expected_sum)), f"Rank {rank} failed the check!"

    print(f"Rank {rank}: Demo finished successfully.")
    cleanup()

if __name__ == "__main__":
    # We want to run this on all 8 GPUs.
    world_size = 8
    if torch.cuda.device_count() < world_size:
        print(f"Error: This script requires {world_size} GPUs, but only {torch.cuda.device_count()} were found.")
    else:
        print(f"Found {torch.cuda.device_count()} GPUs. Spawning {world_size} processes for the arithmetic demo.")
        # Use mp.spawn to launch 'world_size' processes, each running the 'arithmetic_worker' function.
        mp.spawn(arithmetic_worker, args=(world_size,), nprocs=world_size, join=True)

