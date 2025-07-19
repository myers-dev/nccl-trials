import sys
import torch

def format_bytes(size_bytes):
    """Converts a size in bytes to a human-readable format (GB, MB, KB)."""
    if size_bytes == 0:
        return "0B"
    power = 1024
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    n = 0
    while size_bytes >= power and n < len(power_labels) -1 :
        size_bytes /= power
        n += 1
    return f"{size_bytes:.2f} {power_labels[n]}"

def query_cuda_devices():
    """
    Queries and reports detailed information about available CUDA devices using PyTorch.
    """
    print("--- PyTorch CUDA Device Query ---")

    if not torch.cuda.is_available():
        print("\nCUDA is not available. PyTorch did not find any CUDA-enabled GPUs.")
        return

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Version (used by PyTorch): {torch.version.cuda}")
    try:
        print(f"NVIDIA Driver Version: {torch.cuda.get_driver_version()}")
    except Exception:
        print("NVIDIA Driver Version: Not available through PyTorch.")

    device_count = torch.cuda.device_count()
    print(f"\nFound {device_count} CUDA-enabled device(s).")
    print("=" * 60)

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name}")
        print("-" * 40)
        print(f"  CUDA Capability:         {props.major}.{props.minor}")
        print(f"  Total Memory:            {format_bytes(props.total_memory)}")
        print(f"  Multi-Processor Count:   {props.multi_processor_count}")
        print("=" * 60)

if __name__ == "__main__":
    query_cuda_devices()

