import torch
import os

def get_device():
    """
    Get the appropriate device (CUDA GPU or CPU) for PyTorch operations.
    Also handles environment variable settings for performance optimization.
    
    Returns:
        torch.device: The device to use for PyTorch tensors
    """
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        # Set environment variables for CUDA optimization
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = torch.device("cuda")
        
        # Print device properties
        device_properties = torch.cuda.get_device_properties(0)
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Capability: {device_properties.major}.{device_properties.minor}")
        print(f"  Total Memory: {device_properties.total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")
        # Set environment variables for CPU optimization if using Intel CPU
        if "Intel" in os.popen("lscpu").read():
            print("  Intel CPU detected. Setting optimizations.")
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
            os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        device = torch.device("cpu")
        
    return device

if __name__ == "__main__":
    # Test the device detection
    device = get_device()
    print(f"Selected device: {device}")
    
    # Create a test tensor and move it to the device
    x = torch.randn(1000, 1000)
    x = x.to(device)
    
    # Perform a simple operation to test
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        start.record()
    
    result = torch.matmul(x, x)
    
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        print(f"Operation time: {start.elapsed_time(end):.2f} ms")
    
    print("Device test completed successfully.")