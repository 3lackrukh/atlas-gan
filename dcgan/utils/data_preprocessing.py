import torch
from torchvision import datasets, transforms
import os
import sys

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.device_utils import get_device

def download_and_preprocess_mnist(data_dir='./data', batch_size=128):
    """
    Downloads the MNIST dataset and applies preprocessing transformations.
    
    Args:
        data_dir (str): Directory to store the dataset
        batch_size (int): Batch size for data loaders
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to use (CUDA or CPU)
    """
    # Get the appropriate device
    device = get_device()
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transformations
    # For GANs, we typically normalize to [-1, 1] instead of [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Determine the number of workers based on CPU count and device type
    num_workers = 0 if device.type == 'cuda' else min(4, os.cpu_count())
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"MNIST dataset downloaded and processed.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    return train_loader, test_loader, device

if __name__ == "__main__":
    train_loader, test_loader, device = download_and_preprocess_mnist()
    
    # Visualize a batch to verify data loading
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get a batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Create a grid of images
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        # Rescale from [-1, 1] to [0, 1] for display
        plt.imshow(images[i].squeeze().numpy() * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    plt.savefig('./data/sample_mnist_batch.png')
    plt.close()
    
    print("Sample batch visualization saved to ./data/sample_mnist_batch.png")
    
    # Test moving a batch to the selected device
    images = images.to(device)
    labels = labels.to(device)
    print(f"Successfully moved batch to {device}")