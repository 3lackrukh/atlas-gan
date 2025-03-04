import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from models.baseline import Generator
from utils.device_utils import get_device
from utils.visualization import generate_interpolation
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_samples(config, model_path, num_samples=64):
    """
    Generate samples using a trained generator

    Args:
        config (dict): Configuration parameters
        model_path (str): Path to the generator model weights
        num_samples (int): Number of samples to generate
    """
    # Get device (GPU or CPU)
    device = get_device()

    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Initialize generator
    generator = Generator(
        latent_dim=config["latent_dim"],
        ngf=config["ngf"]
    ).to(device)

    # Load model weights
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Create output directory
    os.makedirs(config["sample_dir"], exist_ok=True)

    print(f"Generating {num_samples} samples...")

    # Generate random noise
    z = torch.randn(num_samples, config["latent_dim"], device=device)

    with torch.no_grad():
        # Generate images
        fake_images = generator(z)

        # Rescale from [-1, 1] to [0, 1] for visualization
        fake_images = (fake_images * 0.5) + 0.5

        # Create grid of images
        n_rows = int(np.sqrt(num_samples))
        fig, axs = plt.subplots(n_rows, n_rows, figsize=(10, 10))
        for ax_i, ax in enumerate(axs.flatten()):
            if ax_i < fake_images.size(0):
                ax.imshow(fake_images[ax_i, 0].cpu().numpy(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()

        # Save samples
        sample_file = os.path.join(
            config["sample_dir"],
            "generated_samples.png"
            )
        plt.savefig(sample_file)
        plt.close()

        print(f"Samples saved to {sample_file}")

    # Generate latent space interpolation
    print("Generating latent space interpolation...")

    # Generate two random points in latent space
    z1 = torch.randn(1, config["latent_dim"], device=device)
    z2 = torch.randn(1, config["latent_dim"], device=device)

    # Generate interpolation
    generate_interpolation(
        generator,
        z1,
        z2,
        steps=10,
        output_dir=config["sample_dir"],
        device=device
    )

    print(f"Interpolation saved to {os.path.join(config['sample_dir'], 'latent_interpolation.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from trained DCGAN")
    parser.add_argument("--config", type=str, default="../configs/baseline.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to generator model weights")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples to generate")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    generate_samples(config, args.model, args.num_samples)