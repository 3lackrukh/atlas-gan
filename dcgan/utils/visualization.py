#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def save_generated_images(
        generator, fixed_noise, epoch,
        output_dir, device, n_rows=8, n_cols=8
        ):
    """
    Generate and save images using the generator

    Args:
        generator (nn.Module): The generator model
        fixed_noise (torch.Tensor): Fixed noise vectors
            for consistent generation
        epoch (int): Current epoch number
        output_dir (str): Directory to save images
        device (torch.device): Device to use (CPU or GPU)
        n_rows (int): Number of rows in the grid
        n_cols (int): Number of columns in the grid

    Returns:
        str: Path to the saved image file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set generator to evaluation mode
    generator.eval()

    with torch.no_grad():
        # Generate images
        fake_images = generator(fixed_noise)

        # Reshape images for display
        fake_images = (fake_images * 0.5) + 0.5  # Rescale [-1, 1] to [0, 1]

        # Create a grid
        grid = make_grid(fake_images, nrow=n_rows, normalize=False)

        # Convert to numpy for display
        img_array = grid.cpu().numpy().transpose((1, 2, 0))

        # Create and save figure
        plt.figure(figsize=(10, 10))
        plt.imshow(img_array, cmap='gray' if img_array.shape[2] == 1 else None)
        plt.axis('off')

        # Save the image
        filename = os.path.join(output_dir, f'epoch_{epoch}.png')
        plt.savefig(filename)
        plt.close()

    # Set generator back to training mode
    generator.train()

    return filename


def plot_losses(g_losses, d_losses, output_dir):
    """
    Plot generator and discriminator losses

    Args:
        g_losses (list): Generator losses
        d_losses (list): Discriminator losses
        output_dir (str): Directory to save the plot

    Returns:
        str: Path to the saved image file
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the image
    filename = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(filename)
    plt.close()

    return filename


def generate_interpolation(
        generator, z1, z2, steps=10,
        output_dir=None, device=None
        ):
    """
    Generate images by interpolating between two points in latent space

    Args:
        generator (nn.Module): The generator model
        z1 (torch.Tensor): First latent vector
        z2 (torch.Tensor): Second latent vector
        steps (int): Number of interpolation steps
        output_dir (str): Directory to save the visualization
        device (torch.device): Device to use (CPU or GPU)

    Returns:
        torch.Tensor: Tensor of generated images
    """
    # Set generator to evaluation mode
    generator.eval()

    # Create interpolation vectors
    alphas = np.linspace(0, 1, steps)
    z_interp = []

    for alpha in alphas:
        z = z1 * (1 - alpha) + z2 * alpha
        z_interp.append(z)

    z_interp = torch.cat(z_interp, dim=0)

    with torch.no_grad():
        # Generate images
        fake_images = generator(z_interp)

        # Reshape images for display
        fake_images = (fake_images * 0.5) + 0.5  # Rescale [-1, 1] to [0, 1]

        if output_dir:
            # Create a grid
            grid = make_grid(fake_images, nrow=steps, normalize=False)

            # Convert to numpy for display
            img_array = grid.cpu().numpy().transpose((1, 2, 0))

            # Create and save figure
            plt.figure(figsize=(20, 5))
            plt.imshow(
                img_array, cmap='gray' if img_array.shape[2] == 1 else None
                )
            plt.axis('off')

            # Save the image
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, 'latent_interpolation.png')
            plt.savefig(filename)
            plt.close()

    # Set generator back to training mode
    generator.train()

    return fake_images
