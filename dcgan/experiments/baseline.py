#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import Generator, Discriminator
from utils.data_preprocessing import download_and_preprocess_mnist
from utils.device_utils import get_device

def train_dcgan(config):
    """
    Train the DCGAN model with the given configuration
    
    Args:
        config (dict): Configuration parameters
    """
    # Initialize wandb
    run = wandb.init(project="dcgan-mnist", config=config)

    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Get device (GPU or CPU)
    device = get_device()

    # Load data
    train_loader, _, _ = download_and_preprocess_mnist(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"]
    )

    # Initialize models
    generator = Generator(
        latent_dim=config["latent_dim"],
        ngf=config["ngf"]
    ).to(device)

    discriminator = Discriminator(
        ndf=config["ndf"]
    ).to(device)

    # Log model architectures
    wandb.watch(generator, log="all", log_freq=100)
    wandb.watch(discriminator, log="all", log_freq=100)

    # Binary Cross Entropy loss
    criterion = nn.BCELoss()

    # Setup optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], 0.999)
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], 0.999)
    )

    # Create fixed noise for visualizing the generator's progress
    fixed_noise = torch.randn(64, config["latent_dim"], device=device)

    # Create directories for saving results
    os.makedirs(config["sample_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

    # Labels for real and fake data
    real_label = 1.0
    fake_label = 0.0

    # Training loop
    print("Starting training...")

    for epoch in range(config["n_epochs"]):
        g_losses = []
        d_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']}")
        for i, (real_imgs, _) in enumerate(pbar):
            # Configure batch size for last batch which might be smaller
            batch_size = real_imgs.size(0)

            # Move data to device
            real_imgs = real_imgs.to(device)

            # -----------------
            # Train Discriminator
            # -----------------

            optimizer_d.zero_grad()

            # Loss on real images
            real_labels = torch.full((batch_size,), real_label, device=device)
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)
            d_loss_real.backward()
            d_x = real_output.mean().item()

            # Loss on fake images
            noise = torch.randn(batch_size, config["latent_dim"], device=device)
            fake_imgs = generator(noise)
            fake_labels = torch.full((batch_size,), fake_label, device=device)
            fake_output = discriminator(fake_imgs.detach())  # Detach to avoid training generator
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss_fake.backward()
            d_g_z1 = fake_output.mean().item()

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()

            # -----------------
            # Train Generator
            # -----------------

            optimizer_g.zero_grad()

            # Generate fake images and compute loss
            # The generator wants the discriminator to think these are real
            fake_labels.fill_(real_label)  # Use real labels for generator loss
            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, fake_labels)
            g_loss.backward()
            d_g_z2 = fake_output.mean().item()

            optimizer_g.step()

            # Store losses for logging
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            # Update progress bar
            pbar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item(),
                'D(x)': d_x,
                'D(G(z))': d_g_z2
            })

            # Log to wandb at regular intervals
            if i % 50 == 0:
                wandb.log({
                    "D_loss": d_loss.item(),
                    "G_loss": g_loss.item(),
                    "D(x)": d_x,
                    "D(G(z))": d_g_z2,
                    "epoch": epoch,
                    "batch": i
                })

        # Log epoch averages
        wandb.log({
            "epoch_D_loss": np.mean(d_losses),
            "epoch_G_loss": np.mean(g_losses),
            "epoch": epoch
        })

        # Generate and save sample images
        with torch.no_grad():
            generator.eval()
            fake_imgs = generator(fixed_noise).detach().cpu()
            generator.train()

            # Rescale from [-1, 1] to [0, 1] for visualization
            fake_imgs = (fake_imgs * 0.5) + 0.5

            # Create grid of images
            fig, axs = plt.subplots(8, 8, figsize=(10, 10))
            for ax_i, ax in enumerate(axs.flatten()):
                if ax_i < fake_imgs.size(0):
                    ax.imshow(fake_imgs[ax_i, 0], cmap='gray')
                ax.axis('off')
            plt.tight_layout()

            # Save locally
            sample_file = os.path.join(config["sample_dir"], f"epoch_{epoch+1}.png")
            plt.savefig(sample_file)
            plt.close()

            # Log to wandb
            wandb.log({
                "generated_samples": wandb.Image(sample_file),
                "epoch": epoch
            })

        # Save model checkpoint
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save(generator.state_dict(), 
                      os.path.join(config["model_dir"], f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), 
                      os.path.join(config["model_dir"], f"discriminator_epoch_{epoch+1}.pth"))

    # Save final model
    torch.save(generator.state_dict(), 
              os.path.join(config["model_dir"], "generator_final.pth"))
    torch.save(discriminator.state_dict(), 
              os.path.join(config["model_dir"], "discriminator_final.pth"))

    # Finish wandb run
    wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST dataset")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_dcgan(config)