#!/usr/bin/env
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator Network

    Transforms a random noise vector into a synthetic image.
    for MNIST (28x28 grayscale images),
    generated dims 1x28x28.
    """
    def __init__(self, latent_dim=100, ngf=64):
        """
        Initialize the Generator

        Args:
            latent_dim (int): Dimension of the latent space (noise vector)
            ngf (int): Size of feature maps in generator, controls capacity
        """
        super(Generator, self).__init__()

        # Start with 1x1 representation that will be upsampled
        self.main = nn.Sequential(
            # Input is latent vector z
            # First deconvolution
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size: (ngf*4) x 4 x 4

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Size: (ngf*2) x 8 x 8

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size: ngf x 16 x 16

            # Final layer - output 1 channel (grayscale)
            # Output values should be in range [-1, 1] after normalization
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Size: 1 x 32 x 32 (crop or downsample to 1 x 28 x 28)
        )

        # Initialize weights with mean=0, std=0.02
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes weights according to DCGAN paper"""
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, z):
        """
        Forward pass

        Args:
            z (torch.Tensor): Rnadom noise vector
                of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Generated image
        """
        # Reshape to work with conv layers
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.main(z)
        # Center crop from 32 x 32 to 28 x 28 to match MNIST dims
        img = img[:, :, 2:30, 2:30]
        return img
