# DCGAN Project with MNIST

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST dataset. The project uses PyTorch and includes various experiments with architecture modifications, hyperparameter tuning, and precision changes.

## Project Structure

```
dcgan/
├── data/                  # Storage for MNIST dataset and preprocessed versions
├── models/                # DCGAN model architectures
│   └── baseline.py        # Baseline DCGAN implementation
├── utils/                 # Helper functions
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── device_utils.py        # GPU/CPU device selection utilities
│   └── visualization.py       # Functions for visualizing outputs
├── experiments/           # Experiment scripts
│   └── baseline.py        # Baseline training script
├── configs/               # Configuration files
│   └── baseline.yaml      # Baseline model configuration
├── logs/                  # Training logs and generated images
│   ├── models/            # Saved model checkpoints
│   └── samples/           # Generated sample images
├── docker-compose.yml     # Docker compose configuration
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Environment Setup

This project uses Docker to provide a consistent environment that works on any system, automatically using GPU acceleration when available (CUDA) or falling back to CPU when not.

### Quick Start

```bash
# Build and start the container
docker-compose up -d

# Enter the container
docker-compose exec dcgan bash

# Inside the container, download and process the dataset
python utils/data_preprocessing.py

# Run the baseline DCGAN training
python experiments/baseline.py
```

## DCGAN Architecture

The implemented DCGAN follows the architecture guidelines from the original DCGAN paper:

### Generator
- Takes a random noise vector (latent_dim=100) and generates 28x28 grayscale images
- Uses transposed convolutions for upsampling
- BatchNorm in all layers except the output layer
- ReLU activations in all layers except the output layer (Tanh)

### Discriminator
- Takes 28x28 grayscale images and outputs a probability of the image being real
- Uses strided convolutions instead of pooling layers
- BatchNorm in all layers except the first and last
- LeakyReLU activations (alpha=0.2)
- Sigmoid activation in the output layer

## Experiment Tracking

This project uses Weights & Biases (WandB) for experiment tracking. Before running experiments, you need to set up your WandB account and login inside the Docker container:

```bash
wandb login
```

The training script logs:
- Generator and discriminator losses
- Generated image samples
- Model gradients and parameters
- Hyperparameters

## Experiments

The project includes the following experiments:

1. **Baseline DCGAN**: Standard implementation following the original paper
2. **Architecture Variations**: (To be implemented)
3. **Hyperparameter Tuning**: (To be implemented)
4. **Precision Changes**: (To be implemented)

## License

[MIT License](LICENSE)