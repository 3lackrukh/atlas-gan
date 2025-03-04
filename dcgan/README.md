# DCGAN Project with MNIST

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the MNIST dataset. The project uses PyTorch and includes various experiments with architecture modifications, hyperparameter tuning, and precision changes.

## Project Structure

```
dcgan/
├── data/                  # Storage for MNIST dataset and preprocessed versions
├── models/
    └── baseline.py            # baseline DCGAN architecture
├── utils/                 # Helper functions
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   └── device_utils.py        # GPU/CPU device selection utilities
├── experiments/           # Experiment scripts (to be implemented)
├── docker-compose.yml     # Docker compose configuration
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── SETUP_INSTRUCTIONS.md  # Detailed setup instructions
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
```

See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for detailed setup instructions, troubleshooting, and additional commands.

## Dataset

The project uses the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is automatically downloaded and preprocessed when running the data preprocessing script.

## Experiment Tracking

This project uses Weights & Biases (WandB) for experiment tracking. Before running experiments, set up your WandB account and login inside the Docker container:

```bash
wandb login
```

## Experiments

The project includes the following experiments:

1. **Architecture Variations**: Modifications to the DCGAN architecture
2. **Hyperparameter Tuning**: Experimenting with learning rates, batch sizes, etc.
3. **Precision Changes**: Tests with different floating-point precisions

## License

[MIT License](LICENSE)