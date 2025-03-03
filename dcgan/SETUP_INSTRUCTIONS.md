# Docker Setup Instructions for DCGAN Project

This document provides detailed instructions for setting up the Docker environment for the DCGAN project. The setup is designed to work on any system, automatically using CUDA if available or falling back to CPU if not.

## Prerequisites

1. Install Docker:
   - [Docker for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Docker for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - [Docker for Linux](https://docs.docker.com/engine/install/)

2. For GPU support (NVIDIA only):
   - Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd dcgan
```

### 2. Build and start the Docker container

```bash
docker-compose up -d
```

This command will:
- Build the Docker image with PyTorch and all dependencies
- Start a container in detached mode
- Mount your project directory inside the container

### 3. Verify the setup

```bash
# Enter the running container
docker-compose exec dcgan bash

# Once inside the container, run the device verification script
python utils/device_utils.py
```

You should see output indicating whether CUDA is available and which device (GPU or CPU) will be used.

## Common Commands

- **Enter the container shell:**
  ```bash
  docker-compose exec dcgan bash
  ```

- **View container logs:**
  ```bash
  docker-compose logs
  ```

- **Stop the container:**
  ```bash
  docker-compose down
  ```

- **Restart the container:**
  ```bash
  docker-compose restart
  ```

## Troubleshooting

- **CUDA not detected:** If you have an NVIDIA GPU but CUDA is not detected, ensure the NVIDIA Container Toolkit is properly installed and Docker is configured to use the NVIDIA runtime.

- **Permission issues with mounted volumes:** If you encounter permission issues with files created inside the container, you may need to adjust the user permissions or use Docker's user namespace remapping feature.

- **Container fails to start:** Check the Docker logs for details:
  ```bash
  docker-compose logs
  ```
