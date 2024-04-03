# Land Use and Land Cover Classification

This repository contains a deep learning project aimed at classifying land use and cover using the EUROSAT dataset. The project utilizes a neural network model developed with PyTorch and trained to classify satellite images into 10 different land cover categories with high accuracy.

## Project Overview

The project is structured into a series of Jupyter notebooks that guide through different stages of machine learning workflow:

1. `Data Preparation`: Extracts and organizes the EUROSAT dataset for training and testing.
2. `Model Design`: Constructs and modularizes the neural network architecture for satellite image classification.
3. `Model Training`: Covers the model initialization, training with performance tracking, and model saving for future use.
4. `Model Deployment`: Demonstrates the process to load the trained model and classify new images.

Each notebook is designed to be clear and concise, ensuring that the methodology is easily understood and reproducible.

## Repository Contents

- `landneuralnetwork.py`: Python module containing the neural network code.
- `nn_model.pth`: Saved trained model.
- Jupyter Notebooks: Step-by-step guide to the entire machine learning pipeline.

## Environment Setup

To set up the development environment, follow these steps:

```bash
# Install Anaconda from the official site: https://conda.io/projects/conda/en/latest/index.html

# Create a virtual environment with Conda
conda create --name venv-land

# Check if the environment is created
conda env list

# Activate the virtual environment
conda activate venv-land

# Deactivate the virtual environment
conda deactivate venv-land

# Delete the virtual environment
conda env remove --name venv-land
```

## Tools and Libraries
- Anaconda
- Python
- PyTorch
- Jupyter Notebook
- Visual Studio Code
- Hugging Face Libraries
- Git for version control
- NVIDIA CUDA for GPU processing

## Model Architecture
The model LandClassifierNet is a convolutional neural network with the following structure:

- Convolutional layers for feature extraction.
- Dropout layers to prevent overfitting.
- Fully connected layers for classification.

The architecture details and training process are documented in the Jupyter notebooks included in this repository.

## Usage
After setting up the environment and cloning the repository, navigate to the notebook directory and start Jupyter Notebook

## Contributing
Contributions to this project are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## Acknowledgments
- EUROSAT for providing the dataset.
- PyTorch Community for the comprehensive documentation and support.
- Data Science Academy
