#!/bin/bash
set -e

echo "Setting up environment for NeRF benchmarks..."

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Compile CUDA extensions for torch-ngp
echo "Compiling CUDA extensions for torch-ngp..."
cd torch_ngp
pip install ./raymarching
pip install ./gridencoder
pip install ./shencoder
pip install ./freqencoder
cd ..

echo "Setup completed successfully."
