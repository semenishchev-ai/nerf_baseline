#!/bin/bash
set -e

# Base variables
BASE_DIR="/home/a_semenishchev/VS/iter_2/experiments/baseline_torch_ngp"
DATA_DIR="/home/a_semenishchev/VS/iter_2/experiments/baseline_ngp/data/lego"
CONDA_RUN="conda run --no-capture-output -n baseline_ngp"
ITERATIONS=10000

cd $BASE_DIR

echo "=========================================================="
echo "Training pure torch-ngp for $ITERATIONS iterations..."
echo "=========================================================="

# Run torch-ngp training
$CONDA_RUN python main_nerf.py "$DATA_DIR" \
    --workspace trial_lego \
    -O \
    --bound 1.0 \
    --scale 0.8 \
    --dt_gamma 0 \
    --iters $ITERATIONS \
    > train.log 2>&1

echo "=========================================================="
echo "Training completed for torch-ngp. Results saved in trial_lego/log_*.txt"
echo "=========================================================="
