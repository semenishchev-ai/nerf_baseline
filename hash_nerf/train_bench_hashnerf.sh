#!/bin/bash
# Запуск HashNeRF-pytorch на датасете Lego (10k итераций)

# Переходим в директорию скрипта
cd "$(dirname "$0")"

# Путь к данным (Blender Lego)
DATA_DIR="../baseline_ngp/data/lego"

# Запуск через conda
conda run --no-capture-output -n baseline_ngp python run_nerf.py \
    --config configs/lego.txt \
    --datadir "$DATA_DIR" \
    --expname bench_lego_10k \
    --N_iters 10000 \
    --i_print 100 \
    --i_testset 10000 \
    --i_weights 10000 \
    --i_video 10000 \
    --white_bkgd \
    --lrate 0.01 \
    --lrate_decay 10 \
    --finest_res 512
