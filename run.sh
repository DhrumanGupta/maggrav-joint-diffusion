#!/bin/bash

source .venv/bin/activate

# accelerate launch src/train.py --epochs 100 --sample_every 5 --num_samples 50000 --batch_size 256 --sample_every 1 --disable_wandb --progress

# nohup python -m src.preprocessing \
#     --data-root /home/aritra/Loop3D-noddyverse-fa1fbd5/noddyverse_data \
#     --output ./output \
#     --verify-orientation \
#     --num-parallel-tars 4 \
#     --num-workers-per-tar 48 \
#     --temp-root "/scratch/dhruman_gupta/noddyverse_data" \
#     > preprocessing.log 2>&1 &

# nohup python src/concat_zarrs.py /scratch/dhruman_gupta/noddyverse_preprocessed/output/ /scratch/dhruman_gupta/noddyverse_preprocessed/output-final --batch-size 128 --workers 32 --blosc-threads 4 --compressor zstd --clevel 3 > concat_zarrs.log 2>&1 &

nohup accelerate launch src/train_vae.py --config config/train_vae.yaml > logs/train_vae.log 2>&1 &