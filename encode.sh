export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# accelerate launch -m src.data.encode_latents --input_zarr /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --output_zarr ./zarrs/encoded-latents_0.001.zarr --checkpoint outputs/vae_attention_sweep/kl_weight_0.001/vae_attention_checkpoint_step_1000020.pt --save_logvar --copy_metadata --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_samples $NUM_SAMPLES

# accelerate launch -m src.data.encode_latents --input_zarr /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --output_zarr ./zarrs/encoded-latents_0.005.zarr --checkpoint outputs/vae_attention_sweep/kl_weight_0.005/vae_attention_checkpoint_step_700000.pt --save_logvar --copy_metadata --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_samples $NUM_SAMPLES

# accelerate launch -m src.data.encode_latents --input_zarr /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --output_zarr ./zarrs/encoded-latents_0.01.zarr --checkpoint outputs/vae_attention_sweep/kl_weight_0.01/vae_attention_checkpoint_step_700000.pt --save_logvar --copy_metadata --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_samples $NUM_SAMPLES

# python -m src.evaluate.latent_pca --latents_zarr zarrs/encoded-latents_0.001.zarr/ --output_dir outputs/latent_pca/pca_0.001 --batch_size 128 --num_workers 8 --n_components 10 --use_gpu --num_gpus 7

# python -m src.evaluate.latent_pca --latents_zarr zarrs/encoded-latents_0.005.zarr/ --output_dir outputs/latent_pca/pca_0.005 --batch_size 128 --num_workers 8 --n_components 10 --use_gpu --num_gpus 7

# python -m src.evaluate.latent_pca --latents_zarr zarrs/encoded-latents_0.01.zarr/ --output_dir outputs/latent_pca/pca_0.01 --batch_size 128 --num_workers 8 --n_components 10 --use_gpu --num_gpus 7

# python -m src.evaluate.latent_pca --latents_zarr /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old/ --output_dir outputs/latent_pca/raw --batch_size 128 --num_workers 16 --n_components 10 --max_samples 100000

############################
# SAME CODE
############################

# BATCH_SIZE=12
# NUM_WORKERS=4
# NUM_SAMPLES=100000

# KL_WEIGHTS=(0.001 0.005)
# LATENT_CHANNELS=(2 4)

# # Calculate NUM_GPUS from CUDA_VISIBLE_DEVICES
# NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# for kl_weight in ${KL_WEIGHTS[@]}; do
#     for latent_channels in ${LATENT_CHANNELS[@]}; do
#         accelerate launch -m src.data.encode_latents --input_zarr /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --output_zarr ./zarrs/encoded-latents/latent_channels_${latent_channels}/kl_weight_${kl_weight}.zarr --checkpoint outputs/vae_attention_sweep/latent_channels_${latent_channels}/kl_weight_${kl_weight}/vae_attention_checkpoint_step_1000008.pt --save_logvar --copy_metadata --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_samples $NUM_SAMPLES
#     done
# done

# # Do latent PCA for each of the encoded latents
# for latent_channels in ${LATENT_CHANNELS[@]}; do
#     for kl_weight in ${KL_WEIGHTS[@]}; do
#         python -m src.evaluate.latent_pca --latents_zarr ./zarrs/encoded-latents/latent_channels_${latent_channels}/kl_weight_${kl_weight}.zarr --output_dir outputs/latent_pca/latent_channels_${latent_channels}/kl_weight_${kl_weight} --batch_size 128 --num_workers 8 --n_components 10 --use_gpu --num_gpus $NUM_GPUS
#     done
# done

# # Now also run evaluation for VAE
# # accelerate launch -m src.evaluate.evaluate_vae_attention --zarr_path /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --checkpoint ./outputs/vae_attention_sweep/kl_weight_0.01/vae_attention_checkpoint_step_1000008.pt --num_samples 100 --plot --taus 0 0.5 1.0

# # TAUS="0 0.5 1.0 4"
# TAUS="0"
# for latent_channels in ${LATENT_CHANNELS[@]}; do
#     for kl_weight in ${KL_WEIGHTS[@]}; do
#         accelerate launch -m src.evaluate.evaluate_vae_attention --zarr_path /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --checkpoint ./outputs/vae_attention_sweep_3/latent_channels_${latent_channels}/kl_weight_${kl_weight}/vae_attention_checkpoint_step_1000008.pt --num_samples 100 --plot --taus $TAUS
#     done
# done


# accelerate launch -m src.train.train_vae_attention --config config/train_vae_attention_sweep.yaml

# Encode latents
# BATCH_SIZE=12
# NUM_WORKERS=4

# accelerate launch -m src.data.encode_latents --input_zarr /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --output_zarr ./zarrs/encoded-latents-filtered.zarr --checkpoint outputs/vae_attention_sweep_filtered/latent_channels_4/kl_weight_0.01/vae_attention_checkpoint_step_1000008.pt --save_logvar --copy_metadata --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --use_filter_cache --filter_cache_path ./cache/train_filter_stats.pt --susc_active_threshold_log10 -0.8 --min_active_frac 0.01 --low_info_keep_prob 0.02

# # Do latent PCA for the encoded latents
# python -m src.evaluate.latent_pca --config config/evaluate_latent_pca_encoded_filtered.yaml


# Eval

# TAUS="0"
# # TAUS="0 0.5 1.0 4"
# # accelerate launch -m src.evaluate.evaluate_vae_attention --zarr_path /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --checkpoint ./outputs/vae_attention_sweep_filtered/latent_channels_4/kl_weight_0.01/vae_attention_checkpoint_step_1000008.pt --num_samples 100 --plot --taus $TAUS
# accelerate launch -m src.evaluate.evaluate_vae_attention --zarr_path /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --checkpoint ./outputs/vae_attention_sweep_2/kl_weight_0.01/vae_attention_checkpoint_step_1000020.pt --num_samples 100 --plot --taus $TAUS

# Encode samples
# Use the ckpt outputs/vae_attention_sweep_3/latent_channels_4/kl_weight_0.001/vae_attention_checkpoint_step_1000008.pt
# No filtering

BATCH_SIZE=12
NUM_WORKERS=4

NUM_SAMPLES=10000000

accelerate launch -m src.data.encode_latents --stats_path cache/vae_stats.json --input_zarr /scratch/dhruman_gupta/noddyverse_preprocessed/output-final-old --output_zarr ./zarrs/encoded-latents-attention-sweep-3.zarr --checkpoint outputs/vae_attention_sweep_3/latent_channels_4/kl_weight_0.001/vae_attention_checkpoint_step_1000008.pt --save_logvar --copy_metadata --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --num_samples $NUM_SAMPLES