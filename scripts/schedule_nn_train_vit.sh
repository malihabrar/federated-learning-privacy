#!/bin/bash
# ViT federated training scheduler (GPU).
# One job per (alpha, seed, method) — same conditions as LeNet for direct comparison.
#
# Job count:
#   prune: 13 alphas × 3 seeds = 39 jobs
#   clip:  1       × 3 seeds =  3 jobs
#   total: 42 jobs
#
# ViT uses lr=1e-4 (set in nn_train.py) and starts from the CIFAR-10 fine-tuned checkpoint.
# Estimated runtime per job: ~4–6 hours on GPU.

ALPHAS_PRUNE=( 1 0.95 0.90 0.05 0.00 )
SEEDS=( 10 20 30 )

queue="gpul40s"
cores=4
mem="rusage[mem=8GB]"
maxmem="9GB"
walltime="08:00"

mkdir -p logs/train_vit/

# prune method: all alphas
for X in "${ALPHAS_PRUNE[@]}"; do
  for Y in "${SEEDS[@]}"; do
    bsub \
      -q ${queue} \
      -n ${cores} \
      -J "TRAIN_vit_prune_${X}_${Y}" \
      -R "${mem}" \
      -M ${maxmem} \
      -R "span[hosts=1]" \
      -W ${walltime} \
      -gpu "num=1" \
      -o "logs/train_vit/out_prune_alpha_${X}_seed_${Y}.log" \
      -e "logs/train_vit/err_prune_alpha_${X}_seed_${Y}.log" \
      scripts/run_nn_train.sh --alpha $X --seed $Y --method prune --model vit
  done
done

# clip method
for Y in "${SEEDS[@]}"; do
  bsub \
    -q ${queue} \
    -n ${cores} \
    -J "TRAIN_vit_clip_${Y}" \
    -R "${mem}" \
    -M ${maxmem} \
    -R "span[hosts=1]" \
    -W ${walltime} \
    -gpu "num=1" \
    -o "logs/train_vit/out_clip_seed_${Y}.log" \
    -e "logs/train_vit/err_clip_seed_${Y}.log" \
    scripts/run_nn_train.sh --seed $Y --method clip --model vit
done
