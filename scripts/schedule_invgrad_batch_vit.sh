#!/bin/bash
# InvGrad attack scheduler for ViT (GPU — A100/L40s).
# One job per (alpha, seed, method) — each job runs all 30 images internally.
#
# Job count:
#   prune: 5 alphas × 3 seeds = 15 jobs
#   clip:  1       × 3 seeds =  3 jobs
#   total: 18 jobs
#
# Estimated runtime per job: ~2–4 h on A100 (30 images × 2000 iters × 3 restarts, ViT 86M params).

ALPHAS_PRUNE=( 1 0.95 0.90 0.05 0.00 )
SEEDS=( 10 20 30 )

queue="gpul40s"
cores=4
mem="rusage[mem=8GB]"
maxmem="9GB"
walltime="06:00"

mkdir -p logs/invgrad_vit/

# prune method: all alphas
for X in "${ALPHAS_PRUNE[@]}"; do
  for Y in "${SEEDS[@]}"; do
    bsub \
      -q ${queue} \
      -n ${cores} \
      -J "INVG_vit_prune_${X}_${Y}" \
      -R "${mem}" \
      -M ${maxmem} \
      -R "span[hosts=1]" \
      -W ${walltime} \
      -gpu "num=1" \
      -o "logs/invgrad_vit/out_prune_alpha_${X}_seed_${Y}.log" \
      -e "logs/invgrad_vit/err_prune_alpha_${X}_seed_${Y}.log" \
      scripts/run_invgrad_batch.sh --alpha $X --seed $Y --method prune --model vit --iters 2000 --restarts 3 --device cuda
  done
done

# clip method
for Y in "${SEEDS[@]}"; do
  bsub \
    -q ${queue} \
    -n ${cores} \
    -J "INVG_vit_clip_${Y}" \
    -R "${mem}" \
    -M ${maxmem} \
    -R "span[hosts=1]" \
    -W ${walltime} \
    -gpu "num=1" \
    -o "logs/invgrad_vit/out_clip_seed_${Y}.log" \
    -e "logs/invgrad_vit/err_clip_seed_${Y}.log" \
    scripts/run_invgrad_batch.sh --alpha 1 --seed $Y --method clip --model vit --iters 2000 --restarts 3 --device cuda
done
