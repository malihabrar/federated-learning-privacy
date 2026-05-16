#!/bin/bash
# InvGrad attack scheduler (GPU — A100, CC 8.0, compatible with installed PyTorch).
# One job per (alpha, seed, method) — each job runs all 30 images internally.
#
# Job count:
#   prune: 5 alphas × 3 seeds = 15 jobs
#   clip:  1       × 3 seeds =  3 jobs
#   total: 18 jobs
#
# Estimated runtime per job: ~30–45 min on A100 (30 images × 4000 iters × 1 restart).

ALPHAS_PRUNE=( 1 0.95 0.90 0.05 0.00 )
SEEDS=( 10 20 30 )

queue="gpul40s"
cores=4
mem="rusage[mem=4GB]"
maxmem="5GB"
walltime="03:00"

mkdir -p logs/invgrad/

# prune method: all alphas
for X in "${ALPHAS_PRUNE[@]}"; do
  for Y in "${SEEDS[@]}"; do
    bsub \
      -q ${queue} \
      -n ${cores} \
      -J "INVG_prune_${X}_${Y}" \
      -R "${mem}" \
      -M ${maxmem} \
      -R "span[hosts=1]" \
      -W ${walltime} \
      -gpu "num=1" \
      -o "logs/invgrad/out_prune_alpha_${X}_seed_${Y}.log" \
      -e "logs/invgrad/err_prune_alpha_${X}_seed_${Y}.log" \
      scripts/run_invgrad_batch.sh --alpha $X --seed $Y --method prune --restarts 1
  done
done

# clip method
for Y in "${SEEDS[@]}"; do
  bsub \
    -q ${queue} \
    -n ${cores} \
    -J "INVG_clip_${Y}" \
    -R "${mem}" \
    -M ${maxmem} \
    -R "span[hosts=1]" \
    -W ${walltime} \
    -gpu "num=1" \
    -o "logs/invgrad/out_clip_seed_${Y}.log" \
    -e "logs/invgrad/err_clip_seed_${Y}.log" \
    scripts/run_invgrad_batch.sh --alpha 1 --seed $Y --method clip --restarts 1
done
