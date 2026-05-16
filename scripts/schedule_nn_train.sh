#!/bin/bash
# Federated training scheduler.
# One job per (alpha, seed, method) — covers all experiment conditions.
#
# Job count:
#   prune: 13 alphas × 3 seeds = 39 jobs
#   clip:  1       × 3 seeds =  3 jobs
#   total: 42 jobs

ALPHAS_PRUNE=( 1 0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.90 0.05 0.00 )
SEEDS=( 10 20 30 )

queue="hpc"
cores=4
mem="rusage[mem=4GB]"
maxmem="5GB"
walltime="12:00"

mkdir -p logs/train/

# prune method: all alphas
for X in "${ALPHAS_PRUNE[@]}"; do
  for Y in "${SEEDS[@]}"; do
    bsub \
      -q ${queue} \
      -n ${cores} \
      -J "TRAIN_prune_${X}_${Y}" \
      -R "${mem}" \
      -M ${maxmem} \
      -R "span[hosts=1]" \
      -W ${walltime} \
      -o "logs/train/out_prune_alpha_${X}_seed_${Y}.log" \
      -e "logs/train/err_prune_alpha_${X}_seed_${Y}.log" \
      scripts/run_nn_train.sh --alpha $X --seed $Y --method prune
  done
done

# clip method
for Y in "${SEEDS[@]}"; do
  bsub \
    -q ${queue} \
    -n ${cores} \
    -J "TRAIN_clip_${Y}" \
    -R "${mem}" \
    -M ${maxmem} \
    -R "span[hosts=1]" \
    -W ${walltime} \
    -o "logs/train/out_clip_seed_${Y}.log" \
    -e "logs/train/err_clip_seed_${Y}.log" \
    scripts/run_nn_train.sh --seed $Y --method clip
done
