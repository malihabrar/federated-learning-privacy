#!/bin/bash
# Hyperparameter tuning for InvGrad.
# Submits one job per stage.
#
# Usage:
#   bash scripts/schedule_tune_invgrad.sh 1              # Stage 1: TV x LR grid
#   bash scripts/schedule_tune_invgrad.sh 2 0.01 0.1    # Stage 2: iters x restarts grid

STAGE=${1:-1}
SEED=10
N_IMAGES=3
DEVICE=cpu

queue="hpc"
cores=2
mem="rusage[mem=4GB]"
maxmem="5GB"
walltime="04:00"

mkdir -p logs/tune_invgrad/

if [ "$STAGE" == "1" ]; then
    echo "Submitting Stage 1: TV x LR grid (16 combos x ${N_IMAGES} images)"
    bsub \
        -q ${queue} \
        -n ${cores} \
        -J "tune_invgrad_stage1" \
        -R "${mem}" \
        -M ${maxmem} \
        -R "span[hosts=1]" \
        -W ${walltime} \
        -o "logs/tune_invgrad/out_stage1.log" \
        -e "logs/tune_invgrad/err_stage1.log" \
        scripts/run_tune_invgrad.sh --stage 1 --seed ${SEED} --n_images ${N_IMAGES} --device ${DEVICE}
    echo "Submitted. Check logs/tune_invgrad/out_stage1.log"
    echo "Once done, run: bash scripts/schedule_tune_invgrad.sh 2 <best_tv> <best_lr>"

elif [ "$STAGE" == "2" ]; then
    TV=${2:?Usage: schedule_tune_invgrad.sh 2 <TV> <LR>}
    LR=${3:?Usage: schedule_tune_invgrad.sh 2 <TV> <LR>}
    echo "Submitting Stage 2: iters x restarts grid (12 combos x ${N_IMAGES} images, TV=${TV} LR=${LR})"
    bsub \
        -q ${queue} \
        -n ${cores} \
        -J "tune_invgrad_stage2" \
        -R "${mem}" \
        -M ${maxmem} \
        -R "span[hosts=1]" \
        -W ${walltime} \
        -o "logs/tune_invgrad/out_stage2.log" \
        -e "logs/tune_invgrad/err_stage2.log" \
        scripts/run_tune_invgrad.sh --stage 2 --seed ${SEED} --n_images ${N_IMAGES} --tv ${TV} --lr ${LR} --device ${DEVICE}
    echo "Submitted. Check logs/tune_invgrad/out_stage2.log"

else
    echo "Unknown stage: $STAGE"
    exit 1
fi
