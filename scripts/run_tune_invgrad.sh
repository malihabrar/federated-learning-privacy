#!/bin/bash
# Runner script for tune_invgrad.py — called by schedule_tune_invgrad.sh via bsub.

STAGE=1
SEED=10
N_IMAGES=3
DEVICE=cpu
TV=""
LR=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --stage)     STAGE="$2";     shift ;;
    --seed)      SEED="$2";      shift ;;
    --n_images)  N_IMAGES="$2";  shift ;;
    --device)    DEVICE="$2";    shift ;;
    --tv)        TV="$2";        shift ;;
    --lr)        LR="$2";        shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

module load python3/3.11.9
source /zhome/39/3/205397/plgp/.venv/bin/activate

CMD="python tune_invgrad.py --stage $STAGE --seed $SEED --n_images $N_IMAGES --device $DEVICE"
[ -n "$TV" ] && CMD="$CMD --tv $TV"
[ -n "$LR" ] && CMD="$CMD --lr $LR"

echo "Running: $CMD"
$CMD
