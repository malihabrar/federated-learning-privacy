#!/bin/bash
# Job runner: InvGrad attack for one (alpha, seed, method) across all images.
# Called by schedule_invgrad_batch.sh via bsub.
# Loops over 30 images internally — one GPU job per (alpha, seed, method).

ALPHA=""
SEED=""
METHOD="prune"
MODEL="lenet"
N_IMAGES=30
ITERS=4000
RESTARTS=3
DEVICE="cuda"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --alpha)    ALPHA="$2";    shift ;;
    --seed)     SEED="$2";     shift ;;
    --method)   METHOD="$2";   shift ;;
    --model)    MODEL="$2";    shift ;;
    --n_images) N_IMAGES="$2"; shift ;;
    --iters)    ITERS="$2";    shift ;;
    --restarts) RESTARTS="$2"; shift ;;
    --device)   DEVICE="$2";   shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

module load python3/3.11.9
source /zhome/39/3/205397/plgp/.venv/bin/activate

echo "Starting InvGrad batch: alpha=${ALPHA} seed=${SEED} method=${METHOD} model=${MODEL}"
echo "Images: 0-$((N_IMAGES-1)), iters=${ITERS}, restarts=${RESTARTS}, device=${DEVICE}"
echo "Started at: $(date)"

ALPHA_FMT=$(printf "%.2f" ${ALPHA})

FAILED=0
SKIPPED=0
for IMG in $(seq 0 $((N_IMAGES-1))); do
  SHELVE="results/invgrad/bin/img_${IMG}_seed_${SEED}_alpha_${ALPHA_FMT}_method_${METHOD}_model_${MODEL}.db"
  if [ -f "${SHELVE}" ]; then
    echo "SKIP: img=${IMG} (already done)"
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  python single_image_invgrad.py \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --img_inx ${IMG} \
    --method ${METHOD} \
    --model ${MODEL} \
    --iters ${ITERS} \
    --restarts ${RESTARTS} \
    --device ${DEVICE}

  if [ $? -ne 0 ]; then
    echo "FAILED: img=${IMG}"
    FAILED=$((FAILED+1))
  fi
done

echo "Finished at: $(date)"
echo "Skipped: ${SKIPPED}/${N_IMAGES} (already done)"
echo "Failed:  ${FAILED}/${N_IMAGES}"
