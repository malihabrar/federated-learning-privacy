#!/bin/bash
# InvGrad batch runner for Lightning.ai (no HPC module/venv needed).
# Runs all 18 conditions sequentially: 5 alphas x 3 seeds + clip x 3 seeds.
# Skip logic: skips images already done (shelve .db exists).
# Run from: /teamspace/studios/this_studio/plgp/
# Usage: bash scripts/run_lightning_invgrad.sh

ALPHAS_PRUNE=(1 0.95 0.90 0.05 0.00)
SEEDS=(10 20 30)
N_IMAGES=30
ITERS=2000
RESTARTS=3
DEVICE="cuda"

cd /teamspace/studios/this_studio

echo "=== Lightning.ai InvGrad batch ==="
echo "Started at: $(date)"

run_condition() {
  local ALPHA=$1
  local SEED=$2
  local METHOD=$3
  local ALPHA_FMT=$(printf "%.2f" ${ALPHA})

  echo ""
  echo "--- alpha=${ALPHA} seed=${SEED} method=${METHOD} ---"

  local FAILED=0
  local SKIPPED=0

  for IMG in $(seq 0 $((N_IMAGES-1))); do
    SHELVE="results/invgrad/bin/img_${IMG}_seed_${SEED}_alpha_${ALPHA_FMT}_method_${METHOD}_model_lenet_iters_${ITERS}_restarts_${RESTARTS}.db"
    if [ -f "${SHELVE}" ]; then
      SKIPPED=$((SKIPPED+1))
      continue
    fi

    python single_image_invgrad.py \
      --alpha ${ALPHA} \
      --seed ${SEED} \
      --img_inx ${IMG} \
      --method ${METHOD} \
      --model lenet \
      --iters ${ITERS} \
      --restarts ${RESTARTS} \
      --device ${DEVICE}

    if [ $? -ne 0 ]; then
      echo "FAILED: img=${IMG}"
      FAILED=$((FAILED+1))
    fi
  done

  echo "Skipped: ${SKIPPED}/${N_IMAGES} | Failed: ${FAILED}/${N_IMAGES}"
}

# prune conditions
for ALPHA in "${ALPHAS_PRUNE[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_condition $ALPHA $SEED prune
  done
done

# clip condition
for SEED in "${SEEDS[@]}"; do
  run_condition 1 $SEED clip
done

echo ""
echo "=== All done at: $(date) ==="
