#!/bin/bash
# Job runner: iDLG attack for one (alpha, seed, method) across all images × restarts.
# Called by schedule_idlg_batch.sh via bsub.
# Loops over 30 images × 5 restarts = 150 calls to single_image_idlg.py internally.

ALPHA=""
SEED=""
METHOD="prune"
N_IMAGES=30
N_RESTARTS=5

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --alpha)      ALPHA="$2";      shift ;;
    --seed)       SEED="$2";       shift ;;
    --method)     METHOD="$2";     shift ;;
    --n_images)   N_IMAGES="$2";   shift ;;
    --n_restarts) N_RESTARTS="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

module load python3/3.11.9
source /zhome/39/3/205397/plgp/.venv/bin/activate

echo "Starting iDLG batch: alpha=${ALPHA} seed=${SEED} method=${METHOD}"
echo "Images: 0-$((N_IMAGES-1)), Restarts: 0-$((N_RESTARTS-1))"
echo "Started at: $(date)"

ALPHA_FMT=$(printf "%.2f" ${ALPHA})

FAILED=0
SKIPPED=0
for IMG in $(seq 0 $((N_IMAGES-1))); do
  for ITE in $(seq 0 $((N_RESTARTS-1))); do
    SHELVE="results/idlg_experiments/bin/img_${IMG}_seed_${SEED}_alpha_${ALPHA_FMT}_method_${METHOD}_${ITE}.db"
    if [ -f "${SHELVE}" ]; then
      echo "SKIP: img=${IMG} ite=${ITE} (already done)"
      SKIPPED=$((SKIPPED+1))
      continue
    fi

    python single_image_idlg.py \
      --alpha ${ALPHA} \
      --seed ${SEED} \
      --img_inx ${IMG} \
      --iteration ${ITE} \
      --method ${METHOD}

    if [ $? -ne 0 ]; then
      echo "FAILED: img=${IMG} ite=${ITE}"
      FAILED=$((FAILED+1))
    fi
  done
done

echo "Finished at: $(date)"
echo "Skipped: ${SKIPPED}/$((N_IMAGES * N_RESTARTS)) (already done)"
echo "Failed:  ${FAILED}/$((N_IMAGES * N_RESTARTS))"
