#!/bin/bash
# Job runner: federated training for one (alpha, seed, method).
# Called by schedule_nn_train.sh via bsub.

ALPHA=""
SEED=""
METHOD=""
MODEL=""
MAX_ROUNDS=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --alpha)      ALPHA="$2";      shift ;;
    --seed)       SEED="$2";       shift ;;
    --method)     METHOD="$2";     shift ;;
    --model)      MODEL="$2";      shift ;;
    --max_rounds) MAX_ROUNDS="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

cd /work3/s234533/plgp

module load python3/3.11.9
source /zhome/39/3/205397/plgp/.venv/bin/activate

CMD="python nn_train.py"
[ -n "$ALPHA" ]  && CMD="$CMD --alpha $ALPHA"
[ -n "$SEED" ]   && CMD="$CMD --seed $SEED"
[ -n "$METHOD" ] && CMD="$CMD --method $METHOD"
[ -n "$MODEL" ]      && CMD="$CMD --model $MODEL"
[ -n "$MAX_ROUNDS" ] && CMD="$CMD --max_rounds $MAX_ROUNDS"

$CMD
