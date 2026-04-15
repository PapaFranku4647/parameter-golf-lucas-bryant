#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a 1xH100 pod after downloading sp1024 data:
#   python3 data/cached_challenge_fineweb.py --variant sp1024
#   bash records/track_non_record_16mb/2026-04-14_TWEO_EarlyCosine_OutlierReg_SP1024_4h/run_h100_80m_pair.sh

OUT_DIR="records/track_non_record_16mb/2026-04-14_TWEO_EarlyCosine_OutlierReg_SP1024_4h/h100_80m_seed999"
SCRIPT="records/track_non_record_16mb/2026-04-14_TWEO_EarlyCosine_OutlierReg_SP1024_4h/train_gpt.py"
mkdir -p "$OUT_DIR"

for spec in \
  "seed999_base 999 fixed 0 0 3 0" \
  "seed999_tweo_cosdecay_lam0002_tau5_d3000 999 cosine 0.0002 0 5 3000"
do
  set -- $spec
  name=$1
  seed=$2
  schedule=$3
  lam=$4
  lam_final=$5
  tau=$6
  decay=$7

  run_id="h100_80m_${name}"

  echo "==== ${run_id} schedule=${schedule} lambda=${lam} final=${lam_final} tau=${tau} decay=${decay} ===="

  SEED="$seed" \
  RUN_ID="$run_id" \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  MAX_WALLCLOCK_SECONDS=4800 \
  WARMDOWN_ITERS=1200 \
  VAL_LOSS_EVERY=4000 \
  TRAIN_LOG_EVERY=1000 \
  TWEO_LAMBDA="$lam" \
  TWEO_LAMBDA_FINAL="$lam_final" \
  TWEO_LAMBDA_SCHEDULE="$schedule" \
  TWEO_DECAY_STEPS="$decay" \
  TWEO_START_STEP=0 \
  TWEO_RAMP_STEPS=0 \
  TWEO_TAU="$tau" \
  TWEO_P=4 \
  TWEO_ACT_STATS_BATCHES=1 \
  TWEO_ACT_STATS_EVERY=4000 \
  torchrun --standalone --nproc_per_node=1 "$SCRIPT"

  mv "logs/${run_id}.txt" "$OUT_DIR/"
done

echo "Logs written to $OUT_DIR"
