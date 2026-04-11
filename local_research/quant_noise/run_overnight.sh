#!/bin/bash
# =============================================================================
# Overnight Quant-Noise Experiment (~10-11 hours total)
# =============================================================================
# Runs two long QN experiments back-to-back. Compare results against your
# existing 13780-step baseline (1.2270 int8 roundtrip, no int4 data).
#
# Key insight: your 2000-step runs had WARMDOWN_ITERS=1200, meaning warmdown
# started at step 800 — only 40% of training at full LR. QN needs full-LR
# steps to learn noise robustness. These runs give ~11K steps at full LR.
#
# Expected time: ~5 hrs each = ~10 hrs total on RTX 4080
# =============================================================================

set -e  # Stop on first error
cd ~/src/parameter-golf-lucas-bryant

COMMON_ARGS="DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=13000 \
WARMDOWN_ITERS=1200 \
WARMUP_STEPS=20 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=500 \
SEED=42 \
QUANT_BITS=4"

echo "=========================================="
echo "Starting overnight Quant-Noise experiments"
echo "Time: $(date)"
echo "=========================================="

# --- RUN 1: QN p=0.05, 13K steps, int4+int8 roundtrip ---
# Lower noise = less pre-quant penalty. At 2K steps the gap was 0.037 BPB.
# With 11K steps at full LR, this gap should shrink to ~0.01-0.02.
echo ""
echo "=== RUN 1/2: 9L QN p=0.05, 13000 steps ==="
echo "Start: $(date)"
echo ""

QUANT_NOISE_P=0.05 \
QUANT_BITS=4 \
ITERATIONS=13000 \
WARMDOWN_ITERS=1200 \
WARMUP_STEPS=20 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
RUN_ID=overnight_qn005_13k \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=500 \
python local_research/quant_noise/train_gpt_quant_int.py

echo ""
echo "=== RUN 1 COMPLETE: $(date) ==="
echo ""

# --- RUN 2: QN p=0.10, 13K steps, int4+int8 roundtrip ---
# Higher noise = bigger int4 gap reduction (44% at 2K steps).
# With more training, pre-quant should improve significantly.
echo ""
echo "=== RUN 2/2: 9L QN p=0.10, 13000 steps ==="
echo "Start: $(date)"
echo ""

QUANT_NOISE_P=0.10 \
QUANT_BITS=4 \
ITERATIONS=13000 \
WARMDOWN_ITERS=1200 \
WARMUP_STEPS=20 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=0 \
RUN_ID=overnight_qn010_13k \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=500 \
python local_research/quant_noise/train_gpt_quant_int.py

echo ""
echo "=== RUN 2 COMPLETE: $(date) ==="
echo ""

echo "=========================================="
echo "ALL OVERNIGHT RUNS COMPLETE: $(date)"
echo "=========================================="
echo ""
echo "Compare against your existing baseline:"
echo "  Baseline 13780 steps: pre-quant=1.2204, int8=1.2270 (no int4 data)"
echo ""
echo "Quick comparison:"
echo "  grep 'final_int.*roundtrip_exact' logs/overnight_*.txt"
echo ""
echo "Key questions answered:"
echo "  1. Does QN pre-quant gap close with longer training?"
echo "  2. Does QN still reduce int4 gap at convergence?"
echo "  3. Is int4+QN post-quant competitive with int8 baseline?"