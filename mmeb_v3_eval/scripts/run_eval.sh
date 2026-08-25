#!/usr/bin/env bash
# Single-node multi-GPU MMEB-v3 evaluation for WeMM-Embedding.
#
# Usage:
#   MODEL_PATH=/path/to/WeMM_Embedding DATA_BASEDIR=/path/to/MMEB-V3 bash scripts/run_eval.sh
#
# Optional env overrides:
#   MODALITIES="image video visdoc text tool memory gui"  (default: all non-audio)
#   OUTPUT_DIR=exps/wemm_embedding
#   BATCH_SIZE=8  NPROC_PER_NODE=8  MASTER_PORT=29677
#   BS_TEXT / BS_GUI / MAXLEN_TEXT / MAXLEN_GUI  (per-modality overrides)
#   EXTRA_ARGS="--query_limit 100"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the WeMM-Embedding checkpoint directory}"
DATA_BASEDIR="${DATA_BASEDIR:-data/MMEB-V3}"
# `--data_basedir` only resolves the media roots from the yaml. Dataset metadata
# is looked up through src/constant/dataset_hflocal_path.py, which reads this
# env var; without it the lookups miss and every task falls back to downloading
# from the Hub (and the gui tasks, which have no Hub fallback, fail outright).
export MMEB_V3_DATA_DIR="${MMEB_V3_DATA_DIR:-$DATA_BASEDIR}"
OUTPUT_DIR="${OUTPUT_DIR:-exps/wemm_embedding}"
MODALITIES="${MODALITIES:-image video visdoc text tool memory gui}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29677}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

for MODALITY in $MODALITIES; do
  # Per-modality batch size / sequence length. Long-document retrieval (text)
  # and GUI multi-screenshot trajectories (gui) need a long --max_len and a
  # small batch: large batches over 100k-token sequences can stall or OOM on
  # ~40GB GPUs. Override via BS_<MODALITY> / MAXLEN_<MODALITY> env vars.
  case "$MODALITY" in
    text)
      BS="${BS_TEXT:-1}"; MAXLEN_ARG="--max_len ${MAXLEN_TEXT:-262144}" ;;
    gui)
      BS="${BS_GUI:-1}";  MAXLEN_ARG="--max_len ${MAXLEN_GUI:-262144}" ;;
    *)
      BS="${BATCH_SIZE:-8}"; MAXLEN_ARG="" ;;
  esac
  echo "==> Modality: $MODALITY (batch_size $BS $MAXLEN_ARG)"
  torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" --max_restarts=0 eval.py \
    --pooling last \
    --normalize true \
    --model_backbone wemm_embedding \
    --model_name "$MODEL_PATH" \
    --per_device_eval_batch_size "$BS" \
    --dataset_config "experiments/public/eval/${MODALITY}.yaml" \
    --encode_output_path "$OUTPUT_DIR/$MODALITY" \
    --data_basedir "$DATA_BASEDIR" \
    $MAXLEN_ARG \
    $EXTRA_ARGS
done
echo "==> All modalities done. Results under $OUTPUT_DIR"
