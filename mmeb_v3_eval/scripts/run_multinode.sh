#!/usr/bin/env bash
# Multi-node multi-GPU MMEB-v3 evaluation for WeMM-Embedding (torchrun elastic
# conventions; run the SAME command on every node with a different NODE_RANK).
#
# Example (2 nodes x 8 GPUs):
#   node0: MODEL_PATH=/path/to/ckpt DATA_BASEDIR=/data/MMEB-V3 \
#          NNODES=2 NODE_RANK=0 MASTER_ADDR=node0.ip MASTER_PORT=29677 \
#          bash scripts/run_multinode.sh
#   node1: ... NODE_RANK=1 ... (everything else identical)
#
# Rank 0 writes checkpoints/scores to its own OUTPUT_DIR. Other nodes may use a
# different (e.g. node-local) OUTPUT_DIR: the resume decision is made on rank 0
# and broadcast, so restarts are safe either way. Scores live on rank 0.
#
# If your cluster's NCCL defaults are misconfigured (e.g. a broken RDMA plugin),
# set the usual NCCL_* overrides in the environment before calling this script
# (e.g. NCCL_IB_DISABLE=1, NCCL_SOCKET_IFNAME=<iface with the node IP>).
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
EXTRA_ARGS="${EXTRA_ARGS:-}"

NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK (0..NNODES-1) for this node}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node address}"
MASTER_PORT="${MASTER_PORT:-29677}"

MODALITY_IDX=0
for MODALITY in $MODALITIES; do
  MODALITY_IDX=$((MODALITY_IDX + 1))
  # Use a distinct rendezvous port per modality: rapid back-to-back torchruns
  # reusing one port intermittently hit stale TCPStore state and rendezvous
  # timeouts on some clusters.
  MODALITY_PORT=$((MASTER_PORT + MODALITY_IDX))
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
  echo "==> Modality: $MODALITY (node $NODE_RANK/$NNODES, port $MODALITY_PORT, batch_size $BS $MAXLEN_ARG)"
  torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MODALITY_PORT" \
    --max_restarts=0 eval.py \
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
echo "==> Node $NODE_RANK done. Results under $OUTPUT_DIR"
