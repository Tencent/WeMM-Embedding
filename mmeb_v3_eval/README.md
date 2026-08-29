# WeMM-Embedding MMEB-v3 Evaluation

MMEB-v3 evaluation for the WeMM-Embedding model, built directly on top of the
official [TIGER-AI-Lab/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec)
evaluation pipeline (vendored at commit `2638a8413fda4b98668a29ea763b4898814bfea7`).

This repository exists so that anyone can reproduce our MMEB-v3 numbers. It is
the official VLM2Vec code with the smallest possible diff:

1. **Multi-node multi-GPU inference.** `eval.py` now shards data and gates
   result writes on the *global* rank, so standard `torchrun --nnodes=N`
   works. Single-node usage is unchanged.
2. **`wemm_embedding` model backbone.** A new backbone implementing
   WeMM-Embedding preprocessing (chat template + `process_vision_info` +
   `model.embedding()`) with batched inference, registered the same way as the
   official backbones.
3. **Dataset instructions aligned with our released model.** Instruction
   strings are hardcoded in the per-dataset parser files.
4. **Video frame count aligned with our released numbers.** The official
   `experiments/public/eval/video.yaml` samples `num_frames: 8` per video;
   our reported MMEB-v3 video scores use 64 frames per video, so the
   vendored yaml sets `num_frames: 64` (the on-disk frame dumps already
   contain 64 frames per video, `max_frames_saved: 64`).

Everything else — dataset parsers, candidate generation, metrics
(`RankingMetrics`), scoring, output format — is untouched official code.

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

One command downloads the raw archives from the official Hugging Face repos
(pinned revisions, including the `video_tasks/frames/video_ret.tar.gz` video
retrieval frames) and extracts them into the eval-ready layout:

```bash
DATA_ROOT=/path/to/MMEB-V3 bash scripts/download_data.sh
```

The script is a thin wrapper around the official
`experiments/public/data/dataset_setup_v3.py`; see its header comments for
exactly what is fetched from where. Audio and OmniSET data are skipped
(WeMM-Embedding has no audio tower and the eval configs do not use OmniSET).

The evaluation reads both media and dataset metadata from that extracted root,
through two separate knobs that the run scripts derive from `DATA_BASEDIR`:

- `--data_basedir` resolves the media roots declared in the yaml
  (`image_root`, `video_root`, `query_file`, ...).
- `MMEB_V3_DATA_DIR` resolves the metadata paths in
  `src/constant/dataset_hflocal_path.py` (`image-query/`,
  `video-tasks/data/`, `visdoc-tasks/data/`, `gui-tasks/GAE-*`).

Set both if you invoke `eval.py` directly. Datasets whose metadata is not
staged locally fall back to the Hugging Face Hub (set `HTTPS_PROXY` if your
cluster needs one); the gui tasks have no such fallback and require
`MMEB_V3_DATA_DIR`.

## Single-node evaluation

```bash
MODEL_PATH=/path/to/WeMM_Embedding \
DATA_BASEDIR=/path/to/MMEB-V3 \
OUTPUT_DIR=exps/wemm_embedding \
bash scripts/run_eval.sh
```

Useful overrides: `MODALITIES="image video"`, `BATCH_SIZE=8`,
`NPROC_PER_NODE=8`.

Per-modality settings used for our reported numbers:

- `text`: `BATCH_SIZE=1 EXTRA_ARGS="--max_len 262144"` — LongEmbed documents
  are far longer than the 8192-token default truncation (narrativeqa corpus
  docs reach ~476k tokens) and would be cut in half; batch size 1 avoids
  activation-memory spikes on ~40GB GPUs.
- `gui`: `BATCH_SIZE=1 EXTRA_ARGS="--max_len 262144"` — Mind2Web candidates
  pack up to 10 full-page screenshots each (resized to at most 8 MP), so a
  single sample can exceed 100k tokens; larger batches can exceed SDPA kernel
  limits or GPU memory.
- All other modalities run with the defaults (`BATCH_SIZE=8`).

`scripts/run_multinode.sh` already applies these per-modality settings
automatically (overridable via `BS_<MODALITY>` / `MAXLEN_<MODALITY>`).

## Multi-node evaluation

Run on every node with an identical command except `NODE_RANK`
(all nodes must share `OUTPUT_DIR` and `DATA_BASEDIR`, e.g. via NFS):

```bash
MODEL_PATH=/path/to/WeMM_Embedding DATA_BASEDIR=/path/to/MMEB-V3 \
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29677 \
bash scripts/run_multinode.sh
```

Each rank encodes a strided shard of every dataset; embeddings are gathered
across all ranks and rank 0 merges, scores and writes
`<dataset>_score.json` / `<dataset>_pred.jsonl` under
`$OUTPUT_DIR/<modality>/`, exactly as in the official pipeline. Completed
datasets are skipped on re-run (embedding pickles are reused).

## Notes & limitations

- **Audio tasks are not supported** (WeMM-Embedding has no audio tower). The
  default `MODALITIES` in the run scripts therefore exclude `audio`.
- `<embedding>` token: appended automatically by the model-side chat
  template / tokenizer post-processor — do not insert it manually.
- Video inputs follow the WeMM frame-bundle convention: `video_grid_thw` is
  expanded framewise (one grid row per frame) after the processor call.
- The number of frames evaluated per video comes from `num_frames` in the
  dataset yaml (see point 4 above), sampled uniformly from the on-disk frame
  dump; `--video_max_frames` is a training-side argument and does not affect
  evaluation. Per-message vision sampling (`fps=1`, `max_frames=64`) is fixed
  in the `wemm_embedding` processor.
