#!/usr/bin/env bash
# Download and prepare the MMEB-V3 evaluation data for WeMM-Embedding.
#
# This script fetches the raw archives from the official Hugging Face repos
# (pinned revisions) and then extracts them into the eval-ready layout via
# experiments/public/data/dataset_setup_v3.py.
#
# Usage:
#   DATA_ROOT=/path/to/MMEB-V3 bash scripts/download_data.sh
#
# Notes:
#   - Audio tasks are excluded (WeMM-Embedding has no audio tower), as are
#     omniset.tar.gz and mscoco_omni (not used by the MMEB-V3 eval configs).
#   - video_mret frames come from TIGER-Lab/MMEB-V2's single archive
#     (video-tasks/frames/video_mret.tar.gz), matching the official VLM2Vec
#     MMEB-v2 layout; the MMEB-V3 split parts (video_mret.tar.gz-*) are
#     therefore excluded from the V3 download.
#   - Video QA / retrieval / classification frames (video_qa.tar.gz-*,
#     video_ret.tar.gz, video_cls.tar.gz) and the image-query metadata come
#     from VLM2Vec/MMEB-V3 (its image-query/ tree mirrors
#     ziyjiang/MMEB_Test_Instruct).
#   - Dataset metadata (annotations, qrels) lands under the same DATA_ROOT and
#     is picked up via MMEB_V3_DATA_DIR, which the run scripts set from
#     DATA_BASEDIR. Anything not staged locally falls back to the Hugging Face
#     Hub at runtime (set HTTPS_PROXY if your cluster needs one).
#   - Requires the `hf` CLI (pip install "huggingface_hub[cli]") and ~150GB
#     of free disk for the full download plus extraction.
set -euo pipefail

V3_REPO="VLM2Vec/MMEB-V3"
V3_REVISION="4a5560b2b64384204b6fea8a82ea986eba51f5aa"
V2_REPO="TIGER-Lab/MMEB-V2"
V2_REVISION="e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1"
V2_IMAGE_FILE="image-tasks/mmeb_v1.tar.gz"
V2_VIDEO_MRET_FILE="video-tasks/frames/video_mret.tar.gz"

DATA_ROOT="${DATA_ROOT:-data/MMEB-V3}"
MAX_WORKERS="${HF_MAX_WORKERS:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

V2_SOURCE_ROOT="$DATA_ROOT/.sources/MMEB-V2-$V2_REVISION"

mkdir -p "$DATA_ROOT" "$V2_SOURCE_ROOT"

echo "==> [1/3] Downloading $V3_REPO @ $V3_REVISION (excluding audio/omniset/mret-parts)"
hf download "$V3_REPO" \
    --repo-type dataset \
    --revision "$V3_REVISION" \
    --local-dir "$DATA_ROOT" \
    --exclude 'audio_tasks/**' \
    --exclude 'omniset.tar.gz' \
    --exclude 'mscoco_omni/**' \
    --exclude 'video_tasks/frames/video_mret.tar.gz-*' \
    --max-workers "$MAX_WORKERS"

echo "==> [2/3] Downloading $V2_REPO @ $V2_REVISION ($V2_IMAGE_FILE, $V2_VIDEO_MRET_FILE)"
hf download "$V2_REPO" \
    "$V2_IMAGE_FILE" \
    "$V2_VIDEO_MRET_FILE" \
    --repo-type dataset \
    --revision "$V2_REVISION" \
    --local-dir "$V2_SOURCE_ROOT" \
    --max-workers "$MAX_WORKERS"

# Wire the V2 archives into the layout that dataset_setup_v3.py expects.
# - mmeb_v1.tar.gz: consumed from the raw image_tasks/ dir (skipped if V3
#   already shipped one).
if [ ! -e "$DATA_ROOT/image_tasks/mmeb_v1.tar.gz" ]; then
    ln -s "$V2_SOURCE_ROOT/$V2_IMAGE_FILE" "$DATA_ROOT/image_tasks/mmeb_v1.tar.gz"
fi
# - video_mret: the setup script only handles MMEB-V3's split parts, so the
#   V2 single archive is extracted here directly; the setup script then sees
#   video-tasks/frames/video_mret/ already present and marks it done.
if [ ! -d "$DATA_ROOT/video-tasks/frames/video_mret" ]; then
    mkdir -p "$DATA_ROOT/video-tasks/frames"
    echo "==> Extracting V2 video_mret.tar.gz -> video-tasks/frames/"
    tar -xzf "$V2_SOURCE_ROOT/$V2_VIDEO_MRET_FILE" -C "$DATA_ROOT/video-tasks/frames/"
fi

echo "==> [3/3] Extracting raw archives into the eval-ready layout"
python3 "$REPO_ROOT/experiments/public/data/dataset_setup_v3.py" \
    --root "$DATA_ROOT"

# Visdoc page-level tasks: the MMEB-V3 visdoc-tasks.data.tar.gz ships the
# original (buggy) ViDoSeek-page / MMLongBench-page BEIR data. VLM2Vec fixed
# these in separate HF repos; overlay them so the local-path loader picks up
# the fixed corpus/queries/qrels.
echo "==> [fix] Overlaying VLM2Vec/ViDoSeek-page-fixed and VLM2Vec/MMLongBench-page-fixed"
hf download "VLM2Vec/ViDoSeek-page-fixed" \
    --repo-type dataset \
    --local-dir "$DATA_ROOT/visdoc-tasks/data/ViDoSeek-page" \
    --max-workers "$MAX_WORKERS"
hf download "VLM2Vec/MMLongBench-page-fixed" \
    --repo-type dataset \
    --local-dir "$DATA_ROOT/visdoc-tasks/data/MMLongBench" \
    --max-workers "$MAX_WORKERS"

echo "==> Data preparation finished. Sanity check:"
python3 "$REPO_ROOT/experiments/public/data/dataset_setup_v3.py" \
    --root "$DATA_ROOT" --check-only || true
echo "    (audio-tasks / omniset entries reported as missing are expected:"
echo "     audio is not supported and omniset is not part of the eval configs)"
