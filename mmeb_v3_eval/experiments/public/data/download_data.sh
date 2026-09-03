#!/usr/bin/env bash

# You may need to install git-lfs to download data properly

set -ex

### Training data

# 1. Change to desired directory for storing data
DATA_BASEDIR=data/vlm2vec_train
mkdir -p "$DATA_BASEDIR"
pushd "$DATA_BASEDIR"

# 2.1 Download image data and unzip
git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-train
pushd MMEB-train
python unzip_file.py
popd

# 2.2 Download LLaVA-Hound video data
sudo apt-get install -y git-lfs
git clone https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/ video
pushd video/

pushd train_300k/
for f in *.tar.gz; do tar -xzvf "$f"; done
popd

pushd train_600k/
for f in *.tar.gz; do tar -xzvf "$f"; done
popd

# 2.3 Process LLaVA-Hound annotation files
# We need to generate `video_qa_240k.jsonl` for training, which is a subset of `video_240k_caption_15k.jsonl`.
pushd video_instruction/train/sft/
if [ -f "video_240k_caption_15k.jsonl" ]; then
    echo "Generating video_qa_240k.jsonl..."
    head -n 240000 video_240k_caption_15k.jsonl > video_qa_240k.jsonl
else
    echo "Warning: video_240k_caption_15k.jsonl not found. Skipping video_qa_240k.jsonl generation."
fi
popd

popd

# 3. Adjust the data path in the data config yaml, refer to `experiments/public/train/train_alltasks.yaml`

popd

### Eval data

DATA_BASEDIR=data/vlm2vec_eval
mkdir -p "$DATA_BASEDIR"
pushd "$DATA_BASEDIR"

git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-V2 .
pushd image-tasks/
tar -xzvf mmeb_v1.tar.gz
mv MMEB/* .
rmdir MMEB
popd

pushd video-tasks/frames/
mkdir video_cls
mkdir video_ret
# `video_mret` and `video_qa` are part of the archives already so they'll be created.
tar -xzvf video_cls.tar.gz -C video_cls
tar -xzvf video_mret.tar.gz
tar -xzvf video_ret.tar.gz -C video_ret
cat video_qa.tar.gz-0{0..4} | tar -xzv
popd

popd
