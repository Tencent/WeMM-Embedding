<h1 align="center">WeMM-Embedding: WeChat Multi-Modal Embedding</h1>

<p align="center">
  <a href="https://huggingface.co/collections/tencent/wemm-embedding">
    <img src="https://img.shields.io/badge/🤗-Hugging%20Face-yellow" alt="Hugging Face">
  </a>
  <a href="https://arxiv.org/abs/2608.24053">
    <img src="https://img.shields.io/badge/📄-Technical%20Report-red" alt="Technical Report">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

WeMM-Embedding is a family of universal multimodal embedding models developed by the WeChat Vision team. It provides unified representations for text, images, videos, visual documents, and interleaved multimodal inputs, achieving state-of-the-art performance across multiple benchmarks covering diverse tasks and domains.

<p align="center">
  <a href="assets/performance-overview.pdf">
    <img src="assets/performance-overview.png" width="100%" alt="WeMM-Embedding Performance Overview">
  </a>
</p>

## Model Zoo

| Model | Matryoshka dimensions | Hugging Face |
| --- | --- | --- |
| WeMM-Embedding-2B | `64, 128, 256, 512, 1024, 2048` | [🤗 Link](https://huggingface.co/tencent/WeMM-Embedding-2B) |
| WeMM-Embedding-4B | `64, 128, 256, 512, 1024, 2560` | [🤗 Link](https://huggingface.co/tencent/WeMM-Embedding-4B) |
| WeMM-Embedding-9B | `64, 128, 256, 512, 1024, 2048, 4096` | [🤗 Link](https://huggingface.co/tencent/WeMM-Embedding-9B) |

All models support text, images, videos, visual documents, and interleaved multimodal inputs. Embeddings are obtained from the last-layer hidden state at the dedicated `<embedding>` token position, followed by L2 normalization. Audio input is not currently supported.


## Installation

```bash
pip install -r requirements.txt
```

## Transformers
We recommend using `transformers==5.2.0` for inference and reproducibility, as newer versions may differ in preprocessing behavior.

```bash
python examples/transformers_inference.py \
  --model /path/to/WeMM-Embedding-2B \
  --image /path/to/image.jpg \
  --video /path/to/video.mp4 \
  --dimension 2048
```

The example produces independent text, image, and video embeddings. Omit `--dimension` for the full embedding dimension.

## Sentence Transformers

```bash
python examples/sentence_transformers_inference.py \
  --model /path/to/WeMM-Embedding-2B \
  --image /path/to/image.jpg \
  --video /path/to/video.mp4 \
  --dimension 2048
```

`SentenceTransformer` loads the model directly, so a Hugging Face model id such as `tencent/WeMM-Embedding-2B` also works in place of a local path. Text, image, and video inputs go through `SentenceTransformer.encode()`, and MRL is selected with `--dimension`.

## Serving

Tested versions: vLLM `0.27.0` and SGLang `0.5.9`.

vLLM:

```bash
MODEL_PATH=/path/to/WeMM-Embedding-2B
vllm serve "$MODEL_PATH" \
  --runner pooling \
  --chat-template "$MODEL_PATH/embedding_chat_template.jinja"
```

SGLang:

```bash
MODEL_PATH=/path/to/WeMM-Embedding-2B
python scripts/patch_sglang_video.py
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --is-embedding \
  --enable-precise-embedding-interpolation
```

Equivalent one-command wrappers are available in `scripts/serve_vllm.sh` and `scripts/serve_sglang.sh`.

## Matryoshka Embeddings

For a supported dimension `d`, truncate the full embedding and normalize it again:

```python
embedding = torch.nn.functional.normalize(embedding[..., :d], dim=-1)
```

On MMEB-v2, the 2B model at 256 dimensions retains 98.7% of its full-dimensional image and video performance.

## Evaluation

### MMEB-v2

Results on 78 datasets from Table 1 of the [technical report](assets/WeMM_Embedding_tech_report.pdf). Image and video tasks use Hit@1, while visual-document tasks use NDCG@5. Higher is better.

| Model | Size | AVG | Image | Video | VisDoc |
| --- | ---: | ---: | ---: | ---: | ---: |
| VLM2Vec | 2B | 47.8 | 59.7 | 29.0 | 44.0 |
| GME | 2B | 55.4 | 51.9 | 33.9 | 76.8 |
| VLM2Vec-V2 | 2B | 59.3 | 64.9 | 34.9 | 69.2 |
| Qwen3-VL-Embedding | 2B | 73.2 | 75.0 | 61.9 | 79.2 |
| DME-Small† | 2B | 74.8 | 75.9 | 65.6 | 79.9 |
| **WeMM-Embedding** | **2B** | **77.9** | **79.6** | **70.8** | **80.7** |
| **WeMM-Embedding** | **4B** | **79.2** | **80.8** | **72.1** | **82.0** |
| VLM2Vec | 8B | 53.2 | 65.5 | 34.0 | 49.1 |
| GME | 8B | 59.2 | 56.0 | 38.6 | 79.3 |
| Qwen3-VL-Embedding | 8B | 77.8 | 80.1 | 67.1 | 82.4 |
| DME-Medium† | 9B | 78.4 | 79.8 | 70.8 | 82.0 |
| **WeMM-Embedding** | **9B** | **80.6** | **81.9** | **74.3** | **83.3** |

† Closed-source leaderboard submission without publicly released model weights or a public inference endpoint.

### MMEB-v3

Results on all 190 tasks from Table 2 of the [technical report](assets/WeMM_Embedding_tech_report.pdf). V3-All includes the 78 MMEB-v2 tasks, 53 text tasks, 47 agent tasks, 11 audio tasks, and MCMR. Unsupported tasks are assigned a score of zero.

| Model | Size | V3-All | Text | Agent | MCMR | Audio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VLM2Vec-V2 | 2B | 38.3 | 24.5 | 28.7 | 4.1 | 0.0 |
| Omni-Embed-Nemotron | 3B | 43.5 | 39.2 | 36.5 | 26.1 | 36.5 |
| E5-Omni | 3B | 44.6 | 26.7 | 36.9 | 31.9 | 30.8 |
| Qwen3-VL-Embedding | 2B | 50.9 | 39.2 | 39.3 | 42.0 | 0.0 |
| **WeMM-Embedding** | **2B** | **56.0** | **45.3** | **45.1** | **42.5** | **0.0** |
| **WeMM-Embedding** | **4B** | **58.2** | **47.9** | **49.0** | **41.9** | **0.0** |
| WAVE | 7B | 26.3 | 13.7 | 11.3 | 8.9 | 31.8 |
| VLM2Vec | 8B | 32.9 | 22.2 | 19.7 | 0.9 | 0.0 |
| LCO-Embedding-Omni | 7B | 40.6 | 32.4 | 27.8 | 20.0 | 43.2 |
| GME | 8B | 43.6 | 37.1 | 35.6 | 27.3 | 0.0 |
| E5-Omni | 7B | 47.1 | 26.9 | 36.7 | 41.1 | 43.0 |
| Tianmu-Emb-Uni | 8B | 53.3 | 43.6 | 39.4 | 38.8 | 38.9 |
| Qwen3-VL-Embedding | 8B | 53.5 | 42.5 | 38.4 | 38.0 | 0.0 |
| **WeMM-Embedding** | **9B** | **59.5** | **48.8** | **51.0** | **49.3** | **0.0** |

Text results use NDCG@5; agent, MCMR, and audio results use Hit@1.

`mmeb_v3_eval/` contains the MMEB-v3 evaluation code used to produce our reported numbers. It is the official [TIGER-AI-Lab/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) pipeline with a minimal diff: multi-node multi-GPU inference (`torchrun --nnodes=N`), a `wemm_embedding` backbone implementing our preprocessing and batched inference, dataset instructions aligned with the released model, and 64-frame video sampling. Data download, single-node and multi-node commands are documented in `mmeb_v3_eval/README.md`.

```bash
cd mmeb_v3_eval
DATA_ROOT=/path/to/MMEB-V3 bash scripts/download_data.sh
MODEL_PATH=/path/to/WeMM-Embedding-2B DATA_BASEDIR=/path/to/MMEB-V3 \
OUTPUT_DIR=exps/wemm_embedding bash scripts/run_eval.sh
```

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```bibtex
@article{wemm-embedding,
      title={WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report}, 
      author={Junjie Zhou and Ke Mei and Lei Li and Tianyi Wang and Fengyun Rao and Jing Lyu},
      year={2026},
      eprint={2608.24053},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.24053}, 
}
```

## License

Unless otherwise noted, Tencent-authored code in this repository is released under the
[Apache License 2.0](LICENSE).

Third-party components retain their original licenses and copyright notices. Please review
the corresponding source files before use.
