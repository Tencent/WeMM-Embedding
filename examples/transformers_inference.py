#!/usr/bin/env python3
"""Minimal WeMM-Embedding inference with Transformers."""

import argparse

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode text, image, and video inputs with WeMM-Embedding."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_samples(image_path: str, video_path: str) -> dict[str, list[dict]]:
    return {
        "text": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "A dog is running on a beach.",
                    }
                ],
            }
        ],
        "image": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    }
                ],
            }
        ],
        "video": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    }
                ],
            }
        ],
    }


def encode(processor, model, messages, device: str, dimension: int | None):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)
    else:
        video_metadata = None

    inputs = processor(
        text=prompt,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        embedding = model.embedding(**inputs).float()

    if dimension is not None:
        supported = getattr(model.config, "matryoshka_dimensions", None)
        if supported is not None and dimension not in supported:
            raise ValueError(f"Supported dimensions: {supported}")

        embedding = F.normalize(embedding[..., :dimension], dim=-1)

    return embedding.cpu()


def main() -> None:
    args = parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = model.to(args.device)
    model.eval()

    samples = build_samples(args.image, args.video)

    for modality, messages in samples.items():
        embedding = encode(
            processor,
            model,
            messages,
            device=args.device,
            dimension=args.dimension,
        )
        print(
            modality,
            tuple(embedding.shape),
            embedding.norm(dim=-1).tolist(),
        )


if __name__ == "__main__":
    main()
