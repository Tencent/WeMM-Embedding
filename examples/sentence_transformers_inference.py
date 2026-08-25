#!/usr/bin/env python3
"""Minimal WeMM-Embedding inference with Sentence Transformers."""

import argparse

from sentence_transformers import SentenceTransformer


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


def build_samples(image_path: str, video_path: str) -> dict[str, object]:
    return {
        "text": "A dog is running on a beach.",
        "image": {"image": image_path},
        "video": {"video": video_path},
    }


def main() -> None:
    args = parse_args()

    model = SentenceTransformer(
        args.model,
        trust_remote_code=True,
        device=args.device,
    )

    if args.dimension is not None:
        supported = getattr(
            model[0].auto_model.config, "matryoshka_dimensions", None
        )
        if supported is not None and args.dimension not in supported:
            raise ValueError(f"Supported dimensions: {supported}")

    samples = build_samples(args.image, args.video)

    embeddings = model.encode(
        list(samples.values()),
        batch_size=1,
        truncate_dim=args.dimension,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )

    for modality, embedding in zip(samples, embeddings):
        print(
            modality,
            tuple(embedding.shape),
            embedding.norm(dim=-1).item(),
        )


if __name__ == "__main__":
    main()
