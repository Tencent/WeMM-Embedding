#!/usr/bin/env python3

import argparse

from wemm_sentence_transformers import load_wemm_sentence_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model = load_wemm_sentence_transformer(args.model, device=args.device)
    inputs = [
        "A dog is running on a beach.",
        {"image": args.image, "text": "A dog is running on a"
                                      " beach."},  # text is optional; can add arbitrary caption/query text
        {"video": args.video},
    ]
    embeddings = model.encode(
        inputs,
        batch_size=1,
        truncate_dim=args.dimension,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    print("shape:", tuple(embeddings.shape))
    print("norms:", embeddings.norm(dim=1).cpu().tolist())


if __name__ == "__main__":
    main()
