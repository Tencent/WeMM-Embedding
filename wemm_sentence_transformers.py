#!/usr/bin/env python3
"""Sentence-Transformers adapter for WeMM-Embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import InputModule
from transformers import AutoModel, AutoProcessor


def _as_messages(
    value: Any,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(value, str):
        text = f"{prompt or ''}{value}"
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    if isinstance(value, list):
        if not value or not all(
            isinstance(message, dict) and "role" in message for message in value
        ):
            raise TypeError("A list input must be one chat-style conversation.")
        messages = [dict(message) for message in value]
        if prompt:
            messages.insert(0, {"role": "user", "content": prompt})
        return messages

    if not isinstance(value, dict):
        raise TypeError(
            "Input must be text, a chat conversation, or a multimodal dict."
        )
    if "messages" in value:
        return _as_messages(value["messages"], prompt=prompt)

    content: list[dict[str, Any]] = []
    for modality in ("image", "video"):
        if value.get(modality) is not None:
            content.append({"type": modality, modality: value[modality]})
    if value.get("text") is not None:
        content.append(
            {"type": "text", "text": f"{prompt or ''}{value['text']}"}
        )
    elif prompt:
        content.append({"type": "text", "text": prompt})
    if not content:
        raise ValueError("Input must contain text, image, or video.")
    return [{"role": "user", "content": content}]


class WeMMInputModule(InputModule):
    config_keys = ["model_name_or_path"]
    save_in_root = False

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | torch.device | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        self.auto_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.embedding_token_id = self.tokenizer.convert_tokens_to_ids(
            "<embedding>"
        )
        text_config = getattr(
            self.auto_model.config,
            "text_config",
            self.auto_model.config,
        )
        self.embedding_dimension = int(text_config.hidden_size)

    @property
    def modalities(self):
        return [
            "text",
            "image",
            "video",
            "message",
            ("image", "text"),
            ("text", "video"),
        ]

    @property
    def max_seq_length(self) -> int:
        return int(self.tokenizer.model_max_length)

    def preprocess(
        self,
        inputs: list[Any],
        prompt: str | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | Any]:
        conversations = [
            _as_messages(value, prompt=prompt)
            for value in inputs
        ]
        prompts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in conversations
        ]
        images, videos, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            video_metadata = None

        features = self.processor(
            text=prompts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        input_ids = features["input_ids"]
        positions = features["attention_mask"].sum(dim=1) - 1
        batch = torch.arange(input_ids.shape[0])
        terminal_ids = input_ids[batch, positions]
        if not torch.all(terminal_ids == self.embedding_token_id):
            raise RuntimeError("Each input must end with <embedding>.")
        features["modality"] = "message"
        return dict(features)

    def forward(
        self,
        features: dict[str, torch.Tensor | Any],
        **_: Any,
    ) -> dict[str, torch.Tensor | Any]:
        model_inputs = {
            key: value
            for key, value in features.items()
            if key != "modality"
        }
        inner = getattr(self.auto_model, "model", None)
        if inner is not None and hasattr(inner, "rope_deltas"):
            inner.rope_deltas = None
        features["sentence_embedding"] = self.auto_model.embedding(
            **model_inputs
        ).float()
        return features

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension

    def save(
        self,
        output_path: str,
        *args: Any,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        del args, safe_serialization, kwargs
        self.save_config(output_path)


def load_wemm_sentence_transformer(
    model_name_or_path: str | Path,
    *,
    device: str | torch.device = "cuda:0",
) -> SentenceTransformer:
    return SentenceTransformer(
        modules=[WeMMInputModule(str(model_name_or_path), device=device)],
        device=str(device),
        similarity_fn_name="cosine",
    )
