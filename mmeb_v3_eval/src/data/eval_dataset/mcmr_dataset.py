import os
from typing import Any, Dict, List

from datasets import Dataset

from src.data.dataset.mcmr_dataset import (
    DEFAULT_QUERY_INSTRUCTION,
    load_mcmr_candidate_rows,
    load_mcmr_query_rows,
    resolve_mcmr_paths,
)
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    ImageVideoInstance,
    RESOLUTION_MAPPING,
    add_metainfo_hook,
)
from src.model.processor import VLM_IMAGE_TOKENS, process_input_text


DATASET_PARSER_NAME = "mcmr"


def create_empty_image_dict(image_resolution):
    return {
        "bytes": [None],
        "paths": [None],
        "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)],
    }


@add_metainfo_hook
def data_prepare_query(batch_dict, *args, **kwargs):
    model_backbone = kwargs["model_backbone"]
    image_resolution = kwargs["image_resolution"]

    query_texts: List[List[str]] = []
    query_images: List[List[Any]] = []
    dataset_infos: List[Dict[str, Any]] = []

    batch_size = len(batch_dict["qry_text"])
    for qry_text, qry_id, label_names in zip(
        batch_dict["qry_text"],
        batch_dict["qry_id"],
        batch_dict.get("label_names", [[]] * batch_size),
    ):
        if not qry_text:
            continue

        query_text = process_input_text("", model_backbone, text=qry_text).strip()
        query_texts.append([query_text])
        query_images.append([create_empty_image_dict(image_resolution)])
        dataset_infos.append({
            "qry_id": qry_id,
            "label_name": [str(x) for x in label_names],
            "cand_names": [],
        })

    # Global retrieval mode: candidates come from corpus.
    cand_texts = [[] for _ in query_texts]
    cand_images = [[] for _ in query_texts]

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos,
    }


@add_metainfo_hook
def data_prepare_candidate(batch_dict, *args, **kwargs):
    model_backbone = kwargs["model_backbone"]
    image_resolution = kwargs["image_resolution"]

    cand_texts: List[List[str]] = []
    cand_images: List[List[Dict[str, Any]]] = []
    dataset_infos: List[Dict[str, Any]] = []

    for cand_text, cand_id, cand_image_path in zip(
        batch_dict["cand_text"],
        batch_dict["cand_id"],
        batch_dict["cand_image_path"],
    ):
        if not cand_text or not cand_image_path:
            continue

        # Layout follows the reference format "<image>\nTitle: ...": the image
        # token is immediately followed by a newline before the text fields.
        fused_text = f"{VLM_IMAGE_TOKENS[model_backbone]}\n{cand_text}".strip()

        cand_texts.append([fused_text])
        cand_images.append([ImageVideoInstance(
            bytes=[None],
            paths=[cand_image_path],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict()])
        dataset_infos.append({"cand_names": [str(cand_id)]})

    return {
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos,
    }


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_mcmr_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", "MCMR")
    subset_name = kwargs.get("subset_name", "MCMR_")
    query_file = kwargs.get("query_file")
    candidate_file = kwargs.get("candidate_file")
    image_root = kwargs.get("image_root")
    data_path = kwargs.get("data_path")
    query_instruction = kwargs.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)

    query_file, candidate_file, image_root = resolve_mcmr_paths(
        data_path=data_path,
        query_file=query_file,
        candidate_file=candidate_file,
        image_root=image_root,
    )

    if not query_file or not os.path.isfile(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")
    if not candidate_file or not os.path.isfile(candidate_file):
        raise FileNotFoundError(f"Candidate file not found: {candidate_file}")
    if not image_root or not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")

    query_data = load_mcmr_query_rows(
        query_file=query_file,
        query_instruction=query_instruction,
    )
    candidate_data = load_mcmr_candidate_rows(
        candidate_file=candidate_file,
        image_root=image_root,
    )

    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)
    if isinstance(num_sample_per_subset, str) and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if isinstance(num_sample_per_subset, int) and num_sample_per_subset > 0:
        query_data = query_data[:num_sample_per_subset]

    qry_dataset = Dataset.from_list(query_data)
    cand_dataset = Dataset.from_list(candidate_data)

    kwargs["model_backbone"] = model_args.model_backbone
    kwargs["image_resolution"] = data_args.image_resolution
    kwargs["global_dataset_name"] = f"{dataset_name}/{subset_name}"

    qry_dataset = qry_dataset.map(
        lambda x: data_prepare_query(x, **kwargs),
        batched=True,
        batch_size=64,
        remove_columns=["qry_id", "qry_text", "label_names"],
        drop_last_batch=False,
    )
    qry_dataset = qry_dataset.select_columns(
        ["query_text", "query_image", "cand_text", "cand_image", "dataset_infos", "global_dataset_name"]
    )

    corpus = cand_dataset.map(
        lambda x: data_prepare_candidate(x, **kwargs),
        batched=True,
        batch_size=128,
        remove_columns=["cand_id", "cand_text", "cand_image_path"],
        drop_last_batch=False,
    )
    corpus = corpus.select_columns(["cand_text", "cand_image", "dataset_infos", "global_dataset_name"])

    print(f"Loaded {dataset_name}/{subset_name}: queries={len(query_data)}, candidates={len(candidate_data)}")
    print(f"query_file={query_file}")
    print(f"candidate_file={candidate_file}")
    print(f"image_root={image_root}")

    return qry_dataset, corpus
