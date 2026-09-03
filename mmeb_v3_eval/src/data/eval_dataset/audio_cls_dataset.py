"""
Data processing utilities for audio classification tasks (shared by training and evaluation).
"""

import glob
import os
from typing import Any, Dict, List, Tuple

import datasets
import numpy as np
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset


def _get_label_fields(batch: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Pick an available label field name and return its value list.
    Prefer integer labels, then fall back to string labels.
    """
    int_cands = ["label", "target", "classID", "class_id", "instrument_family", "fold", "emotion_id"]
    str_cands = [
        "category",
        "label_name",
        "class",
        "classname",
        "instrument_family_str",
        "emotion",
        "major_emotion",
    ]
    for name in int_cands:
        if name in batch:
            return name, batch[name]
    for name in str_cands:
        if name in batch:
            return name, batch[name]
    raise ValueError(f"No label field found in batch keys={list(batch.keys())}")


def _extract_audio_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize audio into a {path?, bytes?} dict for the collator to read.
    Supports the HuggingFace datasets array format.
    """
    if "audio" in row and isinstance(row["audio"], dict):
        audio = row["audio"]
        audio_path = audio.get("path")
        audio_bytes = audio.get("bytes")

        if audio_bytes is None and "array" in audio:
            import io
            import soundfile as sf

            array = audio["array"]
            sampling_rate = int(audio.get("sampling_rate", 16000))

            if isinstance(array, list):
                array = np.array(array)
            array = np.asarray(array, dtype=np.float32)
            if array.ndim > 1:
                array = array.squeeze()

            # torch 2.11 torchaudio.save routes through torchcodec and cannot encode to an
            # in-memory BytesIO; use soundfile to materialise WAV bytes from the decoded array.
            buffer = io.BytesIO()
            sf.write(buffer, array, sampling_rate, format="WAV", subtype="PCM_16")
            audio_bytes = buffer.getvalue()

    else:
        audio_path = row.get("audio_path") or row.get("path") or None
        audio_bytes = row.get("audio_bytes") or None

    return {"path": audio_path, "bytes": audio_bytes}


def data_prepare(batch_dict, **kwargs):
    """
    Output fields:
    - query_audio: list[dict]  {path?, bytes?}
    - query_audio_path: list[str|None] audio path placeholder (None if unavailable)
    - query_text: empty-string placeholder
    - query_image: None placeholder
    - cand_text: all class names (reused within the same batch)
    - cand_image: list of None placeholders (same length as cand_text)
    - dataset_infos: {label_id, label_name, cand_names}
    """
    label_field, labels = _get_label_fields(batch_dict)
    if kwargs.get("label_field_override") and kwargs["label_field_override"] in batch_dict:
        label_field = kwargs["label_field_override"]
        labels = batch_dict[label_field]

    label_names = None
    if kwargs.get("label_name_field_override") and kwargs["label_name_field_override"] in batch_dict:
        label_names = batch_dict[kwargs["label_name_field_override"]]
    if label_names is None:
        for key in ["category", "label_name", "class", "classname", "instrument_family_str",
                    "emotion", "major_emotion"]:
            if key in batch_dict:
                label_names = batch_dict[key]
                break
    if label_names is None:
        label_names = [str(x) for x in labels]

    all_label_names = kwargs["all_label_names"]
    label2id = {name: idx for idx, name in enumerate(all_label_names)}

    query_audio, query_audio_paths, query_texts, query_images, cand_texts, cand_images, dataset_infos = (
        [], [], [], [], [], [], [])
    for lbl, lbl_name, row_idx in zip(labels, label_names, range(len(labels))):
        row_dict = {k: v[row_idx] for k, v in batch_dict.items()}
        audio_obj = _extract_audio_obj(row_dict)
        query_audio.append(audio_obj)
        q_path = audio_obj.get("path") or row_dict.get("path") or row_dict.get("audio_path") or row_dict.get("filename")
        query_audio_paths.append(q_path)
        query_text = build_query_text(kwargs["dataset_name"])
        assert isinstance(
            query_text,
            list) and len(query_text) == 1 and isinstance(
            query_text[0],
            str) and query_text[0].strip()
        query_texts.append(query_text)
        query_images.append([None])
        cand_texts.append(all_label_names)
        cand_images.append([None] * len(all_label_names))
        lid = label2id.get(lbl_name, int(lbl) if isinstance(lbl, int) else 0)
        dataset_infos.append({"label_id": lid, "label_name": lbl_name, "cand_names": all_label_names})

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audio,
        "query_audio_path": query_audio_paths,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos,
    }


# -------- NSynth --------
def _load_nsynth_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    if "-1k" in dataset_path:
        data_dir = os.path.join(dataset_path, "eval")
        query_file = os.path.join(data_dir, "query.parquet")
        if os.path.exists(query_file):
            parquet_files = [query_file]
        else:
            raise FileNotFoundError(f"query.parquet not found under {data_dir}")
    else:
        data_dir = os.path.join(dataset_path, "data")
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"NSynth parquet files not found under {data_dir}")

    split_name = split or "train"
    ds_dict = datasets.load_dataset("parquet", data_files={split_name: parquet_files})
    return ds_dict[split_name]


def _build_nsynth_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    int_candidates = [
        ("label_id", "label"),
        ("instrument_family", "instrument_family_str"),
        ("instrument_family_id", "instrument_family_str"),
        ("label", "label_name"),
        ("target", "label_name"),
        ("class_id", "class"),
        ("classID", "class"),
    ]
    str_candidates = ["instrument_family_str", "label_name", "category", "class"]

    label_id_field = None
    label_name_field = None
    for int_field, str_field in int_candidates:
        if int_field in dataset.column_names:
            label_id_field = int_field
            if str_field in dataset.column_names:
                label_name_field = str_field
            break

    if label_id_field:
        label_ids = [int(x) for x in dataset[label_id_field]]
        if label_name_field:
            label_names = dataset[label_name_field]
        else:
            label_names = [str(x) for x in label_ids]
        id_to_name = {}
        for lid, lname in zip(label_ids, label_names):
            lid = int(lid)
            id_to_name.setdefault(lid, str(lname))
        max_id = max(id_to_name.keys())
        all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
        return all_label_names, label_id_field, label_name_field

    for str_field in str_candidates:
        if str_field in dataset.column_names:
            label_names = dataset[str_field]
            label_name_field = str_field
            break
    else:
        raise ValueError(f"NSynth: no usable label field in {dataset.column_names}")

    all_label_names = sorted(list(set(label_names)))
    return all_label_names, None, label_name_field


# -------- ESC-50 --------
def _load_esc50_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
    if not parquet_files:
        # MMEB-V3 local cache stores ESC-50 shards under esc50/data/.
        parquet_files = sorted(glob.glob(os.path.join(dataset_path, "data", "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"ESC-50 parquet files not found under {dataset_path}")
    split_name = split or "train"
    ds_dict = datasets.load_dataset("parquet", data_files={split_name: parquet_files})
    return ds_dict[split_name]


def _build_esc50_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    targets = dataset["target"]
    categories = dataset["category"]
    id_to_name = {}
    for tid, cat in zip(targets, categories):
        tid = int(tid)
        id_to_name.setdefault(tid, str(cat))
    max_id = max(id_to_name.keys())
    all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
    return all_label_names, "target", "category"


# -------- UrbanSound8K --------
def _load_urbansound8k_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    csv_path = os.path.join(dataset_path, "csv_files", "test.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"UrbanSound8K csv not found: {csv_path}")
    import pandas as pd

    df = pd.read_csv(csv_path)
    filename_col = None
    for cand in ("slice_file_name", "slice_filename", "file_name", "filename"):
        if cand in df.columns:
            filename_col = cand
            break
    if filename_col is None and "path" not in df.columns:
        raise ValueError(f"UrbanSound8K csv missing slice_file_name (columns={list(df.columns)})")

    if "path" not in df.columns:
        def _map_path(x):
            fold = x["fold"]
            fname = x[filename_col]
            return os.path.join(dataset_path, "audio", f"fold{fold}", fname)

        df["path"] = df.apply(_map_path, axis=1)
    else:
        df["path"] = df["path"].apply(
            lambda p: p if os.path.isabs(p) else os.path.join(dataset_path, p)
        )

    dataset = datasets.Dataset.from_pandas(df)
    return dataset


def _build_urbansound8k_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    id_field = None
    if "classID" in dataset.column_names:
        id_field = "classID"
    if id_field is None:
        # Fallback: classname only (no numeric labels provided)
        name_field = "class" if "class" in dataset.column_names else "classname"
        if name_field not in dataset.column_names:
            raise ValueError(f"UrbanSound8K: no class name field in {dataset.column_names}")
        label_names = [str(x) for x in dataset[name_field]]
        all_label_names = sorted(set(label_names))
        return all_label_names, name_field, name_field

    if "class" in dataset.column_names:
        name_field = "class"
    elif "classname" in dataset.column_names:
        name_field = "classname"
    else:
        raise ValueError(f"UrbanSound8K: no class name field in {dataset.column_names}")

    label_ids = [int(x) for x in dataset[id_field]]
    label_names = [str(x) for x in dataset[name_field]]
    id_to_name = {}
    for lid, lname in zip(label_ids, label_names):
        if lid not in id_to_name:
            id_to_name[lid] = lname
    max_id = max(id_to_name.keys())
    all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
    return all_label_names, id_field, name_field


# -------- CREMA-D --------
def _load_cremad_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    import pandas as pd

    # expected CSV: "AudioWAV/xxxx.wav", "Emotion"
    csv_path = os.path.join(dataset_path, "processed.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "path" not in df.columns:
            df["path"] = df["file"].apply(lambda x: os.path.join(dataset_path, "AudioWAV", x))
        dataset = datasets.Dataset.from_pandas(df)
        return dataset

    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "data", "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"CREMA-D processed.csv or parquet not found under {dataset_path}")
    ds_dict = datasets.load_dataset("parquet", data_files={"data": parquet_files})
    return ds_dict["data"]


def _build_cremad_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    cols = dataset.column_names
    if "label" in cols:
        id_field = "label"
    else:
        id_field = None

    if "label_name" in cols:
        name_field = "label_name"
    elif "major_emotion" in cols:
        name_field = "major_emotion"
    elif "emotion" in cols:
        name_field = "emotion"
    elif "label" in cols:
        name_field = "label"
    else:
        raise ValueError(f"CREMA-D: no label name field in {cols}")

    if id_field is not None:
        label_ids = [int(x) for x in dataset[id_field]]
        label_names = [str(x) for x in dataset[name_field]]
        id_to_name = {}
        for lid, lname in zip(label_ids, label_names):
            if lid not in id_to_name:
                id_to_name[lid] = lname
        max_id = max(id_to_name.keys())
        all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
        return all_label_names, id_field, name_field

    label_names = [str(x) for x in dataset[name_field]]
    all_label_names = sorted(set(label_names))
    return all_label_names, name_field, name_field


# -------- SpeechCommands --------
def _load_speechcommand_dataset(path_info: Tuple[str, str, str]) -> datasets.Dataset:
    dataset_path, subset, split = path_info
    split_name = (split or "train").lower()
    if split_name in {"test", "eval"}:
        parquet_path = os.path.join(dataset_path, "query_eval.parquet")
    else:
        parquet_path = os.path.join(dataset_path, "query_train.parquet")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"SpeechCommands parquet not found: {parquet_path}")

    ds_dict = datasets.load_dataset("parquet", data_files={"data": [parquet_path]})
    return ds_dict["data"]


def _build_speechcommand_label_pool(dataset: datasets.Dataset) -> Tuple[List[str], str, str]:
    cols = dataset.column_names

    if "label_id" in cols:
        id_field = "label_id"
    elif "label" in cols:
        id_field = "label"
    else:
        id_field = None

    if "label_name" in cols:
        name_field = "label_name"
    elif "label" in cols:
        name_field = "label"
    else:
        raise ValueError(f"SpeechCommands: missing label name field, columns={cols}")

    if id_field is not None:
        label_ids = [int(x) for x in dataset[id_field]]
        label_names = [str(x) for x in dataset[name_field]]
        id_to_name = {}
        for lid, lname in zip(label_ids, label_names):
            if lid not in id_to_name:
                id_to_name[lid] = lname
        max_id = max(id_to_name.keys())
        all_label_names = [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
        return all_label_names, id_field, name_field

    all_label_names = sorted(list(set(str(x) for x in dataset[name_field])))
    return all_label_names, None, name_field


def build_audio_cls_dataset(dataset_name: str, path_info: Tuple[str, str, str], **kwargs):
    """
    Build the retrieval-style audio classification evaluation dataset; returns (dataset, corpus).
    """
    if dataset_name == "NSynth":
        dataset = _load_nsynth_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_nsynth_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        if label_field_override:
            kwargs["label_field_override"] = label_field_override
        if label_name_field_override:
            kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "ESC-50":
        dataset = _load_esc50_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_esc50_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        kwargs["label_field_override"] = label_field_override
        kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "UrbanSound8K":
        dataset = _load_urbansound8k_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_urbansound8k_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        if label_field_override:
            kwargs["label_field_override"] = label_field_override
        if label_name_field_override:
            kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "CREMA-D":
        dataset = _load_cremad_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_cremad_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        if label_field_override:
            kwargs["label_field_override"] = label_field_override
        if label_name_field_override:
            kwargs["label_name_field_override"] = label_name_field_override
    elif dataset_name == "SpeechCommands":
        dataset = _load_speechcommand_dataset(path_info)
        all_label_names, label_field_override, label_name_field_override = _build_speechcommand_label_pool(dataset)
        kwargs["all_label_names"] = all_label_names
        kwargs["label_field_override"] = label_field_override
        kwargs["label_name_field_override"] = label_name_field_override
    else:
        dataset = load_hf_dataset(path_info)
        dataset = sample_dataset(dataset, **kwargs)

        label_field, labels = _get_label_fields(dataset)
        label_names = None
        for key in ["category", "label_name", "class", "instrument_family_str"]:
            if key in dataset.column_names:
                label_names = dataset[key]
                break
        if label_names is None:
            label_names = [str(x) for x in dataset[label_field]]
        all_label_names = sorted(list(set(label_names)))

        kwargs["all_label_names"] = all_label_names
        kwargs.pop("label_field_override", None)
        kwargs.pop("label_name_field_override", None)

    kwargs["dataset_name"] = dataset_name

    # If the audio column was auto-cast to a datasets.Audio feature (e.g. CREMA-D's
    # struct<bytes,path>), accessing it would trigger a torchcodec decode that is broken
    # in this env. Cast to decode=False so _extract_audio_obj gets raw {bytes,path} and
    # the collator decodes via soundfile.
    if "audio" in dataset.column_names and isinstance(dataset.features.get("audio"), datasets.Audio):
        dataset = dataset.cast_column("audio", datasets.Audio(decode=False))

    dataset = sample_dataset(dataset, **kwargs)

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=256,
        drop_last_batch=False,
        load_from_cache_file=False,
    )
    dataset = dataset.select_columns(
        [
            "query_text",
            "query_image",
            "query_audio",
            "query_audio_path",
            "cand_text",
            "cand_image",
            "dataset_infos",
        ]
    )
    corpus = None
    return dataset, corpus


DATASET_PARSER_NAME = "audio_cls"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_audio_cls_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.pop("dataset_name")
    if dataset_name in EVAL_DATASET_LOCAL_PATH:
        path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
    else:
        path_info = EVAL_DATASET_HF_PATH[dataset_name]

    return build_audio_cls_dataset(dataset_name, path_info, **kwargs)
