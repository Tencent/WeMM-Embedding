"""
SpeechCOCO evaluation data preparation (audio -> image retrieval)
- cand_image comes from image.bytes in the corpus parquet (no image root dir needed)
- queries do not carry the candidate pool, avoiding CPU/memory blowups
- corpus follows the schema expected by the framework: ['cand_text','cand_image','dataset_infos']
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import datasets

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.model.processor import process_input_text


def _extract_audio_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    # query parquet schema: audio={"bytes","path"}
    audio = row.get("audio", None)
    if isinstance(audio, dict):
        return {"path": audio.get("path"), "bytes": audio.get("bytes")}
    return {"path": row.get("audio_path") or row.get("path"), "bytes": row.get("audio_bytes")}


def _image_to_cand_image_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    # corpus parquet schema: image={"bytes","path"}
    img = row.get("image", None)
    if not isinstance(img, dict):
        raise ValueError("[SpeechCOCO] corpus row['image'] is not a dict")
    b = img.get("bytes", None)
    if b is None:
        raise ValueError("[SpeechCOCO] corpus image.bytes is None")
    return {"paths": [None], "bytes": [b], "resolutions": [None]}


def _as_dataset(obj: Any, prefer_split: Optional[str] = None) -> datasets.Dataset:
    if isinstance(obj, datasets.Dataset):
        return obj
    if isinstance(obj, datasets.DatasetDict):
        if prefer_split and prefer_split in obj:
            return obj[prefer_split]
        keys = list(obj.keys())
        if not keys:
            raise ValueError("Empty DatasetDict")
        return obj[keys[0]]
    raise TypeError(f"Unsupported dataset object type: {type(obj)}")


def _find_local_parquet_files(dataset_path: str, kind: str) -> List[str]:
    patterns = [
        os.path.join(dataset_path, f"*{kind}*.parquet"),
        os.path.join(dataset_path, "eval", f"*{kind}*.parquet"),
        os.path.join(dataset_path, kind, "*.parquet"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))


def _load_speechcoco_local(dataset_path: str) -> Tuple[datasets.Dataset, datasets.Dataset]:
    query_files = _find_local_parquet_files(dataset_path, "query")
    corpus_files = _find_local_parquet_files(dataset_path, "corpus")

    if not query_files or not corpus_files:
        raise FileNotFoundError(
            f"[SpeechCOCO] local parquet not complete under {dataset_path}: "
            f"query_files={len(query_files)}, corpus_files={len(corpus_files)}"
        )

    query_dataset = datasets.load_dataset("parquet", data_files={"data": query_files}, split="data")
    corpus_dataset = datasets.load_dataset("parquet", data_files={"data": corpus_files}, split="data")
    return query_dataset, corpus_dataset


def _load_speechcoco_hf_by_subset(repo: str, split: Optional[str]) -> Tuple[datasets.Dataset, datasets.Dataset]:
    errors = []
    query_subsets = ["query", "queries"]
    corpus_subsets = ["corpus", "candidates"]

    for q_subset in query_subsets:
        for c_subset in corpus_subsets:
            try:
                q_ds = _as_dataset(load_hf_dataset((repo, q_subset, split)), split)
                c_ds = _as_dataset(load_hf_dataset((repo, c_subset, split)), split)
                if len(q_ds) > 0 and len(c_ds) > 0:
                    return q_ds, c_ds
            except Exception as e:
                errors.append(f"subset({q_subset},{c_subset}): {e}")

    raise RuntimeError("; ".join(errors) if errors else "no subset loading attempts")


def _pick_repo_parquet_files(repo_files: List[str], token: str) -> List[str]:
    out = []
    for f in repo_files:
        low = f.lower()
        base = os.path.basename(low)
        if not low.endswith(".parquet"):
            continue
        if token not in base:
            continue
        if "qrels" in base:
            continue
        out.append(f)
    return sorted(out)


def _load_speechcoco_hf_by_files(repo: str) -> Tuple[datasets.Dataset, datasets.Dataset]:
    try:
        from huggingface_hub import list_repo_files
    except Exception as e:
        raise RuntimeError(f"huggingface_hub is required for file listing fallback: {e}")

    repo_files = list_repo_files(repo_id=repo, repo_type="dataset")
    query_files = _pick_repo_parquet_files(repo_files, "query")
    corpus_files = _pick_repo_parquet_files(repo_files, "corpus")

    if not query_files or not corpus_files:
        raise FileNotFoundError(
            f"[SpeechCOCO] cannot find query/corpus parquet in repo {repo}. "
            f"query_files={len(query_files)}, corpus_files={len(corpus_files)}"
        )

    query_urls = [f"hf://datasets/{repo}/{fp}" for fp in query_files]
    corpus_urls = [f"hf://datasets/{repo}/{fp}" for fp in corpus_files]

    query_dataset = datasets.load_dataset("parquet", data_files={"data": query_urls}, split="data")
    corpus_dataset = datasets.load_dataset("parquet", data_files={"data": corpus_urls}, split="data")
    return query_dataset, corpus_dataset


def _load_query_and_corpus(path_info: Tuple[str, Optional[str], Optional[str]]
                           ) -> Tuple[datasets.Dataset, datasets.Dataset]:
    dataset_path, _, split = path_info

    if os.path.isdir(dataset_path):
        return _load_speechcoco_local(dataset_path)

    try:
        return _load_speechcoco_hf_by_subset(dataset_path, split)
    except Exception:
        return _load_speechcoco_hf_by_files(dataset_path)


def build_speechcoco_audio_image_dataset(path_info, **kwargs) -> Tuple[datasets.Dataset, Any]:
    _, _, _ = path_info

    model_backbone = kwargs.get("model_backbone", None)
    if model_backbone is None:
        raise ValueError("[SpeechCOCO] model_backbone is required")

    query_dataset, corpus_dataset = _load_query_and_corpus(path_info)
    query_dataset = sample_dataset(query_dataset, **kwargs)

    if corpus_dataset is None:
        raise ValueError("[SpeechCOCO] corpus_dataset is None (audio->image requires a corpus parquet)")

    cand_inst = process_input_text(
        "Understand the content of the provided image.",
        model_backbone,
        add_image_token=True,
    )

    def _corpus_map_fn(row, idx):
        cid = row.get("corpus-id") or row.get("id")
        if cid is None:
            raise ValueError(f"[SpeechCOCO] corpus row missing id. keys={list(row.keys())}")
        cid = str(cid)
        return {
            "cand_text": [cand_inst],
            "cand_image": [_image_to_cand_image_obj(row)],
            "dataset_infos": {"cand_names": [cid], "corpus_id": cid, "label_name": cid},
        }

    corpus = corpus_dataset.map(_corpus_map_fn, with_indices=True, remove_columns=corpus_dataset.column_names)

    if "corpus-id" in corpus_dataset.column_names:
        cand_names = [str(x) for x in corpus_dataset["corpus-id"]]
    elif "id" in corpus_dataset.column_names:
        cand_names = [str(x) for x in corpus_dataset["id"]]
    else:
        raise ValueError(f"[SpeechCOCO] corpus missing candidate id columns. got={corpus_dataset.column_names}")

    seen = set()
    uniq = []
    for x in cand_names:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    image_id2idx = {cid: i for i, cid in enumerate(uniq)}

    out_query_text, out_query_image, out_query_audio, out_query_audio_path = [], [], [], []
    out_cand_text, out_cand_image, out_infos = [], [], []

    for row in query_dataset:
        audio_obj = _extract_audio_obj(row)
        q_path = audio_obj.get("path") or row.get("path") or row.get("audio_path")

        img_id = row.get("image_id") or row.get("image") or row.get("image_name") or row.get("id")
        if img_id is None:
            raise ValueError("[SpeechCOCO] query missing image_id")
        img_id = str(img_id)

        label_idx = image_id2idx.get(img_id, -1)
        if label_idx < 0:
            raise ValueError(f"[SpeechCOCO] GT image_id not in corpus: {img_id}")

        audio_id = row.get("query-id") or row.get("id")

        out_query_text.append(build_query_text("SpeechCOCO"))
        out_query_image.append([None])
        out_query_audio.append(audio_obj)
        out_query_audio_path.append(q_path)

        out_cand_text.append([])
        out_cand_image.append([])
        out_infos.append({
            "label_cand_id": label_idx,
            "label_name": img_id,
            "audio_id": audio_id,
            "image_id": img_id,
        })

    dataset = datasets.Dataset.from_dict({
        "query_text": out_query_text,
        "query_image": out_query_image,
        "query_audio": out_query_audio,
        "query_audio_path": out_query_audio_path,
        "cand_text": out_cand_text,
        "cand_image": out_cand_image,
        "dataset_infos": out_infos,
    })

    return dataset, corpus


DATASET_PARSER_NAME = "audio_ret_speechcoco"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_speechcoco_dataset(model_args, data_args, **kwargs):
    eval_type = kwargs.get("eval_type", None)
    if eval_type is None:
        eval_type = getattr(data_args, "eval_type", None)

    if eval_type != "global":
        raise ValueError(
            "[SpeechCOCO] SpeechCOCO must use global-corpus evaluation. "
            "Please set `eval_type: global` in yaml task config."
        )

    mb = getattr(model_args, "model_backbone", None)
    if mb is None:
        raise ValueError("[SpeechCOCO] model_args.model_backbone is None")
    kwargs["model_backbone"] = mb

    # -------- 2) resolve path_info --------
    dataset_name = kwargs.get("dataset_name", None)
    if dataset_name is None:
        raise ValueError("[SpeechCOCO] missing kwargs['dataset_name']")

    path_info = kwargs.get("path_info_override", None)
    if path_info is None:
        from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
        from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH

        local_path_info = EVAL_DATASET_LOCAL_PATH.get(dataset_name)
        hf_path_info = EVAL_DATASET_HF_PATH.get(dataset_name)

        if local_path_info is not None and os.path.exists(local_path_info[0]):
            path_info = local_path_info
        elif hf_path_info is not None:
            path_info = hf_path_info
        else:
            path_info = local_path_info

    if path_info is None:
        raise ValueError(
            f"[SpeechCOCO] Unknown dataset_name={dataset_name}. "
            "Set path_info_override=(<hf_repo_or_local_dir>, <subset_or_None>, <split_or_None>) "
            "for custom uploaded datasets."
        )

    # -------- 3) build dataset + corpus --------
    dataset, corpus = build_speechcoco_audio_image_dataset(path_info, **kwargs)

    need_cols = [
        "query_text",
        "query_image",
        "query_audio",
        "query_audio_path",
        "cand_text",
        "cand_image",
        "dataset_infos",
    ]
    missing = [c for c in need_cols if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"[SpeechCOCO] query dataset missing columns: {missing}. "
            f"got={dataset.column_names}. "
            "Fix build_speechcoco_audio_image_dataset() to always output these fields."
        )

    dataset = dataset.select_columns(need_cols)

    if corpus is None:
        raise ValueError("[SpeechCOCO] corpus is None, but SpeechCOCO needs global corpus pool")

    if not all(k in corpus.column_names for k in ["cand_text", "cand_image", "dataset_infos"]):
        raise ValueError(
            f"[SpeechCOCO] corpus schema invalid, expected columns "
            f"['cand_text','cand_image','dataset_infos'], got={corpus.column_names}"
        )

    return dataset, corpus
