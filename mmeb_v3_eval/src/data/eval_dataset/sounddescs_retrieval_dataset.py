"""
SoundDescs text -> audio retrieval evaluation dataset:
- queries keep only query_* and label info
- candidates are returned separately via the corpus, avoiding duplicating
  the whole candidate pool in every query row
"""

import glob
import os
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import datasets

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset
from src.utils.dataset_utils import load_qrels_mapping, sample_dataset


def _extract_audio_obj(row: Dict[str, Any], keep_audio_bytes: bool) -> Dict[str, Any]:
    """Extract the audio object from flat columns or a nested audio field."""
    path = row.get("audio_path") or row.get("path")
    bytes_data = row.get("audio_bytes")

    if path is None and "audio" in row and isinstance(row["audio"], dict):
        audio_struct = row["audio"]
        path = audio_struct.get("path") or path
        bytes_data = audio_struct.get("bytes") if bytes_data is None else bytes_data

    if path is not None:
        return {"path": path, "bytes": (bytes_data if keep_audio_bytes else None)}

    return {"path": None, "bytes": bytes_data}


def _resolve_audio_path(
    raw_path: Optional[str],
    dataset_path: str,
    audio_root: Optional[str],
) -> Optional[str]:
    if not raw_path:
        return None

    candidates: List[str] = []
    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        if audio_root:
            candidates.append(os.path.join(audio_root, raw_path))
        candidates.append(os.path.join(dataset_path, raw_path))
        candidates.append(os.path.join(dataset_path, "eval", raw_path))

    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _materialize_audio_bytes(audio_bytes: bytes, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not os.path.isfile(out_path):
        with open(out_path, "wb") as f:
            f.write(audio_bytes)


def _extract_text(row: Dict[str, Any]) -> str:
    for key in ["text", "caption", "sentence", "query"]:
        value = row.get(key, None)
        if isinstance(value, str) and value.strip():
            return value
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"No text field found. keys={list(row.keys())}")


def _load_parquet_stream(pattern: str, columns: Optional[List[str]] = None):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {pattern}")

    kw = {
        "path": "parquet",
        "data_files": {"data": files},
        "streaming": True,
    }
    if columns is not None:
        # Only request columns that actually exist (streaming load raises a *lazy*
        # ArrowInvalid during iteration otherwise, e.g. requesting "path" when the
        # schema only has "audio_path"). Filter up front against the real schema.
        import pyarrow.parquet as pq
        present = set(pq.read_schema(files[0]).names)
        columns = [c for c in columns if c in present]
        if columns:
            kw["columns"] = columns

    try:
        return datasets.load_dataset(**kw)["data"]
    except Exception:
        if columns is None:
            raise
        kw.pop("columns", None)
        return datasets.load_dataset(**kw)["data"]


def _get_data_pattern(dataset_path: str, name: str) -> str:
    if "-1k" in dataset_path:
        return os.path.join(dataset_path, "eval", f"{name}*.parquet")
    return os.path.join(dataset_path, name, "*.parquet")


def _prepare_corpus(
    dataset_path: str,
    keep_audio_bytes: bool,
    required_cids: Optional[set],
    audio_root: Optional[str],
    materialize_missing_audio: bool,
    audio_cache_dir: Optional[str],
) -> Tuple[datasets.Dataset, Dict[str, int], List[str]]:
    corpus_pattern = _get_data_pattern(dataset_path, "corpus")

    preferred_columns = ["corpus-id", "id", "audio_path", "path"]
    need_audio_bytes = keep_audio_bytes or materialize_missing_audio
    if need_audio_bytes:
        preferred_columns.append("audio_bytes")
    corpus_stream = _load_parquet_stream(corpus_pattern, columns=preferred_columns)

    iterator = iter(corpus_stream)
    first_row = next(iterator, None)
    if first_row is None:
        raise ValueError(f"Empty corpus parquet: {corpus_pattern}")

    first_audio = _extract_audio_obj(first_row, keep_audio_bytes=need_audio_bytes)
    if first_audio["path"] is None and first_audio["bytes"] is None:
        corpus_stream = _load_parquet_stream(corpus_pattern, columns=None)
        iterator = iter(corpus_stream)
        first_row = next(iterator, None)
        if first_row is None:
            raise ValueError(f"Empty corpus parquet after fallback: {corpus_pattern}")

    rows = chain([first_row], iterator)
    seen_cids = set()
    cand_names: List[str] = []
    corpus_rows: List[Dict[str, Any]] = []

    for row in rows:
        cid = row.get("corpus-id") or row.get("id")
        audio_obj = _extract_audio_obj(row, keep_audio_bytes=need_audio_bytes)
        if cid is None and audio_obj.get("path"):
            cid = os.path.splitext(os.path.basename(audio_obj["path"]))[0]
        if cid is None:
            raise ValueError(f"Corpus row has no candidate id. keys={list(row.keys())}")

        cid = str(cid)
        if required_cids is not None and cid not in required_cids:
            continue

        if cid in seen_cids:
            continue

        resolved_path = _resolve_audio_path(
            raw_path=audio_obj.get("path"),
            dataset_path=dataset_path,
            audio_root=audio_root,
        )
        audio_bytes = audio_obj.get("bytes")
        final_audio: Dict[str, Any]
        if resolved_path is not None:
            final_audio = {"path": resolved_path, "bytes": (audio_bytes if keep_audio_bytes else None)}
        elif materialize_missing_audio and audio_bytes is not None:
            out_dir = audio_cache_dir or os.path.join(dataset_path, "audio_cache")
            raw_name = os.path.basename(audio_obj.get("path") or "")
            ext = os.path.splitext(raw_name)[1] if raw_name else ".wav"
            filename = f"{cid}{ext or '.wav'}"
            out_path = os.path.join(out_dir, filename)
            _materialize_audio_bytes(audio_bytes, out_path)
            final_audio = {"path": out_path, "bytes": (audio_bytes if keep_audio_bytes else None)}
        elif keep_audio_bytes and audio_bytes is not None:
            final_audio = {"path": None, "bytes": audio_bytes}
        else:
            raise FileNotFoundError(
                f"SoundDescs audio is unavailable for cid={cid}: path={audio_obj.get('path')}. "
                "Either provide local audio files (audio_root) or enable bytes fallback/materialization."
            )

        seen_cids.add(cid)
        cand_names.append(cid)

        corpus_rows.append(
            {
                "cand_text": ["[AUDIO]"],
                "cand_image": [None],
                "cand_audio": [final_audio],
                "dataset_infos": {"cand_names": [cid]},
            }
        )

        if required_cids is not None and len(seen_cids) >= len(required_cids):
            break

    if not corpus_rows:
        raise ValueError("No valid candidates found in SoundDescs corpus.")

    cand_id2idx = {cid: idx for idx, cid in enumerate(cand_names)}
    return datasets.Dataset.from_list(corpus_rows), cand_id2idx, cand_names


def _prepare_queries(dataset_path: str, **kwargs) -> datasets.Dataset:
    query_pattern = _get_data_pattern(dataset_path, "query")
    queries_stream = _load_parquet_stream(query_pattern, columns=None)
    queries_ds = datasets.Dataset.from_list(list(queries_stream))
    return sample_dataset(queries_ds, **kwargs)


def _load_qrels(dataset_path: str) -> Dict[str, Dict[str, int]]:
    qrels_pattern = _get_data_pattern(dataset_path, "qrels")
    qrels_stream = _load_parquet_stream(
        qrels_pattern,
        columns=["query-id", "corpus-id", "score"],
    )
    qrels_ds = datasets.Dataset.from_list(list(qrels_stream))
    return load_qrels_mapping(qrels_ds)


def _collect_required_cids(
    queries_ds: datasets.Dataset,
    qrels: Dict[str, Dict[str, int]],
) -> set:
    required: set = set()
    for row in queries_ds:
        qid = _select_query_id(row)
        rels = qrels.get(qid, {})
        for cid, score in rels.items():
            if score > 0:
                required.add(str(cid))
    return required


def _select_query_id(row: Dict[str, Any]) -> str:
    for key in ["query-id", "query_id", "id", "qid"]:
        if key in row and row[key] is not None:
            return str(row[key])
    raise ValueError(f"Query row has no id fields. keys={list(row.keys())}")


def _build_query_dataset(
    queries_ds: datasets.Dataset,
    qrels: Dict[str, Dict[str, int]],
    cand_id2idx: Dict[str, int],
    cand_names: List[str],
    include_cand_names: bool,
) -> datasets.Dataset:
    out_query_text: List[List[str]] = []
    out_query_image: List[List[None]] = []
    out_query_audio: List[None] = []
    out_infos: List[Dict[str, Any]] = []

    for row in queries_ds:
        qid = _select_query_id(row)
        raw_text = _extract_text(row)
        query_text = build_query_text("SoundDescs", raw_text)
        assert (
            isinstance(query_text, list)
            and len(query_text) == 1
            and isinstance(query_text[0], str)
            and query_text[0].strip()
        )

        rels = qrels.get(qid, {})
        if rels:
            rel_cid = str(max(rels.items(), key=lambda x: x[1])[0])
            label_idx = cand_id2idx.get(rel_cid, -1)
        else:
            rel_cid = None
            label_idx = -1

        info = {
            "label_cand_id": label_idx,
            "query_id": qid,
            "corpus_id": rel_cid,
            "label_name": rel_cid,
        }
        if include_cand_names:
            info["cand_names"] = cand_names

        out_query_text.append(query_text)
        out_query_image.append([None])
        out_query_audio.append(None)
        out_infos.append(info)

    return datasets.Dataset.from_dict(
        {
            "query_text": out_query_text,
            "query_image": out_query_image,
            "query_audio": out_query_audio,
            "dataset_infos": out_infos,
        }
    )


def build_sounddescs_text_audio_dataset(path_info: Tuple[str, str, str], **kwargs):
    dataset_path = path_info[0]

    keep_audio_bytes = bool(kwargs.get("keep_audio_bytes", False))
    materialize_missing_audio = bool(kwargs.get("materialize_missing_audio", True))
    audio_root = kwargs.get("audio_root", None)
    audio_cache_dir = kwargs.get("audio_cache_dir", os.path.join(dataset_path, "audio_cache"))
    eval_type = kwargs.get("eval_type", "local")
    include_cand_names = eval_type != "global"

    qrels = _load_qrels(dataset_path)
    queries_ds = _prepare_queries(dataset_path, **kwargs)
    required_cids = _collect_required_cids(queries_ds, qrels)

    corpus, cand_id2idx, cand_names = _prepare_corpus(
        dataset_path,
        keep_audio_bytes=keep_audio_bytes,
        required_cids=required_cids,
        audio_root=audio_root,
        materialize_missing_audio=materialize_missing_audio,
        audio_cache_dir=audio_cache_dir,
    )
    query_dataset = _build_query_dataset(
        queries_ds=queries_ds,
        qrels=qrels,
        cand_id2idx=cand_id2idx,
        cand_names=cand_names,
        include_cand_names=include_cand_names,
    )

    query_dataset = query_dataset.select_columns(
        ["query_text", "query_image", "query_audio", "dataset_infos"]
    )
    corpus = corpus.select_columns(["cand_text", "cand_image", "cand_audio", "dataset_infos"])
    return query_dataset, corpus


DATASET_PARSER_NAME = "sounddescs_text_audio"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_sounddescs_text_audio_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.get("dataset_name", "SoundDescs")
    path_info = EVAL_DATASET_LOCAL_PATH.get(dataset_name, EVAL_DATASET_HF_PATH.get(dataset_name))
    if path_info is None:
        raise ValueError(f"Unknown dataset_name={dataset_name}")
    return build_sounddescs_text_audio_dataset(path_info, **kwargs)
