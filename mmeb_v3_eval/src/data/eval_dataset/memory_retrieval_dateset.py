import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from datasets import Dataset

from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    RESOLUTION_MAPPING,
    add_metainfo_hook,
)
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS


DATASET_PARSER_NAME = "memory_retrieval"
SCIFACT_INSTRUCTION_KEY = "LMEB_SciFact"


def create_empty_image_dict(image_resolution):
    return {
        "bytes": [None],
        "paths": [None],
        "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)],
    }


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_parquet(path: str) -> List[dict]:
    table = pq.read_table(path)
    return table.to_pylist()


def _load_rows(path: str) -> List[dict]:
    if path.endswith(".parquet"):
        return _read_parquet(path)
    return _read_jsonl(path)


def _read_qrels_tsv(path: str) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue

            if len(parts) >= 4 and parts[1].upper() == "Q0":
                qid, did, score_raw = parts[0], parts[2], parts[3]
            else:
                qid, did, score_raw = parts[0], parts[1], parts[2]

            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 1.0

            if score <= 0:
                continue

            qid = str(qid)
            did = str(did)
            qrels.setdefault(qid, {})[did] = score
    return qrels


def _parse_qrels_rows(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    for row in rows:
        qid = (
            row.get("query_id")
            or row.get("query-id")
            or row.get("qid")
            or row.get("q_id")
            or row.get("id")
        )
        did = (
            row.get("corpus_id")
            or row.get("corpus-id")
            or row.get("doc_id")
            or row.get("document_id")
            or row.get("p_id")
        )
        if qid is None or did is None:
            continue

        score = row.get("label", row.get("score", 1))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 1.0

        if score <= 0:
            continue

        qid = str(qid)
        did = str(did)
        qrels.setdefault(qid, {})[did] = score
    return qrels


def _load_qrels(path: str) -> Dict[str, Dict[str, float]]:
    if path.endswith(".tsv"):
        return _read_qrels_tsv(path)
    return _parse_qrels_rows(_load_rows(path))


def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def _find_memory_root(path: Path) -> Optional[Path]:
    p = path.resolve()
    for parent in [p, *p.parents]:
        if parent.name == "memory-tasks":
            return parent
    return None


def _load_task_instructions(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    memory_root = _find_memory_root(dataset_dir)
    if memory_root is None:
        return {}

    instruction_path = memory_root / "task_instructions.json"
    if not instruction_path.is_file():
        return {}

    with open(instruction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def _dataset_instruction_key(dataset_name: str) -> str:
    if dataset_name == "SciFact":
        return SCIFACT_INSTRUCTION_KEY
    return dataset_name


def _resolve_query_instruction(
    instructions: Dict[str, Dict[str, str]],
    dataset_name: str,
    task_key: str,
) -> str:
    ds_key = _dataset_instruction_key(dataset_name)
    ds_cfg = instructions.get(ds_key, {})
    if not isinstance(ds_cfg, dict):
        return ""

    if ds_key == SCIFACT_INSTRUCTION_KEY:
        return ds_cfg.get(SCIFACT_INSTRUCTION_KEY, "")

    subtask = task_key.split("/")[-1] if task_key else ""
    if subtask in ds_cfg:
        return ds_cfg[subtask]

    # Some datasets (e.g. PeerQA) store instruction under dataset-level key.
    if ds_key in ds_cfg:
        return ds_cfg[ds_key]

    # Root-level single-task dataset: use the only available instruction.
    if not subtask and len(ds_cfg) == 1:
        only_val = next(iter(ds_cfg.values()))
        if isinstance(only_val, str):
            return only_val

    return ""


def _prefix_id(raw_id: str, prefix: str) -> str:
    rid = str(raw_id)
    if not prefix:
        return rid
    return f"{prefix}::{rid}"


def _read_corpus_rows(corpus_file: str) -> List[dict]:
    rows = _load_rows(corpus_file)
    parsed = []
    for row in rows:
        did = (
            row.get("_id")
            or row.get("id")
            or row.get("doc_id")
            or row.get("document_id")
            or row.get("corpus-id")
        )
        if did is None:
            continue

        title = row.get("title") or ""
        text = row.get("text") or row.get("content") or ""
        if title and text:
            text = f"{title}\n{text}"
        if not text:
            continue

        parsed.append({"cand_id": str(did), "cand_text": text})
    return parsed


def _query_text(row: dict) -> str:
    return (
        row.get("text")
        or row.get("query")
        or row.get("query_content")
        or row.get("short_query")
        or ""
    )


def _discover_task_specs(dataset_dir: Path) -> List[dict]:
    query_candidates = sorted(dataset_dir.rglob("queries.jsonl"))
    query_candidates += sorted(dataset_dir.rglob("query.jsonl"))
    query_candidates += sorted(dataset_dir.rglob("queries.parquet"))
    query_candidates += sorted(dataset_dir.rglob("query.parquet"))

    specs: List[dict] = []
    seen = set()
    for query_file in query_candidates:
        qdir = query_file.parent

        qrels_file = _pick_first_existing(
            [
                qdir / "qrels.tsv",
                qdir / "qrels.jsonl",
                qdir / "qrels.parquet",
                qdir / "qrels" / "test.jsonl",
                qdir / "qrels" / "test.parquet",
            ]
        )
        if qrels_file is None:
            continue

        corpus_file = _pick_first_existing(
            [
                qdir / "corpus.jsonl",
                qdir / "corpus.parquet",
                qdir / "documents.jsonl",
                qdir / "documents.parquet",
            ]
        )

        if corpus_file is None:
            cursor = qdir
            while cursor != dataset_dir and dataset_dir in cursor.parents:
                cursor = cursor.parent
                corpus_file = _pick_first_existing(
                    [
                        cursor / "corpus.jsonl",
                        cursor / "corpus.parquet",
                        cursor / "documents.jsonl",
                        cursor / "documents.parquet",
                    ]
                )
                if corpus_file is not None:
                    break

        if corpus_file is None:
            continue

        task_key = os.path.relpath(str(qdir), str(dataset_dir)).replace("\\", "/")
        corpus_key = os.path.relpath(str(corpus_file.parent), str(dataset_dir)).replace("\\", "/")

        if task_key == ".":
            task_key = ""
        if corpus_key == ".":
            corpus_key = ""

        sign = (str(query_file), str(corpus_file), str(qrels_file), task_key, corpus_key)
        if sign in seen:
            continue
        seen.add(sign)

        # Optional per-query candidate restriction list (e.g. DeepPlanning,
        # PeerQA): maps a query/scene id to the doc ids it should retrieve from.
        query_candidates_file = _pick_first_existing(
            [
                qdir / "candidates.jsonl",
                qdir / "candidates.parquet",
                qdir / "query_candidates.jsonl",
                qdir / "query_candidates.parquet",
            ]
        )

        specs.append(
            {
                "task_key": task_key,
                "corpus_key": corpus_key,
                "query_file": str(query_file),
                "candidate_file": str(corpus_file),
                "qrels_file": str(qrels_file),
                "query_candidates_file": str(query_candidates_file) if query_candidates_file else None,
            }
        )

    return specs


def _load_query_candidates(path: Optional[str], corpus_key: str) -> Dict[str, List[str]]:
    """Load per-query candidate doc-id lists, keyed by raw (unprefixed) query id."""
    if not path or not os.path.isfile(path):
        return {}
    mapping: Dict[str, List[str]] = {}
    for row in _load_rows(path):
        qid = (
            row.get("scene_id")
            or row.get("_id")
            or row.get("id")
            or row.get("qid")
            or row.get("query_id")
            or row.get("query-id")
        )
        doc_ids = (
            row.get("candidate_doc_ids")
            or row.get("candidate_ids")
            or row.get("candidates")
            or row.get("doc_ids")
        )
        if qid is None or not doc_ids:
            continue
        mapping[str(qid)] = [_prefix_id(doc_id, corpus_key) for doc_id in doc_ids]
    return mapping


@add_metainfo_hook
def data_prepare_query(batch_dict, *args, **kwargs):
    model_backbone = kwargs["model_backbone"]
    image_resolution = kwargs["image_resolution"]

    batch_size = len(batch_dict["qry_text"])
    query_texts, query_images, dataset_infos = [], [], []

    for qry_text, qry_id, label_names, rel_scores, cand_names in zip(
        batch_dict["qry_text"],
        batch_dict["qry_id"],
        batch_dict["label_names"],
        batch_dict.get("rel_scores", [None] * batch_size),
        batch_dict.get("cand_names", [[]] * batch_size),
    ):
        if not qry_text:
            continue

        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])

        query_texts.append([qry_text])
        query_images.append([create_empty_image_dict(image_resolution)])
        dataset_infos.append(
            {
                "qry_id": qry_id,
                "label_name": label_names,
                # keep "cand_names" aligned with the (empty) per-query cand_text
                # lists; the per-query retrieval pool rides in "pool_names".
                "cand_names": [],
                "pool_names": cand_names or [],
                "rel_scores": rel_scores,
            }
        )

    cand_texts = [[] for _ in query_texts]
    cand_images = [[] for _ in query_texts]

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos,
    }


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_memory_retrieval_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    data_path = kwargs.get("data_path")

    query_file = kwargs.get("query_file")
    candidate_file = kwargs.get("candidate_file")
    qrels_file = kwargs.get("qrels_file")

    if not data_path and dataset_name in EVAL_DATASET_LOCAL_PATH:
        data_path = EVAL_DATASET_LOCAL_PATH[dataset_name][0]

    if data_path is None and not (query_file and candidate_file and qrels_file):
        raise ValueError(
            f"Missing data_path for dataset '{dataset_name}'. "
            "Provide data_path or explicit query_file/candidate_file/qrels_file."
        )

    task_specs: List[dict] = []
    dataset_dir = Path(data_path).resolve() if data_path else None

    if query_file and candidate_file and qrels_file:
        task_specs.append(
            {
                "task_key": subset_name or "",
                "corpus_key": subset_name or "",
                "query_file": query_file,
                "candidate_file": candidate_file,
                "qrels_file": qrels_file,
                "query_candidates_file": kwargs.get("query_candidates_file"),
            }
        )
    else:
        if dataset_dir is None or not dataset_dir.is_dir():
            raise FileNotFoundError(f"Memory dataset directory not found: {data_path}")
        task_specs = _discover_task_specs(dataset_dir)

    if not task_specs:
        raise FileNotFoundError(
            f"No memory retrieval task found under: {data_path}. "
            "Expected queries.* + qrels.* + corpus.* files."
        )

    task_instructions = _load_task_instructions(dataset_dir) if dataset_dir else {}

    corpus_cache: Dict[str, List[dict]] = {}
    candidate_map: Dict[str, str] = {}
    query_rows_final: List[dict] = []

    for spec in task_specs:
        q_file = spec["query_file"]
        c_file = spec["candidate_file"]
        r_file = spec["qrels_file"]
        task_key = spec["task_key"]
        corpus_key = spec["corpus_key"]

        if not os.path.isfile(q_file):
            raise FileNotFoundError(f"Query file not found: {q_file}")
        if not os.path.isfile(c_file):
            raise FileNotFoundError(f"Candidate file not found: {c_file}")
        if not os.path.isfile(r_file):
            raise FileNotFoundError(f"Qrels file not found: {r_file}")

        if c_file not in corpus_cache:
            corpus_cache[c_file] = _read_corpus_rows(c_file)

        for cand in corpus_cache[c_file]:
            cand_id = _prefix_id(cand["cand_id"], corpus_key)
            if cand_id not in candidate_map:
                candidate_map[cand_id] = cand["cand_text"]

        qrels = _load_qrels(r_file)
        q_rows = _load_rows(q_file)
        query_candidates = _load_query_candidates(spec.get("query_candidates_file"), corpus_key)

        instruction = _resolve_query_instruction(task_instructions, dataset_name, task_key)

        for row in q_rows:
            qid_raw = (
                row.get("_id")
                or row.get("id")
                or row.get("qid")
                or row.get("query_id")
                or row.get("query-id")
            )
            if qid_raw is None:
                continue

            qid_raw = str(qid_raw)
            q_text = _query_text(row)
            if instruction:
                q_text = f"{instruction}\n{q_text}".strip()
            if not q_text:
                continue

            rel = qrels.get(qid_raw, {})
            label_names = [_prefix_id(doc_id, corpus_key) for doc_id in rel.keys()]
            rel_scores = [rel[doc_id] for doc_id in rel.keys()] if rel else None

            query_rows_final.append(
                {
                    "qry_id": _prefix_id(qid_raw, task_key),
                    "qry_text": q_text,
                    "label_names": label_names,
                    "rel_scores": rel_scores,
                    "cand_names": query_candidates.get(qid_raw, []),
                }
            )

    if query_rows_final and all(len(x.get("label_names", [])) == 0 for x in query_rows_final):
        raise ValueError(
            f"No positive labels loaded for '{dataset_name}'. "
            "Please check qrels/query/corpus ID format."
        )

    candidate_rows_final = [
        {"cand_id": cid, "cand_text": ctext}
        for cid, ctext in candidate_map.items()
    ]

    qry_dataset = Dataset.from_list(query_rows_final)

    kwargs["model_backbone"] = model_args.model_backbone
    kwargs["image_resolution"] = data_args.image_resolution
    kwargs["global_dataset_name"] = f"{dataset_name}/{subset_name or 'all'}"

    qry_dataset = qry_dataset.map(
        lambda x: data_prepare_query(x, **kwargs),
        batched=True,
        batch_size=64,
        remove_columns=["qry_text", "qry_id", "label_names", "rel_scores", "cand_names"],
        drop_last_batch=False,
    )

    corpus_rows = []
    for cand in candidate_rows_final:
        corpus_rows.append(
            {
                "cand_text": [cand["cand_text"]],
                "cand_image": [create_empty_image_dict(data_args.image_resolution)],
                "dataset_infos": {"cand_names": [cand["cand_id"]]},
            }
        )

    corpus = Dataset.from_list(corpus_rows)

    print(
        f"Loaded memory dataset '{dataset_name}' with "
        f"{len(query_rows_final)} queries, {len(candidate_rows_final)} candidates, {len(task_specs)} tasks"
    )

    return qry_dataset, corpus
