import json
import os
import glob
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from datasets import Dataset

from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    add_metainfo_hook,
    RESOLUTION_MAPPING,
)
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS


DATASET_PARSER_NAME = "complex_text_retrieve"

# Per-task query instructions, keyed by subset name ("<group>/<task>").
TASK_QUERY_INSTRUCTIONS = {
    "BRIGHT/aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "BRIGHT/biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "BRIGHT/earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "BRIGHT/economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "BRIGHT/leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    "BRIGHT/pony": "Given a Pony question, retrieve relevant passages that help answer the question.",
    "BRIGHT/psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "BRIGHT/robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
    "BRIGHT/stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "BRIGHT/sustainable_living": "Given a Sustainable Living post, retrieve relevant passages that help answer the"
                                 " post.",
    "BRIGHT/theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "BRIGHT/theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
    "R2MED/Bioinformatics": "Given a Bioinformatics post, retrieve relevant passages that help answer the post.",
    "R2MED/Biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "R2MED/IIYi-Clinical": "Given a Clinical Case, retrieve similar cases that help diagnose the case.",
    "R2MED/MedQA-Diag": "Given a Medical Exam, retrieve relevant passages that help answer the exam.",
    "R2MED/MedXpertQA-Exam": "Given a Medical Exam, retrieve relevant passages that help answer the exam.",
    "R2MED/Medical-Sciences": "Given a Medical Science post, retrieve relevant passages that help answer the post.",
    "R2MED/PMC-Clinical": "Given a Clinical Case, retrieve similar cases that help diagnose the case.",
    "R2MED/PMC-Treatment": "Given a Clinical Case, retrieve relevant passages that help answer the case.",
}


def create_empty_image_dict(image_resolution):
    """Create empty image placeholder for text-only tasks."""
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


def _read_tsv(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("query"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            rows.append({"query-id": parts[0], "corpus-id": parts[1], "score": parts[2]})
    return rows


def _load_rows(path: str) -> List[dict]:
    if path.endswith(".parquet"):
        return _read_parquet(path)
    if path.endswith(".tsv"):
        return _read_tsv(path)
    return _read_jsonl(path)


def _pick_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _glob_first(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return sorted(matches)[0]
    return None


def _resolve_dataset_dir(data_path: Optional[str], subset_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not data_path:
        return None, None
    if subset_name:
        direct = os.path.join(data_path, subset_name)
        if os.path.isdir(direct):
            return direct, None
        parent = os.path.join(data_path, os.path.dirname(subset_name))
        if os.path.isdir(parent):
            return parent, os.path.basename(subset_name)
        return data_path, os.path.basename(subset_name)
    return data_path, None


def _find_query_file(dataset_dir: str, task_name: Optional[str]) -> Optional[str]:
    candidates = []
    # BRIGHT-style examples parquet
    examples_dir = os.path.join(dataset_dir, "examples")
    if os.path.isdir(examples_dir) and task_name:
        candidates.append(_glob_first([os.path.join(examples_dir, f"{task_name}-*.parquet")]))

    candidates.append(_pick_first([
        os.path.join(dataset_dir, "queries.jsonl"),
        os.path.join(dataset_dir, "query.jsonl"),
        os.path.join(dataset_dir, "queries.parquet"),
        os.path.join(dataset_dir, "query.parquet"),
        os.path.join(dataset_dir, "queris.jsonl"),  # typo in some datasets
        os.path.join(dataset_dir, "queris.parquet"),
    ]))

    if task_name:
        candidates.append(_glob_first([
            os.path.join(dataset_dir, f"{task_name}*.queries*.jsonl"),
            os.path.join(dataset_dir, f"{task_name}*.queries*.parquet"),
            os.path.join(dataset_dir, f"{task_name}*.query*.jsonl"),
            os.path.join(dataset_dir, f"{task_name}*.query*.parquet"),
            os.path.join(dataset_dir, f"*{task_name}*.queries*.jsonl"),
            os.path.join(dataset_dir, f"*{task_name}*.queries*.parquet"),
        ]))

    # NanoBEIR-style folders
    candidates.append(_glob_first([
        os.path.join(dataset_dir, "queries", "*.parquet"),
        os.path.join(dataset_dir, "queries", "*.jsonl"),
    ]))

    return _pick_first([c for c in candidates if c])


def _find_corpus_file(dataset_dir: str, task_name: Optional[str]) -> Optional[str]:
    candidates = []
    # BRIGHT-style documents parquet
    documents_dir = os.path.join(dataset_dir, "documents")
    if os.path.isdir(documents_dir) and task_name:
        candidates.append(_glob_first([os.path.join(documents_dir, f"{task_name}-*.parquet")]))

    candidates.append(_pick_first([
        os.path.join(dataset_dir, "corpus.jsonl"),
        os.path.join(dataset_dir, "corpus.parquet"),
        os.path.join(dataset_dir, "documents.jsonl"),
        os.path.join(dataset_dir, "documents.parquet"),
    ]))

    if task_name:
        candidates.append(_glob_first([
            os.path.join(dataset_dir, f"{task_name}*.corpus*.jsonl"),
            os.path.join(dataset_dir, f"{task_name}*.corpus*.parquet"),
            os.path.join(dataset_dir, f"*{task_name}*.corpus*.jsonl"),
            os.path.join(dataset_dir, f"*{task_name}*.corpus*.parquet"),
        ]))

    # NanoBEIR-style folders
    candidates.append(_glob_first([
        os.path.join(dataset_dir, "corpus", "*.parquet"),
        os.path.join(dataset_dir, "corpus", "*.jsonl"),
    ]))

    return _pick_first([c for c in candidates if c])


def _find_qrels_file(dataset_dir: str, task_name: Optional[str]) -> Optional[str]:
    candidates = []
    candidates.append(_pick_first([
        os.path.join(dataset_dir, "qrels.jsonl"),
        os.path.join(dataset_dir, "qrels.parquet"),
        os.path.join(dataset_dir, "qrels_changed", "test.jsonl"),
        os.path.join(dataset_dir, "qrels_changed", "test.parquet"),
        os.path.join(dataset_dir, "qrels_og", "test.jsonl"),
        os.path.join(dataset_dir, "qrels_og", "test.parquet"),
        os.path.join(dataset_dir, "qrels_reversed", "test.jsonl"),
        os.path.join(dataset_dir, "qrels_reversed", "test.parquet"),
    ]))

    if task_name:
        candidates.append(_glob_first([
            os.path.join(dataset_dir, f"{task_name}*.qrels*.jsonl"),
            os.path.join(dataset_dir, f"{task_name}*.qrels*.parquet"),
            os.path.join(dataset_dir, f"*{task_name}*.qrels*.jsonl"),
            os.path.join(dataset_dir, f"*{task_name}*.qrels*.parquet"),
        ]))

    candidates.append(_glob_first([
        os.path.join(dataset_dir, "qrels", "*.parquet"),
        os.path.join(dataset_dir, "qrels", "*.jsonl"),
    ]))

    return _pick_first([c for c in candidates if c])


def _parse_qrels(qrels_rows: List[dict]) -> Dict[str, Dict[str, float]]:
    qrels = {}
    for row in qrels_rows:
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


def _base_query_text(row: dict) -> str:
    return (
        row.get("text")
        or row.get("query")
        or row.get("query_content")
        or row.get("short_query")
        or ""
    )


def _infer_query_text(row: dict, qrels_variant: Optional[str], subset_name: Optional[str]) -> str:
    subset_lower = (subset_name or "").lower()
    base_query = _base_query_text(row)
    if "infosearch" in subset_lower:
        instruction = row.get("instruction_changed", "")
        if instruction:
            return f"{instruction} {base_query}".strip()
        return base_query
    if "followir" in subset_lower:
        instruction = row.get("instruction_og", "")
        if instruction:
            return f"{instruction} {base_query}".strip()
        return base_query

    if qrels_variant == "changed" and row.get("instruction_changed"):
        return row.get("instruction_changed", "")
    if qrels_variant == "reversed" and row.get("instruction_reversed"):
        return row.get("instruction_reversed", "")
    if qrels_variant == "og" and row.get("instruction_og"):
        return row.get("instruction_og", "")
    instruction = TASK_QUERY_INSTRUCTIONS.get(subset_name or "", "")
    if instruction:
        return f"{instruction}\n{base_query}"
    return base_query


def _parse_queries(
    query_rows: List[dict],
    qrels: Optional[Dict[str, Dict[str, float]]],
    qrels_variant: Optional[str],
    subset_name: Optional[str],
) -> List[dict]:
    queries = []
    for row in query_rows:
        qid = (
            row.get("_id")
            or row.get("id")
            or row.get("qid")
            or row.get("query_id")
            or row.get("query-id")
        )
        if qid is None:
            continue
        qid = str(qid)
        text = _infer_query_text(row, qrels_variant, subset_name)
        if not text:
            continue

        label_names = []
        rel_scores = None
        if qrels is not None:
            rel = qrels.get(qid, {})
            label_names = list(rel.keys())
            if rel:
                rel_scores = [rel[doc_id] for doc_id in label_names]
        else:
            gold_ids = row.get("gold_ids") or row.get("gold_ids_long") or []
            if isinstance(gold_ids, str):
                try:
                    gold_ids = json.loads(gold_ids)
                except json.JSONDecodeError:
                    gold_ids = []
            label_names = [str(x) for x in gold_ids if x is not None]

        queries.append({
            "qry_id": qid,
            "qry_text": text,
            "label_names": label_names,
            "rel_scores": rel_scores,
        })
    return queries


def _parse_corpus(corpus_rows: List[dict]) -> List[dict]:
    corpus = []
    for row in corpus_rows:
        did = (
            row.get("_id")
            or row.get("id")
            or row.get("doc_id")
            or row.get("document_id")
            or row.get("corpus-id")
        )
        if did is None:
            continue
        did = str(did)
        title = row.get("title") or ""
        text = row.get("text") or row.get("content") or ""
        if title and text:
            text = f"{title}\n{text}"
        if not text:
            continue
        corpus.append({
            "cand_id": did,
            "cand_text": text,
        })
    return corpus


@add_metainfo_hook
def data_prepare_query(batch_dict, *args, **kwargs):
    model_backbone = kwargs["model_backbone"]
    image_resolution = kwargs["image_resolution"]

    batch_size = len(batch_dict["qry_text"])
    query_texts, query_images, dataset_infos = [], [], []

    for qry_text, qry_id, label_names, rel_scores in zip(
        batch_dict["qry_text"],
        batch_dict["qry_id"],
        batch_dict["label_names"],
        batch_dict.get("rel_scores", [None] * batch_size),
    ):
        if not qry_text:
            continue
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])

        query_texts.append([qry_text])
        empty_image = create_empty_image_dict(image_resolution)
        query_images.append([empty_image])
        dataset_infos.append({
            "qry_id": qry_id,
            "label_name": label_names,
            "cand_names": [],
            "rel_scores": rel_scores,
        })

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

    cand_texts, cand_images, dataset_infos = [], [], []
    for cand_text, cand_id in zip(batch_dict["cand_text"], batch_dict["cand_id"]):
        if not cand_text:
            continue
        if model_backbone != PHI3V:
            cand_text = cand_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])

        cand_texts.append([cand_text])
        empty_image = create_empty_image_dict(image_resolution)
        cand_images.append([empty_image])
        dataset_infos.append({"cand_name": cand_id})

    return {
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos,
    }


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_complex_text_retrieve_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    query_file = kwargs.get("query_file")
    candidate_file = kwargs.get("candidate_file")
    qrels_file = kwargs.get("qrels_file")
    data_path = kwargs.get("data_path", None)

    dataset_dir = None
    task_name = None
    if query_file and os.path.isdir(query_file):
        dataset_dir = query_file
    elif candidate_file and os.path.isdir(candidate_file):
        dataset_dir = candidate_file
    elif data_path:
        dataset_dir, task_name = _resolve_dataset_dir(data_path, subset_name)

    if dataset_dir is not None:
        if query_file is None or os.path.isdir(query_file):
            query_file = _find_query_file(dataset_dir, task_name)
        if candidate_file is None or os.path.isdir(candidate_file):
            candidate_file = _find_corpus_file(dataset_dir, task_name)
        if qrels_file is None or os.path.isdir(qrels_file):
            qrels_file = _find_qrels_file(dataset_dir, task_name)

    if not query_file or not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")
    if not candidate_file or not os.path.exists(candidate_file):
        raise FileNotFoundError(f"Candidate file not found: {candidate_file}")
    if qrels_file and not os.path.exists(qrels_file):
        raise FileNotFoundError(f"Qrels file not found: {qrels_file}")

    qrels_variant = None
    if qrels_file:
        if "qrels_changed" in qrels_file:
            qrels_variant = "changed"
        elif "qrels_reversed" in qrels_file:
            qrels_variant = "reversed"
        elif "qrels_og" in qrels_file:
            qrels_variant = "og"

    print(f"Loading queries from: {query_file}")
    print(f"Loading candidates from: {candidate_file}")
    if qrels_file:
        print(f"Loading qrels from: {qrels_file}")
    print(f"Subset: {subset_name}")

    query_rows = _load_rows(query_file)
    corpus_rows = _load_rows(candidate_file)
    qrels = _parse_qrels(_load_rows(qrels_file)) if qrels_file and os.path.exists(qrels_file) else None

    query_data = _parse_queries(query_rows, qrels, qrels_variant, subset_name)
    if query_data and all(len(x.get("label_names", [])) == 0 for x in query_data):
        raise ValueError(
            f"No positive labels loaded for subset '{subset_name}'. "
            f"Please check qrels_file path and qrels/query ID format."
        )
    candidate_data = _parse_corpus(corpus_rows)

    qry_dataset = Dataset.from_list(query_data)

    kwargs["model_backbone"] = model_args.model_backbone
    kwargs["image_resolution"] = data_args.image_resolution
    kwargs["global_dataset_name"] = f"{dataset_name}/{subset_name}"

    qry_dataset = qry_dataset.map(
        lambda x: data_prepare_query(x, **kwargs),
        batched=True,
        batch_size=64,
        remove_columns=["qry_text", "qry_id", "label_names", "rel_scores"],
        drop_last_batch=False,
    )

    corpus_rows = []
    for cand in candidate_data:
        empty_image = create_empty_image_dict(data_args.image_resolution)
        corpus_rows.append({
            "cand_text": [cand["cand_text"]],
            "cand_image": [empty_image],
            "dataset_infos": {"cand_names": [cand["cand_id"]]},
        })

    corpus = Dataset.from_list(corpus_rows)
    print(f"Loaded {len(query_data)} queries and {len(candidate_data)} candidates")

    return qry_dataset, corpus
