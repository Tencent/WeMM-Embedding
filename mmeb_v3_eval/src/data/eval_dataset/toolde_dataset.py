from datasets import Dataset
import os
import json
import math


class DatasetWithLength:
    """Wrap a Dataset object so that len() works correctly."""

    def __init__(self, dataset, num_rows):
        self.dataset = dataset
        self._num_rows = num_rows

    def __len__(self):
        return self._num_rows

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, key):
        return self.dataset[key]


DATASET_PARSER_NAME = "toolde"
from src.data.eval_dataset.base_eval_dataset import (  # noqa: E402
    AutoEvalPairDataset,
    add_metainfo_hook,
    RESOLUTION_MAPPING,
)
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS  # noqa: E402


def create_empty_image_dict(image_resolution):
    """Create an empty image dict for text-only data."""
    return {
        "bytes": [None],
        "paths": [None],
        "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]
    }


@add_metainfo_hook
def data_prepare_query(batch_dict, *args, **kwargs):
    """
    Prepare query data for the ToolRet dataset.
    """
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']

    batch_size = len(batch_dict['qry_text'])
    query_texts, query_images, dataset_infos = [], [], []

    for qry_text, qry_id, label_names, subtask in \
        zip(batch_dict['qry_text'],
            batch_dict['qry_id'],
            batch_dict['label_names'],
            batch_dict.get('subtask', [None] * batch_size)):

        if not qry_text:
            print("empty query text")
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
            "subtask": subtask,
        })

    if len(query_texts) == 0:
        print('something went wrong in query preparation')

    cand_texts = [[] for _ in query_texts]
    cand_images = [[] for _ in query_texts]

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos
    }


@add_metainfo_hook
def data_prepare_candidate(batch_dict, *args, **kwargs):
    """
    Prepare candidate data for the ToolRet dataset.
    """
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']

    cand_texts, cand_images, dataset_infos = [], [], []

    for cand_text, cand_id in zip(batch_dict['cand_text'], batch_dict['cand_id']):
        if not cand_text:
            print("empty candidate text")
            continue

        if model_backbone != PHI3V:
            cand_text = cand_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])

        cand_texts.append([cand_text])

        empty_image = create_empty_image_dict(image_resolution)
        cand_images.append([empty_image])

        dataset_infos.append({
            "cand_name": cand_id,
        })

    if len(cand_texts) == 0:
        print('something went wrong in candidate preparation')

    return {
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos
    }


def load_query_data(query_file_path):
    """
    Load the query file.
    Supports Parquet, JSONL, and JSON formats.
    Fields: id, query, label (or labels), subtask
    """
    import pandas as pd

    queries = []

    if query_file_path.endswith('.parquet'):
        df = pd.read_parquet(query_file_path)
        for row in df.to_dict(orient="records"):
            if "subtask" not in row:
                row["subtask"] = row.get("category", None)
            queries.append(_parse_query_data(row))
    elif query_file_path.endswith('.json') and not query_file_path.endswith('.jsonl'):
        with open(query_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            if isinstance(data_list, list):
                for data in data_list:
                    queries.append(_parse_query_data(data))
    else:
        with open(query_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                queries.append(_parse_query_data(data))

    return queries


def format_tool_candidate(documentation):
    """Render a tool documentation entry as structured plain text.

    The raw ``documentation`` field is a JSON blob; dumping it verbatim hurts
    retrieval quality, so render the meaningful fields line by line.
    """
    value = documentation
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if not isinstance(value, dict):
        return str(value)
    lines = []
    if value.get("name"):
        lines.append(f"Tool: {value['name']}")
    if value.get("description"):
        lines.append(f"Description: {value['description']}")
    for key in (
        "parameters", "doc_arguments", "api_call", "domain", "framework",
        "tool_profile",
    ):
        field = value.get(key)
        if not field:
            continue
        rendered = field if isinstance(field, str) else json.dumps(
            field, ensure_ascii=False
        )
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def _parse_query_data(data):
    """
    Parse a single query item.
    Fields: id, query, label (or labels), subtask
    """
    instruction = data.get("instruction", "")
    query_text = data.get("query", "")
    if instruction:
        query_text = f"{instruction}\n{query_text}".strip()

    label_data = data.get('label', None)
    if label_data is None:
        label_data = data.get('labels', [])
    if isinstance(label_data, float) and math.isnan(label_data):
        label_data = []
    pos_ids = []

    labels = label_data
    if isinstance(label_data, str):
        label_text = label_data.strip()
        if label_text:
            try:
                labels = json.loads(label_text)
            except json.JSONDecodeError:
                labels = label_text
        else:
            labels = []

    if isinstance(labels, list):
        for label in labels:
            if isinstance(label, dict) and 'id' in label:
                pos_ids.append(label['id'])
            elif isinstance(label, str):
                pos_ids.append(label)
    elif isinstance(labels, dict) and 'id' in labels:
        pos_ids.append(labels['id'])
    elif isinstance(labels, str):
        pos_ids.append(labels)

    return {
        'qry_id': data.get('id', ''),
        'qry_text': query_text,
        'label_names': pos_ids,
        'subtask': data.get('subtask', None),
    }


def load_candidate_data(candidate_file_path):
    """
    Load the candidate file.
    Supports Parquet and JSONL formats.
    Fields: id, documentation
    """
    import pandas as pd

    candidates = []

    if candidate_file_path.endswith('.parquet'):
        df = pd.read_parquet(candidate_file_path)
        for _, row in df.iterrows():
            candidates.append({
                'cand_id': row.get('id', ''),
                'cand_text': format_tool_candidate(row.get('documentation', '')),
            })
    else:
        with open(candidate_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                candidates.append({
                    'cand_id': data.get('id', ''),
                    'cand_text': format_tool_candidate(data.get('documentation', '')),
                })
    return candidates


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_toolde_dataset(model_args, data_args, *args, **kwargs):
    """
    Load the ToolDe evaluation dataset.

    Data layout:
        - queries are grouped by subtask: web/apigen, code/xxx, customized/xxx
        - candidates are grouped by category: web, code, customized
        - evaluation retrieves within the category: a web query is only
          matched against web candidates

    Args:
        dataset_name: dataset name, defaults to "toolde"
        subset_name: subtask name in the form "category/task", e.g. "web/apigen"
        query_file: path to the query file
        candidate_file: path to the candidate file (per category, e.g. web, code, customized)
        data_path: data directory path (optional)
    """
    subset_name = kwargs.get("subset_name")

    if subset_name and '/' in subset_name:
        category, task_name = subset_name.split('/', 1)
    else:
        category = subset_name
        task_name = subset_name

    query_file = kwargs.get("query_file")
    candidate_file = kwargs.get("candidate_file")
    data_path = kwargs.get("data_path", None)

    if not query_file:
        raise ValueError("Missing required argument: query_file")
    if not candidate_file:
        raise ValueError("Missing required argument: candidate_file")

    if data_path:
        if query_file and not os.path.isabs(query_file):
            query_file = os.path.join(data_path, query_file)
        if candidate_file and not os.path.isabs(candidate_file):
            candidate_file = os.path.join(data_path, candidate_file)

    if query_file and os.path.isdir(query_file):
        query_dir = query_file
        possible_query_paths = []
        if subset_name:
            possible_query_paths.extend([
                os.path.join(query_dir, f"{subset_name}.jsonl"),  # web/apigen.jsonl
                os.path.join(query_dir, f"{subset_name}.json"),   # web/apigen.json
            ])
        if task_name:
            possible_query_paths.extend([
                os.path.join(query_dir, f"{task_name}.jsonl"),  # apigen.jsonl
                os.path.join(query_dir, f"{task_name}.json"),   # apigen.json
            ])
        if category and task_name:
            possible_query_paths.append(os.path.join(query_dir, category, f"{task_name}.jsonl"))  # web/apigen.jsonl
        query_file = None
        for path in possible_query_paths:
            if os.path.exists(path) and os.path.isfile(path):
                query_file = path
                break

        if query_file is None:
            import glob
            search_patterns = []
            if task_name:
                search_patterns.extend([
                    os.path.join(query_dir, f"*{task_name}*.jsonl"),
                    os.path.join(query_dir, f"*{task_name}*.json"),
                ])
                if category:
                    search_patterns.extend([
                        os.path.join(query_dir, category, f"*{task_name}*.jsonl"),
                        os.path.join(query_dir, category, f"*{task_name}*.json"),
                    ])
            if not search_patterns:
                search_patterns = [
                    os.path.join(query_dir, "*.jsonl"),
                    os.path.join(query_dir, "*.json"),
                ]
            for pattern in search_patterns:
                matches = sorted(glob.glob(pattern))
                if matches:
                    query_file = matches[0]
                    break

        if query_file is None:
            raise FileNotFoundError(f"Query file not found in directory {query_dir}. Tried patterns:"
                                    f" {possible_query_paths}")
    elif query_file and not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")

    if candidate_file and os.path.isdir(candidate_file):
        possible_candidate_paths = []
        if category:
            possible_candidate_paths.extend([
                os.path.join(candidate_file, f"{category}.parquet"),  # web.parquet
                os.path.join(candidate_file, f"{category}.jsonl"),    # web.jsonl
            ])
        possible_candidate_paths.extend([
            os.path.join(candidate_file, "candidates.parquet"),   # candidates.parquet
            os.path.join(candidate_file, "candidates.jsonl"),     # candidates.jsonl
        ])
        candidate_file_path = None
        for path in possible_candidate_paths:
            if os.path.exists(path) and os.path.isfile(path):
                candidate_file_path = path
                break

        if candidate_file_path is None:
            import glob
            search_patterns = [
                os.path.join(candidate_file, "*.parquet"),
                os.path.join(candidate_file, "*.jsonl"),
                os.path.join(candidate_file, "*.json"),
            ]
            for pattern in search_patterns:
                matches = sorted(glob.glob(pattern))
                if matches:
                    candidate_file_path = matches[0]
                    break

        if candidate_file_path is None:
            raise FileNotFoundError(f"Candidate file not found in directory {candidate_file}. Tried patterns:"
                                    f" {possible_candidate_paths}")
        candidate_file = candidate_file_path
    elif candidate_file and not os.path.exists(candidate_file):
        raise FileNotFoundError(f"Candidate file not found: {candidate_file}")

    print(f"Loading queries from: {query_file}")
    print(f"Loading candidates from: {candidate_file}")
    print(f"Category: {category}, Subset: {subset_name}")

    query_data = load_query_data(query_file)
    candidate_data = load_candidate_data(candidate_file)

    print(f"Loaded {len(query_data)} queries and {len(candidate_data)} candidates for category '{category}'")

    qry_dataset = Dataset.from_list(query_data)
    num_rows = len(query_data)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'

    qry_dataset = qry_dataset.map(
        lambda x: data_prepare_query(x, **kwargs),
        batched=True,
        batch_size=64,
        remove_columns=['qry_text', 'qry_id', 'label_names', 'subtask'],
        drop_last_batch=False
    )

    try:
        _ = len(qry_dataset)
    except (TypeError, AttributeError):
        qry_dataset = DatasetWithLength(qry_dataset, num_rows)

    corpus_rows = []
    for cand in candidate_data:
        empty_image = create_empty_image_dict(data_args.image_resolution)
        corpus_rows.append({
            "cand_text": [cand['cand_text']],
            "cand_image": [empty_image],
            "dataset_infos": {
                "cand_names": [cand['cand_id']],
            }
        })

    corpus = Dataset.from_list(corpus_rows)

    print(f"Created query dataset with {len(query_data)} samples")
    print(f"Created corpus with {len(corpus_rows)} candidates")

    return qry_dataset, corpus
