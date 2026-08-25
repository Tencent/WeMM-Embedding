import datasets
from datasets import load_dataset
from src.utils.basic_utils import print_rank
import os


def sample_dataset(dataset, **kwargs):
    dataset_name = kwargs.get("dataset_name", "UNKNOWN-DATASET")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)

    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if type(num_sample_per_subset) is int and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print_rank(f"Subsample {dataset_name} to {len(dataset)} samples")

    return dataset


def load_qrels_mapping(qrels):
    """
    Returns:
        {
            "qid1": {"docA": 2, "docB": 1},
            "qid2": {"docC": 3},
            ...
        }
    """
    qrels_mapping = {}

    for row in qrels:
        qid = row["query-id"]
        docid = row["corpus-id"]
        score = row["score"]

        if score > 0:
            if qid not in qrels_mapping:
                qrels_mapping[qid] = {}
            # keep the higher score if already exists
            existing_score = qrels_mapping[qid].get(docid, 0)
            qrels_mapping[qid][docid] = max(existing_score, score)

    return qrels_mapping


def local_dataset_exists(local_path: str, subset: str = None) -> bool:
    """Check whether a locally staged dataset is actually usable.

    A base data directory may exist while only some subsets are staged under
    it. For directory-based datasets with a subset name, require the subset
    subdirectory to be present, so callers can fall back to the HuggingFace
    Hub instead of crashing with "BuilderConfig '<subset>' not found".
    """
    if not os.path.exists(local_path):
        return False
    if subset and os.path.isdir(local_path):
        return os.path.isdir(os.path.join(local_path, subset))
    return True


def load_hf_dataset(hf_path):
    if len(hf_path) == 4 and hf_path[3] == "local":
        return load_local_hf_dataset(hf_path[0], hf_path[1], hf_path[2])

    repo, subset, split = hf_path
    if subset and split:
        return load_dataset(repo, subset, split=split)
    elif subset:
        return load_dataset(repo, subset)
    elif split:
        return load_dataset(repo, split=split)
    else:
        return load_dataset(repo)


def load_local_hf_dataset(dataset_path: str, subset: str = None, split: str = None):
    """
    Loads a Hugging Face dataset from local Parquet files or JSON/JSONL files.
    Args:
        dataset_path (str): The base path to the dataset directory or file
        subset (str, optional): The name of the subdirectory containing the data files (e.g., "corpus").
        split (str, optional): Which split of the data to load (e.g., "train", "test").
    Returns: Dataset or DatasetDict: The loaded dataset.
    """
    if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        dataset = datasets.load_dataset("json", data_files=dataset_path)
        if isinstance(dataset, datasets.DatasetDict):
            available_splits = list(dataset.keys())
            if len(available_splits) == 1 and split and split not in available_splits:
                dataset = dataset[available_splits[0]]
            elif split and split in available_splits:
                dataset = dataset[split]
        return dataset
    else:
        if subset:
            subset_path = os.path.join(dataset_path, subset)
            if os.path.isdir(subset_path):
                import glob
                parquet_files = []
                split_parquet_files = []
                if split:
                    split_parquet_files = sorted(glob.glob(os.path.join(subset_path, f"{split}-*.parquet")))
                    if not split_parquet_files:
                        split_parquet_files = sorted(glob.glob(os.path.join(subset_path, f"{split}*.parquet")))
                    parquet_files = split_parquet_files
                if not parquet_files:
                    parquet_files = sorted(glob.glob(os.path.join(subset_path, "*.parquet")))
                if parquet_files:
                    if split and not split_parquet_files:
                        print_rank(
                            f"[load_local_hf_dataset] split='{split}' has no matching parquet in {subset_path}, "
                            f"fallback to all parquet files."
                        )
                    import pandas as pd
                    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
                    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
                    return dataset

                try:
                    dataset = datasets.load_dataset("imagefolder", data_dir=subset_path, split=split)
                    return dataset
                except Exception as e:
                    print(f"Failed to load as imagefolder: {e}, trying standard method...")

        if subset and split:
            dataset = datasets.load_dataset(dataset_path, subset, split=split)
        elif subset:
            dataset = datasets.load_dataset(dataset_path, subset)
        elif split:
            dataset = datasets.load_dataset(dataset_path, split=split)
        else:
            dataset = datasets.load_dataset(dataset_path)
        return dataset


def load_hf_dataset_multiple_subset(hf_path, subset_names):
    """
    Load and concatenate multiple subsets from a Hugging Face dataset (e.g. MVBench)
    """
    repo, _, split = hf_path
    subsets = []
    for subset_name in subset_names:
        dataset = load_dataset(repo, subset_name, split=split)
        new_column = [subset_name] * len(dataset)
        dataset = dataset.add_column("subset", new_column)
        subsets.append(dataset)
    dataset = datasets.concatenate_datasets(subsets)

    return dataset
