from abc import ABCMeta, abstractmethod
from functools import wraps
from datasets import Dataset, Features, Value, Sequence


# Schema for evaluation dataset, not used in the code.
EVAL_QRY_FEATURES = Features(**{
    "query_text": Sequence(Value(dtype='string')),  # Only one element, but make it as a sequence for collator usage
    "query_image": Sequence({
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    }),
    "cand_text": Sequence(Value(dtype='string')),  # Here it's only for hard negatives.
    "cand_video": Sequence({
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    }),
    "dataset_infos": {
        "cand_names": Sequence(Value(dtype='string')),
        "label_name": Value(dtype='string')
    }
})

EVAL_CAND_FEATURES = Features(**{
    "cand_text": Sequence(Value(dtype='string')),  # Only one element, but make it as a sequence for collator usage
    "cand_image": Sequence({
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    }),
    "dataset_infos": {
        "cand_name": Value(dtype='string'),
    },
})

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


class ImageVideoInstance:
    """
    len(bytes) == len(path) == len(resolution) == 1: image
    len(bytes) == len(path) == len(resolution) > 1: multi-image / video
    """

    def __init__(self, bytes, paths, resolutions):
        assert len(bytes) == len(paths) == len(resolutions)
        self.bytes = bytes
        self.paths = paths
        self.resolutions = resolutions

    def to_dict(self):
        return {
            "bytes": self.bytes,
            "paths": self.paths,
            "resolutions": self.resolutions,
        }


class AutoEvalPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoEvalPairDataset.registry:
            AutoEvalPairDataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs)
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                # print(f"Adding {dataset_name}")
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass


def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data.get('query_text', batch_data.get('cand_text', [])))
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        return batch_data

    return wrapper


def generate_cand_dataset(dataset, corpus):
    """
    Used for generating candidate datasets.
    Flatten candidates, merge with corpus, deduplication
    Supports cand_image and cand_video (for video datasets)
    """
    cand_rows = []
    all_cand_name = set()

    def _normalize_audio_item(a):
        # Common pattern in datasets: cand_audio stores a single-item list.
        if isinstance(a, list) and len(a) == 1:
            return a[0]
        return a

    for row in dataset:
        cand_visual_key = None
        if "cand_video" in row:
            cand_visual_key = "cand_video"
        elif "cand_image" in row:
            cand_visual_key = "cand_image"

        if (
            ("cand_text" not in row)
            or (cand_visual_key is None)
            or ("dataset_infos" not in row)
            or (not isinstance(row["dataset_infos"], dict))
            or ("cand_names" not in row["dataset_infos"])
        ):
            continue

        cand_audio_seq = row.get("cand_audio", None)
        if cand_audio_seq is not None:
            assert len(cand_audio_seq) == len(row["cand_text"])
        else:
            cand_audio_seq = [None] * len(row["cand_text"])

        assert len(row["cand_text"]) == len(row[cand_visual_key]) == len(row["dataset_infos"]["cand_names"]) == len(
            cand_audio_seq)
        for cand_text, cand_visual, cand_name, cand_audio in zip(
            row["cand_text"], row[cand_visual_key], row["dataset_infos"]["cand_names"], cand_audio_seq
        ):
            if cand_name not in all_cand_name:
                cand_row = {
                    "cand_text": [cand_text],
                    "dataset_infos": {"cand_name": cand_name},
                }
                cand_row[cand_visual_key] = [cand_visual]
                if cand_audio is not None:
                    cand_row["cand_audio"] = _normalize_audio_item(cand_audio)
                cand_rows.append(cand_row)
                all_cand_name.add(cand_name)

    if corpus is not None:
        for row in corpus:
            corpus_visual_key = "cand_video" if "cand_video" in row else "cand_image"

            assert len(row["cand_text"]) == len(row[corpus_visual_key]) == len(row["dataset_infos"]["cand_names"]) == 1
            cand_name = row["dataset_infos"]["cand_names"][0]
            if cand_name not in all_cand_name:
                corpus_row = {
                    "cand_text": row["cand_text"],
                    "dataset_infos": {"cand_name": row["dataset_infos"]["cand_names"][0]},
                }
                corpus_row[corpus_visual_key] = row[corpus_visual_key]
                if "cand_audio" in row:
                    # corpus row is one candidate; keep a single audio item.
                    corpus_audio = row["cand_audio"]
                    if isinstance(corpus_audio, list) and len(corpus_audio) == 1:
                        corpus_audio = corpus_audio[0]
                    corpus_row["cand_audio"] = _normalize_audio_item(corpus_audio)
                cand_rows.append(corpus_row)
                all_cand_name.add(cand_name)

    cand_dataset = Dataset.from_list(cand_rows)
    return cand_dataset
