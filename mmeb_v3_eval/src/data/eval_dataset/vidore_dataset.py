import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    add_metainfo_hook,
    RESOLUTION_MAPPING,
    ImageVideoInstance,
)
from src.utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping
from src.model.processor import process_input_text
from src.utils.basic_utils import print_master
from PIL import Image
import io

TASK_INST_QRY = "Find a document image that matches the given query:"
TASK_INST_TGT = "Describe the content of this document image."
# ColPali models use no prompts
# TASK_INST_QRY = ""
# TASK_INST_TGT = ""


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    qrels_mapping = kwargs['qrels_mapping']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for query_id, query in zip(batch_dict['query-id'], batch_dict['query']):
        query_texts.append([process_input_text(f"{query}\n{TASK_INST_QRY}", model_backbone)])
        query_images.append([None])
        cand_text, cand_image, cand_names, label_names = [], [], [], []
        rel_scores = []

        for corpus_id, rel_score in qrels_mapping[query_id].items():
            image_path = f'{image_root}/{corpus_id}.png'
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'Image path {image_path} not found.')
            cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True))
            cand_image.append(ImageVideoInstance(
                bytes=[None],
                paths=[image_path],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict())
            cand_names.append(corpus_id)
            label_names.append(corpus_id)
            rel_scores.append(rel_score)
        cand_texts.append(cand_text)
        cand_images.append(cand_image)
        dataset_infos.append({
            "cand_names": cand_names,
            "label_name": label_names,
            "rel_scores": rel_scores,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


def corpus_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    cand_texts, cand_images, dataset_infos = [], [], []
    for corpus_id, image in zip(batch_dict['corpus-id'], batch_dict['image']):
        image_path = f'{image_root}/{corpus_id}.png'

        if not os.path.exists(image_path):
            os.makedirs(image_root, exist_ok=True)
            if isinstance(image, Image.Image):
                image.save(image_path)
            elif isinstance(image, dict):
                if image.get("bytes", None) is not None:
                    Image.open(io.BytesIO(image["bytes"])).save(image_path)
                elif image.get("path", None) is not None:
                    image_path = image["path"]
                else:
                    raise ValueError(f"Unsupported image dict format for corpus_id={corpus_id}:"
                                     f" keys={list(image.keys())}")
            elif isinstance(image, (bytes, bytearray)):
                Image.open(io.BytesIO(image)).save(image_path)
            else:
                raise ValueError(f"Unsupported image type for corpus_id={corpus_id}: {type(image)}")

        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True)])
        cand_images.append([ImageVideoInstance(
            bytes=[None],
            paths=[image_path],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict()])
        dataset_infos.append({
            "cand_names": [corpus_id],
        })

    return {"cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "vidore"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_vidore_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs['dataset_name']

    if dataset_name in EVAL_DATASET_LOCAL_PATH:
        hf_dataset_name = EVAL_DATASET_LOCAL_PATH[dataset_name][0]
        lang = EVAL_DATASET_LOCAL_PATH[dataset_name][1]
        hf_dataset_split = EVAL_DATASET_LOCAL_PATH[dataset_name][2]
        if os.path.exists(hf_dataset_name):
            print(f"Loading {dataset_name} from local path: {hf_dataset_name}")
            dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split, "local"))
            if lang is not None:
                dataset = dataset.filter(lambda example: example["language"] == lang)
            qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split, "local"))
            corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split, "local"))
        else:
            print(f"Local path {hf_dataset_name} not found, falling back to HuggingFace Hub")
            hf_dataset_name = EVAL_DATASET_HF_PATH[dataset_name][0]
            lang = EVAL_DATASET_HF_PATH[dataset_name][1]
            hf_dataset_split = EVAL_DATASET_HF_PATH[dataset_name][2]
            dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
            if lang is not None:
                dataset = dataset.filter(lambda example: example["language"] == lang)
            qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
            corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
    else:
        hf_dataset_name = EVAL_DATASET_HF_PATH[dataset_name][0]
        lang = EVAL_DATASET_HF_PATH[dataset_name][1]
        hf_dataset_split = EVAL_DATASET_HF_PATH[dataset_name][2]
        # BEIR format
        dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
        if lang is not None:
            dataset = dataset.filter(lambda example: example["language"] == lang)
        qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
        corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
    qrels_mapping = load_qrels_mapping(qrels)
    dataset = sample_dataset(dataset, **kwargs)
    print_master(f"Loaded {kwargs['dataset_name']}")
    print_master(f"#hf_dataset_name={hf_dataset_name}")
    print_master(f"#hf_dataset_split={hf_dataset_split}")
    print_master(f"#queries={len(dataset)}")
    print_master(f"#cand={len(corpus)}")
    print_master(f"#qrels={len(qrels)}")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['qrels_mapping'] = qrels_mapping

    corpus = corpus.map(lambda x: corpus_prepare(x, **kwargs), batched=True,
                        batch_size=2048, num_proc=1,
                        drop_last_batch=False, load_from_cache_file=False)
    corpus = corpus.select_columns(['cand_text', 'cand_image', 'dataset_infos'])
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=2048, num_proc=1,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, corpus
