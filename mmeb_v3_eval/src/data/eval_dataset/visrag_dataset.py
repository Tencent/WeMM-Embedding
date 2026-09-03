import os
import hashlib

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


TASK_INST_QRY = "Find a document image that matches the given query:"
TASK_INST_TGT = "Describe the content of this document image."
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

        for image_name, rel_score in qrels_mapping[query_id].items():
            # some image_name are super long...
            base, ext = os.path.splitext(image_name)
            short_base = base[:50] + "_" + hashlib.md5(
                image_name.encode('utf-8')).hexdigest()[:8]  # Truncate base, add original filename hash
            new_imagename = short_base + ext
            image_path = f'{image_root}/{new_imagename}'
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'Image path {image_path} not found.')
            cand_text.append(process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True))
            cand_image.append(ImageVideoInstance(
                bytes=[None],
                paths=[image_path],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict())
            cand_names.append(image_name)
            label_names.append(image_name)
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
    for image_name, image in zip(batch_dict['corpus-id'], batch_dict['image']):
        # some image_name are super long...
        base, ext = os.path.splitext(image_name)
        short_base = base[:50] + "_" + hashlib.md5(
            image_name.encode('utf-8')).hexdigest()[:8]  # Truncate base, add original filename hash
        new_imagename = short_base + ext
        image_path = f'{image_root}/{new_imagename}'
        if not os.path.exists(image_path):
            os.makedirs(image_root, exist_ok=True)
            if isinstance(image, dict):
                image_bytes = image.get("bytes", None)
                if image_bytes is None and image.get("path") and os.path.exists(image["path"]):
                    with open(image["path"], "rb") as f:
                        image_bytes = f.read()
                if image_bytes is None:
                    raise ValueError(f"Image bytes missing for {image_name}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
            else:
                image.save(image_path)
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_image_token=True)])
        cand_images.append([ImageVideoInstance(
            bytes=[None],
            paths=[image_path],
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
        ).to_dict()])
        dataset_infos.append({
            "cand_names": [image_name],
        })

    return {"cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "visrag"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_visrag_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs['dataset_name']

    if dataset_name in EVAL_DATASET_LOCAL_PATH:
        local_path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
        hf_dataset_name = local_path_info[0]
        hf_dataset_split = local_path_info[2]

        if os.path.exists(hf_dataset_name):
            print(f"Loading {dataset_name} from local path: {hf_dataset_name}")
            qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split, "local"))
            corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split, "local"))
            dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split, "local"))
        else:
            print(f"Local path not found for {dataset_name}, falling back to HuggingFace Hub")
            hf_dataset_name = EVAL_DATASET_HF_PATH[dataset_name][0]
            hf_dataset_split = EVAL_DATASET_HF_PATH[dataset_name][2]
            qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
            corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
            dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
    else:
        hf_dataset_name = EVAL_DATASET_HF_PATH[dataset_name][0]
        hf_dataset_split = EVAL_DATASET_HF_PATH[dataset_name][2]
        # BEIR format
        qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
        corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
        dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))

    qrels_mapping = load_qrels_mapping(qrels)
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['qrels_mapping'] = qrels_mapping

    corpus = corpus.map(lambda x: corpus_prepare(x, **kwargs), batched=True,
                        batch_size=1024, num_proc=4,
                        drop_last_batch=False, load_from_cache_file=False)
    corpus = corpus.select_columns(['cand_text', 'cand_image', 'dataset_infos'])
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=1024, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, corpus
