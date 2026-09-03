import os

from datasets import Dataset
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    add_metainfo_hook,
    RESOLUTION_MAPPING,
    ImageVideoInstance,
)
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.eval_dataset.video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    image_resolution = kwargs['image_resolution']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    dataset_name = kwargs['dataset_name']
    model_backbone = kwargs['model_backbone']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_id, label in zip(batch_dict['video_id'], batch_dict['pos_text']):
        video_path = os.path.join(video_root, video_id + '.mp4')
        frame_dir = os.path.join(frame_root, video_id)
        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        query_texts.append([process_input_text(DATASET_INSTRUCTION[dataset_name],
                           model_backbone, add_video_token=True)])
        query_images.append([ImageVideoInstance(
            bytes=[None] * len(video_frame_paths),
            paths=video_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
        ).to_dict()])

        cand_texts.append([label])
        cand_images.append([None])
        dataset_info = {
            "cand_names": [label],
            "label_name": label,
        }
        dataset_infos.append(dataset_info)

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "video_classification"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_video_class_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs['dataset_name']

    if dataset_name in EVAL_DATASET_LOCAL_PATH:
        local_path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
        local_path = local_path_info[0]
        if os.path.exists(local_path):
            print(f"Loading {dataset_name} from local path: {local_path}")
            dataset = load_hf_dataset((local_path, local_path_info[1], local_path_info[2], "local"))
        else:
            print(f"Local path {local_path} not found, falling back to HuggingFace Hub")
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
    else:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])

    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    # Restrict the label corpus to classes actually present in the eval split:
    # classes without any test video can never be correct and only add
    # distractors (e.g. Kinetics-700 eval videos cover 519 of 700 classes).
    all_labels = VIDEOCLS_LABEL_MAPPING[dataset_name]
    present_labels = set(dataset['pos_text'])
    corpus_labels = [label for label in all_labels if label in present_labels]
    if not corpus_labels:
        corpus_labels = all_labels

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    corpus = Dataset.from_list([{
        "cand_text": [label],
        "cand_image": [None],
        "dataset_infos": {"cand_names": [label]}} for label in corpus_labels])

    return dataset, corpus
