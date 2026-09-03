import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    add_metainfo_hook,
    RESOLUTION_MAPPING,
    ImageVideoInstance,
)
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text


TASK_INST_QRY = "Retrieve a video that matches this description:"
TASK_INST_TGT = "Describe the visual content of this video."


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    video_root, frame_root = kwargs['video_root'], kwargs['frame_root']
    model_backbone = kwargs['model_backbone']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_name, video_path, captions in zip(batch_dict['video_id'], batch_dict["video"], batch_dict['caption']):

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=captions[0])])
        query_images.append([None])

        video_path = os.path.join(video_root, video_path)
        frame_dir = os.path.join(frame_root, video_name)
        if os.path.exists(frame_dir):
            video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        elif os.path.exists(video_path):
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
            video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        else:
            print(f"Skipping missing video and frames: {video_path}")
            continue

        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)])
        cand_images.append([ImageVideoInstance(
            bytes=[None] * len(video_frame_paths),
            paths=video_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
        ).to_dict()])
        dataset_infos.append({
            "cand_names": [video_name],
            "label_name": video_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "msvd"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_msvd_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs['dataset_name']
    if dataset_name in EVAL_DATASET_LOCAL_PATH:
        local_path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
        if os.path.exists(local_path_info[0]):
            print(f"Loading {dataset_name} from local path: {local_path_info[0]}")
            dataset = load_hf_dataset(local_path_info + ("local",))
        else:
            print(f"Local path {local_path_info[0]} not found, falling back to HuggingFace Hub")
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
    else:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    corpus = None  # No additional corpus

    return dataset, corpus
