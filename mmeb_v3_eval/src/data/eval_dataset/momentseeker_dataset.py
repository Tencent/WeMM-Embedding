import os

from datasets import load_dataset
from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.utils.dataset_utils import load_hf_dataset, sample_dataset

from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.data.eval_dataset.base_eval_dataset import ImageVideoInstance
from src.utils.vision_utils.vision_utils import load_frames, save_frames
from src.model.processor import process_input_text

TASK_INST_QRY_TEXT = "Find the clip that corresponds to the given text:"
TASK_INST_QRY_IMG = "Select the video clip that aligns with the given text and image:"
TASK_INST_QRY_VIDEO = "Find the clip that corresponds to the given sentence and video segment:"
TASK_INST_TGT = "Understand the content of the provided video clip."


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution = kwargs['image_resolution']
    # metadata
    num_video_frames = kwargs["num_video_frames"]
    model_backbone = kwargs["model_backbone"]
    video_root, clip_root, frame_root = kwargs["video_root"], kwargs["clip_root"], kwargs["frame_root"]

    query_texts, query_images, cand_texts, cand_clip_images, dataset_infos = [], [], [], [], []
    for query, positive_frames, negative_frames, input_frames in \
            zip(batch_dict['query'], batch_dict["positive_frames"], batch_dict["negative_frames"], batch_dict[
                "input_frames"]):

        if (input_frames.endswith(".mp4")):
            query_texts.append([process_input_text(TASK_INST_QRY_VIDEO,
                               model_backbone, text=query, add_video_token=True)])
            query_video_name = input_frames.split(".mp4")[0].replace("/", "_")
            if query_video_name == 'movie101_77':  # TODO @yuepeng a buggy video?
                pass
            query_frame_dir = os.path.join(frame_root, "video_frames", query_video_name)
            if not os.path.exists(query_frame_dir):
                query_video_path = os.path.join(video_root, input_frames)
                save_frames(video_path=query_video_path,
                            frame_dir=query_frame_dir,
                            max_frames_saved=num_video_frames)
            qry_frame_paths = load_frames(query_frame_dir)
            query_images.append([ImageVideoInstance(
                bytes=[None] * len(qry_frame_paths),
                paths=qry_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
            ).to_dict()])
        elif (input_frames.endswith(".jpg")):
            query_texts.append([process_input_text(TASK_INST_QRY_IMG,
                               model_backbone, text=query, add_image_token=True)])
            input_image_path = os.path.join(frame_root, "", f"query_{input_frames}")
            query_images.append([ImageVideoInstance(
                bytes=[None],
                paths=[input_image_path],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict()])
        else:
            query_texts.append([process_input_text(TASK_INST_QRY_TEXT, model_backbone, text=query)])
            query_images.append([None])

        pos_clip_paths = [entry["output_path"] for entry in positive_frames]
        neg_clip_paths = [entry["output_path"] for entry in negative_frames]

        pos_clip_name, cand_clip_names, cand_frames = [], [], []
        for path in pos_clip_paths:
            cand_clip_name = path.replace("/", "_").split(".mp4")[0]
            cand_clip_frame_dir = os.path.join(frame_root, "video_frames", cand_clip_name)
            if not os.path.exists(cand_clip_frame_dir):
                cand_clip_abs_path = os.path.join(clip_root, path)
                save_frames(
                    video_path=cand_clip_abs_path,
                    frame_dir=cand_clip_frame_dir,
                    max_frames_saved=num_video_frames)
            pos_clip_frames = load_frames(cand_clip_frame_dir)
            cand_frames.append(ImageVideoInstance(
                bytes=[None] * len(pos_clip_frames),
                paths=pos_clip_frames,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(pos_clip_frames),
            ).to_dict())
            cand_clip_names.append(cand_clip_frame_dir)
            pos_clip_name.append(cand_clip_frame_dir)
        for path in neg_clip_paths:
            cand_clip_name = path.replace("/", "_").split(".mp4")[0]
            cand_clip_frame_dir = os.path.join(frame_root, "video_frames", cand_clip_name)
            if not os.path.exists(cand_clip_frame_dir):
                cand_clip_abs_path = os.path.join(clip_root, path)
                save_frames(
                    video_path=cand_clip_abs_path,
                    frame_dir=cand_clip_frame_dir,
                    max_frames_saved=num_video_frames)
            neg_clip_frames = load_frames(cand_clip_frame_dir)
            cand_frames.append(ImageVideoInstance(
                bytes=[None] * len(neg_clip_frames),
                paths=neg_clip_frames,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(neg_clip_frames),
            ).to_dict())
            cand_clip_names.append(cand_clip_frame_dir)
        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)]
                          * len(pos_clip_paths + neg_clip_paths))
        cand_clip_images.append(cand_frames)
        dataset_infos.append({
            "cand_names": cand_clip_names,
            "label_name": pos_clip_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_clip_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "momentseeker"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_momentseeker_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs.get('dataset_name')

    if kwargs.get("data_path", None) is not None:
        dataset = load_dataset("json", data_files=kwargs["data_path"])
        dataset = dataset["train"]
    else:
        if dataset_name in EVAL_DATASET_LOCAL_PATH:
            local_path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
            local_path = local_path_info[0]
            if os.path.exists(local_path):
                print(f"Loading {dataset_name} from local path: {local_path}")
                dataset = load_hf_dataset(local_path_info + ("local",))
            else:
                print(f"Local path {local_path} not found, falling back to HuggingFace Hub")
                dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
        else:
            dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])

    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = kwargs['dataset_name'] if kwargs['dataset_name'] else DATASET_PARSER_NAME

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=2048, num_proc=1,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    return dataset, None
