import os

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.utils.vision_utils.vision_utils import process_video_frames, load_frames, qa_template
from src.model.processor import VLM_VIDEO_TOKENS
import cv2


def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token}{prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_PROMPT = (
    "Given a video and a question, select the most accurate answer from the provided candidates. Return"
    " only the exact text of your chosen answer. Question:")
OPTIONS = ['A', 'B', 'C', 'D']


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']
    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_id, query, answer, qid, _type, a0, a1, a2, a3, a4 in \
            zip(batch_dict['video'], batch_dict['question'], batch_dict['answer'],
                batch_dict['qid'], batch_dict['type'], batch_dict['a0'], batch_dict['a1'],
                batch_dict['a2'], batch_dict['a3'], batch_dict['a4']):
        options = [a0, a1, a2, a3, a4]
        query = process_query(query, prompt=TASK_PROMPT, video_token=VLM_VIDEO_TOKENS[model_backbone])
        # qa_template labels the options "(A) ..." inside the query; candidates
        # must carry the same labels so the model returns the exact labeled text.
        query, labeled_options, labeled_answer, _ = qa_template(query, options, options[answer])
        query_texts.append([query])
        video_path = f'{video_root}/{video_id}.mp4'
        frame_dir = f'{frame_root}/{video_id}'
        frames = load_frames(frame_dir)
        if not frames:
            print(f'Extracting frames for: {video_path}')
            os.makedirs(frame_dir, exist_ok=True)
            assert os.path.exists(video_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames_saved)
            frame_idx = 0
            saved_frames = 0
            while saved_frames < max_frames_saved:
                assert cap.isOpened(), "not cap.isOpened()"
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to specific frame
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frame_dir, f"{saved_frames:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                saved_frames += 1
                frame_idx += step
            cap.release()
            print(f'[{DATASET_PARSER_NAME}] Extracted #frames: {saved_frames}, dumped to {frame_dir}')

        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        # print(f'[{DATASET_PARSER_NAME}] Loaded #frames: {len(qry_frame_paths)}, from {frame_dir}')
        qry_frames = {
            "bytes": [None] * len(qry_frame_paths),
            "paths": qry_frame_paths,
            "resolutions": [None] * len(qry_frame_paths),
        }
        query_images.append([qry_frames])
        cand_texts.append(labeled_options)
        cand_images.append([None] * len(labeled_options))
        dataset_info = {
            "question_id": qid,
            "video_id": video_id,
            "query": query,
            "cand_names": labeled_options,
            "answer": labeled_answer,
            "label_name": labeled_answer,
            "answer_idx": answer,
            "type": _type,
            "qry_frame_paths": qry_frame_paths,
        }
        dataset_infos.append(dataset_info)
    if len(query_texts) == 0:
        print('something went wrong')
    # print_rank(f"dataset.map(): global_dataset_name={kwargs.get('global_dataset_name', DATASET_PARSER_NAME)},
    # batch_size={batch_size}, processed_batch_size={len(query_texts)}")
    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "nextqa"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_nextqa_dataset(model_args, data_args, *args, **kwargs):
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

    # dataset = dataset.filter(lambda example: example['video'] == 4740931975)
    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)

    return dataset, None
