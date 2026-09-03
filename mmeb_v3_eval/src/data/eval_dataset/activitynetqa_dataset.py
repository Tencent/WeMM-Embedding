import json
import os
import shutil

from datasets import load_dataset

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.utils.vision_utils.vision_utils import process_video_frames, load_frames
from src.model.processor import VLM_VIDEO_TOKENS
import random
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
OPTIONS = ['yes', 'no']


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']
    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_name, query, answer, question_id in \
            zip(batch_dict['video_name'], batch_dict['question'], batch_dict['answer'], batch_dict['question_id']):
        query = process_query(query + '? (A) yes; (B)'
                                      ' no.', prompt=TASK_PROMPT, video_token=VLM_VIDEO_TOKENS[model_backbone])
        query_texts.append([query])
        video_path = f'{video_root}/v_{video_name}.mp4'
        frame_dir = f'{frame_root}/v_{video_name}'
        frames = load_frames(frame_dir)
        if not frames:
            # print(f'Extracting frames for: {video_path}')
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
            # print(f'[{DATASET_PARSER_NAME}] Extracted #frames: {saved_frames}, dumped to {frame_dir}')

        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        # print(f'[{DATASET_PARSER_NAME}] Loaded #frames: {len(qry_frame_paths)}, from {frame_dir}')
        qry_frames = {
            "bytes": [None] * len(qry_frame_paths),
            "paths": qry_frame_paths,
            "resolutions": [None] * len(qry_frame_paths),
        }
        query_images.append([qry_frames])
        cand_texts.append(OPTIONS)
        cand_images.append([None] * len(OPTIONS))
        dataset_info = {
            "question_id": question_id,
            "video_id": video_name,
            "query": query,
            "cand_names": OPTIONS,
            "answer": answer,
            "label_name": answer,
            "answer_idx": OPTIONS.index(answer),
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


def sub_sample(video_dir, video_export_dir):
    dataset = load_dataset(EVAL_DATASET_HF_PATH, split="test")
    print(f"Loading {EVAL_DATASET_HF_PATH}, {len(dataset)} samples")
    yesno_count = 0
    yesno_mp4_count = 0
    row_idx = []
    for row_id, row in enumerate(dataset):
        if row['answer'] not in ['yes', 'no']:
            print(row)
        else:
            yesno_count += 1
            if not os.path.exists(video_dir + 'v_' + row['video_name'] + '.mp4'):
                continue
            yesno_mp4_count += 1
            row_idx.append(row_id)
        pass
    print(f'yesno_count={yesno_count}')
    print(f'yesno_mp4_count={yesno_mp4_count}')
    yesno_dataset = dataset.select(row_idx)
    random_ids = random.sample(range(len(yesno_dataset)), 1000)
    random_ids = sorted(random_ids)
    yesno_dataset = yesno_dataset.select(random_ids)
    yesno_dataset.save_to_disk("data/ActivityNetQA/")
    with open('data/activitynetqa.jsonl', 'w') as f:
        for row in yesno_dataset:
            f.write(f'{json.dumps(row)}\n')
            video_path = video_dir + 'v_' + row['video_name'] + '.mp4'
            video_export_path = f'{video_export_dir}/r_{row["video_name"]}.mp4'
            shutil.copyfile(video_path, video_export_path)
    print("Done")
    exit()


DATASET_PARSER_NAME = "activitynetqa"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_activitynetqa_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs['dataset_name']

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

    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['video_export_dir'] = kwargs.get("video_export_dir", None)
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    return dataset, None
