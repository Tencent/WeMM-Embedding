import os

from datasets import load_dataset

from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from src.utils.dataset_utils import sample_dataset
from src.utils.vision_utils.vision_utils import process_video_frames, load_frames, qa_template
from src.model.processor import VLM_VIDEO_TOKENS
import datasets
import cv2


def process_query(query, prompt, video_token=''):
    if prompt:
        query = f'{video_token} {prompt} {query}'
    else:
        query = f'{query} {video_token}'
    return query


TASK_PROMPT = (
    "Given a video and a question, select the most accurate answer from the provided candidates. Return"
    " only the exact text of your chosen answer. Question: ")
OPTIONS = ['A', 'B', 'C', 'D']


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    model_backbone = kwargs['model_backbone']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']
    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for video_id, query, answer, question_type, options, subset, image in \
            zip(batch_dict['id'], batch_dict['question'], batch_dict['answer'],
                batch_dict['question_type'], batch_dict['options'], batch_dict['subset'], batch_dict['image']):
        if question_type != 'multiple-choice':
            continue
        query = process_query(query, prompt=TASK_PROMPT, video_token=VLM_VIDEO_TOKENS[model_backbone])
        query, cands, _, _ = qa_template(query, options, answer)
        query_texts.append([query])
        video_path = f'{video_root}/{subset}/{video_id}.mp4'
        frame_dir = f'{frame_root}/{subset}/{video_id}'
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
        cand_texts.append([o[o.find('. '):].strip('. ') for o in options])
        cand_images.append([None] * len(options))
        dataset_info = {
            "video_id": video_id,
            "query": query,
            "cand_names": options,
            "answer": options[answer],
            "label_name": answer,
            "answer_idx": answer,
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


subset_names = ['Perception', 'Comprehension', 'Adaptation']

DATASET_PARSER_NAME = "videommmu"
DATASET_HF_PATH = "lmms-lab/VideoMMMU"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_videommmu_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get('dataset_name', DATASET_PARSER_NAME)
    use_local = False

    if dataset_name in EVAL_DATASET_LOCAL_PATH:
        local_path_info = EVAL_DATASET_LOCAL_PATH[dataset_name]
        local_path = local_path_info[0]
        if os.path.exists(local_path):
            print(f"Loading {dataset_name} from local path: {local_path}")
            use_local = True
            base_path = local_path

    if not use_local:
        print(f"Loading {dataset_name} from HuggingFace Hub: {DATASET_HF_PATH}")

    subsets = []
    for subset_name in subset_names:
        if use_local:
            dataset = load_dataset(base_path, subset_name, split="test")  # pylint: disable=used-before-assignment
        else:
            dataset = load_dataset(DATASET_HF_PATH, subset_name, split="test")
        new_column = [subset_name] * len(dataset)
        dataset = dataset.add_column("subset", new_column)
        subsets.append(dataset)
    dataset = datasets.concatenate_datasets(subsets)
    # TODO filter the dataset by question_type=='multiple-choice'

    print(f"Loading {DATASET_HF_PATH}, {len(dataset)} samples")
    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME
    dataset = sample_dataset(dataset, **kwargs)
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)

    return dataset, None
