import os
import subprocess
from typing import List, Dict, Any

import datasets
from datasets import Audio

from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    add_metainfo_hook,
    ImageVideoInstance,
    RESOLUTION_MAPPING,
)
from src.constant.dataset_hflocal_path import BASE_RAW_DATA_DIR
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames
from src.model.processor import process_input_text


TASK_INST_TGT = "Understand the content of the provided video."


def _resolve_existing_dir(candidates: List[str]) -> str:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return ""


def _case_variants(path: str) -> List[str]:
    if not path:
        return []
    variants = [path]
    variants.append(path.replace("/audio-tasks/AVE/", "/audio-tasks/ave/"))
    variants.append(path.replace("/audio-tasks/ave/", "/audio-tasks/AVE/"))
    variants.append(path.replace("/audio-tasks/AVE", "/audio-tasks/ave"))
    variants.append(path.replace("/audio-tasks/ave", "/audio-tasks/AVE"))
    dedup = []
    seen = set()
    for p in variants:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _audio_filename(video_id: str, clip_id: str) -> str:
    return f"{video_id}_{clip_id}.wav"


def _extract_clip_audio_ffmpeg(video_path: str, start: float, end: float, out_wav: str) -> bool:
    if os.path.isfile(out_wav):
        return True
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    duration = max(float(end) - float(start), 0.05)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(float(start)),
        "-t",
        str(duration),
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        out_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return os.path.isfile(out_wav)


def _parse_split_file(split_file: str) -> List[Dict[str, Any]]:
    """
    Parse an AVE split file (e.g. testSet.txt) with the format:
    Category&VideoID&Quality&StartTime&EndTime
    """
    samples = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Category"):
                continue
            parts = line.split("&")
            if len(parts) < 5:
                continue
            category, video_id, quality, start, end = parts[:5]
            clip_id = f"{video_id}_{int(float(start))}_{int(float(end))}"
            samples.append(
                {
                    "category": category,
                    "video_id": video_id,
                    "quality": quality,
                    "start": float(start),
                    "end": float(end),
                    "clip_id": clip_id,
                }
            )
    return samples


@add_metainfo_hook
def data_prepare(batch_dict, all_video_ids=None, **kwargs):
    """
    Build samples for AVE audio->video retrieval:
    - query_audio: audio object (already decoded by HF datasets)
    - query_text: query instruction text
    - query_image: None placeholder
    - cand_video: candidate video pool (globally fixed, 8 frames per video)
    - cand_text: candidate video descriptions (globally fixed)
    - dataset_infos: ground-truth candidate IDs and label info
    """
    assert all_video_ids is not None and len(all_video_ids) > 0

    video_root = kwargs.get("video_root", "")
    frame_root = kwargs.get("frame_root", "")
    num_frames = kwargs.get("num_frames", 8)
    max_frames_saved = kwargs.get("max_frames_saved", 100)
    image_resolution = kwargs.get("image_resolution", "low")
    model_backbone = kwargs.get("model_backbone")

    cand_names = all_video_ids
    cand_text = [process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True) for _ in cand_names]
    cand_image = []

    for vid in cand_names:
        video_path = os.path.join(video_root, f"{vid}.mp4")
        frame_dir = os.path.join(frame_root, vid)

        save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        cand_image.append(ImageVideoInstance(
            bytes=[None] * len(video_frame_paths),
            paths=video_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
        ).to_dict())

    vid2idx = {vid: i for i, vid in enumerate(cand_names)}

    query_texts, query_images, query_audios = [], [], []
    dataset_infos = []

    for category, video_id, clip_id, audio_obj in zip(
        batch_dict["category"],
        batch_dict["video_id"],
        batch_dict["clip_id"],
        batch_dict["query_audio"],
    ):
        query_audios.append(audio_obj)
        query_text = build_query_text("AVE")
        query_texts.append(query_text)
        assert isinstance(
            query_text,
            list) and len(query_text) == 1 and isinstance(
            query_text[0],
            str) and query_text[0].strip()
        query_images.append([None])

        label_idx = vid2idx[video_id]

        info = {
            "label_cand_id": label_idx,
            "label_name": video_id,
            "cand_names": cand_names,
            "query_id": clip_id,
            "corpus_id": video_id,
            "category": category,
        }
        dataset_infos.append(info)

    bs = len(query_texts)
    return {
        "query_text": query_texts,
        "query_image": query_images,
        "query_audio": query_audios,

        "cand_text": [cand_text] * bs,
        "cand_video": [cand_image] * bs,
        "cand_audio": [[]] * bs,

        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "ave_retrieval"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_ave_retrieval_dataset(model_args, data_args, **kwargs):
    """
    Loader for the AVE audio->video retrieval dataset.
    Each audio clip queries its corresponding full video.

    Args:
        data_path: AVE data root directory
        split_file: split file, defaults to testSet.txt
        audio_root: audio files root, defaults to {data_path}/audios
        video_root: video files root, defaults to {data_path}/AVE
    """
    data_path_input = kwargs.get(
        "data_path", os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "AVE", "AVE_Dataset")
    )
    data_path = _resolve_existing_dir(
        _case_variants(data_path_input)
        + [os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "ave", "AVE", "AVE_Dataset")]
    )
    if not data_path:
        raise AssertionError(
            "AVE data_path not found."
            " tried="
            f"{_case_variants(data_path_input) + [os.path.join(BASE_RAW_DATA_DIR, 'audio-tasks', 'ave', 'AVE', 'AVE_Dataset')]}"  # noqa: E501
        )

    split_file = kwargs.get("split_file", "testSet.txt")
    video_root_input = kwargs.get("video_root", os.path.join(data_path, "AVE"))
    video_root = _resolve_existing_dir(_case_variants(video_root_input) + [os.path.join(data_path, "AVE")])

    # Prefer configured audio_root if valid. If missing/empty, we will extract clips from mp4.
    audio_root_input = kwargs.get("audio_root", os.path.join(data_path, "audios"))
    resolved_audio_root = _resolve_existing_dir(_case_variants(audio_root_input))
    use_precomputed_audio = bool(resolved_audio_root)
    audio_root = resolved_audio_root if resolved_audio_root else audio_root_input
    if not audio_root:
        audio_root = os.path.join(data_path, "audios_extracted_16k")
    os.makedirs(audio_root, exist_ok=True)

    split_path = os.path.join(data_path, split_file)
    assert os.path.isfile(split_path), f"AVE split file not found: {split_path}"
    assert os.path.isdir(video_root), f"Video directory not found: {video_root}"

    samples = _parse_split_file(split_path)
    video_filtered = []
    for s in samples:
        video_path = os.path.join(video_root, f"{s['video_id']}.mp4")
        if os.path.isfile(video_path):
            video_filtered.append(s)
    samples = video_filtered

    max_samples = kwargs.get("max_samples", None)
    seed = kwargs.get("seed", 17)
    if max_samples is not None and max_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, max_samples)

    filtered = []
    missing_audio = 0
    extract_failed = 0
    for s in samples:
        video_path = os.path.join(video_root, f"{s['video_id']}.mp4")
        audio_path = os.path.join(audio_root, _audio_filename(s["video_id"], s["clip_id"]))
        if use_precomputed_audio:
            if not os.path.isfile(audio_path):
                missing_audio += 1
                continue
        else:
            ok = _extract_clip_audio_ffmpeg(video_path, s["start"], s["end"], audio_path)
            if not ok:
                extract_failed += 1
                continue
        s = dict(s)
        s["audio_path"] = audio_path
        filtered.append(s)
    samples = filtered
    assert len(samples) > 0, (
        f"AVE has no valid samples after filtering. "
        f"use_precomputed_audio={use_precomputed_audio}, missing_audio={missing_audio},"
        f" extract_failed={extract_failed}, "
        f"video_root={video_root}, audio_root={audio_root}"
    )

    audio_files = [
        s["audio_path"]
        for s in samples
    ]

    dataset = datasets.Dataset.from_dict(
        {
            "category": [s["category"] for s in samples],
            "video_id": [s["video_id"] for s in samples],
            "clip_id": [s["clip_id"] for s in samples],
            # ✅ HF Audio column expects dicts: {"path":..., "bytes": None}
            "query_audio": [{"path": p, "bytes": None} for p in audio_files],
        }
    )

    dataset = dataset.cast_column("query_audio", Audio())

    all_video_ids = sorted(list(set(dataset["video_id"])))

    kwargs["audio_root"] = audio_root
    kwargs["video_root"] = video_root
    kwargs["frame_root"] = kwargs.get("frame_root", os.path.join(data_path, "frames"))
    kwargs["num_frames"] = kwargs.get("num_frames", 8)
    kwargs["max_frames_saved"] = kwargs.get("max_frames_saved", 100)
    kwargs["image_resolution"] = data_args.image_resolution
    kwargs["model_backbone"] = model_args.model_backbone

    def _prepare(x):
        return data_prepare(x, all_video_ids=all_video_ids, **kwargs)

    dataset = dataset.map(
        _prepare,
        batched=True,
        batch_size=128,
        drop_last_batch=False,
        load_from_cache_file=False,
    )

    dataset = dataset.select_columns(
        ["query_text", "query_image", "query_audio", "cand_text", "cand_video", "dataset_infos"]
    )

    corpus = None
    return dataset, corpus
