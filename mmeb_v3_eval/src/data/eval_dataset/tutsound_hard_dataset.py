"""
TUTSound (hard) audio event grounding evaluation dataset
- text labels -> audio segment retrieval (strict segment-level matching)
"""

import os
from typing import Any, Dict, List, Tuple

import datasets
import soundfile as sf
import types

from src.constant.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.audio_instruction_utils import build_query_text
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset


def _read_evaluate_file(eval_path: str) -> List[Tuple[str, str, float, float, str]]:
    rows = []
    with open(eval_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = line.split()
            if len(parts) < 5:
                continue
            fp = parts[0].strip()
            scene = parts[1].strip()
            onset = float(parts[2])
            offset = float(parts[3])
            label = parts[4].strip()
            rows.append((fp, scene, onset, offset, label))
    return rows


def _segment_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return 0.0 if union <= 0 else inter / union


def _seg_id(fp: str, start: float, end: float) -> str:
    return f"{fp}|{start:.3f}-{end:.3f}"


def _normalize_segment(start: float, end: float, audio_dur: float, min_dur: float = 0.05) -> Tuple[float, float]:
    s = max(0.0, float(start))
    e = float(end)
    if e <= s:
        e = s + float(min_dur)
    if e > audio_dur:
        e = audio_dur
    if e <= s:
        s = max(0.0, audio_dur - float(min_dur))
        e = audio_dur
    if e <= s:
        e = s + 1e-3
    return s, e


def build_tutsound_hard_audio_dataset(path_info: Tuple[str, str, str], **kwargs):
    dataset_path, _, _ = path_info
    base_dir = os.path.join(dataset_path, "TUT-sound-events-2017-development")
    neg_win_len = float(kwargs.get("neg_win_len", 1.0))
    neg_stride = float(kwargs.get("neg_stride", 0.5))
    iou_thresh = float(kwargs.get("neg_iou_thresh", 0.1))
    neg_max_per_query = int(kwargs.get("neg_max_per_query", 50))

    all_rows = []
    for fold in ["1", "2", "3", "4"]:
        eval_file_name = f"street_fold{fold}_evaluate.txt"
        eval_file = os.path.join(base_dir, "evaluation_setup", eval_file_name)
        if os.path.isfile(eval_file):
            fold_rows = _read_evaluate_file(eval_file)
            all_rows.extend(fold_rows)
            print(f"TUTSound(hard): loaded fold{fold} with {len(fold_rows)} events")
        else:
            print(f"Warning: TUTSound(hard) evaluate file not found: {eval_file}")

    if not all_rows:
        raise FileNotFoundError("No TUTSound(hard) evaluate files found for any fold")

    gt_by_fp: Dict[str, Dict[str, Any]] = {}
    for fp, scene, onset, offset, label in all_rows:
        if fp not in gt_by_fp:
            gt_by_fp[fp] = {"scene": scene, "events": []}
        gt_by_fp[fp]["events"].append({"label": label, "onset": onset, "offset": offset})

    query_audio = []
    query_text = []
    query_image = []
    cand_text = []
    cand_image = []
    cand_audio = []
    dataset_infos = []

    wav_list = sorted(gt_by_fp.keys())
    cand_text_item = build_query_text("TUTSound")
    assert (
        isinstance(cand_text_item, list)
        and len(cand_text_item) == 1
        and isinstance(cand_text_item[0], str)
        and cand_text_item[0].strip()
    )

    for fp in wav_list:
        abs_path = os.path.normpath(os.path.join(base_dir, fp))
        info = gt_by_fp[fp]
        gt_events = info["events"]

        try:
            _si = sf.info(abs_path)
            audio_info = types.SimpleNamespace(num_frames=_si.frames, sample_rate=_si.samplerate)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio info for {abs_path}: {e}") from e

        audio_dur = float(audio_info.num_frames) / float(audio_info.sample_rate)
        normalized_events = []
        for ev in gt_events:
            seg_s, seg_e = _normalize_segment(ev["onset"], ev["offset"], audio_dur=audio_dur)
            normalized_events.append({"label": ev["label"], "onset": seg_s, "offset": seg_e})
        gt_events = normalized_events
        gt_segments = [(float(ev["onset"]), float(ev["offset"])) for ev in gt_events]

        neg_segments = []
        if neg_win_len > 0 and neg_stride > 0 and audio_dur >= neg_win_len:
            t = 0.0
            while t + neg_win_len <= audio_dur:
                seg_start = t
                seg_end = t + neg_win_len
                max_iou = 0.0
                for gs, ge in gt_segments:
                    max_iou = max(max_iou, _segment_iou(seg_start, seg_end, gs, ge))
                    if max_iou >= iou_thresh:
                        break
                if max_iou < iou_thresh:
                    neg_segments.append((seg_start, seg_end))
                t += neg_stride

        if neg_max_per_query > 0 and len(neg_segments) > neg_max_per_query:
            neg_segments = neg_segments[:neg_max_per_query]

        cand_segments = gt_segments + neg_segments
        cand_names = [_seg_id(fp, s, e) for s, e in cand_segments]
        cand_audio_row = [{"path": abs_path, "start": s, "end": e, "bytes": None} for s, e in cand_segments]
        cand_text_row = [cand_text_item[0]] * len(cand_segments)
        cand_image_row = [None] * len(cand_segments)

        for ev in gt_events:
            query_audio.append({"path": abs_path, "bytes": None})
            query_text.append(build_query_text("TUTSound", ev["label"]))
            query_image.append([None])
            cand_text.append(cand_text_row)
            cand_image.append(cand_image_row)
            cand_audio.append(cand_audio_row)

            primary_seg_id = _seg_id(fp, float(ev["onset"]), float(ev["offset"]))
            dataset_infos.append(
                {
                    "file_path": fp,
                    "scene": info["scene"],
                    "gt_events": gt_events,
                    "query_event": ev,
                    "cand_names": cand_names,
                    # Strict criterion: the exact target segment must be retrieved.
                    "label_name": [primary_seg_id],
                    "primary_label_name": primary_seg_id,
                }
            )

    dataset = datasets.Dataset.from_dict(
        {
            "query_text": query_text,
            "query_image": query_image,
            "query_audio": query_audio,
            "cand_text": cand_text,
            "cand_image": cand_image,
            "cand_audio": cand_audio,
            "dataset_infos": dataset_infos,
        }
    )
    return dataset, None


DATASET_PARSER_NAME = "tutsound_audio_gnd_hard"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_tutsound_hard_audio_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs.get("dataset_name", "TUTSound(hard)")
    path_info = EVAL_DATASET_LOCAL_PATH.get(dataset_name, EVAL_DATASET_HF_PATH.get(dataset_name))
    if path_info is None:
        # Keep compatibility: allow hard task name while sharing TUTSound raw files.
        path_info = EVAL_DATASET_LOCAL_PATH.get("TUTSound", EVAL_DATASET_HF_PATH.get("TUTSound"))
    if path_info is None:
        raise ValueError(f"Unknown dataset_name={dataset_name}, and fallback 'TUTSound' not found")

    return build_tutsound_hard_audio_dataset(path_info, **kwargs)
