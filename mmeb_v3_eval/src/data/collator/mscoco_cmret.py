import os
import sys
import torch

from src.utils.basic_utils import print_rank, print_master
from datasets import load_dataset, Dataset
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    add_metainfo_hook,
    RESOLUTION_MAPPING,
    ImageVideoInstance,
)
from src.model.processor import process_input_text
from src.utils.vision_utils.vision_utils import save_frames, process_video_frames

from typing import Literal


def _enable_omniset_rotary_dtype_patch():
    flag = os.environ.get("OMNISET_ROTARY_DTYPE_PATCH", os.environ.get("MSCOCO_OMNI_ROTARY_DTYPE_PATCH", "0")).lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    try:
        from flash_attn.layers import rotary as layers_rotary
        from flash_attn.ops.triton import rotary as triton_rotary
    except Exception as e:
        print_master(f"OmniSET rotary dtype patch disabled (flash-attn import failed): {e}")
        return

    if getattr(triton_rotary, "_omniset_dtype_patch_applied", False):
        return

    original_apply_rotary = triton_rotary.apply_rotary

    def _apply_rotary_with_dtype_fix(x, cos, sin, *args, **kwargs):
        if isinstance(x, torch.Tensor) and isinstance(cos, torch.Tensor):
            if x.is_floating_point() and x.dtype != cos.dtype:
                x = x.to(dtype=cos.dtype)
        return original_apply_rotary(x, cos, sin, *args, **kwargs)

    triton_rotary.apply_rotary = _apply_rotary_with_dtype_fix
    layers_rotary.apply_rotary = _apply_rotary_with_dtype_fix
    triton_rotary._omniset_dtype_patch_applied = True
    print_master("Enabled OmniSET flash-attn rotary dtype compatibility patch.")


_enable_omniset_rotary_dtype_patch()

# ============== Cross Modality Utilities ==============
MODALITIES = ['T', 'I', 'V', 'A']
MODALITY_NAME_MAPPING = {
    'T': "text",
    'I': 'image',
    'V': 'video',
    'A': 'audio'
}
MODALITY_EXT_MAPPING = {
    'T': 'txt',
    'I': 'jpg',
    'V': 'mp4',
    'A': 'wav'
}
# MODALITY_INST_MAPPING = {
#     'T': "Find the text that best matches the given image and video.",
#     'I': "Find the image that best matches the given text: ",
#     'V': "Find the video that best matches the given text: ",
#     'A': "Find the audio that best matches the given text: "
# }


def coco_filename(id):
    return f'COCO_val2014_{str(id).zfill(12)}'


def coco_filename_with_ext(id, modality='I'):
    ext = MODALITY_EXT_MAPPING[modality]
    return f'{coco_filename(id)}.{ext}'


def coco_id(filename):
    return int(filename.split('_')[-1].split('.')[0])


def get_instruction(input_mod: Literal['T', 'I', 'V', 'A'], query_mod: Literal['T', 'I', 'V', 'A']):
    assert input_mod != query_mod, "Input modality and query modality cannot be the same, or it will be too simple"
    return (
        f"Find the {MODALITY_NAME_MAPPING[query_mod]} that best matches the given"
        f" {MODALITY_NAME_MAPPING[input_mod]}: ")


# this is a strange enough token to separate the text filename and text content,
# since there is no actual text file, use like this f"{filename}.txt{SPECIAL_SEP_TOKEN}{caption}"
SPECIAL_SEP_TOKEN = (
    "|<<<FILENAME|CAPTION>>>"
    "|")


def generate_omnidirectional_dataset(dataset, *args, **kwargs):
    """
    Generate an omnidirectional evaluation dataset. 
    The query always has an image, with different instructions asking for three modalities of targets, e.g. Text, Video, and Audio. The candidate pool for each query instance includes all four modalities. The presence of Image candidates in the candidate pool serves as distractors making the task more challenging. No Image target (Positive) will be queried. 
    Example:
        Query: [Image of a dog] + "Find the Video that best matches this image. " -> 
        Target (Positive): [Video of a dog running]
        Candidate pool: [Video of a dog running, Image of a dog, Audio of a dog barking, Text description of a dog]
    """
    QRY_MODS, INPUT_MODS, CAND_MODS = kwargs['query_mod'], kwargs['input_mod'], kwargs['cand_mod']
    caption_lookup = kwargs.get('caption_lookup', {}) or {}
    caption_lookup = {int(k): v for k, v in caption_lookup.items()}

    img_ids, pos_filenames, qry_instrs = [], [], []
    qry_texts, qry_images, qry_videos, qry_audios = [], [], [], []
    tgt_texts, tgt_images, tgt_videos, tgt_audios = [], [], [], []
    # Generating candidate pools, same for all three query instances based on the same image_id
    for img_id, caption, neg_ids in zip(dataset['image_id'], dataset['qry_text'], dataset['hard_negatives']):
        # always keep T/I/V/A slots; non-selected modalities stay as placeholders
        tgt_lsts = {k: [] for k in MODALITIES}
        candidate_ids_pool = list(set(neg_ids+[img_id]))
        # three nested for loops looks weird but actually saves lots of lines :)
        for tgt_id in candidate_ids_pool:
            tgt_caption = caption_lookup.get(int(tgt_id), caption)
            for qry_mod in CAND_MODS:
                tgt_lsts[qry_mod].append(
                    f"{coco_filename_with_ext(tgt_id, 'T')}{SPECIAL_SEP_TOKEN}{tgt_caption}"
                    if qry_mod == 'T' else coco_filename_with_ext(tgt_id, qry_mod))
                for other_mod in [k for k in MODALITIES if k != qry_mod]:
                    tgt_lsts[other_mod].append("" if other_mod == 'T' else None)

        lengths = {m: len(tgt_lsts[m]) for m in MODALITIES}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Error: Inconsistent candidate pool lengths: {lengths}")

        # we only query the three modalities without image, which is the query modality
        # now generate the query instance (positive sample) for each modality
        for inp_mod in INPUT_MODS:
            for qry_mod in QRY_MODS:
                img_ids.append(img_id)
                # SPECIAL: caption text is not actually a file, but we still name it .txt
                # to distinguish btw the other three modalities in the candidate pool
                pos_filename = coco_filename_with_ext(img_id, qry_mod)
                pos_filenames.append(pos_filename)
                qry_instrs.append(get_instruction(inp_mod, qry_mod))
                queries = {k: ("" if k == 'T' else None) for k in MODALITIES}
                queries[inp_mod] = caption if inp_mod == 'T' else coco_filename_with_ext(img_id, inp_mod)
                qry_texts.append(queries['T'])
                qry_images.append(queries['I'])
                qry_videos.append(queries['V'])
                qry_audios.append(queries['A'])
                tgt_texts.append(tgt_lsts['T'])
                tgt_images.append(tgt_lsts['I'])
                tgt_videos.append(tgt_lsts['V'])
                tgt_audios.append(tgt_lsts['A'])

    return Dataset.from_dict({
        "image_id": img_ids,
        "pos_filename": pos_filenames,
        "qry_instr": qry_instrs,
        "qry_text": qry_texts,
        "qry_image": qry_images,
        "qry_video": qry_videos,
        "qry_audio": qry_audios,
        "tgt_text": tgt_texts,
        "tgt_image": tgt_images,
        "tgt_video": tgt_videos,
        "tgt_audio": tgt_audios
    })


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    num_frames, max_frames_saved = kwargs['num_frames'], kwargs['max_frames_saved']
    image_root, video_root, audio_root, frame_root = kwargs['image_root'], kwargs['video_root'], kwargs[
        'audio_root'], kwargs['frame_root']
    model_backbone = kwargs['model_backbone']

    TGT_INST = "Represent the given text, image, video, or audio."
    query_texts, query_images, query_videos, query_audios = [], [], [], []
    cand_texts, cand_images, cand_videos, cand_audios, dataset_infos = [], [], [], [], []
    for (pos_id, pos_filename, qry_instr,
         qry_txt, qry_img, qry_vid, qry_aud,
         tgt_txts, tgt_imgs, tgt_vids, tgt_auds) in \
        zip(batch_dict['image_id'], batch_dict['pos_filename'], batch_dict['qry_instr'],
            batch_dict['qry_text'], batch_dict['qry_image'], batch_dict['qry_video'], batch_dict['qry_audio'],
            batch_dict['tgt_text'], batch_dict['tgt_image'], batch_dict['tgt_video'], batch_dict['tgt_audio']):

        query_texts.append([f"{qry_instr}{qry_txt}"])
        if qry_img is not None:
            query_images.append([ImageVideoInstance(
                bytes=[None],
                paths=[os.path.join(image_root, qry_img)],
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
            ).to_dict()])
        else:
            query_images.append([None])
        if qry_vid is not None:
            frame_dir = os.path.join(frame_root, qry_vid.split('.')[0])
            save_frames(video_path=os.path.join(video_root, qry_vid),
                        frame_dir=frame_dir,
                        max_frames_saved=max_frames_saved)
            video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
            query_videos.append([ImageVideoInstance(
                bytes=[None] * len(video_frame_paths),
                paths=video_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
            ).to_dict()])
        else:
            query_videos.append([None])
        if qry_aud is not None:
            query_audios.append({"path": os.path.join(audio_root, qry_aud), "bytes": None})
        else:
            query_audios.append(None)

        # processed candidate pool for each sample in a batch
        name_insts, txt_insts, img_insts, vid_insts, aud_insts = [], [], [], [], []
        # compose paths for the files of the candidate pool
        for txt, img, vid, aud in zip(tgt_txts, tgt_imgs, tgt_vids, tgt_auds):
            if txt != "" and SPECIAL_SEP_TOKEN in txt:  # if this is a concrete text candidate (no img, vid, aud input)
                assert img is None and vid is None and aud is None, (
                    "Error: Text candidate should not have other"
                    f" modality candidates. Found txt={txt}, img={img}, vid={vid}, aud={aud}.")
                filename, caption = txt.split(SPECIAL_SEP_TOKEN)
                txt_insts.append(caption)
                name_insts.append(filename)

            if aud is not None:  # if this is audio candidate
                aud_path = os.path.join(audio_root, aud)
                assert os.path.exists(aud_path), f"Audio {aud_path} does not exist."
                aud_insts.append({"path": aud_path, "bytes": None})
                txt_insts.append(f"<|audio_pad|> {TGT_INST} {txt}")  # actually no tgt txt here
                name_insts.append(aud)
            else:
                aud_insts.append(None)

            if img is not None:
                img_path = os.path.join(image_root, img)
                assert os.path.exists(img_path), f"Image {img_path} does not exist."
                img_insts.append(ImageVideoInstance(
                    bytes=[None],
                    paths=[img_path],
                    resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)],
                ).to_dict())
                txt_insts.append(
                    process_input_text(
                        TGT_INST,
                        text=txt,
                        model_backbone=model_backbone,
                        add_image_token=True))
                name_insts.append(img)
            else:
                img_insts.append(None)

            if vid is not None:  # if is video
                vid_path = os.path.join(video_root, vid)
                assert os.path.exists(vid_path), \
                    f"Video {vid_path} does not exist."
                # use the filename without extension as the frame dir name
                frame_dir = os.path.join(frame_root, vid.split('.')[0])
                try:
                    save_frames(video_path=vid_path,
                                frame_dir=frame_dir,
                                max_frames_saved=max_frames_saved)
                    video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
                    vid_insts.append(ImageVideoInstance(
                        bytes=[None] * len(video_frame_paths),
                        paths=video_frame_paths,
                        resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(video_frame_paths),
                    ).to_dict())
                    txt_insts.append(
                        process_input_text(
                            TGT_INST,
                            text=txt,
                            model_backbone=model_backbone,
                            add_video_token=True))
                    name_insts.append(vid)
                # simply skip the cand video if not exist, however, if the pos sample is a video, raise error
                except Exception:
                    print_rank(f"Loading frames for {vid_path} failed.")
                    if vid == pos_filename:
                        raise FileNotFoundError(f"Positive sample video {vid_path} not found or failed to load!")
                    else:
                        print_rank(f"Skipping candidate video {vid_path}.")
                        # vid_insts.append(None)
                        # txt_insts.append(None)
                        # if as expected, no candidate will have more than 1 modality, so we can simply skip,
                        # if we append None here, no inputs will be available
                        continue
            else:
                vid_insts.append(None)

        assert pos_filename in name_insts and len(name_insts) == len(set(name_insts)), \
            f'Error: pos_filename={pos_filename} NOT FOUND in cand_name_instances={name_insts}'
        assert len(name_insts) == len(txt_insts) == len(img_insts) == len(vid_insts) == len(aud_insts), \
            (
                f"Error: Inconsistent candidate instance lengths for pos_filename={pos_filename}: {len(name_insts)}"
                f" names, {len(txt_insts)} txts, {len(img_insts)} imgs, {len(vid_insts)} vids, {len(aud_insts)}"
                " audios.")
        cand_images.append(img_insts)
        cand_videos.append(vid_insts)
        cand_texts.append(txt_insts)
        cand_audios.append(aud_insts)
        dataset_infos.append({
            "cand_names": name_insts,
            "label_name": pos_filename,
            "query": {
                "instruction": qry_instr,
                "text": qry_txt,
                "image": qry_img,
                "video": qry_vid,
                "audio": qry_aud,
            },
        })

    return {
        "query_text": query_texts, "query_image": query_images, "query_video": query_videos,
        "query_audio": query_audios,
        "cand_text": cand_texts, "cand_image": cand_images, "cand_video": cand_videos, "cand_audio": cand_audios,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "omniset"
LEGACY_DATASET_PARSER_NAME = "mscoco_cmret"
DATASET_HF_PATH = "MINGYISU/t2iv"  # legacy remote fallback
LOCAL_CMRET_JSONL = "omniset.jsonl"
LOCAL_CATALOG_JSONL = "catalog.jsonl"


def _load_caption_lookup(local_catalog_path):
    if not local_catalog_path or not os.path.exists(local_catalog_path):
        return {}

    catalog = load_dataset("json", data_files={"catalog": local_catalog_path}, split="catalog")
    id_to_caption = {}
    for row in catalog:
        image_id = int(row["image_id"])
        captions = row.get("captions", [])
        id_to_caption[image_id] = captions[0] if captions else ""
    return id_to_caption


def _resolve_local_omniset_files(**kwargs):
    data_path = kwargs.get("data_path")
    candidate_roots = []

    if data_path:
        data_path = os.path.abspath(data_path)
        if os.path.isfile(data_path):
            catalog_file = os.path.join(os.path.dirname(data_path), LOCAL_CATALOG_JSONL)
            return data_path, catalog_file if os.path.exists(catalog_file) else None
        if os.path.isdir(data_path):
            candidate_roots.append(data_path)

    for key in ["image_root", "video_root", "audio_root", "frame_root"]:
        root = kwargs.get(key)
        if root:
            candidate_roots.append(os.path.dirname(os.path.abspath(root)))

    visited = set()
    for root in candidate_roots:
        root = os.path.abspath(root)
        if root in visited:
            continue
        visited.add(root)

        cmret_file = os.path.join(root, LOCAL_CMRET_JSONL)
        if not os.path.exists(cmret_file):
            alt_file = os.path.join(root, "test.jsonl")
            if os.path.exists(alt_file):
                cmret_file = alt_file
            else:
                continue

        catalog_file = os.path.join(root, LOCAL_CATALOG_JSONL)
        return cmret_file, catalog_file if os.path.exists(catalog_file) else None

    return None, None


def _load_local_cmret_dataset(local_cmret_path, local_catalog_path):
    dataset = load_dataset("json", data_files={"test": local_cmret_path}, split="test")

    if "qry_text" not in dataset.column_names:
        id_to_caption = _load_caption_lookup(local_catalog_path)
        if not id_to_caption:
            raise FileNotFoundError(
                f"`qry_text` is missing in {local_cmret_path}, and caption file {local_catalog_path} was not found."
            )

        def attach_qry_text(example):
            image_id = int(example["image_id"])
            caption = id_to_caption.get(image_id, "")
            if not caption:
                caption = " ".join(example.get("objects", []))
            return {"qry_text": caption}

        dataset = dataset.map(attach_qry_text, desc="Attach qry_text from local catalog")

    return dataset


@AutoEvalPairDataset.register(LEGACY_DATASET_PARSER_NAME)
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_omniset_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]

    local_cmret_path, local_catalog_path = _resolve_local_omniset_files(**kwargs)
    caption_lookup = {}
    if local_cmret_path is not None:
        print_master(f"Loading local OmniSET file: {local_cmret_path}")
        if local_catalog_path:
            print_master(f"Using local catalog file: {local_catalog_path}")
        dataset = _load_local_cmret_dataset(local_cmret_path, local_catalog_path)
        caption_lookup = _load_caption_lookup(local_catalog_path)
    else:
        print_master(f"Loading remote dataset: {DATASET_HF_PATH}")
        dataset = load_dataset(DATASET_HF_PATH, split="test")

    if not caption_lookup and "image_id" in dataset.column_names and "qry_text" in dataset.column_names:
        caption_lookup = {
            int(i): c
            for i, c in zip(dataset["image_id"], dataset["qry_text"])
            if c is not None
        }

    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print_master(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['caption_lookup'] = caption_lookup

    # print_master(f"Start preparing dataset {dataset_name} with model backbone {model_args.model_backbone}
    # and image resolution {data_args.image_resolution}. Total number of samples: {len(dataset)}.")
    dataset = generate_omnidirectional_dataset(dataset, *args, **kwargs)
    # TODO: DEBUGGING PURPOSE
    os.makedirs("debug_cm_input", exist_ok=True)
    dataset.to_json(f"debug_cm_input/{dataset_name}_debug.json")
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text",
                                      "query_image",
                                      "query_video",
                                      "query_audio",
                                      "cand_text",
                                      "cand_image",
                                      "cand_video",
                                      "cand_audio",
                                      "dataset_infos"])

    return dataset, None


# Backward-compatible import name for older code paths.
load_mscoco_cmret_dataset = load_omniset_dataset
