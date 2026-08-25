import datetime
import csv
import logging
import json
import random
import time

import numpy as np
import os
import pickle
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig
from datasets import Dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from src.utils.eval_utils.metrics import RankingMetrics
from src.model.model import MMEBModel
from src.model.processor import (
    get_backbone_name,
    load_processor,
    COLPALI,
    QWEN3_VL,
    QWEN2_5_OMNI,
    NVOMNIEMBED,
    WAVE,
    E5_OMNI,
    JINA_OMNI,
    LCO_OMNI,
    WEMM_EMBEDDING,
)
from src.utils.basic_utils import batch_to_device, print_rank, print_master

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)


def format_duration(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def materialize_gae_scoring_view(qry_embeds, gt_infos, cand_embed_dict, dataset_name):
    """Materialize the scoring view for GAE GUI-agent retrieval tasks.

    The GAE query sets contain a few duplicated query rows, and some query IDs
    also appear in the corpus as candidates. Follow the GAE scoring protocol:
    deduplicate query rows by query ID (keep the first row's embedding and the
    last row's metadata), substitute the corpus embedding for every query ID
    that is also a corpus candidate, and exclude those shared IDs from the
    candidate pool so a query can never match itself.
    """
    qry_ids = [
        str(info.get("qry_id") or f"{dataset_name}_qry_{index}")
        for index, info in enumerate(gt_infos)
    ]
    first_row, last_info, ordered_ids = {}, {}, []
    for index, (qry_id, info) in enumerate(zip(qry_ids, gt_infos)):
        if qry_id not in first_row:
            first_row[qry_id] = index
            ordered_ids.append(qry_id)
        last_info[qry_id] = info
    materialized_embeds = np.array(
        qry_embeds[[first_row[qry_id] for qry_id in ordered_ids]], copy=True
    )
    materialized_infos = []
    for qry_id in ordered_ids:
        info = dict(last_info[qry_id])
        info["qry_id"] = qry_id
        materialized_infos.append(info)

    materialized_cands = cand_embed_dict
    shared_ids = [qry_id for qry_id in ordered_ids if qry_id in cand_embed_dict]
    if shared_ids:
        query_rows = {qry_id: index for index, qry_id in enumerate(ordered_ids)}
        for qry_id in shared_ids:
            materialized_embeds[query_rows[qry_id]] = cand_embed_dict[qry_id]
        shared_set = set(shared_ids)
        materialized_cands = {
            cand_id: embed
            for cand_id, embed in cand_embed_dict.items()
            if cand_id not in shared_set
        }
    print_master(
        f"[{dataset_name}] GAE scoring view: source_queries={len(qry_ids)} "
        f"materialized_queries={len(ordered_ids)} "
        f"duplicate_query_rows={len(qry_ids) - len(ordered_ids)} "
        f"shared_query_cand_ids={len(shared_ids)}"
    )
    return materialized_embeds, materialized_infos, materialized_cands


def append_dataset_timing_row(csv_path: str, row: dict):
    if not csv_path:
        return
    parent_dir = os.path.dirname(csv_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    header = [
        "model_name",
        "model_backbone",
        "modality",
        "dataset_name",
        "start_time",
        "end_time",
        "duration_seconds",
        "duration_hms",
        "load_seconds",
        "query_seconds",
        "cand_seconds",
        "score_seconds",
        "do_query",
        "do_cand",
        "status",
        "error",
    ]
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})


def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


def encode_embeddings(
    model: MMEBModel,
    loader: DataLoader,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    full_dataset: Dataset,
    encode_side: str,
    description: str = "Encoding"
) -> tuple[np.ndarray, list]:
    """
    Encodes embeddings for a given dataset using the model, handling both standard and
    late-interaction models in a DDP-safe manner.
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Check if the model is a late-interaction type
    is_late_interaction = (model_args.model_backbone == COLPALI)

    local_embeds = []
    local_gt_infos = []
    local_max_len = 0

    model.eval()
    with torch.no_grad():
        for inputs, dataset_info in tqdm(loader, desc=f"{description} (rank {local_rank})", disable=local_rank > 0):
            inputs = batch_to_device(inputs, training_args.device)
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # Determine if encoding query or target based on available keys
                if encode_side == "qry":
                    out_key = "qry_reps"
                    gt_infos = dataset_info
                else:
                    out_key = "tgt_reps"
                    gt_infos = [info["cand_name"] for info in dataset_info]

                if model_args.model_backbone in {QWEN3_VL, QWEN2_5_OMNI,
                                                 NVOMNIEMBED, WAVE, E5_OMNI, JINA_OMNI, LCO_OMNI, WEMM_EMBEDDING}:
                    # Bucketed micro-batching by grid_thw to keep visual seq_len aligned.
                    batch_size = inputs["input_ids"].shape[0]
                    device = inputs["input_ids"].device

                    pixel_values = inputs.get("pixel_values", None)
                    image_grid_thw = inputs.get("image_grid_thw", None)
                    pixel_values_videos = inputs.get("pixel_values_videos", None)
                    video_grid_thw = inputs.get("video_grid_thw", None)

                    def _build_visual_spans(patch_tensor, grid_tensor):
                        """
                        Build per-sample [start, end) spans for flattened visual patches.
                        patch_tensor is expected to be [sum_i(t_i*h_i*w_i), hidden_dim].
                        grid_tensor is expected to be [batch_size, 3].
                        """
                        if not isinstance(patch_tensor, torch.Tensor):
                            return None
                        if not isinstance(grid_tensor, torch.Tensor):
                            return None
                        if grid_tensor.dim() != 2 or grid_tensor.shape[1] != 3:
                            return None
                        if grid_tensor.shape[0] != batch_size:
                            return None

                        counts = (grid_tensor[:, 0].long() * grid_tensor[:, 1].long()
                                  * grid_tensor[:, 2].long()).tolist()
                        if any(c < 0 for c in counts):
                            return None

                        total = int(sum(counts))
                        if patch_tensor.shape[0] != total:
                            return None

                        spans = []
                        start = 0
                        for c in counts:
                            end = start + int(c)
                            spans.append((start, end))
                            start = end
                        return spans

                    image_spans = _build_visual_spans(pixel_values, image_grid_thw)
                    video_spans = _build_visual_spans(pixel_values_videos, video_grid_thw)

                    def _get_batch_aligned_item(value, idx):
                        if value is None:
                            return None
                        if isinstance(value, list):
                            if len(value) == batch_size:
                                return value[idx]
                            return None
                        if isinstance(value, torch.Tensor):
                            if value.dim() > 0 and value.shape[0] == batch_size:
                                return value[idx]
                            return None
                        return None

                    def _grid_to_key(grid_item):
                        if grid_item is None:
                            return None
                        if isinstance(grid_item, torch.Tensor):
                            return tuple(grid_item.detach().cpu().reshape(-1).tolist())
                        if isinstance(grid_item, (list, tuple)):
                            # list of int
                            try:
                                flat = []
                                for x in grid_item:
                                    if isinstance(x, torch.Tensor):
                                        flat.extend(x.detach().cpu().reshape(-1).tolist())
                                    elif isinstance(x, (list, tuple)):
                                        flat.extend(list(x))
                                    else:
                                        flat.append(x)
                                return tuple(flat)
                            except Exception:
                                return str(grid_item)
                        return str(grid_item)

                    def _bucket_key(i):
                        # For flattened visual patch tensors, never probe modality by pixel_values[_videos][i],
                        # because dim0 is token length rather than batch size.
                        img_gt = _get_batch_aligned_item(image_grid_thw, i)
                        vid_gt = _get_batch_aligned_item(video_grid_thw, i)
                        has_img = img_gt is not None
                        has_vid = vid_gt is not None

                        if has_vid:
                            return ("vid", _grid_to_key(vid_gt))
                        if has_img:
                            return ("img", _grid_to_key(img_gt))
                        return ("none",)

                    buckets = {}
                    for i in range(batch_size):
                        k = _bucket_key(i)
                        buckets.setdefault(k, []).append(i)

                    reps_out = torch.empty((batch_size, model.rep_dim), device=device, dtype=torch.float32)

                    def _slice_flat_visual(v, idxs, spans):
                        if spans is None:
                            return None
                        pieces = []
                        for j in idxs:
                            s, e = spans[j]
                            pieces.append(v[s:e])
                        if len(pieces) == 0:
                            return v.new_empty((0, *v.shape[1:]))
                        return torch.cat(pieces, dim=0)

                    def _slice_value(k, v, idxs):
                        if v is None:
                            return None
                        if k == "pixel_values" and isinstance(v, torch.Tensor):
                            sliced = _slice_flat_visual(v, idxs, image_spans)
                            if sliced is not None:
                                return sliced
                        if k == "pixel_values_videos" and isinstance(v, torch.Tensor):
                            sliced = _slice_flat_visual(v, idxs, video_spans)
                            if sliced is not None:
                                return sliced
                        if isinstance(v, torch.Tensor):
                            if v.dim() > 0 and v.shape[0] == batch_size:
                                return v[idxs]
                            return v
                        if isinstance(v, list):
                            if len(v) == batch_size:
                                return [v[j] for j in idxs]
                            return v
                        return None

                    def _slice_inputs(inputs_dict, idxs):
                        sub = {}
                        for kk, vv in inputs_dict.items():
                            sub[kk] = _slice_value(kk, vv, idxs)
                        return sub

                    for _, idxs in buckets.items():
                        if len(idxs) == 0:
                            continue
                        sub_inputs = _slice_inputs(inputs, idxs)

                        if encode_side == "qry":
                            output = model(qry=sub_inputs)
                        else:
                            output = model(tgt=sub_inputs)

                        sub_reps = output[out_key].detach()
                        # sub_reps shape: [len(idxs), D]
                        reps_out[idxs] = sub_reps.to(reps_out.dtype)

                    reps = reps_out
                    local_gt_infos.extend(gt_infos)

                else:
                    if encode_side == "qry":
                        output = model(qry=inputs)
                    else:
                        output = model(tgt=inputs)
                    reps = output[out_key].detach()
                    local_gt_infos.extend(gt_infos)

            if is_late_interaction and reps.dim() == 3:
                local_max_len = max(local_max_len, reps.shape[1])

            local_embeds.append(reps)

    if not local_embeds:
        # Handle cases where a rank gets no data
        return np.array([]), []

    # === DDP Synchronization and Padding for Late-Interaction Models ===
    if is_late_interaction:
        if dist.is_initialized():
            # 1. Find the global maximum sequence length across all ranks
            local_max_len_tensor = torch.tensor(local_max_len, device=training_args.device)
            dist.all_reduce(local_max_len_tensor, op=dist.ReduceOp.MAX)
            global_max_len = local_max_len_tensor.item()
        else:
            global_max_len = local_max_len

        # 2. Pad all local embeddings to the global max length
        padded_embeds = []
        for reps_batch in local_embeds:
            if reps_batch.dim() == 3:
                B, L, H = reps_batch.shape
                padding_size = global_max_len - L
                padded_batch = F.pad(reps_batch, (0, 0, 0, padding_size), "constant", 0)
                padded_embeds.append(padded_batch)
            else:  # Should not happen if model is consistently late-interaction
                padded_embeds.append(reps_batch)

        embeds_tensor = torch.cat(padded_embeds, dim=0).contiguous()
    else:  # Standard dense models
        embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()

    # === Gather embeddings and keys from all ranks ===
    if dist.is_initialized() and full_dataset.num_rows >= world_size:
        print_master(f"Gathering {encode_side} embeddings across all ranks...")

        # all_gather_into_tensor requires each rank input to have the same shape.
        # Pad along the batch dimension to the max local count, then trim by counts.
        embeds_tensor = embeds_tensor.to(training_args.device)
        local_count = torch.tensor([embeds_tensor.shape[0]], device=training_args.device, dtype=torch.int64)
        all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        counts = [int(c.item()) for c in all_counts]
        max_count = max(counts) if counts else int(local_count.item())

        if embeds_tensor.shape[0] < max_count:
            pad_shape = (max_count,) + tuple(embeds_tensor.shape[1:])
            padded = torch.zeros(pad_shape, dtype=embeds_tensor.dtype, device=training_args.device)
            if embeds_tensor.shape[0] > 0:
                padded[: embeds_tensor.shape[0]] = embeds_tensor
            embeds_tensor = padded

        output_shape = (world_size * max_count,) + tuple(embeds_tensor.shape[1:])
        gathered_embeds_tensor = torch.empty(output_shape, dtype=embeds_tensor.dtype, device=training_args.device)
        dist.all_gather_into_tensor(gathered_embeds_tensor, embeds_tensor)

        # Trim the padding using per-rank counts, preserving rank order
        if sum(counts) > 0:
            offset = 0
            slices = []
            for c in counts:
                if c > 0:
                    slices.append(gathered_embeds_tensor[offset: offset + c])
                offset += max_count
            final_embeddings = torch.cat(slices, dim=0).cpu().float().numpy() if slices else np.array([])
        else:
            final_embeddings = np.array([])
        # Gather metadata, for which all_gather_object is appropriate
        gathered_gt_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_gt_infos, local_gt_infos)
        all_gt_infos = [key for rank_keys in gathered_gt_infos for key in rank_keys]
    else:
        all_gt_infos = local_gt_infos
        final_embeddings = embeds_tensor.cpu().float().numpy()

    return final_embeddings, all_gt_infos


def main():
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        # Configurable collective timeout: when a rank is stuck (e.g. bad I/O),
        # a shorter timeout fails the run fast so an outer retry loop can resume
        # instead of idling for an hour. Long scoring phases on rank 0 also wait
        # inside collectives on other ranks, so do not set this too low.
        timeout_minutes = int(os.environ.get("MMEB_DIST_TIMEOUT_MINUTES", "60"))
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=timeout_minutes))
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # DEBUG PRINTS for Distributed Setup
    print_master("Distributed init debug info:")
    print_master(f"RANK: {os.environ.get('RANK')}")
    print_master(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print_master(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print_master(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print_master(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    if dist.is_initialized():
        print_rank(f"dist.get_rank(): {dist.get_rank()}")
        print_rank(f"dist.get_world_size(): {dist.get_world_size()}")

    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            local_rank_arg = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(local_rank_arg)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    if torch.cuda.is_available():
        training_args.device = torch.device("cuda")
    else:
        training_args.device = torch.device("cpu")

    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        training_args.device = torch.device(f"cuda:{local_rank}")

    # --- Model Loading ---
    if not getattr(model_args, "model_backbone", None):
        hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_args.model_backbone)
    print_master(f'Model Backbone: {model_args.model_backbone}')
    print_master(f'Using device: {training_args.device}')
    # --- DDP-Safe Model Loading ---
    # Step 1: Only the master process (rank 0) downloads the model.
    if rank == 0:
        processor = load_processor(model_args, data_args)
        model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
        print_master(f"[rank=0] Loading the model from Huggingface: {model_args.model_name}...")
    # Step 2: All processes wait here. The non-master processes will pause
    # until the master process (rank 0) finishes downloading and exits this barrier.
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
    # Step 3: Now that the model is cached, the non-master processes load it from the local cache.
    if rank != 0:
        print_rank("Loading the model from cache...")
        processor = load_processor(model_args, data_args)
        time.sleep(random.randint(2 * rank, 3 * rank))
        model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model.eval()  # pylint: disable=used-before-assignment  # assigned in the Step 2 branch above
    model = model.to(training_args.device, dtype=torch.bfloat16)
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)
    eval_modality = os.environ.get("EVAL_MODALITY", "")
    dataset_timing_log = os.environ.get("EVAL_DATASET_TIMING_LOG", "")

    # --- Main Evaluation Loop ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        dataset_start_ts = time.time()
        dataset_start_human = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        load_seconds = 0.0
        query_seconds = 0.0
        cand_seconds = 0.0
        score_seconds = 0.0
        dataset_status = "success"
        dataset_error = ""
        do_query = False
        do_cand = False

        try:
            if dist.is_initialized():
                dist.barrier(device_ids=[local_rank])
            print_master(f"--- Evaluating {dataset_name} ---")

            current_batch_size = training_args.per_device_eval_batch_size
            if dataset_name in ["Charades-STA", "QVHighlight", "MomentSeeker", "YouCook2", "Video-MME"]:
                current_batch_size = min(current_batch_size, 4)
                print_master(f"Reduced batch size to {current_batch_size} for {dataset_name}")

            query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
            cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
            dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")
            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")

            # Resume short-circuit: an existing score file marks the dataset as
            # fully done, so skip it without re-encoding. With per-node
            # (non-shared) output dirs, rank 0 decides and broadcasts.
            if dist.is_initialized():
                done_flag = torch.zeros(1, dtype=torch.int64, device=f"cuda:{local_rank}")
                if rank == 0:
                    done_flag[0] = int(os.path.exists(score_path))
                dist.broadcast(done_flag, src=0)
                score_done = bool(done_flag[0].item())
            else:
                score_done = os.path.exists(score_path)
            if score_done:
                if rank == 0:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    print_master(f"Score of {dataset_name} (loaded from previous run): {score_path}")
                    formatted = {k: f"{v:.4f}" for k, v in score_dict.items() if isinstance(v, (int, float))}
                    print_master(formatted)
                continue

            # Checkpoint files are written by rank 0 only. With per-node (non-shared)
            # output dirs, the resume decision must be made on rank 0 and broadcast,
            # otherwise ranks diverge (some encode, some skip) and collectives deadlock.
            if dist.is_initialized():
                skip_flags = torch.zeros(2, dtype=torch.int64, device=f"cuda:{local_rank}")
                if rank == 0:
                    skip_flags[0] = int(not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path))
                    skip_flags[1] = int(not os.path.exists(cand_embed_path))
                dist.broadcast(skip_flags, src=0)
                do_query, do_cand = bool(skip_flags[0].item()), bool(skip_flags[1].item())
            else:
                do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
                do_cand = not os.path.exists(cand_embed_path)

            load_start_ts = time.time()
            if do_query or do_cand:
                if data_args.data_basedir is not None:
                    # Construct full paths for data files if --data_basedir is provided
                    for key in [
                        "image_root",
                        "video_root",
                        "frame_root",
                        "clip_root",
                        "audio_root",
                        "data_path",
                        "query_file",
                        "candidate_file",
                        "qrels_file",
                    ]:
                        if data_args.data_basedir and task_config.get(key):
                            task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

                try:
                    full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(
                        model_args=model_args, data_args=data_args, **task_config)
                    full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
                    eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
                    # Pad datasets to be divisible by world_size before splitting
                    if dist.is_initialized():
                        padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                        padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                        eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=rank, world_size=world_size)
                        eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=rank, world_size=world_size)
                    else:
                        padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
                except Exception as e:
                    print_master(f"Failed to load dataset {dataset_name}, skipping {dataset_name}")
                    import traceback
                    traceback.print_exc()
                    print_master(e)
                    raise
            load_seconds = time.time() - load_start_ts

            # --- 1. Compute Query Embeddings ---
            if do_query:
                query_start_ts = time.time()
                print_master("Encoding queries...")
                eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
                eval_qry_loader = DataLoader(
                    eval_qry_dataset,  # pylint: disable=possibly-used-before-assignment
                    batch_size=current_batch_size,
                    collate_fn=eval_qry_collator,
                    num_workers=training_args.dataloader_num_workers,
                    timeout=int(
                        os.environ.get(
                            "MMEB_DATALOADER_TIMEOUT",
                            "600")) if training_args.dataloader_num_workers > 0 else 0)
                query_embeds, gt_infos = encode_embeddings(
                    model, eval_qry_loader, training_args, model_args, padded_qry_dataset,
                    encode_side="qry", description=f"Queries for {dataset_name}")
                query_embeds = query_embeds[:len(full_eval_qry_dataset)]  # world_size>1, trim the padded data points
                gt_infos = gt_infos[:len(full_eval_qry_dataset)]
                if rank == 0:
                    os.makedirs(os.path.dirname(query_embed_path), exist_ok=True)
                    os.makedirs(os.path.dirname(dataset_info_path), exist_ok=True)
                    with open(query_embed_path, 'wb') as f:
                        pickle.dump(query_embeds, f)
                    with open(dataset_info_path, 'w') as f:
                        for info in gt_infos:
                            f.write(json.dumps(info) + '\n')
                    print_master(f"Saved query embeddings to {query_embed_path}")
                if dist.is_initialized():
                    dist.barrier(device_ids=[local_rank])
                query_seconds = time.time() - query_start_ts

            # --- 2. Compute Candidate Embeddings ---
            if do_cand:
                cand_start_ts = time.time()
                print_master("Encoding candidates...")
                eval_cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "cand")
                eval_cand_loader = DataLoader(
                    eval_cand_dataset,  # pylint: disable=possibly-used-before-assignment
                    batch_size=current_batch_size,
                    collate_fn=eval_cand_collator,
                    num_workers=training_args.dataloader_num_workers,
                    timeout=int(
                        os.environ.get(
                            "MMEB_DATALOADER_TIMEOUT",
                            "600")) if training_args.dataloader_num_workers > 0 else 0)

                cand_embeds, all_cand_ids = encode_embeddings(
                    model, eval_cand_loader, training_args, model_args, padded_cand_dataset,
                    encode_side="cand", description=f"Candidates for {dataset_name}")
                cand_embeds = cand_embeds[:len(full_eval_cand_dataset)]  # world_size>1, trim the padded data points
                all_cand_ids = all_cand_ids[:len(full_eval_cand_dataset)]

                if rank == 0:
                    cand_embed_dict = {cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)}
                    os.makedirs(os.path.dirname(cand_embed_path), exist_ok=True)
                    with open(cand_embed_path, 'wb') as f:
                        pickle.dump(cand_embed_dict, f)
                    print_master(f"Saved candidate embeddings to {cand_embed_path}")
                cand_seconds = time.time() - cand_start_ts

            if dist.is_initialized():
                dist.barrier(device_ids=[local_rank])

            # --- 3. Compute Scores (on master rank only) ---
            if rank == 0:
                score_start_ts = time.time()
                score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
                score_loaded_from_cache = False
                if os.path.exists(score_path):
                    try:
                        with open(score_path, "r") as f:
                            score_dict = json.load(f)
                        print_master(f"Score of {dataset_name} (loaded from previous run): {score_path}")
                        formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
                        print_master(formatted)
                        score_loaded_from_cache = True
                    except Exception:
                        print_master(f"Failed to load score for {dataset_name}, will recompute {dataset_name}")

                if not score_loaded_from_cache:
                    with open(query_embed_path, 'rb') as f:
                        qry_embeds = pickle.load(f)
                    with open(cand_embed_path, 'rb') as f:
                        cand_embed_dict = pickle.load(f)
                    gt_infos = [json.loads(line) for line in open(dataset_info_path)]
                    pred_dicts = []

                    eval_type = str(task_config.get("eval_type", "global")).strip().lower()
                    rank_against_all_candidates = eval_type == "global"
                    if dataset_name.startswith("GAE-") and rank_against_all_candidates:
                        qry_embeds, gt_infos, cand_embed_dict = materialize_gae_scoring_view(
                            qry_embeds, gt_infos, cand_embed_dict, dataset_name
                        )
                    if not rank_against_all_candidates:
                        # A query's retrieval pool comes from "pool_names" when the
                        # dataset provides explicit per-query candidate lists
                        # (e.g. LMEB memory tasks), else from "cand_names".
                        def _local_pool(info):
                            pool = info.get("pool_names", None)
                            if isinstance(pool, list) and len(pool) > 0:
                                return pool
                            return info.get("cand_names", None)

                        missing_local_cands = any(
                            not isinstance(_local_pool(info), list) or len(_local_pool(info) or []) == 0
                            for info in gt_infos
                        )
                        if missing_local_cands:
                            print_master(
                                f"[{dataset_name}] eval_type='{eval_type}' but some queries have empty candidate"
                                " pools; "
                                f"fallback to global ranking."
                            )
                            rank_against_all_candidates = True
                    if rank_against_all_candidates:
                        cand_keys = list(cand_embed_dict.keys())
                        cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])

                        # Compute a score matrix with shape [N_q, N_c].
                        if qry_embeds.ndim == 3:  # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                            qry_embed = torch.from_numpy(qry_embeds)
                            cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                            scores = processor.score(
                                qry_embed, cand_embeds, batch_size=64)  # use ColPali score function
                            score_matrix = (
                                scores.detach().cpu().numpy()
                                if isinstance(scores, torch.Tensor) else np.array(scores))
                        else:  # Dense
                            score_matrix = np.dot(qry_embeds, cand_embeds.T)

                        ranked_candids = np.argsort(-score_matrix, axis=1)
                        for qid, (ranked_candid, gt_info) in tqdm(enumerate(zip(ranked_candids, gt_infos)),
                                                                  desc=f"Calculating scores for {dataset_name}"):
                            ranked_idx = (
                                ranked_candid.tolist()
                                if isinstance(ranked_candid, np.ndarray) else list(ranked_candid))
                            rel_docids = (
                                gt_info["label_name"]
                                if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]])
                            rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None
                            assert rel_scores is None or len(rel_docids) == len(rel_scores)

                            pred_names = [cand_keys[i] for i in ranked_idx]
                            pred_scores = [float(score_matrix[qid][i]) for i in ranked_idx]
                            pred_dicts.append({
                                "query": gt_info.get("query", None),
                                "prediction": pred_names,
                                "prediction_scores": pred_scores,
                                "label": rel_docids,
                                "rel_scores": rel_scores,
                            })
                    else:
                        for qid, (qry_embed, gt_info) in tqdm(enumerate(zip(qry_embeds, gt_infos)),
                                                              desc=f"Calculating scores for {dataset_name}"):
                            pool_names = _local_pool(gt_info)
                            cand_embeds = np.stack([cand_embed_dict[key] for key in pool_names])
                            if qry_embeds.ndim == 3:  # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                                qry_embed = torch.from_numpy(np.array(qry_embed)).unsqueeze(0)
                                cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                                scores = processor.score(
                                    qry_embed, cand_embeds, batch_size=1024)  # use ColPali score function
                                score_vector = scores.squeeze(0).detach().cpu().numpy() if isinstance(
                                    scores, torch.Tensor) else np.array(scores).squeeze(0)
                            else:
                                score_vector = np.dot(qry_embed, cand_embeds.T)

                            ranked_candids = np.argsort(-score_vector)
                            ranked_idx = (
                                ranked_candids.tolist()
                                if isinstance(ranked_candids, np.ndarray) else list(ranked_candids))
                            rel_docids = (
                                gt_info["label_name"]
                                if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]])
                            rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None

                            assert rel_scores is None or len(rel_docids) == len(rel_scores)

                            # Debug: inspect first sample top10
                            if qid == 0 and dataset_name == "QVHighlight":
                                top10_idx = ranked_idx[:10]
                                top10_names = [pool_names[i] for i in top10_idx]
                                print_master(f"[DEBUG QVHighlight] label: {rel_docids}, top10_idx: {top10_idx},"
                                             f" top10_names: {top10_names}")

                            pred_names = [pool_names[i] for i in ranked_idx]
                            pred_scores = [float(score_vector[i]) for i in ranked_idx]
                            pred_dicts.append({
                                "query": gt_info.get("query", None),
                                "prediction": pred_names,
                                "prediction_scores": pred_scores,
                                "label": rel_docids,
                                "rel_scores": rel_scores,
                            })

                    score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
                    pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")

                    metrics_to_report = task_config["metrics"] if task_config.get("metrics", None) is not None else [
                        "hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
                    metrics = RankingMetrics(metrics_to_report)
                    score_dict = metrics.evaluate(pred_dicts)
                    formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
                    score_dict["num_pred"] = len(pred_dicts)
                    score_dict["num_data"] = len(gt_infos)
                    print_master(f"Score of {dataset_name}:")
                    print_master(formatted)
                    print_master(f"Outputting final score to: {score_path}")
                    with open(score_path, "w") as f:
                        json.dump(score_dict, f, indent=4)
                    with open(pred_path, "w") as f:
                        for pred in pred_dicts:
                            f.write(json.dumps(pred) + '\n')
                score_seconds = time.time() - score_start_ts

        except Exception as e:
            dataset_status = "failed"
            dataset_error = f"{type(e).__name__}: {e}"
            raise
        finally:
            dataset_end_human = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dataset_duration_seconds = time.time() - dataset_start_ts
            if rank == 0:
                print_master(
                    f"[Timing] dataset={dataset_name}, status={dataset_status}, "
                    f"total={format_duration(dataset_duration_seconds)}, "
                    f"load={format_duration(load_seconds)}, "
                    f"query={format_duration(query_seconds)}, "
                    f"cand={format_duration(cand_seconds)}, "
                    f"score={format_duration(score_seconds)}"
                )
                append_dataset_timing_row(
                    dataset_timing_log,
                    {
                        "model_name": model_args.model_name,
                        "model_backbone": model_args.model_backbone,
                        "modality": eval_modality,
                        "dataset_name": dataset_name,
                        "start_time": dataset_start_human,
                        "end_time": dataset_end_human,
                        "duration_seconds": f"{dataset_duration_seconds:.2f}",
                        "duration_hms": format_duration(dataset_duration_seconds),
                        "load_seconds": f"{load_seconds:.2f}",
                        "query_seconds": f"{query_seconds:.2f}",
                        "cand_seconds": f"{cand_seconds:.2f}",
                        "score_seconds": f"{score_seconds:.2f}",
                        "do_query": do_query,
                        "do_cand": do_cand,
                        "status": dataset_status,
                        "error": dataset_error,
                    },
                )


if __name__ == "__main__":
    main()
