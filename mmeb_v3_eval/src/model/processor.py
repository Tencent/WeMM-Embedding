# pylint: disable=no-name-in-module
# (Optional backbones such as baseline_backbone / qwen2_vl_tokenselection /
#  qwen2_5_vl live behind try/except import guards or lazy-loading branches,
#  which pylint's static analysis cannot follow.)
import logging
import warnings
from collections import defaultdict

import PIL
from transformers.image_utils import ChannelDimension

try:
    from src.model.baseline_backbone.colpali import ColPaliProcessor
    _COLPALI_IMPORT_ERROR = None
except Exception as exc:
    ColPaliProcessor = None
    _COLPALI_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

import torch  # noqa: E402
import numpy as np  # noqa: E402
from src.utils.basic_utils import print_master  # noqa: E402

try:
    from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
    _LLAVA_IMPORT_ERROR = None
except Exception as exc:
    LlavaNextForConditionalGeneration = None
    _LLAVA_IMPORT_ERROR = exc

try:
    from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
    _PHI3_IMPORT_ERROR = None
except Exception as exc:
    Phi3VForCausalLM = None
    _PHI3_IMPORT_ERROR = exc
try:
    from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    _QWEN2_VL_IMPORT_ERROR = None
except Exception as exc:
    Qwen2VLForConditionalGeneration = None
    Qwen2VLProcessor = None
    _QWEN2_VL_IMPORT_ERROR = exc

try:
    from src.model.vlm_backbone.qwen2_vl_tokenselection import (
        Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration,
        Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor,
    )
    _QWEN2_VL_TS_IMPORT_ERROR = None
except Exception as exc:
    Qwen2VLTokenSelectionForConditionalGeneration = None
    Qwen2VLTokenSelectionProcessor = None
    _QWEN2_VL_TS_IMPORT_ERROR = exc

try:
    from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import InternVideo2_Stage2
    _INTERNVIDEO_IMPORT_ERROR = None
except Exception as exc:
    InternVideo2_Stage2 = None
    _INTERNVIDEO_IMPORT_ERROR = exc

try:
    from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    _QWEN25_VL_IMPORT_ERROR = None
except Exception as exc:
    Qwen2_5_VLForConditionalGeneration = None
    _QWEN25_VL_IMPORT_ERROR = exc

try:
    from src.model.vlm_backbone.qwen2_5_vl_tokenselection import (
        Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    )
    _QWEN25_VL_TS_IMPORT_ERROR = None
except Exception as exc:
    Qwen2_5_VL_TokenSelectionForConditionalGeneration = None
    _QWEN25_VL_TS_IMPORT_ERROR = exc

try:
    from src.model.vlm_backbone.omni_embed import OmniEmbedForConditionalGeneration
    _OMNI_EMBED_IMPORT_ERROR = None
except Exception as exc:
    OmniEmbedForConditionalGeneration = None
    _OMNI_EMBED_IMPORT_ERROR = exc
from src.model.wave_official_utils import load_wave_official_processor_class  # noqa: E402


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000
TEXT_ONLY_MAX_LEN = 2048


def _text_only_max_length(max_length):
    return TEXT_ONLY_MAX_LEN if max_length is None else max_length


def _infer_image_min_size(processor, fallback=28):
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    try:
        patch_val = int(patch_size)
        merge_val = int(merge_size)
    except Exception:
        return fallback
    if patch_val > 0 and merge_val > 0:
        return patch_val * merge_val
    return fallback


def _pad_to_min_image_size(image, min_size):
    if not isinstance(image, PIL.Image.Image):
        return image
    width, height = image.size
    if width >= min_size and height >= min_size:
        return image
    if width <= 0 or height <= 0:
        return image
    new_width = max(width, min_size)
    new_height = max(height, min_size)
    if new_width == width and new_height == height:
        return image
    if image.mode == "RGBA":
        fill = (0, 0, 0, 0)
    elif image.mode == "RGB":
        fill = (0, 0, 0)
    else:
        fill = 0
    canvas = PIL.Image.new(image.mode, (new_width, new_height), color=fill)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    canvas.paste(image, (left, top))
    return canvas


def _item_to_orig_index(item):
    if isinstance(item, tuple):
        return int(item[0])
    return int(item)


def _cat_tensors_with_auto_pad(parts, key=None):
    if len(parts) == 0:
        raise ValueError("Cannot concatenate empty tensor parts.")
    if len(parts) == 1:
        return parts[0]

    normalized_parts = []
    for p in parts:
        if not isinstance(p, torch.Tensor):
            raise TypeError(f"Expected tensor part, got {type(p)} for key={key}.")
        normalized_parts.append(p.reshape(1) if p.dim() == 0 else p)
    parts = normalized_parts

    ndim = parts[0].dim()
    if any(p.dim() != ndim for p in parts):
        shapes = [tuple(p.shape) for p in parts]
        raise RuntimeError(f"Cannot merge tensors with different ranks for key={key}: {shapes}")

    try:
        return torch.cat(parts, dim=0)
    except RuntimeError:
        if ndim == 1:
            return torch.cat(parts, dim=0)

    max_tail = [max(p.shape[d] for p in parts) for d in range(1, ndim)]
    if all(all(p.shape[d] == max_tail[d - 1] for d in range(1, ndim)) for p in parts):
        return torch.cat(parts, dim=0)

    pad_value = False if parts[0].dtype == torch.bool else 0
    padded_parts = []
    for p in parts:
        target_shape = (p.shape[0], *max_tail)
        if tuple(p.shape) == target_shape:
            padded_parts.append(p)
            continue
        padded = p.new_full(target_shape, pad_value)
        slices = (slice(None),) + tuple(slice(0, p.shape[d]) for d in range(1, ndim))
        padded[slices] = p
        padded_parts.append(padded)
    return torch.cat(padded_parts, dim=0)


def _cat_arrays_with_auto_pad(parts, key=None):
    if len(parts) == 0:
        raise ValueError("Cannot concatenate empty ndarray parts.")
    if len(parts) == 1:
        return parts[0]

    normalized_parts = []
    for p in parts:
        if not isinstance(p, np.ndarray):
            raise TypeError(f"Expected ndarray part, got {type(p)} for key={key}.")
        normalized_parts.append(p.reshape(1) if p.ndim == 0 else p)
    parts = normalized_parts

    ndim = parts[0].ndim
    if any(p.ndim != ndim for p in parts):
        shapes = [tuple(p.shape) for p in parts]
        raise RuntimeError(f"Cannot merge ndarrays with different ranks for key={key}: {shapes}")

    try:
        return np.concatenate(parts, axis=0)
    except ValueError:
        if ndim == 1:
            return np.concatenate(parts, axis=0)

    max_tail = [max(p.shape[d] for p in parts) for d in range(1, ndim)]
    if all(all(p.shape[d] == max_tail[d - 1] for d in range(1, ndim)) for p in parts):
        return np.concatenate(parts, axis=0)

    pad_value = False if parts[0].dtype == np.bool_ else 0
    padded_parts = []
    for p in parts:
        target_shape = (p.shape[0], *max_tail)
        if tuple(p.shape) == target_shape:
            padded_parts.append(p)
            continue
        padded = np.full(target_shape, pad_value, dtype=p.dtype)
        slices = (slice(None),) + tuple(slice(0, p.shape[d]) for d in range(1, ndim))
        padded[slices] = p
        padded_parts.append(padded)
    return np.concatenate(padded_parts, axis=0)


def _merge_group_outputs(ordered_groups, total_batch_size):
    """
    Merge per-group processor outputs back to original sample order.
    Tensor values are merged by batch dimension and auto-padded on non-batch dims when needed.
    """
    merged = {}
    all_keys = set()
    for _, out in ordered_groups:
        all_keys.update(out.keys())
    # These keys are packed by modality count (num_images/num_videos), not by batch size.
    # Filling missing rows with zeros breaks global image/video indices in Qwen2.5-Omni RoPE.
    packed_index_keys = {
        "image_grid_thw",
        "video_grid_thw",
        "video_second_per_grid",
        # Audio features are also packed by num_audios (not always batch size).
        # Filling missing rows with zeros can create invalid zero-length audio rows
        # and crash Qwen2.5-Omni audio tower.
        "input_features",
        "feature_attention_mask",
        "audio_feature_lengths",
        # WAVE official BEATs path consumes raw waveform tensors separately.
        "input_raw_wav",
    }

    for key in all_keys:
        example = None
        for _, out in ordered_groups:
            val = out.get(key, None)
            if val is not None:
                example = val
                break
        if example is None:
            continue

        if isinstance(example, torch.Tensor):
            slots = [None] * total_batch_size
            used_slots = False
            concat_parts = []
            ref = None
            for items, out in ordered_groups:
                v = out.get(key, None)
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    ref = ref if ref is not None else v
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == len(items):
                    used_slots = True
                    for row_i, item in enumerate(items):
                        orig_i = _item_to_orig_index(item)
                        if 0 <= orig_i < total_batch_size:
                            slots[orig_i] = v[row_i: row_i + 1]
                elif isinstance(v, torch.Tensor):
                    concat_parts.append(v)

            if used_slots:
                if ref is not None and key not in packed_index_keys:
                    for i, x in enumerate(slots):
                        if x is None:
                            slots[i] = torch.zeros(
                                (1, *ref.shape[1:]),
                                device=ref.device,
                                dtype=ref.dtype,
                            )
                parts = [x for x in slots if isinstance(x, torch.Tensor)]
                if len(parts) > 0:
                    merged[key] = _cat_tensors_with_auto_pad(parts, key=key)
                    continue
            if len(concat_parts) > 0:
                merged[key] = _cat_tensors_with_auto_pad(concat_parts, key=key)
            continue

        if isinstance(example, np.ndarray):
            slots = [None] * total_batch_size
            used_slots = False
            concat_parts = []
            ref = None
            for items, out in ordered_groups:
                v = out.get(key, None)
                if v is None:
                    continue
                if isinstance(v, np.ndarray):
                    ref = ref if ref is not None else v
                if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == len(items):
                    used_slots = True
                    for row_i, item in enumerate(items):
                        orig_i = _item_to_orig_index(item)
                        if 0 <= orig_i < total_batch_size:
                            slots[orig_i] = v[row_i: row_i + 1]
                elif isinstance(v, np.ndarray):
                    concat_parts.append(v)

            if used_slots:
                if ref is not None and key not in packed_index_keys:
                    for i, x in enumerate(slots):
                        if x is None:
                            slots[i] = np.zeros((1, *ref.shape[1:]), dtype=ref.dtype)
                parts = [x for x in slots if isinstance(x, np.ndarray)]
                if len(parts) > 0:
                    merged[key] = _cat_arrays_with_auto_pad(parts, key=key)
                    continue
            if len(concat_parts) > 0:
                merged[key] = _cat_arrays_with_auto_pad(concat_parts, key=key)
            continue

        if isinstance(example, list):
            slots = [None] * total_batch_size
            used_slots = False
            merged_list = []
            for items, out in ordered_groups:
                v = out.get(key, None)
                if v is None:
                    continue
                if isinstance(v, list) and len(v) == len(items):
                    used_slots = True
                    for row_i, item in enumerate(items):
                        orig_i = _item_to_orig_index(item)
                        if 0 <= orig_i < total_batch_size:
                            slots[orig_i] = v[row_i]
                elif isinstance(v, list):
                    merged_list.extend(v)
                else:
                    merged_list.append(v)

            if used_slots:
                merged[key] = [x for x in slots if x is not None]
            elif len(merged_list) > 0:
                merged[key] = merged_list
            continue

        for _, out in ordered_groups:
            v = out.get(key, None)
            if v is not None:
                merged[key] = v
                break

    return merged


def _install_qwen_omni_warning_filters():
    root_logger = logging.getLogger()
    if getattr(root_logger, "_suppress_qwen_omni_warnings", False):
        return

    class _SuppressQwenOmniWarnings(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "System prompt modified, audio output may not work as expected." in msg:
                return False
            if "video processor config saved in `preprocessor.json` file which is deprecated" in msg:
                return False
            if "Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'mrope_section'}" in msg:
                return False
            if msg.startswith("Unused or unrecognized kwargs:"):
                # This warning is very noisy for omni multimodal preprocessing.
                return False
            return True

    warning_filter = _SuppressQwenOmniWarnings()
    target_loggers = [
        root_logger,
        logging.getLogger("transformers"),
        logging.getLogger("transformers.image_utils"),
        logging.getLogger("transformers.video_processing_utils"),
    ]

    for lg in target_loggers:
        lg.addFilter(warning_filter)
        for handler in lg.handlers:
            handler.addFilter(warning_filter)

    # Suppress very noisy tokenizer warning in omni multimodal eval:
    # `max_length` is ignored when padding=True and truncation is disabled.
    warnings.filterwarnings(
        "ignore",
        message=r".*max_length.*ignored.*padding.*no truncation strategy.*",
        category=UserWarning,
        module=r"transformers\.tokenization_utils_base",
    )

    root_logger._suppress_qwen_omni_warnings = True


PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
QWEN3_VL = 'qwen3_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'
QWEN2_5_OMNI = 'qwen2_5_omni'  # Qwen2.5-Omni / Omni-Embed
NVOMNIEMBED = 'nvomniembed'  # NVIDIA omni-embed-nemotron
WAVE = 'wave'  # WAVE official (Qwen2.5-Omni-Thinker)
INTERNVIDEO2 = 'internvideo2'
GME = 'gme'  # QWEN2-VL
LamRA = 'lamra'  # QWEN2-VL
LamRA_QWEN2_5 = 'lamra_qwen25'  # QWEN2.5-VL
COLPALI = 'colpali'  # PaliGemma-3B
E5_V = 'e5_v'  # Llava_next
E5_OMNI = 'e5_omni'  # Haon-Chen/e5-omni-7B (Qwen2.5-Omni-Thinker)
JINA_OMNI = 'jina_omni'  # jinaai/jina-embeddings-v5-omni-small (Qwen3 custom)
LCO_OMNI = 'lco_omni'  # LCO-Embedding/LCO-Embedding-Omni-7B (Qwen2.5-Omni-Thinker)
WEMM_EMBEDDING = 'wemm_embedding'  # WeMM-Embedding (Qwen3.5-VL based, trust_remote_code)
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
    'qwen2_5_vl_tokenselection': QWEN2_5_VL_TOKENSELECTION,
    'qwen3_vl': QWEN3_VL,
    'qwen2_5_omni': QWEN2_5_OMNI,
    'qwen2_5_omni_thinker': QWEN2_5_OMNI,
    'wave': WAVE,
    'nvomniembed': NVOMNIEMBED,
    'internvideo2': INTERNVIDEO2,
    'gme': GME,
    'lamra': LamRA,
    'lamra_qwen25': LamRA,
    'colpali': COLPALI,
    'e5_v': E5_V,
    'e5_omni': E5_OMNI,
    'jina_omni': JINA_OMNI,
    'jina_embeddings_v5_omni': JINA_OMNI,
    'lco_omni': LCO_OMNI,
    'wemm_embedding': WEMM_EMBEDDING,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    QWEN3_VL: "<|image_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_OMNI: "<|image_pad|>",
    NVOMNIEMBED: "<|image_pad|>",
    WAVE: "<|IMAGE|>",
    GME: "<|image_pad|>",
    LamRA: "<|image_pad|>",
    LamRA_QWEN2_5: "<|image_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
    E5_OMNI: "<|image_pad|>",
    JINA_OMNI: "<|image_pad|>",
    LCO_OMNI: "<|image_pad|>",
    WEMM_EMBEDDING: "<|image_pad|>",
}

VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    QWEN3_VL: "<|video_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_OMNI: "<|video_pad|>",
    NVOMNIEMBED: "<|video_pad|>",
    WAVE: "<|VIDEO|>",
    GME: "<|video_pad|>",
    LamRA: "<|video_pad|>",
    LamRA_QWEN2_5: "<|video_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
    E5_V: "<image>",
    E5_OMNI: "<|video_pad|>",
    JINA_OMNI: "<|video_pad|>",
    LCO_OMNI: "<|video_pad|>",
    WEMM_EMBEDDING: "<|video_pad|>",
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    QWEN2_5_OMNI: OmniEmbedForConditionalGeneration,
    E5_OMNI: OmniEmbedForConditionalGeneration,
    LCO_OMNI: OmniEmbedForConditionalGeneration,
    INTERNVIDEO2: InternVideo2_Stage2,
    E5_V: LlavaNextForConditionalGeneration,
}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    Note: due to this change, https://github.com/huggingface/transformers/commit/9215cc62d4366072aacafa4e44028c1ca187167b#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356L1102
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    if model_args.model_backbone == PHI3V:
        from src.model.baseline_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from src.model.baseline_backbone.llava_next import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
    elif model_args.model_backbone in [QWEN2_VL, GME, LamRA]:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer, size=size
        )
    elif model_args.model_backbone == QWEN2_VL_TOKENSELECTION:
        from src.model.vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone in [QWEN2_5_VL, LamRA_QWEN2_5]:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {
            "shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)
    elif model_args.model_backbone == QWEN3_VL:
        try:
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        except Exception as e:
            raise ImportError(
                "Qwen3-VL-Embedding requires transformers>=4.57.0 "
                "(current environment cannot import transformers.models.qwen3_vl)."
            ) from e
        processor = Qwen3VLProcessor.from_pretrained(model_name_or_path)
        if data_args is not None and hasattr(processor, "image_processor"):
            if getattr(data_args, "resize_min_pixels", None) is not None:
                processor.image_processor.min_pixels = data_args.resize_min_pixels
            if getattr(data_args, "resize_max_pixels", None) is not None:
                processor.image_processor.max_pixels = data_args.resize_max_pixels
    elif model_args.model_backbone == QWEN2_5_VL_TOKENSELECTION:
        # TODO: qwen2.5 token selection not working yet
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.image_processing_qwen2_5_vl import (
            Qwen2_5_VLImageProcessor,
        )
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        min_pixels, max_pixels = None, None
        if data_args is not None:
            min_pixels, max_pixels = data_args.resize_min_pixels, data_args.resize_max_pixels
        size = {
            "shortest_edge": min_pixels, "longest_edge": max_pixels, "min_pixels": min_pixels, "max_pixels": max_pixels}
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path, size=size)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone == QWEN2_5_OMNI:
        # Qwen2.5-Omni / Omni-Embed: use official processor with apply_chat_template.
        from src.model.olm_backbone.qwen2_5_moni.processing_qwen2_5_omni import Qwen2_5OmniEmbeddingProcessor
        _install_qwen_omni_warning_filters()

        processor_path = model_args.processor_name if model_args.processor_name else model_args.model_name
        processor = Qwen2_5OmniEmbeddingProcessor.from_pretrained(processor_path, trust_remote_code=True)
        root_logger = logging.getLogger()
        if not getattr(root_logger, "_suppress_qwen_omni_prompt_warning", False):
            class _SuppressQwenOmniPromptWarning(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    msg = record.getMessage()
                    return "System prompt modified, audio output may not work as expected." not in msg

            root_logger.addFilter(_SuppressQwenOmniPromptWarning())
            root_logger._suppress_qwen_omni_prompt_warning = True
    elif model_args.model_backbone == NVOMNIEMBED:
        from transformers import AutoProcessor
        _install_qwen_omni_warning_filters()
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    elif model_args.model_backbone in (E5_OMNI, LCO_OMNI):
        from transformers import AutoProcessor
        _install_qwen_omni_warning_filters()
        processor_path = model_args.processor_name if model_args.processor_name else model_name_or_path
        proc_kwargs = {}
        if data_args is not None and getattr(data_args, "resize_use_processor", True):
            if getattr(data_args, "resize_max_pixels", None) is not None:
                proc_kwargs["max_pixels"] = data_args.resize_max_pixels
            if getattr(data_args, "resize_min_pixels", None) is not None:
                proc_kwargs["min_pixels"] = data_args.resize_min_pixels
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True, **proc_kwargs)
        for attr in ("image_processor", "video_processor"):
            sub_processor = getattr(processor, attr, None)
            if sub_processor is None:
                continue
            if proc_kwargs.get("max_pixels") is not None:
                sub_processor.max_pixels = proc_kwargs["max_pixels"]
            if proc_kwargs.get("min_pixels") is not None:
                sub_processor.min_pixels = proc_kwargs["min_pixels"]
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and hasattr(tok, "padding_side"):
            tok.padding_side = "left"
    elif model_args.model_backbone == JINA_OMNI:
        from transformers import AutoProcessor
        processor_path = model_args.processor_name if model_args.processor_name else model_name_or_path
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    elif model_args.model_backbone == WAVE:
        _install_qwen_omni_warning_filters()
        processor_path = model_args.processor_name if model_args.processor_name else model_name_or_path
        Qwen2_5OmniProcessor = load_wave_official_processor_class()
        processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)

    elif model_args.model_backbone == INTERNVIDEO2:
        return None
    elif model_args.model_backbone == COLPALI:
        from transformers import AutoProcessor
        if ColPaliProcessor is None:
            raise ImportError(
                "ColPaliProcessor is unavailable due to import error; "
                "install compatible dependencies to use COLPALI backbone."
            ) from _COLPALI_IMPORT_ERROR
        processor = ColPaliProcessor.from_pretrained(model_args.model_name)
    elif model_args.model_backbone == WEMM_EMBEDDING:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        if data_args is not None:
            for attr in ("image_processor", "video_processor"):
                sub_processor = getattr(processor, attr, None)
                if sub_processor is None:
                    continue
                if getattr(data_args, "resize_min_pixels", None) is not None:
                    sub_processor.min_pixels = data_args.resize_min_pixels
                if getattr(data_args, "resize_max_pixels", None) is not None:
                    sub_processor.max_pixels = data_args.resize_max_pixels
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, (
        f"Unknown backbone name {hf_config.model_type}.Supported"
        f" models are {SUPPORTED_MODELS}")
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO: NOT FINISHED YET!
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        # in theory, each batch item should contain a list of frames, but we still check for exceptions here
        # if no images as input (not likely to happen in mmeb pro cases)
        if images is None or (type(images) is list and any(i is None for i in images)):
            inputs = processor(
                images=None,
                text=text,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
        else:
            image_exists = True
            # in theory, valid images should be a list of frames
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            inputs = processor(images=images, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask,
        # 'texts': texts,
        # 'images': visual_inputs,
    }
    image_exists = any([p is not None for p in pixel_values])
    if image_exists:
        pixel_values = torch.from_numpy(np.array(pixel_values)).float()
        pixel_values_shape = pixel_values.shape
        pixel_values = pixel_values.reshape(pixel_values_shape[0] * pixel_values_shape[1], *pixel_values_shape[2:])
        image_sizes = torch.tensor(np.array(image_sizes)).long()
        image_sizes_shape = image_sizes.shape
        image_sizes = image_sizes.reshape(image_sizes_shape[0] * image_sizes_shape[1], *image_sizes_shape[2:])
        inputs['pixel_values'] = torch.from_numpy(np.array(pixel_values)).float()
        inputs['image_sizes'] = torch.tensor(np.array(image_sizes)).long()
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(
                text,
                None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    # 1. iterate each pair and process, since processors do not support
    # processing for mixed batch (contains data w/ and w/o visual inputs)
    for text, visual_input in zip(texts, visual_inputs):
        if visual_input is None or (type(visual_input) is list and any(i is None for i in visual_input)):
            # Text-only path. All visual inputs must be valid when a visual token is present.
            inputs = processor(
                text=[text],
                images=None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            try:
                if vlm_image_token in text:
                    if isinstance(visual_input, PIL.Image.Image):
                        # images is a single image
                        visual_input = [visual_input]
                    for iid, image in enumerate(visual_input):
                        # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                        if image.size[0] < 28 or image.size[1] < 28:
                            image = image.resize((56, 56))
                            visual_input[iid] = image
                    inputs = processor(
                        text=[text], images=visual_input, return_tensors="np", max_length=max_length,
                        truncation=(max_length is not None), input_data_format=ChannelDimension.LAST)
                elif vlm_video_token in text:
                    # TODO: check text/video data validity
                    inputs = processor(
                        text=[text], videos=[visual_input], return_tensors="np", max_length=max_length,
                        truncation=(max_length is not None), input_data_format=ChannelDimension.LAST)
                else:
                    raise NotImplementedError(f"No visual token found ({vlm_image_token} or {vlm_video_token}) in"
                                              f" the text: {text}")
            except Exception as e:
                for i in visual_input:
                    print(i.filename)
                raise e
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types:
    #     Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs


def Qwen3_VL_Embedding_process_fn(model_inputs: dict, processor, max_length=None):
    """
    Qwen3-VL-Embedding official-style preprocessing:
    - build chat conversation (system + user content)
    - apply_chat_template
    - process_vision_info
    - processor(text=..., images=..., videos=...)
    """
    texts = model_inputs.get("text", []) or []
    images = model_inputs.get("images", None)
    videos = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)

    if audios is not None and any(a is not None for a in audios):
        raise ValueError(
            "Qwen3-VL-Embedding does not support audio inputs in this eval pipeline. "
            "Please evaluate it on text/image/video modalities."
        )

    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

    def _strip_special_tokens(text: str) -> str:
        if text is None:
            return ""
        stripped = text
        all_mm_tokens = set(VLM_IMAGE_TOKENS.values()) | set(VLM_VIDEO_TOKENS.values())
        for tok in all_mm_tokens:
            if tok:
                stripped = stripped.replace(tok, " ")
        return " ".join(stripped.split())

    def _normalize_video_frames(video_item):
        if video_item is None:
            return None
        if not isinstance(video_item, list):
            return video_item
        flat = []
        for frame in video_item:
            if frame is None:
                continue
            if isinstance(frame, list):
                flat.extend([x for x in frame if x is not None])
            else:
                flat.append(frame)
        return flat if flat else None

    conversations = []
    for idx, text in enumerate(texts):
        content = []

        image_item = None
        if images is not None and idx < len(images):
            image_item = images[idx]
        if image_item is not None:
            if isinstance(image_item, list):
                for image in image_item:
                    if image is not None:
                        content.append({"type": "image", "image": image})
            else:
                content.append({"type": "image", "image": image_item})

        video_item = None
        if videos is not None and idx < len(videos):
            video_item = _normalize_video_frames(videos[idx])
        if video_item is not None:
            content.append({"type": "video", "video": video_item})

        clean_text = _strip_special_tokens(text if isinstance(text, str) else str(text))
        if not clean_text:
            clean_text = " "
        content.append({"type": "text", "text": clean_text})

        conversations.append([
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content},
        ])

    def _apply_chat_template(conversation):
        if hasattr(processor, "apply_chat_template"):
            return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        raise AttributeError("Qwen3-VL processor/tokenizer has no apply_chat_template")

    chat_texts = []
    for conv in conversations:
        text = _apply_chat_template(conv)
        if isinstance(text, list):
            text = "".join([x if isinstance(x, str) else str(x) for x in text])
        if not isinstance(text, str):
            text = str(text)
        chat_texts.append(text)

    try:
        image_inputs, video_inputs, _video_kwargs = process_vision_info(conversations, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(conversations)

    def _has_visual_payload(visual_inputs):
        if visual_inputs is None:
            return False
        if not isinstance(visual_inputs, list):
            return True
        for item in visual_inputs:
            if item is None:
                continue
            if isinstance(item, list):
                if any(x is not None for x in item):
                    return True
            else:
                return True
        return False

    has_visual_inputs = _has_visual_payload(image_inputs) or _has_visual_payload(video_inputs)
    effective_max_length = 8192 if max_length is None else int(max_length)
    # NOTE:
    # Qwen3-VL fast image processor may fail on some raw resolutions when do_resize=False
    # (invalid patch reshape). Use processor default resizing behavior for robustness.
    processor_kwargs = dict(
        text=chat_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if has_visual_inputs:
        # Keep multimodal placeholders intact; truncation may break image/video token alignment.
        processor_kwargs["truncation"] = False
    else:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = effective_max_length
    try:
        outputs = processor(**processor_kwargs)
    except RuntimeError as e:
        if "shape" in str(e) and "invalid for input of size" in str(e):
            logger.warning("Qwen3-VL preprocess failed with raw size; retrying with do_resize=True.")
            outputs = processor(**processor_kwargs, do_resize=True)
        else:
            raise
    return outputs


WEMM_IMAGE_KWARGS = {
    "min_pixels": 64 * 32 * 32,
    "max_pixels": 8192 * 32 * 32,
}
WEMM_VIDEO_KWARGS = {
    "min_pixels": 4 * 32 * 32,
    "max_pixels": 256 * 32 * 32,
    "total_pixels": 8192 * 32 * 32,
    "sample_fps": 1,
    "fps": 1,
    "max_frames": 64,
}


def Wemm_Embedding_process_fn(model_inputs: dict, processor, max_length=None):
    """
    WeMM-Embedding preprocessing:
    - build user-only chat conversation (image/video content items carry their
      own sampling kwargs, which qwen_vl_utils forwards to the processor)
    - apply_chat_template (no generation prompt; the model-side template /
      tokenizer post-processor appends the final <embedding> token)
    - process_vision_info with video metadata
    - processor(text=..., images=..., videos=..., video_metadata=...)
    - expand video_grid_thw framewise (frame-bundle convention)
    """
    texts = model_inputs.get("text", []) or []
    images = model_inputs.get("images", None)
    videos = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)

    if audios is not None and any(a is not None for a in audios):
        raise ValueError(
            "WeMM-Embedding does not support audio inputs in this eval pipeline. "
            "Please evaluate it on text/image/video modalities."
        )

    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

    def _strip_special_tokens(text: str) -> str:
        # Remove the dataset-parser's media placeholder and trim the ends, but
        # preserve the original inner layout (newlines between fields/options
        # are part of the prompt and must survive into the chat template).
        if text is None:
            return ""
        stripped = text
        all_mm_tokens = set(VLM_IMAGE_TOKENS.values()) | set(VLM_VIDEO_TOKENS.values())
        for tok in all_mm_tokens:
            if tok:
                stripped = stripped.replace(tok, "")
        return stripped.strip()

    def _normalize_video_frames(video_item):
        if video_item is None:
            return None
        if not isinstance(video_item, list):
            return video_item
        flat = []
        for frame in video_item:
            if frame is None:
                continue
            if isinstance(frame, list):
                flat.extend([x for x in frame if x is not None])
            else:
                flat.append(frame)
        return flat if flat else None

    image_token = VLM_IMAGE_TOKENS[WEMM_EMBEDDING]
    conversations = []
    for idx, text in enumerate(texts):
        content = []
        raw_text = text if isinstance(text, str) else str(text)
        # The dataset parser embeds the media placeholder in the prompt; use it
        # to recover the intended modality before stripping it below.
        wants_video = VLM_VIDEO_TOKENS[WEMM_EMBEDDING] in raw_text

        image_item = None
        if images is not None and idx < len(images):
            image_item = images[idx]
        if image_item is not None and not isinstance(image_item, list):
            image_item = [image_item]
        if image_item is not None:
            image_item = [image for image in image_item if image is not None] or None

        video_item = None
        if videos is not None and idx < len(videos):
            video_item = _normalize_video_frames(videos[idx])

        interleaved = False
        if video_item is not None:
            content.append({"type": "video", "video": video_item, **WEMM_VIDEO_KWARGS})
        elif image_item is not None and wants_video:
            # Video frames delivered through the image channel (video tasks).
            content.append({"type": "video", "video": image_item, **WEMM_VIDEO_KWARGS})
        elif image_item is not None and image_token in raw_text:
            # Multi-image prompts with inline placeholders (e.g. GUI trajectory
            # candidates: "Observation 1: <img> Action 1: ... Observation 2:
            # <img> ..."): interleave images at their placeholder positions
            # instead of front-loading them, preserving the prompt layout.
            interleaved = True
            parts = raw_text.split(image_token)
            img_iter = iter(image_item[:16])
            for part_idx, part in enumerate(parts):
                part = part.strip()
                if part:
                    content.append({"type": "text", "text": part})
                if part_idx < len(parts) - 1:
                    image = next(img_iter, None)
                    if image is not None:
                        content.append({"type": "image", "image": image, **WEMM_IMAGE_KWARGS})
        elif image_item is not None:
            for image in image_item[:16]:
                content.append({"type": "image", "image": image, **WEMM_IMAGE_KWARGS})

        if not interleaved:
            clean_text = _strip_special_tokens(raw_text)
            if clean_text or not content:
                content.append({"type": "text", "text": clean_text})

        conversations.append([{"role": "user", "content": content}])

    chat_texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in conversations
    ]

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations, return_video_kwargs=True
        )

    video_metadatas = None
    if video_inputs is not None and len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
        video_values, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_values), list(video_metadatas)

    def _has_visual_payload(visual_inputs):
        if visual_inputs is None:
            return False
        if not isinstance(visual_inputs, list):
            return True
        for item in visual_inputs:
            if item is None:
                continue
            if isinstance(item, list):
                if any(x is not None for x in item):
                    return True
            else:
                return True
        return False

    has_visual_inputs = _has_visual_payload(image_inputs) or _has_visual_payload(video_inputs)
    effective_max_length = 8192 if max_length is None else int(max_length)
    processor_kwargs = dict(
        text=chat_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if video_metadatas is not None:
        processor_kwargs["video_metadata"] = video_metadatas
    if isinstance(video_kwargs, dict):
        processor_kwargs.update(video_kwargs)
    if has_visual_inputs:
        # Keep multimodal placeholders intact; truncation may break image/video token alignment.
        processor_kwargs["truncation"] = False
    else:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = effective_max_length
    outputs = processor(**processor_kwargs)

    # Frame-bundle convention: repeat each video grid row by its temporal size
    # so that every frame has its own grid entry with t=1.
    video_grid_thw = outputs.get("video_grid_thw", None)
    if isinstance(video_grid_thw, torch.Tensor) and video_grid_thw.ndim == 2 and video_grid_thw.shape[1] == 3:
        temporal = video_grid_thw[:, 0]
        if bool(torch.all(temporal >= 1)) and bool(torch.any(temporal > 1)):
            expanded = torch.repeat_interleave(video_grid_thw, temporal, dim=0)
            expanded[:, 0] = 1
            outputs["video_grid_thw"] = expanded
    return outputs


def Gme_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    inputs = {
        'texts': model_inputs['text'],
        'images': model_inputs['images'],
    }
    return inputs


def Qwen2_VL_TokenSelection_process_fn(model_inputs: dict, processor: Qwen2VLTokenSelectionProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images) is list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(
                text=[text],
                images=None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            patch_pos.append(None)
            select_mask.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            image_exists = True
            # TODO only
            # handling multi-image data from videos, cannot deal with mixed image + video data
            if VLM_IMAGE_TOKENS[QWEN2_VL] in text:
                inputs = processor(
                    text=[text], images=[images], return_tensors="np", max_length=None,
                    truncation=False, input_data_format=ChannelDimension.LAST)
            elif VLM_VIDEO_TOKENS[QWEN2_VL] in text:
                assert len(images) > 1, f"Video data must have more than 1 frame, got {len(images)}"
                inputs = processor(
                    text=[text], videos=[images], return_tensors="np", max_length=None,
                    truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
                if 'patch_pos' in inputs:
                    patch_pos.append(inputs['patch_pos'])
                if 'select_mask' in inputs:
                    select_mask.append(inputs['select_mask'])
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                patch_pos.append(None)
                select_mask.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

    if image_exists:
        if patch_pos:
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            key_tmp = [
                torch.from_numpy(v) if v is not None else (
                    torch.zeros(patch_pos_shape_for_padding) -
                    1) for v in patch_pos]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)
        if select_mask:
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.ones(
                select_mask_shape_for_padding).bool() for v in select_mask]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)

    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types:
    #     Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long()
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    inputs['patch_pos'] = patch_pos
    inputs['select_mask'] = select_mask

    return inputs


def InternVL_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO not working yet
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(
                text,
                None,
                return_tensors="np",
                max_length=_text_only_max_length(max_length),
                truncation=True,
            )
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def ColPali_process_fn(model_inputs: dict, processor, max_length=None):
    texts, images = model_inputs['text'], model_inputs['images']

    input_ids_batch = []
    attention_mask_batch = []
    pixel_values_batch = []

    for text, image in zip(texts, images):
        if image is not None:
            inputs = processor.process_images([image])
            pixel_values_batch.append(inputs['pixel_values'])
        else:
            inputs = processor.process_queries([text], max_length=_text_only_max_length(max_length))
            pixel_values_batch.append(None)

        input_ids_batch.append(inputs['input_ids'].squeeze().tolist())
        attention_mask_batch.append(inputs['attention_mask'].squeeze().tolist())

    # Pad input_ids and attention_mask
    padded_text_inputs = processor.tokenizer.pad(
        {'input_ids': input_ids_batch, 'attention_mask': attention_mask_batch},
        return_tensors="pt"
    )

    final_input_ids = padded_text_inputs['input_ids']
    final_attention_mask = padded_text_inputs['attention_mask']

    # Handle pixel_values
    if any(pv is not None for pv in pixel_values_batch):
        # Find a representative shape for pixel_values
        representative_pv_shape = None
        for pv in pixel_values_batch:
            if pv is not None:
                representative_pv_shape = pv.shape
                break

        processed_pixel_values = []
        for pv in pixel_values_batch:
            if pv is None:
                # Create a zero tensor of the representative shape
                processed_pixel_values.append(torch.zeros(representative_pv_shape))
            else:
                processed_pixel_values.append(pv)
        final_pixel_values = torch.cat(processed_pixel_values)
    else:
        # No images in the batch at all
        batch_size = len(texts)
        # SigLIP expects 3 channels (RGB) and a square image of 448x448 based on
        # the error (1024 patches = 32x32 patches, with patch_size=14, 32*14=448)
        default_channels = 3
        default_height = 448
        default_width = 448
        final_pixel_values = torch.zeros(batch_size, default_channels, default_height, default_width)

    return {
        'input_ids': final_input_ids,
        'attention_mask': final_attention_mask,
        'pixel_values': final_pixel_values,
    }


def InternVideo2_process_fn(model_inputs: dict, processor, max_length=None):
    if all(x is None for x in model_inputs["images"]):
        # Text side
        from src.model.baseline_backbone.internvideo2.modeling_internvideo2 import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        inputs = tokenizer(
            model_inputs["text"],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt")
    else:
        # Video side
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert from PIL image to tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # ImageNet std
        ])
        frame_list = model_inputs["images"]
        # to make image inputs be exact 4 frames
        # Case 1: frame_list is flat (not a list of lists), e.g., [PIL, PIL, ...]
        if type(frame_list[0]) is not list:
            frame_list = [[img.copy() for _ in range(4)] for img in frame_list]
        # Case 2: frame_list is already a list of lists, ensure each has exactly 4 images
        elif type(frame_list[0]) is list and len(frame_list[0]) != 4:
            new_list = []
            for frames in frame_list:
                if len(frames) < 4:
                    frames = frames + [frames[-1].copy() for _ in range(4 - len(frames))]
                elif len(frames) > 4:
                    # Sample 4 indices uniformly across the sequence
                    indices = np.linspace(0, len(frames) - 1, num=4, dtype=int)
                    frames = [frames[i] for i in indices]
                new_list.append(frames)
            frame_list = new_list
        pixel_values = [
            torch.stack([preprocess(img) for img in frames], dim=0)  # (num_frames, C, H, W)
            for frames in frame_list
        ]

        pixel_values = torch.stack(pixel_values, dim=0)  # (B, num_frames, C, H, W)
        inputs = {'pixel_values': pixel_values}

    return inputs


def Omni_process_fn(model_inputs: dict, processor, max_length=None):
    """
    Qwen2.5-Omni / Omni-Embed processor for EVAL (collate-time).
    Safe rules:
    - Always returns torch tensors for input_ids / attention_mask (LongTensor).
    - Supports text-only / text+image / text+video / text+audio (audio optional).
    - Does NOT rely on Dataset.map storing tensors (avoid Arrow -> list issue).
    """
    import torch
    import PIL
    import numpy as np
    import torchaudio
    import io
    import random

    texts = model_inputs.get("text", [])
    images = model_inputs.get("images", None)
    videos = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)         # could be None or list, keep as 'audios' for compatibility
    audio_sample_rate = model_inputs.get("audio_sample_rate", None)

    if texts is None:
        texts = []
    if not texts:
        raise ValueError("Omni_process_fn: at least one text is required.")

    def _normalize_text_item(t):
        # Keep tokenizer input shape stable: always return a single plain string.
        if t is None:
            return "None"
        if isinstance(t, str):
            s = t
        elif isinstance(t, (list, tuple)):
            if len(t) == 0:
                s = "None"
            else:
                flat = [str(x).strip() for x in t if x is not None and str(x).strip()]
                s = " ".join(flat) if flat else "None"
        else:
            s = str(t)
        s = s.strip()
        return s if s else "None"

    texts = [_normalize_text_item(t) for t in texts]

    # Ensure all arrays have the same length as texts
    batch_size = len(texts)
    if images is None:
        images = [None] * batch_size
    if videos is None:
        videos = [None] * batch_size
    if audios is None:
        audios = [None] * batch_size

    # Check lengths and pad/truncate if necessary
    if len(images) != batch_size:
        if len(images) < batch_size:
            images.extend([None] * (batch_size - len(images)))
        else:
            images = images[:batch_size]
    if len(videos) != batch_size:
        if len(videos) < batch_size:
            videos.extend([None] * (batch_size - len(videos)))
        else:
            videos = videos[:batch_size]
    if len(audios) != batch_size:
        if len(audios) < batch_size:
            audios.extend([None] * (batch_size - len(audios)))
        else:
            audios = audios[:batch_size]

    if not any(t and str(t).strip() for t in texts):
        raise ValueError("Omni_process_fn: at least one non-empty text is required.")

    base_processor = getattr(processor, "base", processor)
    min_image_size = _infer_image_min_size(processor)

    def _squeeze_leading_ones(x, max_squeeze=2):
        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            return x
        for _ in range(max_squeeze):
            if x.dim() >= 1 and x.size(0) == 1:
                x = x.squeeze(0)
            else:
                break
        return x

    def _to_pil_image(x):
        if x is None or isinstance(x, str):
            return x
        if isinstance(x, PIL.Image.Image):
            return _pad_to_min_image_size(x, min_image_size)
        arr = None
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.dim() == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        elif isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
        if arr is None:
            return x
        if arr.dtype.kind == "f":
            arr = (arr * 255.0).clip(0, 255).astype("uint8")
        elif arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype("uint8")
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return _pad_to_min_image_size(PIL.Image.fromarray(arr), min_image_size)

    def _to_video_input(v):
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, list):
            frames = [f for f in v if f is not None]
            return [_to_pil_image(f) for f in frames] if frames else None
        if isinstance(v, (torch.Tensor, np.ndarray)):
            t = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            if t.ndim != 4:
                raise ValueError(f"Video tensor must be 4D, got shape {tuple(t.shape)}")
            if t.shape[-1] in (1, 3):
                frames = [t[i] for i in range(t.shape[0])]
            else:
                frames = [np.transpose(t[i], (1, 2, 0)) for i in range(t.shape[0])]
            return [_to_pil_image(f) for f in frames]
        return v

    def _to_audio_input(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor):
            if a.ndim != 1:
                raise ValueError(f"Audio must be 1D torch.Tensor waveform, got {type(a)}"
                                 f" shape={getattr(a,'shape',None)}")
            return a.detach().cpu().numpy()
        if isinstance(a, np.ndarray):
            return a
        return np.asarray(a)

    def _load_audio_item(item, target_sr, min_samples, max_samples):
        if item is None:
            return None

        if isinstance(item, dict) and "array" in item:
            wav = torch.tensor(item["array"], dtype=torch.float32)
            sr = int(item.get("sampling_rate", target_sr))
            if wav.ndim > 1:
                wav = wav.mean(0)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            if wav.numel() < min_samples:
                return None
            if wav.numel() > max_samples:
                start = random.randint(0, wav.numel() - max_samples)
                wav = wav[start: start + max_samples]
            return wav

        if isinstance(item, torch.Tensor):
            wav = item.float()
            if wav.ndim > 1:
                wav = wav.mean(0)
            if wav.numel() < min_samples:
                return None
            if wav.numel() > max_samples:
                start = random.randint(0, wav.numel() - max_samples)
                wav = wav[start: start + max_samples]
            return wav

        if isinstance(item, np.ndarray):
            wav = torch.tensor(item, dtype=torch.float32)
            if wav.ndim > 1:
                wav = wav.mean(0)
            if wav.numel() < min_samples:
                return None
            if wav.numel() > max_samples:
                start = random.randint(0, wav.numel() - max_samples)
                wav = wav[start: start + max_samples]
            return wav

        if isinstance(item, dict):
            a_path = item.get("path") or item.get("audio_path") or item.get("video_path")
            a_bytes = item.get("bytes", None)
            start_t = float(item.get("start", 0.0) or 0.0)
            end_v = item.get("end", None)
            end_t = float(end_v) if end_v is not None else None

            if a_bytes is not None:
                wave, sr = torchaudio.load(io.BytesIO(a_bytes))
            elif a_path:
                info = torchaudio.info(a_path)
                sr = info.sample_rate
                frame_offset = int(start_t * sr)
                num_frames = int((end_t - start_t) * sr) if end_t is not None else -1
                if num_frames == 0:
                    return None
                wave, _ = torchaudio.load(a_path, frame_offset=frame_offset, num_frames=num_frames)
            else:
                return None

            if wave.dim() > 1:
                wave = wave.mean(0)
            if sr != target_sr:
                wave = torchaudio.functional.resample(wave, sr, target_sr)
            if wave.numel() < min_samples:
                return None
            if wave.numel() > max_samples:
                start = random.randint(0, wave.numel() - max_samples)
                wave = wave[start: start + max_samples]
            return wave

        return None

    # -----------------------------------------
    # 1) Build normalized batches (AutoProcessor path)
    # -----------------------------------------
    images_batch = []
    videos_batch = []
    audios_batch = []
    has_video = False
    has_image = False
    target_sr = int(audio_sample_rate or 16000)
    min_audio_samples = int(target_sr * 0.025)
    max_audio_seconds = model_inputs.get("audio_max_seconds", None)
    if max_audio_seconds is not None:
        max_audio_samples = int(float(max_audio_seconds) * target_sr)
    else:
        max_audio_frames = int(model_inputs.get("audio_max_frames", 1024))
        max_audio_samples = max_audio_frames * 160
    for text, image, video, audio in zip(texts, images, videos, audios):
        if not text.strip():
            raise ValueError("Omni_process_fn: empty text is not allowed.")

        # If video is not provided but image is a multi-frame list, treat it as video.
        if video is None and isinstance(image, list):
            non_null = [v for v in image if v is not None]
            if len(non_null) > 1:
                video = image
                image = None

        image_in = _to_pil_image(image)
        video_in = _to_video_input(video)
        if audio is None:
            audio_in = None
        else:
            audio_in = _load_audio_item(audio, target_sr, min_audio_samples, max_audio_samples)
            if audio_in is not None:
                audio_in = _to_audio_input(audio_in)
        if isinstance(image_in, list):
            image_in = [im for im in image_in if im is not None]
            if not image_in:
                image_in = None
        if isinstance(image_in, PIL.Image.Image):
            w, h = image_in.size
            if w < 28 or h < 28:
                image_in = None
        if isinstance(image_in, list):
            filtered = []
            for im in image_in:
                if isinstance(im, PIL.Image.Image):
                    w, h = im.size
                    if w < 28 or h < 28:
                        continue
                filtered.append(im)
            image_in = filtered if filtered else None
        if isinstance(video_in, list):
            filtered = []
            for im in video_in:
                if isinstance(im, PIL.Image.Image):
                    w, h = im.size
                    if w < 28 or h < 28:
                        continue
                filtered.append(im)
            video_in = filtered if filtered else None

        has_image = has_image or (image_in is not None)
        has_video = has_video or (video_in is not None)
        images_batch.append(image_in)
        videos_batch.append(video_in)
        audios_batch.append(audio_in)

    # Group by modality signature to avoid processor constraints
    # For Qwen2.5-Omni, ensure consistent modality across each group
    idxs = list(range(len(texts)))
    groups = {}

    for i in idxs:
        has_image = images_batch[i] is not None
        has_video = videos_batch[i] is not None
        has_audio = audios_batch[i] is not None

        if has_image and has_video:
            raise ValueError(f"Sample {i}: cannot have both image and video")

        has_visual = has_image or has_video

        # Group by exact modality combination
        if has_visual and has_audio:
            group_key = "visual_audio"
        elif has_visual and not has_audio:
            group_key = "visual_only"
        elif has_audio and not has_visual:
            group_key = "audio_only"
        else:
            group_key = "text_only"

        groups.setdefault(group_key, []).append(i)

    ordered_groups = []
    for group_key, sub in groups.items():
        # For audio groups, filter to only include samples with valid audio
        if group_key in ["visual_audio", "audio_only"]:
            valid_sub = [i for i in sub if audios_batch[i] is not None]
            if not valid_sub:
                continue  # Skip this group if no valid audio samples
            sub = valid_sub

        sub_texts = [texts[i] for i in sub]
        sub_images = [images_batch[i] for i in sub]
        sub_videos = [videos_batch[i] for i in sub]
        sub_audios = [audios_batch[i] for i in sub]

        kwargs = dict(
            text=sub_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        if group_key == "text_only":
            if max_length is None:
                kwargs["max_length"] = 2048
            else:
                kwargs["max_length"] = min(int(max_length), 2048)
        elif max_length is not None:
            kwargs["max_length"] = max_length

        has_images = any(im is not None for im in sub_images)
        has_videos = any(v is not None for v in sub_videos)

        if has_images and has_videos:
            raise ValueError("Omni_process_fn: cannot mix images and videos in the same group.")

        if has_images:
            kwargs["images"] = sub_images
        if has_videos:
            kwargs["videos"] = sub_videos

        # For audio groups, all samples should have valid audio by construction
        if group_key in ["visual_audio", "audio_only"]:
            kwargs["audio"] = sub_audios  # All should be valid
            # Keep consistent with training collator: explicitly pass sampling rate.
            # Without this, Qwen2.5-Omni may extract degenerate audio features (T=1).
            kwargs["audio_kwargs"] = {"sampling_rate": int(target_sr)}

        outputs = base_processor(**kwargs)
        ordered_groups.append((sub, outputs))

    outputs = {}
    if ordered_groups:
        ordered_groups = sorted(ordered_groups, key=lambda t: t[0][0] if len(t[0]) > 0 else -1)
        outputs = _merge_group_outputs(ordered_groups, total_batch_size=len(texts))

    if "audio_attention_mask" in outputs and "feature_attention_mask" not in outputs:
        outputs["feature_attention_mask"] = outputs.pop("audio_attention_mask")

    feats = outputs.get("input_features", None)
    fam = outputs.get("feature_attention_mask", None)
    if isinstance(feats, torch.Tensor) and feats.dim() == 3:
        if feats.shape[1] != 128 and feats.shape[2] == 128:
            feats = feats.transpose(1, 2)
        if fam is None:
            fam = torch.ones(feats.shape[0], feats.shape[2], dtype=torch.long)
        if isinstance(fam, torch.Tensor) and fam.dim() == 2:
            if fam.shape[1] != feats.shape[2]:
                min_len = min(fam.shape[1], feats.shape[2])
                feats = feats[:, :, :min_len]
                fam = fam[:, :min_len]
            outputs["input_features"] = feats
            outputs["feature_attention_mask"] = fam
            # Force audio lengths to match feature mask to avoid RoPE index overflow.
            outputs["audio_feature_lengths"] = fam.sum(-1).long()

    # Ensure input_ids/attention_mask alignment.
    ids = outputs.get("input_ids", None)
    am = outputs.get("attention_mask", None)
    if isinstance(ids, torch.Tensor) and isinstance(am, torch.Tensor):
        if ids.dim() == 2 and am.dim() == 2 and ids.shape[1] != am.shape[1]:
            min_len = min(ids.shape[1], am.shape[1])
            outputs["input_ids"] = ids[:, :min_len]
            outputs["attention_mask"] = am[:, :min_len]

    # squeeze any leading singleton dims produced by processor
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in outputs:
            outputs[key] = _squeeze_leading_ones(outputs.get(key), max_squeeze=2)

    return outputs


def NVOmni_process_fn(model_inputs: dict, processor, max_length=None):
    """
    NVIDIA omni-embed-nemotron processor for EVAL (collate-time).
    Align with the official example: apply_chat_template + process_mm_info.
    """
    texts = model_inputs.get("text", []) or []
    images = model_inputs.get("images", None)
    videos = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)
    audio_sample_rate = model_inputs.get("audio_sample_rate", None)
    resize_min_pixels = model_inputs.get("_resize_min_pixels", None)
    resize_max_pixels = model_inputs.get("_resize_max_pixels", None)
    # WAVE-only compatibility flag: raw wav is consumed by WAVE BEATs branch.
    keep_input_raw_wav = bool(model_inputs.get("_keep_input_raw_wav", False))
    min_image_size = _infer_image_min_size(processor)

    def _strip_special_tokens(text: str) -> str:
        if text is None:
            return ""
        stripped = text
        # Remove placeholder tokens from upstream prompts (e.g. <|VIDEO|>, <|video_pad|>).
        # Multimodal placeholders are injected again by apply_chat_template/content items.
        all_mm_tokens = set(VLM_IMAGE_TOKENS.values()) | set(VLM_VIDEO_TOKENS.values())
        for tok in all_mm_tokens:
            if tok:
                stripped = stripped.replace(tok, " ")
        return " ".join(stripped.split())

    # Build Qwen-Omni style documents with per-sample content.
    documents = []
    for idx, t in enumerate(texts):
        # Keep multimodal placeholders at the front so right-side truncation does not
        # drop <vision_start>/<image|video|audio> tokens and desync *_grid_thw.
        content = []

        if images is not None and idx < len(images):
            img_item = images[idx]
            if isinstance(img_item, list):
                for im in img_item:
                    if im is not None:
                        content.append({"type": "image", "image": _pad_to_min_image_size(im, min_image_size)})
            elif img_item is not None:
                content.append({"type": "image", "image": _pad_to_min_image_size(img_item, min_image_size)})

        if videos is not None and idx < len(videos):
            vid_item = videos[idx]
            if vid_item is not None:
                if isinstance(vid_item, list):
                    normalized_frames = []
                    for frame in vid_item:
                        if frame is None:
                            continue
                        if isinstance(frame, list):
                            for nested_frame in frame:
                                if nested_frame is None:
                                    continue
                                normalized_frames.append(_pad_to_min_image_size(nested_frame, min_image_size))
                        else:
                            normalized_frames.append(_pad_to_min_image_size(frame, min_image_size))
                    vid_item = normalized_frames if normalized_frames else None
                if vid_item is not None:
                    content.append({"type": "video", "video": vid_item})

        if audios is not None and idx < len(audios):
            aud_item = audios[idx]
            if isinstance(aud_item, torch.Tensor):
                aud_item = aud_item.detach().cpu().numpy()
            elif aud_item is not None and not isinstance(aud_item, np.ndarray):
                aud_item = np.asarray(aud_item, dtype=np.float32)
            if aud_item is not None:
                content.append({"type": "audio", "audio": aud_item})

        clean_text = _strip_special_tokens(t)
        if not clean_text:
            clean_text = " "
        content.append({"type": "text", "text": clean_text})

        documents.append([{"role": "user", "content": content}])

    def _is_allowed_media(value, media_type: str) -> bool:
        if isinstance(value, str):
            return True
        if media_type == "image":
            return isinstance(value, PIL.Image.Image)
        if media_type == "video":
            return isinstance(value, list)  # list of frames/objects
        if media_type == "audio":
            return isinstance(value, (np.ndarray, torch.Tensor))
        return False

    def _validate_document_schema(doc, sample_idx: int):
        if not isinstance(doc, dict):
            raise ValueError(f"NVOmni schema error: sample[{sample_idx}] is not dict: {type(doc)}")
        if "role" not in doc or "content" not in doc:
            raise ValueError(f"NVOmni schema error: sample[{sample_idx}] missing 'role' or 'content'")
        if not isinstance(doc["content"], list):
            raise ValueError(f"NVOmni schema error: sample[{sample_idx}].content is not list: {type(doc['content'])}")
        for item_idx, item in enumerate(doc["content"]):
            if not isinstance(item, dict):
                raise ValueError(
                    f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] is not dict: {type(item)}"
                )
            if "type" not in item:
                raise ValueError(
                    f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] missing 'type'"
                )
            item_type = item["type"]
            if item_type == "text":
                if not isinstance(item.get("text", None), str):
                    raise ValueError(
                        f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}].text must be str, got"
                        f" {type(item.get('text', None))}"
                    )
            elif item_type in {"image", "video", "audio"}:
                key = item_type
                if key not in item:
                    raise ValueError(
                        f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] missing '{key}'"
                    )
                if not _is_allowed_media(item[key], item_type):
                    raise ValueError(
                        f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}].{key} "
                        f"has invalid type {type(item[key])}; allow str or supported objects"
                    )
            else:
                raise ValueError(
                    f"NVOmni schema error: sample[{sample_idx}].content[{item_idx}] unsupported type '{item_type}'"
                )

    for i, doc_list in enumerate(documents):
        if not isinstance(doc_list, list) or len(doc_list) != 1:
            raise ValueError(f"NVOmni schema error: sample[{i}] must be single-message list, got {type(doc_list)}"
                             f" len={len(doc_list) if isinstance(doc_list, list) else 'n/a'}")
        _validate_document_schema(doc_list[0], i)

    # Align with official example: apply_chat_template + process_mm_info.
    try:
        from qwen_omni_utils import process_mm_info as _process_mm_info
    except Exception:
        def _process_mm_info(docs, use_audio_in_video=False):
            del use_audio_in_video
            _audios, _images, _videos = [], [], []
            for msg in docs:
                for part in msg.get("content", []):
                    ptype = part.get("type")
                    if ptype == "image":
                        _images.append(part.get("image", None))
                    elif ptype == "video":
                        _videos.append(part.get("video", None))
                    elif ptype == "audio":
                        _audios.append(part.get("audio", None))
            return _audios or None, _images or None, _videos or None

    def _apply_chat_template(doc):
        # Try both wrapper processor and its underlying base/tokenizer objects.
        candidates = [processor]
        base = getattr(processor, "base", None)
        if base is not None:
            candidates.append(base)

        for obj in candidates:
            if hasattr(obj, "apply_chat_template"):
                return obj.apply_chat_template(doc, add_generation_prompt=False, tokenize=False)
            tok = getattr(obj, "tokenizer", None)
            if tok is not None and hasattr(tok, "apply_chat_template"):
                return tok.apply_chat_template(doc, add_generation_prompt=False, tokenize=False)

        raise AttributeError("Processor has no apply_chat_template; install a compatible AutoProcessor/tokenizer.")

    def _normalize_chat_text(value):
        if value is None:
            return " "
        if isinstance(value, list):
            return "".join([v if isinstance(v, str) else str(v) for v in value]) or " "
        if not isinstance(value, str):
            return str(value)
        return value

    def _unwrap_single(x):
        if x is None:
            return None
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        return x

    def _normalize_video_value(v):
        # qwen_omni_utils.process_mm_info returns a list of videos.
        # For single-video samples this is usually [[frame1, ...]], unwrap one level.
        v = _unwrap_single(v)
        if v is None:
            return None
        if isinstance(v, list):
            flat = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, list):
                    flat.extend([x for x in item if x is not None])
                else:
                    flat.append(item)
            v = flat if flat else None
        return v

    sample_records = []
    for doc in documents:
        text = _normalize_chat_text(_apply_chat_template(doc))
        a_i, i_i, v_i = _process_mm_info(doc, use_audio_in_video=False)
        aud = _unwrap_single(a_i)
        img = _unwrap_single(i_i)
        vid = _normalize_video_value(v_i)
        video_len = len(vid) if isinstance(vid, list) else None
        sig = (img is not None, vid is not None, aud is not None, video_len)
        sample_records.append({
            "text": text,
            "audio": aud,
            "image": img,
            "video": vid,
            "sig": sig,
        })

    def _call_processor_for_group(records):
        texts_g = [r["text"] for r in records]
        images_g = [r["image"] for r in records]
        videos_g = [r["video"] for r in records]
        audios_g = [r["audio"] for r in records]

        has_images = records[0]["sig"][0]
        has_videos = records[0]["sig"][1]
        has_audios = records[0]["sig"][2]
        assert all(r["sig"] == records[0]["sig"] for r in records)

        if has_images:
            assert all(x is not None for x in images_g)
        else:
            assert all(x is None for x in images_g)
        if has_videos:
            assert all(x is not None for x in videos_g)
        else:
            assert all(x is None for x in videos_g)
        if has_audios:
            assert all(x is not None for x in audios_g)
        else:
            assert all(x is None for x in audios_g)
        if has_images and has_videos:
            raise ValueError("NVOmni_process_fn: cannot mix images and videos in one group.")

        # Qwen2.5 Omni image processor requires a uniform list shape.
        if has_images and any(isinstance(x, list) for x in images_g):
            images_g = [x if isinstance(x, list) else [x] for x in images_g]

        # Text-only groups can keep short caps.
        # For multimodal groups, truncation may cut <image>/<video>/<audio> placeholders
        # while visual/audio features stay full, causing token/feature mismatch in forward.
        if not has_images and not has_videos and not has_audios:
            if max_length is None:
                text_max_length = 2048
            else:
                text_max_length = min(int(max_length), 2048)
            text_kwargs = {
                "truncation": True,
                "padding": True,
                "max_length": text_max_length,
            }
        else:
            # Keep all multimodal placeholders so token count matches extracted features.
            text_kwargs = {
                "truncation": False,
                "padding": True,
            }
        # Follow DataArguments resize knobs when provided; fallback matches DataArguments defaults.
        effective_min_pixels = int(resize_min_pixels) if resize_min_pixels is not None else (28 * 28 * 4)
        effective_max_pixels = int(resize_max_pixels) if resize_max_pixels is not None else (28 * 28 * 1280)
        videos_kwargs = {
            "min_pixels": effective_min_pixels,
            "max_pixels": effective_max_pixels,
            "use_audio_in_video": False,
        }
        images_kwargs = {
            "min_pixels": effective_min_pixels,
            "max_pixels": effective_max_pixels,
        }
        audio_kwargs = {"max_length": 2048000}
        if audio_sample_rate is not None:
            audio_kwargs["sampling_rate"] = int(audio_sample_rate)

        mm_kwargs = {
            "return_tensors": "pt",
            "text_kwargs": text_kwargs,
            "videos_kwargs": videos_kwargs,
            "images_kwargs": images_kwargs,
        }
        if has_images:
            mm_kwargs["images"] = images_g
        if has_videos:
            mm_kwargs["videos"] = videos_g
        if has_audios:
            mm_kwargs["audio_kwargs"] = audio_kwargs
            mm_kwargs["audio"] = audios_g

        outputs = processor(text=texts_g, **mm_kwargs)

        # Keep raw waveform only when explicitly requested (WAVE official BEATs path).
        if has_audios and keep_input_raw_wav:
            raw_wavs = []
            max_len = 0
            for a in audios_g:
                if isinstance(a, torch.Tensor):
                    wav = a.detach().float().cpu().reshape(-1)
                elif isinstance(a, np.ndarray):
                    wav = torch.from_numpy(a.astype(np.float32, copy=False)).reshape(-1)
                else:
                    wav = torch.as_tensor(np.asarray(a, dtype=np.float32)).reshape(-1)
                raw_wavs.append(wav)
                if wav.numel() > max_len:
                    max_len = int(wav.numel())

            if max_len > 0 and len(raw_wavs) > 0:
                raw_batch = torch.zeros((len(raw_wavs), max_len), dtype=torch.float32)
                for i, wav in enumerate(raw_wavs):
                    if wav.numel() > 0:
                        raw_batch[i, : wav.numel()] = wav
                outputs["input_raw_wav"] = raw_batch

        return outputs

    groups = defaultdict(list)  # sig -> list[(orig_i, record)]
    for i, r in enumerate(sample_records):
        groups[r["sig"]].append((i, r))

    group_outs = {}
    for sig, items in groups.items():
        records = [r for _, r in items]
        out = _call_processor_for_group(records)
        group_outs[sig] = (items, out)

    # Merge group outputs back to original sample order.
    B = len(sample_records)
    ordered_groups = sorted(group_outs.values(), key=lambda t: t[0][0][0] if len(t[0]) > 0 else -1)
    inputs = _merge_group_outputs(ordered_groups, total_batch_size=B)

    feats = inputs.get("input_features", None)
    fam = inputs.get("feature_attention_mask", None)
    if isinstance(feats, torch.Tensor) and feats.dim() == 3:
        # Expect (B, C=128, T) for qwen2_5_omni audio.
        if feats.shape[1] != 128 and feats.shape[2] == 128:
            feats = feats.transpose(1, 2)
        if fam is None:
            fam = torch.ones(feats.shape[0], feats.shape[2], dtype=torch.long)
        if isinstance(fam, torch.Tensor) and fam.dim() == 2:
            if fam.shape[1] != feats.shape[2]:
                min_len = min(fam.shape[1], feats.shape[2])
                feats = feats[:, :, :min_len]
                fam = fam[:, :min_len]
            inputs["input_features"] = feats
            inputs["feature_attention_mask"] = fam

    # Drop qwen2_5_omni-specific fields to avoid forward arg mismatch.
    inputs.pop("audio_attention_mask", None)
    inputs.pop("audio_feature_lengths", None)
    return inputs


def e5_v_prompt_template(text, add_video_token, add_image_token):
    llama3_template = (
        '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        ' \n')
    if text is not None and add_video_token is False and add_image_token is False:  # only text
        prompt = llama3_template.format('{}\nSummary above sentence in one word: '.format(text))
    if text is None and add_video_token:  # only video
        prompt = llama3_template.format('<image>\nSummary above video in one word: ')
    if text is None and add_image_token:  # only image
        prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    if text is not None and add_video_token:  # video + text
        prompt = llama3_template.format('<image>\n{}\nSummary above video and text in one word: '.format(text))
    if text is not None and add_image_token:
        prompt = llama3_template.format('<image>\n{}\nSummary above image and text in one word: '.format(text))

    return prompt  # pylint: disable=possibly-used-before-assignment  # callers guarantee add_video_token/add_image_token is set


PROMPT_TEMPLATE_DICT = {
    "e5_v": e5_v_prompt_template,
}


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    # Formulate input text based on text, special token and instruction.
    # TBD: Reorganize the hard-code part for baselines such as internvideo2
    if model_backbone == "internvideo2":
        return text
    elif model_backbone in [GME, LamRA, LamRA_QWEN2_5]:
        if text:
            return instruction + " " + text  # GME and LamRA do not need special tokens
        else:
            return instruction + " "
    elif model_backbone == E5_V:
        return PROMPT_TEMPLATE_DICT[model_backbone](text, add_video_token, add_image_token)
    elif model_backbone in (E5_OMNI, JINA_OMNI):
        prompt = instruction if instruction else ""
        if text:
            prompt = prompt + text
        return prompt
    elif model_backbone == LCO_OMNI:
        if add_image_token:
            suffix = "\nSummarize the above image in one word:"
        elif add_video_token:
            suffix = "\nSummarize the above video in one word:"
        elif instruction and "audio" in instruction.lower():
            suffix = "\nSummarize the above audio in one word:"
        else:
            suffix = "\nSummarize the above text in one word:"
        if text and not add_image_token and not add_video_token:
            return text + suffix
        return suffix

    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt

    return prompt


def E5Omni_process_fn(model_inputs: dict, processor, max_length=None):
    """e5-omni eval preprocessing.

    e5-omni requires add_generation_prompt=True and an appended <|endoftext|>
    anchor for last-token pooling. Patch both processor and tokenizer owners because
    NVOmni_process_fn may call either apply_chat_template implementation.
    """
    owners = []
    for obj in [processor, getattr(processor, "tokenizer", None)]:
        if obj is not None and hasattr(obj, "apply_chat_template") and obj not in owners:
            owners.append(obj)
    originals = {id(owner): owner.apply_chat_template for owner in owners}

    def _make_patch(orig):
        def _patched(messages, **kwargs):
            kwargs["add_generation_prompt"] = True
            result = orig(messages, **kwargs)
            if isinstance(result, str):
                return result + "<|endoftext|>"
            if isinstance(result, list):
                return [item + "<|endoftext|>" if isinstance(item, str) else item for item in result]
            return result
        return _patched

    for owner in owners:
        owner.apply_chat_template = _make_patch(originals[id(owner)])
    try:
        return NVOmni_process_fn(model_inputs, processor, max_length=max_length)
    finally:
        for owner in owners:
            owner.apply_chat_template = originals[id(owner)]


def LCOOmni_process_fn(model_inputs: dict, processor, max_length=None):
    """LCO-Embedding-Omni eval preprocessing.

    LCO requires add_generation_prompt=True so last-token pooling lands on the
    assistant anchor. It does not append <|endoftext|>.
    """
    owners = []
    for obj in [processor, getattr(processor, "tokenizer", None)]:
        if obj is not None and hasattr(obj, "apply_chat_template") and obj not in owners:
            owners.append(obj)
    originals = {id(owner): owner.apply_chat_template for owner in owners}

    def _make_patch(orig):
        def _patched(messages, **kwargs):
            kwargs["add_generation_prompt"] = True
            return orig(messages, **kwargs)
        return _patched

    for owner in owners:
        owner.apply_chat_template = _make_patch(originals[id(owner)])
    try:
        return NVOmni_process_fn(model_inputs, processor, max_length=max_length)
    finally:
        for owner in owners:
            owner.apply_chat_template = originals[id(owner)]


process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    QWEN3_VL: Qwen3_VL_Embedding_process_fn,
    WEMM_EMBEDDING: Wemm_Embedding_process_fn,
    QWEN2_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    # Keep qwen2_5_omni aligned with nemotron-style multimodal preprocessing.
    QWEN2_5_OMNI: NVOmni_process_fn,
    NVOMNIEMBED: NVOmni_process_fn,
    WAVE: NVOmni_process_fn,
    E5_OMNI: E5Omni_process_fn,
    JINA_OMNI: NVOmni_process_fn,
    LCO_OMNI: LCOOmni_process_fn,
    INTERNVIDEO2: InternVideo2_process_fn,
    GME: Gme_process_fn,
    LamRA: Gme_process_fn,
    LamRA_QWEN2_5: Gme_process_fn,
    COLPALI: ColPali_process_fn,
    E5_V: Llava_NEXT_process_fn,
}
