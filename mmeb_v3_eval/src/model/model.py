# pylint: disable=no-name-in-module  # baseline_backbone is an optional dependency behind a try/except import guard
from typing import Dict
import os
import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, AutoConfig
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    _PEFT_IMPORT_ERROR = None
except Exception as exc:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None
    _PEFT_IMPORT_ERROR = exc

from src.arguments import ModelArguments
from src.model.processor import (
    LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL,
    backbone2model, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V,
    INTERNVIDEO2, GME, VLM_IMAGE_TOKENS, LamRA, LamRA_QWEN2_5, COLPALI, QWEN2_5_OMNI,
    NVOMNIEMBED, WAVE, QWEN3_VL, WEMM_EMBEDDING,
    E5_OMNI, JINA_OMNI, LCO_OMNI
)
from src.model.wave_official_utils import load_wave_official_model_classes
try:
    from src.model.baseline_backbone.colpali import ColPali
    _COLPALI_IMPORT_ERROR = None
except Exception as exc:
    ColPali = None
    _COLPALI_IMPORT_ERROR = exc
try:
    from src.model.baseline_backbone.gme.gme_inference import GmeQwen2VL
    _GME_IMPORT_ERROR = None
except Exception as exc:
    GmeQwen2VL = None
    _GME_IMPORT_ERROR = exc

try:
    from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL
    _LAMRA_IMPORT_ERROR = None
except Exception as exc:
    LamRAQwen2VL = None
    _LAMRA_IMPORT_ERROR = exc

try:
    from src.model.baseline_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL
    _LAMRA25_IMPORT_ERROR = None
except Exception as exc:
    LamRAQwen25VL = None
    _LAMRA25_IMPORT_ERROR = exc
try:
    from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
    _PHI3_IMPORT_ERROR = None
except Exception as exc:
    Phi3VForCausalLM = None
    _PHI3_IMPORT_ERROR = exc

try:
    from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
    _LLAVA_IMPORT_ERROR = None
except Exception as exc:
    LlavaNextForConditionalGeneration = None
    _LLAVA_IMPORT_ERROR = exc

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


def _disable_qwen35_gdn_fast_path():
    """Force the transformers-native PyTorch fallback for Qwen3.5 GatedDeltaNet.

    The optional fla/triton fast path can crash with illegal memory access on
    some GPUs (e.g. sm_80 A800). The native fallback is always correct, so we
    prefer it for evaluation stability and cross-machine reproducibility.

    Note: the torch fallback is orders of magnitude slower on long sequences
    (video frames x batch), which can stall one rank long enough to trip the
    distributed collective timeout in multi-node runs. Set
    WEMM_QWEN35_GDN_FAST_PATH=1 to keep the fla/triton fast path when it works
    on your hardware.
    """
    if os.environ.get("WEMM_QWEN35_GDN_FAST_PATH", "0") == "1":
        return
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception:
        return
    for attr in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        setattr(modeling_qwen3_5, attr, None)
    modeling_qwen3_5.is_fast_path_available = False


def _patch_qwen_omni_flash_rotary_dtype():
    """Patch Qwen2.5-Omni vision flash-attn rotary dtype mismatch."""
    try:
        from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni
        from flash_attn.layers.rotary import apply_rotary_emb
    except Exception:
        return False

    cls = getattr(qwen_omni, "Qwen2_5OmniVisionFlashAttention2", None)
    if cls is None:
        return False
    if getattr(cls._apply_rotary_pos_emb_flashatt, "_vlm2vec_patched", False):
        return True

    def _apply_rotary_pos_emb_flashatt(self, tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # Route through the library's pure-torch vision rotary instead of flash_attn's
        # Triton kernel: the Triton path raises "Triton Error [CUDA]: invalid argument"
        # on long/high-res vision sequences (e.g. GUI screenshots, long video). The
        # pure-torch path is numerically identical to the sdpa path; varlen flash
        # attention itself is unaffected.
        if hasattr(qwen_omni, "apply_rotary_pos_emb_vision"):
            return qwen_omni.apply_rotary_pos_emb_vision(tensor, freqs)
        tensor_ = tensor.float()
        cos = freqs.cos().to(tensor_.dtype)
        sin = freqs.sin().to(tensor_.dtype)
        return apply_rotary_emb(tensor_, cos, sin).type_as(tensor)

    _apply_rotary_pos_emb_flashatt._vlm2vec_patched = True
    cls._apply_rotary_pos_emb_flashatt = _apply_rotary_pos_emb_flashatt
    return True


def _require_peft():
    if LoraConfig is None or get_peft_model is None or PeftModel is None:
        raise ImportError(
            "peft is required for LoRA paths but failed to import. "
            "Please install compatible peft/transformers versions."
        ) from _PEFT_IMPORT_ERROR


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(
        self,
        encoder: PreTrainedModel,
        pooling: str = "last",
        normalize: bool = False,
        temperature: float = 0.02,
    ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling          # <-- string mode
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # inside MMEBModel.__init__(...)
        self.rep_dim = self._infer_rep_dim(encoder=self.encoder)

        if self.rep_dim is None or int(self.rep_dim) <= 0:
            raise ValueError(f"Cannot infer rep_dim from config. Got rep_dim={self.rep_dim}")
        self.rep_dim = int(self.rep_dim)

    def _infer_rep_dim(self, encoder: PreTrainedModel):
        """
        Infer representation dimension from common config layouts.
        Keep legacy priority first, then add nested-config fallbacks for models
        like Qwen3-VL where hidden size lives under text_config.
        """
        cfg = getattr(encoder, "config", None)
        model_cfg = getattr(getattr(encoder, "model", None), "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        vision_cfg = getattr(cfg, "vision_config", None)

        for val in (
            # legacy paths (preserve behavior for existing models)
            getattr(cfg, "hidden_size", None),
            getattr(cfg, "d_model", None),
            getattr(cfg, "embed_dim", None),
            getattr(model_cfg, "hidden_size", None),
            # nested paths (e.g., qwen3_vl)
            getattr(text_cfg, "hidden_size", None),
            getattr(text_cfg, "d_model", None),
            getattr(text_cfg, "embed_dim", None),
            getattr(vision_cfg, "out_hidden_size", None),
            getattr(vision_cfg, "hidden_size", None),
        ):
            if val is not None:
                return val
        return None

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ----------------------------
    # Pooling (string mode)
    # ----------------------------
    def _pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        hidden_states: (B, L, D)
        attention_mask: (B, L) with 1 for valid tokens, 0 for pads
        self.pooling is a STRING mode: 'last' | 'mean' | 'cls'
        """
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=hidden_states.device)
            else:
                attention_mask = attention_mask.to(hidden_states.device).long()

        mode = getattr(self, "pooling", "last")  # string config
        if mode in ("last_token", "eos"):
            mode = "last"

        if mode == "eos":
            mode = "last"

        if mode == "cls":
            reps = hidden_states[:, 0]  # (B, D)

        elif mode == "mean":
            if attention_mask is None:
                reps = hidden_states.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (B, L, 1)
                reps = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

        elif mode == "last":
            if attention_mask is None:
                reps = hidden_states[:, -1]
            else:
                # Works for both left-padding and right-padding:
                # pick the largest position whose mask==1 in each sequence.
                seq_len = attention_mask.size(1)
                pos = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand_as(attention_mask)
                masked_pos = pos.masked_fill(attention_mask == 0, -1)
                last_idx = masked_pos.max(dim=1).values.clamp_min(0)
                reps = hidden_states[
                    torch.arange(hidden_states.size(0), device=hidden_states.device),
                    last_idx
                ]
        else:
            raise ValueError(f"Unknown pooling mode: {mode}. Expected one of ['last','mean','cls'].")

        if getattr(self, "normalize", False):
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)

        return reps

    # ----------------------------
    # Encode Input
    # ----------------------------
    def encode_input(self, input):
        if input is None:
            return None

        backbone = getattr(self, "model_backbone", None)

        # ===== INTERNVIDEO2 =====
        if backbone == INTERNVIDEO2:
            if "input_ids" in input:
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input.get("attention_mask", None),
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                pooled_output = self.encoder.text_proj(pooled_text_embeds)
                pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)
                return pooled_output
            else:
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)
                vfeat = self.encoder.vision_proj(vfeat)
                vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
                return vfeat

        # ===== GME / LamRA =====
        elif backbone in [GME, LamRA, LamRA_QWEN2_5]:
            texts = [t.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + "\n", "") for t in input["texts"]]
            images = []
            for imgs in input["images"]:
                if isinstance(imgs, list):
                    imgs = imgs[len(imgs) // 2]
                    assert not isinstance(imgs, list)
                images.append(imgs)
            pooled_output = self.encoder.get_fused_embeddings(texts=texts, images=images)
            return pooled_output

        # ===== COLPALI =====
        elif backbone == COLPALI:
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output

        # ===== LLAVA_NEXT =====
        elif backbone == LLAVA_NEXT:
            input = dict(input)  # avoid in-place side effects
            input["pixel_values"] = input["pixel_values"].squeeze(dim=1)
            input["image_sizes"] = input["image_sizes"].squeeze(dim=1)
            outputs = self.encoder(**input, return_dict=True, output_hidden_states=True, use_cache=False)
            hidden_states = (
                outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state)
            attn_mask = input.get("attention_mask", None)
            return self._pooling(hidden_states, attn_mask)

        # ===== JINA_OMNI =====
        elif backbone == JINA_OMNI:
            extra_keys = {"texts", "images", "audios"}
            model_input = {k: v for k, v in input.items() if k not in extra_keys}
            dev = self.device
            for k in ("input_ids", "attention_mask"):
                if k in model_input and isinstance(model_input[k], torch.Tensor):
                    model_input[k] = model_input[k].long().to(dev)
            for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                if k in model_input and isinstance(model_input[k], torch.Tensor):
                    model_input[k] = model_input[k].to(dev)
            for k in ("input_features", "feature_attention_mask"):
                if k in model_input and isinstance(model_input[k], torch.Tensor):
                    model_input[k] = model_input[k].to(dev)
            emb = self.encoder.embed(**model_input)
            return F.normalize(emb, p=2, dim=-1) if getattr(self, "normalize", False) else emb

        # ===== WEMM_EMBEDDING =====
        elif backbone == WEMM_EMBEDDING:
            extra_keys = {"texts", "images", "image_paths", "audios"}
            model_input = {k: v for k, v in input.items() if k not in extra_keys}
            # rope_deltas is generation-oriented state cached on the inner model;
            # reset it so batches of varying size/modality stay independent.
            inner = getattr(self.encoder, "model", None)
            if inner is not None and hasattr(inner, "rope_deltas"):
                inner.rope_deltas = None
            emb = self.encoder.embedding(**model_input)
            return F.normalize(emb, p=2, dim=-1) if getattr(self, "normalize", False) else emb

        # ===== QWEN2.5 OMNI / NVOMNI / WAVE / E5_OMNI / LCO_OMNI =====
        elif backbone in {QWEN2_5_OMNI, NVOMNIEMBED, WAVE, E5_OMNI, LCO_OMNI}:
            is_wave = backbone == WAVE
            # 1) Drop debug-only fields that the model does not accept.
            EXTRA_KEYS = {"texts", "images", "audios"}
            raw_input = input
            model_input = {k: v for k, v in raw_input.items() if k not in EXTRA_KEYS}

            # device follow model
            dev = self.device

            # 2) Force input_ids / attention_mask to LongTensor and move to device.
            def _to_long_tensor(x, device=None):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    x = x.long()
                    return x.to(device) if device is not None else x
                t = torch.tensor(x, dtype=torch.long)
                return t.to(device) if device is not None else t

            for k in ("input_ids", "attention_mask"):
                if k in model_input:
                    model_input[k] = _to_long_tensor(model_input[k], device=dev)

            # 3) Align audio field names between processor and model.
            # processor: input_features, audio_attention_mask, audio_feature_lengths
            # model:     input_features, feature_attention_mask, audio_feature_lengths
            if "audio_attention_mask" in model_input and "feature_attention_mask" not in model_input:
                model_input["feature_attention_mask"] = model_input.pop("audio_attention_mask")

            if "audio_values" in model_input and "input_features" not in model_input:
                model_input["input_features"] = model_input.pop("audio_values")

            # Remove redundant keys.
            for k in ("audio_values", "audio_features", "audios"):
                model_input.pop(k, None)

            # 4) Detect available modalities.
            def _has_nonempty(container, key: str) -> bool:
                if key not in container or container[key] is None:
                    return False
                v = container[key]
                if isinstance(v, list):
                    return not all(item is None for item in v)
                return True

            has_image = _has_nonempty(model_input, "pixel_values") or _has_nonempty(model_input, "image_grid_thw")
            has_video = _has_nonempty(model_input, "pixel_values_videos") or _has_nonempty(
                model_input, "video_grid_thw")
            has_audio = (
                _has_nonempty(model_input, "input_features")
                or _has_nonempty(model_input, "feature_attention_mask")
                or _has_nonempty(model_input, "audio_feature_lengths")
            )
            has_multimodal = has_image or has_video or has_audio

            def _pad_and_stack_3d(feats_list, pad_value=0.0):
                # list[Tensor] with [128,T] or [1,128,T] -> Tensor [B,128,Tmax]
                normed = []
                Tmax = 0
                for x in feats_list:
                    if x is None:
                        continue
                    if not isinstance(x, torch.Tensor):
                        x = torch.tensor(x)
                    if x.dim() == 3 and x.size(0) == 1:
                        x = x.squeeze(0)
                    if x.dim() != 2:
                        raise ValueError(f"input_features item must be [128,T] or [1,128,T], got {tuple(x.shape)}")
                    Tmax = max(Tmax, x.size(-1))
                    normed.append(x)
                if len(normed) == 0:
                    return None
                out = []
                for x in normed:
                    pad_t = Tmax - x.size(-1)
                    if pad_t > 0:
                        x = F.pad(x, (0, pad_t), value=pad_value)
                    out.append(x)
                return torch.stack(out, dim=0)

            def _pad_and_stack_2d(mask_list, pad_value=0):
                # list[Tensor] with [T] or [1,T] -> Tensor [B,Tmax]
                normed = []
                Tmax = 0
                for x in mask_list:
                    if x is None:
                        continue
                    if not isinstance(x, torch.Tensor):
                        x = torch.tensor(x)
                    if x.dim() == 2 and x.size(0) == 1:
                        x = x.squeeze(0)
                    if x.dim() != 1:
                        raise ValueError(f"feature_attention_mask item must be [T] or [1,T], got {tuple(x.shape)}")
                    Tmax = max(Tmax, x.numel())
                    normed.append(x)
                if len(normed) == 0:
                    return None
                out = []
                for x in normed:
                    pad_t = Tmax - x.numel()
                    if pad_t > 0:
                        x = F.pad(x, (0, pad_t), value=pad_value)
                    out.append(x)
                return torch.stack(out, dim=0).long()

            if "input_features" in model_input and isinstance(model_input["input_features"], list):
                model_input["input_features"] = _pad_and_stack_3d(model_input["input_features"], pad_value=0.0)

            if "feature_attention_mask" in model_input and isinstance(model_input["feature_attention_mask"], list):
                model_input["feature_attention_mask"] = _pad_and_stack_2d(model_input[
                    "feature_attention_mask"], pad_value=0)

            if "audio_feature_lengths" in model_input and isinstance(model_input["audio_feature_lengths"], list):
                model_input["audio_feature_lengths"] = torch.tensor(model_input[
                    "audio_feature_lengths"], dtype=torch.long)

            if "input_features" in model_input and isinstance(model_input["input_features"], torch.Tensor):
                model_input["input_features"] = model_input["input_features"].to(dtype=torch.float32)
            if (
                "feature_attention_mask") in model_input and isinstance(model_input[
                    "feature_attention_mask"], torch.Tensor):
                model_input["feature_attention_mask"] = model_input["feature_attention_mask"].to(dtype=torch.long)

            # 4.1) Sanitize audio rows: drop zero-length rows to avoid Qwen2.5-Omni audio tower
            # receiving empty sequences (can crash in avg_pool1d with length 0).
            feats = model_input.get("input_features", None)
            fam = model_input.get("feature_attention_mask", None)
            if isinstance(feats, torch.Tensor) and isinstance(fam, torch.Tensor):
                # Ensure feature/mask time dims are aligned.
                if feats.dim() >= 3 and fam.dim() == 2:
                    feat_t = feats.shape[-1]
                    mask_t = fam.shape[-1]
                    if feat_t != mask_t:
                        min_t = min(feat_t, mask_t)
                        feats = feats[..., :min_t]
                        fam = fam[:, :min_t]

                    if feats.shape[-1] == 0:
                        model_input.pop("input_features", None)
                        model_input.pop("feature_attention_mask", None)
                        model_input.pop("audio_feature_lengths", None)
                        model_input.pop("input_raw_wav", None)
                    else:
                        audio_lens = fam.sum(dim=1)
                        invalid_audio = audio_lens <= 0
                        if torch.any(invalid_audio):
                            feats = feats.clone()
                            fam = fam.clone()
                            feats[invalid_audio] = 0
                            fam[invalid_audio] = 0
                            valid_audio_rows = torch.nonzero(fam.sum(dim=1) > 0, as_tuple=False).squeeze(1)
                            if valid_audio_rows.numel() == 0:
                                model_input.pop("input_features", None)
                                model_input.pop("feature_attention_mask", None)
                                model_input.pop("audio_feature_lengths", None)
                                model_input.pop("input_raw_wav", None)
                            else:
                                model_input["input_features"] = feats.index_select(0, valid_audio_rows)
                                model_input["feature_attention_mask"] = fam.index_select(0, valid_audio_rows)
                                afl = model_input.get("audio_feature_lengths", None)
                                if isinstance(afl, torch.Tensor) and afl.dim() > 0:
                                    if afl.size(0) == fam.size(0):
                                        model_input["audio_feature_lengths"] = afl.index_select(0, valid_audio_rows)
                                    else:
                                        model_input.pop("audio_feature_lengths", None)
                                raw_wav = model_input.get("input_raw_wav", None)
                                if isinstance(raw_wav, torch.Tensor) and raw_wav.dim(
                                ) > 0 and raw_wav.size(0) == fam.size(0):
                                    model_input["input_raw_wav"] = raw_wav.index_select(0, valid_audio_rows)
                                elif isinstance(raw_wav, list) and len(raw_wav) == fam.size(0):
                                    valid_idx = valid_audio_rows.detach().cpu().tolist()
                                    model_input["input_raw_wav"] = [raw_wav[i] for i in valid_idx]
                        else:
                            model_input["input_features"] = feats
                            model_input["feature_attention_mask"] = fam

            # Recompute modality flags after audio sanitization.
            has_image = _has_nonempty(model_input, "pixel_values") or _has_nonempty(model_input, "image_grid_thw")
            has_video = _has_nonempty(model_input, "pixel_values_videos") or _has_nonempty(
                model_input, "video_grid_thw")
            has_audio = (
                _has_nonempty(model_input, "input_features")
                or _has_nonempty(model_input, "feature_attention_mask")
                or _has_nonempty(model_input, "audio_feature_lengths")
            )
            has_multimodal = has_image or has_video or has_audio

            # 5) forward
            if has_multimodal or is_wave:
                forward_kwargs = {
                    **model_input,
                    "output_hidden_states": True,
                    "return_dict": True,
                    "use_cache": False,
                }
                if is_wave:
                    forward_kwargs["pred_embeds"] = bool(getattr(self, "wave_pred_embeds", True))
                if backbone == E5_OMNI:
                    cache_position = torch.arange(0, model_input["input_ids"].shape[1], device=dev)
                    prepared = self.encoder.base_model.prepare_inputs_for_generation(
                        **model_input, use_cache=True, cache_position=cache_position,
                    )
                    outputs = self.encoder.base_model(
                        **prepared,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    outputs = self.encoder(**forward_kwargs)
                attn_mask = model_input.get("attention_mask", None)
            else:
                # Text-only path: use the text encoder only.
                visual_keys = {
                    "pixel_values", "image_grid_thw",
                    "pixel_values_videos", "video_grid_thw", "second_per_grid_ts",
                    "input_features", "feature_attention_mask", "audio_feature_lengths",
                    "audio_values", "audio_attention_mask", "audio_features", "audios",
                }
                text_only_input = {k: v for k, v in model_input.items() if k not in visual_keys}

                # Avoid flash-attn alignment issues from inconsistent padding_side.
                for m in [self.encoder, getattr(self.encoder, "model", None)]:
                    if m is None:
                        continue
                    if hasattr(m, "padding_side"):
                        m.padding_side = "left"
                    if hasattr(m, "config") and hasattr(m.config, "padding_side"):
                        m.config.padding_side = "left"

                # Keep text-only and multimodal paths aligned for omni backbones by
                # using the full model forward (can expose `embeddings` consistently).
                if backbone in {NVOMNIEMBED, QWEN2_5_OMNI, E5_OMNI, LCO_OMNI}:
                    if backbone == E5_OMNI:
                        cache_position = torch.arange(0, text_only_input["input_ids"].shape[1], device=dev)
                        prepared = self.encoder.base_model.prepare_inputs_for_generation(
                            **text_only_input, use_cache=True, cache_position=cache_position,
                        )
                        outputs = self.encoder.base_model(
                            **prepared,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    else:
                        outputs = self.encoder(
                            **text_only_input,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False,
                        )
                else:
                    outputs = self.encoder.model(
                        **text_only_input,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                attn_mask = text_only_input.get("attention_mask", None)

            if hasattr(outputs, "mllm_embeds") and outputs.mllm_embeds is not None:
                emb = outputs.mllm_embeds
                if getattr(self, "normalize", False):
                    emb = F.normalize(emb, p=2, dim=-1)
                return emb

            if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
                emb = outputs.embeddings
                if getattr(self, "normalize", False):
                    emb = F.normalize(emb, p=2, dim=-1)
                return emb

            hidden_states = (
                outputs.hidden_states[-1]
                if getattr(outputs, "hidden_states", None) is not None
                else outputs.last_hidden_state)

            if attn_mask is not None:
                if not isinstance(attn_mask, torch.Tensor):
                    attn_mask = torch.tensor(attn_mask, dtype=torch.long, device=hidden_states.device)
                else:
                    attn_mask = attn_mask.to(hidden_states.device).long()

            return self._pooling(hidden_states, attn_mask)

        # ===== Fallback: all other HF backbones =====
        else:
            forward_kwargs = dict(return_dict=True, output_hidden_states=True, use_cache=False)
            if backbone == QWEN3_VL:
                # Qwen3-VL ForConditionalGeneration defaults to full-sequence logits (very memory heavy).
                # We only need hidden states for embedding pooling, so keep logits to a minimal slice.
                forward_kwargs["logits_to_keep"] = 1
            outputs = self.encoder(**input, **forward_kwargs)
            hidden_states = (
                outputs.hidden_states[-1]
                if getattr(outputs, "hidden_states", None) is not None
                else outputs.last_hidden_state)
            attn_mask = input.get("attention_mask", None)
            if attn_mask is not None:
                if not isinstance(attn_mask, torch.Tensor):
                    attn_mask = torch.tensor(attn_mask, dtype=torch.long, device=hidden_states.device)
                else:
                    attn_mask = attn_mask.to(hidden_states.device).long()
            return self._pooling(hidden_states, attn_mask)

    # ----------------------------
    # Build / Load (keep your original)
    # ----------------------------
    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f"Loading backbone [{model_backbone}] from {model_args.model_name}")

        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == QWEN3_VL:
            try:
                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            except Exception as e:
                raise ImportError(
                    "Qwen3-VL-Embedding requires transformers>=4.57.0 "
                    "(cannot import transformers.models.qwen3_vl)."
                ) from e
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False

            from .utils import parse_layer_type
            lm_qwen_layer = 28
            vis_qwen_layer = 32
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,
                vis_skip_layer=vis_skip_layer,
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name,
                **kwargs,
                config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )

        if model_args.lora:
            _require_peft()
            print_master(f"Loading lora adapter from {base_model}")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(","),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False,
            )
            if hasattr(base_model, "model") and base_model.model is not None:
                base_model.model = get_peft_model(base_model.model, lora_config)
            else:
                base_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config_source = (
            model_args.model_name
            if getattr(model_args, "lora", False) and model_args.checkpoint_path
            else model_name_or_path)
        config = None

        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, "model_backbone", model_backbone)

        print_master(f"Loading backbone [{model_args.model_backbone}] from {model_name_or_path}")

        if model_args.model_backbone == WAVE:
            Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration = load_wave_official_model_classes()
            config = Qwen2_5OmniThinkerConfig.from_pretrained(config_source)
            config.use_cache = False
            config.padding_side = "left"
            try:
                config._attn_implementation = "flash_attention_2"
                if hasattr(config, "vision_config"):
                    config.vision_config._attn_implementation = "flash_attention_2"
            except Exception as e:
                print_master(f"Warning: Could not set flash_attention_2 for WAVE: {e}")

            # WAVE official classify settings.
            config.train_classify = bool(getattr(model_args, "wave_train_classify", True))
            config.classify_type = getattr(model_args, "wave_classify_type", "all_layer")
            config.sim_temperature = float(getattr(model_args, "temperature", 0.02))

            wave_use_beats = bool(getattr(model_args, "wave_use_beats", False))
            wave_beats_path = getattr(model_args, "wave_beats_path", None)
            wave_beats_only = bool(getattr(model_args, "wave_beats_only", False))
            if hasattr(config, "audio_config"):
                if wave_use_beats:
                    if wave_beats_path:
                        config.audio_config.beats_path = wave_beats_path
                else:
                    config.audio_config.beats_path = "No"
                config.audio_config.beats_only = wave_beats_only

            base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
            )
        elif model_args.model_backbone in {
            LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, QWEN2_5_OMNI, E5_V
        }:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            try:
                config._attn_implementation = "flash_attention_2"
                if hasattr(config, "vision_config"):
                    config.vision_config._attn_implementation = "flash_attention_2"
            except Exception as e:
                print_master(f"Warning: Could not set flash_attention_2 for {model_args.model_backbone}: {e}")

            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
            )
        elif model_args.model_backbone in {E5_OMNI, LCO_OMNI}:
            import importlib.util
            attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            )
            for module in [base_model, getattr(base_model, "model", None), getattr(base_model, "base_model", None)]:
                if module is None:
                    continue
                if hasattr(module, "padding_side"):
                    module.padding_side = "left"
                if hasattr(module, "config") and hasattr(module.config, "padding_side"):
                    module.config.padding_side = "left"
        elif model_args.model_backbone == QWEN3_VL:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            try:
                from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            except Exception as e:
                raise ImportError(
                    "Qwen3-VL-Embedding requires transformers>=4.57.0 "
                    "(cannot import transformers.models.qwen3_vl)."
                ) from e
            try:
                config._attn_implementation = "flash_attention_2"
                if hasattr(config, "vision_config"):
                    config.vision_config._attn_implementation = "flash_attention_2"
            except Exception as e:
                print_master(f"Warning: Could not set flash_attention_2 for {model_args.model_backbone}: {e}")
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
            )
        elif model_args.model_backbone == WEMM_EMBEDDING:
            _disable_qwen35_gdn_fast_path()
            # sdpa matches the released evaluation setup; the flash_attn varlen
            # path can hit illegal memory access with framewise video grids on
            # some GPU/driver combinations.
            attn_impl = "sdpa"
            base_model = AutoModel.from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
        elif model_args.model_backbone == NVOMNIEMBED:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            try:
                import importlib
                _has_fa2 = importlib.util.find_spec("flash_attn") is not None
                _attn_impl = "flash_attention_2" if _has_fa2 else "sdpa"
                if _attn_impl == "flash_attention_2" and not _patch_qwen_omni_flash_rotary_dtype():
                    _attn_impl = "sdpa"
                config._attn_implementation = _attn_impl
                if hasattr(config, "vision_config"):
                    config.vision_config._attn_implementation = _attn_impl
            except Exception as e:
                print_master(f"Warning: Could not set attn_implementation for {model_args.model_backbone}: {e}")
            base_model = AutoModel.from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
                trust_remote_code=True,
            )
        elif model_args.model_backbone == JINA_OMNI:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            try:
                config._attn_implementation = "sdpa"
                if hasattr(config, "vision_config"):
                    config.vision_config._attn_implementation = "sdpa"
            except Exception as e:
                print_master(f"Warning: Could not set sdpa for {model_args.model_backbone}: {e}")
            base_model = AutoModel.from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config,
                trust_remote_code=True,
                default_task="retrieval",
            )
        elif model_args.model_backbone == PHI3V:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name, **kwargs, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            base_model.padding_side = "right"
        elif model_args.model_backbone == INTERNVIDEO2:
            print_master(f"Loading backbone [{model_args.model_backbone}] from"
                         f" {'src/model/vlm_backbone/internvideo2/'}")
            config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/", trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                "src/model/vlm_backbone/internvideo2/", config=config, trust_remote_code=True
            )
        elif model_args.model_backbone == GME:
            base_model = GmeQwen2VL(model_args.model_name, processor=kwargs["processor"])
            if config is None:
                # GME remote code enforces transformers<4.52; keep local Qwen2-VL config path.
                config = AutoConfig.from_pretrained(config_source, trust_remote_code=False)
            setattr(base_model, "config", config)
        elif model_args.model_backbone == LamRA:
            base_model = LamRAQwen2VL(model_args.model_name)
            if config is None:
                config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
            setattr(base_model, "config", config)
        elif model_args.model_backbone == LamRA_QWEN2_5:
            base_model = LamRAQwen25VL(model_args.model_name)
            if config is None:
                config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
            setattr(base_model, "config", config)
        elif model_args.model_backbone == COLPALI:
            if ColPali is None:
                raise ImportError(
                    "ColPali is unavailable due to import error; install compatible dependencies "
                    "to use COLPALI backbone."
                ) from _COLPALI_IMPORT_ERROR
            base_model = ColPali.from_pretrained(model_args.model_name)
            if config is None:
                config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
            setattr(base_model, "config", config)
        else:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
            )

        if model_args.lora:
            _require_peft()
            print_master(f"Loading LoRA from {model_name_or_path}")
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            # Qwen2.5-Omni wrapper note:
            #   OmniEmbedForConditionalGeneration.forward -> self.base_model(...)
            # so LoRA must be attached to `base_model.base_model.model` (thinker.model),
            # not only to wrapper attribute `base_model.model`.
            if (
                model_args.model_backbone in {QWEN2_5_OMNI, E5_OMNI, LCO_OMNI}
                and hasattr(base_model, "base_model")
                and getattr(base_model.base_model, "model", None) is not None
            ):
                lora_model = PeftModel.from_pretrained(
                    base_model.base_model.model, model_name_or_path, config=lora_config, is_trainable=is_trainable
                )
                if not is_trainable:
                    lora_model = lora_model.merge_and_unload()
                base_model.base_model.model = lora_model
                # Keep wrapper mirror attribute in sync for downstream checks.
                if hasattr(base_model, "model"):
                    base_model.model = lora_model
                print_master("LoRA attached to qwen2_5_omni thinker.model.")
                encoder = base_model
            elif model_args.model_backbone in {QWEN2_VL, QWEN2_VL_TOKENSELECTION}:
                # VLM2Vec adapters for Qwen2-VL are exported against the full
                # conditional-generation model. Attaching to `base_model.model`
                # causes widespread key mismatch and near-random retrieval scores.
                lora_model = PeftModel.from_pretrained(
                    base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable
                )
                if not is_trainable:
                    lora_model = lora_model.merge_and_unload()
                encoder = lora_model
            elif model_args.model_backbone == COLPALI:
                # ColPali adapter keys are exported against the full ColPali wrapper
                # (includes `custom_text_proj` and nested `model.language_model` path).
                # Injecting into `base_model.model` causes large key mismatch.
                lora_model = PeftModel.from_pretrained(
                    base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable
                )
                if not is_trainable:
                    lora_model = lora_model.merge_and_unload()
                encoder = lora_model
            elif hasattr(base_model, "model") and base_model.model is not None:
                lora_model = PeftModel.from_pretrained(
                    base_model.model, model_name_or_path, config=lora_config, is_trainable=is_trainable
                )
                if not is_trainable:
                    lora_model = lora_model.merge_and_unload()
                base_model.model = lora_model
                encoder = base_model
            else:
                lora_model = PeftModel.from_pretrained(
                    base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable
                )
                if not is_trainable:
                    lora_model = lora_model.merge_and_unload()
                encoder = lora_model
            model = cls(
                encoder=encoder,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )

        model.model_backbone = model_args.model_backbone
        model.wave_pred_embeds = bool(getattr(model_args, "wave_pred_embeds", True))
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size
        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self.encoder, "enable_input_require_grads"):
            self.encoder.enable_input_require_grads()
