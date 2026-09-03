from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple
import io
import os
import random
import numpy as np
import torch
import torchaudio
from PIL import Image
from transformers import ProcessorMixin

_OMNI_CHAT_TEMPLATE_INFO_PRINTED = False


@dataclass
class OmniAutoProcessorCollator:
    processor: ProcessorMixin
    data_args: Any
    model_args: Any
    training_args: Any
    batch_size: Optional[int] = None
    _printed_proc_info: bool = False

    # ---------- helpers ----------
    def _clean_image_list(self, imgs):
        """Remove None frames; return None if empty."""
        if imgs is None:
            return None
        if isinstance(imgs, list):
            imgs = [im for im in imgs if im is not None]
            return imgs if len(imgs) > 0 else None
        return imgs  # single PIL or None

    def _split_visual(self, visual):
        """
        Split visual into image or video.
        - image: PIL.Image or list length 1
        - video: list length >1
        Returns (image, video).
        """
        if visual is None:
            return None, None
        if isinstance(visual, list):
            visual = [v for v in visual if v is not None]
            if not visual:
                return None, None
            if len(visual) > 1:
                return None, visual
            return [visual[0]], None
        return [visual], None

    def _random_window(self, items, max_frames: int):
        if items is None or not isinstance(items, list):
            return items
        if max_frames <= 0 or len(items) <= max_frames:
            return items
        start = random.randint(0, len(items) - max_frames)
        return items[start: start + max_frames]

    def _load_image_from_dict(self, raw_images: dict, example: dict):
        """
        raw_images format assumed similar to your current code:
          - 'resolutions' list determines num images/frames
          - 'bytes' or 'paths' optional lists
        """
        if not isinstance(raw_images, dict) or "resolutions" not in raw_images:
            return None
        visual = []
        num_images = len(raw_images["resolutions"])
        for i in range(num_images):
            b = raw_images.get("bytes", [None]*num_images)[i] if "bytes" in raw_images else None
            p = raw_images.get("paths", [None]*num_images)[i] if "paths" in raw_images else None

            if b is not None:
                im = Image.open(io.BytesIO(b)).convert("RGB")
            elif p is not None:
                with Image.open(p) as img:
                    im = img.convert("RGB")
            else:
                im = None
            visual.append(im)

        # optional video frame subsample
        max_frames = getattr(self.data_args, "video_max_frames", 0) or 0
        if max_frames > 0:
            visual = self._random_window(visual, max_frames)

        # optional resize each frame to square
        frame_size = getattr(self.data_args, "video_frame_size", None)
        if frame_size:
            visual = [(im.resize((frame_size, frame_size)) if im is not None else None) for im in visual]

        return self._clean_image_list(visual)

    def _load_audio_batch(self, audio_items: List[Any]) -> Tuple[List[Optional[torch.Tensor]], int]:
        # Unified audio config: use same fields as eval
        target_sr = int(getattr(self.data_args, "audio_sample_rate", 16000) or 16000)
        min_audio_samples = getattr(self.data_args, "audio_min_samples", None)
        if min_audio_samples is None:
            min_audio_samples = int(target_sr * 0.025)  # 25ms

        # Unified max audio length: audio_max_seconds takes precedence over audio_max_samples
        max_audio_seconds = getattr(self.data_args, "audio_max_seconds", None)
        if max_audio_seconds is not None:
            max_audio_samples = int(float(max_audio_seconds) * target_sr)
        else:
            max_audio_samples = getattr(self.data_args, "audio_max_samples", None)
            if max_audio_samples is None:
                # Fallback to legacy config for compatibility
                max_audio_frames = int(getattr(self.data_args, "audio_max_frames", 1024))
                max_audio_samples = max_audio_frames * 160

        out = []
        for item in audio_items:
            if item is None:
                out.append(None)
                continue

            # HF dataset style: {"array":..., "sampling_rate":...}
            if isinstance(item, dict) and "array" in item:
                wav = torch.tensor(item["array"], dtype=torch.float32)
                sr = int(item.get("sampling_rate", target_sr))
                if wav.ndim > 1:
                    wav = wav.mean(0)
                if sr != target_sr:
                    wav = torchaudio.functional.resample(wav, sr, target_sr)
                if wav.numel() < min_audio_samples:
                    out.append(None)
                    continue
                if wav.numel() > max_audio_samples:
                    start = random.randint(0, wav.numel() - max_audio_samples)
                    wav = wav[start: start + max_audio_samples]
                out.append(wav)
                continue

            # Tensor wav
            if isinstance(item, torch.Tensor):
                wav = item.float()
                if wav.ndim > 1:
                    wav = wav.mean(0)
                if wav.numel() < min_audio_samples:
                    out.append(None)
                    continue
                if wav.numel() > max_audio_samples:
                    start = random.randint(0, wav.numel() - max_audio_samples)
                    wav = wav[start: start + max_audio_samples]
                out.append(wav)
                continue

            # Dict with path/bytes (+ optional start/end)
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
                        out.append(None)
                        continue
                    wave, _ = torchaudio.load(a_path, frame_offset=frame_offset, num_frames=num_frames)
                else:
                    out.append(None)
                    continue

                if wave.numel() < min_audio_samples:
                    out.append(None)
                    continue
                if wave.ndim > 1:
                    wave = wave.mean(0)
                if sr != target_sr:
                    wave = torchaudio.functional.resample(wave, sr, target_sr)
                if wave.numel() > max_audio_samples:
                    start = random.randint(0, wave.numel() - max_audio_samples)
                    wave = wave[start: start + max_audio_samples]
                out.append(wave)
                continue

            raise ValueError(f"Unsupported audio item type: {type(item)}")

        return out, target_sr

    def _extract_raw(self, examples: List[dict], text_key: str, image_key: str, audio_key: Optional[str]):
        texts, images, audios = [], [], []
        for ex in examples:
            if ex is None or not ex:
                texts.append(" ")
                images.append(None)
                audios.append(None)
                continue

            t = ex.get(text_key, " ")
            raw_img = ex.get(image_key, None)
            raw_aud = ex.get(audio_key, None) if audio_key else None

            # if list wrappers exist
            if isinstance(t, list):
                if len(t) == 0:
                    t = " "
                elif all(isinstance(x, str) for x in t):
                    valid_texts = [x for x in t if x is not None and x.strip()]
                    if len(valid_texts) == 0:
                        t = " "
                    elif len(valid_texts) == 1:
                        t = valid_texts[0]
                    else:
                        t = random.choice(valid_texts)
                else:
                    t = t[0]
            if isinstance(raw_img, list):
                raw_img = raw_img[0] if len(raw_img) > 0 else None
            if isinstance(raw_aud, list):
                raw_aud = raw_aud[0] if len(raw_aud) > 0 else None

            # normalize image/video
            if isinstance(raw_img, dict):
                img = self._load_image_from_dict(raw_img, ex)
            elif isinstance(raw_img, list):
                img = self._clean_image_list(raw_img)
            else:
                img = raw_img  # PIL or None

            texts.append(t)
            images.append(img)
            audios.append(raw_aud)

        return texts, images, audios

    def _sig(self, img, vid, has_audio: bool):
        has_a = bool(has_audio)
        has_i = img is not None
        has_v = vid is not None
        return (has_i, has_v, has_a)

    def _process_group(self, texts, images, videos, audios, max_length: int):
        kwargs = dict(
            text=texts,
            text_kwargs={
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
            },
        )
        if max_length is not None:
            kwargs["text_kwargs"]["max_length"] = max_length

        has_images = any(im is not None for im in images)
        has_videos = any(v is not None for v in videos)
        has_audios = any(a is not None for a in audios)

        if has_images and has_videos:
            raise ValueError("OmniAutoProcessorCollator: cannot mix images and videos in the same group.")

        if has_images:
            kwargs["images"] = images
        if has_videos:
            kwargs["videos"] = videos
        audio_present = None
        if has_audios:
            # Build per-sample audio list, keep batch aligned; missing audio -> dummy zeros.
            audio_present = []
            audio_np = []
            audio_invalid = False
            dummy_len = 1
            for a in audios:
                if isinstance(a, torch.Tensor) and a.numel() > 0:
                    if a.dim() > 1:
                        a = a.squeeze()
                    if a.dim() == 1:
                        audio_present.append(True)
                        audio_np.append(a.detach().cpu().numpy().astype(np.float32, copy=False))
                        continue
                    if os.environ.get("VLM2VEC_DEBUG"):
                        print(f"[DEBUG] Invalid audio tensor shape: {a.shape}")
                    audio_invalid = True
                    break
                elif a is None:
                    audio_present.append(False)
                    audio_np.append(np.zeros((dummy_len,), dtype=np.float32))
                else:
                    if os.environ.get("VLM2VEC_DEBUG"):
                        print(f"[DEBUG] Invalid audio tensor: {type(a)}")
                    audio_invalid = True
                    break
            if not audio_invalid and len(audio_np) == len(texts):
                kwargs["audio"] = audio_np
                target_sr = int(getattr(self.data_args, "audio_sample_rate", 16000) or 16000)
                kwargs["audio_kwargs"] = {"sampling_rate": target_sr}
                if os.environ.get("VLM2VEC_DEBUG"):
                    print(f"[DEBUG] Passing {len(audio_np)} audios to processor (dummy for missing)")
            else:
                if os.environ.get("VLM2VEC_DEBUG"):
                    print("[DEBUG] Audio invalid or count mismatch, skipping audio processing")
                has_audios = False

        # Audio parameters are only passed when has_audios is True and all audios are valid
        if "audio" in kwargs:
            assert isinstance(kwargs["audio"], list)
            assert len(kwargs["audio"]) == len(kwargs["text"])
            assert all(isinstance(a, np.ndarray) and a.ndim == 1 for a in kwargs["audio"])
            assert all(a.dtype in (np.float32, np.float64) for a in kwargs["audio"])

        base_processor = getattr(self.processor, "base", self.processor)
        if not self._printed_proc_info and os.environ.get("VLM2VEC_DEBUG"):
            print("PROC TYPE:", type(base_processor))
            print("KWARGS KEYS:", sorted(list(kwargs.keys())))
            for attr in ["tokenizer", "image_processor", "feature_extractor", "audio_processor", "base"]:
                print(attr, hasattr(base_processor, attr))
            self._printed_proc_info = True
        outputs = base_processor(**kwargs)

        # If we injected dummy audio, zero out features & masks for missing samples.
        if audio_present is not None and "input_features" in outputs:
            missing = torch.tensor([not x for x in audio_present], dtype=torch.bool)
            if missing.any():
                if isinstance(outputs.get("input_features"), torch.Tensor):
                    outputs["input_features"][missing] = 0
                if isinstance(outputs.get("feature_attention_mask"), torch.Tensor):
                    outputs["feature_attention_mask"][missing] = 0
        return outputs

    def _tensor_only(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _should_use_chat_template(self) -> bool:
        try:
            from src.model.processor import NVOMNIEMBED, QWEN2_5_OMNI
        except Exception:
            return False
        backbone = getattr(self.model_args, "model_backbone", None)
        # Unify Qwen2.5-Omni and Nemotron to the same official multimodal path:
        # apply_chat_template + process_mm_info.
        if backbone not in {NVOMNIEMBED, QWEN2_5_OMNI}:
            return False
        # Only enable if processor exposes apply_chat_template
        if hasattr(self.processor, "apply_chat_template"):
            return True
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "apply_chat_template"):
            return True
        return False

    def _process_group_chat_template(self, texts, images, videos, audios, max_length: int):
        """
        Apply official multimodal path: apply_chat_template + process_mm_info.
        Uses NVOmni_process_fn which wraps both behaviors.
        """
        from src.model.processor import NVOmni_process_fn

        inputs = {"text": texts}
        if images is not None and any(im is not None for im in images):
            inputs["images"] = images
        if videos is not None and any(v is not None for v in videos):
            inputs["videos"] = videos
        if audios is not None and any(a is not None for a in audios):
            inputs["audios"] = audios
            inputs["audio_sample_rate"] = int(getattr(self.data_args, "audio_sample_rate", 16000) or 16000)

        outputs = NVOmni_process_fn(
            model_inputs=inputs,
            processor=self.processor,
            max_length=max_length,
        )
        return self._tensor_only(outputs)

    # ---------- main ----------
    def __call__(self, examples: List[dict]):
        # Ensure qwen omni warning filters are active in dataloader worker processes too.
        if not getattr(self, "_qwen_warning_filter_ready", False):
            try:
                from src.model.processor import _install_qwen_omni_warning_filters
                _install_qwen_omni_warning_filters()
            except Exception:
                pass
            self._qwen_warning_filter_ready = True

        # fixed batch size check
        if self.batch_size is not None and len(examples) < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got {len(examples)}")

        # raw
        q_texts, q_imgs_raw, q_auds = self._extract_raw(examples, "query_text", "query_image", "query_audio")
        p_texts, p_imgs_raw, p_auds = self._extract_raw(examples, "pos_text", "pos_image", "pos_audio")

        # decode audio tensors only for those that exist (avoid dummy audio)
        q_wavs, q_sr = self._load_audio_batch(q_auds)
        p_wavs, p_sr = self._load_audio_batch(p_auds)
        if os.environ.get("VLM2VEC_DEBUG"):
            print(f"[DEBUG] q_audio batch len={len(q_wavs)}"
                  f" shapes/numel={[None if w is None else (tuple(w.shape), int(w.numel())) for w in q_wavs]}")
            print(f"[DEBUG] p_audio batch len={len(p_wavs)}"
                  f" shapes/numel={[None if w is None else (tuple(w.shape), int(w.numel())) for w in p_wavs]}")

        q_imgs, q_vids = [], []
        p_imgs, p_vids = [], []
        for qi, pi in zip(q_imgs_raw, p_imgs_raw):
            q_img, q_vid = self._split_visual(qi)
            p_img, p_vid = self._split_visual(pi)
            q_imgs.append(q_img)
            q_vids.append(q_vid)
            p_imgs.append(p_img)
            p_vids.append(p_vid)

        # valid mask: any sample with audio specified but failed to load -> invalid
        valid = []
        for qa, qw, pa, pw in zip(q_auds, q_wavs, p_auds, p_wavs):
            ok = True
            if qa is not None and qw is None:
                ok = False
            if pa is not None and pw is None:
                ok = False
            valid.append(ok)

        # replace invalid samples with pure text dummy (keeps batch size stable)
        for i, ok in enumerate(valid):
            if not ok:
                q_texts[i], p_texts[i] = " ", " "
                q_imgs[i], p_imgs[i] = None, None
                q_vids[i], p_vids[i] = None, None
                q_wavs[i], p_wavs[i] = None, None

        # group by modality signature to keep processor assumptions clean
        # Re-group after invalid sample replacement to ensure consistent signatures
        idxs = list(range(len(examples)))
        groups = {}
        for i in idxs:
            # Only consider samples with valid modalities; group by has_audio to avoid mixed audio in-group
            q_has_audio = q_wavs[i] is not None
            p_has_audio = p_wavs[i] is not None
            q_sig = self._sig(q_imgs[i], q_vids[i], q_has_audio)
            p_sig = self._sig(p_imgs[i], p_vids[i], p_has_audio)
            s = (q_sig, p_sig)
            groups.setdefault(s, []).append(i)

        raw_max_len = getattr(self.data_args, "max_len", None)
        max_len = 256 if raw_max_len is None else int(raw_max_len)
        use_chat_template = self._should_use_chat_template()
        global _OMNI_CHAT_TEMPLATE_INFO_PRINTED
        if use_chat_template and not _OMNI_CHAT_TEMPLATE_INFO_PRINTED:
            print("[INFO] OmniAutoProcessorCollator: using apply_chat_template + process_mm_info")
            _OMNI_CHAT_TEMPLATE_INFO_PRINTED = True

        def _cat_tensors_with_auto_pad(parts: List[torch.Tensor], key: str):
            if len(parts) == 0:
                raise ValueError("Cannot concatenate empty tensor parts.")
            if len(parts) == 1:
                return parts[0]

            normed = []
            for p in parts:
                if not isinstance(p, torch.Tensor):
                    raise TypeError(f"Expected tensor part for key={key}, got {type(p)}.")
                normed.append(p.reshape(1) if p.dim() == 0 else p)
            parts = normed

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
            padded = []
            for p in parts:
                target_shape = (p.shape[0], *max_tail)
                if tuple(p.shape) == target_shape:
                    padded.append(p)
                    continue
                x = p.new_full(target_shape, pad_value)
                slices = (slice(None),) + tuple(slice(0, p.shape[d]) for d in range(1, ndim))
                x[slices] = p
                padded.append(x)
            return torch.cat(padded, dim=0)

        def _cat_arrays_with_auto_pad(parts: List[np.ndarray], key: str):
            if len(parts) == 0:
                raise ValueError("Cannot concatenate empty ndarray parts.")
            if len(parts) == 1:
                return parts[0]

            normed = []
            for p in parts:
                if not isinstance(p, np.ndarray):
                    raise TypeError(f"Expected ndarray part for key={key}, got {type(p)}.")
                normed.append(p.reshape(1) if p.ndim == 0 else p)
            parts = normed

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
            padded = []
            for p in parts:
                target_shape = (p.shape[0], *max_tail)
                if tuple(p.shape) == target_shape:
                    padded.append(p)
                    continue
                x = np.full(target_shape, pad_value, dtype=p.dtype)
                slices = (slice(None),) + tuple(slice(0, p.shape[d]) for d in range(1, ndim))
                x[slices] = p
                padded.append(x)
            return np.concatenate(padded, axis=0)

        def _merge(out_list: List[Dict[str, Any]], idx_chunks: List[List[int]], total_bsz: int) -> Dict[str, Any]:
            # Merge per-group outputs back to original sample order.
            all_keys = set()
            for out in out_list:
                all_keys.update(out.keys())

            final: Dict[str, Any] = {}
            debug_merge = os.environ.get("VLM2VEC_DEBUG_MERGE_SHAPES", "").lower()
            # These keys are packed by modality count (num_images/num_videos), not by batch size.
            # Filling missing rows with zeros corrupts global image/video indices in Qwen2.5-Omni.
            packed_index_keys = {"image_grid_thw", "video_grid_thw", "video_second_per_grid"}

            for k in all_keys:
                example = None
                for out in out_list:
                    if k in out and out[k] is not None:
                        example = out[k]
                        break
                if example is None:
                    continue

                if isinstance(example, torch.Tensor):
                    slots = [None] * total_bsz
                    ref = None
                    extra = []
                    for out, chunk in zip(out_list, idx_chunks):
                        if k not in out or out[k] is None:
                            continue
                        v = out[k]
                        if not isinstance(v, torch.Tensor):
                            continue
                        ref = ref if ref is not None else v
                        if v.dim() > 0 and v.shape[0] == len(chunk):
                            for j, orig_i in enumerate(chunk):
                                slots[orig_i] = v[j: j + 1]
                        else:
                            extra.append(v)

                    if any(x is not None for x in slots):
                        if ref is not None and k not in packed_index_keys:
                            for i, x in enumerate(slots):
                                if x is None:
                                    slots[i] = torch.zeros(
                                        (1, *ref.shape[1:]),
                                        device=ref.device,
                                        dtype=ref.dtype,
                                    )
                        parts = [x for x in slots if isinstance(x, torch.Tensor)]
                        if debug_merge and len(parts) > 0:
                            shapes = [tuple(c.shape) for c in parts]
                            if len(set(shapes)) != 1:
                                print(f"[DEBUG][merge] key={k} reordered_shapes={shapes}")
                        if len(parts) > 0:
                            final[k] = _cat_tensors_with_auto_pad(parts, key=k)
                            continue
                    if len(extra) > 0:
                        final[k] = _cat_tensors_with_auto_pad(extra, key=k)
                    continue

                if isinstance(example, np.ndarray):
                    slots = [None] * total_bsz
                    ref = None
                    extra = []
                    for out, chunk in zip(out_list, idx_chunks):
                        if k not in out or out[k] is None:
                            continue
                        v = out[k]
                        if isinstance(v, torch.Tensor):
                            v = v.detach().cpu().numpy()
                        if not isinstance(v, np.ndarray):
                            continue
                        ref = ref if ref is not None else v
                        if v.ndim > 0 and v.shape[0] == len(chunk):
                            for j, orig_i in enumerate(chunk):
                                slots[orig_i] = v[j: j + 1]
                        else:
                            extra.append(v)

                    if any(x is not None for x in slots):
                        if ref is not None and k not in packed_index_keys:
                            for i, x in enumerate(slots):
                                if x is None:
                                    slots[i] = np.zeros((1, *ref.shape[1:]), dtype=ref.dtype)
                        parts = [x for x in slots if isinstance(x, np.ndarray)]
                        if debug_merge and len(parts) > 0:
                            shapes = [tuple(c.shape) for c in parts]
                            if len(set(shapes)) != 1:
                                print(f"[DEBUG][merge] key={k} reordered_shapes={shapes}")
                        if len(parts) > 0:
                            final[k] = _cat_arrays_with_auto_pad(parts, key=k)
                            continue
                    if len(extra) > 0:
                        final[k] = _cat_arrays_with_auto_pad(extra, key=k)
                    continue

                if isinstance(example, list):
                    slots = [None] * total_bsz
                    per_sample = True
                    for out, chunk in zip(out_list, idx_chunks):
                        if k not in out or out[k] is None:
                            continue
                        v = out[k]
                        if isinstance(v, list) and len(v) == len(chunk):
                            for j, orig_i in enumerate(chunk):
                                slots[orig_i] = v[j]
                        else:
                            per_sample = False
                            break
                    if per_sample and any(x is not None for x in slots):
                        final[k] = slots
                    else:
                        tmp = []
                        for out in out_list:
                            if k not in out or out[k] is None:
                                continue
                            v = out[k]
                            if isinstance(v, list):
                                tmp.extend(v)
                            else:
                                tmp.append(v)
                        final[k] = tmp
                    continue

                for out in out_list:
                    if k in out and out[k] is not None:
                        final[k] = out[k]
                        break

            return final

        # process qry/pos in matching grouping to preserve alignment
        qry_outs = []
        pos_outs = []
        order_chunks = []

        for s, sub in groups.items():
            # keep sub order stable
            sub_q_text = [q_texts[i] for i in sub]
            sub_q_img = [q_imgs[i] for i in sub]
            sub_q_vid = [q_vids[i] for i in sub]
            sub_q_wav = [q_wavs[i] for i in sub]

            sub_p_text = [p_texts[i] for i in sub]
            sub_p_img = [p_imgs[i] for i in sub]
            sub_p_vid = [p_vids[i] for i in sub]
            sub_p_wav = [p_wavs[i] for i in sub]

            if use_chat_template:
                q_proc = self._process_group_chat_template(sub_q_text, sub_q_img, sub_q_vid, sub_q_wav, max_len)
                p_proc = self._process_group_chat_template(sub_p_text, sub_p_img, sub_p_vid, sub_p_wav, max_len)
            else:
                q_proc = self._process_group(sub_q_text, sub_q_img, sub_q_vid, sub_q_wav, max_len)
                p_proc = self._process_group(sub_p_text, sub_p_img, sub_p_vid, sub_p_wav, max_len)

            qry_outs.append(q_proc)
            pos_outs.append(p_proc)
            order_chunks.append(sub)

        # merge
        qry_batch = _merge(qry_outs, order_chunks, len(examples))
        pos_batch = _merge(pos_outs, order_chunks, len(examples))

        # attach metadata
        qry_batch["valid_example_mask"] = torch.tensor(valid, dtype=torch.bool)
        pos_batch["valid_example_mask"] = torch.tensor(valid, dtype=torch.bool)

        # Optional: preserve candidate identity for multi-positive contrastive loss.
        pair_ids = []
        has_pair_id = False
        for ex in examples:
            pid = ex.get("pair_id", None) if isinstance(ex, dict) else None
            if pid is None:
                pair_ids.append(-1)
            else:
                has_pair_id = True
                pair_ids.append(int(pid))
        if has_pair_id:
            pair_id_tensor = torch.tensor(pair_ids, dtype=torch.long)
            qry_batch["pair_id"] = pair_id_tensor
            pos_batch["pair_id"] = pair_id_tensor

        # for debug / hash you can still attach raw texts if you want
        qry_batch["text"] = q_texts
        pos_batch["text"] = p_texts

        return qry_batch, pos_batch
