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


@dataclass
class OmniEvalAutoProcessorCollator:
    processor: ProcessorMixin
    data_args: Any
    model_args: Any
    training_args: Any
    batch_size: Optional[int] = None

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

    def _load_audio_batch_eval(self, audio_items: List[Any]) -> Tuple[List[Optional[torch.Tensor]], int]:
        """
        Eval version: deterministic audio loading and cropping
        """
        target_sr = int(getattr(self.data_args, "audio_sample_rate", 16000) or 16000)
        min_audio_samples = getattr(self.data_args, "audio_min_samples", None)
        if min_audio_samples is None:
            min_audio_samples = int(target_sr * 0.025)  # 25ms

        # Unified audio config: same as train collator
        max_audio_seconds = getattr(self.data_args, "audio_max_seconds", None)
        if max_audio_seconds is not None:
            max_audio_samples = int(float(max_audio_seconds) * target_sr)
        else:
            max_audio_samples = getattr(self.data_args, "audio_max_samples", None)
            if max_audio_samples is None:
                # Fallback to legacy config for compatibility
                max_audio_frames = int(getattr(self.data_args, "audio_max_frames", 1024))
                max_audio_samples = max_audio_frames * 160

        # Eval crop strategy: head/center/multi_crop (deterministic)
        eval_crop = getattr(self.data_args, "eval_crop", "head")

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
                    # Eval deterministic cropping
                    if eval_crop == "head":
                        wav = wav[:max_audio_samples]
                    elif eval_crop == "center":
                        start = (wav.numel() - max_audio_samples) // 2
                        wav = wav[start:start + max_audio_samples]
                    elif eval_crop == "multi_crop":
                        # For multi_crop, we could return multiple crops
                        # For simplicity, use center crop as default
                        start = (wav.numel() - max_audio_samples) // 2
                        wav = wav[start:start + max_audio_samples]
                    else:
                        # Default to head
                        wav = wav[:max_audio_samples]
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
                    # Eval deterministic cropping
                    if eval_crop == "head":
                        wav = wav[:max_audio_samples]
                    elif eval_crop == "center":
                        start = (wav.numel() - max_audio_samples) // 2
                        wav = wav[start:start + max_audio_samples]
                    elif eval_crop == "multi_crop":
                        start = (wav.numel() - max_audio_samples) // 2
                        wav = wav[start:start + max_audio_samples]
                    else:
                        wav = wav[:max_audio_samples]
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
                    # Eval deterministic cropping
                    if eval_crop == "head":
                        wave = wave[:max_audio_samples]
                    elif eval_crop == "center":
                        start = (wave.numel() - max_audio_samples) // 2
                        wave = wave[start:start + max_audio_samples]
                    elif eval_crop == "multi_crop":
                        start = (wave.numel() - max_audio_samples) // 2
                        wave = wave[start:start + max_audio_samples]
                    else:
                        wave = wave[:max_audio_samples]
                out.append(wave)
                continue

            raise ValueError(f"Unsupported audio item type: {type(item)}")

        return out, target_sr

    def _random_window(self, items, max_frames: int):
        if items is None or not isinstance(items, list):
            return items
        if max_frames <= 0 or len(items) <= max_frames:
            return items
        start = random.randint(0, len(items) - max_frames)
        return items[start: start + max_frames]

    def _extract_raw_eval(self, examples: List[dict], text_key: str, image_key: str, audio_key: Optional[str]):
        """
        Extract raw data for eval, similar to train version but adapted for eval datasets
        """
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
                t = t[0] if len(t) > 0 else " "
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

    def _sig(self, img, vid, aud):
        has_a = aud is not None
        has_i = img is not None
        has_v = vid is not None
        return (has_i, has_v, has_a)

    def _process_group_eval(self, texts, images, videos, audios, max_length: int):
        """
        Process a group of examples with same modality signature for eval
        """
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
        if has_images and has_videos:
            raise ValueError("OmniEvalAutoProcessorCollator: cannot mix images and videos in the same group.")
        if has_images:
            kwargs["images"] = images
        if has_videos:
            kwargs["videos"] = videos
        audio_present = None
        if any(a is not None for a in audios):
            audio_present = []
            audio_np = []
            dummy_len = 1
            audio_invalid = False
            for a in audios:
                if isinstance(a, torch.Tensor) and a.numel() > 0:
                    if a.dim() > 1:
                        a = a.squeeze()
                    if a.dim() == 1:
                        audio_present.append(True)
                        audio_np.append(a.detach().cpu().numpy().astype(np.float32, copy=False))
                        continue
                    audio_invalid = True
                    break
                elif a is None:
                    audio_present.append(False)
                    audio_np.append(np.zeros((dummy_len,), dtype=np.float32))
                else:
                    audio_invalid = True
                    break
            if not audio_invalid and len(audio_np) == len(texts):
                kwargs["audio"] = audio_np
            else:
                audio_present = None

        outputs = self.processor(**kwargs)
        if audio_present is not None and "input_features" in outputs:
            missing = torch.tensor([not x for x in audio_present], dtype=torch.bool)
            if missing.any():
                if isinstance(outputs.get("input_features"), torch.Tensor):
                    outputs["input_features"][missing] = 0
                if isinstance(outputs.get("feature_attention_mask"), torch.Tensor):
                    outputs["feature_attention_mask"][missing] = 0
        return outputs

    def __call__(self, examples: List[dict]):
        """
        Eval collator with modality grouping to avoid mixed-modality batch constraints
        """
        # fixed batch size check
        if self.batch_size is not None and len(examples) < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got {len(examples)}")

        # Extract data based on eval dataset schema
        # Support both query/cand and direct text/image/audio formats
        if "query_text" in examples[0]:
            # Query-candidate format (retrieval tasks)
            q_texts, q_imgs_raw, q_auds = self._extract_raw_eval(examples, "query_text", "query_image", "query_audio")
            c_texts, c_imgs_raw, c_auds = self._extract_raw_eval(examples, "cand_text", "cand_image", "cand_audio")
        else:
            # Direct format (classification/other tasks)
            q_texts, q_imgs_raw, q_auds = self._extract_raw_eval(examples, "text", "image", "audio")
            c_texts, c_imgs_raw, c_auds = [], [], []  # No candidates

        # Load audio with eval strategy (deterministic)
        q_wavs, q_sr = self._load_audio_batch_eval(q_auds)
        if c_auds:
            c_wavs, c_sr = self._load_audio_batch_eval(c_auds)
        else:
            c_wavs, _ = [], q_sr

        # Split visuals
        q_imgs, q_vids = [], []
        c_imgs, c_vids = [], []
        for qi, ci in zip(q_imgs_raw, c_imgs_raw or [None] * len(q_imgs_raw)):
            q_img, q_vid = self._split_visual(qi)
            c_img, c_vid = self._split_visual(ci) if ci is not None else (None, None)
            q_imgs.append(q_img)
            q_vids.append(q_vid)
            c_imgs.append(c_img)
            c_vids.append(c_vid)

        # Validate audio loading
        valid = []
        for i, (qa, qw) in enumerate(zip(q_auds, q_wavs)):
            ok = True
            if qa is not None and qw is None:
                ok = False
            # For retrieval tasks, also check candidates
            if c_auds and i < len(c_auds):
                ca, cw = c_auds[i], c_wavs[i]
                if ca is not None and cw is None:
                    ok = False
            valid.append(ok)

        # Replace invalid samples with pure text dummy
        for i, ok in enumerate(valid):
            if not ok:
                q_texts[i], c_texts[i] = " ", " "
                q_imgs[i], c_imgs[i] = None, None
                q_vids[i], c_vids[i] = None, None
                q_wavs[i], c_wavs[i] = None, None

        # Group by modality signature to keep processor assumptions clean
        idxs = list(range(len(examples)))
        groups = {}
        for i in idxs:
            q_sig = self._sig(q_imgs[i], q_vids[i], q_wavs[i])
            c_sig = self._sig(c_imgs[i], c_vids[i], c_wavs[i]) if c_imgs else (False, False, False)
            groups.setdefault((q_sig, c_sig), []).append(i)

        raw_max_len = getattr(self.data_args, "max_len", None)
        max_len = 256 if raw_max_len is None else int(raw_max_len)

        def _merge(out_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Ensure missing batch-first keys are filled so batch dims align.
            batch_first_keys = {
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "position_ids",
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
                "labels",
            }
            key_refs = {}
            for out in out_list:
                for k, v in out.items():
                    if k in batch_first_keys and isinstance(v, torch.Tensor):
                        key_refs.setdefault(k, v)
            if key_refs:
                for out in out_list:
                    bsz = None
                    for v in out.values():
                        if isinstance(v, torch.Tensor):
                            bsz = v.shape[0]
                            break
                    if bsz is None:
                        continue
                    for k, ref in key_refs.items():
                        if k not in out:
                            out[k] = torch.zeros(
                                (bsz, *ref.shape[1:]),
                                device=ref.device,
                                dtype=ref.dtype,
                            )
            merged = {}
            for out in out_list:
                for k, v in out.items():
                    merged.setdefault(k, []).append(v)
            final = {}
            debug_merge = os.environ.get("VLM2VEC_DEBUG_MERGE_SHAPES", "").lower()
            for k, chunks in merged.items():
                if isinstance(chunks[0], torch.Tensor):
                    if debug_merge:
                        shapes = [tuple(c.shape) for c in chunks]
                        if len(set(shapes)) != 1:
                            print(f"[DEBUG][merge] key={k} shapes={shapes}")
                    final[k] = torch.cat(chunks, dim=0)
                else:
                    if debug_merge:
                        lens = [len(c) for c in chunks]
                        if len(set(lens)) != 1:
                            print(f"[DEBUG][merge] key={k} lens={lens}")
                    tmp = []
                    for c in chunks:
                        tmp.extend(c)
                    final[k] = tmp
            return final

        # Process qry/pos in matching grouping to preserve alignment
        qry_outs = []
        cand_outs = []
        order_chunks = []

        for s, sub in groups.items():
            # keep sub order stable
            sub_q_text = [q_texts[i] for i in sub]
            sub_q_img = [q_imgs[i] for i in sub]
            sub_q_vid = [q_vids[i] for i in sub]
            sub_q_wav = [q_wavs[i] for i in sub]

            if c_texts:
                sub_c_text = [c_texts[i] for i in sub]
                sub_c_img = [c_imgs[i] for i in sub]
                sub_c_vid = [c_vids[i] for i in sub]
                sub_c_wav = [c_wavs[i] for i in sub]

            q_proc = self._process_group_eval(sub_q_text, sub_q_img, sub_q_vid, sub_q_wav, max_len)
            if c_texts:
                c_proc = self._process_group_eval(sub_c_text, sub_c_img, sub_c_vid, sub_c_wav, max_len)
            else:
                c_proc = None

            qry_outs.append(q_proc)
            if c_proc:
                cand_outs.append(c_proc)
            order_chunks.append(sub)

        # merge
        qry_batch = _merge(qry_outs)
        if cand_outs:
            cand_batch = _merge(cand_outs)
        else:
            cand_batch = {}

        # attach metadata
        qry_batch["valid_example_mask"] = torch.tensor(valid, dtype=torch.bool)
        if cand_batch:
            cand_batch["valid_example_mask"] = torch.tensor(valid, dtype=torch.bool)

        # for debug / hash you can still attach raw texts if you want
        qry_batch["text"] = q_texts
        if c_texts:
            cand_batch["text"] = c_texts

        # Return format depends on whether we have candidates
        if cand_batch:
            return qry_batch, cand_batch
        else:
            return qry_batch
