import logging
import os
from dataclasses import dataclass
from typing import Dict, Any
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments
import torch
try:
    from qwen_vl_utils import smart_resize
except ImportError:
    from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import smart_resize
from PIL import Image
from src.model.processor import (
    QWEN2_VL,
    QWEN2_5_VL,
    QWEN3_VL,
    QWEN2_VL_TOKENSELECTION,
    QWEN2_5_VL_TOKENSELECTION,
    QWEN2_5_OMNI,
    NVOMNIEMBED,
    WAVE,
    E5_OMNI,
    JINA_OMNI,
    LCO_OMNI,
    process_vlm_inputs_fns,
)

import io

logger = logging.getLogger(__name__)
PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000


@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        examples = {'text': [e[0] for e in examples], 'images': [e[1] for e in examples]}
        inputs = process_vlm_inputs_fns[self.model_args.model_backbone](examples,
                                                                        processor=self.processor,
                                                                        max_length=self.data_args.max_len)
        inputs['texts'] = examples['text']
        inputs['images'] = examples['images']
        inputs['image_paths'] = [i.filename if hasattr(i, 'filename') else None for i in examples['images']]
        return inputs


@dataclass
class CLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(images=image, return_tensors="pt")
                image_exist = True
                pixel_values.append(image_inputs['pixel_values'])
            if text:
                text_exist = True
            text_inputs = self.txt_processors(
                text, padding=getattr(self.data_args, "padding", True),
                max_length=self.data_args.max_len, truncation=True, return_tensors="pt")
            input_ids.append(text_inputs["input_ids"].squeeze(0))
        if text_exist:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.txt_processors.pad_token_id
            )
            attention_mask = input_ids.ne(self.txt_processors.pad_token_id)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0] == pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs


@dataclass
class OpenCLIPCollator:
    data_args: DataArguments
    vis_processors: AutoProcessor
    txt_processors: AutoTokenizer

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, attention_mask = [], [], []
        image_exist, text_exist = False, False
        for example in examples:
            text, image = example
            if image is not None:
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_inputs = self.vis_processors(image).unsqueeze(0)
                image_exist = True
                pixel_values.append(image_inputs)
            if text:
                text_exist = True
            text_inputs = self.txt_processors(text)
            input_ids.append(text_inputs)
        if text_exist:
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = input_ids.ne(0)
        if image_exist:
            pixel_values = torch.cat(pixel_values, dim=0)
        if text_exist and image_exist:
            assert input_ids.size()[0] == pixel_values.size()[0]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }

        return inputs


@dataclass
class MultimodalEvalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    encode_side: str

    def _load_image_from_dict(self, raw_images: Dict[str, Any]):
        """raw_images is ImageVideoInstance.to_dict() output"""
        visual_input = []
        assert "resolutions" in raw_images, "need raw_images['resolutions'] to determine num images"

        num_images = len(raw_images["paths"])
        for image_idx in range(num_images):
            b = raw_images["bytes"][image_idx] if "bytes" in raw_images else None
            p = raw_images["paths"][image_idx] if "paths" in raw_images else None
            image_resolution = raw_images["resolutions"][image_idx] if "resolutions" in raw_images else None
            if isinstance(p, str) and p.strip() == "":
                p = None

            if b is None and p is None:
                image = None
            else:
                # Robustness: a corrupted/truncated image file should not hang or
                # crash the whole eval; fall back to a blank image so that sample
                # indices stay aligned with the dataset.
                try:
                    if b is not None:
                        image = Image.open(io.BytesIO(b)).convert("RGB")
                    elif os.path.exists(p):
                        image = Image.open(p).convert("RGB")
                    else:
                        # Compatibility fallback: some GUI samples may reference
                        # missing local files; treat as no-image samples.
                        image = None
                except Exception as exc:
                    print(f"[warn] failed to load image (path={p}), using blank fallback: {exc}")
                    image = Image.new("RGB", (32, 32))

            if (not self.data_args.resize_use_processor) and image is not None and image_resolution:
                image = image.resize(image_resolution)

            if image is not None and (image_resolution is None and self.data_args.image_decay_factor is not None):
                assert self.model_args.model_backbone in [
                    QWEN2_VL, QWEN2_5_VL, QWEN3_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], \
                    "image_decay_factor is only supported for Qwen models"
                max_pixels = max(
                    self.data_args.resize_min_pixels,
                    self.data_args.resize_max_pixels * (self.data_args.image_decay_factor ** (num_images - image_idx))
                )
                width, height = image.size
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    min_pixels=self.data_args.resize_min_pixels,
                    max_pixels=max_pixels,
                )
                image = image.resize((resized_width, resized_height))

            visual_input.append(image)

        return visual_input

    def _get_batch_inputs(self, batch, text_keyname, visual_keyname, audio_keyname, visual_mode: str):
        """
        visual_mode:
          - "image": passed to processor(images=...)
          - "video": passed to processor(videos=...)
        """
        texts, visuals, audios = [], [], []

        for example in batch:
            if example is None or not example:
                # empty fallback
                texts.append("  ")
                visuals.append(None)
                audios.append(None)
                continue

            ex_text = example[text_keyname]
            ex_visual = example[visual_keyname]
            ex_audio = example.get(audio_keyname, None)

            # ex_text: list[str]
            if isinstance(ex_visual, dict):
                for t in ex_text:
                    visual_input = self._load_image_from_dict(ex_visual)  # List[PIL] or []
                    texts.append(t)
                    visuals.append(visual_input if visual_input else None)
                    audios.append(ex_audio)

            elif isinstance(ex_visual, list) and len(ex_visual) > 0 and isinstance(ex_visual[0], dict):
                for t, raw_images in zip(ex_text, ex_visual):
                    visual_input = self._load_image_from_dict(raw_images)  # List[PIL]
                    texts.append(t)
                    visuals.append(visual_input if visual_input else None)
                    audios.append(ex_audio)

            elif visual_mode == "video":
                for t, video_frames in zip(ex_text, ex_visual):
                    texts.append(t)
                    # Downsample video frames to avoid excessive tokens/OOM.
                    if isinstance(video_frames, list):
                        sample_n_frames = int(self.data_args.video_max_frames or 8)
                        if sample_n_frames > 0 and len(video_frames) > sample_n_frames:
                            if sample_n_frames == 1:
                                video_frames = [video_frames[0]]
                            else:
                                idxs = [
                                    round(i * (len(video_frames) - 1) / (sample_n_frames - 1))
                                    for i in range(sample_n_frames)
                                ]
                                video_frames = [video_frames[i] for i in idxs]
                        frame_size = getattr(self.data_args, "video_frame_size", None)
                        if frame_size:
                            video_frames = [
                                (f.resize((frame_size, frame_size), resample=Image.BICUBIC) if f is not None else None)
                                for f in video_frames
                            ]
                    visuals.append(video_frames)  # List[PIL.Image]
                    audios.append(ex_audio)

            else:
                for t, visual_input in zip(ex_text, ex_visual):
                    texts.append(t)
                    if visual_input is None:
                        visuals.append(None)
                    elif isinstance(visual_input, list):
                        visuals.append(visual_input)          # already List[PIL]
                    else:
                        visuals.append([visual_input])        # ✅ PIL -> [PIL]
                    audios.append(ex_audio)

        inputs = {"text": texts}
        if visual_mode == "video":
            inputs["videos"] = visuals
        else:
            inputs["images"] = visuals

        if any(a is not None for a in audios):
            inputs["audios"] = audios
        return inputs

    def _tensor_only(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _omni_process_batch(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Use Omni/NVOmni process fn directly with the inputs, since grouping is now handled in processor.py
        """
        # Validate inputs before processing
        if not inputs.get("text"):
            # Fallback for empty batch
            return {"input_ids": torch.empty(0, 0, dtype=torch.long),
                    "attention_mask": torch.empty(0, 0, dtype=torch.long)}

        try:
            # Call the processor directly (grouping is handled inside processor now)
            from src.model.processor import (
                Omni_process_fn,
                NVOmni_process_fn,
                E5Omni_process_fn,
                LCOOmni_process_fn,
                NVOMNIEMBED,
                QWEN2_5_OMNI,
                WAVE,
                E5_OMNI,
                JINA_OMNI,
                LCO_OMNI,
            )
            # Keep eval behavior consistent with training: use apply_chat_template
            # + process_mm_info for qwen/omni-family models.
            if self.model_args.model_backbone == E5_OMNI:
                process_fn = E5Omni_process_fn
            elif self.model_args.model_backbone == LCO_OMNI:
                process_fn = LCOOmni_process_fn
            elif self.model_args.model_backbone in {NVOMNIEMBED, QWEN2_5_OMNI, WAVE, JINA_OMNI}:
                process_fn = NVOmni_process_fn
            else:
                process_fn = Omni_process_fn
            omni_inputs = dict(inputs)
            # input_raw_wav is only needed by WAVE official BEATs branch.
            omni_inputs["_keep_input_raw_wav"] = (self.model_args.model_backbone == WAVE)
            # Let omni processor follow DataArguments image resize knobs.
            omni_inputs["_resize_min_pixels"] = getattr(self.data_args, "resize_min_pixels", None)
            omni_inputs["_resize_max_pixels"] = getattr(self.data_args, "resize_max_pixels", None)
            outputs = process_fn(
                model_inputs=omni_inputs,
                processor=self.processor,
                max_length=self.data_args.max_len
            )
            return self._tensor_only(outputs)
        except Exception as e:
            print(f"Error in _omni_process_batch: {e}")
            print(f"Inputs keys: {list(inputs.keys())}")
            print(f"Text length: {len(inputs.get('text', []))}")
            print(f"Images length: {len(inputs.get('images', [])) if inputs.get('images') else 'None'}")
            print(f"Audios length: {len(inputs.get('audios', [])) if inputs.get('audios') else 'None'}")
            raise

    def __call__(self, examples):
        """
        examples: dict with keys:
          - query_text/query_image/query_audio
          - cand_text/cand_image or cand_video/cand_audio
        """
        use_omni = self.model_args.model_backbone in {QWEN2_5_OMNI, NVOMNIEMBED, WAVE, E5_OMNI, JINA_OMNI, LCO_OMNI}
        # Ensure qwen omni warning filters are active in dataloader worker processes too.
        if use_omni and not getattr(self, "_qwen_warning_filter_ready", False):
            try:
                from src.model.processor import _install_qwen_omni_warning_filters
                _install_qwen_omni_warning_filters()
            except Exception:
                pass
            self._qwen_warning_filter_ready = True

        if self.encode_side == "qry":
            inputs = self._get_batch_inputs(
                examples,
                text_keyname="query_text",
                visual_keyname="query_image",
                audio_keyname="query_audio",
                visual_mode="image",
            )
        else:
            if "cand_video" in examples[0]:
                inputs = self._get_batch_inputs(
                    examples,
                    text_keyname="cand_text",
                    visual_keyname="cand_video",
                    audio_keyname="cand_audio",
                    visual_mode="video",
                )
            else:
                inputs = self._get_batch_inputs(
                    examples,
                    text_keyname="cand_text",
                    visual_keyname="cand_image",
                    audio_keyname="cand_audio",
                    visual_mode="image",
                )

        if "audios" in inputs:
            # Imported here so that text/image/video evaluation needs no audio stack.
            # torch 2.11 torchaudio lacks .info and can't encode/decode BytesIO
            # (torchcodec); use soundfile for decoding.
            import torchaudio
            import soundfile as sf

            target_sr = getattr(self.data_args, "audio_sample_rate", 16000) or 16000
            min_audio_samples = getattr(self.data_args, "audio_min_samples", None)
            if min_audio_samples is None:
                min_audio_samples = int(target_sr * 0.025)  # 25ms

            max_audio_seconds = getattr(self.data_args, "audio_max_seconds", None)
            if max_audio_seconds is not None:
                max_audio_samples = int(float(max_audio_seconds) * target_sr)
            else:
                max_audio_samples = getattr(self.data_args, "audio_max_samples", None)
                if max_audio_samples is None:
                    max_audio_frames = int(getattr(self.data_args, "audio_max_frames", 1024))
                    max_audio_samples = max_audio_frames * 160
                else:
                    max_audio_samples = int(max_audio_samples)

            eval_crop = getattr(self.data_args, "eval_crop", "head")

            def _crop_audio(wav: torch.Tensor) -> torch.Tensor | None:
                if wav is None:
                    return None
                if wav.numel() < min_audio_samples:
                    return None
                if wav.numel() > max_audio_samples:
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
                return wav

            def _safe_resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
                if wav is None or wav.numel() == 0 or src_sr == dst_sr:
                    return wav
                return torchaudio.functional.resample(wav, src_sr, dst_sr)

            audio_tensors = []
            for audio_item in inputs["audios"]:
                if audio_item is None:
                    audio_tensors.append(None)
                    continue

                # 0) datasets torchcodec AudioDecoder (Audio feature, decode=True): grab the
                # raw encoded bytes/path and fall through to soundfile decoding (avoids torchcodec).
                if type(audio_item).__name__ == "AudioDecoder":
                    enc = getattr(audio_item, "_hf_encoded", None)
                    if isinstance(enc, dict) and (enc.get("bytes") or enc.get("path")):
                        audio_item = {"path": enc.get("path"), "bytes": enc.get("bytes")}
                    else:
                        _s = audio_item.get_all_samples()
                        wav = _s.data
                        if wav.dim() > 1:
                            wav = wav.mean(0)
                        wav = _safe_resample(wav, _s.sample_rate, target_sr)
                        audio_tensors.append(_crop_audio(wav))
                        continue

                # 1) HF decoded audio dict: {"array":..., "sampling_rate":...}
                if isinstance(audio_item, dict) and ("array" in audio_item):
                    arr = audio_item["array"]
                    sr = int(audio_item.get("sampling_rate", target_sr))
                    wav = torch.tensor(arr, dtype=torch.float32)
                    if wav.ndim > 1:
                        wav = wav.mean(0)
                    wav = _safe_resample(wav, sr, target_sr)
                    audio_tensors.append(_crop_audio(wav))
                    continue

                # 2) Tensor waveform
                if isinstance(audio_item, torch.Tensor):
                    wav = audio_item.float()
                    if wav.ndim > 1:
                        wav = wav.mean(0)
                    audio_tensors.append(_crop_audio(wav))
                    continue

                # 3) dict path/bytes (+ optional start/end)
                if isinstance(audio_item, dict):
                    a_path = audio_item.get("path") or audio_item.get("audio_path") or audio_item.get("video_path")
                    if (
                        a_path is not None
                        and not os.path.isabs(a_path)
                        and getattr(self.data_args, "data_basedir", None) is not None
                    ):
                        a_path = os.path.join(self.data_args.data_basedir, a_path)
                    a_bytes = audio_item.get("bytes", None)
                    start_t = float(audio_item.get("start", 0.0))
                    end_t = audio_item.get("end", None)

                    if a_bytes is not None:
                        _d, sr = sf.read(io.BytesIO(a_bytes), dtype="float32")
                        wave = torch.from_numpy(_d.mean(axis=1) if _d.ndim > 1 else _d).float()
                    elif a_path:
                        _si = sf.info(a_path)
                        sr = _si.samplerate
                        total_frames = int(getattr(_si, "frames", 0) or 0)
                        frame_offset = max(0, int(start_t * sr))
                        if total_frames > 0:
                            frame_offset = min(frame_offset, max(0, total_frames - 1))

                        if end_t is not None:
                            seg_frames = int((float(end_t) - start_t) * sr)
                            num_frames = max(1, seg_frames)
                            if total_frames > 0:
                                remain = max(1, total_frames - frame_offset)
                                num_frames = min(num_frames, remain)
                        else:
                            num_frames = -1

                        _d, sr = sf.read(a_path, start=frame_offset,
                                         frames=(num_frames if num_frames and num_frames > 0 else -1),
                                         dtype="float32")
                        wave = torch.from_numpy(_d.mean(axis=1) if _d.ndim > 1 else _d).float()
                    else:
                        raise ValueError("audio item missing array/path/bytes")

                    if wave.dim() > 1:
                        wave = wave.mean(0)
                    wave = _safe_resample(wave, sr, target_sr)
                    audio_tensors.append(_crop_audio(wave))
                    continue

                raise ValueError(f"Unsupported audio item type: {type(audio_item)}")

            inputs["audios"] = audio_tensors
            inputs["audio_sample_rate"] = target_sr

        if use_omni:
            processed_inputs = self._omni_process_batch(inputs)
        else:
            process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
            processed_inputs = process_fn(inputs, processor=self.processor, max_length=self.data_args.max_len)
        dataset_infos = [e["dataset_infos"] for e in examples]
        return processed_inputs, dataset_infos
