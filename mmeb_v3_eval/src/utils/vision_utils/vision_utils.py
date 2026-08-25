import os
import re
import subprocess
import math

import numpy as np
import pandas as pd
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
try:
    from torchvision.io import write_video
except ImportError:  # torchvision >=0.26 removed video IO; not needed for text/image eval
    def write_video(*args, **kwargs):
        raise RuntimeError("torchvision.io.write_video is unavailable in this torchvision version")
from torchvision.utils import save_image

from src.utils.vision_utils import video_transforms


VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def qa_template(question, candidates, answer):
    question = f"{question}\n"
    question += "Options:\n"
    answer_idx = -1
    options = []
    for idx, c in enumerate(candidates):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        options.append(f"({chr(ord('A') + idx)}) {c}")
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"
    return question, options, answer, answer_idx


def is_url(url):
    return re.match(regex, url) is not None


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def download_url(input_path):
    output_dir = "cache"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    img_data = requests.get(input_path).content
    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def temporal_random_crop(vframes, num_frames, frame_interval):
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert end_frame_ind - start_frame_ind >= num_frames
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indice]
    return video


def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video


def get_transforms_image(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "Image size must be square for center crop"
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform


def read_image_from_path(path, transform=None, transform_name="center", num_frames=1, image_size=(256, 256)):
    image = pil_loader(path)
    if transform is None:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
    image = transform(image)
    video = image.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    video = video.permute(1, 0, 2, 3)
    return video


def read_video_from_path(path, transform=None, transform_name="center", image_size=(256, 256)):
    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
    if transform is None:
        transform = get_transforms_video(image_size=image_size, name=transform_name)
    video = transform(vframes)  # T C H W
    video = video.permute(1, 0, 2, 3)
    return video


def read_from_path(path, image_size, transform_name="center"):
    if is_url(path):
        path = download_url(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext.lower() in VID_EXTENSIONS:
        return read_video_from_path(path, image_size=image_size, transform_name=transform_name)
    else:
        assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
        return read_image_from_path(path, image_size=image_size, transform_name=transform_name)


def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1, 1), force_video=False):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))
        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
    return save_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def resize_crop_to_fill(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i: i + th, j: j + tw])


# adopted from LLaVA-Hound-DPO
def get_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image


def load_frames(frames_dir, filter_func=None):
    """
    Load image frames from a directory, with an optional filter function.
    """
    def natural_sort_key(filename):
        """Extract numbers from filenames for correct sorting."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

    results = []
    if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
        return []
    frame_names = os.listdir(frames_dir)
    frame_names = sorted(frame_names, key=natural_sort_key)
    for frame_name in frame_names:
        ext = os.path.splitext(frame_name)[-1].lower()
        if ext.lower() in IMAGE_EXTENSIONS:
            if filter_func is None or filter_func(frame_name):
                image_path = f"{frames_dir}/{frame_name}"
                results.append(image_path)
    return results


def sample_frames(frames, num_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration-1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
        except Exception:
            break
        sampled_frames.append(single_frame_path)
    # If total frame numbers is less than num_segments, append the last images to achieve
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames


def process_video_frames(frame_dir, num_frames=None):
    """
    Load and sample frames as input into the model.

    Sampling policy: when the directory holds at most num_frames frames, all
    frames are kept in natural order; otherwise num_frames frames are selected
    with round-uniform indices over the full range (first and last included).
    """
    if num_frames == 0:
        return []
    frames = load_frames(frame_dir)
    if num_frames is None or not frames:
        return frames
    if len(frames) <= num_frames:
        return frames
    indices = [round(i * (len(frames) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [frames[i] for i in indices]


def save_frames(video_path, frame_dir, max_frames_saved, file_name_prefix=''):
    """
    Saves frames if only raw video is available.
    """
    if (not os.path.exists(frame_dir)
            or not any(f.lower().endswith(tuple(IMAGE_EXTENSIONS)) for f in os.listdir(frame_dir))):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File {video_path} does not exist")
        os.makedirs(frame_dir, exist_ok=True)
        total_frames = get_total_frames(video_path)
        if total_frames == 0:
            print("No frames found in video.")
            return
        if total_frames <= max_frames_saved:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / max_frames_saved
            frame_indices = [math.floor(i * step) for i in range(max_frames_saved)]
        select_expr = "+".join([f"eq(n\\,{i})" for i in frame_indices])
        output_pattern = os.path.join(frame_dir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", video_path,
            "-vf", f"select='{select_expr}'",
            "-vsync", "vfr",  # ensure only selected frames are output
            output_pattern
        ]
        subprocess.run(cmd, check=True)


def get_total_frames(video_path):
    """Use ffprobe to get total frame count."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip()

    if output.isdigit():
        return int(output)

    # Fallback: use nb_frames (less reliable but often available)
    fallback_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
    fallback_output = fallback_result.stdout.strip()

    if fallback_output.isdigit():
        return int(fallback_output)

    raise ValueError(
        f"Could not determine total frames for {video_path}. Outputs: '{output}', fallback: '{fallback_output}'")
