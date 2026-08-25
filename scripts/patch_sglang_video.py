#!/usr/bin/env python3

import importlib.metadata
import py_compile
import shutil
from pathlib import Path

import sglang


EXPECTED_VERSION = "0.5.9"
REPLACEMENTS = (
    (
        "IMAGE_FACTOR = 28",
        "IMAGE_FACTOR = 32  # patch_size=16 * spatial_merge_size=2",
    ),
    (
        "    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)",
        "    idx = torch.linspace(0, total_frames - 1, nframes).round().long().cpu().numpy()",
    ),
    (
        """    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )
""",
        """    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
""",
    ),
)


def main() -> None:
    version = importlib.metadata.version("sglang")
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Expected sglang=={EXPECTED_VERSION}, found {version}")

    path = (
        Path(sglang.__file__).resolve().parent
        / "srt"
        / "multimodal"
        / "processors"
        / "qwen_vl.py"
    )
    text = path.read_text(encoding="utf-8")
    if all(new in text for _, new in REPLACEMENTS):
        print("SGLang video preprocessing is ready.")
        return

    updated = text
    for old, new in REPLACEMENTS:
        if updated.count(old) != 1:
            raise RuntimeError(f"Unexpected SGLang source: {old.splitlines()[0]}")
        updated = updated.replace(old, new, 1)

    backup = path.with_suffix(path.suffix + ".original")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print("SGLang video preprocessing is ready.")


if __name__ == "__main__":
    main()
