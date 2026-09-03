#!/usr/bin/env python3
import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path


RAW_LAYOUT = {
    "image_root": "image_tasks",
    "audio_root": "audio_tasks",
    "gui_root": "gui_tasks",
    "memory_root": "memory_tasks",
    "text_root": "text_tasks",
    "tool_root": "tool_tasks",
    "video_root": "video_tasks",
    "visdoc_root": "visdoc_tasks",
    "omniset_archive": "omniset.tar.gz",
}


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as tar:
        for member in tar.getmembers():
            member_path = dest_dir / member.name
            if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(dest_dir)


def safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            member_path = dest_dir / member.filename
            if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip archive: {member.filename}")
        zf.extractall(dest_dir)


def copytree_if_missing(src: Path, dst: Path) -> None:
    if not src.exists() or _done_marker(dst).exists():
        return
    if dst.exists():
        print(f"[clean partial] {dst}")
        _remove_path(dst)
    shutil.copytree(src, dst)
    _mark_done(dst)


def materialize_tree_if_missing(src: Path, dst: Path) -> None:
    if _done_marker(dst).exists():
        print(f"[skip] {dst}")
        return
    if not src.exists():
        if dst.exists():
            print(f"[skip existing] {dst} (source not found: {src})")
            _mark_done(dst)
            return
        raise FileNotFoundError(f"Source directory not found: {src}")
    if dst.exists():
        print(f"[skip existing] {dst}")
        _mark_done(dst)
        return
    print(f"[copy] {src} -> {dst}")
    shutil.copytree(src, dst)
    _mark_done(dst)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _done_marker(path: Path) -> Path:
    return path / ".done"


def _mark_done(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _done_marker(path).touch()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def extract_if_missing(archive_path: Path, target_path: Path, dest_dir: Path) -> None:
    if _done_marker(target_path).exists():
        print(f"[skip] {target_path}")
        return
    if not archive_path.exists():
        if target_path.exists():
            print(f"[skip existing] {target_path} (archive not found: {archive_path})")
            _mark_done(target_path)
            return
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if target_path.exists():
        print(f"[clean partial] {target_path}")
        _remove_path(target_path)
    print(f"[extract] {archive_path} -> {dest_dir}")
    safe_extract_tar(archive_path, dest_dir)
    if not target_path.exists():
        raise FileNotFoundError(f"Expected extracted path not found: {target_path}")
    _mark_done(target_path)


def extract_zip_if_missing(archive_path: Path, target_path: Path, dest_dir: Path,
                           lowercase_name: str | None = None) -> None:
    if _done_marker(target_path).exists():
        print(f"[skip] {target_path}")
        return
    if target_path.exists():
        print(f"[clean partial] {target_path}")
        _remove_path(target_path)
    if lowercase_name:
        lowercase_dir = dest_dir / lowercase_name
        if lowercase_dir.exists():
            print(f"[clean partial] {lowercase_dir}")
            _remove_path(lowercase_dir)
    print(f"[extract] {archive_path} -> {dest_dir}")
    safe_extract_zip(archive_path, dest_dir)
    if lowercase_name:
        lowercase_dir = dest_dir / lowercase_name
        if lowercase_dir.exists() and not target_path.exists():
            lowercase_dir.rename(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Expected extracted path not found: {target_path}")
    _mark_done(target_path)


def prepare_image_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["image_root"]
    out_root = root / "image-tasks"
    ensure_dir(out_root)

    extract_if_missing(raw_root / "mmeb_v1.tar.gz", out_root / "MMEB", out_root)
    extract_if_missing(raw_root / "MCMR.tar.gz", out_root / "MCMR", out_root)


def prepare_audio_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["audio_root"]
    out_root = root / "audio-tasks"
    ensure_dir(out_root)

    archives = sorted(raw_root.glob("*.tar"))
    for archive_path in archives:
        with tarfile.open(archive_path, mode="r:*") as tar:
            top_level = tar.getmembers()[0].name.split("/")[0]
        extract_if_missing(archive_path, out_root / top_level, out_root)


def prepare_gui_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["gui_root"]
    out_root = root / "gui-tasks"
    ensure_dir(out_root)

    copytree_if_missing(raw_root / "GAE-GUIAct", out_root / "GAE-GUIAct")
    copytree_if_missing(raw_root / "GAE-Mind2Web", out_root / "GAE-Mind2Web")

    gui_zip = raw_root / "GUIAct.zip"
    # GUIAct.zip contains a top-level guiact/ directory, and the parquet metadata
    # references images as "guiact/<file>.png" relative to image_root=gui-tasks/GUIAct/.
    # Extract into GUIAct/ so the final layout is gui-tasks/GUIAct/guiact/<file>.png.
    extract_zip_if_missing(gui_zip, out_root / "GUIAct" / "guiact", out_root / "GUIAct")

    mind2web_zip = raw_root / "Mind2Web.zip"
    # Same nested layout: gui-tasks/Mind2Web/mind2web/<file>.png.
    extract_zip_if_missing(mind2web_zip, out_root / "Mind2Web" / "mind2web", out_root / "Mind2Web")


# The eval config expects memory-tasks/<Category>/<Dataset>. Some releases of
# memory-tasks.tar ship a flat layout (memory-tasks/<Dataset>); create the
# category-level symlinks when they are missing.
MEMORY_CATEGORY_DIRS = {
    "Episodic": ["KnowMeBench"],
    "Dialogue": ["REALTALK"],
    "Semantic": ["PeerQA"],
    "Procedural": ["DeepPlanning"],
}


def prepare_memory_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["memory_root"]
    extract_if_missing(raw_root / "memory-tasks.tar", root / "memory-tasks", root)
    memory_root = root / "memory-tasks"
    for category, datasets in MEMORY_CATEGORY_DIRS.items():
        for dataset in datasets:
            link = memory_root / category / dataset
            target = memory_root / dataset
            if link.exists() or not target.exists():
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(target, link.parent), link)


def prepare_text_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["text_root"]
    extract_if_missing(raw_root / "text_tasks.tar.part-000", root / "text-tasks", root)


def prepare_tool_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["tool_root"]
    extract_if_missing(raw_root / "tool_tasks.tar.gz", root / "tool-tasks", root)


def prepare_visdoc_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["visdoc_root"]
    out_root = root / "visdoc-tasks"
    ensure_dir(out_root)

    # Preferred HF layout: visdoc_tasks/*.tar.gz -> visdoc-tasks/{images,data}.
    # Fallback keeps compatibility with earlier local layouts that stored archives
    # directly under visdoc-tasks/.
    images_archive = raw_root / "visdoc-tasks.images.tar.gz"
    if not images_archive.exists():
        images_archive = out_root / "visdoc-tasks.images.tar.gz"
    data_archive = raw_root / "visdoc-tasks.data.tar.gz"
    if not data_archive.exists():
        data_archive = out_root / "visdoc-tasks.data.tar.gz"

    extract_if_missing(images_archive, out_root / "images", out_root)
    extract_if_missing(data_archive, out_root / "data", out_root)


def _extract_video_tar(archive_path: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    with tarfile.open(archive_path, mode="r:*") as tar:
        for member in tar.getmembers():
            member_path = dest_dir / member.name
            if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(dest_dir)


def _extract_video_parts(parts: list[Path], dest_dir: Path) -> None:
    ensure_dir(dest_dir)

    class _ConcatReader:
        def __init__(self, file_paths: list[Path]):
            self._paths = file_paths
            self._handles = [p.open("rb") for p in file_paths]
            self._index = 0

        def read(self, size: int = -1) -> bytes:
            if size == 0:
                return b""
            chunks = []
            remaining = size
            while self._index < len(self._handles):
                handle = self._handles[self._index]
                chunk = handle.read() if size < 0 else handle.read(remaining)
                if chunk:
                    chunks.append(chunk)
                    if size < 0:
                        continue
                    remaining -= len(chunk)
                    if remaining == 0:
                        break
                else:
                    handle.close()
                    self._index += 1
            return b"".join(chunks)

        def close(self) -> None:
            for handle in self._handles:
                try:
                    handle.close()
                except Exception:
                    pass

    reader = _ConcatReader(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                member_path = dest_dir / member.name
                if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
                    raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
                tar.extract(member, dest_dir)
    finally:
        reader.close()


def _preferred_or_legacy_archive(preferred: Path, legacy: Path) -> Path:
    return preferred if preferred.exists() else legacy


def _preferred_or_legacy_parts(preferred_glob: list[Path], legacy_glob: list[Path]) -> list[Path]:
    return preferred_glob if preferred_glob else legacy_glob


def prepare_video_tasks(root: Path) -> None:
    raw_root = root / RAW_LAYOUT["video_root"]
    raw_frames_root = raw_root / "frames"
    out_root = root / "video-tasks"
    out_frames_root = out_root / "frames"
    ensure_dir(out_frames_root)

    # Preferred HF layout: video_tasks/data and video_tasks/frames/*.tar.gz.
    # Evaluation code reads video-tasks/data and video-tasks/frames/{video_*}.
    materialize_tree_if_missing(raw_root / "data", out_root / "data")

    single_archives = {
        "video_cls": _preferred_or_legacy_archive(
            raw_frames_root / "video_cls.tar.gz",
            out_frames_root / "video_cls.tar.gz",
        ),
        "video_ret": _preferred_or_legacy_archive(
            raw_frames_root / "video_ret.tar.gz",
            out_frames_root / "video_ret.tar.gz",
        ),
    }
    for task_name, archive_path in single_archives.items():
        dest_dir = out_frames_root / task_name
        if _done_marker(dest_dir).exists():
            print(f"[skip] {dest_dir}")
            continue
        if not archive_path.exists():
            if dest_dir.exists():
                print(f"[skip existing] {dest_dir} (archive not found: {archive_path})")
                _mark_done(dest_dir)
                continue
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        if dest_dir.exists():
            print(f"[clean partial] {dest_dir}")
            _remove_path(dest_dir)
        print(f"[extract] {archive_path} -> {dest_dir}")
        _extract_video_tar(archive_path, dest_dir)
        _mark_done(dest_dir)

    split_archives = {
        "video_mret": _preferred_or_legacy_parts(
            sorted(raw_frames_root.glob("video_mret.tar.gz-*")),
            sorted(out_frames_root.glob("video_mret.tar.gz-*")),
        ),
        "video_qa": _preferred_or_legacy_parts(
            sorted(raw_frames_root.glob("video_qa.tar.gz-*")),
            sorted(out_frames_root.glob("video_qa.tar.gz-*")),
        ),
    }
    for task_name, parts in split_archives.items():
        dest_dir = out_frames_root / task_name
        if _done_marker(dest_dir).exists():
            print(f"[skip] {dest_dir}")
            continue
        if not parts:
            if dest_dir.exists():
                print(f"[skip existing] {dest_dir} (split archives not found)")
                _mark_done(dest_dir)
                continue
            raise FileNotFoundError(f"Missing split archive parts for {task_name}")
        if dest_dir.exists():
            print(f"[clean partial] {dest_dir}")
            _remove_path(dest_dir)
        print(f"[extract] {task_name} parts -> {dest_dir}")
        _extract_video_parts(parts, dest_dir)
        _mark_done(dest_dir)


def _omniset_ready(out_root: Path) -> bool:
    required_paths = [
        out_root / "omniset.jsonl",
        out_root / "catalog.jsonl",
        out_root / "val2014",
        out_root / "videos",
        out_root / "audios",
        out_root / "frames_omni",
    ]
    return all(path.exists() for path in required_paths)


def prepare_omniset(root: Path) -> None:
    archive_path = root / RAW_LAYOUT["omniset_archive"]
    out_root = root / "omniset"

    if _done_marker(out_root).exists() and _omniset_ready(out_root):
        print(f"[skip] {out_root}")
        return
    if _omniset_ready(out_root):
        print(f"[skip existing] {out_root}")
        _mark_done(out_root)
        return
    if not archive_path.exists():
        # OmniSET (cross-modality/audio) is optional: none of the standard
        # MMEB-V3 eval configs need it, so a missing archive is not fatal.
        print(f"[skip] OmniSET archive not found (optional): {archive_path}")
        return
    if out_root.exists():
        print(f"[clean partial] {out_root}")
        _remove_path(out_root)

    print(f"[extract] {archive_path} -> {root}")
    safe_extract_tar(archive_path, root)
    if not _omniset_ready(out_root):
        raise FileNotFoundError(f"Expected OmniSET layout not found under: {out_root}")
    _mark_done(out_root)


def materialize_image_query(root: Path, image_query_source: Path | None, image_query_mode: str) -> None:
    dst = root / "image-query"
    if dst.exists():
        print(f"[skip] {dst}")
        return
    if image_query_source is None:
        print("[warn] image-query is still missing. Pass --image-query-source to wire it into the local layout.")
        return

    if not image_query_source.exists():
        raise FileNotFoundError(f"image-query source not found: {image_query_source}")

    if image_query_mode == "symlink":
        print(f"[symlink] {dst} -> {image_query_source}")
        os.symlink(image_query_source, dst)
    else:
        print(f"[copy] {image_query_source} -> {dst}")
        shutil.copytree(image_query_source, dst)


def check_layout(root: Path) -> list[str]:
    issues = []

    required_ready_paths = [
        root / "image-tasks" / "MMEB",
        root / "image-tasks" / "MCMR",
        root / "audio-tasks",
        root / "gui-tasks" / "GUIAct",
        root / "gui-tasks" / "Mind2Web",
        root / "gui-tasks" / "GAE-GUIAct",
        root / "gui-tasks" / "GAE-Mind2Web",
        root / "memory-tasks",
        root / "text-tasks",
        root / "tool-tasks",
        root / "visdoc-tasks" / "images",
        root / "video-tasks" / "frames" / "video_cls",
        root / "video-tasks" / "frames" / "video_ret",
        root / "video-tasks" / "frames" / "video_mret",
        root / "video-tasks" / "frames" / "video_qa",
        root / "omniset" / "omniset.jsonl",
        root / "omniset" / "catalog.jsonl",
        root / "omniset" / "val2014",
        root / "omniset" / "videos",
        root / "omniset" / "audios",
        root / "omniset" / "frames_omni",
        root / "image-query",
    ]
    for path in required_ready_paths:
        if not path.exists():
            issues.append(f"missing: {path}")

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local MMEB-V3 eval-ready directories from raw archives.")
    parser.add_argument("--root", type=Path, required=True, help="Path to the MMEB-V3 raw dataset root.")
    parser.add_argument(
        "--image-query-source",
        type=Path,
        default=None,
        help="Optional local path for image-query metadata (for example, a local clone/download of"
             " ziyjiang/MMEB_Test_Instruct).",
    )
    parser.add_argument(
        "--image-query-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to attach image-query into the MMEB-V3 root when --image-query-source is given.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report missing eval-ready directories without extracting anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"MMEB-V3 root not found: {root}")

    if args.check_only:
        issues = check_layout(root)
        if not issues:
            print("[ok] MMEB-V3 layout looks complete.")
            return
        print("[check] MMEB-V3 is still missing:")
        for issue in issues:
            print(f"  - {issue}")
        return

    prepare_image_tasks(root)
    prepare_audio_tasks(root)
    prepare_gui_tasks(root)
    prepare_memory_tasks(root)
    prepare_text_tasks(root)
    prepare_tool_tasks(root)
    prepare_visdoc_tasks(root)
    prepare_video_tasks(root)
    prepare_omniset(root)
    materialize_image_query(root, args.image_query_source, args.image_query_mode)

    issues = check_layout(root)
    if issues:
        print("[done] Extraction finished, but some eval-ready pieces are still missing:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[done] MMEB-V3 local layout is ready.")


if __name__ == "__main__":
    main()
