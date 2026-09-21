"""Regression tests for the GUI trajectory dataset parser.

Run with:  python -m unittest discover -s mmeb_v3_eval/tests
"""

import importlib.util
import os
import sys
import types
import unittest

EVAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)


def _load_gui_dataset():
    """Load src/data/eval_dataset/gui_dataset.py without running the package
    __init__ files, whose star-imports pull in the optional video/audio readers
    (cv2, decord, imageio, soundfile) that the eval itself never needs here.
    """
    for name in ("src.data", "src.data.eval_dataset"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [os.path.join(EVAL_ROOT, *name.split("."))]
            sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(
        "src.data.eval_dataset.gui_dataset",
        os.path.join(EVAL_ROOT, "src", "data", "eval_dataset", "gui_dataset.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    gui_dataset = _load_gui_dataset()
except ImportError as exc:  # pragma: no cover - depends on the local install
    raise unittest.SkipTest(f"gui_dataset is not importable here: {exc}")

process_multi_images = gui_dataset.process_multi_images
IMAGE_TOKEN = gui_dataset.VLM_IMAGE_TOKENS[gui_dataset.PHI3V]

ROOT = "gui-tasks/Mind2Web"


class ProcessMultiImagesTest(unittest.TestCase):
    def test_distinct_screenshots_are_passed_through_in_order(self):
        paths = ["t/step_0.png", "t/step_1.png", "t/step_2.png"]
        self.assertEqual(
            process_multi_images(ROOT, paths),
            [os.path.join(ROOT, p) for p in paths],
        )

    def test_repeated_screenshots_are_kept_once_per_placeholder(self):
        # An action that leaves the screen unchanged (e.g. typing into a field)
        # makes a trajectory reference the same screenshot twice. Both steps
        # still have their own placeholder, so both must keep an image.
        paths = ["t/step_0.png", "t/step_0.png", "t/step_2.png"]
        self.assertEqual(
            process_multi_images(ROOT, paths),
            [os.path.join(ROOT, p) for p in paths],
        )

    def test_one_image_per_image_placeholder(self):
        # The invariant the interleaved multi-image branch of the processor
        # relies on: image count == placeholder count, so screenshot N lands on
        # "Observation N".
        paths = ["t/step_0.png"] * 4 + ["t/step_4.png"]
        text = "".join(
            f"Observation {i}: {IMAGE_TOKEN}\nAction {i}: click()\n"
            for i in range(1, len(paths) + 1)
        )
        self.assertEqual(len(process_multi_images(ROOT, paths)), text.count(IMAGE_TOKEN))

    def test_accepts_a_python_literal_list_and_blank_entries(self):
        self.assertEqual(
            process_multi_images(ROOT, '["t/a.png", "", "t/a.png"]'),
            [os.path.join(ROOT, "t/a.png"), "", os.path.join(ROOT, "t/a.png")],
        )

    def test_empty_input_yields_a_single_blank_entry(self):
        self.assertEqual(process_multi_images(ROOT, None), [""])
        self.assertEqual(process_multi_images(ROOT, []), [""])


if __name__ == "__main__":
    unittest.main()
