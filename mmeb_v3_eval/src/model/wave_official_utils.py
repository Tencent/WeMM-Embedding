import sys
from pathlib import Path


def ensure_wave_official_import_path() -> Path:
    """Add vendored WAVE official code root to sys.path once."""
    # .../OLM2Vec/OLM2Vec/src/model/wave_official_utils.py -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    wave_root = repo_root / "third_party" / "wave_official"
    if not wave_root.exists():
        raise FileNotFoundError(f"WAVE official code not found: {wave_root}")
    wave_root_str = str(wave_root)
    if wave_root_str not in sys.path:
        sys.path.insert(0, wave_root_str)
    return wave_root


def load_wave_official_classes():
    """Lazy import WAVE official processor/config/model classes."""
    ensure_wave_official_import_path()
    from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    return Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration


def load_wave_official_processor_class():
    ensure_wave_official_import_path()
    from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor

    return Qwen2_5OmniProcessor


def load_wave_official_model_classes():
    ensure_wave_official_import_path()
    from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    return Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
