from typing import Optional, List


DATASET_INSTRUCTION = {
    # Audio Classification
    "ESC-50": "Recognize the environmental sound category of the audio.",
    "UrbanSound8K": "Recognize the urban sound category of the audio.",
    "NSynth": "Recognize the musical instrument category of the audio.",
    "SpeechCommands": "Recognize the spoken word category in the audio.",
    "CREMA-D": "Recognize the emotion expressed in the speech audio.",

    # Text-to-Audio Retrieval
    "Clotho": "Retrieve audio clips that best match the given textual description.",
    "SoundDescs": "Retrieve audio samples that best match the given sound description.",

    # Audio-to-Image
    "SpeechCOCO": "Retrieve images that best match the given spoken description.",

    # Audio-to-Video
    "AVE": "Retrieve the video that best matches the given audio event.",

    # Audio Event Grounding
    "TUTSound": "Retrieve the sound event categories that best match the given audio.",
}


def build_query_text(dataset_name: str, raw_text: Optional[str] = None) -> List[str]:
    """
    Unified query text builder for audio datasets.

    Args:
        dataset_name: dataset name; must exist in DATASET_INSTRUCTION
        raw_text: raw query text, optional

    Returns:
        A list containing a single non-empty string
    """
    if dataset_name not in DATASET_INSTRUCTION:
        raise KeyError(f"Dataset '{dataset_name}' not found in DATASET_INSTRUCTION. Available:"
                       f" {list(DATASET_INSTRUCTION.keys())}")

    instr = DATASET_INSTRUCTION[dataset_name]

    if raw_text is None or raw_text.strip() == "":
        return [instr]
    else:
        return [f"{instr}\nQuery: {raw_text.strip()}"]
