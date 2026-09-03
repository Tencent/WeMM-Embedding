# Video Classification
from .video_classification_datasets import load_video_class_dataset  # noqa: F401
from .ssv2_dataset import load_ssv2_dataset  # noqa: F401

# Video QA
from .videomme_dataset import load_videomme_dataset  # noqa: F401
from .mvbench_dataset import load_mvbench_dataset  # noqa: F401
from .nextqa_dataset import load_nextqa_dataset  # noqa: F401
from .egoschema_dataset import load_egoschema_dataset  # noqa: F401
from .activitynetqa_dataset import load_activitynetqa_dataset  # noqa: F401
from .videommmu_dataset import load_videommmu_dataset  # noqa: F401

# Video Retrieval
from .msrvtt_dataset import load_msrvtt_dataset  # noqa: F401
from .didemo_dataset import load_didemo_dataset  # noqa: F401
from .msvd_dataset import load_msvd_dataset  # noqa: F401
from .youcook2_dataset import load_youcook2_dataset  # noqa: F401
from .vatex_dataset import load_vatex_dataset  # noqa: F401

from .gui_dataset import load_gui_dataset  # noqa: F401

# Temporal Grounding
from .moment_retrieval_datasets import load_moment_retrieval_dataset  # noqa: F401
from .momentseeker_dataset import load_momentseeker_dataset  # noqa: F401

# MMEB
from .image_cls_dataset import load_image_cls_dataset  # noqa: F401
from .image_qa_dataset import load_image_qa_dataset  # noqa: F401
from .image_t2i_eval import load_image_t2i_dataset  # noqa: F401
from .image_i2t_eval import load_image_i2t_dataset  # noqa: F401
from .image_i2i_vg_dataset import load_image_i2i_vg_dataset  # noqa: F401
from .mcmr_dataset import load_mcmr_dataset  # noqa: F401
from src.data.collator.mscoco_cmret import load_mscoco_cmret_dataset  # noqa: F401

# VisDoc
from .vidore_dataset import load_vidore_dataset  # noqa: F401
from .visrag_dataset import load_visrag_dataset  # noqa: F401

# ToolDe
from .toolde_dataset import load_toolde_dataset  # noqa: F401

# Audio Classification
from .audio_cls_dataset import load_audio_cls_dataset  # noqa: F401

# Audio Retrieval
from .sounddescs_retrieval_dataset import load_sounddescs_text_audio_dataset  # noqa: F401
from .clotho_dataset import load_clotho_text_audio_dataset  # noqa: F401
from .speechcoco_retrieval_dataset import load_speechcoco_dataset  # noqa: F401
from .ave_retreival_dateset import load_ave_retrieval_dataset  # noqa: F401

# Audio Grounding
from .tutsound_dataset import load_tutsound_audio_dataset  # noqa: F401
from .tutsound_hard_dataset import load_tutsound_hard_audio_dataset  # noqa: F401

# Text Retrieval
from .complex_text_retrieve import load_complex_text_retrieve_dataset  # noqa: F401
from .memory_retrieval_dateset import load_memory_retrieval_dataset  # noqa: F401
