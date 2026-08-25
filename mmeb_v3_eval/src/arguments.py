from dataclasses import dataclass, field
from typing import List
import torch

try:
    from transformers import TrainingArguments as HFTrainingArguments
    _TRAINING_ARGS_IMPORT_ERROR = None
except Exception as exc:
    _TRAINING_ARGS_IMPORT_ERROR = exc

    class HFTrainingArguments:  # type: ignore[override]
        """Lightweight fallback to keep inference-time imports working."""

        def __init__(self, *args, **kwargs):
            pass

        def __post_init__(self):
            pass

        @property
        def device(self):
            return torch.device("cpu")


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "huggingface model name or path"})
    model_type: str = field(default=None, metadata={"help": "model type, typically includes in config file, but"
                                                            " sometimes needs mannually add"})
    processor_name: str = field(default=None, metadata={"help": "processor_name, huggingface model name or path"})
    model_backbone: str = field(default=None, metadata={"help": "HF model type"})
    checkpoint_path: str = field(default=None, metadata={"help": "a local model path, could be a LoRA version"})
    pooling: str = field(default='last', metadata={"help": "pooling method for encoder"})
    normalize: bool = field(default=False, metadata={"help": "normalize query and passage representations"})
    temperature: float = field(default=0.02, metadata={"help": "temperature for softmax"})
    lora: bool = field(default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"})
    lora_r: int = field(default=16, metadata={"help": "lora r"})
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "lora dropout"})
    lora_target_modules: str = field(default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj"
                                             ",gate_proj,up_proj", metadata={"help": "lora target modules"})
    full_finetune: bool = field(default=False, metadata={"help": "train all parameters (disable embedding-only"
                                                                 " freezing)"})
    num_crops: int = field(default=16, metadata={"help": "number of crops used in image encoder"})
    uigraph_use: bool = field(default=False, metadata={"help": "Enable ui graph for token selection"})
    uigraph_diff: int = field(default=1, metadata={"help": "Pixel difference used for constructing ui graph for"
                                                           " token selection"})
    uigraph_rand: bool = field(default=False, metadata={"help": "Enable random graph construction for token selection"})
    uimask_ratio: float = field(default=0.5, metadata={"help": "Specify the percentage of patch tokens to skip per"
                                                               " component for token selection"})
    uimask_rand: bool = field(default=False, metadata={"help": "Enable random token selection instead of uniform"
                                                               " selection"})
    lm_skip_layer: str = field(default='[1,28,0]', metadata={"help": "Specify the layers of the language model to"
                                                                     " skip for token selection"})
    vis_skip_layer: str = field(default='[1,32,0]', metadata={"help": "Specify the layers of the vision model to"
                                                                      " skip for token selection"})
    # WAVE official inference options (only used when --model_backbone wave)
    wave_train_classify: bool = field(default=True, metadata={"help": "Enable WAVE classify head inference path"})
    wave_classify_type: str = field(default="all_layer", metadata={"help": "WAVE classify type: last_layer |"
                                                                           " all_layer"})
    wave_pred_embeds: bool = field(default=True, metadata={"help": "Request WAVE model to return embedding outputs"})
    wave_use_beats: bool = field(default=False, metadata={"help": "Enable WAVE BEATs branch"})
    wave_beats_path: str = field(default=None, metadata={"help": "Path to BEATs checkpoint required by WAVE when"
                                                                 " wave_use_beats=true"})
    wave_beats_only: bool = field(default=False, metadata={"help": "Use BEATs-only audio branch in WAVE"})


@dataclass
class DataArguments:
    dataset_config: str = field(default=None, metadata={"help": "yaml file with dataset configuration"})
    data_basedir: str = field(default=None, metadata={"help": "Expect an absolute path to the base directory of"
                                                              " all datasets. If set, it will be prepended to each"
                                                              " dataset path"})
    dataset_name: str = field(default=None, metadata={"help": "huggingface dataset name"})
    subset_name: List[str] = field(default=None, metadata={"help": "Useful for datasets with subsets"})
    dataset_split: str = field(default='train', metadata={"help": "dataset split"})
    num_sample_per_subset: int = field(default=None, metadata={"help": "number of training samples per subset"})
    image_dir: str = field(default=None, metadata={"help": "Image directory path"})
    encode_output_path: str = field(default=None, metadata={"help": "encode output path"})
    max_len: int = field(default=None, metadata={"help": "The maximum total input sequence length after"
                                                         " tokenization. Use with caution, since it may truncate"
                                                         " text prompts due to large image lengths."},)
    embedding_type: str = field(default="", metadata={"help": "embedding type"})
    image_resolution: str = field(default=None, metadata={"help": "for models i.e. LLaVA-next and Qwen, resize"
                                                                  " images first, none means using original image"
                                                                  " resolution. This is only works when"
                                                                  " `--resize_use_processor false`."})
    resize_use_processor: bool = field(default=True, metadata={"help": "Resize visual inputs insides processor,"
                                                                       " e.g. Qwen2VLImageProcessor, instead of by"
                                                                       " our code."})
    resize_min_pixels: int = field(default=None, metadata={"help": "Override the min pixels of the model's"
                                                                   " image/video processor. None (default) keeps"
                                                                   " the limits from the checkpoint's own"
                                                                   " processor_config.json, which is the"
                                                                   " recommended setting for WeMM-Embedding."})
    resize_max_pixels: int = field(default=None, metadata={"help": "Override the max pixels of the model's"
                                                                   " image/video processor. None (default) keeps"
                                                                   " the limits from the checkpoint's own"
                                                                   " processor_config.json, which is the"
                                                                   " recommended setting for WeMM-Embedding. NOTE:"
                                                                   " the legacy VLM2Vec default (28*28*1280 ~ 1M"
                                                                   " pixels) heavily downscales tall"
                                                                   " document/infographic images and costs several"
                                                                   " points on such datasets; do not use it for"
                                                                   " WeMM-Embedding."})
    image_decay_factor: float = field(default=None, metadata={"help": "The image decay factor for resizing"
                                                                      " temporal images"})
    num_hardneg: int = field(default=0, metadata={"help": "hard negative number"})
    video_max_frames: int = field(default=8, metadata={"help": "Max video frames per sample in training"})
    video_frame_size: int = field(default=224, metadata={"help": "Square frame size for video frames (pixels)"})

    # Audio configuration (unified for train/eval)
    audio_sample_rate: int = field(default=16000, metadata={"help": "Audio sample rate for resampling"})
    audio_max_seconds: float = field(default=None, metadata={"help": "Maximum audio duration in seconds. If set,"
                                                                     " takes precedence over audio_max_samples"})
    audio_max_samples: int = field(default=None, metadata={"help": "Maximum audio samples. Used if"
                                                                   " audio_max_seconds is not set"})
    audio_min_samples: int = field(default=None, metadata={"help": "Minimum audio samples to keep (shorter audios"
                                                                   " are discarded)"})

    # Audio cropping strategies
    train_crop: str = field(default="random", metadata={"help": "Audio cropping strategy for training: 'random'"})
    eval_crop: str = field(default="head", metadata={"help": "Audio cropping strategy for evaluation: 'head',"
                                                             " 'center', or 'multi_crop'"})


@dataclass
class TrainingArguments(HFTrainingArguments):
    device: torch.device = field(default=None, metadata={"help": "device for training/inference"})
    image_encoder_freeze: bool = field(default=False, metadata={"help": "huggingface model name"})
    output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})
    resume_from: str = field(default="none", metadata={"help": "`auto` will detect if any previous checkpoints"
                                                               " should be resumed. or specify specific step of"
                                                               " the checkpoint."})
    project_name: str = field(default=None, metadata={"help": "project name"})
    logging_steps: int = field(default=1, metadata={"help": "logging steps"})
    num_train_epochs: int = field(default=1, metadata={"help": "number of training epochs"})
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=2, metadata={"help": "query side subset size"})
    gc_p_chunk_size: int = field(default=2, metadata={"help": "target side subset size"})
    interleave_stopping_strategy: str = field(default="all_exhausted", metadata={"help": "all_exhausted or"
                                                                                         " first_exhausted"})
    homogeneous_batch_size_per_device: float = field(default=0, metadata={"help": "Specify number of consecutive"
                                                                                  " samples from the same dataset"
                                                                                  " PER DEVICE. 0/None means"
                                                                                  " random mixing."})
    interleave_batch_size: float = field(default=0, metadata={"help": "[DEPRECATED] Use"
                                                                      " `homogeneous_batch_size_per_device`."})
    export_full_checkpoint: int = field(default=0, metadata={"help": "Export full HF dir at a specific checkpoint"
                                                                     " step; 0 disables."})
    loss_stage: str = field(default="infonce", metadata={"help": "Training loss stage: infonce | jepa | mixed"})
    loss_alpha: float = field(default=0.5, metadata={"help": "Alpha for mixed loss: alpha*jepa + (1-alpha)*infonce"})
    jepa_predictor_hidden: int = field(default=0, metadata={"help": "Hidden dim for JEPA predictor MLP. <=0 means"
                                                                    " use emb dim."})

    def __post_init__(self):
        # Ensure base TrainingArguments initialization runs (sets distributed_state, etc.)
        super().__post_init__()
        if _TRAINING_ARGS_IMPORT_ERROR is not None:
            # In fallback mode we keep a minimal object for import-time compatibility.
            self.distributed_state = None
            self.device = torch.device("cpu")
            return
        # Some transformers versions don't define distributed_state; keep it for later access.
        if not hasattr(self, "distributed_state"):
            self.distributed_state = None
        # Mirror the base property so callers using the field get the actual device.
        try:
            self.device = super().device
        except Exception:
            pass


@dataclass
class MTEBArguments:
    device: str = field(default="cuda", metadata={"help": "use cuda for single GPU inference, if multiple GPUs are"
                                                          " available it will use DP automatically"})
    batch_size_per_device: int = field(default=16, metadata={"help": ""})
    max_length: int = field(default=512, metadata={"help": ""})
    eval_output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})
    task_types: List[str] = field(default=None, metadata={"help": ""})
    tasks: List[str] = field(default=None, metadata={"help": ""})
    prompt_family: List[str] = field(default=None, metadata={"help": ""})
